import asyncio
import logging
import time
import copy
import random
from copy import deepcopy
import os

import nashpy
import aioprocessing as aio
import torch
import numpy as np
import scipy as sp
from torch import nn
from minimax_q.worker import Learner, Actor, ReplayBuffer, Transition, LocalBuffer, PriorityTree, Block, value_rescale, inverse_value_rescale
from minimax_q.minimaxq import Minimax, move_size, mon_embed_size
from minimax_q.r2d2 import Network, AgentState, uk_mon_idx, uk_move_idx,move_lookup, dex_lookup, noop_idx, combined_lookup
import minimax_q.config as config


torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.set_num_threads(1)


def get_epsilon(actor_id: int, base_eps: float = config.base_eps, alpha: float = config.alpha, num_actors: int = config.num_actors):
    exponent = 1 + actor_id / (2 * num_actors-1) * alpha
    return base_eps**exponent


def wrap_run(actor):
    asyncio.run(actor.run())


def train(num_actors=config.num_actors, log_interval=config.log_interval):
    env = Minimax(None, 200)

    model = Network(env.action_space.shape[0], env.observation_space.shape[0], config.hidden_dim)
    del env
    model.share_memory()
    sample_queue_list = [aio.AioQueue() for _ in range(num_actors)]
    batch_queue = aio.AioQueue(8)
    priority_queue = aio.AioQueue(8)

    buffer = ReplayBuffer(sample_queue_list, batch_queue, priority_queue)
    learner = Learner(batch_queue, priority_queue, model)
    actors = [Actor(model, get_epsilon(i), sample_queue_list[i]) for i in range(num_actors)]

    actor_procs = [aio.AioProcess(target=wrap_run, args=(actor,)) for actor in actors]
    for proc in actor_procs:
        proc.start()

    buffer_proc = aio.AioProcess(target=buffer.run)
    buffer_proc.start()

    learner.run()

    buffer_proc.join()

    for proc in actor_procs:
        proc.terminate()


class ReplayBufferSync:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer: list[Block] = [None] * capacity
        self.priority_tree = PriorityTree(capacity, config.prio_exponent, config.importance_sampling_exponent)
        self.size = 0
        self.ptr = 0
        
    def __len__(self):
        return self.size
    
    def add(self, block, priority):
        self.priority_tree.update(self.ptr, priority)
        self.buffer[self.ptr] = block
        self.ptr = (self.ptr + 1) % self.capacity
        self.size += 1
        
    def update_priorities(self, idxes, td_errors, old_ptr):
        if self.ptr > old_ptr:
            # range from [old_ptr, self.seq_ptr)
            mask = (idxes < old_ptr) | (idxes >= self.ptr)
            idxes = idxes[mask]
            td_errors = td_errors[mask]
        elif self.ptr < old_ptr:
            # range from [0, self.seq_ptr) & [old_ptr, self,capacity)
            mask = (idxes < old_ptr) & (idxes >= self.ptr)
            idxes = idxes[mask]
            td_errors = td_errors[mask]
        self.priority_tree.update(idxes, td_errors)

    def sample(self, num_samples) -> (np.ndarray, list[Block]):
        idxes, is_weights = self.priority_tree.sample(num_samples)
        blocks = []
        for idx in idxes:
            blocks.append(self.buffer[idx])
        return idxes, blocks, torch.from_numpy(is_weights), self.ptr
        

async def train_single_actor():
    env = Minimax(None)

    start_iter = 3720
    batch_size = 64
    games_per_run = 16
    min_buffer_size = 2 * batch_size
    model = Network(env.action_space.shape[0], env.observation_size, config.hidden_dim)
    model.load_state_dict(torch.load(f"/home/mathy/PycharmProjects/stockfisk/models/{start_iter}.pth"))
    target_model = deepcopy(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, eps=config.eps)
    loss_fn = nn.MSELoss(reduction="mean")
    del env
    #model.share_memory()

    local_buffer = LocalBuffer()
    replay_buffer = ReplayBufferSync(400)
    agents = (Minimax(model), Minimax(model))
    num_updates = start_iter
    gamma_n = config.gamma ** config.forward_steps
    gamma_arr = config.gamma ** torch.arange(0, config.forward_steps).float()
    while True:
        for badd in range(games_per_run):
            print(num_updates, badd)
            await agents[0].battle_against(agents[1], 1)
            num_turns = agents[0].turn
            if num_turns <= 10:
                print("short")
                continue
            for i in range(num_turns):
                actions = (agents[0].actions[i], agents[1].actions[i])
                states = (agents[0].history[i], agents[1].history[i])
                qs = (agents[0].qs[i], agents[1].qs[i])
                max_len = max(len(actions[0]), len(actions[1]))
                for j in range(2):
                    while len(actions[j]) < max_len:
                        actions[j].append(noop_idx)
                        c = copy.deepcopy(states[j][-1])
                        c.reward = 0
                        c.embeddings_self = {noop_idx: np.zeros(mon_embed_size+move_size)}
                        c.legal_moves_idx = [noop_idx]
                        states[j].append(c)
                        qs[j].append(qs[j][-1])
                for k in range(max_len):
                    transitions = []
                    for j in range(2):
                        t = Transition(
                            sp.sparse.coo_array(states[j][k].obs),
                            actions[j][k],
                            actions[1-j][k],
                            states[j][k].legal_moves_idx,
                            states[j][k].legal_moves_opponent_idx,
                            states[j][k].unknown_pokemon,
                            states[j][k].unknown_moves,
                            states[j][k].reward,
                            qs[j][k],
                            states[j][k].hidden_state,
                            states[j][k].embeddings_self,
                            states[j][k].embeddings_opponent,
                        )
                        transitions.append(t)
                    local_buffer.add(transitions[0], transitions[1])
            temp = agents[0].calc_reward(list(agents[0].battles.values())[-1])
            blocks, priorities = local_buffer.finish((temp, -temp))
            local_buffer.reset()
            replay_buffer.add(blocks[0], priorities[0])
            replay_buffer.add(blocks[1], priorities[1])
            agents[0].reset()
            agents[1].reset()
        if replay_buffer.size < min_buffer_size:
            continue

        idxes, blocks, is_weights, old_ptr = replay_buffer.sample(batch_size)
        losses = torch.zeros(batch_size)
        new_prio = np.zeros(len(blocks))
        for sample_idx, block in enumerate(blocks):
            try:
                predicted_qs = torch.zeros(block.size)
                target_qs = torch.zeros(block.size)
                rewards = torch.from_numpy(block.reward).float()
                hidden = None
                target_hidden = hidden
                # i think that since we start the hidden state at all 0s and are doing
                # entire episodes as batches this makes sense as hidden state initialization
                for i in range(block.size):
                    # do forward pass
                    state = AgentState(
                        block.obs[i].toarray(),
                        hidden,
                        block.legal_actions[i],
                        block.predicted_legal_actions[i],
                        block.unrevealed_pokemon[i],
                        block.unrevealed_moves[i],
                        rewards[i],
                        block.embeddings_self[i],
                        block.embeddings_opponent[i]
                    )
                    #print(combined_lookup[block.action[i]], [combined_lookup[x] for x in block.legal_actions[i]])
                    #print(combined_lookup[block.opponent_action[i]], [combined_lookup[x] for x in block.predicted_legal_actions[i]])
                    #print()
                    q_mat, hidden = model(state)
                    a = block.legal_actions[i].index(block.action[i])
                    opp_action = block.opponent_action[i]
                    # this is pretty hacky
                    # the second condition is a workaround for when a pokemon faints from a switching move like u-turn
                    if opp_action == noop_idx or opp_action != noop_idx and noop_idx in block.predicted_legal_actions[i]:
                        assert len(block.predicted_legal_actions[i]) == 1
                        b = 0
                    elif opp_action in block.predicted_legal_actions[i]:
                        b = block.predicted_legal_actions[i].index(block.opponent_action[i])
                    elif opp_action in dex_lookup:
                        b = block.predicted_legal_actions[i].index(uk_mon_idx)
                    elif opp_action in move_lookup:
                        b = block.predicted_legal_actions[i].index(uk_move_idx)
                    else:
                        print("huh, weird thing in action lookup")
                        continue
                    q_val = q_mat[a, b]
                    predicted_qs[i] = q_val
                    if i > config.burn_in_steps + config.forward_steps:
                        q_mat = q_mat.detach().numpy()
                        game = nashpy.Game(q_mat)
                        a, b = next(game.support_enumeration())
                        state.hidden_state = target_hidden
                        q_mat, target_hidden = target_model(state)
                        q_mat = q_mat.detach().numpy()
                        q = nashpy.Game(q_mat)[a, b][0]
                        # oops i forgot to add reward, silly uwu
                        reward_range = rewards[i+2 - config.forward_steps: i+2]
                        reward_sum = torch.dot(gamma_arr, reward_range)
                        target = value_rescale(gamma_n * q + inverse_value_rescale(reward_sum))
                        target_qs[i - config.forward_steps] = target
                losses[sample_idx] = loss_fn(
                    predicted_qs[config.burn_in_steps:-config.forward_steps],
                    target_qs[config.burn_in_steps:-config.forward_steps])
                error = torch.abs(predicted_qs[config.burn_in_steps:-config.forward_steps]-
                               target_qs[config.burn_in_steps:-config.forward_steps])
                priority = torch.mean(error) * .9 + torch.max(error) * .1
            except (KeyError, ValueError, AssertionError) as e:
                print("oopsy", e)
                priority = 0
                losses[sample_idx] = 0
            new_prio[sample_idx] = priority
        loss = (is_weights * losses).mean()
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm)
        optimizer.step()
        num_updates += 1
        agents[0].set_model(model)
        agents[1].set_model(model)
        replay_buffer.update_priorities(idxes, new_prio, old_ptr)
        if num_updates % 20 == 0:
            target_model.load_state_dict(model.state_dict())
            torch.save(model.state_dict(),
                       os.path.join("/home/mathy/PycharmProjects/stockfisk/models", f'{num_updates}.pth'))




