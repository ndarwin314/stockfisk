import asyncio
import logging
import time
import random
from copy import deepcopy
import os

import nashpy
import aioprocessing as aio
import torch
import numpy as np
from torch import nn
from minimax_q.worker import Learner, Actor, ReplayBuffer, Transition, LocalBuffer, PriorityTree, Block, value_rescale, inverse_value_rescale
from minimax_q.minimaxq import Minimax
from minimax_q.r2d2 import Network, AgentState, uk_mon_idx, uk_move_idx,move_lookup, dex_lookup
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
    env = Minimax(None, None)

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
        return idxes, self.buffer[idxes], is_weights
        

async def train_single_actor():
    env = Minimax(None, None)

    model = Network(env.action_space.shape[0], env.observation_space.shape[0], config.hidden_dim)
    target_model = deepcopy(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, eps=config.eps)
    loss_fn = nn.MSELoss(reduction='none')
    del env
    #model.share_memory()

    local_buffer = LocalBuffer()
    replay_buffer = ReplayBufferSync(1000)
    agents = (Minimax(model), Minimax(model))

    gamma_n = config.gamma ** config.forward_steps
    num_updates = 0
    while True:
        print(num_updates)
        for _ in range(10):
            await agents[0].battle_against(agents[1], 1)
            print("battle")
            num_turns = len(agents[0].history)
            for i in range(num_turns):
                states = (agents[0].history[i], agents[1].history[i])
                transitions = []
                for j in range(2):
                    t = Transition(
                        states[j].obs,
                        agents[j].actions[i],
                        agents[1 - j].actions[i],
                        states[i].legal_moves_idx,
                        states[i].legal_moves_opponent_idx,
                        states[i].unknown_pokemon,
                        states[i].unknown_moves,
                        0,
                        agents[0].qs[i],
                        states[i].hidden_state)
                    transitions.append(t)
                if i == num_turns - 1:
                    for j in range(2):
                        transitions[j].reward = 1 if agents[j].n_battles_won() == 1 else -1
                local_buffer.add(transitions[0], transitions[1])

            blocks, priorities = local_buffer.finish()
            replay_buffer.add(blocks[0], priorities[0])
            replay_buffer.add(blocks[1], priorities[1])
        agents[0].reset()
        agents[1].reset()
        if replay_buffer.size < 500:
            continue

        idxes, blocks, is_weights = replay_buffer.sample(32)
        losses = np.zeros(32)
        for sample_idx, block in enumerate(blocks):
            predicted_qs = np.zeros(block.size)
            target_qs = np.zeros(block.size)
            hidden = None
            target_hidden = hidden
            # i think that since we start the hidden state at all 0s and are doing
            # entire episodes as batches this makes sense as hidden state initialization
            for i in range(block.size - config.forward_steps):
                # do forward pass
                state = AgentState(
                    block.obs[i],
                    hidden,
                    block.legal_actions[i],
                    block.predicted_legal_actions[i],
                    block.unrevealed_pokemon[i],
                    block.unrevealed_moves[i]
                )
                q_mat, hidden = model(state)
                predicted_qs[i] = q_mat
                a = block.legal_actions[i].find(block.action[i])
                opp_action = block.opponent_action[i]
                if opp_action in block.predicted_legal_actions[i]:
                    b = block.predicted_legal_actions[i].find(block.opponent_action[i])
                elif opp_action in dex_lookup:
                    b = block.predicted_legal_actions[i].find(uk_mon_idx)
                elif opp_action in move_lookup:
                    b = block.predicted_legal_actions[i].find(uk_move_idx)
                else:
                    print("huh, weird thing in action lookup")
                    continue
                q_val = q_mat[a, b]
                predicted_qs[i] = q_val

                if i > config.burn_in_steps + config.forward_steps:
                    game = nashpy.Game(q_mat)
                    a, b = next(game.support_enumeration())
                    state.hidden_state = target_hidden
                    q_mat, target_hidden = target_model(state)
                    target_qs[i - config.forward_steps] = nashpy.Game(q_mat)[a, b]
            losses[sample_idx] = loss_fn(
                predicted_qs[config.burn_in_steps:-config.forward_steps],
                target_qs[config.burn_in_steps:-config.forward_steps]).mean()
        loss = (is_weights * losses).mean()
        loss.backward()
        optimizer.step()
        num_updates += 1
        agents[0].set_model(model)
        agents[1].set_model(model)
        if num_updates % 10 == 0:
            target_model.load_state_dict(model.state_dict())
            torch.save(model.state_dict(),
                       os.path.join("/Users/ndarwin/PycharmProjectsk/stockfisk/models", f'{num_updates}.pth'))




