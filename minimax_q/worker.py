'''Replay buffer, learner and actor'''
import time
import os
import math
from copy import deepcopy
from typing import List, Tuple, Any
import aioprocessing as aio
from dataclasses import dataclass
import asyncio

import numpy.random
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import nashpy
from poke_env.player.env_player import EnvPlayer

from minimax_q.r2d2 import Network, AgentState
from minimax_q.minimaxq import Minimax
from minimax_q.priority_tree import PriorityTree
import minimax_q.config as config


def value_rescale(value, eps=1e-3):
    return value.sign() * ((value.abs() + 1).sqrt() - 1) + eps * value


def inverse_value_rescale(value, eps=1e-3):
    temp = ((1 + 4 * eps * (value.abs() + 1 + eps)).sqrt() - 1) / (2 * eps)
    return value.sign() * (temp.square() - 1)


############################## Replay Buffer ##############################


@dataclass
class Block:
    obs: torch.tensor
    action: np.array
    opponent_action: np.array
    legal_actions: Any
    predicted_legal_actions: Any
    unrevealed_pokemon: np.ndarray
    unrevealed_moves: np.ndarray
    reward: np.array
    hidden: list[tuple[torch.tensor, torch.tensor]]
    size: int


class ReplayBuffer:
    def __init__(self, sample_queue_list, batch_queue, priority_queue, buffer_capacity=config.buffer_capacity,
                 alpha=config.prio_exponent, beta=config.importance_sampling_exponent,
                 batch_size=config.batch_size):

        self.buffer_capacity = buffer_capacity
        self.block_len = config.block_length

        self.block_ptr = 0

        # since im making sequences variable length
        # i'll do a generous overestimate for the capacity of the tree
        self.max_num_blocks = buffer_capacity // 20

        self.priority_tree = PriorityTree(self.max_num_blocks, alpha, beta)

        self.batch_size = batch_size

        self.env_steps = 0

        self.num_episodes = 0
        self.episode_reward = 0

        self.training_steps = 0
        self.last_training_steps = 0
        self.sum_loss = 0

        self.lock = aio.AioLock()

        self.size = 0
        self.last_size = 0

        self.buffer = [None] * self.max_num_blocks

        self.sample_queue_list, self.batch_queue, self.priority_queue = sample_queue_list, batch_queue, priority_queue

    def __len__(self):
        return self.size

    def run(self):
        background_thread = aio.AioProcess(target=self.add_data, daemon=True)
        background_thread.start()

        background_thread = aio.AioProcess(target=self.prepare_data, daemon=True)
        background_thread.start()

        background_thread = aio.AioProcess(target=self.update_data, daemon=True)
        background_thread.start()

        log_interval = config.log_interval

        while True:
            print(f'buffer size: {self.size}')
            print(f'buffer update speed: {(self.size - self.last_size) / log_interval}/s')
            self.last_size = self.size
            print(f'number of environment steps: {self.env_steps}')
            if self.num_episodes != 0:
                print(f'average episode return: {self.episode_reward / self.num_episodes:.4f}')
                # print(f'average episode return: {self.episode_reward/self.num_episodes:.4f}')
                self.episode_reward = 0
                self.num_episodes = 0
            print(f'number of training steps: {self.training_steps}')
            print(f'training speed: {(self.training_steps - self.last_training_steps) / log_interval}/s')
            if self.training_steps != self.last_training_steps:
                print(f'loss: {self.sum_loss / (self.training_steps - self.last_training_steps):.4f}')
                self.last_training_steps = self.training_steps
                self.sum_loss = 0
            self.last_env_steps = self.env_steps
            print()

            if self.training_steps == config.training_steps:
                break
            else:
                time.sleep(log_interval)

    def prepare_data(self):
        while self.size < config.learning_starts:
            time.sleep(1)

        while True:
            if not self.batch_queue.full():
                data = self.sample_batch()
                self.batch_queue.put(data)
            else:
                time.sleep(0.1)

    def add_data(self):
        while True:
            for sample_queue in self.sample_queue_list:
                if not sample_queue.empty():
                    data = sample_queue.get_nowait()
                    self.add(*data)

    def update_data(self):

        while True:
            if not self.priority_queue.empty():
                data = self.priority_queue.get_nowait()
                self.update_priorities(*data)
            else:
                time.sleep(0.1)

    def add(self, block: Block, priority: np.array, episode_reward: float):

        with self.lock:

            idxes = np.arange(self.block_ptr * self.seq_pre_block, (self.block_ptr + 1) * self.seq_pre_block,
                              dtype=np.int64)

            self.priority_tree.update(idxes, priority)

            if self.buffer[self.block_ptr] is not None:
                self.size -= np.sum(self.buffer[self.block_ptr].learning_steps).item()

            self.size += np.sum(block.learning_steps).item()

            self.buffer[self.block_ptr] = block

            self.env_steps += np.sum(block.learning_steps, dtype=np.int32)

            self.block_ptr = (self.block_ptr + 1) % self.num_blocks
            if episode_reward:
                self.episode_reward += episode_reward
                self.num_episodes += 1

    def sample_batch(self):
        '''sample one batch of training data'''
        batch_obs, batch_last_action, batch_last_reward, batch_hidden, batch_action, batch_reward, batch_gamma = (
            [], [], [], [], [], [], [])
        batch_last_opponent_action, batch_opponent_action = [], []
        burn_in_steps, learning_steps, forward_steps = [], [], []

        with self.lock:
            idxes, is_weights = self.priority_tree.sample(self.batch_size)

            block_idxes = idxes // self.seq_pre_block
            sequence_idxes = idxes % self.seq_pre_block

            for block_idx, sequence_idx in zip(block_idxes, sequence_idxes):
                block = self.buffer[block_idx]

                assert sequence_idx < block.num_sequences, 'index is {} but size is {}'.format(sequence_idx,
                                                                                               self.seq_pre_block_buf[
                                                                                                   block_idx])

                burn_in_step = block.burn_in_steps[sequence_idx]
                learning_step = block.learning_steps[sequence_idx]
                forward_step = block.forward_steps[sequence_idx]

                start_idx = block.burn_in_steps[0] + np.sum(block.learning_steps[:sequence_idx])

                obs = block.obs[start_idx - burn_in_step:start_idx + learning_step + forward_step]
                last_action = block.last_action[start_idx - burn_in_step:start_idx + learning_step + forward_step]
                last_opponent_action = block.last_opponent_action[start_idx - burn_in_step:start_idx + learning_step + forward_step]
                last_reward = block.last_reward[start_idx - burn_in_step:start_idx + learning_step + forward_step]
                obs, last_action, last_reward = torch.from_numpy(obs), torch.from_numpy(last_action), torch.from_numpy(
                    last_reward)

                start_idx = np.sum(block.learning_steps[:sequence_idx])
                end_idx = start_idx + block.learning_steps[sequence_idx]
                action = block.action[start_idx:end_idx]
                reward = block.n_step_reward[start_idx:end_idx]
                gamma = block.gamma[start_idx:end_idx]
                hidden = block.hidden[sequence_idx]

                batch_obs.append(obs)
                batch_last_action.append(last_action)
                batch_last_opponent_action.append((action))
                batch_last_reward.append(last_reward)
                batch_action.append(action)
                batch_opponent_action.append(action)
                batch_reward.append(reward)
                batch_gamma.append(gamma)
                batch_hidden.append(hidden)

                burn_in_steps.append(burn_in_step)
                learning_steps.append(learning_step)
                forward_steps.append(forward_step)

            batch_obs = pad_sequence(batch_obs, batch_first=True)
            batch_last_action = pad_sequence(batch_last_action, batch_first=True)
            batch_last_opponent_action = pad_sequence(batch_last_opponent_action, batch_first=True)
            batch_last_reward = pad_sequence(batch_last_reward, batch_first=True)

            is_weights = np.repeat(is_weights, learning_steps)

            data = (
                batch_obs,
                batch_last_action,
                batch_last_opponent_action,
                batch_last_reward,
                torch.from_numpy(np.stack(batch_hidden)).transpose(0, 1),

                torch.from_numpy(np.concatenate(batch_action)).unsqueeze(1),
                torch.from_numpy(np.concatenate(batch_opponent_action)).unsqueeze(1),
                torch.from_numpy(np.concatenate(batch_reward)),
                torch.from_numpy(np.concatenate(batch_gamma)),

                torch.ByteTensor(burn_in_steps),
                torch.ByteTensor(learning_steps),
                torch.ByteTensor(forward_steps),

                idxes,
                torch.from_numpy(is_weights.astype(np.float32)),
                self.block_ptr,

                self.env_steps
            )

        return data

    def update_priorities(self, idxes: np.ndarray, td_errors: np.ndarray, old_ptr: int, loss: float):
        """Update priorities of sampled transitions"""
        with self.lock:

            # discard the idxes that already been replaced by new data in replay buffer during training
            if self.block_ptr > old_ptr:
                # range from [old_ptr, self.seq_ptr)
                mask = (idxes < old_ptr * self.seq_pre_block) | (idxes >= self.block_ptr * self.seq_pre_block)
                idxes = idxes[mask]
                td_errors = td_errors[mask]
            elif self.block_ptr < old_ptr:
                # range from [0, self.seq_ptr) & [old_ptr, self,capacity)
                mask = (idxes < old_ptr * self.seq_pre_block) & (idxes >= self.block_ptr * self.seq_pre_block)
                idxes = idxes[mask]
                td_errors = td_errors[mask]

            self.priority_tree.update(idxes, td_errors)

        self.training_steps += 1
        self.sum_loss += loss


############################## Learner ##############################

def calculate_mixed_td_errors(td_error, learning_steps):
    start_idx = 0
    mixed_td_errors = np.empty(learning_steps.shape, dtype=td_error.dtype)
    for i, steps in enumerate(learning_steps):
        mixed_td_errors[i] = 0.9 * td_error[start_idx:start_idx + steps].max() + 0.1 * td_error[
                                                                                       start_idx:start_idx + steps].mean()
        start_idx += steps

    return mixed_td_errors


class Learner:
    def __init__(self, batch_queue, priority_queue, model, grad_norm: int = config.grad_norm,
                 lr: float = config.lr, eps: float = config.eps,
                 target_net_update_interval: int = config.target_net_update_interval,
                 save_interval: int = config.save_interval):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.online_net = deepcopy(model)
        self.online_net.to(self.device)
        self.online_net.train()
        self.target_net = deepcopy(self.online_net)
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr, eps=eps)
        self.loss_fn = nn.MSELoss(reduction='none')
        self.grad_norm = grad_norm
        self.batch_queue = batch_queue
        self.priority_queue = priority_queue
        self.num_updates = 0
        self.done = False

        self.target_net_update_interval = target_net_update_interval
        self.save_interval = save_interval

        self.batched_data = []

        self.shared_model = model

    def store_weights(self):
        self.shared_model.load_state_dict(self.online_net.state_dict())

    def prepare_data(self):
        while True:
            if not self.batch_queue.empty() and len(self.batched_data) < 4:
                data = self.batch_queue.get_nowait()
                self.batched_data.append(data)
            else:
                time.sleep(0.1)

    def run(self):
        background_thread = aio.AioProcess(target=self.prepare_data, daemon=True)
        background_thread.start()
        time.sleep(2)

        start_time = time.time()
        while self.num_updates < config.training_steps:

            while not self.batched_data:
                time.sleep(1)
            data = self.batched_data.pop(0)

            (batch_obs, batch_last_action, batch_last_opponent_action, batch_last_reward, batch_hidden, batch_action,
             batch_opponent_action, batch_n_step_reward, batch_n_step_gamma, burn_in_steps, learning_steps,
             forward_steps, idxes, is_weights, old_ptr, env_steps) = data
            batch_obs, batch_last_action, batch_last_opponent_action, batch_last_reward = (
                batch_obs.to(self.device), batch_last_action.to(self.device), batch_last_opponent_action.to(self.device),
                batch_last_reward.to(self.device))
            batch_hidden, batch_action, batch_opponent_action = (
                batch_hidden.to(self.device), batch_action.to(self.device), batch_opponent_action.to(self.device))
            batch_n_step_reward, batch_n_step_gamma = batch_n_step_reward.to(self.device), batch_n_step_gamma.to(
                self.device)
            is_weights = is_weights.to(self.device)

            batch_obs, batch_last_action, batch_last_opponent_action \
                = batch_obs.float(), batch_last_action.float(), batch_last_opponent_action.float()
            batch_action, batch_opponent_action = batch_action.long(), batch_opponent_action.long()
            burn_in_steps, learning_steps, forward_steps = burn_in_steps, learning_steps, forward_steps

            # wtf is this
            batch_hidden = (batch_hidden[:1], batch_hidden[1:])

            batch_obs = batch_obs / 255

            # not doing double q because its confusing with minimax

            # compute

            # get game matrices
            batch_next_qs = \
                self.target_net.calculate_q(
                    batch_obs, batch_action, batch_opponent_action, batch_last_reward,
                    batch_hidden, burn_in_steps, learning_steps, forward_steps)

            # this part doesnt seem like it can be parallelized unfortunately
            # also idk if this is the correct axis
            target_qs = torch.zeros(range(batch_next_qs.shape[-1]))
            for i in range(batch_next_qs.shape[-1]):
                # computing nash equilibria and utilities
                game = nashpy.Game(batch_next_qs[0])
                a, b = game.support_enumeration(tol=1e-6)
                q_val = game[a, b][0]
                target_qs[i] = q_val


            # TODO: probably need to do some dumb shit with squeezing and whatever
            # what is batch_n_step_reward
            target_q = self.value_rescale(
                batch_n_step_reward + batch_n_step_gamma * self.inverse_value_rescale(q_val))

            batch_q = self.online_net.calculate_q(batch_obs, batch_last_action, batch_last_opponent_action,
                                                  batch_last_reward, batch_hidden, burn_in_steps, learning_steps).gather(1, batch_action).squeeze(1)

            loss = (is_weights * self.loss_fn(batch_q, target_q)).mean()

            td_errors = (target_q - batch_q).detach().clone().squeeze().abs().cpu().float().numpy()

            priorities = calculate_mixed_td_errors(td_errors, learning_steps.numpy())

            # automatic mixed precision training
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.online_net.parameters(), self.grad_norm)
            self.optimizer.step()

            self.num_updates += 1

            self.priority_queue.put((idxes, priorities, old_ptr, loss.item()))

            # store new weights in shared memory
            if self.num_updates % 4 == 0:
                self.store_weights()

            # update target net
            if self.num_updates % self.target_net_update_interval == 0:
                self.target_net.load_state_dict(self.online_net.state_dict())

            # save model
            if self.num_updates % self.save_interval == 0:
                torch.save((self.online_net.state_dict(), self.num_updates, env_steps, (time.time() - start_time) / 60),
                           os.path.join('models', f'{self.num_updates}.pth'))

    @staticmethod
    def value_rescale(value, eps=1e-3):
        return value.sign() * ((value.abs() + 1).sqrt() - 1) + eps * value

    @staticmethod
    def inverse_value_rescale(value, eps=1e-3):
        temp = ((1 + 4 * eps * (value.abs() + 1 + eps)).sqrt() - 1) / (2 * eps)
        return value.sign() * (temp.square() - 1)


############################## Actor ##############################

# TODO: for morning NOah, update this to have two moves
# and figure out how to use both sides for training data
@dataclass
class Transition:
    observation: torch.tensor
    action_self: int
    action_opponent: int
    options_self: list[int]
    options_opponent: list[int]
    unrevealed_pokemon: list[int]
    unrevealed_moves: list[int]
    reward: int
    q_estimate: float
    hidden: tuple[torch.tensor, torch.tensor]


class LocalBuffer:
    '''store transitions of one episode'''

    def __init__(self, forward_steps: int = config.forward_steps,
                 burn_in_steps=config.burn_in_steps, learning_steps: int = config.learning_steps,
                 gamma: float = config.gamma, hidden_dim: int = config.hidden_dim,
                 block_length: int = config.block_length):

        self.obs_buffer = None
        self.gamma = gamma
        self.hidden_dim = hidden_dim
        self.forward_steps = forward_steps
        self.learning_steps = learning_steps
        self.burn_in_steps = burn_in_steps
        self.block_length = block_length
        self.curr_burn_in_steps = 0
        self.done = False
        self.size = 0
        self.transitions: tuple[list[Transition], list[Transition]] = ([], [])

    def __len__(self):
        return self.size

    def reset(self):
        self.curr_burn_in_steps = 0
        self.size = 0
        self.done = False
        self.transitions = ([], [])

    def add(self, transition1, transition2):
        self.transitions[0].append(transition1)
        self.transitions[1].append(transition2)
        self.size += 1

    def finish(self) -> (list[Block], list[float]):
        assert self.size <= self.block_length
        burn_steps = self.burn_in_steps
        forward_steps = self.forward_steps
        training_steps = self.size - burn_steps - forward_steps

        assert training_steps > 0

        gamma_arr = self.gamma ** -np.arange(forward_steps+1)
        # compute TD errors
        # i think this is correct but im not 100% sure
        # either way it seems reasonable-ish at least
        priorities = []
        blocks = []
        for i in range(2):
            errors = np.zeros(forward_steps)
            rewards = [t.reward for t in self.transitions[i]]
            rewards = rewards[burn_steps:-forward_steps]
            qs = [t.q_estimate for t in self.transitions[i]]
            for j in range(forward_steps):
                pred = np.dot(rewards[j:j+forward_steps], gamma_arr[:-1])
                pred += gamma_arr[-1] * inverse_value_rescale(qs[j+forward_steps])
                pred = value_rescale(pred)
                errors[i] = abs(pred - qs[i])

            priority = np.mean(errors) * .9 + np.max(errors) * .1
            priorities.append(priority)

            block = Block(
                np.array([t.obs for t in self.transitions[i]]),
                np.array([t.action for t in self.transitions[i]]),
                np.array([t.opponent_action for t in self.transitions[i]]),
                np.array([t.options_self for t in self.transitions[i]]),
                np.array([t.options_opponent for t in self.transitions[i]]),
                np.array([t.unrevealed_pokemon for t in self.transitions[i]]),
                np.array([t.unrevealed_moves for t in self.transitions[i]]),
                np.array([t.reward for t in self.transitions[i]]),
                [t.hidden for t in self.transitions[i]],
                self.size,
            )
            blocks.append(block)
        return blocks, priorities


class Actor:
    def __init__(self, model, epsilon: float, sample_queue):
        self.hidden_dim = config.hidden_dim
        self.model = model
        self.agents: (Minimax, Minimax) = (Minimax(model), Minimax(model))
        self.model.eval()
        self.local_buffer = LocalBuffer()

        self.epsilon = epsilon
        self.shared_model = self.model
        self.sample_queue = sample_queue
        self.max_episode_steps = config.max_episode_steps
        self.block_length = config.block_length

    async def run(self):
        num_games = 0
        while True:
            self.reset()
            await self.agents[0].battle_against(self.agents[1], 1)
            print("battle")
            num_turns = len(self.agents[0].history)
            for i in range(num_turns):
                states = (self.agents[0].history[i], self.agents[1].history[i])
                transitions = []
                for j in range(2):
                    t = Transition(
                        states[j].obs,
                        self.agents[j].actions[i],
                        self.agents[1-j].actions[i],
                        states[i].legal_moves_idx,
                        states[i].legal_moves_opponent_idx,
                        0,
                        self.agents[0].qs[i],
                        states[i].hidden_state)
                    transitions.append(t)
                if i == num_turns - 1:
                    for j in range(2):
                        transitions[j].reward = 1 if self.agents[j].n_battles_won() == 1 else -1
                self.local_buffer.add(transitions[0], transitions[1])

            num_games += num_turns
            blocks, priorities = self.local_buffer.finish()
            self.sample_queue.put(blocks[0])
            self.sample_queue.put(blocks[1])

            if num_games % 20 == 0:
                self.update_weights()

    def update_weights(self):
        '''load the latest weights from shared model'''
        self.model.load_state_dict(self.shared_model.state_dict())
        for agent in self.agents:
            agent.set_model(self.model)

    def reset(self):
        for agent in self.agents:
            agent.reset()
        self.local_buffer.reset()

