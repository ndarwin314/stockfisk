'''Replay buffer, learner and actor'''
import time
import os
from copy import deepcopy
from typing import List, Tuple, Any

import aioprocessing as aio
import torch
import torch.nn as nn
import numpy as np
import nashpy
from poke_env.concurrency import handle_threaded_coroutines

from minimax_q.utils import Transition, LocalBuffer, Block, ReplayBuffer
from minimax_q.minimaxq import Minimax
import minimax_q.config as config


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
        return value.sign() * ((abs(value) + 1).sqrt() - 1) + eps * value

    @staticmethod
    def inverse_value_rescale(value, eps=1e-3):
        temp = ((1 + 4 * eps * (abs(value) + 1 + eps)).sqrt() - 1) / (2 * eps)
        return value.sign() * (temp.square() - 1)


############################## Actor ##############################

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
            await handle_threaded_coroutines(self.agents[0].battle_against(self.agents[1], 1))
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

