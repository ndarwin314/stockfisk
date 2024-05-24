'''Neural network model'''
from dataclasses import dataclass, field
from typing import Tuple, Optional
from collections import namedtuple
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import minimax_q.config as config
import poke_env.data as data
import numpy as np

gen_data = data.gen_data.GenData(8)

pokedex = gen_data.pokedex
pokedex["UNKNOWN"] = None
reverse_dex_lookup = {k: i for i, k in enumerate(pokedex)}
dex_lookup = {v: k for k, v in reverse_dex_lookup.items()}
uk_mon_idx = reverse_dex_lookup["UNKNOWN"]
dex_size = len(pokedex)
chart = gen_data.type_chart
moves = gen_data.moves
moves["UNKNOWN"] = None
reverse_move_lookup = {k: i+dex_size for i, k in enumerate(moves)}
move_lookup = {v: k for k, v in reverse_move_lookup.items()}
uk_move_idx = reverse_move_lookup["UNKNOWN"]
moves_size = len(pokedex)
action_embed_size = dex_size + moves_size

Option = namedtuple('Option',
                    ('pokemon', "moves", "unrevealed_pokemon", "unrevealed_moves"))


def embed_all_moves(idxs, unrevealed_mons: int, unrevealed_moves: int):
    embedding = np.zeros(action_embed_size)
    embedding[idxs] = 1
    if unrevealed_mons != 0:
        embedding[reverse_dex_lookup["UNKNOWN"]] = unrevealed_mons
    if unrevealed_moves != 0:
        embedding[reverse_move_lookup["UNKNOWN"]] = unrevealed_moves
    return embedding


def embed_idx(idx1, idx2, mon, move):
    embedding = np.zeros(2 * action_embed_size)
    embedding[idx1] = 1
    if idx2 == uk_mon_idx:
        embedding[idx2 + action_embed_size] = mon
    elif idx2 == uk_move_idx:
        embedding[idx2 + action_embed_size] = move
    else:
        embedding[idx2 + action_embed_size] = 1
    return embedding



@dataclass
class AgentState:
    obs: torch.Tensor
    hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    legal_moves_idx: list[int] = None
    legal_moves_opponent_idx: list[int] = None
    unknown_pokemon: int = 0
    unknown_moves: int = 0

    def __post_init__(self):
        temp1 = embed_all_moves(self.legal_moves_idx, 0, 0)
        temp2 = embed_all_moves(self.legal_moves_opponent_idx, self.unknown_pokemon, self.unknown_moves)
        self.recurrent_input = torch.concat([self.obs, temp1, temp2])
        bad = [[embed_idx(idx1, idx2, self.unknown_pokemon, self.unknown_moves)for idx1 in self.legal_moves_idx] for idx2 in self.legal_moves_opponent_idx]
        self.advantage_input = torch.tensor(bad)

    def update(self, obs, last_legal_idx, last_legal_oppponent_idx, hidden):
        self.obs = torch.from_numpy(obs).unsqueeze(0)
        self.legal_moves_idx = last_legal_idx
        self.legal_moves_opponent_idx = last_legal_oppponent_idx
        self.hidden_state = hidden


class Network(nn.Module):
    def __init__(self, action_dim, obs_dim, hidden_dim):
        super().__init__()

        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim

        self.max_forward_steps = config.forward_steps

        self.recurrent = nn.LSTM(obs_dim + 2 * action_dim + 1, self.hidden_dim, batch_first=True)

        # instead of learning a mapping from hidden -> action x action,
        # we learn a map from hidden, action, action ->  1
        # and evaluate across action x action inputs
        # this allows us to only keep moves that are legal
        self.advantage = nn.Sequential(
            nn.Linear(self.hidden_dim + 2 * action_dim, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, 1)
        )

        self.value = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, 1)
        )

    def forward(self, state: AgentState):

        recurrent_input = state.recurrent_input

        _, recurrent_output = self.recurrent(recurrent_input, state.hidden_state)

        hidden = recurrent_output[0]

        adv = self.advantage(torch.concat([state.hidden_state, state.advantage_input]))
        val = self.value(hidden)
        q_value = val + adv - adv.mean(1, keepdim=True)

        return q_value, recurrent_output

    def calculate_q_(self, obs, last_action, last_opponent_action, last_reward,
                     hidden_state, burn_in_steps, learning_steps, forward_steps, legal_idx, legal_opponent_idx):
        # obs shape: (batch_size, seq_len, obs_shape)
        batch_size, max_seq_len, *_ = obs.size()

        obs = obs.reshape(-1, * self.obs_shape)
        last_action = last_action.view(-1, self.action_dim)
        last_opponent_action = last_opponent_action.view(-1, self.action_dim)
        last_reward = last_reward.view(-1, 1)
        latent = self.feature(obs)

        seq_len = burn_in_steps + learning_steps + forward_steps

        recurrent_input = torch.cat((latent, last_action, last_opponent_action, last_reward), dim=1)
        recurrent_input = recurrent_input.view(batch_size, max_seq_len, -1)

        recurrent_input = pack_padded_sequence(recurrent_input, seq_len, batch_first=True, enforce_sorted=False)

        self.recurrent.flatten_parameters()
        recurrent_output, _ = self.recurrent(recurrent_input, hidden_state)

        recurrent_output, _ = pad_packed_sequence(recurrent_output, batch_first=True)

        seq_start_idx = burn_in_steps + self.max_forward_steps
        forward_pad_steps = torch.minimum(self.max_forward_steps - forward_steps, learning_steps)

        hidden = []
        for hidden_seq, start_idx, end_idx, padding_length in zip(recurrent_output, seq_start_idx, seq_len,
                                                                  forward_pad_steps):
            hidden.append(hidden_seq[start_idx:end_idx])
            if padding_length > 0:
                hidden.append(hidden_seq[end_idx - 1:end_idx].repeat(padding_length, 1))

        hidden = torch.cat(hidden)

        assert hidden.size(0) == torch.sum(learning_steps)

        adv = self.advantage(hidden)
        val = self.value(hidden)[legal_idx, legal_opponent_idx]
        q_value = val + adv - adv.mean(1, keepdim=True)

        return q_value

    # TODO: rewrite this bullshit using the new syntax
    def calculate_q(self, obs, last_action, last_opponent_action, last_reward, hidden_state, burn_in_steps,
                    learning_steps, legal_idx, legal_opponent_idx):
        # obs shape: (batch_size, seq_len, obs_shape)
        batch_size, max_seq_len, *_ = obs.size()

        obs = obs.reshape(-1, *self.obs_shape)
        last_action = last_action.view(-1, self.action_dim)
        last_opponent_action = last_opponent_action.view(-1, self.action_dim)
        last_reward = last_reward.view(-1, 1)

        latent = self.feature(obs)

        seq_len = burn_in_steps + learning_steps

        recurrent_input = torch.cat((latent, last_action, last_opponent_action, last_reward), dim=1)
        recurrent_input = recurrent_input.view(batch_size, max_seq_len, -1)
        recurrent_input = pack_padded_sequence(recurrent_input, seq_len, batch_first=True, enforce_sorted=False)

        # self.recurrent.flatten_parameters()
        recurrent_output, _ = self.recurrent(recurrent_input, hidden_state)

        recurrent_output, _ = pad_packed_sequence(recurrent_output, batch_first=True)

        hidden = torch.cat([output[burn_in:burn_in + learning] for output, burn_in, learning in
                            zip(recurrent_output, burn_in_steps, learning_steps)], dim=0)

        adv = self.advantage(hidden)
        val = self.value(hidden)[legal_idx, legal_opponent_idx]

        q_value = val + adv - adv.mean(1, keepdim=True)

        return q_value
