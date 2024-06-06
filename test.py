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

env = Minimax(None)
start_iter = 2740
batch_size = 64
games_per_run = 16
min_buffer_size = 2 * batch_size
model = Network(env.action_space.shape[0], env.observation_size, config.hidden_dim)
model.load_state_dict(torch.load(f"/home/mathy/PycharmProjects/stockfisk/models/{start_iter}.pth"))
model.eval()
agent = Minimax(model)
agent.ladder(1000)

