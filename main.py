from poke_env.player import Player, RandomPlayer

import poke_env.data as data
from minimax_q.r2d2 import Network
from minimax_q.minimaxq import Minimax
import minimax_q.config as config
import asyncio
import time
import torch
from gymnasium.utils.env_checker import check_env
from minimax_q.train import train_single_actor, train
from poke_env import AccountConfiguration, ShowdownServerConfiguration


async def test():
    env = Minimax(None)
    start_iter = 2740
    model = Network(env.action_space.shape[0], env.observation_size, config.hidden_dim)
    model.load_state_dict(torch.load(f"/home/mathy/PycharmProjects/stockfisk/models/{start_iter}.pth"))
    model.eval()
    agent = Minimax(model, epsilon=.05)
    await agent.ladder(1000)


if __name__ == "__main__":
    asyncio.run(train())

