import asyncio
import time

import numpy as np
import torch
import matplotlib.pyplot as plt

from poke_env.player import RandomPlayer, MaxBasePowerPlayer, SimpleHeuristicsPlayer
from poke_env.ps_client import AccountConfiguration

from minimax_q.minimaxq import Minimax
from minimax_q.r2d2 import Network
import minimax_q.config as config


async def compare_to_random(end_epoch):
    con = AccountConfiguration("random 3", None)
    random = RandomPlayer(battle_format="gen8randombattle", account_configuration=con)
    max_player = MaxBasePowerPlayer(battle_format="gen8randombattle")
    heuristic_player = SimpleHeuristicsPlayer(battle_format="gen8randombattle")
    step_size = 100
    num_battles = 100
    start = 40
    x = np.arange(start, end_epoch, step_size)
    y = np.zeros((3, len(range(start, end_epoch, step_size))))
    con = AccountConfiguration("Minimax 5", None)
    agent = Minimax(None, epsilon=1e-3, account_configuration=con)
    model = Network(agent.action_space.shape[0], agent.observation_size, config.hidden_dim)
    model.eval()
    agent.set_model(model)
    start_time = time.time()
    for i, epoch in enumerate(range(start, end_epoch, step_size)):
        print(i, epoch)
        model.load_state_dict(torch.load(f"/home/mathy/PycharmProjects/stockfisk/models/{epoch}.pth"))
        await agent.battle_against(random, num_battles)
        y[0, i] = agent.n_won_battles
        agent.reset()
        await agent.battle_against(max_player, num_battles)
        y[1, i] = agent.n_won_battles
        await agent.battle_against(heuristic_player, num_battles)
        y[2, i] = agent.n_won_battles
        agent.reset()
        print(y[:, i])
        print(time.time() - start_time)
    y /= num_battles
    np.save(f"x_arr_{end_epoch}", x)
    np.save(f"y_arr_{end_epoch}", y)
    plt.plot(x, y.T)
    plt.show()

asyncio.run(compare_to_random(3820))

