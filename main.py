from poke_env.player import Player, RandomPlayer

import poke_env.data as data
from minimax_q.r2d2 import Network
from minimax_q.minimaxq import Minimax
import minimax_q.config as config
import asyncio
import time
from gymnasium.utils.env_checker import check_env
from minimax_q.train import train_single_actor


async def test():
    start = time.time()
    model = Network(Minimax.action_space.shape[0], Minimax.observation_space.shape[0], config.hidden_dim)
    opponent = RandomPlayer(battle_format="gen8randombattle")
    train_env = Minimax(model=model)
    print("test")
    await train_env.battle_against(opponent, n_battles=1)

    print(
        "Max damage player won %d / 1 battles [this took %f seconds]"
        % (
            train_env.n_won_battles, time.time() - start
        )
    )


if __name__ == "__main__":
    asyncio.run(train_single_actor())

