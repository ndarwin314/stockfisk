from poke_env.player import Player, RandomPlayer

import poke_env.data as data
from minimax_q.minimaxq import Minimax
from gymnasium.utils.env_checker import check_env
from minimax_q.train import train

if __name__ == "__main__":
    opponent = RandomPlayer(battle_format="gen8randombattle")
    test_env = Minimax(
        opponent=opponent, start_challenging=True
    )
    check_env(test_env)
    test_env.close()

