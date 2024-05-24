import numpy as np
import torch
from gymnasium.spaces import Space, Box, MultiDiscrete
import nashpy

from poke_env.player import Gen8EnvSinglePlayer, EnvPlayer
from poke_env.environment import AbstractBattle, Battle

from .r2d2 import Network, AgentState, Option
from .r2d2 import reverse_move_lookup, reverse_dex_lookup, move_lookup, dex_lookup, chart, action_embed_size
from .r2d2 import uk_move_idx, uk_mon_idx

action_space = np.ones(action_embed_size)
action_space[uk_move_idx] = 5
action_space[uk_mon_idx] = 6


def get_options_both(battle: Battle) -> tuple[Option, Option]:
    self_moves = battle.available_moves
    self_switches = battle.available_switches
    opponent_switches = [mon[3:] for mon in battle.opponent_team if not battle.opponent_team[mon].fainted]
    opponent_unrevealed = 6 - len(opponent_switches)
    if opponent_unrevealed != 0:
        opponent_switches.append("UNKNOWN")
    for data in battle.opponent_team.values():
        if data.active:
            opponent_moves = list(data.moves.keys())
            break
    else:
        raise ValueError("no active pokemon in opponents team")
    opponent_unrevealed_moves = 4 - len(opponent_moves)
    if opponent_unrevealed_moves != 0:
        opponent_moves.append("UNKNOWN")
    return (Option(self_moves, self_switches, 0, 0),
            Option(opponent_moves, opponent_switches,
             opponent_unrevealed, opponent_unrevealed_moves))


def get_option_idxs(option: Option) -> list[int]:
    idxs = []
    for mon in option.pokemon:
        idxs.append(reverse_dex_lookup[mon])
    for move in option.moves:
        idxs.append(reverse_move_lookup[move])
    return idxs


def decode_move(idx):
    return dex_lookup.get(idx, move_lookup[idx])


low = np.array([-1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
high = np.array([3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1])


class Minimax(Gen8EnvSinglePlayer):
    action_space = MultiDiscrete(action_space)
    observation_space = Box(low, high)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy: Network | None = None
        #self.history = []
        self.hidden = None

    def set_policy(self, policy: Network):
        self.policy = policy

    # TODO: add dynamax shit
    def choose_move(self, battle: Battle) -> str:
        # steps:
        # 1. decode battle to encoding
        # get possible actions for both players
        options1, options2 = get_options_both(battle)
        self_idxs = get_option_idxs(options1)
        opponent_idxs = get_option_idxs(options2)
        state_embedding = self.embed_battle(battle)
        agent_state = AgentState(
            state_embedding,
            self.hidden,
            self_idxs,
            opponent_idxs,
            options2.unrevealed_pokemon,
            options2.unrevealed_moves)
        # 2. pass into model
        with torch.no_grad():
            q_matrix, hidden = self.model(agent_state)
        self.hidden = hidden
        game = nashpy.Game(q_matrix)
        pa, pb = game.support_enumeration(tol=1e-6)
        move = np.random.choice(agent_state.legal_moves_idx, pa)
        # 3. decode output
        return decode_move(move)

    # TODO: tune this
    def calc_reward(self, last_battle, current_battle) -> float:
        return self.reward_computing_helper(
            current_battle, fainted_value=1.0, hp_value=.5, victory_value=10.0
        )

    # TODO: experiment with more expressive embeddings
    # idea, make big one hot encoded vector then pass it through a feed forward nn
    # before passing it to rnn, similar to applying cnn first for atari games
    def embed_battle(self, battle: AbstractBattle):
        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        move_acc = np.ones(4)
        move_pp = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = move.base_power / 100
            move_acc[i] = move.accuracy / 100
            move_pp[i] = move.current_pp / 24
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    type_1=battle.opponent_active_pokemon.type_1,
                    type_2=battle.opponent_active_pokemon.type_2,
                    type_chart=chart
                )

        # We count how many pokemons have fainted in each team
        fainted_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6
        fainted_mon_opponent = (
                len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )

        final_vector = np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                move_acc,
                move_pp,
                [fainted_mon_team, fainted_mon_opponent],
            ]
        )
        return np.float32(final_vector)

    def describe_embedding(self) -> Space:
        return self.observation_space

    """async def reset(self, challenge=True, **kwargs):
        if challenge:
            self.background_send_challenge(self._opponent.username)
        else:
            self.background_accept_challenge(self._opponent.username)
        self.hidden = None
        self.current_battle = self.agent.current_battle
        self.current_battle.logger = None
        self.last_battle = self.current_battle
        return self._observations.get(), self.get_additional_info()"""



