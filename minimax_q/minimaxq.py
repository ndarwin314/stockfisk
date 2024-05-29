import numpy as np
import torch
from gymnasium.spaces import Space, Box, MultiDiscrete
from gymnasium import Env
import nashpy
from time import sleep

from poke_env.player import Gen8EnvSinglePlayer, Player, BattleOrder, ForfeitBattleOrder
from poke_env.environment import AbstractBattle, Battle, Move, SideCondition, Weather, Field, Status, Pokemon

from minimax_q.r2d2 import Network, AgentState, Option
from minimax_q.r2d2 import reverse_move_lookup, reverse_dex_lookup, move_lookup, dex_lookup, chart, action_embed_size
from minimax_q.r2d2 import uk_move_idx, uk_mon_idx

action_space = np.ones(action_embed_size)
action_space[uk_move_idx] = 5
action_space[uk_mon_idx] = 6


def embed_pokemon(mon: Pokemon):
    health = mon.current_hp_fraction
    one_hot = np.zeros(len(dex_lookup))
    one_hot[reverse_dex_lookup[mon.species]] = 1.0
    moves = np.zeros(len(move_lookup))
    for k, v in mon.moves.items():
        moves[reverse_move_lookup[k] - len(dex_lookup)] = v.current_pp / 24
    status = np.zeros(len(Status))
    if mon.status is not None:
        status[mon.status.value - 1] = 1.0
    boosts = np.zeros(7)
    for i, stat in enumerate(["accuracy", "atk", "def", "evasion", "spa", "spd", "spe"]):
        boosts[i] = mon.boosts.get(stat, 0) / 6
    # TODO: item
    return np.concatenate(
        [
            one_hot,
            np.array([health]),
            moves,
            status,
            boosts])


mon_embed_size = 2297
empty_mon_embed = np.zeros(mon_embed_size)


def get_options_both(battle: Battle) -> tuple[Option, Option]:
    self_moves = [move.id for move in battle.available_moves]
    self_switches = [mon.species for mon in battle.available_switches]
    self_option = Option(self_switches, self_moves, 0, 0)
    # kinda hacky but this should check if the user has a move but the opponent doesnt
    # because either switching from faint/switch move
    if (battle.active_pokemon.fainted) or len(self_moves) == 0:
        opponent_switches = []
        opponent_moves = ["noop"]
        opponent_unrevealed = 0
        opponent_unrevealed_moves = 0
    else:
        opponent_switches = [battle.opponent_team[mon].species for mon in battle.opponent_team if
                             not (battle.opponent_team[mon].fainted or battle.opponent_team[mon].active)]
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
    opponent_option = Option(opponent_switches, opponent_moves, opponent_unrevealed, opponent_unrevealed_moves)
    return self_option, opponent_option


def get_option_idxs(option: Option) -> list[int]:
    idxs = []
    for mon in option.pokemon:
        idxs.append(reverse_dex_lookup[mon])
    for move in option.moves:
        idxs.append(reverse_move_lookup[move])
    return idxs


low = np.array([-1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
high = np.array([3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1])


class Minimax(Player, Env):
    action_space = MultiDiscrete(action_space)
    observation_space = Box(low, high)
    observation_size = 27691

    def __init__(self, model, max_len=100, *args, **kwargs):
        super().__init__(battle_format="gen8randombattle", *args, **kwargs)
        assert 0 < max_len <= 1000
        self.max_len = max_len
        self.turn = 0
        self.policy: Network = model
        self.history = [[] for _ in range(max_len)]
        self.actions = [[] for _ in range(max_len)]
        self.opponent_actions = [[] for _ in range(max_len)]
        self.qs = [[] for _ in range(max_len)]
        self.hidden = None
        self.rng = np.random.default_rng()
        self._reward_buffer: dict[AbstractBattle, float] = {}

    def set_policy(self, policy: Network):
        self.policy = policy

    def decode_move(self, battle: Battle, idx: int):
        if idx in dex_lookup:
            temp = dex_lookup[idx]
            for mon in battle.available_switches:
                if mon.species == temp:
                    temp = mon
                    break
        else:
            temp = move_lookup[idx]
            if temp == "struggle":
                return BattleOrder(Move(temp, 8))
            try:
                temp = battle.active_pokemon.moves[temp]
            except KeyError:
                # probably zoroark bullshit going on
                return BattleOrder(Move(temp, 8))
        return BattleOrder(temp)

    # TODO: add dynamax shit
    def choose_move(self, battle: Battle):
        self.turn = battle.turn
        if self.turn >= self.max_len:
            return ForfeitBattleOrder()
        # steps:
        # 1. decode battle to encoding
        # get possible actions for both players
        options1, options2 = get_options_both(battle)
        self_idxs = get_option_idxs(options1)
        opponent_idxs = get_option_idxs(options2)
        state_embedding = self.embed_battle(battle)
        agent_state = AgentState(
            torch.tensor(state_embedding),
            self.hidden,
            self_idxs,
            opponent_idxs,
            options2.unrevealed_pokemon,
            options2.unrevealed_moves)
        # 2. pass into model
        with torch.no_grad():
            q_matrix, hidden = self.policy(agent_state)
        self.hidden = hidden
        if q_matrix.shape[0] == 0:
            return self.choose_random_move(battle)
        game = nashpy.Game(q_matrix)
        test = game.support_enumeration(tol=1e-6)
        pa, pb = next(test)
        q = game[pa, pb][0]
        self.qs[self.turn-1].append(q)
        move = self.rng.choice(agent_state.legal_moves_idx, p=pa)
        self.actions[self.turn-1].append(move)
        self.history[self.turn-1].append(agent_state)
        # 3. decode output
        #print(self.username, ": ", self.turn, ": ", self.decode_move(battle, move))
        return self.decode_move(battle, move)

    # TODO: tune this
    def calc_reward(self, last_battle, current_battle) -> float:
        return self.reward_computing_helper(
            current_battle, fainted_value=1.0, hp_value=.5, victory_value=10.0
        )

    # TODO: experiment with more expressive embeddings
    # idea, make big one hot encoded vector then pass it through a feed forward nn
    # before passing it to rnn, similar to applying cnn first for atari games
    def embed_battle(self, battle: AbstractBattle):
        # We count how many pokemons have fainted in each team
        fainted_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6
        fainted_mon_opponent = (
                len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )
        weather = np.zeros(len(Weather) * 2)
        for k,v in battle.weather.items():
            num = k.value-1
            weather[num] = 1
            weather[num + len(Weather)] = v
        field = np.zeros(len(Field) * 2)
        for k, v in battle.fields.items():
            num = k.value-1
            field[num] = 1
            field[num + len(Field)] = v
        self_side = np.zeros(len(SideCondition) * 2)
        for k, v in battle.side_conditions.items():
            num = k.value-1
            self_side[num] = 1
            self_side[num + len(SideCondition)] = v
        opponent_side = np.zeros(len(SideCondition) * 2)
        for k, v in battle.side_conditions.items():
            num = k.value-1
            opponent_side[num] = 1
            opponent_side[num + len(SideCondition)] = v
        self_reserve = [embed_pokemon(mon) for mon in battle.available_switches]
        opponent_reserve = [embed_pokemon(battle.opponent_team[mon]) for mon in battle.opponent_team if
                             not (battle.opponent_team[mon].fainted or battle.opponent_team[mon].active)]
        while len(self_reserve) < 5:
            self_reserve.append(empty_mon_embed)
        while len(opponent_reserve) < 5:
            opponent_reserve.append(empty_mon_embed)
        final_vector = np.concatenate(
            [
                np.array([battle.turn]),
                np.array([fainted_mon_team, fainted_mon_opponent]),
                weather,
                field,
                self_side,
                opponent_side,
                embed_pokemon(battle.active_pokemon),
                embed_pokemon(battle.opponent_active_pokemon),
            ] + self_reserve + opponent_reserve
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

    def reset(self, **kwargs):
        super().reset(**kwargs)
        self.hidden = None
        self.history = [[] for _ in range(self.max_len)]
        self.actions = [[] for _ in range(self.max_len)]
        self.opponent_actions = [[] for _ in range(self.max_len)]
        self.qs = [[] for _ in range(self.max_len)]
        self.turn = 0

    def set_model(self, model: Network):
        self.policy = model

    def reward_computing_helper(
        self,
        battle: AbstractBattle,
        *,
        fainted_value: float = 0.0,
        hp_value: float = 0.0,
        number_of_pokemons: int = 6,
        starting_value: float = 0.0,
        status_value: float = 0.0,
        victory_value: float = 1.0,
    ) -> float:
        """A helper function to compute rewards.

        The reward is computed by computing the value of a game state, and by comparing
        it to the last state.

        State values are computed by weighting different factor. Fainted pokemons,
        their remaining HP, inflicted statuses and winning are taken into account.

        For instance, if the last time this function was called for battle A it had
        a state value of 8 and this call leads to a value of 9, the returned reward will
        be 9 - 8 = 1.

        Consider a single battle where each player has 6 pokemons. No opponent pokemon
        has fainted, but our team has one fainted pokemon. Three opposing pokemons are
        burned. We have one pokemon missing half of its HP, and our fainted pokemon has
        no HP left.

        The value of this state will be:

        - With fainted value: 1, status value: 0.5, hp value: 1:
            = - 1 (fainted) + 3 * 0.5 (status) - 1.5 (our hp) = -1
        - With fainted value: 3, status value: 0, hp value: 1:
            = - 3 + 3 * 0 - 1.5 = -4.5

        :param battle: The battle for which to compute rewards.
        :type battle: AbstractBattle
        :param fainted_value: The reward weight for fainted pokemons. Defaults to 0.
        :type fainted_value: float
        :param hp_value: The reward weight for hp per pokemon. Defaults to 0.
        :type hp_value: float
        :param number_of_pokemons: The number of pokemons per team. Defaults to 6.
        :type number_of_pokemons: int
        :param starting_value: The default reference value evaluation. Defaults to 0.
        :type starting_value: float
        :param status_value: The reward value per non-fainted status. Defaults to 0.
        :type status_value: float
        :param victory_value: The reward value for winning. Defaults to 1.
        :type victory_value: float
        :return: The reward.
        :rtype: float
        """
        if battle not in self._reward_buffer:
            self._reward_buffer[battle] = starting_value
        current_value = 0

        for mon in battle.team.values():
            current_value += mon.current_hp_fraction * hp_value
            if mon.fainted:
                current_value -= fainted_value
            elif mon.status is not None:
                current_value -= status_value

        current_value += (number_of_pokemons - len(battle.team)) * hp_value

        for mon in battle.opponent_team.values():
            current_value -= mon.current_hp_fraction * hp_value
            if mon.fainted:
                current_value += fainted_value
            elif mon.status is not None:
                current_value += status_value

        current_value -= (number_of_pokemons - len(battle.opponent_team)) * hp_value

        if battle.won:
            current_value += victory_value
        elif battle.lost:
            current_value -= victory_value

        to_return = current_value - self._reward_buffer[battle]
        self._reward_buffer[battle] = current_value

        return to_return


