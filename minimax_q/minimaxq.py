import copy

import numpy
import numpy as np
import scipy as sp
import torch
from gymnasium.spaces import Space, Box, MultiDiscrete
from gymnasium import Env
import nashpy
from time import sleep
from typing import Union

from poke_env.player import Gen8EnvSinglePlayer, Player, BattleOrder, ForfeitBattleOrder
from poke_env.environment import AbstractBattle, Battle, Move, SideCondition, Weather, Field, Status, Pokemon, PokemonType, EmptyMove

from minimax_q.r2d2 import Network, AgentState, Option
from minimax_q.r2d2 import reverse_move_lookup, reverse_dex_lookup, move_lookup, dex_lookup, chart, action_embed_size
from minimax_q.r2d2 import uk_move_idx, uk_mon_idx

#action_space[uk_move_idx] = 5
#action_space[uk_mon_idx] = 6
boost_list = ["accuracy", "atk", "def", "evasion", "spa", "spd", "spe"]
stat_list = ["hp", "atk", "def", "spa", "spd", "spe"]

rng = np.random.default_rng()

int_types = Union[int, numpy.int64, np.int32, np.int16]


def embed_move(move: Move, real=True) -> np.ndarray:
    base_power = (move.base_power - 0.8) / 100.0
    accuracy = (move.accuracy - 0.8) / 100.0
    current_pp = move.current_pp / 24
    category = move.category.value - 1
    category_encode = np.zeros(3)
    category_encode[category] = 1
    crit_ratio = move.crit_ratio
    try:
        damage = move.damage / 100
    # sue me
    except TypeError:
        damage = 0.8
    drain = move.drain
    heal = move.heal
    boosts = move.boosts
    boost_encode = np.zeros(7)
    if boosts is not None:
        for k, v in boosts.items():
            boost_encode[boost_list.index(k)] = v
    self_boosts = move.self_boost
    self_boost_encode = np.zeros(7)
    if self_boosts is not None:
        for k, v in self_boosts.items():
            self_boost_encode[boost_list.index(k)] = v
    drag = move.force_switch
    priority = move.priority
    recoil = move.recoil
    self_switch = move.self_switch
    secondaries = move.secondary
    # i'm deciding to only deal with status secondary effects since there are too many
    # nvm we are not dealing with it at all
    side_condition = move.side_condition
    side_encode = np.zeros(len(SideCondition))
    if side_condition is not None:
        idx = SideCondition.from_string(side_condition).value - 1
        side_encode[idx] = 1
    type = move.type.value - 1
    type_encode = np.zeros(len(PokemonType))
    type_encode[type] = 1
    scalars = [
        real,
        base_power,
        accuracy,
        current_pp,
        crit_ratio,
        damage,
        drain,
        heal,
        priority,
        recoil,
        self_switch,
        drag
    ]
    test = [np.array(scalars), category_encode, type_encode, side_encode, boost_encode, self_boost_encode]
    return np.concatenate(test)


move_size = 69
empty_move_embed = np.zeros(move_size)
empty_move_embed[0] = 1
noop_move_embed = np.zeros(move_size)


def embed_pokemon(mon: Pokemon):
    health = mon.current_hp_fraction
    alive = not mon.fainted
    stats = np.zeros(6)
    for i, v in enumerate(mon.base_stats.values()):
        stats[i] = v / 100
    move_list = mon.moves
    moves = []
    for v in move_list.values():
        moves.append(embed_move(v))
    while len(moves) < 4:
        moves.append(empty_move_embed)
    type_encode = np.zeros(2 * len(PokemonType))
    type1 = mon.type_1.value - 1
    type_encode[type1] = 1
    type2 = mon.type_2
    if type2 is not None:
        type_encode[type2.value - 1 + len(PokemonType)] = 1

    status = np.zeros(len(Status))
    if mon.status is not None:
        status[mon.status.value - 1] = 1.0
    boosts = np.zeros(7)
    for i, stat in enumerate(boost_list):
        boosts[i] = mon.boosts.get(stat, 0) / 6
    # TODO: item and ability
    arr = np.concatenate(
        [
            np.array([alive, health]),
            stats,
            status,
            boosts,
            type_encode
        ] + moves)
    return arr


mon_embed_size = 2 + 6 + len(Status) + 7 + 2 * len(PokemonType) + 4 * move_size
empty_mon_embed = np.zeros(mon_embed_size)
action_space = np.ones(move_size + mon_embed_size)


def get_options_both(battle: Battle) -> tuple[Option, Option]:
    self_moves = [move.id for move in battle.available_moves]
    move_embeddings = [embed_move(move) for move in battle.available_moves]
    self_switches = [mon.species for mon in battle.available_switches]
    mon_embeddings = [embed_pokemon(mon) for mon in battle.available_switches]
    self_option = Option(self_switches, self_moves, 0, 0, mon_embeddings, move_embeddings)
    # kinda hacky but this should check if the user has a move but the opponent doesnt
    # because either switching from faint/switch move
    if (battle.active_pokemon.fainted) or len(self_moves) == 0:
        opponent_switches = []
        mon_embeddings = []
        opponent_moves = ["noop"]
        move_embeddings = [noop_move_embed]
        opponent_unrevealed = 0
        opponent_unrevealed_moves = 0
    else:
        opponent_switches = [battle.opponent_team[mon].species for mon in battle.opponent_team if
                             not (battle.opponent_team[mon].fainted or battle.opponent_team[mon].active)]
        opponent_unrevealed = 6 - len(opponent_switches)
        mon_embeddings = [embed_pokemon(battle.opponent_team[mon]) for mon in battle.opponent_team if
                             not (battle.opponent_team[mon].fainted or battle.opponent_team[mon].active)]
        if opponent_unrevealed != 0:
            opponent_switches.append("UNKNOWN")
            c = copy.copy(empty_mon_embed)
            c[0] = 1
            mon_embeddings.append(c)
        for data in battle.opponent_team.values():
            if data.active:
                opponent_moves = list(data.moves.keys())
                move_embeddings = [embed_move(move) for move in data.moves.values()]
                break
        else:
            raise ValueError("no active pokemon in opponents team")
        opponent_unrevealed_moves = 4 - len(opponent_moves)
        if opponent_unrevealed_moves != 0:
            opponent_moves.append("UNKNOWN")
            move_embeddings.append(empty_move_embed)
    opponent_option = Option(opponent_switches, opponent_moves, opponent_unrevealed, opponent_unrevealed_moves, mon_embeddings, move_embeddings)
    return self_option, opponent_option


def get_option_idxs(option: Option) -> (list[int], dict[int, np.array]):
    idxs = []
    embeddings = {}
    for mon, embed in zip(option.pokemon, option.pokemon_embeddings):
        idx = reverse_dex_lookup[mon]
        idxs.append(idx)
        embeddings[idx] = np.concatenate([embed, empty_move_embed])
    for move, embed in zip(option.moves, option.move_embeddings):
        idx = reverse_move_lookup[move]
        idxs.append(idx)
        embeddings[idx] = np.concatenate([empty_mon_embed, embed])
    return idxs, embeddings



low = np.array([-1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
high = np.array([3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1])


class Minimax(Player, Env):
    action_space = MultiDiscrete(action_space)
    observation_space = Box(low, high)
    observation_size = 4183

    def __init__(self, model, max_len=200, epsilon=0.2, account_configuration=None, *args, **kwargs):
        super().__init__(battle_format="gen8randombattle", account_configuration=account_configuration, *args, **kwargs)
        assert 0 < max_len <= 1000
        self.max_len = max_len
        self.epsilon = epsilon
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
        self_idxs, self_embeds = get_option_idxs(options1)
        opponent_idxs, opponent_embeds = get_option_idxs(options2)
        state_embedding = self.embed_battle(battle)
        agent_state = AgentState(
            state_embedding,
            self.hidden,
            self_idxs,
            opponent_idxs,
            options2.unrevealed_pokemon,
            options2.unrevealed_moves,
            self.calc_reward(current_battle=battle),
            self_embeds,
            opponent_embeds
        )
        # 2. pass into model
        with torch.no_grad():
            q_matrix, hidden = self.policy(agent_state)
        self.hidden = hidden
        self.hidden = hidden
        if q_matrix.shape[0] == 0:
            return self.choose_random_move(battle)
        game = nashpy.Game(q_matrix)
        test = game.support_enumeration(tol=1e-6)
        try:
            pa, pb = next(test)
            q = game[pa, pb][0]
            print(q)
            self.qs[self.turn - 1].append(q)
            if rng.uniform() > self.epsilon:
                move = self.rng.choice(agent_state.legal_moves_idx, p=pa)
            else:
                move = self.choose_random_move(battle)
        except StopIteration:
            self.qs[self.turn - 1].append(0)
            move = self.choose_random_move(battle)
        if isinstance(move, int_types):
            self.actions[self.turn-1].append(move)
            self.history[self.turn-1].append(agent_state)
            # 3. decode output
            #print(self.username, ": ", self.turn, ": ", self.decode_move(battle, move))
            return self.decode_move(battle, move)
        elif isinstance(move, BattleOrder):
            order = move.order
            if isinstance(order, Move):
                move1 = reverse_move_lookup[order.id]
            elif isinstance(order, Pokemon):
                move1 = reverse_dex_lookup[order.species]
            self.actions[self.turn-1].append(move1)
            self.history[self.turn-1].append(agent_state)
            return move

    # TODO: tune this
    def calc_reward(self, current_battle) -> float:
        return self.reward_computing_helper(
            current_battle, fainted_value=.2, hp_value=.05, status_value=.03, victory_value=3.0
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
        return final_vector

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
        self._battles = {}

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


