from __future__ import annotations

import functools
import logging
from collections import defaultdict, namedtuple
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import IntEnum
from itertools import product
from typing import Any

import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers


class Action(IntEnum):
    NONE = 0
    NORTH = 1
    SOUTH = 2
    WEST = 3
    EAST = 4
    LOAD = 5


class CellEntity(IntEnum):
    OUT_OF_BOUNDS = 0
    EMPTY = 1
    FOOD = 2
    AGENT = 3


@dataclass(eq=False)
class Player:
    controller: Any | None = None
    position: tuple[int, int] | None = None
    level: int = 0
    field_size: tuple[int, int] | None = None
    score: float = 0.0
    reward: float = 0.0
    history: list[Any] = field(default_factory=list)
    current_step: int = 0

    def setup(
        self,
        position: tuple[int, int],
        level: int,
        field_size: tuple[int, int],
    ) -> None:
        self.history = []
        self.position = position
        self.level = int(level)
        self.field_size = field_size
        self.score = 0.0

    def set_controller(self, controller: Any) -> None:
        self.controller = controller

    def step(self, obs: Any) -> Any:
        return self.controller._step(obs)

    @property
    def name(self) -> str:
        return self.controller.name if self.controller else "Player"


def env(**kwargs: Any):
    """AEC-style PettingZoo environment."""
    render_mode = kwargs.get("render_mode", None)
    internal_render_mode = "human" if render_mode == "ansi" else render_mode
    aec_env = raw_env(**{**kwargs, "render_mode": internal_render_mode})

    if render_mode == "ansi":
        aec_env = wrappers.CaptureStdoutWrapper(aec_env)

    aec_env = wrappers.AssertOutOfBoundsWrapper(aec_env)
    aec_env = wrappers.OrderEnforcingWrapper(aec_env)
    return aec_env


def raw_env(**kwargs: Any):
    """Convert the simultaneous-action ParallelEnv into an AEC env."""
    return parallel_to_aec(parallel_env(**kwargs))


def parallel_env(**kwargs: Any) -> "LevelBasedForagingParallelEnv":
    return LevelBasedForagingParallelEnv(**kwargs)


class LevelBasedForagingParallelEnv(ParallelEnv):
    """Level-Based Foraging as a PettingZoo Parallel environment.

    This is a native PettingZoo port of the upstream Gymnasium LBF game. The
    transition rules intentionally follow the original implementation: invalid
    state-dependent actions become ``NONE``, movement collisions cancel all
    colliding moves, and adjacent loading agents share normalized food rewards.
    """

    metadata = {
        "name": "level_based_foraging_v0",
        "render_modes": ["human", "rgb_array"],
        "is_parallelizable": True,
        "render_fps": 5,
    }

    Observation = namedtuple(
        "Observation",
        ["field", "actions", "players", "game_over", "sight", "current_step"],
    )
    PlayerObservation = namedtuple(
        "PlayerObservation",
        ["position", "level", "history", "reward", "is_self"],
    )
    action_set = [Action.NORTH, Action.SOUTH, Action.WEST, Action.EAST, Action.LOAD]

    def __init__(
        self,
        *,
        n_agents: int = 2,
        players: int | None = None,
        min_player_level: int | Iterable[int] = 1,
        max_player_level: int | Iterable[int] = 2,
        min_food_level: int | Iterable[int] = 1,
        max_food_level: int | Iterable[int] | None = None,
        field_size: tuple[int, int] = (8, 8),
        max_num_food: int = 1,
        sight: int | None = None,
        max_episode_steps: int | None = 50,
        max_cycles: int | None = None,
        force_coop: bool = False,
        normalize_reward: bool = True,
        grid_observation: bool = False,
        observe_agent_levels: bool = True,
        penalty: float = 0.0,
        fixed_layout_seed: int | None = None,
        layout: str | None = None,
        render_mode: str | None = None,
        render_cell_size: int = 48,
    ) -> None:
        if players is not None:
            n_agents = players
        if n_agents < 1:
            raise ValueError("n_agents must be at least 1.")
        if len(field_size) != 2:
            raise ValueError("field_size must be a (rows, columns) tuple.")
        rows, cols = int(field_size[0]), int(field_size[1])
        if rows < 3 or cols < 3:
            raise ValueError("field_size must be at least 3x3.")
        if max_num_food < 1:
            raise ValueError("max_num_food must be at least 1.")
        if max_episode_steps is not None and max_episode_steps < 1:
            raise ValueError("max_episode_steps must be positive when provided.")
        if max_cycles is not None:
            if max_cycles < 1:
                raise ValueError("max_cycles must be positive when provided.")
            max_episode_steps = max_cycles
        if sight is None:
            sight = max(rows, cols)
        if sight < 1:
            raise ValueError("sight must be at least 1.")
        if render_cell_size < 8:
            raise ValueError("render_cell_size must be at least 8.")
        if render_mode not in self.metadata["render_modes"] and render_mode is not None:
            raise ValueError(f"Unsupported render_mode: {render_mode!r}.")

        self.logger = logging.getLogger(__name__)
        self.n_agents = int(n_agents)
        self.field = np.zeros((rows, cols), dtype=np.int32)
        self.players = [Player() for _ in range(self.n_agents)]
        self.min_food_level = self._level_array(
            "min_food_level",
            min_food_level,
            max_num_food,
        )
        self.max_food_level = (
            None
            if max_food_level is None
            else self._level_array("max_food_level", max_food_level, max_num_food)
        )
        if self.max_food_level is not None:
            self._validate_level_bounds(
                "food",
                self.min_food_level,
                self.max_food_level,
            )

        self.max_num_food = int(max_num_food)
        self._food_spawned = 0.0
        self.min_player_level = self._level_array(
            "min_player_level",
            min_player_level,
            self.n_agents,
        )
        self.max_player_level = self._level_array(
            "max_player_level",
            max_player_level,
            self.n_agents,
        )
        self._validate_level_bounds(
            "player",
            self.min_player_level,
            self.max_player_level,
        )

        self.sight = int(sight)
        self.force_coop = bool(force_coop)
        self.penalty = float(penalty)
        self.max_episode_steps = max_episode_steps
        self.max_cycles = max_episode_steps
        self._normalize_reward = bool(normalize_reward)
        self._grid_observation = bool(grid_observation)
        self._observe_agent_levels = bool(observe_agent_levels)
        self.fixed_layout_seed = (
            None
            if fixed_layout_seed is None
            else int(fixed_layout_seed)
        )
        self.layout = self._normalize_layout(layout)
        if self.layout == "close_coop":
            self._validate_close_coop_layout_config()
        self.render_mode = render_mode
        self.render_cell_size = int(render_cell_size)

        self.possible_agents = [f"agent_{idx}" for idx in range(self.n_agents)]
        self.agent_name_mapping = {
            agent: idx for idx, agent in enumerate(self.possible_agents)
        }
        self.agents: list[str] = []
        self._action_spaces = {
            agent: spaces.Discrete(len(Action)) for agent in self.possible_agents
        }
        self._observation_space = self._get_observation_space()

        state_low = np.concatenate(
            (
                np.zeros(self.rows * self.cols, dtype=np.float32),
                np.tile(np.array([0, 0, 0], dtype=np.float32), self.n_agents),
            )
        )
        state_high = np.concatenate(
            (
                np.full(
                    self.rows * self.cols,
                    self._max_food_level_for_space(),
                    dtype=np.float32,
                ),
                np.tile(
                    np.array(
                        [
                            self.rows - 1,
                            self.cols - 1,
                            int(np.max(self.max_player_level)),
                        ],
                        dtype=np.float32,
                    ),
                    self.n_agents,
                ),
            )
        )
        self.state_space = spaces.Box(
            low=state_low,
            high=state_high,
            dtype=np.float32,
        )

        self._valid_actions: dict[Player, list[Action]] = {}
        self._game_over = False
        self.current_step = 0
        self.np_random, self.np_random_seed = seeding.np_random(None)
        self._viewer = None
        self._last_observations: dict[str, np.ndarray] = {}
        self._last_infos: dict[str, dict[str, Any]] = {}
        self._last_load_attempted_agents: frozenset[str] = frozenset()
        self._last_failed_load_agents: frozenset[str] = frozenset()
        self._last_successful_load_agents: frozenset[str] = frozenset()
        self._last_coop_load_violation_agents: frozenset[str] = frozenset()

    @staticmethod
    def _level_array(
        name: str,
        value: int | Iterable[int],
        length: int,
    ) -> np.ndarray:
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            levels = np.asarray(list(value), dtype=np.int32)
            if len(levels) != length:
                raise ValueError(f"{name} must be a scalar or length {length}.")
            return levels
        return np.full(length, int(value), dtype=np.int32)

    @staticmethod
    def _validate_level_bounds(
        kind: str,
        minimum: np.ndarray,
        maximum: np.ndarray,
    ) -> None:
        for idx, (min_level, max_level) in enumerate(zip(minimum, maximum)):
            if min_level > max_level:
                raise ValueError(
                    f"min_{kind}_level must be <= max_{kind}_level for item {idx}."
                )

    @staticmethod
    def _normalize_layout(layout: str | None) -> str | None:
        if layout is None:
            return None
        normalized = str(layout).strip().lower().replace("-", "_")
        if normalized in {"", "none", "random", "default"}:
            return None
        if normalized == "close_coop":
            return normalized
        raise ValueError(f"Unsupported LBF layout: {layout!r}.")

    @property
    def field_size(self) -> tuple[int, int]:
        return self.field.shape

    @property
    def rows(self) -> int:
        return int(self.field.shape[0])

    @property
    def cols(self) -> int:
        return int(self.field.shape[1])

    @property
    def game_over(self) -> bool:
        return self._game_over

    @property
    def unwrapped(self) -> "LevelBasedForagingParallelEnv":
        return self

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str):
        _ = agent
        return self._observation_space

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str):
        return self._action_spaces[agent]

    def seed(self, seed: int | None = None) -> list[int | None]:
        if seed is not None:
            self.np_random, self.np_random_seed = seeding.np_random(seed)
        return [self.np_random_seed]

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        _ = options
        reset_seed = (
            self.fixed_layout_seed
            if self.fixed_layout_seed is not None
            else seed
        )
        if reset_seed is not None:
            self.np_random, self.np_random_seed = seeding.np_random(reset_seed)

        self.field = np.zeros(self.field_size, dtype=np.int32)
        for player in self.players:
            player.position = None
            player.level = 0
            player.reward = 0.0
        if self.layout == "close_coop":
            self._spawn_close_coop_layout()
        else:
            self.spawn_players(self.min_player_level, self.max_player_level)
            player_levels = sorted(player.level for player in self.players)
            derived_max_food = np.full(
                self.max_num_food,
                sum(player_levels[: min(3, len(player_levels))]),
                dtype=np.int32,
            )
            self.spawn_food(
                self.max_num_food,
                min_levels=self.min_food_level,
                max_levels=self.max_food_level
                if self.max_food_level is not None
                else derived_max_food,
            )
        self.current_step = 0
        self._game_over = False
        self.agents = self.possible_agents[:]
        self._clear_load_events()
        self._gen_valid_moves()

        self._last_observations = self._make_parallel_observations()
        self._last_infos = {agent: self._info_for_agent(agent) for agent in self.agents}

        if self.render_mode == "human":
            self.render()

        return dict(self._last_observations), dict(self._last_infos)

    def step(self, actions: dict[str, Any]):
        if not self.agents:
            return {}, {}, {}, {}, {}

        current_agents = self.agents[:]
        missing = [agent for agent in current_agents if agent not in actions]
        if missing:
            raise KeyError(f"Missing actions for agents: {missing}")

        ordered_actions: list[Action] = []
        for agent in current_agents:
            action = int(actions[agent])
            if not self.action_space(agent).contains(action):
                raise ValueError(f"Invalid action {action} for {agent!r}.")
            player = self.players[self.agent_name_mapping[agent]]
            candidate = Action(action)
            ordered_actions.append(
                candidate if candidate in self._valid_actions[player] else Action.NONE
            )

        self.current_step += 1
        self._clear_load_events()
        coop_ready_players = self._coop_load_ready_players()
        coop_violation_players = {
            player
            for player, action in zip(self.players, ordered_actions, strict=True)
            if player in coop_ready_players and action != Action.LOAD
        }
        self._last_coop_load_violation_agents = self._agent_ids_for_players(
            coop_violation_players
        )

        for player in self.players:
            player.reward = 0.0
            player.current_step = self.current_step

        loading_players: set[Player] = set()
        collisions: dict[tuple[int, int], list[Player]] = defaultdict(list)

        for player, action in zip(self.players, ordered_actions):
            assert player.position is not None
            row, col = player.position
            if action == Action.NONE:
                collisions[player.position].append(player)
            elif action == Action.NORTH:
                collisions[(row - 1, col)].append(player)
            elif action == Action.SOUTH:
                collisions[(row + 1, col)].append(player)
            elif action == Action.WEST:
                collisions[(row, col - 1)].append(player)
            elif action == Action.EAST:
                collisions[(row, col + 1)].append(player)
            elif action == Action.LOAD:
                collisions[player.position].append(player)
                loading_players.add(player)

        attempted_players = frozenset(loading_players)
        self._last_load_attempted_agents = self._agent_ids_for_players(attempted_players)

        for target, players in collisions.items():
            if len(players) == 1:
                players[0].position = target

        failed_players, successful_players = self._process_loading_players(
            loading_players
        )
        self._last_failed_load_agents = self._agent_ids_for_players(failed_players)
        self._last_successful_load_agents = self._agent_ids_for_players(
            successful_players
        )

        for player in self.players:
            player.score += player.reward

        all_food_collected = bool(self.field.sum() == 0)
        time_limit_hit = (
            self.max_episode_steps is not None
            and self.current_step >= self.max_episode_steps
        )
        env_termination = all_food_collected
        env_truncation = bool(time_limit_hit and not all_food_collected)
        self._game_over = bool(env_termination or env_truncation)
        self._gen_valid_moves()

        observations = self._make_parallel_observations(current_agents)
        rewards = {
            agent: float(self.players[self.agent_name_mapping[agent]].reward)
            for agent in current_agents
        }
        terminations = {agent: env_termination for agent in current_agents}
        truncations = {agent: env_truncation for agent in current_agents}
        infos = {agent: self._info_for_agent(agent) for agent in current_agents}
        if env_termination or env_truncation:
            reason = "all_food_collected" if env_termination else "max_episode_steps"
            for info in infos.values():
                info["episode_done"] = True
                info["episode_end_reason"] = reason
            self.agents = []

        self._last_observations = observations
        self._last_infos = infos

        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminations, truncations, infos

    def observe(self, agent: str) -> np.ndarray:
        if agent not in self.possible_agents:
            raise KeyError(f"Unknown agent {agent!r}.")
        return self._make_parallel_observations([agent])[agent]

    def state(self) -> np.ndarray:
        player_state = []
        for player in self.players:
            if player.position is None:
                player_state.extend((-1.0, -1.0, 0.0))
            else:
                player_state.extend(
                    (
                        float(player.position[0]),
                        float(player.position[1]),
                        float(player.level),
                    )
                )
        return np.concatenate(
            (
                self.field.astype(np.float32).reshape(-1),
                np.asarray(player_state, dtype=np.float32),
            )
        )

    def render(self):
        if self.render_mode is None:
            return None
        if self._viewer is None:
            from .rendering import Viewer

            self._viewer = Viewer(cell_size=self.render_cell_size)
        return self._viewer.render(
            self,
            return_rgb_array=self.render_mode == "rgb_array",
        )

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    def neighborhood(
        self,
        row: int,
        col: int,
        distance: int = 1,
        ignore_diag: bool = False,
    ):
        if not ignore_diag:
            return self.field[
                max(row - distance, 0) : min(row + distance + 1, self.rows),
                max(col - distance, 0) : min(col + distance + 1, self.cols),
            ]

        return (
            self.field[
                max(row - distance, 0) : min(row + distance + 1, self.rows), col
            ].sum()
            + self.field[
                row, max(col - distance, 0) : min(col + distance + 1, self.cols)
            ].sum()
        )

    def adjacent_food(self, row: int, col: int) -> int:
        return int(
            self.field[max(row - 1, 0), col]
            + self.field[min(row + 1, self.rows - 1), col]
            + self.field[row, max(col - 1, 0)]
            + self.field[row, min(col + 1, self.cols - 1)]
        )

    def adjacent_food_location(self, row: int, col: int) -> tuple[int, int] | None:
        if row > 0 and self.field[row - 1, col] > 0:
            return row - 1, col
        if row < self.rows - 1 and self.field[row + 1, col] > 0:
            return row + 1, col
        if col > 0 and self.field[row, col - 1] > 0:
            return row, col - 1
        if col < self.cols - 1 and self.field[row, col + 1] > 0:
            return row, col + 1
        return None

    def adjacent_players(self, row: int, col: int) -> list[Player]:
        adjacent = []
        for player in self.players:
            assert player.position is not None
            prow, pcol = player.position
            if (abs(prow - row) == 1 and pcol == col) or (
                abs(pcol - col) == 1 and prow == row
            ):
                adjacent.append(player)
        return adjacent

    def spawn_food(
        self,
        max_num_food: int,
        min_levels: np.ndarray,
        max_levels: np.ndarray,
    ) -> None:
        food_count = 0
        attempts = 0
        min_levels = max_levels if self.force_coop else min_levels

        food_permutation = self.np_random.permutation(max_num_food)
        min_levels = min_levels[food_permutation]
        max_levels = max_levels[food_permutation]

        while food_count < max_num_food and attempts < 1000:
            attempts += 1
            row = int(self.np_random.integers(1, self.rows - 1))
            col = int(self.np_random.integers(1, self.cols - 1))

            if (
                self.neighborhood(row, col).sum() > 0
                or self.neighborhood(row, col, distance=2, ignore_diag=True) > 0
                or not self._is_empty_location(row, col)
            ):
                continue

            min_level = int(min_levels[food_count])
            max_level = int(max_levels[food_count])
            self.field[row, col] = (
                min_level
                if min_level == max_level
                else int(self.np_random.integers(min_level, max_level + 1))
            )
            food_count += 1

        self._food_spawned = float(self.field.sum())

    def spawn_players(
        self,
        min_player_levels: np.ndarray,
        max_player_levels: np.ndarray,
    ) -> None:
        player_permutation = self.np_random.permutation(len(self.players))
        min_player_levels = min_player_levels[player_permutation]
        max_player_levels = max_player_levels[player_permutation]

        for player, min_level, max_level in zip(
            self.players,
            min_player_levels,
            max_player_levels,
        ):
            attempts = 0
            player.reward = 0.0
            while attempts < 1000:
                row = int(self.np_random.integers(0, self.rows))
                col = int(self.np_random.integers(0, self.cols))
                if self._is_empty_location(row, col):
                    level = int(self.np_random.integers(min_level, max_level + 1))
                    player.setup((row, col), level, self.field_size)
                    break
                attempts += 1
            if player.position is None:
                raise RuntimeError("Unable to spawn all players after 1000 attempts.")

    def _validate_close_coop_layout_config(self) -> None:
        if self.n_agents != 2:
            raise ValueError("layout='close_coop' requires exactly two agents.")
        if self.max_num_food != 1:
            raise ValueError("layout='close_coop' requires max_num_food=1.")
        if np.any(self.min_player_level > 1) or np.any(self.max_player_level < 1):
            raise ValueError(
                "layout='close_coop' requires player level ranges to include 1."
            )
        max_food_levels = (
            self.max_food_level
            if self.max_food_level is not None
            else np.asarray([self._max_food_level_for_space()], dtype=np.int32)
        )
        if int(self.min_food_level[0]) > 2 or int(max_food_levels[0]) < 2:
            raise ValueError(
                "layout='close_coop' requires the food level range to include 2."
            )

    def _spawn_close_coop_layout(self) -> None:
        food_row = self.rows // 2
        food_col = self.cols // 2
        left_position = (food_row, food_col - 1)
        right_position = (food_row, food_col + 1)

        self.field[food_row, food_col] = 2
        self._food_spawned = 2.0
        for player, position in zip(
            self.players,
            (left_position, right_position),
            strict=True,
        ):
            player.setup(position, 1, self.field_size)
            player.reward = 0.0
            player.current_step = 0

    def get_valid_actions(self) -> list[tuple[Action, ...]]:
        return list(product(*[self._valid_actions[player] for player in self.players]))

    def test_make_gym_obs(self):
        return self._make_ordered_observations()

    def test_gen_valid_moves(self) -> bool:
        try:
            self._gen_valid_moves()
        except Exception:
            return False
        return True

    def _get_observation_space(self):
        max_food_level = self._max_food_level_for_space()
        max_player_level = int(np.max(self.max_player_level))
        if not self._grid_observation:
            player_obs_len = 3 if self._observe_agent_levels else 2
            food_low = [-1, -1, 0] * self.max_num_food
            food_high = [
                self.rows - 1,
                self.cols - 1,
                max_food_level,
            ] * self.max_num_food
            if self._observe_agent_levels:
                player_low = [-1, -1, 0] * len(self.players)
                player_high = [
                    self.rows - 1,
                    self.cols - 1,
                    max_player_level,
                ] * len(self.players)
            else:
                player_low = [-1, -1] * len(self.players)
                player_high = [self.rows - 1, self.cols - 1] * len(self.players)
            low_obs = np.array(food_low + player_low, dtype=np.float32)
            high_obs = np.array(food_high + player_high, dtype=np.float32)
            assert len(low_obs) == self.max_num_food * 3 + player_obs_len * len(
                self.players
            )
        else:
            grid_shape = (1 + 2 * self.sight, 1 + 2 * self.sight)
            agents_min = np.zeros(grid_shape, dtype=np.float32)
            agents_high_value = max_player_level if self._observe_agent_levels else 1
            agents_max = np.full(grid_shape, agents_high_value, dtype=np.float32)
            foods_min = np.zeros(grid_shape, dtype=np.float32)
            foods_max = np.full(grid_shape, max_food_level, dtype=np.float32)
            access_min = np.zeros(grid_shape, dtype=np.float32)
            access_max = np.ones(grid_shape, dtype=np.float32)
            low_obs = np.stack([agents_min, foods_min, access_min])
            high_obs = np.stack([agents_max, foods_max, access_max])

        return spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)

    def _max_food_level_for_space(self) -> int:
        if self.max_food_level is not None:
            return int(np.max(self.max_food_level))
        player_levels = sorted(int(level) for level in self.max_player_level)
        return int(sum(player_levels[: min(3, len(player_levels))]))

    def _is_empty_location(self, row: int, col: int) -> bool:
        if self.field[row, col] != 0:
            return False
        for player in self.players:
            if player.position and row == player.position[0] and col == player.position[1]:
                return False
        return True

    def _is_valid_action(self, player: Player, action: Action) -> bool:
        assert player.position is not None
        row, col = player.position
        if action == Action.NONE:
            return True
        if action == Action.NORTH:
            return row > 0 and self.field[row - 1, col] == 0
        if action == Action.SOUTH:
            return row < self.rows - 1 and self.field[row + 1, col] == 0
        if action == Action.WEST:
            return col > 0 and self.field[row, col - 1] == 0
        if action == Action.EAST:
            return col < self.cols - 1 and self.field[row, col + 1] == 0
        if action == Action.LOAD:
            return self.adjacent_food(row, col) > 0

        self.logger.error("Undefined action %s from %s", action, player.name)
        raise ValueError(f"Undefined action {action}")

    def _gen_valid_moves(self) -> None:
        self._valid_actions = {
            player: [action for action in Action if self._is_valid_action(player, action)]
            for player in self.players
        }

    def _clear_load_events(self) -> None:
        self._last_load_attempted_agents = frozenset()
        self._last_failed_load_agents = frozenset()
        self._last_successful_load_agents = frozenset()
        self._last_coop_load_violation_agents = frozenset()

    def _agent_ids_for_players(self, players: Iterable[Player]) -> frozenset[str]:
        selected = set(players)
        return frozenset(
            self.possible_agents[index]
            for index, player in enumerate(self.players)
            if player in selected
        )

    def _coop_load_ready_players(self) -> frozenset[Player]:
        ready: set[Player] = set()
        for food_row, food_col in zip(*self.field.nonzero()):
            food_level = int(self.field[food_row, food_col])
            adjacent = self.adjacent_players(int(food_row), int(food_col))
            if len(adjacent) < 2:
                continue
            if sum(player.level for player in adjacent) < food_level:
                continue
            ready.update(
                player
                for player in adjacent
                if int(player.level) < food_level
            )
        return frozenset(ready)

    def _process_loading_players(
        self,
        loading_players: set[Player],
    ) -> tuple[frozenset[Player], frozenset[Player]]:
        failed_players: set[Player] = set()
        successful_players: set[Player] = set()
        while loading_players:
            player = loading_players.pop()
            food_location = self.adjacent_food_location(*player.position)
            if food_location is None:
                continue
            frow, fcol = food_location
            food = int(self.field[frow, fcol])

            adj_players = self.adjacent_players(frow, fcol)
            adj_players = [
                candidate
                for candidate in adj_players
                if candidate in loading_players or candidate is player
            ]
            adj_player_level = sum(candidate.level for candidate in adj_players)
            loading_players = loading_players - set(adj_players)

            if adj_player_level < food:
                for candidate in adj_players:
                    candidate.reward -= self.penalty
                failed_players.update(adj_players)
                continue

            for candidate in adj_players:
                candidate.reward = float(candidate.level * food)
                if self._normalize_reward and self._food_spawned > 0:
                    candidate.reward /= float(adj_player_level * self._food_spawned)
            successful_players.update(adj_players)
            self.field[frow, fcol] = 0
        return frozenset(failed_players), frozenset(successful_players)

    def _transform_to_neighborhood(
        self,
        center: tuple[int, int],
        sight: int,
        position: tuple[int, int],
    ) -> tuple[int, int]:
        return (
            position[0] - center[0] + min(sight, center[0]),
            position[1] - center[1] + min(sight, center[1]),
        )

    def _make_obs(self, player: Player):
        assert player.position is not None
        return self.Observation(
            actions=self._valid_actions[player],
            players=[
                self.PlayerObservation(
                    position=self._transform_to_neighborhood(
                        player.position,
                        self.sight,
                        other.position,
                    ),
                    level=other.level,
                    is_self=other == player,
                    history=other.history,
                    reward=other.reward if other == player else None,
                )
                for other in self.players
                if other.position is not None
                and min(
                    self._transform_to_neighborhood(
                        player.position,
                        self.sight,
                        other.position,
                    )
                )
                >= 0
                and max(
                    self._transform_to_neighborhood(
                        player.position,
                        self.sight,
                        other.position,
                    )
                )
                <= 2 * self.sight
            ],
            field=np.copy(self.neighborhood(*player.position, self.sight)),
            game_over=self.game_over,
            sight=self.sight,
            current_step=self.current_step,
        )

    def _make_observations(self):
        return [self._make_obs(player) for player in self.players]

    def _make_obs_array(self, observation) -> np.ndarray:
        obs = np.zeros(self._observation_space.shape, dtype=np.float32)
        seen_players = [p for p in observation.players if p.is_self] + [
            p for p in observation.players if not p.is_self
        ]

        for i in range(self.max_num_food):
            obs[3 * i] = -1
            obs[3 * i + 1] = -1
            obs[3 * i + 2] = 0

        for i, (row, col) in enumerate(zip(*np.nonzero(observation.field))):
            obs[3 * i] = row
            obs[3 * i + 1] = col
            obs[3 * i + 2] = observation.field[row, col]

        player_obs_len = 3 if self._observe_agent_levels else 2
        player_offset = self.max_num_food * 3
        for i in range(len(self.players)):
            obs[player_offset + player_obs_len * i] = -1
            obs[player_offset + player_obs_len * i + 1] = -1
            if self._observe_agent_levels:
                obs[player_offset + player_obs_len * i + 2] = 0

        for i, player_obs in enumerate(seen_players):
            obs[player_offset + player_obs_len * i] = player_obs.position[0]
            obs[player_offset + player_obs_len * i + 1] = player_obs.position[1]
            if self._observe_agent_levels:
                obs[player_offset + player_obs_len * i + 2] = player_obs.level

        return obs

    def _make_global_grid_arrays(self) -> np.ndarray:
        grid_rows = self.rows + 2 * self.sight
        grid_cols = self.cols + 2 * self.sight
        grid_shape = (grid_rows, grid_cols)

        agents_layer = np.zeros(grid_shape, dtype=np.float32)
        for player in self.players:
            assert player.position is not None
            row, col = player.position
            agents_layer[row + self.sight, col + self.sight] = (
                player.level if self._observe_agent_levels else 1
            )

        foods_layer = np.zeros(grid_shape, dtype=np.float32)
        foods_layer[
            self.sight : self.sight + self.rows,
            self.sight : self.sight + self.cols,
        ] = self.field.copy()

        access_layer = np.ones(grid_shape, dtype=np.float32)
        access_layer[: self.sight, :] = 0.0
        access_layer[self.sight + self.rows :, :] = 0.0
        access_layer[:, : self.sight] = 0.0
        access_layer[:, self.sight + self.cols :] = 0.0
        for player in self.players:
            assert player.position is not None
            row, col = player.position
            access_layer[row + self.sight, col + self.sight] = 0.0
        food_rows, food_cols = self.field.nonzero()
        for row, col in zip(food_rows, food_cols):
            access_layer[row + self.sight, col + self.sight] = 0.0

        return np.stack([agents_layer, foods_layer, access_layer])

    def _make_grid_observations(self) -> tuple[np.ndarray, ...]:
        layers = self._make_global_grid_arrays()
        observations = []
        for player in self.players:
            assert player.position is not None
            row, col = player.position
            observations.append(
                layers[
                    :,
                    row : row + 2 * self.sight + 1,
                    col : col + 2 * self.sight + 1,
                ].astype(np.float32, copy=False)
            )
        return tuple(observations)

    def _make_ordered_observations(self) -> tuple[np.ndarray, ...]:
        if self._grid_observation:
            observations = self._make_grid_observations()
        else:
            observations = tuple(
                self._make_obs_array(obs) for obs in self._make_observations()
            )

        for obs in observations:
            assert self._observation_space.contains(obs), (
                f"obs space error: obs: {obs}, obs_space: {self._observation_space}"
            )
        return observations

    def _make_parallel_observations(
        self,
        agents: list[str] | None = None,
    ) -> dict[str, np.ndarray]:
        agent_names = self.possible_agents if agents is None else agents
        ordered_observations = self._make_ordered_observations()
        return {
            agent: ordered_observations[self.agent_name_mapping[agent]]
            for agent in agent_names
        }

    def _info_for_agent(self, agent: str) -> dict[str, Any]:
        player = self.players[self.agent_name_mapping[agent]]
        return {
            "position": player.position,
            "level": int(player.level),
            "score": float(player.score),
            "valid_actions": tuple(int(action) for action in self._valid_actions[player]),
            "load_attempted": agent in self._last_load_attempted_agents,
            "load_failed": agent in self._last_failed_load_agents,
            "load_successful": agent in self._last_successful_load_agents,
            "coop_load_ok": agent not in self._last_coop_load_violation_agents,
        }

    def __str__(self) -> str:
        return (
            "level_based_foraging"
            f"<{self.rows}x{self.cols}-{self.n_agents}p-{self.max_num_food}f>"
        )
