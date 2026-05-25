from __future__ import annotations

# Adapted from Jumanji Connector, Copyright 2022 InstaDeep Ltd.,
# licensed under the Apache License, Version 2.0.

import colorsys
import functools
from enum import IntEnum
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers

EMPTY = 0
PATH = AGENT_INITIAL_VALUE = 1
POSITION = 2
TARGET = 3
RESERVATION_CONFLICT_RESERVED_CELLS = {
    0: ((2, 2),),
    1: tuple(),
}
RESERVATION_BENCHMARK_GENERATORS = {
    "reservation_conflict",
    "reservation_priority_conflict",
}


class ConnectorAction(IntEnum):
    Noop = 0
    Up = 1
    Right = 2
    Down = 3
    Left = 4


_ACTION_DELTAS = {
    ConnectorAction.Noop: (0, 0),
    ConnectorAction.Up: (-1, 0),
    ConnectorAction.Right: (0, 1),
    ConnectorAction.Down: (1, 0),
    ConnectorAction.Left: (0, -1),
}


def env(**kwargs: Any):
    """AEC-style PettingZoo environment."""
    aec_env = raw_env(**kwargs)
    aec_env = wrappers.AssertOutOfBoundsWrapper(aec_env)
    aec_env = wrappers.OrderEnforcingWrapper(aec_env)
    return aec_env


def raw_env(**kwargs: Any):
    """Convert the simultaneous-action ParallelEnv into an AEC env."""
    return parallel_to_aec(parallel_env(**kwargs))


def parallel_env(**kwargs: Any) -> "ConnectorParallelEnv":
    return ConnectorParallelEnv(**kwargs)


def get_path(agent_id: int) -> int:
    return PATH + 3 * int(agent_id)


def get_position(agent_id: int) -> int:
    return POSITION + 3 * int(agent_id)


def get_target(agent_id: int) -> int:
    return TARGET + 3 * int(agent_id)


def get_agent_id(value: int) -> int:
    return 0 if int(value) == 0 else (int(value) - 1) // 3 + 1


def is_path(value: int) -> bool:
    return int(value) > 0 and (int(value) - PATH) % 3 == 0


def is_position(value: int) -> bool:
    return int(value) > 0 and (int(value) - POSITION) % 3 == 0


def is_target(value: int) -> bool:
    return int(value) > 0 and (int(value) - TARGET) % 3 == 0


def _value_owned_by_agent(value: int, agent_id: int) -> bool:
    return int(value) in {
        get_path(agent_id),
        get_position(agent_id),
        get_target(agent_id),
    }


def move_position(position: np.ndarray, action: int) -> np.ndarray:
    row_delta, col_delta = _ACTION_DELTAS[ConnectorAction(int(action))]
    return np.asarray(
        [int(position[0]) + row_delta, int(position[1]) + col_delta],
        dtype=np.int32,
    )


def _is_repeated_later(positions: np.ndarray) -> np.ndarray:
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError(
            "positions must have shape (n_agents, 2), "
            f"but got {positions.shape}."
        )
    if len(positions) == 0:
        return np.zeros(0, dtype=bool)
    same = np.all(positions[:, None, :] == positions[None, :, :], axis=-1)
    later = np.arange(len(positions))[None, :] > np.arange(len(positions))[:, None]
    return np.any(same & later, axis=1)


class ConnectorParallelEnv(ParallelEnv):
    """PettingZoo Parallel port of Jumanji's Connector routing environment."""

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "connector_v0",
        "is_parallelizable": True,
        "render_fps": 10,
    }

    def __init__(
        self,
        *,
        grid_size: int = 10,
        n_agents: int | None = None,
        max_cycles: int | None = None,
        generator: str = "random_walk",
        reward_mode: str = "dense",
        connected_reward: float = 1.0,
        timestep_reward: float = -0.03,
        temperature: float = 1.0,
        generator_max_steps: int | None = None,
        fixed_layout_seed: int | None = None,
        render_mode: str | None = None,
        render_cell_size: int = 32,
    ) -> None:
        if n_agents is None:
            n_agents = 10
        if max_cycles is None:
            max_cycles = 50

        if grid_size < 2:
            raise ValueError("grid_size must be at least 2.")
        if n_agents < 1:
            raise ValueError("n_agents must be at least 1.")
        if grid_size * grid_size < 2 * n_agents:
            raise ValueError("grid_size must provide at least two cells per agent.")
        if max_cycles is not None and max_cycles < 1:
            raise ValueError("max_cycles must be positive when provided.")
        if generator not in {
            "random_walk",
            "uniform",
            *RESERVATION_BENCHMARK_GENERATORS,
        }:
            raise ValueError(
                "generator must be 'random_walk', 'uniform', "
                "'reservation_conflict', or 'reservation_priority_conflict'."
            )
        if generator in RESERVATION_BENCHMARK_GENERATORS and (
            int(grid_size) != 5 or int(n_agents) != 2
        ):
            raise ValueError(
                f"{generator} requires grid_size=5 and n_agents=2."
            )
        if reward_mode not in {"dense", "shared_dense", "sparse", "shared_sparse"}:
            raise ValueError(
                "reward_mode must be one of: dense, shared_dense, sparse, shared_sparse."
            )
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        if generator_max_steps is not None and generator_max_steps < 1:
            raise ValueError("generator_max_steps must be positive when provided.")
        if render_mode not in self.metadata["render_modes"] and render_mode is not None:
            raise ValueError(f"Unsupported render_mode: {render_mode!r}.")
        if render_cell_size < 4:
            raise ValueError("render_cell_size must be at least 4.")

        self.grid_size = int(grid_size)
        self.n_agents = int(n_agents)
        self.max_cycles = None if max_cycles is None else int(max_cycles)
        self.generator = generator
        self.reward_mode = reward_mode
        self.connected_reward = float(connected_reward)
        self.timestep_reward = float(timestep_reward)
        self.temperature = float(temperature)
        self.generator_max_steps = (
            None if generator_max_steps is None else int(generator_max_steps)
        )
        self.fixed_layout_seed = (
            None if fixed_layout_seed is None else int(fixed_layout_seed)
        )
        self.render_mode = render_mode
        self.render_cell_size = int(render_cell_size)

        self.possible_agents = [f"agent_{idx}" for idx in range(self.n_agents)]
        self.agent_name_mapping = {
            agent: idx for idx, agent in enumerate(self.possible_agents)
        }
        self.agents: list[str] = []

        self._action_spaces = {
            agent: spaces.Discrete(len(ConnectorAction))
            for agent in self.possible_agents
        }
        self._obs_dim = self.grid_size * self.grid_size + len(ConnectorAction) + 2
        obs_low = np.concatenate(
            (
                np.zeros(self.grid_size * self.grid_size, dtype=np.float32),
                np.zeros(len(ConnectorAction), dtype=np.float32),
                np.array([0.0, 0.0], dtype=np.float32),
            )
        )
        step_high = np.inf if self.max_cycles is None else float(self.max_cycles)
        obs_high = np.concatenate(
            (
                np.full(
                    self.grid_size * self.grid_size,
                    3 * self.n_agents,
                    dtype=np.float32,
                ),
                np.ones(len(ConnectorAction), dtype=np.float32),
                np.array([step_high, max(self.n_agents - 1, 0)], dtype=np.float32),
            )
        )
        self._observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32,
        )
        self.state_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_agents * self._obs_dim,),
            dtype=np.float32,
        )

        self.np_random, self.np_random_seed = seeding.np_random(None)
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.starts = np.zeros((self.n_agents, 2), dtype=np.int32)
        self.targets = np.zeros((self.n_agents, 2), dtype=np.int32)
        self.positions = np.zeros((self.n_agents, 2), dtype=np.int32)
        self.action_mask = np.ones((self.n_agents, len(ConnectorAction)), dtype=bool)
        self.num_moves = 0
        self._last_observations: dict[str, np.ndarray] = {}
        self._last_infos: dict[str, dict[str, Any]] = {}
        self._last_reservation_violations: frozenset[tuple[int, int]] = frozenset()
        self._human_screen = None

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str):
        _ = agent
        return self._observation_space

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str):
        return self._action_spaces[agent]

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        _ = options
        if seed is not None:
            self.np_random, self.np_random_seed = seeding.np_random(seed)

        if self.fixed_layout_seed is not None:
            runtime_random = self.np_random
            runtime_random_seed = self.np_random_seed
            self.np_random, self.np_random_seed = seeding.np_random(
                self.fixed_layout_seed
            )
            try:
                self._generate_board()
            finally:
                self.np_random = runtime_random
                self.np_random_seed = runtime_random_seed
        else:
            self._generate_board()

        self.num_moves = 0
        self.agents = self.possible_agents[:]
        self._last_reservation_violations = frozenset()
        self._recompute_action_mask()
        self._last_observations = self._observations_for(self.possible_agents)
        self._last_infos = self._infos_for(self.possible_agents)

        if self.render_mode == "human":
            self.render()

        return dict(self._last_observations), dict(self._last_infos)

    def _generate_board(self) -> None:
        if self.generator == "reservation_conflict":
            self._generate_reservation_conflict_board()
        elif self.generator == "reservation_priority_conflict":
            self._generate_reservation_priority_conflict_board()
        elif self.generator == "uniform":
            self._generate_uniform_board()
        else:
            self._generate_random_walk_board()

    def step(self, actions: dict[str, Any]):
        if not self.agents:
            return {}, {}, {}, {}, {}

        current_agents = self.agents[:]
        missing = [agent for agent in current_agents if agent not in actions]
        if missing:
            raise KeyError(f"Missing actions for agents: {missing}")

        ordered_actions = np.zeros(self.n_agents, dtype=np.int64)
        for agent in current_agents:
            action = int(actions[agent])
            if not self.action_space(agent).contains(action):
                raise ValueError(f"Invalid action {action} for {agent!r}.")
            ordered_actions[self.agent_name_mapping[agent]] = action

        old_connected = self._connected_mask()
        self._step_agents(ordered_actions, old_connected=old_connected)
        self.num_moves += 1
        self._recompute_action_mask()

        rewards_array = self._rewards(old_connected)
        agent_done = self._connected_or_blocked_mask()
        all_done = bool(agent_done.all())
        timeout = self.max_cycles is not None and self.num_moves >= self.max_cycles
        terminated = all_done
        truncated = bool(timeout and not all_done)

        discounts = np.where(agent_done, 0.0, 1.0).astype(np.float32)
        if terminated:
            discounts[:] = 0.0

        rewards = {
            agent: float(rewards_array[self.agent_name_mapping[agent]])
            for agent in current_agents
        }
        terminations = {agent: bool(terminated) for agent in current_agents}
        truncations = {agent: bool(truncated) for agent in current_agents}

        end_reason = None
        if terminated:
            end_reason = "all_connected_or_blocked"
        elif truncated:
            end_reason = "max_cycles"

        observations = self._observations_for(current_agents)
        infos = self._infos_for(
            current_agents,
            discounts=discounts,
            episode_end_reason=end_reason,
        )

        self._last_observations.update(observations)
        self._last_infos = infos

        if terminated or truncated:
            self.agents = []

        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminations, truncations, infos

    def observe(self, agent: str) -> np.ndarray:
        return self._observation_for_index(self.agent_name_mapping[agent])

    def state(self) -> np.ndarray:
        if not self._last_observations:
            raise RuntimeError("Call reset() before state().")
        return np.concatenate(
            [self._observation_for_index(idx) for idx in range(self.n_agents)]
        ).astype(np.float32, copy=False)

    def render(self):
        if self.render_mode is None:
            return None
        frame = self._render_rgb_array(self.render_cell_size)
        if self.render_mode == "rgb_array":
            return frame

        import pygame

        pygame.init()
        height, width = frame.shape[:2]
        if self._human_screen is None:
            self._human_screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("Connector")
        surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        self._human_screen.blit(surface, (0, 0))
        pygame.display.flip()
        return None

    def close(self) -> None:
        if self._human_screen is not None:
            import pygame

            pygame.display.quit()
            self._human_screen = None

    @property
    def unwrapped(self):
        return self

    def _generate_uniform_board(self) -> None:
        cells = self.grid_size * self.grid_size
        flat = self.np_random.choice(cells, size=2 * self.n_agents, replace=False)
        self.starts = np.column_stack(
            np.divmod(flat[: self.n_agents], self.grid_size)
        ).astype(np.int32)
        self.targets = np.column_stack(
            np.divmod(flat[self.n_agents :], self.grid_size)
        ).astype(np.int32)
        self.positions = self.starts.copy()
        self._populate_training_grid()

    def _generate_reservation_conflict_board(self) -> None:
        self.starts = np.asarray([(2, 1), (4, 2)], dtype=np.int32)
        self.targets = np.asarray([(2, 3), (0, 2)], dtype=np.int32)
        self.positions = self.starts.copy()
        self._populate_training_grid()

    def _generate_reservation_priority_conflict_board(self) -> None:
        self.starts = np.asarray([(2, 1), (3, 2)], dtype=np.int32)
        self.targets = np.asarray([(2, 3), (0, 2)], dtype=np.int32)
        self.positions = self.starts.copy()
        self._populate_training_grid()

    def _generate_random_walk_board(self) -> None:
        max_steps = (
            self.generator_max_steps
            if self.generator_max_steps is not None
            else int(self.grid_size * 1.5)
        )
        last_error: Exception | None = None
        for _ in range(200):
            try:
                self._try_generate_random_walk_board(max_steps)
                return
            except ValueError as exc:
                last_error = exc
        raise RuntimeError("Failed to generate a Connector random-walk board.") from last_error

    def _try_generate_random_walk_board(self, max_steps: int) -> None:
        grid, starts, positions = self._initialize_random_walk_agents()
        action_mask = self._action_masks_for(positions, np.full_like(positions, -1), grid)
        step_count = 1

        while bool(action_mask[:, 1:].any()) and step_count < max_steps:
            actions = np.zeros(self.n_agents, dtype=np.int64)
            for idx in range(self.n_agents):
                actions[idx] = self._select_random_walk_action(
                    action_mask[idx],
                    starts[idx],
                    positions[idx],
                )
            positions, grid = self._step_positions_for_generator(
                positions,
                grid,
                actions,
            )
            action_mask = self._action_masks_for(positions, np.full_like(positions, -1), grid)
            step_count += 1

        self.starts = starts.astype(np.int32, copy=False)
        self.targets = positions.astype(np.int32, copy=True)
        self.positions = self.starts.copy()
        self._populate_training_grid()

    def _initialize_random_walk_agents(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        starts = np.zeros((self.n_agents, 2), dtype=np.int32)
        positions = np.zeros((self.n_agents, 2), dtype=np.int32)

        for agent_id in range(self.n_agents):
            valid_start_mask = (grid == EMPTY) & ~self._surrounded_mask(grid)
            start = self._sample_coordinate(valid_start_mask)
            grid[tuple(start)] = get_path(agent_id)

            adjacency = self._adjacency_mask(tuple(start))
            first_move = self._sample_coordinate(adjacency & (grid == EMPTY))
            grid[tuple(first_move)] = get_position(agent_id)

            starts[agent_id] = start
            positions[agent_id] = first_move

        return grid, starts, positions

    def _sample_coordinate(self, mask: np.ndarray) -> np.ndarray:
        choices = np.flatnonzero(mask.ravel())
        if len(choices) == 0:
            raise ValueError("No valid coordinates available.")
        flat = int(self.np_random.choice(choices))
        return np.asarray(np.unravel_index(flat, mask.shape), dtype=np.int32)

    def _select_random_walk_action(
        self,
        action_mask: np.ndarray,
        start_position: np.ndarray,
        current_position: np.ndarray,
    ) -> int:
        if not action_mask[1:].any():
            return int(ConnectorAction.Noop)

        displacement = current_position - start_position
        dot_products = np.asarray(
            [-displacement[0], displacement[1], displacement[0], -displacement[1]],
            dtype=np.float64,
        )
        logits = dot_products / self.temperature
        logits = logits - float(np.max(logits))
        probs = np.exp(logits) * action_mask[1:].astype(np.float64)
        total = float(probs.sum())
        if total <= 0.0:
            return int(ConnectorAction.Noop)
        probs = probs / total
        return int(self.np_random.choice(np.arange(1, 5), p=probs))

    def _step_positions_for_generator(
        self,
        positions: np.ndarray,
        grid: np.ndarray,
        actions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        new_positions = np.asarray(
            [move_position(position, action) for position, action in zip(positions, actions, strict=True)],
            dtype=np.int32,
        )
        collided = _is_repeated_later(new_positions)
        new_positions = np.where(collided[:, None], positions, new_positions)
        noop = np.all(new_positions == positions, axis=-1)

        next_grid = grid.copy()
        for idx in range(self.n_agents):
            if noop[idx]:
                continue
            next_grid[tuple(positions[idx])] += PATH - POSITION
            next_grid[tuple(new_positions[idx])] += get_position(idx)

        return new_positions.astype(np.int32, copy=False), next_grid

    def _populate_training_grid(self) -> None:
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.grid[tuple(self.starts.T)] = np.asarray(
            [get_position(idx) for idx in range(self.n_agents)],
            dtype=np.int32,
        )
        self.grid[tuple(self.targets.T)] = np.asarray(
            [get_target(idx) for idx in range(self.n_agents)],
            dtype=np.int32,
        )

    def _step_agents(self, actions: np.ndarray, *, old_connected: np.ndarray) -> None:
        legal_action_taken = self.action_mask[np.arange(self.n_agents), actions]
        proposed_positions = np.asarray(
            [move_position(position, action) for position, action in zip(self.positions, actions, strict=True)],
            dtype=np.int32,
        )
        new_positions = np.where(
            legal_action_taken[:, None],
            proposed_positions,
            self.positions,
        )
        collided = _is_repeated_later(new_positions)
        connecting = np.asarray(
            [is_target(self.grid[tuple(position)]) for position in new_positions],
            dtype=bool,
        )
        noop = np.all(new_positions == self.positions, axis=-1)

        next_grid = self.grid.copy()
        for idx in range(self.n_agents):
            if collided[idx] or noop[idx]:
                continue
            next_grid[tuple(self.positions[idx])] += PATH - POSITION
            if connecting[idx]:
                next_grid[tuple(new_positions[idx])] += POSITION - TARGET
            else:
                next_grid[tuple(new_positions[idx])] += get_position(idx)

        final_positions = np.where(collided[:, None], self.positions, new_positions).astype(
            np.int32,
            copy=False,
        )
        self._last_reservation_violations = self._reservation_violations_for(
            self.positions,
            final_positions,
            old_connected,
        )
        self.positions = final_positions
        self.grid = next_grid

    def _recompute_action_mask(self) -> None:
        self.action_mask = self._action_masks_for(self.positions, self.targets, self.grid)

    def _action_masks_for(
        self,
        positions: np.ndarray,
        targets: np.ndarray,
        grid: np.ndarray,
    ) -> np.ndarray:
        masks = np.ones((self.n_agents, len(ConnectorAction)), dtype=bool)
        for idx in range(self.n_agents):
            connected = bool(np.array_equal(positions[idx], targets[idx]))
            for action in (
                ConnectorAction.Up,
                ConnectorAction.Right,
                ConnectorAction.Down,
                ConnectorAction.Left,
            ):
                next_position = move_position(positions[idx], int(action))
                row, col = int(next_position[0]), int(next_position[1])
                in_bounds = 0 <= row < self.grid_size and 0 <= col < self.grid_size
                if not in_bounds or connected:
                    masks[idx, int(action)] = False
                    continue
                value = int(grid[row, col])
                masks[idx, int(action)] = value == EMPTY or value == get_target(idx)
        return masks

    def _surrounded_mask(self, grid: np.ndarray) -> np.ndarray:
        occupied = grid > EMPTY
        surrounded = np.zeros_like(occupied, dtype=bool)
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                neighbors = self._valid_neighbors(row, col)
                surrounded[row, col] = all(occupied[n_row, n_col] for n_row, n_col in neighbors)
        return surrounded

    def _adjacency_mask(self, coordinate: tuple[int, int]) -> np.ndarray:
        row, col = coordinate
        mask = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        for n_row, n_col in self._valid_neighbors(int(row), int(col)):
            mask[n_row, n_col] = True
        return mask

    def _valid_neighbors(self, row: int, col: int) -> tuple[tuple[int, int], ...]:
        neighbors: list[tuple[int, int]] = []
        for row_delta, col_delta in ((-1, 0), (1, 0), (0, 1), (0, -1)):
            n_row = row + row_delta
            n_col = col + col_delta
            if 0 <= n_row < self.grid_size and 0 <= n_col < self.grid_size:
                neighbors.append((n_row, n_col))
        return tuple(neighbors)

    def _connected_mask(self) -> np.ndarray:
        return np.all(self.positions == self.targets, axis=-1)

    def _connected_or_blocked_mask(self) -> np.ndarray:
        return self._connected_mask() | ~self.action_mask[:, 1:].any(axis=1)

    def _reservation_cells_for_agent(self, agent_idx: int) -> tuple[tuple[int, int], ...]:
        if self.generator not in RESERVATION_BENCHMARK_GENERATORS:
            return tuple()
        return tuple(
            (int(row), int(col))
            for row, col in RESERVATION_CONFLICT_RESERVED_CELLS.get(
                int(agent_idx),
                tuple(),
            )
        )

    def _reservation_violations_for(
        self,
        previous_positions: np.ndarray,
        next_positions: np.ndarray,
        old_connected: np.ndarray,
    ) -> frozenset[tuple[int, int]]:
        violations: set[tuple[int, int]] = set()
        for violator_idx in range(self.n_agents):
            previous = tuple(int(value) for value in previous_positions[violator_idx])
            current = tuple(int(value) for value in next_positions[violator_idx])
            if previous == current:
                continue
            for owner_idx in range(self.n_agents):
                if owner_idx == violator_idx or bool(old_connected[owner_idx]):
                    continue
                reserved_cells = set(self._reservation_cells_for_agent(owner_idx))
                if previous in reserved_cells or current in reserved_cells:
                    violations.add((violator_idx, owner_idx))
        return frozenset(violations)

    def _reserved_route_blocked_mask(self) -> np.ndarray:
        connected = self._connected_mask()
        blocked = np.zeros(self.n_agents, dtype=bool)
        for owner_idx in range(self.n_agents):
            if bool(connected[owner_idx]):
                continue
            for row, col in self._reservation_cells_for_agent(owner_idx):
                value = int(self.grid[row, col])
                if value != EMPTY and not _value_owned_by_agent(value, owner_idx):
                    blocked[owner_idx] = True
                    break
        return blocked

    def _rewards(self, old_connected: np.ndarray) -> np.ndarray:
        new_connected = self._connected_mask()
        newly_connected = (~old_connected) & new_connected
        if self.reward_mode in {"dense", "shared_dense"}:
            rewards = (
                self.connected_reward * newly_connected.astype(np.float32)
                + self.timestep_reward * (~old_connected).astype(np.float32)
            )
        else:
            rewards = newly_connected.astype(np.float32)

        if self.reward_mode in {"shared_dense", "shared_sparse"}:
            rewards = np.full(self.n_agents, float(rewards.sum()), dtype=np.float32)
        return rewards.astype(np.float32, copy=False)

    def _observation_for_index(self, idx: int) -> np.ndarray:
        return np.concatenate(
            (
                self.grid.astype(np.float32, copy=False).ravel(),
                self.action_mask[idx].astype(np.float32, copy=False),
                np.asarray([self.num_moves, idx], dtype=np.float32),
            )
        ).astype(np.float32, copy=False)

    def _observations_for(self, agents: list[str]) -> dict[str, np.ndarray]:
        return {
            agent: self._observation_for_index(self.agent_name_mapping[agent])
            for agent in agents
        }

    def _infos_for(
        self,
        agents: list[str],
        *,
        discounts: np.ndarray | None = None,
        episode_end_reason: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        if discounts is None:
            discounts = np.where(self._connected_or_blocked_mask(), 0.0, 1.0).astype(np.float32)
        extras = self._extras()
        connected = self._connected_mask()
        blocked = ~self.action_mask[:, 1:].any(axis=1)
        reserved_route_blocked = self._reserved_route_blocked_mask()
        infos: dict[str, dict[str, Any]] = {}
        for agent in agents:
            idx = self.agent_name_mapping[agent]
            legal_actions = tuple(
                int(action)
                for action, allowed in enumerate(self.action_mask[idx])
                if bool(allowed)
            )
            info: dict[str, Any] = {
                "grid": self.grid.copy(),
                "action_mask": self.action_mask[idx].copy(),
                "global_action_mask": self.action_mask.copy(),
                "position": tuple(int(value) for value in self.positions[idx]),
                "start": tuple(int(value) for value in self.starts[idx]),
                "target": tuple(int(value) for value in self.targets[idx]),
                "connected": bool(connected[idx]),
                "blocked": bool(blocked[idx] and not connected[idx]),
                "reserved_route_blocked": bool(reserved_route_blocked[idx]),
                "reservation_violations": tuple(
                    violation
                    for violation in self._last_reservation_violations
                    if violation[0] == idx
                ),
                "reserved_cells": self._reservation_cells_for_agent(idx),
                "connected_or_blocked": bool(connected[idx] or blocked[idx]),
                "legal_actions": legal_actions,
                "discount": float(discounts[idx]),
                "step_count": int(self.num_moves),
                "fixed_layout_seed": self.fixed_layout_seed,
                **extras,
            }
            if episode_end_reason is not None:
                info["episode_done"] = True
                info["episode_end_reason"] = episode_end_reason
            infos[agent] = info
        return infos

    def _extras(self) -> dict[str, Any]:
        encoded_as_path = (
            AGENT_INITIAL_VALUE + (self.grid - AGENT_INITIAL_VALUE) % 3
        ) == PATH
        total_path_length = int(np.count_nonzero(encoded_as_path) + self.n_agents)
        connected = self._connected_mask()
        return {
            "num_connections": int(np.count_nonzero(connected)),
            "ratio_connections": float(np.mean(connected)),
            "total_path_length": total_path_length,
        }

    def _render_rgb_array(self, cell_size: int) -> np.ndarray:
        size = self.grid_size * cell_size
        frame = np.full((size, size, 3), 244, dtype=np.uint8)
        colors = self._agent_colors()

        for row in range(self.grid_size):
            for col in range(self.grid_size):
                value = int(self.grid[row, col])
                y0 = row * cell_size
                x0 = col * cell_size
                y1 = y0 + cell_size
                x1 = x0 + cell_size
                if value == EMPTY:
                    color = np.asarray([248, 248, 244], dtype=np.uint8)
                    frame[y0:y1, x0:x1] = color
                else:
                    agent_idx = get_agent_id(value) - 1
                    base = colors[agent_idx % len(colors)]
                    if is_path(value):
                        color = (0.72 * np.asarray([248, 248, 244]) + 0.28 * base).astype(np.uint8)
                        frame[y0:y1, x0:x1] = color
                    elif is_position(value):
                        pad = max(cell_size // 6, 2)
                        frame[y0:y1, x0:x1] = np.asarray([248, 248, 244], dtype=np.uint8)
                        frame[y0 + pad : y1 - pad, x0 + pad : x1 - pad] = base
                    elif is_target(value):
                        pad = max(cell_size // 4, 2)
                        frame[y0:y1, x0:x1] = np.asarray([248, 248, 244], dtype=np.uint8)
                        frame[y0 + pad : y1 - pad, x0 + pad : x1 - pad] = base

        line_color = np.asarray([52, 57, 63], dtype=np.uint8)
        frame[::cell_size, :, :] = line_color
        frame[:, ::cell_size, :] = line_color
        frame[-1, :, :] = line_color
        frame[:, -1, :] = line_color
        return frame

    def _agent_colors(self) -> list[np.ndarray]:
        colors: list[np.ndarray] = []
        for idx in range(self.n_agents):
            hue = idx / max(self.n_agents, 1)
            red, green, blue = colorsys.hsv_to_rgb(hue, 0.68, 0.88)
            colors.append(
                np.asarray([red * 255, green * 255, blue * 255], dtype=np.uint8)
            )
        return colors

    def __str__(self) -> str:
        return f"connector<{self.n_agents}ag,{self.grid_size}x{self.grid_size}>"
