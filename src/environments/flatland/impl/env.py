from __future__ import annotations

import functools
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers

_VENDOR_ROOT = Path(__file__).resolve().parents[1] / "_vendor"
if str(_VENDOR_ROOT) in sys.path:
    sys.path.remove(str(_VENDOR_ROOT))
sys.path.insert(0, str(_VENDOR_ROOT))

from flatland.env_generation.env_generator import env_generator
from flatland.envs.grid.rail_env_grid import RailEnvTransitions
from flatland.envs.rail_env import RailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.rail_generators import rail_from_grid_transition_map
from flatland.envs.rail_grid_transition_map import RailGridTransitionMap
from flatland.envs.rail_trainrun_data_structures import Waypoint
from flatland.envs.step_utils.states import TrainState
from flatland.envs.timetable_utils import Line
from flatland.envs.grid4_generators_utils import connect_straight_line_in_grid_map
from flatland.ml.observations.flatten_tree_observation_for_rail_env import (
    FlattenedNormalizedTreeObsForRailEnv,
)


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


def parallel_env(**kwargs: Any) -> "FlatlandParallelEnv":
    return FlatlandParallelEnv(**kwargs)


def _clear_cached_bound_methods(obj: object) -> None:
    seen: set[int] = set()
    for cls in type(obj).__mro__:
        for name in cls.__dict__:
            try:
                member = getattr(obj, name)
            except Exception:
                continue
            cache_clear = getattr(member, "cache_clear", None)
            if not callable(cache_clear):
                continue
            cache_key = getattr(member, "__func__", member)
            cache_id = id(cache_key)
            if cache_id in seen:
                continue
            seen.add(cache_id)
            cache_clear()


class FlatlandParallelEnv(ParallelEnv):
    """PettingZoo Parallel wrapper around the vendored Flatland RailEnv."""

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "flatland_railway_v0",
        "is_parallelizable": True,
        "render_fps": 10,
    }

    def __init__(
        self,
        *,
        n_agents: int = 2,
        x_dim: int = 30,
        y_dim: int = 30,
        n_cities: int = 2,
        max_rail_pairs_in_city: int = 2,
        grid_mode: bool = False,
        max_rails_between_cities: int = 2,
        p_level_free: float = 0.0,
        malfunction_duration_min: int = 20,
        malfunction_duration_max: int = 50,
        malfunction_interval: int | None = 0,
        speed_ratios: dict[float, float] | None = None,
        line_length: int = 2,
        observation_depth: int = 2,
        prediction_depth: int = 20,
        tree_depth: int | None = None,
        predictor_depth: int | None = None,
        observation_radius: int = 2,
        acceleration_delta: float = 1.0,
        braking_delta: float = -1.0,
        max_cycles: int | None = None,
        scenario: str | None = None,
        team_done: bool = False,
        reward_mode: str = "native",
        progress_reward_scale: float = 1.0,
        arrival_bonus: float = 5.0,
        deadlock_penalty: float = 5.0,
        step_penalty: float = 0.01,
        render_mode: str | None = None,
        render_cell_size: int = 24,
        seed: int | None = None,
    ) -> None:
        scenario = "sparse" if scenario is None else str(scenario)
        if tree_depth is not None:
            observation_depth = tree_depth
        if predictor_depth is not None:
            prediction_depth = predictor_depth

        if scenario not in {"sparse", "single_track_meet"}:
            raise ValueError(f"Unsupported Flatland scenario: {scenario!r}.")
        if reward_mode not in {"native", "progress"}:
            raise ValueError(f"Unsupported Flatland reward_mode: {reward_mode!r}.")
        if scenario == "single_track_meet":
            n_agents = 2
            x_dim = 7
            y_dim = 3
            speed_ratios = {1.0: 1.0}
            malfunction_interval = 0

        if n_agents < 1:
            raise ValueError("n_agents must be at least 1.")
        if x_dim < 2 or y_dim < 2:
            raise ValueError("x_dim and y_dim must be at least 2.")
        if observation_depth < 0:
            raise ValueError("observation_depth must be non-negative.")
        if prediction_depth < 1:
            raise ValueError("prediction_depth must be positive.")
        if max_cycles is not None and max_cycles < 1:
            raise ValueError("max_cycles must be positive when provided.")
        if render_cell_size < 4:
            raise ValueError("render_cell_size must be at least 4.")
        if render_mode not in self.metadata["render_modes"] and render_mode is not None:
            raise ValueError(f"Unsupported render_mode: {render_mode!r}.")

        self.n_agents = int(n_agents)
        self.scenario = scenario
        self.team_done = bool(team_done)
        self.reward_mode = str(reward_mode)
        self.progress_reward_scale = float(progress_reward_scale)
        self.arrival_bonus = float(arrival_bonus)
        self.deadlock_penalty = float(deadlock_penalty)
        self.step_penalty = float(step_penalty)
        self.render_mode = render_mode
        self.max_cycles = None if max_cycles is None else int(max_cycles)
        self.render_cell_size = int(render_cell_size)
        self._seed = seed
        self._env_kwargs: dict[str, Any] = {
            "n_agents": int(n_agents),
            "x_dim": int(x_dim),
            "y_dim": int(y_dim),
            "n_cities": int(n_cities),
            "max_rail_pairs_in_city": int(max_rail_pairs_in_city),
            "grid_mode": bool(grid_mode),
            "max_rails_between_cities": int(max_rails_between_cities),
            "p_level_free": float(p_level_free),
            "malfunction_duration_min": int(malfunction_duration_min),
            "malfunction_duration_max": int(malfunction_duration_max),
            "malfunction_interval": malfunction_interval,
            "speed_ratios": speed_ratios,
            "line_length": int(line_length),
            "acceleration_delta": float(acceleration_delta),
            "braking_delta": float(braking_delta),
        }
        self._observation_config = {
            "max_depth": int(observation_depth),
            "predictor": ShortestPathPredictorForRailEnv(max_depth=int(prediction_depth)),
            "observation_radius": int(observation_radius),
        }

        self.possible_agents = [f"agent_{idx}" for idx in range(self.n_agents)]
        self.agent_name_mapping = {
            agent: idx for idx, agent in enumerate(self.possible_agents)
        }
        self.agents: list[str] = []

        raw_observation_space = self._make_observation_builder().get_observation_space(0)
        self._observation_space = spaces.Box(
            low=np.full(raw_observation_space.shape, -1.0, dtype=raw_observation_space.dtype),
            high=np.full(
                raw_observation_space.shape,
                max(float(observation_radius), 1.0),
                dtype=raw_observation_space.dtype,
            ),
            dtype=raw_observation_space.dtype,
        )
        self._action_spaces = {
            agent: spaces.Discrete(len(RailEnvActions)) for agent in self.possible_agents
        }
        self.state_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_agents * gym.spaces.flatdim(self._observation_space),),
            dtype=np.float32,
        )

        self.rail_env = None
        self._last_observations: dict[str, np.ndarray] = {}
        self._last_infos: dict[str, dict[str, Any]] = {}
        self._last_distances: dict[str, float] = {}
        self._last_yield_violations: frozenset[tuple[int, int]] = frozenset()
        self._human_screen = None

    def _make_observation_builder(self) -> FlattenedNormalizedTreeObsForRailEnv:
        return FlattenedNormalizedTreeObsForRailEnv(**self._observation_config)

    def _make_base_env(self, seed: int | None):
        if self.scenario == "single_track_meet":
            return self._make_single_track_meet_env(seed)

        raw_env, observations, infos = env_generator(
            seed=seed,
            obs_builder_object=self._make_observation_builder(),
            **self._env_kwargs,
        )
        return raw_env, observations, infos

    def _make_single_track_meet_env(self, seed: int | None):
        rail_transitions = RailEnvTransitions()
        rail_map = RailGridTransitionMap(
            width=7,
            height=3,
            transitions=rail_transitions,
        )
        connect_straight_line_in_grid_map(
            rail_map,
            (1, 1),
            (1, 5),
            rail_transitions,
        )

        def line_generator(
            rail,
            num_agents: int,
            hints: dict[str, Any] | None = None,
            num_resets: int = 0,
            np_random=None,
        ) -> Line:
            _ = (rail, hints, num_resets, np_random)
            if int(num_agents) != 2:
                raise ValueError("single_track_meet requires exactly two agents.")
            return Line(
                agent_waypoints={
                    0: [
                        [Waypoint((1, 1), 1)],
                        [Waypoint((1, 5), None)],
                    ],
                    1: [
                        [Waypoint((1, 5), 3)],
                        [Waypoint((1, 1), None)],
                    ],
                },
                agent_speeds=[1.0, 1.0],
            )

        raw_env = RailEnv(
            width=7,
            height=3,
            rail_generator=rail_from_grid_transition_map(rail_map),
            line_generator=line_generator,
            number_of_agents=2,
            obs_builder_object=self._make_observation_builder(),
            record_steps=True,
            random_seed=seed,
            acceleration_delta=self._env_kwargs["acceleration_delta"],
            braking_delta=self._env_kwargs["braking_delta"],
        )
        observations, infos = raw_env.reset(random_seed=seed)
        return raw_env, observations, infos

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str):
        _ = agent
        return self._observation_space

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str):
        return self._action_spaces[agent]

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        _ = options
        reset_seed = self._seed if seed is None else seed
        self._cleanup_current_rail_env()
        self.rail_env, observations, flatland_infos = self._make_base_env(reset_seed)
        self.agents = self.possible_agents[:]
        self._last_yield_violations = frozenset()

        self._last_observations = self._agent_dict_from_handles(observations)
        self._last_infos = self._infos_from_flatland(flatland_infos)
        self._last_distances = {
            agent: self._distance_to_target(self.agent_name_mapping[agent])
            for agent in self.possible_agents
        }

        if self.render_mode == "human":
            self.render()

        return dict(self._last_observations), dict(self._last_infos)

    def step(self, actions: dict[str, Any]):
        if not self.agents:
            return {}, {}, {}, {}, {}

        missing = [agent for agent in self.agents if agent not in actions]
        if missing:
            raise KeyError(f"Missing actions for agents: {missing}")

        current_agents = self.agents[:]
        action_dict: dict[int, RailEnvActions] = {}
        for agent in current_agents:
            action = int(actions[agent])
            if not self.action_space(agent).contains(action):
                raise ValueError(f"Invalid action {action} for {agent!r}.")
            action_dict[self.agent_name_mapping[agent]] = RailEnvActions.from_value(action)

        previous_distances = {
            agent: self._distance_to_target(self.agent_name_mapping[agent])
            for agent in current_agents
        }
        previous_done = {
            agent: self.rail_env.agents[self.agent_name_mapping[agent]].state == TrainState.DONE
            for agent in current_agents
        }
        previous_trains = tuple(
            self._train_snapshot(train)
            for train in self.rail_env.agents
        )
        observations, rewards, dones, flatland_infos = self.rail_env.step(action_dict)
        self._last_yield_violations = self._yield_violations_for(
            previous_trains,
            tuple(self._train_snapshot(train) for train in self.rail_env.agents),
        )

        obs_dict = self._agent_dict_from_handles(observations, agents=current_agents)
        infos = self._infos_from_flatland(flatland_infos, agents=current_agents)
        if self.reward_mode == "progress":
            reward_dict = self._progress_rewards(
                rewards,
                previous_distances=previous_distances,
                previous_done=previous_done,
                agents=current_agents,
            )
        else:
            reward_dict = {
                agent: float(rewards[self.agent_name_mapping[agent]])
                for agent in current_agents
            }
        self._last_distances = {
            agent: self._distance_to_target(self.agent_name_mapping[agent])
            for agent in self.possible_agents
        }

        max_cycles_hit = (
            self.max_cycles is not None
            and int(self.rail_env._elapsed_steps) >= self.max_cycles
        )
        internal_time_limit_hit = (
            self.rail_env._max_episode_steps is not None
            and int(self.rail_env._elapsed_steps) >= int(self.rail_env._max_episode_steps)
        )

        terminations: dict[str, bool] = {}
        truncations: dict[str, bool] = {}
        team_terminated = bool(dones.get("__all__", False) and self._all_agents_reached_target())
        team_truncated = bool(
            (max_cycles_hit or internal_time_limit_hit)
            and not team_terminated
        )
        for agent in current_agents:
            handle = self.agent_name_mapping[agent]
            target_reached = self.rail_env.agents[handle].state == TrainState.DONE
            time_limit_hit = max_cycles_hit or (
                internal_time_limit_hit and not self._all_agents_reached_target()
            )
            if self.team_done:
                terminations[agent] = team_terminated
                truncations[agent] = team_truncated
            else:
                terminations[agent] = bool(dones.get(handle, False) and target_reached)
                truncations[agent] = bool(time_limit_hit and not target_reached)
            if max_cycles_hit:
                infos[agent]["episode_end_reason"] = "max_cycles"
            elif internal_time_limit_hit and not target_reached:
                infos[agent]["episode_end_reason"] = "flatland_time_limit"
            if terminations[agent] or truncations[agent]:
                infos[agent]["episode_done"] = True

        self._last_observations.update(obs_dict)
        self._last_infos = infos

        episode_done = bool(dones.get("__all__", False) or max_cycles_hit or team_truncated)
        if self.team_done and not episode_done:
            self.agents = self.possible_agents[:]
        else:
            self.agents = [
                agent
                for agent in current_agents
                if not (terminations[agent] or truncations[agent] or episode_done)
            ]
        if episode_done:
            self.agents = []

        if self.render_mode == "human":
            self.render()

        return obs_dict, reward_dict, terminations, truncations, infos

    def _progress_rewards(
        self,
        native_rewards: dict[int, float],
        *,
        previous_distances: dict[str, float],
        previous_done: dict[str, bool],
        agents: list[str],
    ) -> dict[str, float]:
        deadlocked_handles = set(getattr(self.rail_env.motion_check, "deadlocked", set()))
        reward_dict: dict[str, float] = {}
        for agent in agents:
            handle = self.agent_name_mapping[agent]
            train = self.rail_env.agents[handle]
            previous_distance = previous_distances.get(agent, self._distance_to_target(handle))
            next_distance = self._distance_to_target(handle)
            progress = 0.0
            if np.isfinite(previous_distance) and np.isfinite(next_distance):
                progress = max(previous_distance - next_distance, 0.0)
            reward = float(progress * self.progress_reward_scale)
            reward -= self.step_penalty
            if train.state == TrainState.DONE and not previous_done.get(agent, False):
                reward += self.arrival_bonus
            if handle in deadlocked_handles:
                reward -= self.deadlock_penalty
            if not np.isfinite(reward):
                reward = float(native_rewards.get(handle, 0.0))
            reward_dict[agent] = float(reward)
        return reward_dict

    def render(self):
        if self.render_mode is None or self.rail_env is None:
            return None

        frame = self._render_rgb_array(cell_size=self.render_cell_size)
        if self.render_mode == "rgb_array":
            return frame

        import pygame

        pygame.init()
        height, width = frame.shape[:2]
        if self._human_screen is None:
            self._human_screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("Flatland RailwayEnv")
        surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        self._human_screen.blit(surface, (0, 0))
        pygame.display.flip()
        return None

    def close(self):
        self._cleanup_current_rail_env()
        self.rail_env = None
        self.agents = []
        self._last_observations = {}
        self._last_infos = {}
        self._last_distances = {}
        self._last_yield_violations = frozenset()
        if self._human_screen is not None:
            import pygame

            pygame.display.quit()
            self._human_screen = None

    def _cleanup_current_rail_env(self) -> None:
        rail_env = self.rail_env
        if rail_env is None:
            return

        rail = getattr(rail_env, "rail", None)
        if rail is not None:
            _clear_cached_bound_methods(rail)
        _clear_cached_bound_methods(rail_env)
        self._detach_observation_builder(rail_env)

    @staticmethod
    def _detach_observation_builder(rail_env) -> None:
        obs_builder = getattr(rail_env, "obs_builder", None)
        if obs_builder is None:
            return

        predictor = getattr(obs_builder, "predictor", None)
        if predictor is not None and hasattr(predictor, "env"):
            predictor.env = None
        if hasattr(obs_builder, "env"):
            obs_builder.env = None

    def state(self) -> np.ndarray:
        if not self._last_observations:
            raise RuntimeError("Call reset() before state().")

        zero_obs = np.zeros(self._observation_space.shape, dtype=self._observation_space.dtype)
        flat_obs = []
        for agent in self.possible_agents:
            obs = self._last_observations.get(agent, zero_obs)
            flat_obs.append(gym.spaces.flatten(self._observation_space, obs))
        return np.concatenate(flat_obs).astype(np.float32, copy=False)

    @property
    def unwrapped(self):
        return self.rail_env

    def _agent_dict_from_handles(
        self,
        values: dict[int, Any],
        *,
        agents: list[str] | None = None,
    ) -> dict[str, np.ndarray]:
        selected_agents = self.possible_agents if agents is None else agents
        return {
            agent: np.asarray(values[self.agent_name_mapping[agent]])
            for agent in selected_agents
        }

    def _infos_from_flatland(
        self,
        flatland_infos: dict[str, dict[int, Any]],
        *,
        agents: list[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        selected_agents = self.possible_agents if agents is None else agents
        infos: dict[str, dict[str, Any]] = {}
        for agent in selected_agents:
            handle = self.agent_name_mapping[agent]
            train = self.rail_env.agents[handle] if self.rail_env is not None else None
            state = flatland_infos.get("state", {}).get(handle)
            info = {
                "handle": handle,
                "action_required": bool(
                    flatland_infos.get("action_required", {}).get(handle, False)
                ),
                "malfunction": int(flatland_infos.get("malfunction", {}).get(handle, 0)),
                "speed": float(flatland_infos.get("speed", {}).get(handle, 0.0)),
                "state": state,
                "position": None if train is None else train.position,
                "direction": None if train is None else train.direction,
                "target": None if train is None else train.target,
                "elapsed_steps": 0 if self.rail_env is None else self.rail_env._elapsed_steps,
            }
            if state is not None and hasattr(state, "name"):
                info["state_name"] = state.name
            deadlocked_handles = set()
            stopped_handles = set()
            if self.rail_env is not None and getattr(self.rail_env, "motion_check", None) is not None:
                deadlocked_handles = set(getattr(self.rail_env.motion_check, "deadlocked", set()))
                stopped_handles = set(getattr(self.rail_env.motion_check, "stopped", set()))
            info["deadlocked"] = handle in deadlocked_handles
            info["stopped_by_conflict"] = handle in stopped_handles
            for other_agent, other_handle in self.agent_name_mapping.items():
                if other_handle == handle:
                    continue
                info[f"yields_to_{other_agent}_ok"] = (
                    (handle, other_handle) not in self._last_yield_violations
                )
            info["distance_to_target"] = self._distance_to_target(handle)
            infos[agent] = info
        return infos

    def _all_agents_reached_target(self) -> bool:
        return all(agent.state == TrainState.DONE for agent in self.rail_env.agents)

    def _distance_to_target(self, handle: int) -> float:
        if self.rail_env is None:
            return float("inf")
        train = self.rail_env.agents[handle]
        if train.state == TrainState.DONE:
            return 0.0
        configuration = train.current_configuration
        if configuration is None:
            configuration = train.initial_configuration
        if configuration is None:
            return float("inf")
        try:
            distance_map = self.rail_env.distance_map.get()
            (row, col), direction = configuration
            value = float(distance_map[handle, int(row), int(col), int(direction)])
        except (IndexError, TypeError, ValueError):
            return float("inf")
        return value

    @staticmethod
    def _train_snapshot(train) -> tuple[tuple[int, int] | None, str]:
        position = train.position
        normalized_position = (
            None
            if position is None
            else (int(position[0]), int(position[1]))
        )
        state = train.state
        state_name = state.name if hasattr(state, "name") else str(state)
        return normalized_position, str(state_name)

    @staticmethod
    def _single_track_corridor_position(position: tuple[int, int] | None) -> bool:
        return position is not None and position[0] == 1 and 1 <= position[1] <= 5

    def _yield_violations_for(
        self,
        previous_trains: tuple[tuple[tuple[int, int] | None, str], ...],
        next_trains: tuple[tuple[tuple[int, int] | None, str], ...],
    ) -> frozenset[tuple[int, int]]:
        violations: set[tuple[int, int]] = set()
        for yielder_idx, (previous, current) in enumerate(
            zip(previous_trains, next_trains, strict=True)
        ):
            previous_position, previous_state = previous
            next_position, _ = current
            if (
                next_position == previous_position
                or not self._single_track_corridor_position(next_position)
            ):
                continue
            for priority_idx, (_, priority_state) in enumerate(previous_trains):
                if priority_idx == yielder_idx:
                    continue
                if priority_state == TrainState.DONE.name:
                    continue
                violations.add((yielder_idx, priority_idx))
        return frozenset(violations)

    def _render_rgb_array(self, cell_size: int = 24) -> np.ndarray:
        height = int(self.rail_env.height)
        width = int(self.rail_env.width)
        frame = np.full((height * cell_size, width * cell_size, 3), 245, dtype=np.uint8)

        self._draw_grid(frame, cell_size)
        self._draw_rails(frame, cell_size)
        self._draw_targets(frame, cell_size)
        self._draw_agents(frame, cell_size)
        return frame

    @staticmethod
    def _draw_line(frame: np.ndarray, start: tuple[int, int], end: tuple[int, int], color, width: int = 2):
        x0, y0 = start
        x1, y1 = end
        steps = max(abs(x1 - x0), abs(y1 - y0), 1)
        xs = np.linspace(x0, x1, steps + 1).astype(int)
        ys = np.linspace(y0, y1, steps + 1).astype(int)
        radius = max(width // 2, 1)
        for x, y in zip(xs, ys):
            y_min = max(y - radius, 0)
            y_max = min(y + radius + 1, frame.shape[0])
            x_min = max(x - radius, 0)
            x_max = min(x + radius + 1, frame.shape[1])
            frame[y_min:y_max, x_min:x_max] = color

    @staticmethod
    def _draw_rect(frame: np.ndarray, rect: tuple[int, int, int, int], color):
        x0, y0, x1, y1 = rect
        frame[max(y0, 0):min(y1, frame.shape[0]), max(x0, 0):min(x1, frame.shape[1])] = color

    def _draw_grid(self, frame: np.ndarray, cell_size: int):
        frame[::cell_size, :, :] = 220
        frame[:, ::cell_size, :] = 220

    def _draw_rails(self, frame: np.ndarray, cell_size: int):
        rail_color = np.array([88, 84, 78], dtype=np.uint8)
        direction_offsets = {
            0: (0, -cell_size // 2 + 3),
            1: (cell_size // 2 - 3, 0),
            2: (0, cell_size // 2 - 3),
            3: (-cell_size // 2 + 3, 0),
        }
        for row in range(self.rail_env.height):
            for col in range(self.rail_env.width):
                dirs = set()
                for orientation in range(4):
                    transitions = self.rail_env.rail.get_transitions(((row, col), orientation))
                    for target_direction, is_open in enumerate(transitions):
                        if is_open:
                            dirs.add(orientation)
                            dirs.add(target_direction)
                if not dirs:
                    continue
                cx = col * cell_size + cell_size // 2
                cy = row * cell_size + cell_size // 2
                for direction in dirs:
                    dx, dy = direction_offsets[direction]
                    self._draw_line(frame, (cx, cy), (cx + dx, cy + dy), rail_color, width=3)

    def _draw_targets(self, frame: np.ndarray, cell_size: int):
        target_color = np.array([64, 126, 169], dtype=np.uint8)
        for agent in self.rail_env.agents:
            targets = getattr(agent, "targets", None)
            if targets is None:
                target = getattr(agent, "target", None)
                targets = [] if target is None else [target]
            for target in targets:
                position = getattr(target, "position", target)
                if (
                    isinstance(position, tuple)
                    and len(position) == 2
                    and isinstance(position[0], tuple)
                ):
                    position = position[0]
                row, col = position
                pad = max(cell_size // 4, 3)
                x0 = col * cell_size + pad
                y0 = row * cell_size + pad
                x1 = (col + 1) * cell_size - pad
                y1 = (row + 1) * cell_size - pad
                self._draw_rect(frame, (x0, y0, x1, y0 + 3), target_color)
                self._draw_rect(frame, (x0, y1 - 3, x1, y1), target_color)
                self._draw_rect(frame, (x0, y0, x0 + 3, y1), target_color)
                self._draw_rect(frame, (x1 - 3, y0, x1, y1), target_color)

    def _draw_agents(self, frame: np.ndarray, cell_size: int):
        colors = [
            np.array([213, 60, 55], dtype=np.uint8),
            np.array([0, 145, 234], dtype=np.uint8),
            np.array([0, 170, 112], dtype=np.uint8),
            np.array([225, 155, 48], dtype=np.uint8),
            np.array([128, 91, 172], dtype=np.uint8),
        ]
        direction_offsets = {
            0: (0, -cell_size // 3),
            1: (cell_size // 3, 0),
            2: (0, cell_size // 3),
            3: (-cell_size // 3, 0),
        }
        for idx, agent in enumerate(self.rail_env.agents):
            position = agent.position or agent.initial_position
            if position is None:
                continue
            row, col = position
            cx = col * cell_size + cell_size // 2
            cy = row * cell_size + cell_size // 2
            radius = max(cell_size // 4, 4)
            color = colors[idx % len(colors)]
            yy, xx = np.ogrid[:frame.shape[0], :frame.shape[1]]
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
            frame[mask] = color
            if agent.direction is not None:
                dx, dy = direction_offsets[int(agent.direction)]
                self._draw_line(frame, (cx, cy), (cx + dx, cy + dy), np.array([30, 30, 30], dtype=np.uint8), width=2)

    def __str__(self) -> str:
        return (
            "flatland_railway"
            f"<{self.n_agents}ag,{self._env_kwargs['x_dim']}x{self._env_kwargs['y_dim']}"
            f",{self.scenario}>"
        )
