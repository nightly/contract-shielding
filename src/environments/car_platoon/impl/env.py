from __future__ import annotations

import functools
from enum import IntEnum
from typing import Any

import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers


class CarAction(IntEnum):
    Brake = 0
    Coast = 1
    Accelerate = 2


ACTION_TO_ACCELERATION = {
    CarAction.Brake: -2.0,
    CarAction.Coast: 0.0,
    CarAction.Accelerate: 2.0,
}


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


def parallel_env(**kwargs: Any) -> "CarPlatoonParallelEnv":
    return CarPlatoonParallelEnv(**kwargs)


class CarPlatoonParallelEnv(ParallelEnv):
    """Car platoon benchmark with one uncontrolled front car.

    Cars are stored front-to-back. ``agent_i`` controls car ``i + 1`` and
    observes its own velocity, the front car's velocity, and the gap between
    them.
    """

    metadata = {
        "render_modes": ["human", "ansi"],
        "name": "car_platoon_v0",
        "is_parallelizable": True,
    }

    def __init__(
        self,
        *,
        n_cars: int = 10,
        t_act: float = 1.0,
        min_velocity: float = -10.0,
        max_velocity: float = 20.0,
        min_distance: float = 0.0,
        max_distance: float = 200.0,
        initial_distance: float = 50.0,
        max_cycles: int | None = 100,
        safety_violation_penalty: float = 0.0,
        terminate_on_violation: bool = False,
        render_mode: str | None = None,
    ) -> None:
        if n_cars < 2:
            raise ValueError("n_cars must be at least 2.")
        if t_act <= 0:
            raise ValueError("t_act must be positive.")
        if min_velocity >= max_velocity:
            raise ValueError("min_velocity must be less than max_velocity.")
        if min_distance >= max_distance:
            raise ValueError("min_distance must be less than max_distance.")
        if initial_distance <= min_distance:
            raise ValueError("initial_distance must be greater than min_distance.")
        if render_mode not in self.metadata["render_modes"] and render_mode is not None:
            raise ValueError(f"Unsupported render_mode: {render_mode!r}")

        self.n_cars = int(n_cars)
        self.n_agents = self.n_cars - 1
        self.t_act = float(t_act)
        self.min_velocity = float(min_velocity)
        self.max_velocity = float(max_velocity)
        self.min_distance = float(min_distance)
        self.max_distance = float(max_distance)
        self.initial_distance = float(initial_distance)
        self.max_cycles = None if max_cycles is None else int(max_cycles)
        self.safety_violation_penalty = float(safety_violation_penalty)
        self.terminate_on_violation = bool(terminate_on_violation)
        self.render_mode = render_mode

        self.possible_agents = [f"agent_{idx}" for idx in range(self.n_agents)]
        self.agent_name_mapping = {
            agent: idx for idx, agent in enumerate(self.possible_agents)
        }
        self.agents: list[str] = []

        self._action_spaces = {
            agent: spaces.Discrete(len(CarAction)) for agent in self.possible_agents
        }
        self._observation_space = spaces.Box(
            low=np.array(
                [self.min_velocity, self.min_velocity, self.min_distance],
                dtype=np.float32,
            ),
            high=np.array([self.max_velocity, self.max_velocity, np.inf], dtype=np.float32),
            dtype=np.float32,
        )
        self.state_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_cars + self.n_agents + self.n_cars,),
            dtype=np.float32,
        )

        self.np_random, self.np_random_seed = seeding.np_random(None)
        self.velocities = np.zeros(self.n_cars, dtype=np.float32)
        self.distances = np.full(self.n_agents, self.initial_distance, dtype=np.float32)
        self.damaged = np.zeros(self.n_cars, dtype=bool)
        self.num_moves = 0
        self._last_front_action = int(CarAction.Coast)
        self._last_conservative_follow_violations: frozenset[int] = frozenset()
        self._last_smooth_lead_violations: frozenset[int] = frozenset()

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

        self.agents = self.possible_agents[:]
        self.velocities = np.zeros(self.n_cars, dtype=np.float32)
        self.distances = np.full(self.n_agents, self.initial_distance, dtype=np.float32)
        self.damaged = np.zeros(self.n_cars, dtype=bool)
        self.num_moves = 0
        self._last_front_action = int(CarAction.Coast)
        self._last_conservative_follow_violations = frozenset()
        self._last_smooth_lead_violations = frozenset()

        observations = {agent: self.observe(agent) for agent in self.agents}
        infos = {agent: self._info_for_agent(agent) for agent in self.agents}

        if self.render_mode == "human":
            self.render()

        return observations, infos

    def step(self, actions: dict[str, Any]):
        if not self.agents:
            return {}, {}, {}, {}, {}

        current_agents = self.agents[:]
        missing = [agent for agent in current_agents if agent not in actions]
        if missing:
            raise KeyError(f"Missing actions for agents: {missing}")

        controlled_actions = np.zeros(self.n_agents, dtype=np.int64)
        for agent in current_agents:
            action = int(actions[agent])
            if not self.action_space(agent).contains(action):
                raise ValueError(f"Invalid action {action} for {agent!r}.")
            controlled_actions[self.agent_name_mapping[agent]] = action

        (
            self._last_conservative_follow_violations,
            self._last_smooth_lead_violations,
        ) = self._protocol_violations(controlled_actions)

        front_action = self._sample_front_action()
        self._last_front_action = int(front_action)
        action_indices = np.concatenate(
            ([int(front_action)], controlled_actions),
            dtype=np.int64,
        )

        previous_velocities = self.velocities.astype(np.float64, copy=True)
        next_velocities = previous_velocities.copy()
        for car_idx, action_idx in enumerate(action_indices):
            next_velocities[car_idx] = self._next_velocity(
                previous_velocities[car_idx],
                bool(self.damaged[car_idx]),
                int(action_idx),
            )

        old_relative_velocities = previous_velocities[:-1] - previous_velocities[1:]
        new_relative_velocities = next_velocities[:-1] - next_velocities[1:]
        next_distances = (
            self.distances.astype(np.float64)
            + ((old_relative_velocities + new_relative_velocities) / 2.0) * self.t_act
        )

        crashed = next_distances <= self.min_distance
        for gap_idx, did_crash in enumerate(crashed):
            if not did_crash:
                continue
            next_distances[gap_idx] = self.min_distance
            self.damaged[gap_idx] = True
            self.damaged[gap_idx + 1] = True
            next_velocities[gap_idx] = 0.0
            next_velocities[gap_idx + 1] = 0.0

        self.velocities = next_velocities.astype(np.float32)
        self.distances = next_distances.astype(np.float32)
        self.num_moves += 1

        too_far = self.distances >= self.max_distance
        safety_violations = np.logical_or(crashed, too_far)
        env_termination = bool(self.terminate_on_violation and safety_violations.any())
        env_truncation = (
            self.max_cycles is not None and self.num_moves >= self.max_cycles
        )

        observations = {agent: self.observe(agent) for agent in current_agents}
        rewards = {
            agent: self._reward_for_agent(agent, bool(safety_violations[self.agent_name_mapping[agent]]))
            for agent in current_agents
        }
        terminations = {agent: env_termination for agent in current_agents}
        truncations = {agent: env_truncation for agent in current_agents}
        infos = {agent: self._info_for_agent(agent) for agent in current_agents}

        if env_termination or env_truncation:
            self.agents = []

        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminations, truncations, infos

    def observe(self, agent: str) -> np.ndarray:
        idx = self.agent_name_mapping[agent]
        return np.array(
            [
                self.velocities[idx + 1],
                self.velocities[idx],
                self.distances[idx],
            ],
            dtype=np.float32,
        )

    def state(self) -> np.ndarray:
        return np.concatenate(
            (
                self.velocities.astype(np.float32),
                self.distances.astype(np.float32),
                self.damaged.astype(np.float32),
            )
        )

    def render(self):
        if self.render_mode is None:
            return None
        text = self._render_text()
        if self.render_mode == "ansi":
            return text
        print(text)
        return None

    def close(self) -> None:
        return None

    @property
    def unwrapped(self):
        return self

    def _sample_front_action(self) -> int:
        velocity = float(self.velocities[0])
        weights = np.array(
            [
                0.0
                if velocity <= self.min_velocity
                else (2.0 if velocity > 10.0 else 1.0),
                1.0,
                0.0
                if velocity >= self.max_velocity
                else (2.0 if velocity < 0.0 else 1.0),
            ],
            dtype=np.float64,
        )
        probabilities = weights / weights.sum()
        return int(self.np_random.choice(np.arange(len(CarAction)), p=probabilities))

    def _next_velocity(self, velocity: float, damaged: bool, action: int) -> float:
        if damaged:
            return 0.0
        acceleration = ACTION_TO_ACCELERATION[CarAction(action)]
        return float(np.clip(velocity + acceleration, self.min_velocity, self.max_velocity))

    def _reward_for_agent(self, agent: str, safety_violated: bool) -> float:
        idx = self.agent_name_mapping[agent]
        reward = -float(self.distances[idx])
        if safety_violated:
            reward -= self.safety_violation_penalty
        return reward

    def _info_for_agent(self, agent: str) -> dict[str, Any]:
        idx = self.agent_name_mapping[agent]
        distance = float(self.distances[idx])
        crashed = distance <= self.min_distance
        too_far = distance >= self.max_distance
        return {
            "front_action": int(self._last_front_action),
            "distance": distance,
            "gap_safe": bool(self.min_distance < distance < self.max_distance),
            "crashed": bool(crashed),
            "too_far": bool(too_far),
            "safety_violated": bool(crashed or too_far),
            "damaged": bool(self.damaged[idx + 1]),
            "front_damaged": bool(self.damaged[idx]),
            "conservative_follow_ok": idx not in self._last_conservative_follow_violations,
            "smooth_lead_ok": idx not in self._last_smooth_lead_violations,
        }

    def _protocol_violations(
        self,
        controlled_actions: np.ndarray,
    ) -> tuple[frozenset[int], frozenset[int]]:
        conservative_follow_violations: set[int] = set()
        smooth_lead_violations: set[int] = set()
        for agent_idx, action in enumerate(controlled_actions):
            if int(action) == int(CarAction.Accelerate) and self._follow_gap_risky(agent_idx):
                conservative_follow_violations.add(agent_idx)
            if (
                int(action) == int(CarAction.Brake)
                and agent_idx + 1 < self.n_agents
                and self._follow_gap_risky(agent_idx + 1)
                and not self._front_gap_requires_brake(agent_idx)
            ):
                smooth_lead_violations.add(agent_idx)
        return (
            frozenset(conservative_follow_violations),
            frozenset(smooth_lead_violations),
        )

    def _follow_gap_risky(self, gap_idx: int) -> bool:
        distance = float(self.distances[gap_idx])
        own_velocity = float(self.velocities[gap_idx + 1])
        front_velocity = float(self.velocities[gap_idx])
        return (
            distance <= self.min_distance + self._near_min_gap_margin()
            or own_velocity - front_velocity >= 2.0
            or not self._safe_to_accelerate_if_front_coasts(gap_idx)
        )

    def _front_gap_requires_brake(self, gap_idx: int) -> bool:
        distance = float(self.distances[gap_idx])
        own_velocity = float(self.velocities[gap_idx + 1])
        front_velocity = float(self.velocities[gap_idx])
        return (
            distance <= self.min_distance + self._near_min_gap_margin()
            or own_velocity - front_velocity >= 2.0
            or distance <= self.min_distance
        )

    def _safe_to_accelerate_if_front_coasts(self, gap_idx: int) -> bool:
        front_car_idx = gap_idx
        own_car_idx = gap_idx + 1
        if self.damaged[front_car_idx] or self.damaged[own_car_idx]:
            return False
        next_front_velocity = self._next_velocity(
            float(self.velocities[front_car_idx]),
            bool(self.damaged[front_car_idx]),
            int(CarAction.Coast),
        )
        next_own_velocity = self._next_velocity(
            float(self.velocities[own_car_idx]),
            bool(self.damaged[own_car_idx]),
            int(CarAction.Accelerate),
        )
        previous_relative_velocity = (
            float(self.velocities[front_car_idx])
            - float(self.velocities[own_car_idx])
        )
        next_relative_velocity = next_front_velocity - next_own_velocity
        next_distance = (
            float(self.distances[gap_idx])
            + ((previous_relative_velocity + next_relative_velocity) / 2.0)
            * self.t_act
        )
        return self.min_distance < next_distance < self.max_distance

    def _near_min_gap_margin(self) -> float:
        gap_span = self.max_distance - self.min_distance
        return min(10.0, 0.1 * gap_span)

    def _render_text(self) -> str:
        velocity_text = ", ".join(f"{value:.1f}" for value in self.velocities)
        distance_text = ", ".join(f"{value:.1f}" for value in self.distances)
        damaged_text = ", ".join("1" if value else "0" for value in self.damaged)
        return (
            f"CarPlatoon(step={self.num_moves}, front_action={self._last_front_action})\n"
            f"velocities=[{velocity_text}]\n"
            f"distances=[{distance_text}]\n"
            f"damaged=[{damaged_text}]"
        )

    def __str__(self) -> str:
        return f"car_platoon<{self.n_cars}cars>"
