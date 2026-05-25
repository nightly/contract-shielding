from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import product

from ..impl.env import ACTION_TO_ACCELERATION, CarAction
from src.shield.core import (
    AbstractTransitionOutcome,
    Label,
    LocalAction,
)


CAR_PLATOON_CONTRACT_LABEL_FAMILIES = (
    "conservative_follow_ok",
    "smooth_lead_ok",
    "gap_safe",
    "crashed",
    "too_far",
    "near_min_gap",
    "near_max_gap",
    "closing_fast",
    "opening_fast",
    "safe_to_accelerate_if_front_coasts",
)


def car_platoon_contract_local_alphabet_by_agent(
    agent_ids: tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
    """Return own-gap protocol alphabets for platoon contracts."""
    scopes: dict[str, tuple[str, ...]] = {}
    for agent_idx, agent_id in enumerate(agent_ids):
        scopes[str(agent_id)] = (
            f"agent_{agent_idx}_gap_safe",
            f"agent_{agent_idx}_conservative_follow_ok",
            f"agent_{agent_idx}_smooth_lead_ok",
        )
    return scopes


def car_platoon_contract_diagnostic_alphabet_by_agent(
    agent_ids: tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
    scopes: dict[str, tuple[str, ...]] = {}
    gap_count = len(agent_ids)
    for agent_idx, agent_id in enumerate(agent_ids):
        gap_indices = [agent_idx]
        if agent_idx + 1 < gap_count:
            gap_indices.append(agent_idx + 1)
        protocol_labels = (
            f"agent_{agent_idx}_conservative_follow_ok",
            f"agent_{agent_idx}_smooth_lead_ok",
        )
        gap_labels = tuple(
            f"agent_{gap_idx}_{family}"
            for gap_idx in gap_indices
            for family in CAR_PLATOON_CONTRACT_LABEL_FAMILIES
            if family not in {"conservative_follow_ok", "smooth_lead_ok"}
        )
        scopes[str(agent_id)] = protocol_labels + gap_labels
    return scopes


@dataclass(frozen=True)
class CarPlatoonState:
    velocities: tuple[float, ...]
    distances: tuple[float, ...]
    damaged: tuple[bool, ...]
    conservative_follow_violations: frozenset[int] = frozenset()
    smooth_lead_violations: frozenset[int] = frozenset()
    num_moves: int = 0


class CarPlatoonSafetyModel:
    """Safety labels and local successor support for the car platoon.

    The local successor relation remains pairwise-local for ordinary shields.
    Contract exactness is provided by ``transition_outcomes_for_joint_action``,
    which enumerates the controlled joint action with the stochastic front-car
    action distribution.
    """

    def __init__(self, env) -> None:
        self.env = env.unwrapped if hasattr(env, "unwrapped") else env
        self.agent_ids = tuple(self.env.possible_agents)
        self._agent_index = {
            agent_id: index for index, agent_id in enumerate(self.agent_ids)
        }
        self._actions = tuple(range(self.env.action_space(self.agent_ids[0]).n))
        self.min_velocity = float(self.env.min_velocity)
        self.max_velocity = float(self.env.max_velocity)
        self.min_distance = float(self.env.min_distance)
        self.max_distance = float(self.env.max_distance)
        self.t_act = float(self.env.t_act)
        self.max_cycles = self.env.max_cycles
        self.safety_violation_penalty = float(self.env.safety_violation_penalty)
        self.terminate_on_violation = bool(self.env.terminate_on_violation)
        gap_span = self.max_distance - self.min_distance
        self.near_min_gap_margin = min(10.0, 0.1 * gap_span)
        self.near_max_gap_margin = min(20.0, 0.1 * gap_span)

    def initial_state(self, env: object) -> CarPlatoonState:
        return self.abstract_state(env)

    def abstract_state(self, env: object) -> CarPlatoonState:
        base_env = env.unwrapped if hasattr(env, "unwrapped") else env
        return CarPlatoonState(
            velocities=tuple(self._canonical_number(value) for value in base_env.velocities),
            distances=tuple(self._canonical_number(value) for value in base_env.distances),
            damaged=tuple(bool(value) for value in base_env.damaged),
            conservative_follow_violations=frozenset(
                int(index)
                for index in getattr(
                    base_env,
                    "_last_conservative_follow_violations",
                    frozenset(),
                )
            ),
            smooth_lead_violations=frozenset(
                int(index)
                for index in getattr(
                    base_env,
                    "_last_smooth_lead_violations",
                    frozenset(),
                )
            ),
            num_moves=self._canonical_num_moves(int(getattr(base_env, "num_moves", 0))),
        )

    def local_actions(
        self,
        agent_id: str,
        state: CarPlatoonState,
    ) -> tuple[LocalAction, ...]:
        _ = (agent_id, state)
        return self._actions

    def safety_projection(self, state: CarPlatoonState) -> CarPlatoonState:
        return CarPlatoonState(
            velocities=state.velocities,
            distances=state.distances,
            damaged=state.damaged,
            conservative_follow_violations=state.conservative_follow_violations,
            smooth_lead_violations=state.smooth_lead_violations,
            num_moves=0,
        )

    def label(self, state: CarPlatoonState) -> Label:
        labels: set[str] = set()
        for agent_idx in range(len(state.distances)):
            front_car_idx = agent_idx
            own_car_idx = agent_idx + 1
            distance = float(state.distances[agent_idx])
            own_velocity = float(state.velocities[own_car_idx])
            front_velocity = float(state.velocities[front_car_idx])
            own_damaged = bool(state.damaged[own_car_idx])
            front_damaged = bool(state.damaged[front_car_idx])

            labels.add(
                f"agent_{agent_idx}_velocity_{self._label_number(own_velocity)}"
            )
            labels.add(
                f"agent_{agent_idx}_front_velocity_{self._label_number(front_velocity)}"
            )
            labels.add(f"agent_{agent_idx}_distance_{self._label_number(distance)}")

            if self.min_distance < distance < self.max_distance:
                labels.add(f"agent_{agent_idx}_gap_safe")
            if distance <= self.min_distance:
                labels.add(f"agent_{agent_idx}_crashed")
            if distance >= self.max_distance:
                labels.add(f"agent_{agent_idx}_too_far")
            if distance <= self.min_distance + self.near_min_gap_margin:
                labels.add(f"agent_{agent_idx}_near_min_gap")
            if distance >= self.max_distance - self.near_max_gap_margin:
                labels.add(f"agent_{agent_idx}_near_max_gap")
            if own_velocity - front_velocity >= 2.0:
                labels.add(f"agent_{agent_idx}_closing_fast")
            if front_velocity - own_velocity >= 2.0:
                labels.add(f"agent_{agent_idx}_opening_fast")
            if self._safe_to_accelerate_if_front_coasts(
                distance=distance,
                front_velocity=front_velocity,
                own_velocity=own_velocity,
                front_damaged=front_damaged,
                own_damaged=own_damaged,
            ):
                labels.add(f"agent_{agent_idx}_safe_to_accelerate_if_front_coasts")
            if own_damaged:
                labels.add(f"agent_{agent_idx}_damaged")
            if front_damaged:
                labels.add(f"agent_{agent_idx}_front_damaged")
            if agent_idx not in state.conservative_follow_violations:
                labels.add(f"agent_{agent_idx}_conservative_follow_ok")
            if agent_idx not in state.smooth_lead_violations:
                labels.add(f"agent_{agent_idx}_smooth_lead_ok")

        return frozenset(labels)

    def possible_labels(self) -> tuple[str, ...]:
        labels: set[str] = set()
        velocity_values = range(int(self.min_velocity), int(self.max_velocity) + 1)
        distance_values = range(int(self.min_distance), int(self.max_distance) + 1)
        for agent_idx in range(len(self.agent_ids)):
            for velocity in velocity_values:
                labels.add(
                    f"agent_{agent_idx}_velocity_{self._label_number(float(velocity))}"
                )
                labels.add(
                    f"agent_{agent_idx}_front_velocity_{self._label_number(float(velocity))}"
                )
            for distance in distance_values:
                labels.add(
                    f"agent_{agent_idx}_distance_{self._label_number(float(distance))}"
                )
            labels.add(f"agent_{agent_idx}_gap_safe")
            labels.add(f"agent_{agent_idx}_crashed")
            labels.add(f"agent_{agent_idx}_too_far")
            labels.add(f"agent_{agent_idx}_near_min_gap")
            labels.add(f"agent_{agent_idx}_near_max_gap")
            labels.add(f"agent_{agent_idx}_closing_fast")
            labels.add(f"agent_{agent_idx}_opening_fast")
            labels.add(f"agent_{agent_idx}_safe_to_accelerate_if_front_coasts")
            labels.add(f"agent_{agent_idx}_damaged")
            labels.add(f"agent_{agent_idx}_front_damaged")
            labels.add(f"agent_{agent_idx}_conservative_follow_ok")
            labels.add(f"agent_{agent_idx}_smooth_lead_ok")
        return tuple(sorted(labels))

    def contract_local_alphabet_by_agent(self) -> dict[str, tuple[str, ...]]:
        return car_platoon_contract_local_alphabet_by_agent(self.agent_ids)

    def contract_diagnostic_alphabet_by_agent(self) -> dict[str, tuple[str, ...]]:
        return car_platoon_contract_diagnostic_alphabet_by_agent(self.agent_ids)

    def joint_actions(
        self,
        state: CarPlatoonState,
    ) -> tuple[tuple[LocalAction, ...], ...]:
        return tuple(product(*(self.local_actions(agent_id, state) for agent_id in self.agent_ids)))

    def successors_for_joint_action(
        self,
        state: CarPlatoonState,
        joint_action: tuple[LocalAction, ...],
    ) -> frozenset[CarPlatoonState]:
        normalized_action = tuple(int(action) for action in joint_action)
        if len(normalized_action) != len(self.agent_ids):
            raise ValueError(
                f"Expected {len(self.agent_ids)} local actions, got {len(normalized_action)}."
            )
        return frozenset(
            self._next_state_for_joint_action(
                state,
                lead_action,
                normalized_action,
                advance_time=False,
            )
            for lead_action in self._front_action_support(state, 0)
        )

    def transition_outcomes_for_joint_action(
        self,
        state: CarPlatoonState,
        joint_action: tuple[LocalAction, ...],
    ) -> tuple[AbstractTransitionOutcome[CarPlatoonState], ...]:
        normalized_action = tuple(int(action) for action in joint_action)
        if len(normalized_action) != len(self.agent_ids):
            raise ValueError(
                f"Expected {len(self.agent_ids)} local actions, got {len(normalized_action)}."
            )

        probabilities: defaultdict[CarPlatoonState, float] = defaultdict(float)
        for lead_action, lead_probability in self._front_action_distribution(state):
            next_state = self._next_state_for_joint_action(
                state,
                int(lead_action),
                normalized_action,
                advance_time=True,
            )
            probabilities[next_state] += float(lead_probability)

        outcomes: list[AbstractTransitionOutcome[CarPlatoonState]] = []
        for next_state, probability in probabilities.items():
            safety_violations = self._safety_violations(next_state)
            env_termination = bool(self.terminate_on_violation and any(safety_violations))
            env_truncation = (
                self.max_cycles is not None and next_state.num_moves >= self.max_cycles
            )
            outcomes.append(
                AbstractTransitionOutcome(
                    next_state=next_state,
                    probability=float(probability),
                    rewards=self._rewards_for_state(next_state, safety_violations),
                    terminations={agent_id: env_termination for agent_id in self.agent_ids},
                    truncations={agent_id: env_truncation for agent_id in self.agent_ids},
                    label=self.label(next_state),
                )
            )
        return tuple(outcomes)

    def successors_for_local_action(
        self,
        state: CarPlatoonState,
        agent_id: str,
        action: LocalAction,
    ) -> frozenset[CarPlatoonState]:
        agent_idx = self._agent_index[agent_id]
        front_car_idx = agent_idx
        own_car_idx = agent_idx + 1
        own_action = int(action)

        successors: set[CarPlatoonState] = set()
        for front_action in self._front_action_support(state, front_car_idx):
            velocities = list(state.velocities)
            distances = list(state.distances)
            damaged = list(state.damaged)

            previous_front_velocity = float(velocities[front_car_idx])
            previous_own_velocity = float(velocities[own_car_idx])
            previous_relative_velocity = previous_front_velocity - previous_own_velocity

            next_front_velocity = self._next_velocity(
                previous_front_velocity,
                bool(damaged[front_car_idx]),
                int(front_action),
            )
            next_own_velocity = self._next_velocity(
                previous_own_velocity,
                bool(damaged[own_car_idx]),
                own_action,
            )
            next_relative_velocity = next_front_velocity - next_own_velocity
            next_distance = (
                float(distances[agent_idx])
                + ((previous_relative_velocity + next_relative_velocity) / 2.0)
                * self.t_act
            )

            velocities[front_car_idx] = next_front_velocity
            velocities[own_car_idx] = next_own_velocity
            if next_distance <= self.min_distance:
                next_distance = self.min_distance
                damaged[front_car_idx] = True
                damaged[own_car_idx] = True
                velocities[front_car_idx] = 0.0
                velocities[own_car_idx] = 0.0
            distances[agent_idx] = next_distance

            successors.add(
                CarPlatoonState(
                    velocities=tuple(self._canonical_number(value) for value in velocities),
                    distances=tuple(self._canonical_number(value) for value in distances),
                    damaged=tuple(bool(value) for value in damaged),
                    conservative_follow_violations=(
                        frozenset({agent_idx})
                        if self._conservative_follow_violation(
                            state,
                            agent_idx,
                            own_action,
                        )
                        else frozenset()
                    ),
                    smooth_lead_violations=(
                        frozenset({agent_idx})
                        if self._smooth_lead_violation(state, agent_idx, own_action)
                        else frozenset()
                    ),
                    num_moves=state.num_moves,
                )
            )
        return frozenset(successors)

    def _next_state_for_joint_action(
        self,
        state: CarPlatoonState,
        lead_action: int,
        joint_action: tuple[int, ...],
        *,
        advance_time: bool,
    ) -> CarPlatoonState:
        previous_velocities = [float(value) for value in state.velocities]
        velocities = list(previous_velocities)
        distances = list(state.distances)
        damaged = list(state.damaged)
        actions_by_car = [int(lead_action), *[int(action) for action in joint_action]]

        for car_idx, action in enumerate(actions_by_car):
            velocities[car_idx] = self._next_velocity(
                previous_velocities[car_idx],
                bool(damaged[car_idx]),
                int(action),
            )

        for agent_idx in range(len(joint_action)):
            front_car_idx = agent_idx
            own_car_idx = agent_idx + 1
            previous_relative_velocity = (
                previous_velocities[front_car_idx]
                - previous_velocities[own_car_idx]
            )
            next_relative_velocity = velocities[front_car_idx] - velocities[own_car_idx]
            next_distance = (
                float(distances[agent_idx])
                + ((previous_relative_velocity + next_relative_velocity) / 2.0)
                * self.t_act
            )
            if next_distance <= self.min_distance:
                next_distance = self.min_distance
                damaged[front_car_idx] = True
                damaged[own_car_idx] = True
                velocities[front_car_idx] = 0.0
                velocities[own_car_idx] = 0.0
            distances[agent_idx] = next_distance

        (
            conservative_follow_violations,
            smooth_lead_violations,
        ) = self._protocol_violations(state, joint_action)

        return CarPlatoonState(
            velocities=tuple(self._canonical_number(value) for value in velocities),
            distances=tuple(self._canonical_number(value) for value in distances),
            damaged=tuple(bool(value) for value in damaged),
            conservative_follow_violations=conservative_follow_violations,
            smooth_lead_violations=smooth_lead_violations,
            num_moves=(
                self._canonical_num_moves(state.num_moves + 1)
                if advance_time
                else state.num_moves
            ),
        )

    def _protocol_violations(
        self,
        state: CarPlatoonState,
        joint_action: tuple[int, ...],
    ) -> tuple[frozenset[int], frozenset[int]]:
        conservative_follow_violations: set[int] = set()
        smooth_lead_violations: set[int] = set()
        for agent_idx, action in enumerate(joint_action):
            if self._conservative_follow_violation(state, agent_idx, int(action)):
                conservative_follow_violations.add(agent_idx)
            if self._smooth_lead_violation(state, agent_idx, int(action)):
                smooth_lead_violations.add(agent_idx)
        return (
            frozenset(conservative_follow_violations),
            frozenset(smooth_lead_violations),
        )

    def _conservative_follow_violation(
        self,
        state: CarPlatoonState,
        agent_idx: int,
        action: int,
    ) -> bool:
        return int(action) == int(CarAction.Accelerate) and self._follow_gap_risky(
            state,
            agent_idx,
        )

    def _smooth_lead_violation(
        self,
        state: CarPlatoonState,
        agent_idx: int,
        action: int,
    ) -> bool:
        return (
            int(action) == int(CarAction.Brake)
            and agent_idx + 1 < len(state.distances)
            and self._follow_gap_risky(state, agent_idx + 1)
            and not self._front_gap_requires_brake(state, agent_idx)
        )

    def _follow_gap_risky(self, state: CarPlatoonState, gap_idx: int) -> bool:
        if gap_idx >= len(state.distances):
            return False
        distance = float(state.distances[gap_idx])
        own_velocity = float(state.velocities[gap_idx + 1])
        front_velocity = float(state.velocities[gap_idx])
        return (
            distance <= self.min_distance + self.near_min_gap_margin
            or own_velocity - front_velocity >= 2.0
            or not self._safe_to_accelerate_if_front_coasts(
                distance=distance,
                front_velocity=front_velocity,
                own_velocity=own_velocity,
                front_damaged=bool(state.damaged[gap_idx]),
                own_damaged=bool(state.damaged[gap_idx + 1]),
            )
        )

    def _front_gap_requires_brake(
        self,
        state: CarPlatoonState,
        gap_idx: int,
    ) -> bool:
        if gap_idx >= len(state.distances):
            return False
        distance = float(state.distances[gap_idx])
        own_velocity = float(state.velocities[gap_idx + 1])
        front_velocity = float(state.velocities[gap_idx])
        return (
            distance <= self.min_distance + self.near_min_gap_margin
            or own_velocity - front_velocity >= 2.0
            or distance <= self.min_distance
        )

    def _front_action_distribution(
        self,
        state: CarPlatoonState,
    ) -> tuple[tuple[int, float], ...]:
        velocity = float(state.velocities[0])
        weights = {
            int(CarAction.Brake): (
                0.0
                if velocity <= self.min_velocity
                else (2.0 if velocity > 10.0 else 1.0)
            ),
            int(CarAction.Coast): 1.0,
            int(CarAction.Accelerate): (
                0.0
                if velocity >= self.max_velocity
                else (2.0 if velocity < 0.0 else 1.0)
            ),
        }
        total_weight = sum(weights.values())
        return tuple(
            (action, weight / total_weight)
            for action, weight in sorted(weights.items())
            if weight > 0.0
        )

    def _safety_violations(self, state: CarPlatoonState) -> tuple[bool, ...]:
        return tuple(
            distance <= self.min_distance or distance >= self.max_distance
            for distance in state.distances
        )

    def _rewards_for_state(
        self,
        state: CarPlatoonState,
        safety_violations: tuple[bool, ...],
    ) -> dict[str, float]:
        rewards: dict[str, float] = {}
        for agent_idx, agent_id in enumerate(self.agent_ids):
            reward = -float(state.distances[agent_idx])
            if safety_violations[agent_idx]:
                reward -= self.safety_violation_penalty
            rewards[agent_id] = float(reward)
        return rewards

    def _front_action_support(
        self,
        state: CarPlatoonState,
        front_car_idx: int,
    ) -> tuple[int, ...]:
        if state.damaged[front_car_idx]:
            return (int(CarAction.Coast),)
        if front_car_idx != 0:
            return self._actions

        velocity = float(state.velocities[0])
        supported: list[int] = [int(CarAction.Coast)]
        if velocity > self.min_velocity:
            supported.append(int(CarAction.Brake))
        if velocity < self.max_velocity:
            supported.append(int(CarAction.Accelerate))
        return tuple(sorted(set(supported)))

    def _next_velocity(self, velocity: float, damaged: bool, action: int) -> float:
        if damaged:
            return 0.0
        acceleration = ACTION_TO_ACCELERATION[CarAction(action)]
        return self._canonical_number(
            min(max(float(velocity) + acceleration, self.min_velocity), self.max_velocity)
        )

    def _safe_to_accelerate_if_front_coasts(
        self,
        *,
        distance: float,
        front_velocity: float,
        own_velocity: float,
        front_damaged: bool,
        own_damaged: bool,
    ) -> bool:
        if front_damaged or own_damaged:
            return False
        next_front_velocity = self._next_velocity(
            front_velocity,
            front_damaged,
            int(CarAction.Coast),
        )
        next_own_velocity = self._next_velocity(
            own_velocity,
            own_damaged,
            int(CarAction.Accelerate),
        )
        previous_relative_velocity = float(front_velocity) - float(own_velocity)
        next_relative_velocity = next_front_velocity - next_own_velocity
        next_distance = (
            float(distance)
            + ((previous_relative_velocity + next_relative_velocity) / 2.0)
            * self.t_act
        )
        return self.min_distance < next_distance < self.max_distance

    @staticmethod
    def _canonical_number(value: float) -> float:
        value = float(value)
        rounded = round(value)
        if abs(value - rounded) <= 1e-7:
            return float(rounded)
        return round(value, 6)

    def _canonical_num_moves(self, num_moves: int) -> int:
        if self.max_cycles is None:
            return 0
        return min(int(num_moves), int(self.max_cycles))

    @classmethod
    def _label_number(cls, value: float) -> str:
        value = cls._canonical_number(value)
        rounded = round(value)
        if abs(value - rounded) <= 1e-7:
            integer = int(rounded)
            return f"m{abs(integer)}" if integer < 0 else str(integer)
        return (
            f"{value:.6f}"
            .rstrip("0")
            .rstrip(".")
            .replace("-", "m")
            .replace(".", "p")
        )
