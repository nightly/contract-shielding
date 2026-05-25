from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

from src.environments.flatland import RailEnvActions
from src.shield.core import AbstractTransitionOutcome, Label, LocalAction


FLATLAND_CONTRACT_LABEL_FAMILIES = (
    "deadlocked",
)


def flatland_contract_local_alphabet_by_agent(
    agent_ids: tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
    global_deadlock_props = tuple(
        f"{agent_id}_deadlocked"
        for agent_id in agent_ids
    )
    return {
        str(agent_id): (
            tuple(
                f"{agent_id}_yields_to_{other_agent_id}_ok"
                for other_agent_id in agent_ids
                if other_agent_id != agent_id
            )
            + global_deadlock_props
        )
        for agent_id in agent_ids
    }


def flatland_contract_diagnostic_alphabet_by_agent(
    agent_ids: tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
    return flatland_contract_local_alphabet_by_agent(agent_ids)


@dataclass(frozen=True)
class FlatlandTrainState:
    position: tuple[int, int] | None
    direction: int | None
    state: str
    target: tuple[int, int]
    initial_position: tuple[int, int]
    initial_direction: int


@dataclass(frozen=True)
class FlatlandState:
    trains: tuple[FlatlandTrainState, ...]
    deadlocked: frozenset[int] = frozenset()
    stopped_by_conflict: frozenset[int] = frozenset()
    yield_violations: frozenset[tuple[int, int]] = frozenset()
    step_count: int = 0


class FlatlandSafetyModel:
    """Small deterministic Flatland abstraction for the single-track experiment."""

    def __init__(self, env) -> None:
        self.wrapper = env
        self.env = env.unwrapped if hasattr(env, "unwrapped") else env
        if hasattr(env, "possible_agents"):
            self.agent_ids = tuple(str(agent_id) for agent_id in env.possible_agents)
        else:
            self.agent_ids = tuple(
                f"agent_{idx}"
                for idx in range(int(self.env.get_num_agents()))
            )
        self._agent_index = {
            agent_id: index
            for index, agent_id in enumerate(self.agent_ids)
        }
        self._actions = tuple(int(action) for action in RailEnvActions)
        self.width = int(self.env.width)
        self.height = int(self.env.height)
        self.max_cycles = getattr(env, "max_cycles", None)
        self.progress_reward_scale = float(getattr(env, "progress_reward_scale", 1.0))
        self.arrival_bonus = float(getattr(env, "arrival_bonus", 5.0))
        self.deadlock_penalty = float(getattr(env, "deadlock_penalty", 5.0))
        self.step_penalty = float(getattr(env, "step_penalty", 0.01))

    def initial_state(self, env: object) -> FlatlandState:
        return self.abstract_state(env)

    def abstract_state(self, env: object) -> FlatlandState:
        base_env = env.unwrapped if hasattr(env, "unwrapped") else env
        motion_check = getattr(base_env, "motion_check", None)
        trains = tuple(
            FlatlandTrainState(
                position=_normalize_position(train.position),
                direction=(
                    None
                    if train.direction is None
                    else int(train.direction)
                ),
                state=_state_name(train.state),
                target=_normalize_position(train.target),
                initial_position=_normalize_position(train.initial_position),
                initial_direction=int(train.initial_direction),
            )
            for train in base_env.agents
        )
        raw_stopped = {
            int(handle)
            for handle in getattr(motion_check, "stopped", set())
        }
        return FlatlandState(
            trains=trains,
            deadlocked=frozenset(
                int(handle)
                for handle in getattr(motion_check, "deadlocked", set())
            ),
            stopped_by_conflict=frozenset(
                handle
                for handle in raw_stopped
                if 0 <= handle < len(trains) and trains[handle].position is not None
            ),
            yield_violations=frozenset(
                (int(yielder), int(priority))
                for yielder, priority in getattr(
                    self.wrapper,
                    "_last_yield_violations",
                    frozenset(),
                )
            ),
            step_count=int(getattr(base_env, "_elapsed_steps", 0) or 0),
        )

    def safety_projection(self, state: FlatlandState) -> FlatlandState:
        return FlatlandState(
            trains=state.trains,
            deadlocked=state.deadlocked,
            stopped_by_conflict=frozenset(),
            yield_violations=state.yield_violations,
            step_count=0,
        )

    def local_actions(
        self,
        agent_id: str,
        state: FlatlandState,
    ) -> tuple[LocalAction, ...]:
        train = state.trains[self._agent_index[agent_id]]
        if train.state == "DONE":
            return (int(RailEnvActions.DO_NOTHING),)
        return self._actions

    def label(self, state: FlatlandState) -> Label:
        labels: set[str] = set()
        for agent_idx, train in enumerate(state.trains):
            agent_id = self.agent_ids[agent_idx]
            state_label = _state_label(train.state)
            labels.add(f"{agent_id}_{state_label}")
            if train.position is None:
                labels.add(f"{agent_id}_off_map")
            else:
                row, col = train.position
                labels.add(f"{agent_id}_at_{row}_{col}")
            if train.position == train.target or train.state == "DONE":
                labels.add(f"{agent_id}_at_target")
            if agent_idx in state.deadlocked:
                labels.add(f"{agent_id}_deadlocked")
            if agent_idx in state.stopped_by_conflict:
                labels.add(f"{agent_id}_stopped_by_conflict")
            for other_idx, other_agent_id in enumerate(self.agent_ids):
                if other_idx == agent_idx:
                    continue
                if (agent_idx, other_idx) not in state.yield_violations:
                    labels.add(f"{agent_id}_yields_to_{other_agent_id}_ok")
        return frozenset(labels)

    def possible_labels(self) -> tuple[str, ...]:
        labels: set[str] = set()
        state_labels = (
            "waiting",
            "ready_to_depart",
            "malfunction_off_map",
            "moving",
            "stopped",
            "malfunction",
            "done",
        )
        for agent_idx, agent_id in enumerate(self.agent_ids):
            for row in range(self.height):
                for col in range(self.width):
                    labels.add(f"{agent_id}_at_{row}_{col}")
            labels.add(f"{agent_id}_at_target")
            labels.add(f"{agent_id}_deadlocked")
            labels.add(f"{agent_id}_stopped_by_conflict")
            labels.add(f"{agent_id}_off_map")
            for other_agent_id in self.agent_ids:
                if other_agent_id != agent_id:
                    labels.add(f"{agent_id}_yields_to_{other_agent_id}_ok")
            for state_label in state_labels:
                labels.add(f"{agent_id}_{state_label}")
        return tuple(sorted(labels))

    def contract_local_alphabet_by_agent(self) -> dict[str, tuple[str, ...]]:
        return flatland_contract_local_alphabet_by_agent(self.agent_ids)

    def contract_diagnostic_alphabet_by_agent(self) -> dict[str, tuple[str, ...]]:
        return flatland_contract_diagnostic_alphabet_by_agent(self.agent_ids)

    def joint_actions(
        self,
        state: FlatlandState,
    ) -> tuple[tuple[LocalAction, ...], ...]:
        return tuple(
            product(
                *(
                    self.local_actions(agent_id, state)
                    for agent_id in self.agent_ids
                )
            )
        )

    def successors_for_joint_action(
        self,
        state: FlatlandState,
        joint_action: tuple[LocalAction, ...],
    ) -> frozenset[FlatlandState]:
        return frozenset({self._next_state(state, joint_action)})

    def successors_for_local_action(
        self,
        state: FlatlandState,
        agent_id: str,
        action: LocalAction,
    ) -> frozenset[FlatlandState]:
        agent_idx = self._agent_index[agent_id]
        successors: set[FlatlandState] = set()
        other_action_sets: list[tuple[int, ...]] = []
        for idx, other_agent_id in enumerate(self.agent_ids):
            if idx == agent_idx:
                other_action_sets.append((int(action),))
            else:
                other_action_sets.append(self.local_actions(other_agent_id, state))
        for joint_action in product(*other_action_sets):
            successors.add(self._next_state(state, joint_action))
        return frozenset(successors)

    def transition_outcomes_for_joint_action(
        self,
        state: FlatlandState,
        joint_action: tuple[LocalAction, ...],
    ) -> tuple[AbstractTransitionOutcome[FlatlandState], ...]:
        next_state = self._next_state(state, joint_action)
        terminated = all(train.state == "DONE" for train in next_state.trains)
        truncated = (
            self.max_cycles is not None
            and next_state.step_count >= int(self.max_cycles)
            and not terminated
        )
        return (
            AbstractTransitionOutcome(
                next_state=next_state,
                probability=1.0,
                rewards=self._rewards_for_transition(state, next_state),
                terminations={
                    agent_id: terminated
                    for agent_id in self.agent_ids
                },
                truncations={
                    agent_id: truncated
                    for agent_id in self.agent_ids
                },
                label=self.label(next_state),
            ),
        )

    def _next_state(
        self,
        state: FlatlandState,
        joint_action: tuple[LocalAction, ...],
    ) -> FlatlandState:
        normalized_action = tuple(int(action) for action in joint_action)
        if len(normalized_action) != len(state.trains):
            raise ValueError(
                f"Expected {len(state.trains)} local actions, got {len(normalized_action)}."
            )

        proposals = tuple(
            self._propose_train_update(train, normalized_action[agent_idx])
            for agent_idx, train in enumerate(state.trains)
        )
        current_resources = tuple(
            _resource_for(train.position, agent_idx)
            for agent_idx, train in enumerate(state.trains)
        )
        next_resources = tuple(
            _resource_for(proposal.position, agent_idx)
            for agent_idx, proposal in enumerate(proposals)
        )

        deadlocked: set[int] = set()
        stopped: set[int] = {
            agent_idx
            for agent_idx, (train, proposal) in enumerate(
                zip(state.trains, proposals, strict=True)
            )
            if (
                proposal.state == "STOPPED"
                and proposal.position is not None
                and proposal.position == train.position
            )
        }
        for left_idx, right_idx in product(range(len(state.trains)), repeat=2):
            if left_idx >= right_idx:
                continue
            left_moving = next_resources[left_idx] != current_resources[left_idx]
            right_moving = next_resources[right_idx] != current_resources[right_idx]
            if (
                left_moving
                and right_moving
                and current_resources[left_idx] == next_resources[right_idx]
                and current_resources[right_idx] == next_resources[left_idx]
            ):
                deadlocked.update((left_idx, right_idx))
                stopped.update((left_idx, right_idx))

        target_claims: dict[Any, list[int]] = {}
        for agent_idx, resource in enumerate(next_resources):
            if resource[0] is None or agent_idx in stopped:
                continue
            target_claims.setdefault(resource, []).append(agent_idx)
        for claimants in target_claims.values():
            moving_claimants = [
                agent_idx
                for agent_idx in claimants
                if next_resources[agent_idx] != current_resources[agent_idx]
            ]
            if len(moving_claimants) > 1:
                stopped.update(sorted(moving_claimants)[1:])

        for mover_idx, next_resource in enumerate(next_resources):
            if mover_idx in stopped or next_resource[0] is None:
                continue
            if next_resource == current_resources[mover_idx]:
                continue
            for blocker_idx, current_resource in enumerate(current_resources):
                if blocker_idx == mover_idx:
                    continue
                blocker_stays = next_resources[blocker_idx] == current_resource
                if blocker_stays and next_resource == current_resource:
                    stopped.add(mover_idx)

        yield_violations = _yield_violations_for(
            state.trains,
            current_resources,
            next_resources,
            stopped,
        )

        next_trains: list[FlatlandTrainState] = []
        for agent_idx, (train, proposal) in enumerate(zip(state.trains, proposals, strict=True)):
            if agent_idx in stopped:
                if train.position is None:
                    next_trains.append(train)
                else:
                    next_trains.append(
                        FlatlandTrainState(
                            position=train.position,
                            direction=train.direction,
                            state="STOPPED",
                            target=train.target,
                            initial_position=train.initial_position,
                            initial_direction=train.initial_direction,
                        )
                    )
                continue
            if proposal.position == train.target and proposal.state == "MOVING":
                next_trains.append(
                    FlatlandTrainState(
                        position=None,
                        direction=None,
                        state="DONE",
                        target=train.target,
                        initial_position=train.initial_position,
                        initial_direction=train.initial_direction,
                    )
                )
            else:
                next_trains.append(proposal)

        return FlatlandState(
            trains=tuple(next_trains),
            deadlocked=frozenset(deadlocked),
            stopped_by_conflict=frozenset(stopped),
            yield_violations=frozenset(yield_violations),
            step_count=int(state.step_count) + 1,
        )

    def _propose_train_update(
        self,
        train: FlatlandTrainState,
        action: int,
    ) -> FlatlandTrainState:
        if train.state == "DONE":
            return train
        if train.state == "WAITING":
            return _replace_train(train, state="READY_TO_DEPART")
        if train.state == "READY_TO_DEPART":
            if _is_movement_action(action):
                return FlatlandTrainState(
                    position=train.initial_position,
                    direction=train.initial_direction,
                    state="MOVING",
                    target=train.target,
                    initial_position=train.initial_position,
                    initial_direction=train.initial_direction,
                )
            return train
        if train.state in {"MOVING", "STOPPED", "MALFUNCTION"}:
            if action == int(RailEnvActions.STOP_MOVING):
                return _replace_train(train, state="STOPPED")
            if train.state == "STOPPED" and not _is_movement_action(action):
                return train
            if train.position is None or train.direction is None:
                return train
            next_position = _advance_position(train.position, train.direction)
            return FlatlandTrainState(
                position=next_position,
                direction=train.direction,
                state="MOVING",
                target=train.target,
                initial_position=train.initial_position,
                initial_direction=train.initial_direction,
            )
        return train

    def _rewards_for_transition(
        self,
        state: FlatlandState,
        next_state: FlatlandState,
    ) -> dict[str, float]:
        rewards: dict[str, float] = {}
        for agent_idx, agent_id in enumerate(self.agent_ids):
            train = state.trains[agent_idx]
            next_train = next_state.trains[agent_idx]
            reward = (
                max(
                    self._distance_to_target(train)
                    - self._distance_to_target(next_train),
                    0.0,
                )
                * self.progress_reward_scale
            )
            reward -= self.step_penalty
            if train.state != "DONE" and next_train.state == "DONE":
                reward += self.arrival_bonus
            if agent_idx in next_state.deadlocked:
                reward -= self.deadlock_penalty
            rewards[agent_id] = float(reward)
        return rewards

    @staticmethod
    def _distance_to_target(train: FlatlandTrainState) -> float:
        if train.state == "DONE":
            return 0.0
        position = train.position or train.initial_position
        return float(abs(position[0] - train.target[0]) + abs(position[1] - train.target[1]))


def _normalize_position(position: Any) -> tuple[int, int] | None:
    if position is None:
        return None
    return (int(position[0]), int(position[1]))


def _state_name(state: Any) -> str:
    if hasattr(state, "name"):
        return str(state.name)
    return str(state)


def _state_label(state: str) -> str:
    return str(state).lower()


def _is_movement_action(action: int) -> bool:
    return int(action) in {
        int(RailEnvActions.MOVE_LEFT),
        int(RailEnvActions.MOVE_FORWARD),
        int(RailEnvActions.MOVE_RIGHT),
    }


def _replace_train(train: FlatlandTrainState, **updates: Any) -> FlatlandTrainState:
    values = {
        "position": train.position,
        "direction": train.direction,
        "state": train.state,
        "target": train.target,
        "initial_position": train.initial_position,
        "initial_direction": train.initial_direction,
    }
    values.update(updates)
    return FlatlandTrainState(**values)


def _resource_for(
    position: tuple[int, int] | None,
    agent_idx: int,
) -> tuple[tuple[int, int] | None, int | None]:
    if position is None:
        return (None, int(agent_idx))
    return (position, None)


def _yield_violations_for(
    trains: tuple[FlatlandTrainState, ...],
    current_resources: tuple[tuple[tuple[int, int] | None, int | None], ...],
    next_resources: tuple[tuple[tuple[int, int] | None, int | None], ...],
    stopped: set[int],
) -> set[tuple[int, int]]:
    violations: set[tuple[int, int]] = set()
    for yielder_idx, yielder in enumerate(trains):
        if yielder_idx in stopped:
            continue
        moved = next_resources[yielder_idx] != current_resources[yielder_idx]
        next_position = next_resources[yielder_idx][0]
        if not moved or not _is_single_track_corridor(next_position):
            continue
        for priority_idx, priority in enumerate(trains):
            if priority_idx == yielder_idx:
                continue
            if priority.state == "DONE":
                continue
            violations.add((yielder_idx, priority_idx))
    return violations


def _is_single_track_corridor(position: tuple[int, int] | None) -> bool:
    return position is not None and position[0] == 1 and 1 <= position[1] <= 5


def _advance_position(
    position: tuple[int, int],
    direction: int,
) -> tuple[int, int]:
    row, col = position
    if int(direction) == 0:
        return (row - 1, col)
    if int(direction) == 1:
        return (row, col + 1)
    if int(direction) == 2:
        return (row + 1, col)
    if int(direction) == 3:
        return (row, col - 1)
    raise ValueError(f"Unsupported Flatland direction: {direction!r}")
