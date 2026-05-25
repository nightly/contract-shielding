from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product

import numpy as np

from src.environments.connector.impl.env import (
    EMPTY,
    PATH,
    POSITION,
    RESERVATION_BENCHMARK_GENERATORS,
    RESERVATION_CONFLICT_RESERVED_CELLS,
    TARGET,
    ConnectorAction,
    _is_repeated_later,
    get_path,
    get_position,
    get_target,
    is_target,
    move_position,
)
from src.shield.core import AbstractTransitionOutcome, Label, LocalAction


_ACTION_LABELS = {
    int(ConnectorAction.Noop): "noop",
    int(ConnectorAction.Up): "up",
    int(ConnectorAction.Right): "right",
    int(ConnectorAction.Down): "down",
    int(ConnectorAction.Left): "left",
}


CONNECTOR_CONTRACT_LABEL_FAMILIES = (
    "reserved_route_clear_ok",
    "keeps_reserved_route_clear_ok",
    "keeps_reserved_routes_clear_ok",
    "respects_reservations_ok",
    "respects_reservation_ok",
    "reserved_route_blocked",
    "on_reserved_route",
    "connected",
    "blocked",
)


def connector_contract_local_alphabet_by_agent(
    agent_ids: tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
    return {
        str(agent_id): tuple(
            dict.fromkeys(
                (
                    f"{agent_id}_keeps_reserved_routes_clear_ok",
                    *(
                        f"{agent_id}_keeps_{other_agent_id}_reserved_route_clear_ok"
                        for other_agent_id in agent_ids
                        if other_agent_id != agent_id
                    ),
                    *(
                        f"{owner_agent_id}_reserved_route_clear_ok"
                        for owner_agent_id in agent_ids
                    ),
                    f"{agent_id}_respects_reservations_ok",
                    *(
                        f"{agent_id}_respects_{other_agent_id}_reservation_ok"
                        for other_agent_id in agent_ids
                        if other_agent_id != agent_id
                    ),
                )
            )
        )
        for agent_id in agent_ids
    }


def connector_contract_diagnostic_alphabet_by_agent(
    agent_ids: tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
    reserved_blocking_props = tuple(
        f"{agent_id}_reserved_route_blocked"
        for agent_id in agent_ids
    )
    route_state_props = tuple(
        f"{agent_id}_{label}"
        for agent_id in agent_ids
        for label in ("on_reserved_route", "connected", "blocked")
    )
    return {
        str(agent_id): (
            connector_contract_local_alphabet_by_agent(agent_ids)[str(agent_id)]
            + reserved_blocking_props
            + route_state_props
        )
        for agent_id in agent_ids
    }


def _reservation_grid_projection(
    state: ConnectorState,
    *,
    grid_size: int,
    n_agents: int,
    active_reserved_cells_by_agent: tuple[tuple[tuple[int, int], ...], ...],
) -> tuple[tuple[int, ...], ...]:
    grid = np.zeros((grid_size, grid_size), dtype=np.int32)
    source_grid = np.asarray(state.grid, dtype=np.int32)
    targets = np.asarray(state.targets, dtype=np.int32)
    positions = np.asarray(state.positions, dtype=np.int32)
    grid[tuple(targets.T)] = np.asarray(
        [get_target(idx) for idx in range(n_agents)],
        dtype=np.int32,
    )
    for reserved_cells in active_reserved_cells_by_agent:
        for row, col in reserved_cells:
            value = int(source_grid[row, col])
            if value != EMPTY:
                grid[row, col] = value
    grid[tuple(positions.T)] = np.asarray(
        [get_position(idx) for idx in range(n_agents)],
        dtype=np.int32,
    )
    return _tuple_grid(grid)


@dataclass(frozen=True)
class ConnectorState:
    grid: tuple[tuple[int, ...], ...]
    positions: tuple[tuple[int, int], ...]
    targets: tuple[tuple[int, int], ...]
    reservation_violations: frozenset[tuple[int, int]] = frozenset()
    step_count: int = 0


class ConnectorSafetyModel:
    """Exact Connector transition abstraction for shielding and contracts."""

    def __init__(self, env) -> None:
        self.env = env.unwrapped if hasattr(env, "unwrapped") else env
        self.agent_ids = tuple(str(agent_id) for agent_id in self.env.possible_agents)
        self._agent_index = {
            agent_id: index
            for index, agent_id in enumerate(self.agent_ids)
        }
        self._actions = tuple(int(action) for action in ConnectorAction)
        self.grid_size = int(self.env.grid_size)
        self.n_agents = int(self.env.n_agents)
        self.max_cycles = self.env.max_cycles
        self.reward_mode = str(self.env.reward_mode)
        self.connected_reward = float(self.env.connected_reward)
        self.timestep_reward = float(self.env.timestep_reward)
        self.generator = str(getattr(self.env, "generator", ""))

    def initial_state(self, env: object) -> ConnectorState:
        return self.abstract_state(env)

    def abstract_state(self, env: object) -> ConnectorState:
        base_env = env.unwrapped if hasattr(env, "unwrapped") else env
        return ConnectorState(
            grid=_tuple_grid(base_env.grid),
            positions=_tuple_coords(base_env.positions),
            targets=_tuple_coords(base_env.targets),
            reservation_violations=frozenset(
                (int(violator), int(owner))
                for violator, owner in getattr(
                    base_env,
                    "_last_reservation_violations",
                    frozenset(),
                )
            ),
            step_count=int(getattr(base_env, "num_moves", 0) or 0),
        )

    def safety_projection(self, state: ConnectorState) -> ConnectorState:
        if self.generator in RESERVATION_BENCHMARK_GENERATORS:
            connected = self._connected_mask(state)
            active_reserved_cells_by_agent = tuple(
                tuple()
                if bool(connected[agent_idx])
                else self._reservation_cells_for_agent(agent_idx)
                for agent_idx in range(self.n_agents)
            )
            return replace(
                state,
                grid=_reservation_grid_projection(
                    state,
                    grid_size=self.grid_size,
                    n_agents=self.n_agents,
                    active_reserved_cells_by_agent=active_reserved_cells_by_agent,
                ),
                step_count=0,
            )
        return replace(state, step_count=0)

    def local_actions(
        self,
        agent_id: str,
        state: ConnectorState,
    ) -> tuple[LocalAction, ...]:
        agent_idx = self._agent_index[str(agent_id)]
        mask = self._action_masks_for_state(state)[agent_idx]
        return tuple(
            int(action)
            for action in self._actions
            if bool(mask[int(action)])
        )

    def label(self, state: ConnectorState) -> Label:
        labels: set[str] = set()
        action_mask = self._action_masks_for_state(state)
        connected = self._connected_mask(state)
        blocked = self._blocked_mask(state, action_mask=action_mask)
        reserved_route_blocked = self._reserved_route_blocked_mask(state)
        reserved_route_clear = self._reserved_route_clear_mask(state)
        keeps_reserved_route_clear = self._keeps_reserved_route_clear_mask(state)

        for agent_idx, agent_id in enumerate(self.agent_ids):
            row, col = state.positions[agent_idx]
            target_row, target_col = state.targets[agent_idx]
            labels.add(f"{agent_id}_at_{row}_{col}")
            labels.add(f"{agent_id}_target_{target_row}_{target_col}")
            if bool(connected[agent_idx]):
                labels.add(f"{agent_id}_connected")
            if bool(blocked[agent_idx]):
                labels.add(f"{agent_id}_blocked")
            if bool(reserved_route_blocked[agent_idx]):
                labels.add(f"{agent_id}_reserved_route_blocked")
            if bool(reserved_route_clear[agent_idx]):
                labels.add(f"{agent_id}_reserved_route_clear_ok")
            if (row, col) in self._reservation_cells_for_agent(agent_idx):
                labels.add(f"{agent_id}_on_reserved_route")
            respects_all_reservations = True
            keeps_all_reserved_routes_clear = True
            for other_idx, other_agent_id in enumerate(self.agent_ids):
                if other_idx == agent_idx:
                    continue
                if bool(keeps_reserved_route_clear[agent_idx, other_idx]):
                    labels.add(
                        f"{agent_id}_keeps_{other_agent_id}_reserved_route_clear_ok"
                    )
                else:
                    keeps_all_reserved_routes_clear = False
                if (agent_idx, other_idx) not in state.reservation_violations:
                    labels.add(
                        f"{agent_id}_respects_{other_agent_id}_reservation_ok"
                    )
                else:
                    respects_all_reservations = False
            if keeps_all_reserved_routes_clear:
                labels.add(f"{agent_id}_keeps_reserved_routes_clear_ok")
            if respects_all_reservations:
                labels.add(f"{agent_id}_respects_reservations_ok")
            for action, action_name in _ACTION_LABELS.items():
                if bool(action_mask[agent_idx, action]):
                    labels.add(f"{agent_id}_can_{action_name}")
        return frozenset(labels)

    def possible_labels(self) -> tuple[str, ...]:
        labels: set[str] = set()
        for agent_id in self.agent_ids:
            labels.add(f"{agent_id}_connected")
            labels.add(f"{agent_id}_blocked")
            labels.add(f"{agent_id}_reserved_route_blocked")
            labels.add(f"{agent_id}_reserved_route_clear_ok")
            labels.add(f"{agent_id}_on_reserved_route")
            labels.add(f"{agent_id}_keeps_reserved_routes_clear_ok")
            labels.add(f"{agent_id}_respects_reservations_ok")
            for other_agent_id in self.agent_ids:
                if other_agent_id != agent_id:
                    labels.add(
                        f"{agent_id}_keeps_{other_agent_id}_reserved_route_clear_ok"
                    )
                    labels.add(f"{agent_id}_respects_{other_agent_id}_reservation_ok")
            for action_name in _ACTION_LABELS.values():
                labels.add(f"{agent_id}_can_{action_name}")
            for row in range(self.grid_size):
                for col in range(self.grid_size):
                    labels.add(f"{agent_id}_at_{row}_{col}")
                    labels.add(f"{agent_id}_target_{row}_{col}")
        return tuple(sorted(labels))

    def contract_local_alphabet_by_agent(self) -> dict[str, tuple[str, ...]]:
        return connector_contract_local_alphabet_by_agent(self.agent_ids)

    def contract_diagnostic_alphabet_by_agent(self) -> dict[str, tuple[str, ...]]:
        return connector_contract_diagnostic_alphabet_by_agent(self.agent_ids)

    def joint_actions(
        self,
        state: ConnectorState,
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
        state: ConnectorState,
        joint_action: tuple[LocalAction, ...],
    ) -> frozenset[ConnectorState]:
        return frozenset(
            {
                self._next_state(
                    state,
                    joint_action,
                    advance_time=False,
                )
            }
        )

    def successors_for_local_action(
        self,
        state: ConnectorState,
        agent_id: str,
        action: LocalAction,
    ) -> frozenset[ConnectorState]:
        agent_idx = self._agent_index[str(agent_id)]
        action_sets: list[tuple[int, ...]] = []
        for idx, other_agent_id in enumerate(self.agent_ids):
            if idx == agent_idx:
                action_sets.append((int(action),))
            else:
                action_sets.append(self.local_actions(other_agent_id, state))
        return frozenset(
            self._next_state(state, joint_action, advance_time=False)
            for joint_action in product(*action_sets)
        )

    def transition_outcomes_for_joint_action(
        self,
        state: ConnectorState,
        joint_action: tuple[LocalAction, ...],
    ) -> tuple[AbstractTransitionOutcome[ConnectorState], ...]:
        next_state = self._next_state(state, joint_action, advance_time=True)
        terminated, truncated = self._end_flags(next_state)
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
        state: ConnectorState,
        joint_action: tuple[LocalAction, ...],
        *,
        advance_time: bool,
    ) -> ConnectorState:
        normalized_action = tuple(int(action) for action in joint_action)
        if len(normalized_action) != self.n_agents:
            raise ValueError(
                f"Expected {self.n_agents} local actions, got {len(normalized_action)}."
            )

        grid = np.asarray(state.grid, dtype=np.int32).copy()
        positions = np.asarray(state.positions, dtype=np.int32)
        action_mask = self._action_masks_for_state(state)
        legal_action_taken = action_mask[np.arange(self.n_agents), normalized_action]
        proposed_positions = np.asarray(
            [
                move_position(position, action)
                for position, action in zip(
                    positions,
                    normalized_action,
                    strict=True,
                )
            ],
            dtype=np.int32,
        )
        new_positions = np.where(
            legal_action_taken[:, None],
            proposed_positions,
            positions,
        )
        collided = _is_repeated_later(new_positions)
        connecting = np.asarray(
            [is_target(grid[tuple(position)]) for position in new_positions],
            dtype=bool,
        )
        noop = np.all(new_positions == positions, axis=-1)

        next_grid = grid.copy()
        for idx in range(self.n_agents):
            if bool(collided[idx]) or bool(noop[idx]):
                continue
            next_grid[tuple(positions[idx])] += PATH - POSITION
            if bool(connecting[idx]):
                next_grid[tuple(new_positions[idx])] += POSITION - TARGET
            else:
                next_grid[tuple(new_positions[idx])] += get_position(idx)

        next_positions = np.where(
            collided[:, None],
            positions,
            new_positions,
        ).astype(np.int32, copy=False)
        reservation_violations = self._reservation_violations_for(
            positions,
            next_positions,
            self._connected_mask(state),
        )
        return ConnectorState(
            grid=_tuple_grid(next_grid),
            positions=_tuple_coords(next_positions),
            targets=state.targets,
            reservation_violations=reservation_violations,
            step_count=(
                int(state.step_count) + 1
                if advance_time
                else int(state.step_count)
            ),
        )

    def _action_masks_for_state(self, state: ConnectorState) -> np.ndarray:
        positions = np.asarray(state.positions, dtype=np.int32)
        targets = np.asarray(state.targets, dtype=np.int32)
        grid = np.asarray(state.grid, dtype=np.int32)
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
                masks[idx, int(action)] = (
                    value == 0
                    or value == TARGET + 3 * idx
                )
        return masks

    def _connected_mask(self, state: ConnectorState) -> np.ndarray:
        positions = np.asarray(state.positions, dtype=np.int32)
        targets = np.asarray(state.targets, dtype=np.int32)
        return np.all(positions == targets, axis=-1)

    def _blocked_mask(
        self,
        state: ConnectorState,
        *,
        action_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        if action_mask is None:
            action_mask = self._action_masks_for_state(state)
        return (~self._connected_mask(state)) & ~action_mask[:, 1:].any(axis=1)

    def _connected_or_blocked_mask(self, state: ConnectorState) -> np.ndarray:
        action_mask = self._action_masks_for_state(state)
        return self._connected_mask(state) | self._blocked_mask(
            state,
            action_mask=action_mask,
        )

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

    def _reserved_route_blocked_mask(self, state: ConnectorState) -> np.ndarray:
        connected = self._connected_mask(state)
        grid = np.asarray(state.grid, dtype=np.int32)
        blocked = np.zeros(self.n_agents, dtype=bool)
        for owner_idx in range(self.n_agents):
            if bool(connected[owner_idx]):
                continue
            for row, col in self._reservation_cells_for_agent(owner_idx):
                value = int(grid[row, col])
                if value != EMPTY and not _value_owned_by_agent(value, owner_idx):
                    blocked[owner_idx] = True
                    break
        return blocked

    def _reserved_route_clear_mask(self, state: ConnectorState) -> np.ndarray:
        connected = self._connected_mask(state)
        grid = np.asarray(state.grid, dtype=np.int32)
        clear = np.ones(self.n_agents, dtype=bool)
        for owner_idx in range(self.n_agents):
            if bool(connected[owner_idx]):
                continue
            for row, col in self._reservation_cells_for_agent(owner_idx):
                value = int(grid[row, col])
                if value != EMPTY and not _value_owned_by_agent(value, owner_idx):
                    clear[owner_idx] = False
                    break
        return clear

    def _keeps_reserved_route_clear_mask(self, state: ConnectorState) -> np.ndarray:
        connected = self._connected_mask(state)
        grid = np.asarray(state.grid, dtype=np.int32)
        keeps_clear = np.ones((self.n_agents, self.n_agents), dtype=bool)
        for agent_idx in range(self.n_agents):
            for owner_idx in range(self.n_agents):
                if agent_idx == owner_idx or bool(connected[owner_idx]):
                    continue
                for row, col in self._reservation_cells_for_agent(owner_idx):
                    if _value_is_agent_footprint(int(grid[row, col]), agent_idx):
                        keeps_clear[agent_idx, owner_idx] = False
                        break
        return keeps_clear

    def _end_flags(self, state: ConnectorState) -> tuple[bool, bool]:
        all_done = bool(self._connected_or_blocked_mask(state).all())
        timeout = self.max_cycles is not None and int(state.step_count) >= int(
            self.max_cycles
        )
        terminated = all_done
        truncated = bool(timeout and not all_done)
        return terminated, truncated

    def _rewards_for_transition(
        self,
        state: ConnectorState,
        next_state: ConnectorState,
    ) -> dict[str, float]:
        old_connected = self._connected_mask(state)
        new_connected = self._connected_mask(next_state)
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
        return {
            agent_id: float(rewards[agent_idx])
            for agent_idx, agent_id in enumerate(self.agent_ids)
        }


def _tuple_grid(grid: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(int(value) for value in row)
        for row in np.asarray(grid, dtype=np.int32)
    )


def _tuple_coords(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    return tuple(
        (int(row), int(col))
        for row, col in np.asarray(coords, dtype=np.int32)
    )


def _value_owned_by_agent(value: int, agent_id: int) -> bool:
    return int(value) in {
        get_path(agent_id),
        get_position(agent_id),
        get_target(agent_id),
    }


def _value_is_agent_footprint(value: int, agent_id: int) -> bool:
    return int(value) in {
        get_path(agent_id),
        get_position(agent_id),
    }
