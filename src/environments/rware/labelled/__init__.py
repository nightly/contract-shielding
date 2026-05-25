from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from itertools import product
from typing import Any, MutableMapping

from src.shield.core import AbstractTransitionOutcome, Label, LocalAction


_DIR_ORDER = ("UP", "RIGHT", "DOWN", "LEFT")
_NOOP = 0
_FORWARD = 1
_LEFT = 2
_RIGHT = 3
_TOGGLE_LOAD = 4


RWARE_CONTRACT_LABEL_FAMILIES = (
    "queue_yield_ok",
    "delivery_lane_ok",
    "requested_carrier_progress_ok",
    "in_queue_zone",
    "loaded",
    "carrying_requested",
    "blocking_requested_carrier",
    "in_delivery_zone",
    "at_requested_shelf",
    "in_pickup_zone",
    "requested_carrier_in_queue",
    "requested_carrier_near_goal",
)


def rware_contract_local_alphabet_by_agent(
    agent_ids: tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
    return {
        str(agent_id): (f"{agent_id}_queue_yield_ok",)
        for agent_id in agent_ids
    }


def rware_contract_diagnostic_alphabet_by_agent(
    agent_ids: tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
    formula_props = tuple(
        prop
        for agent_id in agent_ids
        for prop in (
            f"{agent_id}_in_queue_zone",
            f"{agent_id}_carrying_requested",
            f"{agent_id}_blocking_requested_carrier",
        )
    )
    return {
        str(agent_id): (
            f"{agent_id}_queue_yield_ok",
            f"{agent_id}_delivery_lane_ok",
            f"{agent_id}_requested_carrier_progress_ok",
            *formula_props,
            *(
                f"{agent_id}_{family}"
                for family in RWARE_CONTRACT_LABEL_FAMILIES
                if family
                not in {
                    "queue_yield_ok",
                    "delivery_lane_ok",
                    "requested_carrier_progress_ok",
                    "in_queue_zone",
                    "carrying_requested",
                    "blocking_requested_carrier",
                }
            ),
        )
        for agent_id in agent_ids
    }


@dataclass(frozen=True)
class RWAREAgentState:
    x: int
    y: int
    direction: str
    carrying: bool = False
    carrying_requested: bool = False
    carrying_shelf: int | None = None
    has_delivered: bool = False

    @property
    def loaded(self) -> bool:
        return self.carrying or self.carrying_shelf is not None


@dataclass(frozen=True)
class RWAREShelfState:
    shelf_id: int
    x: int
    y: int


@dataclass(frozen=True)
class RWAREState:
    agents: tuple[RWAREAgentState, ...]
    shelves: tuple[RWAREShelfState, ...] = ()
    request_queue: tuple[int, ...] = ()
    standing_shelves: frozenset[tuple[int, int]] = frozenset()
    requested_shelves: frozenset[tuple[int, int]] = frozenset()
    queue_yield_violations: frozenset[int] = frozenset()
    delivery_lane_violations: frozenset[int] = frozenset()
    requested_carrier_progress_violations: frozenset[int] = frozenset()
    step_count: int = 0
    inactive_steps: int = 0


@dataclass(frozen=True)
class RWAREQueueProtocolState:
    agents: tuple[RWAREAgentState, ...]
    requested_shelves: frozenset[tuple[int, int]] = frozenset()
    requested_shelf_unknown: bool = False
    queue_yield_required: frozenset[int] = frozenset()
    queue_yield_clear_forward: frozenset[int] = frozenset()
    requested_carrier_progress_required: frozenset[int] = frozenset()
    queue_yield_violations: frozenset[int] = frozenset()
    delivery_lane_violations: frozenset[int] = frozenset()
    requested_carrier_progress_violations: frozenset[int] = frozenset()
    step_count: int = 0
    inactive_steps: int = 0


class RWARESafetyModel:
    def __init__(
        self,
        env,
        *,
        use_queue_protocol_projection: bool | None = None,
    ) -> None:
        self.env = env.unwrapped if hasattr(env, "unwrapped") else env
        self.agent_ids = tuple(f"agent_{idx}" for idx in range(self.env.n_agents))
        self._agent_index = {
            agent_id: index
            for index, agent_id in enumerate(self.agent_ids)
        }
        self._num_agents = len(self.agent_ids)
        self._actions = tuple(range(self.env.action_space[0].n))
        self._goal_cells = tuple((x, y) for x, y in self.env.goals)
        self._highway_cells = frozenset(
            (x, y)
            for y in range(self.env.grid_size[0])
            for x in range(self.env.grid_size[1])
            if bool(self.env.highways[y, x])
        )
        self._queue_cells = frozenset(self._derive_queue_cells())
        self._delivery_cells = frozenset(
            cell
            for cell in self._queue_cells
            if any(self._manhattan(cell, goal) <= 2 for goal in self._goal_cells)
        )
        self._queue_conflict_relevant_cells_cache: (
            frozenset[tuple[int, int]] | None
        ) = None
        self._queue_conflict_fallback_cell_cache: tuple[int, int] | None = None
        self.benchmark = str(getattr(self.env, "benchmark", "") or "")
        self._use_queue_protocol_projection = (
            self.benchmark == "queue_conflict"
            if use_queue_protocol_projection is None
            else bool(use_queue_protocol_projection)
        )
        self._joint_successor_cache: dict[
            tuple[RWAREState | RWAREQueueProtocolState, tuple[int, ...]],
            frozenset[RWAREState | RWAREQueueProtocolState],
        ] = {}
        self._local_successor_cache: dict[
            tuple[RWAREState | RWAREQueueProtocolState, str, int],
            frozenset[RWAREState | RWAREQueueProtocolState],
        ] = {}
        self._successor_cache_limit = 50_000

    def initial_state(self, env: object) -> RWAREState:
        return self.abstract_state(env)

    def abstract_state(self, env: object) -> RWAREState:
        warehouse = env.unwrapped if hasattr(env, "unwrapped") else env
        shelves = tuple(
            RWAREShelfState(
                shelf_id=int(shelf.id),
                x=int(shelf.x),
                y=int(shelf.y),
            )
            for shelf in warehouse.shelfs
        )
        request_queue = tuple(
            int(shelf.id)
            for shelf in warehouse.request_queue
        )
        agents = tuple(
            RWAREAgentState(
                x=int(agent.x),
                y=int(agent.y),
                direction=agent.dir.name,
                carrying_shelf=(
                    int(agent.carrying_shelf.id)
                    if agent.carrying_shelf is not None
                    else None
                ),
                carrying=agent.carrying_shelf is not None,
                carrying_requested=(
                    agent.carrying_shelf is not None
                    and int(agent.carrying_shelf.id) in request_queue
                ),
                has_delivered=bool(getattr(agent, "has_delivered", False)),
            )
            for agent in warehouse.agents
        )
        return self._make_state(
            agents=agents,
            shelves=shelves,
            request_queue=request_queue,
            queue_yield_violations=frozenset(
                int(index)
                for index in getattr(
                    warehouse,
                    "_last_queue_yield_violations",
                    frozenset(),
                )
            ),
            delivery_lane_violations=frozenset(
                int(index)
                for index in getattr(
                    warehouse,
                    "_last_delivery_lane_violations",
                    frozenset(),
                )
            ),
            requested_carrier_progress_violations=frozenset(
                int(index)
                for index in getattr(
                    warehouse,
                    "_last_requested_carrier_progress_violations",
                    frozenset(),
                )
            ),
            step_count=int(getattr(warehouse, "_cur_steps", 0) or 0),
            inactive_steps=int(getattr(warehouse, "_cur_inactive_steps", 0) or 0),
        )

    def safety_projection(
        self,
        state: RWAREState | RWAREQueueProtocolState,
    ) -> RWAREState | RWAREQueueProtocolState:
        if isinstance(state, RWAREQueueProtocolState):
            return self._canonicalize_queue_protocol_state(state)
        state = self._as_exact_state(state)
        if self.benchmark == "queue_conflict" and self._use_queue_protocol_projection:
            return self._queue_protocol_safety_projection(state)
        if self.benchmark == "queue_conflict":
            return self._queue_conflict_safety_projection(state)
        return self._make_state(
            agents=tuple(
                RWAREAgentState(
                    x=agent.x,
                    y=agent.y,
                    direction=agent.direction,
                    carrying=agent.loaded,
                    carrying_requested=self._agent_carrying_requested(state, agent),
                    carrying_shelf=agent.carrying_shelf,
                    has_delivered=False,
                )
                for agent in state.agents
            ),
            shelves=state.shelves,
            request_queue=state.request_queue,
            queue_yield_violations=state.queue_yield_violations,
            delivery_lane_violations=state.delivery_lane_violations,
            requested_carrier_progress_violations=(
                state.requested_carrier_progress_violations
            ),
        )

    def _queue_protocol_safety_projection(
        self,
        state: RWAREState,
    ) -> RWAREQueueProtocolState:
        state = self._as_exact_state(state)
        queue_yield_required = self._queue_yield_required_indices(state)
        requested_carrier_progress_required = (
            self._requested_carrier_progress_required_indices(state)
        )
        if requested_carrier_progress_required and not queue_yield_required:
            return self._make_requested_carrier_progress_protocol_state(
                state,
                requested_carrier_progress_required=(
                    requested_carrier_progress_required
                ),
            )
        return self._make_queue_yield_protocol_state(
            queue_yield_required=queue_yield_required,
            queue_yield_clear_forward=self._queue_yield_forward_clears_indices(
                state,
                queue_yield_required,
            ),
            requested_carrier_progress_required=(
                requested_carrier_progress_required
            ),
            queue_yield_violations=state.queue_yield_violations,
            delivery_lane_violations=state.delivery_lane_violations,
            requested_carrier_progress_violations=(
                state.requested_carrier_progress_violations
            ),
        )

    def _canonicalize_queue_protocol_state(
        self,
        state: RWAREQueueProtocolState,
    ) -> RWAREQueueProtocolState:
        if (
            state.requested_carrier_progress_required
            and not state.queue_yield_required
        ):
            return self._queue_protocol_safety_projection(
                self._queue_protocol_as_exact_state(state)
            )
        return self._make_queue_yield_protocol_state(
            queue_yield_required=state.queue_yield_required,
            queue_yield_clear_forward=state.queue_yield_clear_forward,
            requested_carrier_progress_required=(
                state.requested_carrier_progress_required
            ),
            queue_yield_violations=state.queue_yield_violations,
            delivery_lane_violations=state.delivery_lane_violations,
            requested_carrier_progress_violations=(
                state.requested_carrier_progress_violations
            ),
        )

    def _queue_conflict_safety_projection(self, state: RWAREState) -> RWAREState:
        state = self._as_exact_state(state)
        relevant_cells = self._queue_conflict_relevant_agent_cells()
        fallback_x, fallback_y = self._queue_conflict_fallback_cell(relevant_cells)
        projected_agents: list[RWAREAgentState] = []
        for agent in state.agents:
            carrying_requested = self._agent_carrying_requested(state, agent)
            if agent.loaded or carrying_requested or (agent.x, agent.y) in relevant_cells:
                projected_agents.append(
                    RWAREAgentState(
                        x=agent.x,
                        y=agent.y,
                        direction=agent.direction,
                        carrying=agent.loaded,
                        carrying_requested=carrying_requested,
                        carrying_shelf=agent.carrying_shelf,
                        has_delivered=False,
                    )
                )
                continue
            projected_agents.append(
                RWAREAgentState(
                    x=fallback_x,
                    y=fallback_y,
                    direction="DOWN",
                    carrying=False,
                    carrying_requested=False,
                    carrying_shelf=None,
                    has_delivered=False,
                )
            )
        return self._make_queue_conflict_state_from_agents(
            state,
            tuple(projected_agents),
        )

    def _make_queue_conflict_state_from_agents(
        self,
        source: RWAREState,
        agents: tuple[RWAREAgentState, ...],
    ) -> RWAREState:
        source = self._as_exact_state(source)
        source_shelves = {
            shelf.shelf_id: shelf
            for shelf in source.shelves
        }
        env_shelves = self._environment_shelves_by_id()
        source_shelves_by_position = {
            (shelf.x, shelf.y): shelf
            for shelf in source.shelves
        }
        env_shelves_by_position = {
            (shelf.x, shelf.y): shelf
            for shelf in env_shelves.values()
        }
        projected_agents: list[RWAREAgentState] = []
        projected_shelves: dict[int, RWAREShelfState] = {}
        projected_request_queue: list[int] = list(source.request_queue)
        local_positions = {
            (agent.x + dx, agent.y + dy)
            for agent in agents
            for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))
        }
        for position in local_positions:
            shelf = source_shelves_by_position.get(position)
            if shelf is None:
                shelf = env_shelves_by_position.get(position)
            if shelf is not None:
                projected_shelves[shelf.shelf_id] = shelf
        for agent_idx, agent in enumerate(agents):
            carrying_requested = self._agent_carrying_requested(source, agent)
            carrying_shelf = agent.carrying_shelf
            if carrying_shelf is not None:
                projected_shelves[carrying_shelf] = RWAREShelfState(
                    carrying_shelf,
                    agent.x,
                    agent.y,
                )
            projected_agents.append(
                RWAREAgentState(
                    x=agent.x,
                    y=agent.y,
                    direction=agent.direction,
                    carrying=agent.loaded,
                    carrying_requested=carrying_requested,
                    carrying_shelf=carrying_shelf,
                    has_delivered=False,
                )
            )
        for shelf in source.shelves:
            env_shelf = env_shelves.get(shelf.shelf_id)
            if env_shelf is None:
                projected_shelves[shelf.shelf_id] = shelf
                continue
            if (shelf.x, shelf.y) != (env_shelf.x, env_shelf.y):
                projected_shelves[shelf.shelf_id] = shelf
        for shelf_id in projected_request_queue:
            if shelf_id in projected_shelves:
                continue
            requested_shelf = source_shelves.get(shelf_id)
            if requested_shelf is None:
                requested_shelf = env_shelves.get(shelf_id)
            if requested_shelf is not None:
                projected_shelves[shelf_id] = requested_shelf
        return self._make_state(
            agents=tuple(projected_agents),
            shelves=tuple(projected_shelves.values()),
            request_queue=tuple(projected_request_queue),
            queue_yield_violations=source.queue_yield_violations,
            delivery_lane_violations=source.delivery_lane_violations,
            requested_carrier_progress_violations=(
                source.requested_carrier_progress_violations
            ),
            step_count=0,
            inactive_steps=0,
        )

    def local_actions(
        self,
        agent_id: str,
        state: RWAREState | RWAREQueueProtocolState,
    ) -> tuple[LocalAction, ...]:
        _ = (agent_id, state)
        return self._actions

    def label(self, state: RWAREState | RWAREQueueProtocolState) -> Label:
        if isinstance(state, RWAREQueueProtocolState):
            return self._label_queue_protocol_state(state)
        state = self._as_exact_state(state)
        labels: set[str] = set()
        shelf_positions = self._shelf_positions(state)
        requested_shelves = frozenset(
            shelf_positions[shelf_id]
            for shelf_id in state.request_queue
            if shelf_id in shelf_positions
        )
        standing_shelves = frozenset(
            position
            for shelf_id, position in shelf_positions.items()
            if shelf_id not in self._carried_shelf_ids(state)
        )
        agent_positions = {
            (agent.x, agent.y): agent_idx
            for agent_idx, agent in enumerate(state.agents)
        }
        standing_requested_shelves = requested_shelves & standing_shelves
        blocking_requested_carrier_indices = self._blocking_requested_carrier_indices(
            state,
            agent_positions,
        )

        for agent_idx, agent in enumerate(state.agents):
            position = (agent.x, agent.y)
            labels.add(f"agent_{agent_idx}_at_{agent.x}_{agent.y}")
            if position in self._goal_cells:
                labels.add(f"agent_{agent_idx}_at_goal")
            if position in self._highway_cells:
                labels.add(f"agent_{agent_idx}_on_highway")
            if position in self._queue_cells:
                labels.add(f"agent_{agent_idx}_in_queue_zone")
            if position in self._delivery_cells:
                labels.add(f"agent_{agent_idx}_in_delivery_zone")
            if agent.loaded:
                labels.add(f"agent_{agent_idx}_loaded")
            if self._agent_carrying_requested(state, agent):
                labels.add(f"agent_{agent_idx}_carrying_requested")
                if position in self._queue_cells:
                    labels.add(f"agent_{agent_idx}_requested_carrier_in_queue")
                if position in self._delivery_cells:
                    labels.add(f"agent_{agent_idx}_requested_carrier_near_goal")
            if (
                not agent.loaded
                and position in standing_requested_shelves
            ):
                labels.add(f"agent_{agent_idx}_at_requested_shelf")
            if (
                not agent.loaded
                and self._is_pickup_zone(position, standing_requested_shelves)
            ):
                labels.add(f"agent_{agent_idx}_in_pickup_zone")
            if agent_idx in blocking_requested_carrier_indices:
                labels.add(f"agent_{agent_idx}_blocking_requested_carrier")
            if agent_idx not in state.queue_yield_violations:
                labels.add(f"agent_{agent_idx}_queue_yield_ok")
            if agent_idx not in state.delivery_lane_violations:
                labels.add(f"agent_{agent_idx}_delivery_lane_ok")
            if agent_idx not in state.requested_carrier_progress_violations:
                labels.add(f"agent_{agent_idx}_requested_carrier_progress_ok")

        for x, y in requested_shelves:
            labels.add(f"requested_shelf_at_{x}_{y}")
        return frozenset(labels)

    def _label_queue_protocol_state(self, state: RWAREQueueProtocolState) -> Label:
        labels: set[str] = set()
        agent_positions = {
            (agent.x, agent.y): agent_idx
            for agent_idx, agent in enumerate(state.agents)
        }
        blocking_requested_carrier_indices = (
            self._blocking_requested_carrier_indices_for_agents(
                state.agents,
                agent_positions,
            )
        )
        for agent_idx, agent in enumerate(state.agents):
            position = (agent.x, agent.y)
            labels.add(f"agent_{agent_idx}_at_{agent.x}_{agent.y}")
            if position in self._goal_cells:
                labels.add(f"agent_{agent_idx}_at_goal")
            if position in self._highway_cells:
                labels.add(f"agent_{agent_idx}_on_highway")
            if position in self._queue_cells:
                labels.add(f"agent_{agent_idx}_in_queue_zone")
            if position in self._delivery_cells:
                labels.add(f"agent_{agent_idx}_in_delivery_zone")
            if agent.loaded:
                labels.add(f"agent_{agent_idx}_loaded")
            if agent.carrying_requested:
                labels.add(f"agent_{agent_idx}_carrying_requested")
                if position in self._queue_cells:
                    labels.add(f"agent_{agent_idx}_requested_carrier_in_queue")
                if position in self._delivery_cells:
                    labels.add(f"agent_{agent_idx}_requested_carrier_near_goal")
            if (
                not agent.loaded
                and not state.requested_shelf_unknown
                and position in state.requested_shelves
            ):
                labels.add(f"agent_{agent_idx}_at_requested_shelf")
            if (
                not agent.loaded
                and not state.requested_shelf_unknown
                and self._is_pickup_zone(position, state.requested_shelves)
            ):
                labels.add(f"agent_{agent_idx}_in_pickup_zone")
            if agent_idx in blocking_requested_carrier_indices:
                labels.add(f"agent_{agent_idx}_blocking_requested_carrier")
            if agent_idx not in state.queue_yield_violations:
                labels.add(f"agent_{agent_idx}_queue_yield_ok")
            if agent_idx not in state.delivery_lane_violations:
                labels.add(f"agent_{agent_idx}_delivery_lane_ok")
            if agent_idx not in state.requested_carrier_progress_violations:
                labels.add(f"agent_{agent_idx}_requested_carrier_progress_ok")
        if not state.requested_shelf_unknown:
            for x, y in state.requested_shelves:
                labels.add(f"requested_shelf_at_{x}_{y}")
        return frozenset(labels)

    def possible_labels(self) -> tuple[str, ...]:
        labels: set[str] = set()
        width = int(self.env.grid_size[1])
        height = int(self.env.grid_size[0])
        for agent_idx in range(self._num_agents):
            for y in range(height):
                for x in range(width):
                    labels.add(f"agent_{agent_idx}_at_{x}_{y}")
            labels.add(f"agent_{agent_idx}_at_goal")
            labels.add(f"agent_{agent_idx}_on_highway")
            labels.add(f"agent_{agent_idx}_in_queue_zone")
            labels.add(f"agent_{agent_idx}_in_delivery_zone")
            labels.add(f"agent_{agent_idx}_loaded")
            labels.add(f"agent_{agent_idx}_carrying_requested")
            labels.add(f"agent_{agent_idx}_blocking_requested_carrier")
            labels.add(f"agent_{agent_idx}_at_requested_shelf")
            labels.add(f"agent_{agent_idx}_in_pickup_zone")
            labels.add(f"agent_{agent_idx}_requested_carrier_in_queue")
            labels.add(f"agent_{agent_idx}_requested_carrier_near_goal")
            labels.add(f"agent_{agent_idx}_queue_yield_ok")
            labels.add(f"agent_{agent_idx}_delivery_lane_ok")
            labels.add(f"agent_{agent_idx}_requested_carrier_progress_ok")
        for y in range(height):
            for x in range(width):
                labels.add(f"requested_shelf_at_{x}_{y}")
        return tuple(sorted(labels))

    def contract_local_alphabet_by_agent(self) -> dict[str, tuple[str, ...]]:
        return rware_contract_local_alphabet_by_agent(self.agent_ids)

    def contract_diagnostic_alphabet_by_agent(self) -> dict[str, tuple[str, ...]]:
        return rware_contract_diagnostic_alphabet_by_agent(self.agent_ids)

    def joint_actions(
        self,
        state: RWAREState | RWAREQueueProtocolState,
    ) -> tuple[tuple[LocalAction, ...], ...]:
        return tuple(product(*(self.local_actions(agent_id, state) for agent_id in self.agent_ids)))

    def successors_for_joint_action(
        self,
        state: RWAREState | RWAREQueueProtocolState,
        joint_action: tuple[LocalAction, ...],
    ) -> frozenset[RWAREState | RWAREQueueProtocolState]:
        normalized_action = tuple(int(action) for action in joint_action)
        cache_key = (state, normalized_action)
        if cache_key not in self._joint_successor_cache:
            self._remember_successors(
                self._joint_successor_cache,
                cache_key,
                self._successors_for_joint_action(
                    state,
                    normalized_action,
                ),
            )
        return self._joint_successor_cache[cache_key]

    def transition_outcomes_for_joint_action(
        self,
        state: RWAREState | RWAREQueueProtocolState,
        joint_action: tuple[LocalAction, ...],
    ) -> tuple[
        AbstractTransitionOutcome[RWAREState | RWAREQueueProtocolState],
        ...,
    ]:
        if isinstance(state, RWAREQueueProtocolState):
            return self._transition_outcomes_for_queue_protocol_state(
                state,
                joint_action,
            )
        state = self._as_exact_state(state)
        normalized_action = tuple(int(action) for action in joint_action)
        if len(normalized_action) != self._num_agents:
            raise ValueError(
                f"Expected {self._num_agents} local actions, got {len(normalized_action)}."
            )
        next_agents, next_shelves, rewards, delivered_shelves, effective_actions = (
            self._apply_joint_action_before_request_replacement(
                state,
                normalized_action,
            )
        )
        if delivered_shelves and self.benchmark == "queue_conflict":
            replacement_shelves_by_id = self._environment_shelves_by_id()
            replacement_shelves_by_id.update(
                {
                    shelf.shelf_id: shelf
                    for shelf in next_shelves
                }
            )
            next_shelves = list(replacement_shelves_by_id.values())

        branches = {
            (state.request_queue, tuple(next_agents)): 1.0
        }
        for shelf_id in delivered_shelves:
            updated: dict[tuple[tuple[int, ...], tuple[RWAREAgentState, ...]], float] = {}
            for (request_queue, branch_agents), branch_probability in branches.items():
                if shelf_id not in request_queue:
                    updated[(request_queue, branch_agents)] = (
                        updated.get((request_queue, branch_agents), 0.0)
                        + branch_probability
                    )
                    continue

                candidates = tuple(
                    candidate.shelf_id
                    for candidate in next_shelves
                    if candidate.shelf_id not in request_queue
                )
                if not candidates:
                    raise ValueError(
                        "RWARE exact transition needs at least one non-requested "
                        "shelf for request replacement."
                    )
                replacement_probability = branch_probability / float(len(candidates))
                replace_index = request_queue.index(shelf_id)
                for candidate_id in candidates:
                    next_queue = list(request_queue)
                    next_queue[replace_index] = candidate_id
                    agents_after_delivery = tuple(
                        self._mark_agent_delivery(agent, shelf_id)
                        for agent in branch_agents
                    )
                    key = (tuple(next_queue), agents_after_delivery)
                    updated[key] = updated.get(key, 0.0) + replacement_probability
            branches = updated

        inactive_steps = 0 if delivered_shelves else state.inactive_steps + 1
        step_count = state.step_count + 1
        max_steps_hit = self.env.max_steps is not None and step_count >= self.env.max_steps
        inactivity_hit = (
            self.env.max_inactivity_steps is not None
            and inactive_steps >= self.env.max_inactivity_steps
        )
        terminations = {
            agent_id: bool(inactivity_hit)
            for agent_id in self.agent_ids
        }
        truncations = {
            agent_id: bool(max_steps_hit)
            for agent_id in self.agent_ids
        }

        probabilities: defaultdict[RWAREState, float] = defaultdict(float)
        for (request_queue, branch_agents), probability in branches.items():
            next_state = self._make_state(
                agents=branch_agents,
                shelves=tuple(sorted(next_shelves, key=lambda shelf: shelf.shelf_id)),
                request_queue=request_queue,
                step_count=step_count,
                inactive_steps=inactive_steps,
            )
            (
                queue_yield_violations,
                delivery_lane_violations,
                requested_carrier_progress_violations,
            ) = self._protocol_violations(state, next_state, effective_actions)
            probabilities[
                replace(
                    next_state,
                    queue_yield_violations=queue_yield_violations,
                    delivery_lane_violations=delivery_lane_violations,
                    requested_carrier_progress_violations=(
                        requested_carrier_progress_violations
                    ),
                )
            ] += probability

        return tuple(
            AbstractTransitionOutcome(
                next_state=next_state,
                probability=float(probability),
                rewards={
                    agent_id: float(rewards[index])
                    for index, agent_id in enumerate(self.agent_ids)
                },
                terminations=terminations,
                truncations=truncations,
                label=self.label(next_state),
            )
            for next_state, probability in probabilities.items()
        )

    def successors_for_local_action(
        self,
        state: RWAREState | RWAREQueueProtocolState,
        agent_id: str,
        action: LocalAction,
    ) -> frozenset[RWAREState | RWAREQueueProtocolState]:
        action = int(action)
        cache_key = (state, agent_id, action)
        if cache_key in self._local_successor_cache:
            return self._local_successor_cache[cache_key]

        agent_idx = self._agent_index[agent_id]
        successors: set[RWAREState] = set()
        for other_actions in product(self._actions, repeat=self._num_agents - 1):
            joint_action = list(other_actions)
            joint_action.insert(agent_idx, action)
            successors.update(
                self.successors_for_joint_action(
                    state,
                    tuple(joint_action),
                )
            )
        result = frozenset(successors)
        self._remember_successors(self._local_successor_cache, cache_key, result)
        return result

    def _remember_successors(
        self,
        cache: MutableMapping[Any, frozenset[RWAREState | RWAREQueueProtocolState]],
        key: Any,
        value: frozenset[RWAREState | RWAREQueueProtocolState],
    ) -> None:
        if len(cache) >= self._successor_cache_limit:
            cache.clear()
        cache[key] = value

    def _successors_for_joint_action(
        self,
        state: RWAREState | RWAREQueueProtocolState,
        joint_action: tuple[int, ...],
    ) -> frozenset[RWAREState | RWAREQueueProtocolState]:
        return frozenset(
            outcome.next_state
            for outcome in self.transition_outcomes_for_joint_action(
                state,
                joint_action,
            )
        )

    def _transition_outcomes_for_queue_protocol_state(
        self,
        state: RWAREQueueProtocolState,
        joint_action: tuple[LocalAction, ...],
    ) -> tuple[AbstractTransitionOutcome[RWAREQueueProtocolState], ...]:
        normalized_action = tuple(int(action) for action in joint_action)
        if len(normalized_action) != self._num_agents:
            raise ValueError(
                f"Expected {self._num_agents} local actions, got {len(normalized_action)}."
            )
        uses_lane_geometry = self._queue_protocol_state_uses_lane_geometry(state)
        coarse_queue_yield_violations = frozenset(
            agent_idx
            for agent_idx in state.queue_yield_required
            if (
                normalized_action[agent_idx] != _FORWARD
                or agent_idx not in state.queue_yield_clear_forward
            )
        )
        queue_yield_violations_for_kernel = (
            frozenset() if uses_lane_geometry else coarse_queue_yield_violations
        )
        requested_carrier_progress_violations = frozenset(
            agent_idx
            for agent_idx in state.requested_carrier_progress_required
            if normalized_action[agent_idx] != _FORWARD
        )
        raw_exact_outcomes = (
            self._exact_queue_protocol_transition_outcomes(
                state,
                normalized_action,
                queue_yield_violations=queue_yield_violations_for_kernel,
                requested_carrier_progress_violations=(
                    requested_carrier_progress_violations
                ),
            )
            if uses_lane_geometry
            else self._off_lane_queue_protocol_transition_outcomes(
                state,
                normalized_action,
                queue_yield_violations=queue_yield_violations_for_kernel,
                requested_carrier_progress_violations=(
                    requested_carrier_progress_violations
                ),
            )
        )
        if (
            uses_lane_geometry
            and state.requested_carrier_progress_required
            and not state.queue_yield_required
        ):
            raw_exact_outcomes = tuple(
                dict.fromkeys(
                    (
                        *raw_exact_outcomes,
                        *self._off_lane_queue_protocol_transition_outcomes(
                            state,
                            normalized_action,
                            queue_yield_violations=queue_yield_violations_for_kernel,
                            requested_carrier_progress_violations=(
                                requested_carrier_progress_violations
                            ),
                        ),
                    )
                )
            )
        next_required_options = (
            self._queue_protocol_requirement_options(include_blocked=False)
            if self._queue_protocol_needs_request_refresh(state)
            else ()
        )
        observed_queue_yield_violation_sets = frozenset(
            next_state.queue_yield_violations
            for next_state in raw_exact_outcomes
        )
        if not observed_queue_yield_violation_sets:
            observed_queue_yield_violation_sets = frozenset(
                {queue_yield_violations_for_kernel}
            )
        post_yield_required_options = (
            self._queue_protocol_post_yield_requirement_options(
                state,
                queue_yield_violations=frozenset(),
            )
            if frozenset() in observed_queue_yield_violation_sets
            else ()
        )
        post_violation_option_groups = tuple(
            (
                self._queue_protocol_post_violation_requirement_options(
                    state,
                    queue_yield_violations=queue_yield_violations,
                ),
                queue_yield_violations,
            )
            for queue_yield_violations in observed_queue_yield_violation_sets
            if queue_yield_violations
        )
        post_yield_progress_blocker_outcomes: list[RWAREQueueProtocolState] = []
        for (
            _next_queue_required,
            _next_queue_yield_clear_forward,
            next_progress_required,
        ) in post_yield_required_options:
            if not next_progress_required:
                continue
            post_yield_progress_blocker_outcomes.extend(
                self._queue_protocol_progress_blocker_ahead_options(
                    replace(
                        state,
                        queue_yield_required=frozenset(),
                        queue_yield_clear_forward=frozenset(),
                        requested_carrier_progress_required=next_progress_required,
                    ),
                    queue_yield_violations=frozenset(),
                    requested_carrier_progress_violations=(
                        requested_carrier_progress_violations
                    ),
                )
            )
        if post_yield_progress_blocker_outcomes:
            raw_exact_outcomes = tuple(
                dict.fromkeys(
                    (*raw_exact_outcomes, *post_yield_progress_blocker_outcomes)
                )
            )
        delivery_lane_options = self._queue_protocol_agent_subsets()
        exact_outcomes = tuple(
            replace(next_state, delivery_lane_violations=delivery_lane_violations)
            for next_state in raw_exact_outcomes
            for delivery_lane_violations in delivery_lane_options
        )
        probability = 1.0 / float(
            len(exact_outcomes)
            + (
                len(next_required_options)
                + len(post_yield_required_options)
                + sum(
                    len(post_violation_required_options)
                    for (
                        post_violation_required_options,
                        _queue_yield_violations,
                    ) in post_violation_option_groups
                )
            )
            * len(delivery_lane_options)
        )
        outcomes: list[AbstractTransitionOutcome[RWAREQueueProtocolState]] = []
        for next_state in exact_outcomes:
            outcomes.append(
                AbstractTransitionOutcome(
                    next_state=next_state,
                    probability=probability,
                    rewards={agent_id: 0.0 for agent_id in self.agent_ids},
                    terminations={agent_id: False for agent_id in self.agent_ids},
                    truncations={agent_id: False for agent_id in self.agent_ids},
                    label=self.label(next_state),
                )
            )
        requirement_option_groups: list[
            tuple[
                tuple[tuple[frozenset[int], frozenset[int], frozenset[int]], ...],
                frozenset[int],
            ]
        ] = []
        if next_required_options:
            requirement_option_groups.append(
                (next_required_options, queue_yield_violations_for_kernel)
            )
        if post_yield_required_options:
            requirement_option_groups.append(
                (post_yield_required_options, frozenset())
            )
        requirement_option_groups.extend(post_violation_option_groups)
        for requirement_options, next_queue_yield_violations in requirement_option_groups:
            for (
                next_queue_required,
                next_queue_yield_clear_forward,
                next_progress_required,
            ) in requirement_options:
                for delivery_lane_violations in delivery_lane_options:
                    next_state = self._make_queue_yield_protocol_state(
                        queue_yield_required=next_queue_required,
                        queue_yield_clear_forward=next_queue_yield_clear_forward,
                        requested_carrier_progress_required=next_progress_required,
                        queue_yield_violations=next_queue_yield_violations,
                        delivery_lane_violations=delivery_lane_violations,
                        requested_carrier_progress_violations=(
                            requested_carrier_progress_violations
                        ),
                    )
                    outcomes.append(
                        AbstractTransitionOutcome(
                            next_state=next_state,
                            probability=probability,
                            rewards={agent_id: 0.0 for agent_id in self.agent_ids},
                            terminations={
                                agent_id: False for agent_id in self.agent_ids
                            },
                            truncations={
                                agent_id: False for agent_id in self.agent_ids
                            },
                            label=self.label(next_state),
                        )
                    )
        return tuple(outcomes)

    def _queue_protocol_needs_request_refresh(
        self,
        state: RWAREQueueProtocolState,
    ) -> bool:
        return (
            not state.queue_yield_required
            and not state.requested_carrier_progress_required
            and not any(agent.carrying_requested for agent in state.agents)
        )

    def _queue_protocol_post_yield_requirement_options(
        self,
        state: RWAREQueueProtocolState,
        *,
        queue_yield_violations: frozenset[int],
    ) -> tuple[tuple[frozenset[int], frozenset[int], frozenset[int]], ...]:
        if not state.queue_yield_required or queue_yield_violations:
            return ()
        options: list[tuple[frozenset[int], frozenset[int], frozenset[int]]] = [
            (frozenset(), frozenset(), frozenset())
        ]
        progress_candidates = frozenset(
            agent_idx
            for agent_idx, agent in enumerate(state.agents)
            if agent.carrying_requested and agent_idx not in state.queue_yield_required
        )
        options.extend(
            (frozenset(), frozenset(), subset)
            for subset in self._queue_protocol_agent_subsets()
            if subset and subset <= progress_candidates
        )
        options.append(
            (
                state.queue_yield_required,
                state.queue_yield_required,
                frozenset(),
            )
        )
        return tuple(dict.fromkeys(options))

    def _queue_protocol_post_violation_requirement_options(
        self,
        state: RWAREQueueProtocolState,
        *,
        queue_yield_violations: frozenset[int],
    ) -> tuple[tuple[frozenset[int], frozenset[int], frozenset[int]], ...]:
        if not state.queue_yield_required or not queue_yield_violations:
            return ()
        return tuple(
            (state.queue_yield_required, clear_subset, frozenset())
            for clear_subset in self._queue_protocol_agent_subsets()
            if clear_subset <= state.queue_yield_required
        )

    def _queue_protocol_state_uses_lane_geometry(
        self,
        state: RWAREQueueProtocolState,
    ) -> bool:
        return bool(
            state.queue_yield_required
            or self._queue_protocol_progress_blocker_ahead(state) is not None
        )

    def _off_lane_queue_protocol_transition_outcomes(
        self,
        state: RWAREQueueProtocolState,
        joint_action: tuple[int, ...],
        *,
        queue_yield_violations: frozenset[int],
        requested_carrier_progress_violations: frozenset[int],
    ) -> tuple[RWAREQueueProtocolState, ...]:
        requirement_options = {
            (
                state.queue_yield_required,
                state.queue_yield_clear_forward,
                state.requested_carrier_progress_required,
            )
        }
        if state.requested_carrier_progress_required:
            requirement_options.update(
                self._queue_protocol_requirement_options(include_blocked=False)
            )
        progress_blocking_teammates = (
            frozenset(
                agent_idx
                for agent_idx, action in enumerate(joint_action)
                if (
                    int(action) == _FORWARD
                    and agent_idx not in state.requested_carrier_progress_required
                )
            )
            if state.requested_carrier_progress_required
            else frozenset()
        )
        if progress_blocking_teammates:
            requirement_options.update(
                self._queue_protocol_requirement_options(include_blocked=True)
            )
        progress_conflict_carriers = frozenset(
            agent_idx
            for agent_idx in state.requested_carrier_progress_required
            if int(joint_action[agent_idx]) in {_FORWARD, _LEFT, _RIGHT}
        )
        if progress_conflict_carriers:
            requirement_options.update(
                self._queue_protocol_requirement_options(include_blocked=True)
            )
        progressed_carriers = frozenset(
            agent_idx
            for agent_idx in state.requested_carrier_progress_required
            if int(joint_action[agent_idx]) == _FORWARD
        )
        if progressed_carriers:
            requirement_options.add(
                (
                    state.queue_yield_required,
                    state.queue_yield_clear_forward,
                    state.requested_carrier_progress_required
                    - progressed_carriers,
                )
            )
        dropped_progress = frozenset(
            agent_idx
            for agent_idx in state.requested_carrier_progress_required
            if int(joint_action[agent_idx]) == _TOGGLE_LOAD
        )
        if dropped_progress:
            requirement_options.add(
                (
                    state.queue_yield_required,
                    state.queue_yield_clear_forward,
                    state.requested_carrier_progress_required
                    - dropped_progress,
                )
            )
        outcomes = [
            self._make_queue_yield_protocol_state(
                queue_yield_required=next_queue_required,
                queue_yield_clear_forward=next_queue_yield_clear_forward,
                requested_carrier_progress_required=next_progress_required,
                queue_yield_violations=queue_yield_violations,
                delivery_lane_violations=frozenset(),
                requested_carrier_progress_violations=(
                    requested_carrier_progress_violations
                ),
            )
            for (
                next_queue_required,
                next_queue_yield_clear_forward,
                next_progress_required,
            ) in requirement_options
        ]
        progressed_carriers = frozenset(
            agent_idx
            for agent_idx in state.requested_carrier_progress_required
            if int(joint_action[agent_idx]) == _FORWARD
        )
        if progressed_carriers:
            outcomes.append(
                self._make_queue_yield_protocol_state(
                    queue_yield_required=state.queue_yield_required,
                    queue_yield_clear_forward=state.queue_yield_clear_forward,
                    requested_carrier_progress_required=(
                        state.requested_carrier_progress_required
                    ),
                    queue_yield_violations=queue_yield_violations,
                    delivery_lane_violations=frozenset(),
                    requested_carrier_progress_violations=(
                        requested_carrier_progress_violations
                        | progressed_carriers
                    ),
                )
            )
        if state.requested_carrier_progress_required:
            outcomes.extend(
                self._queue_protocol_progress_blocker_ahead_options(
                    state,
                    queue_yield_violations=queue_yield_violations,
                    requested_carrier_progress_violations=(
                        requested_carrier_progress_violations
                    ),
                )
            )
        return tuple(dict.fromkeys(outcomes))

    def _exact_queue_protocol_transition_outcomes(
        self,
        state: RWAREQueueProtocolState,
        joint_action: tuple[int, ...],
        *,
        queue_yield_violations: frozenset[int],
        requested_carrier_progress_violations: frozenset[int],
    ) -> tuple[RWAREQueueProtocolState, ...]:
        exact_state = self._queue_protocol_as_exact_state(state)
        next_agents, next_shelves, _rewards, delivered_shelves, effective_actions = (
            self._apply_joint_action_before_request_replacement(
                exact_state,
                joint_action,
            )
        )
        delivered = set(delivered_shelves)
        next_request_queue = tuple(
            shelf_id
            for shelf_id in exact_state.request_queue
            if shelf_id not in delivered
        )

        def _state_from_branch(
            branch_agents: tuple[RWAREAgentState, ...],
            branch_request_queue: tuple[int, ...],
        ) -> RWAREQueueProtocolState:
            next_state = self._make_state(
                agents=branch_agents,
                shelves=tuple(sorted(next_shelves, key=lambda shelf: shelf.shelf_id)),
                request_queue=branch_request_queue,
                queue_yield_violations=queue_yield_violations,
                requested_carrier_progress_violations=(
                    requested_carrier_progress_violations
                ),
            )
            (
                exact_queue_yield_violations,
                delivery_lane_violations,
                exact_progress_violations,
            ) = self._protocol_violations(exact_state, next_state, effective_actions)
            next_state = replace(
                next_state,
                queue_yield_violations=queue_yield_violations
                | exact_queue_yield_violations,
                delivery_lane_violations=delivery_lane_violations,
                requested_carrier_progress_violations=(
                    requested_carrier_progress_violations
                    | exact_progress_violations
                ),
            )
            return self._queue_protocol_safety_projection(next_state)

        delivered_agents = tuple(
            self._mark_agent_delivery(agent, agent.carrying_shelf)
            if agent.carrying_shelf in delivered
            else agent
            for agent in next_agents
        )
        outcomes = {
            _state_from_branch(delivered_agents, next_request_queue)
        }
        off_lane_toggles = tuple(
            agent_idx
            for agent_idx, agent in enumerate(exact_state.agents)
            if (
                int(joint_action[agent_idx]) == _TOGGLE_LOAD
                and agent.carrying_shelf in exact_state.request_queue
                and (agent.x, agent.y) not in self._queue_protocol_relevant_agent_cells()
            )
        )
        for toggle_mask in range(1, 1 << len(off_lane_toggles)):
            branch_agents = list(delivered_agents)
            for bit, agent_idx in enumerate(off_lane_toggles):
                if not toggle_mask & (1 << bit):
                    continue
                agent = branch_agents[agent_idx]
                branch_agents[agent_idx] = RWAREAgentState(
                    x=agent.x,
                    y=agent.y,
                    direction=agent.direction,
                    carrying=False,
                    carrying_requested=False,
                    carrying_shelf=None,
                    has_delivered=False,
                )
            outcomes.add(
                _state_from_branch(
                    tuple(branch_agents),
                    exact_state.request_queue,
                )
            )
        return tuple(outcomes)

    def _apply_joint_action_before_request_replacement(
        self,
        state: RWAREState,
        joint_action: tuple[int, ...],
    ) -> tuple[
        list[RWAREAgentState],
        list[RWAREShelfState],
        list[float],
        tuple[int, ...],
        tuple[int, ...],
    ]:
        state = self._as_exact_state(state)
        agents = list(state.agents)
        shelves_by_id = dict(self._shelves_by_id(state))
        standing_shelves = {
            (shelf.x, shelf.y): shelf.shelf_id
            for shelf in state.shelves
            if shelf.shelf_id not in self._carried_shelf_ids(state)
        }

        targets: dict[int, tuple[int, int]] = {}
        effective_actions: list[int] = list(joint_action)

        for idx, (agent, action) in enumerate(zip(agents, joint_action, strict=True)):
            target = self._requested_location(agent, int(action))
            if (
                int(action) == _FORWARD
                and agent.loaded
                and target != (agent.x, agent.y)
                and target in standing_shelves
            ):
                effective_actions[idx] = _NOOP
                target = (agent.x, agent.y)
            targets[idx] = target

        committed = self._committed_agent_indices(agents, targets)

        for idx, action in enumerate(effective_actions):
            if action == _FORWARD and idx not in committed:
                effective_actions[idx] = _NOOP
                targets[idx] = (agents[idx].x, agents[idx].y)

        next_agents: list[RWAREAgentState] = list(agents)
        rewards = [0.0 for _ in self.agent_ids]
        for idx, (agent, action) in enumerate(zip(agents, effective_actions, strict=True)):
            if action == _FORWARD:
                if agent.carrying_shelf is not None:
                    shelves_by_id[agent.carrying_shelf] = RWAREShelfState(
                        shelf_id=agent.carrying_shelf,
                        x=targets[idx][0],
                        y=targets[idx][1],
                    )
                next_agents[idx] = RWAREAgentState(
                    x=targets[idx][0],
                    y=targets[idx][1],
                    direction=agent.direction,
                    carrying_shelf=agent.carrying_shelf,
                    has_delivered=agent.has_delivered,
                )
            elif action == _LEFT:
                next_agents[idx] = RWAREAgentState(
                    x=agent.x,
                    y=agent.y,
                    direction=self._turn(agent.direction, left=True),
                    carrying_shelf=agent.carrying_shelf,
                    has_delivered=agent.has_delivered,
                )
            elif action == _RIGHT:
                next_agents[idx] = RWAREAgentState(
                    x=agent.x,
                    y=agent.y,
                    direction=self._turn(agent.direction, left=False),
                    carrying_shelf=agent.carrying_shelf,
                    has_delivered=agent.has_delivered,
                )
            elif action == _TOGGLE_LOAD and not agent.loaded and (agent.x, agent.y) in standing_shelves:
                shelf_id = standing_shelves[(agent.x, agent.y)]
                next_agents[idx] = RWAREAgentState(
                    x=agent.x,
                    y=agent.y,
                    direction=agent.direction,
                    carrying_shelf=shelf_id,
                    has_delivered=agent.has_delivered,
                )
            elif action == _TOGGLE_LOAD and agent.loaded and (agent.x, agent.y) not in self._highway_cells:
                if agent.has_delivered and self._reward_type_name == "TWO_STAGE":
                    rewards[idx] += 0.5
                next_agents[idx] = RWAREAgentState(
                    x=agent.x,
                    y=agent.y,
                    direction=agent.direction,
                    carrying_shelf=None,
                    has_delivered=False,
                )

        delivered_shelves: list[int] = []
        for goal_cell in self._goal_cells:
            for agent_idx, agent in enumerate(next_agents):
                if agent.carrying_shelf is None:
                    continue
                if (agent.x, agent.y) != goal_cell:
                    continue
                if agent.carrying_shelf not in state.request_queue:
                    continue
                delivered_shelves.append(agent.carrying_shelf)
                if self._reward_type_name == "GLOBAL":
                    rewards = [reward + 1.0 for reward in rewards]
                elif self._reward_type_name == "INDIVIDUAL":
                    rewards[agent_idx] += 1.0
                elif self._reward_type_name == "TWO_STAGE":
                    rewards[agent_idx] += 0.5
                break

        return (
            next_agents,
            list(shelves_by_id.values()),
            rewards,
            tuple(delivered_shelves),
            tuple(effective_actions),
        )

    @property
    def _reward_type_name(self) -> str:
        reward_type = getattr(self.env, "reward_type", "")
        return str(getattr(reward_type, "name", reward_type))

    def _environment_shelves_by_id(self) -> dict[int, RWAREShelfState]:
        return {
            int(shelf.id): RWAREShelfState(
                shelf_id=int(shelf.id),
                x=int(shelf.x),
                y=int(shelf.y),
            )
            for shelf in getattr(self.env, "shelfs", ())
        }

    def _queue_conflict_relevant_agent_cells(self) -> frozenset[tuple[int, int]]:
        cached = self._queue_conflict_relevant_cells_cache
        if cached is not None:
            return cached

        base_cells = set(self._goal_cells) | set(self._queue_cells)
        cells: set[tuple[int, int]] = set(base_cells)
        cached = frozenset(cells)
        self._queue_conflict_relevant_cells_cache = cached
        return cached

    def _queue_conflict_fallback_cell(
        self,
        relevant_cells: frozenset[tuple[int, int]],
    ) -> tuple[int, int]:
        cached = self._queue_conflict_fallback_cell_cache
        if cached is not None:
            return cached

        shelf_positions = {
            (shelf.x, shelf.y)
            for shelf in self._environment_shelves_by_id().values()
        }
        width = int(self.env.grid_size[1])
        height = int(self.env.grid_size[0])
        for y in range(height):
            for x in range(width - 1, -1, -1):
                if (x, y) in relevant_cells or (x, y) in shelf_positions:
                    continue
                self._queue_conflict_fallback_cell_cache = (x, y)
                return (x, y)
        self._queue_conflict_fallback_cell_cache = (0, 0)
        return (0, 0)

    def _make_state(
        self,
        *,
        agents: tuple[RWAREAgentState, ...],
        shelves: tuple[RWAREShelfState, ...],
        request_queue: tuple[int, ...],
        queue_yield_violations: frozenset[int] = frozenset(),
        delivery_lane_violations: frozenset[int] = frozenset(),
        requested_carrier_progress_violations: frozenset[int] = frozenset(),
        step_count: int = 0,
        inactive_steps: int = 0,
    ) -> RWAREState:
        request_set = frozenset(request_queue)
        normalized_agents = tuple(
            RWAREAgentState(
                x=agent.x,
                y=agent.y,
                direction=agent.direction,
                carrying=agent.loaded,
                carrying_requested=(
                    agent.carrying_shelf in request_set
                    if agent.carrying_shelf is not None
                    else bool(agent.carrying_requested)
                ),
                carrying_shelf=agent.carrying_shelf,
                has_delivered=agent.has_delivered,
            )
            for agent in agents
        )
        carried_shelf_ids = frozenset(
            agent.carrying_shelf
            for agent in normalized_agents
            if agent.carrying_shelf is not None
        )
        shelf_positions = {
            shelf.shelf_id: (shelf.x, shelf.y)
            for shelf in shelves
        }
        return RWAREState(
            agents=normalized_agents,
            shelves=tuple(sorted(shelves, key=lambda shelf: shelf.shelf_id)),
            request_queue=tuple(request_queue),
            standing_shelves=frozenset(
                position
                for shelf_id, position in shelf_positions.items()
                if shelf_id not in carried_shelf_ids
            ),
            requested_shelves=frozenset(
                shelf_positions[shelf_id]
                for shelf_id in request_queue
                if shelf_id in shelf_positions
            ),
            queue_yield_violations=queue_yield_violations,
            delivery_lane_violations=delivery_lane_violations,
            requested_carrier_progress_violations=(
                requested_carrier_progress_violations
            ),
            step_count=int(step_count),
            inactive_steps=int(inactive_steps),
        )

    def _make_protocol_agent(
        self,
        *,
        x: int,
        y: int,
        direction: str,
        loaded: bool,
        carrying_requested: bool,
    ) -> RWAREAgentState:
        return RWAREAgentState(
            x=int(x),
            y=int(y),
            direction=str(direction),
            carrying=bool(loaded),
            carrying_requested=bool(carrying_requested),
            carrying_shelf=None,
            has_delivered=False,
        )

    def _queue_protocol_relevant_agent_cells(self) -> frozenset[tuple[int, int]]:
        return self._queue_conflict_relevant_agent_cells()

    def _queue_protocol_as_exact_state(
        self,
        state: RWAREQueueProtocolState,
    ) -> RWAREState:
        shelves: list[RWAREShelfState] = []
        request_queue: list[int] = []
        exact_agents: list[RWAREAgentState] = []
        next_shelf_id = 1
        for agent in state.agents:
            carrying_shelf: int | None = None
            if agent.loaded:
                carrying_shelf = next_shelf_id
                next_shelf_id += 1
                shelves.append(RWAREShelfState(carrying_shelf, agent.x, agent.y))
                if agent.carrying_requested:
                    request_queue.append(carrying_shelf)
            exact_agents.append(
                RWAREAgentState(
                    x=agent.x,
                    y=agent.y,
                    direction=agent.direction,
                    carrying=agent.loaded,
                    carrying_requested=agent.carrying_requested,
                    carrying_shelf=carrying_shelf,
                    has_delivered=False,
                )
            )
        return self._make_state(
            agents=tuple(exact_agents),
            shelves=tuple(shelves),
            request_queue=tuple(request_queue),
            queue_yield_violations=state.queue_yield_violations,
            delivery_lane_violations=state.delivery_lane_violations,
            requested_carrier_progress_violations=(
                state.requested_carrier_progress_violations
            ),
        )

    def _make_queue_yield_protocol_state(
        self,
        *,
        queue_yield_required: frozenset[int],
        queue_yield_clear_forward: frozenset[int] = frozenset(),
        requested_carrier_progress_required: frozenset[int] = frozenset(),
        queue_yield_violations: frozenset[int] = frozenset(),
        delivery_lane_violations: frozenset[int] = frozenset(),
        requested_carrier_progress_violations: frozenset[int] = frozenset(),
    ) -> RWAREQueueProtocolState:
        queue_cells = tuple(
            cell
            for cell in sorted(self._queue_cells)
            if cell not in self._goal_cells
        )
        blocker_cells = tuple(reversed(queue_cells)) or ((0, 0),)
        carrier_cells = (
            tuple(reversed(queue_cells[:-1]))
            if len(queue_cells) >= 2
            else blocker_cells
        )
        occupied: set[tuple[int, int]] = set()
        relevant_cells = self._queue_conflict_relevant_agent_cells()

        def _claim(candidates: tuple[tuple[int, int], ...], fallback_idx: int) -> tuple[int, int]:
            for cell in candidates:
                if cell not in occupied:
                    occupied.add(cell)
                    return cell
            fallback = self._queue_protocol_fallback_cell(
                relevant_cells | frozenset(occupied),
                fallback_idx,
            )
            occupied.add(fallback)
            return fallback

        queue_yield_required = frozenset(
            idx
            for idx in queue_yield_required
            if 0 <= int(idx) < self._num_agents
        )
        requested_carrier_progress_required = frozenset(
            idx
            for idx in requested_carrier_progress_required
            if 0 <= int(idx) < self._num_agents and idx not in queue_yield_required
        )
        queue_yield_clear_forward = frozenset(
            idx
            for idx in queue_yield_clear_forward
            if idx in queue_yield_required
        )
        agents: list[RWAREAgentState] = []
        for agent_idx in range(self._num_agents):
            if agent_idx in queue_yield_required:
                cell = _claim(blocker_cells, agent_idx)
                agents.append(
                    self._make_protocol_agent(
                        x=cell[0],
                        y=cell[1],
                        direction=(
                            "DOWN"
                            if agent_idx in queue_yield_clear_forward
                            else "UP"
                        ),
                        loaded=False,
                        carrying_requested=False,
                    )
                )
            elif agent_idx in requested_carrier_progress_required:
                fallback = self._queue_protocol_progress_fallback_cell(
                    relevant_cells | frozenset(occupied),
                    agent_idx,
                )
                direction = self._queue_protocol_open_direction(fallback, occupied)
                progress_agent = self._make_protocol_agent(
                    x=fallback[0],
                    y=fallback[1],
                    direction=direction,
                    loaded=True,
                    carrying_requested=True,
                )
                occupied.add(fallback)
                progress_target = self._requested_location(
                    progress_agent,
                    _FORWARD,
                )
                if progress_target != fallback:
                    occupied.add(progress_target)
                agents.append(progress_agent)
            elif queue_yield_required:
                cell = _claim(carrier_cells, agent_idx)
                agents.append(
                    self._make_protocol_agent(
                        x=cell[0],
                        y=cell[1],
                        direction="DOWN",
                        loaded=True,
                        carrying_requested=True,
                    )
                )
            else:
                fallback = self._queue_protocol_fallback_cell(
                    relevant_cells | frozenset(occupied),
                    agent_idx,
                )
                occupied.add(fallback)
                agents.append(
                    self._make_protocol_agent(
                        x=fallback[0],
                        y=fallback[1],
                        direction="DOWN",
                        loaded=False,
                        carrying_requested=False,
                    )
                )
        return RWAREQueueProtocolState(
            agents=tuple(agents),
            requested_shelves=frozenset(),
            requested_shelf_unknown=False,
            queue_yield_required=frozenset(queue_yield_required),
            queue_yield_clear_forward=frozenset(queue_yield_clear_forward),
            requested_carrier_progress_required=frozenset(
                requested_carrier_progress_required
            ),
            queue_yield_violations=frozenset(queue_yield_violations),
            delivery_lane_violations=frozenset(delivery_lane_violations),
            requested_carrier_progress_violations=frozenset(
                requested_carrier_progress_violations
            ),
            step_count=0,
            inactive_steps=0,
        )

    def _queue_protocol_progress_lane_cells(
        self,
    ) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
        non_goal_queue_cells = {
            cell for cell in self._queue_cells if cell not in self._goal_cells
        }
        for x in sorted({cell[0] for cell in non_goal_queue_cells}, reverse=True):
            ys = sorted(
                cell[1] for cell in non_goal_queue_cells if cell[0] == x
            )
            for high_y in sorted(ys, reverse=True):
                cells = ((x, high_y - 2), (x, high_y - 1), (x, high_y))
                if all(cell in non_goal_queue_cells for cell in cells):
                    return cells
        queue_cells = tuple(
            cell
            for cell in sorted(self._queue_cells)
            if cell not in self._goal_cells
        )
        if len(queue_cells) >= 3:
            return queue_cells[0], queue_cells[1], queue_cells[2]
        fallback = self._queue_protocol_fallback_cell(frozenset(), 0)
        return fallback, fallback, fallback

    def _direction_relative_to(
        self,
        *,
        direction: str,
        source_forward: str,
        target_forward: str,
    ) -> str:
        relative_turns = (
            _DIR_ORDER.index(direction) - _DIR_ORDER.index(source_forward)
        ) % len(_DIR_ORDER)
        return _DIR_ORDER[
            (_DIR_ORDER.index(target_forward) + relative_turns) % len(_DIR_ORDER)
        ]

    def _queue_protocol_progress_blocker_ahead(
        self,
        state: RWAREQueueProtocolState,
    ) -> tuple[int, int] | None:
        if state.queue_yield_required:
            return None
        progress_required = frozenset(state.requested_carrier_progress_required)
        if len(progress_required) != 1:
            return None
        carrier_idx = int(next(iter(progress_required)))
        if not 0 <= carrier_idx < len(state.agents):
            return None
        carrier = state.agents[carrier_idx]
        if not carrier.loaded or not carrier.carrying_requested:
            return None
        first_target = self._requested_location(carrier, _FORWARD)
        if first_target == (carrier.x, carrier.y):
            return None
        after_forward = RWAREAgentState(
            x=first_target[0],
            y=first_target[1],
            direction=carrier.direction,
            carrying=True,
            carrying_requested=True,
        )
        second_target = self._requested_location(after_forward, _FORWARD)
        if second_target == first_target:
            return None
        agent_positions = {
            (agent.x, agent.y): agent_idx
            for agent_idx, agent in enumerate(state.agents)
        }
        blocker_idx = agent_positions.get(second_target)
        if blocker_idx is None or blocker_idx == carrier_idx:
            return None
        blocker = state.agents[blocker_idx]
        if (
            blocker.loaded
            or second_target not in self._queue_cells
            or second_target in self._goal_cells
        ):
            return None
        return int(carrier_idx), int(blocker_idx)

    def _make_requested_carrier_progress_protocol_state(
        self,
        state: RWAREState,
        *,
        requested_carrier_progress_required: frozenset[int],
    ) -> RWAREQueueProtocolState:
        projected = self._make_queue_yield_protocol_state(
            queue_yield_required=frozenset(),
            requested_carrier_progress_required=(
                requested_carrier_progress_required
            ),
            queue_yield_violations=state.queue_yield_violations,
            delivery_lane_violations=state.delivery_lane_violations,
            requested_carrier_progress_violations=(
                state.requested_carrier_progress_violations
            ),
        )
        exact_like = RWAREQueueProtocolState(
            agents=tuple(
                RWAREAgentState(
                    x=agent.x,
                    y=agent.y,
                    direction=agent.direction,
                    carrying=agent.loaded,
                    carrying_requested=self._agent_carrying_requested(state, agent),
                    carrying_shelf=None,
                    has_delivered=False,
                )
                for agent in state.agents
            ),
            requested_carrier_progress_required=(
                requested_carrier_progress_required
            ),
        )
        blocker_pair = self._queue_protocol_progress_blocker_ahead(exact_like)
        if blocker_pair is None:
            return projected

        carrier_idx, blocker_idx = blocker_pair
        carrier = exact_like.agents[carrier_idx]
        blocker = exact_like.agents[blocker_idx]
        carrier_cell, middle_cell, blocker_cell = (
            self._queue_protocol_progress_lane_cells()
        )
        occupied = {carrier_cell, middle_cell, blocker_cell}
        relevant_cells = self._queue_conflict_relevant_agent_cells()
        canonical_agents: list[RWAREAgentState] = []
        for agent_idx, agent in enumerate(exact_like.agents):
            if agent_idx == carrier_idx:
                canonical_agents.append(
                    self._make_protocol_agent(
                        x=carrier_cell[0],
                        y=carrier_cell[1],
                        direction="DOWN",
                        loaded=True,
                        carrying_requested=True,
                    )
                )
                continue
            if agent_idx == blocker_idx:
                canonical_agents.append(
                    self._make_protocol_agent(
                        x=blocker_cell[0],
                        y=blocker_cell[1],
                        direction=self._direction_relative_to(
                            direction=blocker.direction,
                            source_forward=carrier.direction,
                            target_forward="DOWN",
                        ),
                        loaded=False,
                        carrying_requested=False,
                    )
                )
                continue
            fallback = self._queue_protocol_fallback_cell(
                relevant_cells | frozenset(occupied),
                agent_idx,
            )
            occupied.add(fallback)
            canonical_agents.append(
                self._make_protocol_agent(
                    x=fallback[0],
                    y=fallback[1],
                    direction="DOWN",
                    loaded=agent.loaded,
                    carrying_requested=agent.carrying_requested,
                )
            )
        return RWAREQueueProtocolState(
            agents=tuple(canonical_agents),
            requested_shelves=frozenset(),
            requested_shelf_unknown=False,
            queue_yield_required=frozenset(),
            queue_yield_clear_forward=frozenset(),
            requested_carrier_progress_required=frozenset(
                requested_carrier_progress_required
            ),
            queue_yield_violations=state.queue_yield_violations,
            delivery_lane_violations=state.delivery_lane_violations,
            requested_carrier_progress_violations=(
                state.requested_carrier_progress_violations
            ),
            step_count=0,
            inactive_steps=0,
        )

    def _queue_protocol_progress_blocker_ahead_options(
        self,
        state: RWAREQueueProtocolState,
        *,
        queue_yield_violations: frozenset[int],
        requested_carrier_progress_violations: frozenset[int],
    ) -> tuple[RWAREQueueProtocolState, ...]:
        progress_required = frozenset(state.requested_carrier_progress_required)
        if not progress_required:
            return ()
        carrier_cell, middle_cell, blocker_cell = (
            self._queue_protocol_progress_lane_cells()
        )
        relevant_cells = self._queue_conflict_relevant_agent_cells()
        options: list[RWAREQueueProtocolState] = []
        for carrier_idx in progress_required:
            if not 0 <= int(carrier_idx) < self._num_agents:
                continue
            for blocker_idx in range(self._num_agents):
                if blocker_idx == carrier_idx:
                    continue
                for blocker_direction in _DIR_ORDER:
                    occupied = {carrier_cell, middle_cell, blocker_cell}
                    agents: list[RWAREAgentState] = []
                    for agent_idx, agent in enumerate(state.agents):
                        if agent_idx == carrier_idx:
                            agents.append(
                                self._make_protocol_agent(
                                    x=carrier_cell[0],
                                    y=carrier_cell[1],
                                    direction="DOWN",
                                    loaded=True,
                                    carrying_requested=True,
                                )
                            )
                            continue
                        if agent_idx == blocker_idx:
                            agents.append(
                                self._make_protocol_agent(
                                    x=blocker_cell[0],
                                    y=blocker_cell[1],
                                    direction=blocker_direction,
                                    loaded=False,
                                    carrying_requested=False,
                                )
                            )
                            continue
                        fallback = self._queue_protocol_fallback_cell(
                            relevant_cells | frozenset(occupied),
                            agent_idx,
                        )
                        occupied.add(fallback)
                        agents.append(
                            self._make_protocol_agent(
                                x=fallback[0],
                                y=fallback[1],
                                direction="DOWN",
                                loaded=agent.loaded,
                                carrying_requested=agent.carrying_requested,
                            )
                        )
                    options.append(
                        RWAREQueueProtocolState(
                            agents=tuple(agents),
                            requested_shelves=frozenset(),
                            requested_shelf_unknown=False,
                            queue_yield_required=frozenset(),
                            queue_yield_clear_forward=frozenset(),
                            requested_carrier_progress_required=progress_required,
                            queue_yield_violations=queue_yield_violations,
                            delivery_lane_violations=frozenset(),
                            requested_carrier_progress_violations=(
                                requested_carrier_progress_violations
                            ),
                            step_count=0,
                            inactive_steps=0,
                        )
                    )
        return tuple(dict.fromkeys(options))

    def _queue_yield_required_indices(self, state: RWAREState) -> frozenset[int]:
        agent_positions = {
            (agent.x, agent.y): agent_idx
            for agent_idx, agent in enumerate(state.agents)
        }
        required: set[int] = set()
        for carrier in state.agents:
            if not self._agent_carrying_requested(state, carrier):
                continue
            target = self._requested_location(carrier, _FORWARD)
            blocker_idx = agent_positions.get(target)
            if blocker_idx is None:
                continue
            blocker = state.agents[blocker_idx]
            blocker_position = (blocker.x, blocker.y)
            if (
                not blocker.loaded
                and blocker_position in self._queue_cells
                and blocker_position not in self._goal_cells
            ):
                required.add(blocker_idx)
        return frozenset(required)

    def _queue_yield_forward_clears_indices(
        self,
        state: RWAREState,
        required: frozenset[int],
    ) -> frozenset[int]:
        clearable: set[int] = set()
        for blocker_idx in required:
            blocker_clears = True
            for other_actions in product(
                self._actions,
                repeat=max(self._num_agents - 1, 0),
            ):
                joint_action = list(other_actions)
                joint_action.insert(blocker_idx, _FORWARD)
                (
                    next_agents,
                    next_shelves,
                    _rewards,
                    delivered_shelves,
                    effective_actions,
                ) = self._apply_joint_action_before_request_replacement(
                    state,
                    tuple(joint_action),
                )
                next_request_queue = tuple(
                    shelf_id
                    for shelf_id in state.request_queue
                    if shelf_id not in set(delivered_shelves)
                )
                next_state = self._make_state(
                    agents=tuple(next_agents),
                    shelves=tuple(next_shelves),
                    request_queue=next_request_queue,
                    step_count=state.step_count + 1,
                    inactive_steps=state.inactive_steps,
                )
                queue_yield_violations, _delivery, _progress = (
                    self._protocol_violations(
                        state,
                        next_state,
                        effective_actions,
                    )
                )
                if blocker_idx in queue_yield_violations:
                    blocker_clears = False
                    break
            if blocker_clears:
                clearable.add(blocker_idx)
        return frozenset(clearable)

    def _requested_carrier_progress_required_indices(
        self,
        state: RWAREState,
    ) -> frozenset[int]:
        agent_positions = {
            (agent.x, agent.y): agent_idx
            for agent_idx, agent in enumerate(state.agents)
        }
        required: set[int] = set()
        for carrier_idx, carrier in enumerate(state.agents):
            if not self._agent_carrying_requested(state, carrier):
                continue
            if (carrier.x, carrier.y) in self._goal_cells:
                continue
            target = self._requested_location(carrier, _FORWARD)
            if target == (carrier.x, carrier.y):
                continue
            blocker_idx = agent_positions.get(target)
            if blocker_idx is None:
                required.add(carrier_idx)
        return frozenset(required)

    def _queue_protocol_agent_subsets(self) -> tuple[frozenset[int], ...]:
        subsets: list[frozenset[int]] = []
        for mask in range(1 << self._num_agents):
            subsets.append(
                frozenset(
                    agent_idx
                    for agent_idx in range(self._num_agents)
                    if mask & (1 << agent_idx)
                )
            )
        return tuple(subsets)

    def _queue_protocol_requirement_options(
        self,
        *,
        include_blocked: bool = True,
    ) -> tuple[tuple[frozenset[int], frozenset[int], frozenset[int]], ...]:
        options: list[tuple[frozenset[int], frozenset[int], frozenset[int]]] = []
        assignments_by_agent = (
            ("none", "queue_clear", "queue_blocked", "progress")
            if include_blocked
            else ("none", "queue_clear", "progress")
        )
        for assignments in product(assignments_by_agent, repeat=self._num_agents):
            queue_required = frozenset(
                agent_idx
                for agent_idx, assignment in enumerate(assignments)
                if assignment in {"queue_clear", "queue_blocked"}
            )
            queue_yield_clear_forward = frozenset(
                agent_idx
                for agent_idx, assignment in enumerate(assignments)
                if assignment == "queue_clear"
            )
            progress_required = frozenset(
                agent_idx
                for agent_idx, assignment in enumerate(assignments)
                if assignment == "progress"
            )
            options.append(
                (
                    queue_required,
                    queue_yield_clear_forward,
                    progress_required,
                )
            )
        return tuple(options)

    def _queue_protocol_fallback_cell(
        self,
        relevant_cells: frozenset[tuple[int, int]],
        agent_idx: int,
        *,
        require_highway: bool = False,
    ) -> tuple[int, int]:
        width = int(self.env.grid_size[1])
        height = int(self.env.grid_size[0])
        cells: list[tuple[int, int]] = []
        for y in range(height):
            for x in range(width - 1, -1, -1):
                if (x, y) in relevant_cells:
                    continue
                if ((x, y) in self._highway_cells) != require_highway:
                    continue
                cells.append((x, y))
        if cells:
            return cells[min(agent_idx, len(cells) - 1)]
        if require_highway:
            return self._queue_protocol_fallback_cell(
                relevant_cells,
                agent_idx,
                require_highway=False,
            )
        return self._queue_conflict_fallback_cell(relevant_cells)

    def _queue_protocol_progress_fallback_cell(
        self,
        relevant_cells: frozenset[tuple[int, int]],
        agent_idx: int,
    ) -> tuple[int, int]:
        width = int(self.env.grid_size[1])
        height = int(self.env.grid_size[0])
        candidates = [
            (x, y)
            for y in range(1, max(height - 1, 1))
            for x in range(width - 2, 0, -1)
            if (
                (x, y) not in relevant_cells
                and (x, y) in self._highway_cells
            )
        ]
        if candidates:
            return candidates[min(agent_idx, len(candidates) - 1)]
        return self._queue_protocol_fallback_cell(
            relevant_cells,
            agent_idx,
            require_highway=True,
        )

    def _queue_protocol_open_direction(
        self,
        cell: tuple[int, int],
        occupied: set[tuple[int, int]],
    ) -> str:
        for direction in ("RIGHT", "DOWN", "LEFT", "UP"):
            probe = RWAREAgentState(cell[0], cell[1], direction)
            target = self._requested_location(probe, _FORWARD)
            if target != cell and target not in occupied:
                return direction
        return "DOWN"

    def _as_exact_state(self, state: RWAREState) -> RWAREState:
        if state.shelves:
            return state

        shelves: list[RWAREShelfState] = []
        shelf_id_by_position: dict[tuple[int, int], int] = {}
        used_shelf_ids: set[int] = set()
        reserved_shelf_ids = set(state.request_queue)
        reserved_shelf_ids.update(
            agent.carrying_shelf
            for agent in state.agents
            if agent.carrying_shelf is not None
        )
        next_shelf_id = 1

        def next_available_shelf_id() -> int:
            nonlocal next_shelf_id
            while (
                next_shelf_id in used_shelf_ids
                or next_shelf_id in reserved_shelf_ids
            ):
                next_shelf_id += 1
            shelf_id = next_shelf_id
            used_shelf_ids.add(shelf_id)
            next_shelf_id += 1
            return shelf_id

        requested_positions = tuple(sorted(state.requested_shelves))
        requested_ids = list(state.request_queue)
        for index, position in enumerate(requested_positions):
            if index < len(requested_ids):
                shelf_id = requested_ids[index]
                used_shelf_ids.add(shelf_id)
            else:
                shelf_id = next_available_shelf_id()
            shelf_id_by_position[position] = shelf_id
            shelves.append(RWAREShelfState(shelf_id, position[0], position[1]))

        for position in sorted(state.standing_shelves):
            if position in shelf_id_by_position:
                continue
            shelf_id = next_available_shelf_id()
            shelf_id_by_position[position] = shelf_id
            shelves.append(RWAREShelfState(shelf_id, position[0], position[1]))

        agents: list[RWAREAgentState] = []
        for agent in state.agents:
            carrying_shelf = agent.carrying_shelf
            if carrying_shelf is None and agent.loaded:
                carrying_shelf = next_available_shelf_id()
                shelves.append(RWAREShelfState(carrying_shelf, agent.x, agent.y))
            elif carrying_shelf is not None:
                used_shelf_ids.add(carrying_shelf)
            agents.append(
                RWAREAgentState(
                    x=agent.x,
                    y=agent.y,
                    direction=agent.direction,
                    carrying=agent.loaded,
                    carrying_requested=agent.carrying_requested,
                    carrying_shelf=carrying_shelf,
                    has_delivered=agent.has_delivered,
                )
            )
            if carrying_shelf is not None:
                shelf_id_by_position[(agent.x, agent.y)] = carrying_shelf

        request_queue: list[int] = list(state.request_queue)
        for position in sorted(state.requested_shelves):
            shelf_id = shelf_id_by_position.get(position)
            if shelf_id is not None and shelf_id not in request_queue:
                request_queue.append(shelf_id)

        for agent in agents:
            if (
                agent.carrying_shelf is not None
                and agent.carrying_requested
                and agent.carrying_shelf not in request_queue
            ):
                request_queue.append(agent.carrying_shelf)

        shelf_ids = {
            shelf.shelf_id
            for shelf in shelves
        }
        env_shelf_positions = {
            shelf_id: (shelf.x, shelf.y)
            for shelf_id, shelf in self._environment_shelves_by_id().items()
        }
        for shelf_id in request_queue:
            if shelf_id in shelf_ids:
                continue
            position = env_shelf_positions.get(shelf_id, (0, 0))
            shelves.append(RWAREShelfState(shelf_id, position[0], position[1]))
            shelf_ids.add(shelf_id)

        return self._make_state(
            agents=tuple(agents),
            shelves=tuple(shelves),
            request_queue=tuple(request_queue),
            queue_yield_violations=state.queue_yield_violations,
            delivery_lane_violations=state.delivery_lane_violations,
            requested_carrier_progress_violations=(
                state.requested_carrier_progress_violations
            ),
            step_count=state.step_count,
            inactive_steps=state.inactive_steps,
        )

    def _shelves_by_id(self, state: RWAREState) -> tuple[tuple[int, RWAREShelfState], ...]:
        state = self._as_exact_state(state)
        return tuple((shelf.shelf_id, shelf) for shelf in state.shelves)

    def _shelf_positions(self, state: RWAREState) -> dict[int, tuple[int, int]]:
        state = self._as_exact_state(state)
        return {
            shelf.shelf_id: (shelf.x, shelf.y)
            for shelf in state.shelves
        }

    def _carried_shelf_ids(self, state: RWAREState) -> frozenset[int]:
        state = self._as_exact_state(state)
        return frozenset(
            agent.carrying_shelf
            for agent in state.agents
            if agent.carrying_shelf is not None
        )

    def _agent_carrying_requested(
        self,
        state: RWAREState,
        agent: RWAREAgentState,
    ) -> bool:
        if not state.shelves:
            return bool(agent.carrying_requested)
        return (
            agent.carrying_shelf is not None
            and agent.carrying_shelf in state.request_queue
        )

    def _mark_agent_delivery(
        self,
        agent: RWAREAgentState,
        delivered_shelf_id: int,
    ) -> RWAREAgentState:
        if agent.carrying_shelf != delivered_shelf_id:
            return agent
        return RWAREAgentState(
            x=agent.x,
            y=agent.y,
            direction=agent.direction,
            carrying=True,
            carrying_requested=False,
            carrying_shelf=agent.carrying_shelf,
            has_delivered=self._reward_type_name == "TWO_STAGE",
        )

    def _derive_queue_cells(self) -> list[tuple[int, int]]:
        if not hasattr(self.env, "column_height"):
            return []
        x_coords = sorted({goal[0] for goal in self._goal_cells})
        return [
            (x, y)
            for x in x_coords
            for y in range(self.env.grid_size[0])
            if y > self.env.grid_size[0] - (self.env.column_height + 3)
        ]

    def _blocking_requested_carrier_indices(
        self,
        state: RWAREState,
        agent_positions: dict[tuple[int, int], int],
    ) -> set[int]:
        return self._blocking_requested_carrier_indices_for_agents(
            state.agents,
            agent_positions,
            state=state,
        )

    def _blocking_requested_carrier_indices_for_agents(
        self,
        agents: tuple[RWAREAgentState, ...],
        agent_positions: dict[tuple[int, int], int],
        *,
        state: RWAREState | None = None,
    ) -> set[int]:
        blocking_indices: set[int] = set()
        for carrier_idx, carrier in enumerate(agents):
            carrying_requested = (
                self._agent_carrying_requested(state, carrier)
                if state is not None
                else bool(carrier.carrying_requested)
            )
            if not carrying_requested:
                continue
            target = self._requested_location(carrier, _FORWARD)
            blocker_idx = agent_positions.get(target)
            if blocker_idx is not None and blocker_idx != carrier_idx:
                blocking_indices.add(blocker_idx)
        return blocking_indices

    def _protocol_violations(
        self,
        state: RWAREState,
        next_state: RWAREState,
        joint_action: tuple[int, ...],
    ) -> tuple[frozenset[int], frozenset[int], frozenset[int]]:
        if self.benchmark != "queue_conflict":
            return frozenset(), frozenset(), frozenset()

        queue_yield_violations: set[int] = set()
        delivery_lane_violations: set[int] = set()
        requested_carrier_progress_violations: set[int] = set()
        agent_positions = {
            (agent.x, agent.y): agent_idx
            for agent_idx, agent in enumerate(state.agents)
        }

        for carrier_idx, carrier in enumerate(state.agents):
            if not self._agent_carrying_requested(state, carrier):
                continue
            target = self._requested_location(carrier, _FORWARD)
            blocker_idx = agent_positions.get(target)
            if blocker_idx is not None and blocker_idx != carrier_idx:
                blocker = state.agents[blocker_idx]
                end_blocker = next_state.agents[blocker_idx]
                if (
                    not blocker.loaded
                    and (blocker.x, blocker.y) in self._queue_cells
                    and (blocker.x, blocker.y) not in self._goal_cells
                ):
                    if (end_blocker.x, end_blocker.y) == target:
                        queue_yield_violations.add(blocker_idx)
            elif (
                target != (carrier.x, carrier.y)
                and int(joint_action[carrier_idx]) != _FORWARD
                and carrier.carrying_shelf in next_state.request_queue
            ):
                requested_carrier_progress_violations.add(carrier_idx)

            lane_cells = self._delivery_lane_cells_for_carrier(carrier)
            for agent_idx, end_agent in enumerate(next_state.agents):
                if agent_idx == carrier_idx:
                    continue
                if end_agent.loaded:
                    continue
                end_position = (end_agent.x, end_agent.y)
                if end_position in lane_cells and end_position not in self._goal_cells:
                    delivery_lane_violations.add(agent_idx)

        return (
            frozenset(queue_yield_violations),
            frozenset(delivery_lane_violations),
            frozenset(requested_carrier_progress_violations),
        )

    def _delivery_lane_cells_for_carrier(
        self,
        carrier: RWAREAgentState,
    ) -> frozenset[tuple[int, int]]:
        if not self._goal_cells:
            return frozenset()
        goal_x, goal_y = self._goal_cells[0]
        if carrier.x != goal_x:
            return frozenset()
        low_y = min(int(carrier.y), int(goal_y))
        high_y = max(int(carrier.y), int(goal_y))
        return frozenset((goal_x, y) for y in range(low_y, high_y + 1))

    def _is_pickup_zone(
        self,
        position: tuple[int, int],
        standing_requested_shelves: frozenset[tuple[int, int]],
    ) -> bool:
        return any(
            self._manhattan(position, shelf_position) <= 1
            for shelf_position in standing_requested_shelves
        )

    @staticmethod
    def _manhattan(left: tuple[int, int], right: tuple[int, int]) -> int:
        return abs(left[0] - right[0]) + abs(left[1] - right[1])

    def _requested_location(self, agent: RWAREAgentState, action: int) -> tuple[int, int]:
        if action != _FORWARD:
            return (agent.x, agent.y)
        if agent.direction == "UP":
            return (agent.x, max(0, agent.y - 1))
        if agent.direction == "DOWN":
            return (agent.x, min(self.env.grid_size[0] - 1, agent.y + 1))
        if agent.direction == "LEFT":
            return (max(0, agent.x - 1), agent.y)
        return (min(self.env.grid_size[1] - 1, agent.x + 1), agent.y)

    def _turn(self, direction: str, *, left: bool) -> str:
        index = _DIR_ORDER.index(direction)
        shift = -1 if left else 1
        return _DIR_ORDER[(index + shift) % len(_DIR_ORDER)]

    def _committed_agent_indices(
        self,
        agents: list[RWAREAgentState],
        targets: dict[int, tuple[int, int]],
    ) -> set[int]:
        starts = {
            (agent.x, agent.y): idx
            for idx, agent in enumerate(agents)
        }
        successors = {
            (agent.x, agent.y): targets[idx]
            for idx, agent in enumerate(agents)
        }
        neighbours: dict[tuple[int, int], set[tuple[int, int]]] = defaultdict(set)
        for idx, agent in enumerate(agents):
            start = (agent.x, agent.y)
            target = targets[idx]
            neighbours[start].add(target)
            neighbours[target].add(start)

        committed: set[int] = set()
        visited: set[tuple[int, int]] = set()
        for node in tuple(neighbours):
            if node in visited:
                continue
            component = self._weak_component(node, neighbours)
            visited.update(component)
            cycle = self._find_cycle(component, successors)
            if cycle:
                if len(cycle) == 2:
                    continue
                committed.update(
                    starts[position]
                    for position in cycle
                    if position in starts
                )
                continue
            committed.update(
                starts[position]
                for position in self._longest_path(component, successors, agents, targets)
                if position in starts
            )
        return committed

    def _weak_component(
        self,
        start: tuple[int, int],
        neighbours: dict[tuple[int, int], set[tuple[int, int]]],
    ) -> set[tuple[int, int]]:
        stack = [start]
        component: set[tuple[int, int]] = set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(neighbours[node] - component)
        return component

    def _find_cycle(
        self,
        component: set[tuple[int, int]],
        successors: dict[tuple[int, int], tuple[int, int]],
    ) -> tuple[tuple[int, int], ...]:
        visited: set[tuple[int, int]] = set()
        for start in component:
            path: list[tuple[int, int]] = []
            path_indices: dict[tuple[int, int], int] = {}
            node = start
            while node in component and node in successors:
                if node in path_indices:
                    return tuple(path[path_indices[node] :])
                if node in visited:
                    break
                path_indices[node] = len(path)
                path.append(node)
                node = successors[node]
            visited.update(path)
        return tuple()

    def _longest_path(
        self,
        component: set[tuple[int, int]],
        successors: dict[tuple[int, int], tuple[int, int]],
        agents: list[RWAREAgentState],
        targets: dict[int, tuple[int, int]],
    ) -> tuple[tuple[int, int], ...]:
        ordered_nodes: list[tuple[int, int]] = []
        for idx, agent in enumerate(agents):
            for node in ((agent.x, agent.y), targets[idx]):
                if node in component and node not in ordered_nodes:
                    ordered_nodes.append(node)
        for node in sorted(component):
            if node not in ordered_nodes:
                ordered_nodes.append(node)

        memo: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {}

        def path_from(node: tuple[int, int]) -> tuple[tuple[int, int], ...]:
            if node in memo:
                return memo[node]
            target = successors.get(node)
            if target is None or target not in component:
                path = (node,)
            else:
                path = (node, *path_from(target))
            memo[node] = path
            return path

        best: tuple[tuple[int, int], ...] = tuple()
        for node in ordered_nodes:
            path = path_from(node)
            if len(path) > len(best):
                best = path
        return best
