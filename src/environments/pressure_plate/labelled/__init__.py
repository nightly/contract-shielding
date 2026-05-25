from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from math import factorial
from itertools import permutations, product

from src.shield.core import (
    AbstractTransitionOutcome,
    Label,
    LocalAction,
)


def pressure_plate_contract_local_alphabet_by_agent(
    agent_ids: tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
    """Return protocol-first local alphabets for the linear pressure-plate task."""
    door_count = max(len(agent_ids) - 1, 0)
    runner_id = agent_ids[-1]
    scopes: dict[str, tuple[str, ...]] = {}
    for agent_idx, agent_id in enumerate(agent_ids):
        if agent_idx < door_count:
            door_idx = agent_idx
            scopes[str(agent_id)] = (
                f"{agent_id}_holds_door_{door_idx}_ok",
                f"door_{door_idx}_open",
                f"{runner_id}_after_door_{door_idx}",
                f"{runner_id}_crossing_door_{door_idx}",
                f"{runner_id}_in_door_{door_idx}",
                f"{runner_id}_waiting_at_door_{door_idx}",
                "goal_achieved",
                f"plate_{door_idx}_pressed",
                f"{agent_id}_on_plate",
                f"{agent_id}_on_plate_{door_idx}",
            )
        else:
            scopes[str(agent_id)] = tuple(
                label
                for door_idx in range(door_count)
                for label in (
                    f"{agent_id}_waits_for_door_{door_idx}_ok",
                    f"{agent_id}_after_door_{door_idx}",
                    f"door_{door_idx}_open",
                    f"{agent_id}_waiting_at_door_{door_idx}",
                    f"{agent_id}_crossing_door_{door_idx}",
                    f"{agent_id}_in_door_{door_idx}",
                    f"{agent_id}_can_return_before_door_{door_idx}",
                )
            ) + ("goal_achieved",)
    return scopes


def pressure_plate_contract_diagnostic_alphabet_by_agent(
    agent_ids: tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
    door_count = max(len(agent_ids) - 1, 0)
    runner_id = agent_ids[-1]
    formula_props = tuple(
        prop
        for door_idx in range(door_count)
        for prop in (
            f"{runner_id}_after_door_{door_idx}",
            f"{runner_id}_crossing_door_{door_idx}",
            f"{runner_id}_in_door_{door_idx}",
            f"{runner_id}_can_return_before_door_{door_idx}",
            f"door_{door_idx}_open",
        )
    )
    shared_phase_props = ("goal_achieved",) + formula_props + tuple(
        prop
        for door_idx in range(door_count)
        for prop in (
            f"plate_{door_idx}_pressed",
            f"agent_{door_idx}_on_plate",
            f"agent_{door_idx}_on_plate_{door_idx}",
            f"{runner_id}_waiting_at_door_{door_idx}",
        )
    )
    scopes: dict[str, tuple[str, ...]] = {}
    for agent_idx, agent_id in enumerate(agent_ids):
        protocol_labels: tuple[str, ...]
        if agent_idx < door_count:
            protocol_labels = (f"{agent_id}_holds_door_{agent_idx}_ok",)
        else:
            protocol_labels = tuple(
                f"{agent_id}_waits_for_door_{door_idx}_ok"
                for door_idx in range(door_count)
            )
        scopes[str(agent_id)] = protocol_labels + shared_phase_props
    return scopes


@dataclass(frozen=True)
class PressurePlateState:
    agent_positions: tuple[tuple[int, int], ...]
    goal_achieved: bool
    hold_door_violations: frozenset[int] = frozenset()
    wait_door_violations: frozenset[tuple[int, int]] = frozenset()
    runner_crossing_doors: frozenset[int] = frozenset()
    num_moves: int = 0


class PressurePlateSafetyModel:
    def __init__(self, env) -> None:
        self.env = env
        self.agent_ids = tuple(env.possible_agents)
        self._agent_index = {
            agent_id: index
            for index, agent_id in enumerate(self.agent_ids)
        }
        self._num_agents = len(self.agent_ids)
        self._actions = tuple(range(env.action_space(self.agent_ids[0]).n))
        self._walls = frozenset((wall.x, wall.y) for wall in env.walls)
        self._goal = (env.goal.x, env.goal.y)
        self._plate_cells = tuple((plate.x, plate.y) for plate in env.plates)
        self._door_cells = tuple(tuple(zip(door.x, door.y)) for door in env.doors)
        self._door_rows = tuple(
            min(y for _, y in door_cells)
            for door_cells in self._door_cells
        )
        self._door_waiting_cells = tuple(
            frozenset((x, y + 1) for x, y in door_cells)
            for door_cells in self._door_cells
        )
        self._max_cycles = env.max_cycles
        self._max_dist = float(env.max_dist)
        self._success_reward = float(getattr(env, "success_reward", 0.0))
        self._timestep_penalty = float(getattr(env, "timestep_penalty", 0.0))
        self._room_boundaries = tuple(int(boundary) for boundary in env.room_boundaries)
        self._runner_idx = self._num_agents - 1

    def initial_state(self, env: object) -> PressurePlateState:
        return self.abstract_state(env)

    def abstract_state(self, env: object) -> PressurePlateState:
        return PressurePlateState(
            agent_positions=tuple((agent.x, agent.y) for agent in env.agent_entities),
            goal_achieved=bool(env.goal.achieved),
            hold_door_violations=frozenset(
                int(index)
                for index in getattr(env, "_last_hold_door_violations", frozenset())
            ),
            wait_door_violations=frozenset(
                (int(agent_idx), int(door_idx))
                for agent_idx, door_idx in getattr(
                    env,
                    "_last_wait_door_violations",
                    frozenset(),
                )
            ),
            runner_crossing_doors=frozenset(
                int(door_idx)
                for door_idx in getattr(env, "_last_runner_crossing_doors", frozenset())
            ),
            num_moves=self._canonical_num_moves(int(getattr(env, "num_moves", 0))),
        )

    def local_actions(
        self,
        agent_id: str,
        state: PressurePlateState,
    ) -> tuple[LocalAction, ...]:
        _ = agent_id
        if state.goal_achieved:
            return (4,)
        return self._actions

    def safety_projection(self, state: PressurePlateState) -> PressurePlateState:
        return PressurePlateState(
            agent_positions=state.agent_positions,
            goal_achieved=state.goal_achieved,
            hold_door_violations=state.hold_door_violations,
            wait_door_violations=state.wait_door_violations,
            runner_crossing_doors=state.runner_crossing_doors,
            num_moves=0,
        )

    def label(self, state: PressurePlateState) -> Label:
        labels: set[str] = set()
        if state.goal_achieved:
            labels.add("goal_achieved")

        for agent_idx, position in enumerate(state.agent_positions):
            x, y = position
            labels.add(f"agent_{agent_idx}_at_{x}_{y}")
            if position == self._goal:
                labels.add(f"agent_{agent_idx}_at_goal")
            for door_idx, door_row in enumerate(self._door_rows):
                if y > door_row:
                    labels.add(f"agent_{agent_idx}_before_door_{door_idx}")
                elif y < door_row:
                    labels.add(f"agent_{agent_idx}_after_door_{door_idx}")
                if position in self._door_waiting_cells[door_idx]:
                    labels.add(f"agent_{agent_idx}_waiting_at_door_{door_idx}")
                if (
                    agent_idx == self._runner_idx
                    and position in self._door_cells[door_idx]
                ):
                    labels.add(f"agent_{agent_idx}_in_door_{door_idx}")
                if (
                    agent_idx == self._runner_idx
                    and door_idx in state.runner_crossing_doors
                ):
                    labels.add(f"agent_{agent_idx}_crossing_door_{door_idx}")
            for plate_idx, plate_cell in enumerate(self._plate_cells):
                if position == plate_cell:
                    labels.add(f"agent_{agent_idx}_on_plate_{plate_idx}")
            if agent_idx < len(self._plate_cells):
                if agent_idx not in state.hold_door_violations:
                    labels.add(f"agent_{agent_idx}_holds_door_{agent_idx}_ok")
            if agent_idx == self._runner_idx:
                for door_idx in range(len(self._door_cells)):
                    if (agent_idx, door_idx) not in state.wait_door_violations:
                        labels.add(f"agent_{agent_idx}_waits_for_door_{door_idx}_ok")
                    if self._runner_can_return_before_door(state, door_idx):
                        labels.add(
                            f"agent_{agent_idx}_can_return_before_door_{door_idx}"
                        )

        for plate_idx, plate_cell in enumerate(self._plate_cells):
            if (
                plate_idx < len(state.agent_positions)
                and state.agent_positions[plate_idx] == plate_cell
            ):
                labels.add(f"plate_{plate_idx}_pressed")
                labels.add(f"agent_{plate_idx}_on_plate")
                labels.add(f"door_{plate_idx}_open")

        return frozenset(labels)

    def possible_labels(self) -> tuple[str, ...]:
        labels: set[str] = {"goal_achieved"}
        for agent_idx in range(self._num_agents):
            for y in range(self.env.height):
                for x in range(self.env.width):
                    labels.add(f"agent_{agent_idx}_at_{x}_{y}")
            labels.add(f"agent_{agent_idx}_at_goal")
            labels.add(f"agent_{agent_idx}_on_plate")
            for door_idx in range(len(self._door_cells)):
                labels.add(f"agent_{agent_idx}_after_door_{door_idx}")
                labels.add(f"agent_{agent_idx}_before_door_{door_idx}")
                labels.add(f"agent_{agent_idx}_waiting_at_door_{door_idx}")
            for plate_idx in range(len(self._plate_cells)):
                labels.add(f"agent_{agent_idx}_on_plate_{plate_idx}")
            if agent_idx < len(self._plate_cells):
                labels.add(f"agent_{agent_idx}_holds_door_{agent_idx}_ok")
            if agent_idx == self._runner_idx:
                for door_idx in range(len(self._door_cells)):
                    labels.add(f"agent_{agent_idx}_waits_for_door_{door_idx}_ok")
                    labels.add(f"agent_{agent_idx}_crossing_door_{door_idx}")
                    labels.add(f"agent_{agent_idx}_in_door_{door_idx}")
                    labels.add(f"agent_{agent_idx}_can_return_before_door_{door_idx}")
        for plate_idx in range(len(self._plate_cells)):
            labels.add(f"plate_{plate_idx}_pressed")
            labels.add(f"door_{plate_idx}_open")
        return tuple(sorted(labels))

    def contract_local_alphabet_by_agent(self) -> dict[str, tuple[str, ...]]:
        return pressure_plate_contract_local_alphabet_by_agent(self.agent_ids)

    def contract_diagnostic_alphabet_by_agent(self) -> dict[str, tuple[str, ...]]:
        return pressure_plate_contract_diagnostic_alphabet_by_agent(self.agent_ids)

    def local_shield_formulas_by_agent(
        self,
        *,
        formula_name: str | None,
        global_formula: str,
    ) -> dict[str, str] | None:
        _ = global_formula
        if formula_name not in {
            "runner_requires_open_doors",
            "runner_crosses_open_doors_strict",
            "runner_requires_return_routes",
        }:
            return None

        door_count = max(len(self.agent_ids) - 1, 0)
        if formula_name == "runner_requires_return_routes":
            runner_id = self.agent_ids[-1]
            runner_stays_out = " & ".join(
                f"!{runner_id}_in_door_{door_idx} & !{runner_id}_after_door_{door_idx}"
                for door_idx in range(door_count)
            )
            return {
                str(agent_id): (
                    f"G({runner_stays_out})"
                    if agent_id == runner_id and runner_stays_out
                    else "t"
                )
                for agent_id in self.agent_ids
            }

        formulas: dict[str, str] = {}
        for agent_idx, agent_id in enumerate(self.agent_ids):
            if agent_idx < door_count:
                formulas[str(agent_id)] = f"G({agent_id}_holds_door_{agent_idx}_ok)"
            else:
                waits = " & ".join(
                    f"{agent_id}_waits_for_door_{door_idx}_ok"
                    for door_idx in range(door_count)
                )
                formulas[str(agent_id)] = f"G({waits})" if waits else "t"
        return formulas

    def contract_seed_formulas_by_agent(
        self,
        *,
        global_formula: str,
    ) -> dict[str, str]:
        door_count = max(len(self.agent_ids) - 1, 0)
        runner_id = self.agent_ids[-1]
        runner_open_doors = self._runner_open_doors_formula(runner_id, door_count)
        runner_strict_crossing = self._runner_strict_crossing_formula(
            runner_id,
            door_count,
        )
        runner_return_routes = self._runner_return_routes_formula(
            runner_id,
            door_count,
        )
        compact_global = self._compact_formula(global_formula)
        compact_open_doors = self._compact_formula(runner_open_doors)
        compact_strict_crossing = self._compact_formula(runner_strict_crossing)
        compact_return_routes = self._compact_formula(runner_return_routes)
        if compact_global not in {
            compact_open_doors,
            compact_strict_crossing,
            compact_return_routes,
        }:
            return {}

        formulas: dict[str, str] = {}
        if compact_global == compact_open_doors:
            for door_idx in range(door_count):
                if door_idx >= len(self.agent_ids):
                    continue
                holder_id = self.agent_ids[door_idx]
                formulas[str(holder_id)] = (
                    "G(("
                    f"{runner_id}_waiting_at_door_{door_idx} -> door_{door_idx}_open"
                    f") & {holder_id}_holds_door_{door_idx}_ok)"
                )
            formulas[str(runner_id)] = runner_open_doors
        elif compact_global == compact_strict_crossing:
            for door_idx in range(door_count):
                if door_idx >= len(self.agent_ids):
                    continue
                holder_id = self.agent_ids[door_idx]
                formulas[str(holder_id)] = (
                    "G(("
                    f"{runner_id}_waiting_at_door_{door_idx} | "
                    f"{runner_id}_in_door_{door_idx}"
                    f") -> X(door_{door_idx}_open))"
                )
            formulas[str(runner_id)] = runner_strict_crossing
        else:
            for door_idx in range(door_count):
                if door_idx >= len(self.agent_ids):
                    continue
                holder_id = self.agent_ids[door_idx]
                formulas[str(holder_id)] = (
                    "G((!goal_achieved & ("
                    f"{runner_id}_waiting_at_door_{door_idx} | "
                    f"{runner_id}_in_door_{door_idx} | "
                    f"{runner_id}_after_door_{door_idx}"
                    f")) -> X(door_{door_idx}_open))"
                )
            formulas[str(runner_id)] = runner_return_routes
        return formulas

    @staticmethod
    def _compact_formula(formula: str) -> str:
        return "".join(str(formula).split())

    @staticmethod
    def _runner_open_doors_formula(
        runner_id: str,
        door_count: int,
    ) -> str:
        conjuncts = (
            f"{runner_id}_after_door_{door_idx} -> door_{door_idx}_open"
            for door_idx in range(door_count)
        )
        return "G((" + ") & (".join(conjuncts) + "))" if door_count else "t"

    @staticmethod
    def _runner_strict_crossing_formula(
        runner_id: str,
        door_count: int,
    ) -> str:
        conjuncts = (
            f"({runner_id}_crossing_door_{door_idx} | "
            f"{runner_id}_in_door_{door_idx}) -> door_{door_idx}_open"
            for door_idx in range(door_count)
        )
        return "G((" + ") & (".join(conjuncts) + "))" if door_count else "t"

    @staticmethod
    def _runner_return_routes_formula(
        runner_id: str,
        door_count: int,
    ) -> str:
        conjuncts = (
            f"(!goal_achieved & ({runner_id}_in_door_{door_idx} | "
            f"{runner_id}_after_door_{door_idx})) -> "
            f"{runner_id}_can_return_before_door_{door_idx}"
            for door_idx in range(door_count)
        )
        return "G((" + ") & (".join(conjuncts) + "))" if door_count else "t"

    def joint_actions(
        self,
        state: PressurePlateState,
    ) -> tuple[tuple[LocalAction, ...], ...]:
        return tuple(product(*(self.local_actions(agent_id, state) for agent_id in self.agent_ids)))

    def successors_for_joint_action(
        self,
        state: PressurePlateState,
        joint_action: tuple[LocalAction, ...],
    ) -> frozenset[PressurePlateState]:
        normalized_action = tuple(int(action) for action in joint_action)
        if len(normalized_action) != self._num_agents:
            raise ValueError(
                f"Expected {self._num_agents} local actions, got {len(normalized_action)}."
            )
        if state.goal_achieved:
            return frozenset({state})
        return frozenset(
            self._next_state_for_move_order(
                state,
                normalized_action,
                move_order,
                advance_time=False,
            )
            for move_order in permutations(range(self._num_agents))
        )

    def transition_outcomes_for_joint_action(
        self,
        state: PressurePlateState,
        joint_action: tuple[LocalAction, ...],
    ) -> tuple[AbstractTransitionOutcome[PressurePlateState], ...]:
        normalized_action = tuple(int(action) for action in joint_action)
        if len(normalized_action) != self._num_agents:
            raise ValueError(
                f"Expected {self._num_agents} local actions, got {len(normalized_action)}."
            )
        if state.goal_achieved:
            return (
                AbstractTransitionOutcome(
                    next_state=state,
                    probability=1.0,
                    rewards=self._rewards_for_state(state, step_reward=False),
                    terminations={agent_id: True for agent_id in self.agent_ids},
                    truncations={agent_id: False for agent_id in self.agent_ids},
                    label=self.label(state),
                ),
            )

        counts: defaultdict[PressurePlateState, int] = defaultdict(int)
        for move_order in permutations(range(self._num_agents)):
            counts[
                self._next_state_for_move_order(
                    state,
                    normalized_action,
                    move_order,
                    advance_time=True,
                )
            ] += 1

        total_orders = float(factorial(self._num_agents))
        outcomes: list[AbstractTransitionOutcome[PressurePlateState]] = []
        for next_state, count in counts.items():
            env_truncation = (
                self._max_cycles is not None and next_state.num_moves >= self._max_cycles
            )
            outcomes.append(
                AbstractTransitionOutcome(
                    next_state=next_state,
                    probability=float(count) / total_orders,
                    rewards=self._rewards_for_state(
                        next_state,
                        goal_transition=next_state.goal_achieved,
                    ),
                    terminations={
                        agent_id: next_state.goal_achieved for agent_id in self.agent_ids
                    },
                    truncations={agent_id: env_truncation for agent_id in self.agent_ids},
                    label=self.label(next_state),
                )
            )
        return tuple(outcomes)

    def successors_for_local_action(
        self,
        state: PressurePlateState,
        agent_id: str,
        action: LocalAction,
    ) -> frozenset[PressurePlateState]:
        agent_idx = self._agent_index[agent_id]
        if state.goal_achieved:
            return frozenset({state})

        successors: set[PressurePlateState] = set()
        for other_actions in product(self._actions, repeat=self._num_agents - 1):
            joint_action = list(other_actions)
            joint_action.insert(agent_idx, int(action))
            successors.update(
                self.successors_for_joint_action(
                    state,
                    tuple(joint_action),
                )
            )
        return frozenset(successors)

    def _next_state_for_move_order(
        self,
        state: PressurePlateState,
        joint_action: tuple[int, ...],
        move_order: tuple[int, ...],
        *,
        advance_time: bool,
    ) -> PressurePlateState:
        closed_doors = {
            cell
            for idx, cells in enumerate(self._door_cells)
            if state.agent_positions[idx] != self._plate_cells[idx]
            for cell in cells
        }

        positions = list(state.agent_positions)
        start_positions = tuple(state.agent_positions)
        start_open_doors = tuple(
            self._door_open_for_positions(start_positions, door_idx)
            for door_idx in range(len(self._door_cells))
        )
        for agent_idx in move_order:
            action = int(joint_action[agent_idx])
            proposed = self._propose_move(positions[agent_idx], action)
            if proposed == positions[agent_idx]:
                continue
            if self._detect_collision(proposed, agent_idx, positions, closed_doors):
                continue
            positions[agent_idx] = proposed

        goal_achieved = any(position == self._goal for position in positions)
        (
            hold_door_violations,
            wait_door_violations,
        ) = self._protocol_violations(
            start_positions=start_positions,
            end_positions=tuple(positions),
            start_open_doors=start_open_doors,
            joint_action=joint_action,
            goal_achieved=goal_achieved,
        )
        return PressurePlateState(
            agent_positions=tuple(positions),
            goal_achieved=goal_achieved,
            hold_door_violations=hold_door_violations,
            wait_door_violations=wait_door_violations,
            runner_crossing_doors=self._runner_crossing_doors(
                start_positions,
                tuple(positions),
            ),
            num_moves=(
                self._canonical_num_moves(state.num_moves + 1)
                if advance_time
                else state.num_moves
            ),
        )

    def _protocol_violations(
        self,
        *,
        start_positions: tuple[tuple[int, int], ...],
        end_positions: tuple[tuple[int, int], ...],
        start_open_doors: tuple[bool, ...],
        joint_action: tuple[int, ...],
        goal_achieved: bool,
    ) -> tuple[frozenset[int], frozenset[tuple[int, int]]]:
        hold_violations: set[int] = set()
        wait_violations: set[tuple[int, int]] = set()
        for door_idx, plate_cell in enumerate(self._plate_cells):
            if door_idx >= len(start_positions):
                continue
            if start_positions[door_idx] != plate_cell:
                continue
            if end_positions[door_idx] == plate_cell:
                continue
            if not goal_achieved:
                hold_violations.add(door_idx)

        if self._runner_idx < len(joint_action):
            runner_start = start_positions[self._runner_idx]
            runner_proposed = self._propose_move(
                runner_start,
                int(joint_action[self._runner_idx]),
            )
            for door_idx, door_cells in enumerate(self._door_cells):
                if bool(start_open_doors[door_idx]):
                    continue
                if runner_proposed in door_cells:
                    wait_violations.add((self._runner_idx, door_idx))

        return frozenset(hold_violations), frozenset(wait_violations)

    def _runner_crossing_doors(
        self,
        start_positions: tuple[tuple[int, int], ...],
        end_positions: tuple[tuple[int, int], ...],
    ) -> frozenset[int]:
        if (
            self._runner_idx >= len(start_positions)
            or self._runner_idx >= len(end_positions)
        ):
            return frozenset()

        start = start_positions[self._runner_idx]
        end = end_positions[self._runner_idx]
        if start == end:
            return frozenset()

        crossing_doors: set[int] = set()
        for door_idx, door_cells in enumerate(self._door_cells):
            door_row = self._door_rows[door_idx]
            moved_into_door = start not in door_cells and end in door_cells
            moved_to_far_side = start in door_cells and end[1] < door_row
            crossed_boundary = start[1] > door_row and end[1] < door_row
            if moved_into_door or moved_to_far_side or crossed_boundary:
                crossing_doors.add(door_idx)
        return frozenset(crossing_doors)

    def _runner_can_return_before_door(
        self,
        state: PressurePlateState,
        door_idx: int,
    ) -> bool:
        if self._runner_idx >= len(state.agent_positions):
            return False
        if door_idx >= len(self._door_rows):
            return False

        start = state.agent_positions[self._runner_idx]
        door_row = self._door_rows[door_idx]
        if start[1] > door_row:
            return True

        closed_doors = {
            cell
            for idx, cells in enumerate(self._door_cells)
            if not self._door_open_for_positions(state.agent_positions, idx)
            for cell in cells
        }
        queue: deque[tuple[int, int]] = deque([start])
        visited: set[tuple[int, int]] = {start}
        while queue:
            x, y = queue.popleft()
            if y > door_row:
                return True
            for neighbor in (
                (x, y - 1),
                (x, y + 1),
                (x - 1, y),
                (x + 1, y),
            ):
                nx, ny = neighbor
                if nx < 0 or ny < 0 or nx >= self.env.width or ny >= self.env.height:
                    continue
                if neighbor in visited:
                    continue
                if neighbor in self._walls or neighbor in closed_doors:
                    continue
                visited.add(neighbor)
                queue.append(neighbor)
        return False

    def _door_open_for_positions(
        self,
        positions: tuple[tuple[int, int], ...],
        door_idx: int,
    ) -> bool:
        return door_idx < len(positions) and positions[door_idx] == self._plate_cells[door_idx]

    def _rewards_for_state(
        self,
        state: PressurePlateState,
        *,
        goal_transition: bool = False,
        step_reward: bool = True,
    ) -> dict[str, float]:
        rewards: dict[str, float] = {}
        for agent_idx, agent_id in enumerate(self.agent_ids):
            agent_loc = state.agent_positions[agent_idx]
            target_loc = (
                self._goal
                if agent_idx == len(state.agent_positions) - 1
                else self._plate_cells[agent_idx]
            )
            curr_room = self._curr_room_reward(agent_loc[1])
            if agent_idx == curr_room:
                reward = -(
                    abs(target_loc[0] - agent_loc[0])
                    + abs(target_loc[1] - agent_loc[1])
                ) / self._max_dist
            else:
                reward = -len(self._room_boundaries) + 1 + curr_room
            if step_reward:
                reward -= self._timestep_penalty
            if goal_transition:
                reward += self._success_reward
            rewards[agent_id] = float(reward)
        return rewards

    def _curr_room_reward(self, agent_y: int) -> int:
        for index, room_level in enumerate(self._room_boundaries):
            if agent_y > room_level:
                return index
        return len(self._room_boundaries) - 1

    def _canonical_num_moves(self, num_moves: int) -> int:
        if self._max_cycles is None:
            return 0
        return min(int(num_moves), int(self._max_cycles))

    def _propose_move(self, position: tuple[int, int], action: int) -> tuple[int, int]:
        x, y = position
        if action == 0:
            return (x, y - 1)
        if action == 1:
            return (x, y + 1)
        if action == 2:
            return (x - 1, y)
        if action == 3:
            return (x + 1, y)
        return position

    def _detect_collision(
        self,
        proposed: tuple[int, int],
        moving_agent_idx: int,
        positions: list[tuple[int, int]],
        closed_doors: set[tuple[int, int]],
    ) -> bool:
        x, y = proposed
        if x < 0 or y < 0 or x >= self.env.width or y >= self.env.height:
            return True
        if proposed in self._walls or proposed in closed_doors:
            return True
        return any(
            proposed == position
            for idx, position in enumerate(positions)
            if idx != moving_agent_idx
        )
