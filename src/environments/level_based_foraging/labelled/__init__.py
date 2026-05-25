from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import re

from ..impl.env import Action
from src.shield.core import AbstractTransitionOutcome, Label, LocalAction


LBF_CONTRACT_LABEL_FAMILIES = (
    "coop_load_ok",
    "failed_load",
    "load_attempted",
    "successful_load",
    "adjacent_food",
    "needs_partner",
    "partner_adjacent_same_food",
    "can_load_solo",
)


def level_based_foraging_contract_local_alphabet_by_agent(
    agent_ids: tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
    shared_labels = tuple(
        f"{agent_id}_failed_load"
        for agent_id in agent_ids
    )
    return {
        str(agent_id): (
            (f"{agent_id}_coop_load_ok",)
            + shared_labels
        )
        for agent_id in agent_ids
    }


def level_based_foraging_contract_diagnostic_alphabet_by_agent(
    agent_ids: tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
    shared_labels = tuple(
        f"{agent_id}_failed_load"
        for agent_id in agent_ids
    ) + ("food_available", "all_food_collected")
    return {
        str(agent_id): (
            (f"{agent_id}_coop_load_ok",)
            + shared_labels
            + ("joint_load_ready",)
            + tuple(
                f"{agent_id}_{family}"
                for family in LBF_CONTRACT_LABEL_FAMILIES
                if family not in {"coop_load_ok", "failed_load"}
            )
        )
        for agent_id in agent_ids
    }


@dataclass(frozen=True)
class LevelBasedForagingState:
    agent_positions: tuple[tuple[int, int], ...]
    agent_levels: tuple[int, ...]
    foods: tuple[tuple[int, int, int], ...]
    failed_loads: frozenset[int] = frozenset()
    successful_loads: frozenset[int] = frozenset()
    load_attempts: frozenset[int] = frozenset()
    coop_load_violations: frozenset[int] = frozenset()
    step_count: int = 0


class LevelBasedForagingSafetyModel:
    """Exact LBF abstraction for the small cooperative one-food benchmark."""

    def __init__(self, env) -> None:
        self.env = env.unwrapped if hasattr(env, "unwrapped") else env
        if int(getattr(self.env, "max_num_food", 0)) != 1:
            raise ValueError(
                "LevelBasedForagingSafetyModel currently supports max_num_food=1."
            )
        self.agent_ids = tuple(str(agent_id) for agent_id in self.env.possible_agents)
        self._agent_index = {
            agent_id: index
            for index, agent_id in enumerate(self.agent_ids)
        }
        self._num_agents = len(self.agent_ids)
        self._actions = tuple(range(self.env.action_space(self.agent_ids[0]).n))
        self.rows = int(self.env.rows)
        self.cols = int(self.env.cols)
        self.max_episode_steps = self.env.max_episode_steps
        self.penalty = float(self.env.penalty)
        self.normalize_reward = bool(getattr(self.env, "_normalize_reward", True))
        spawned = float(getattr(self.env, "_food_spawned", 0.0) or 0.0)
        if spawned <= 0.0:
            spawned = float(sum(food[2] for food in self._foods_from_env(self.env)))
        self._food_spawned = spawned
        self._max_food_level = int(self.env._max_food_level_for_space())

    def initial_state(self, env: object) -> LevelBasedForagingState:
        return self.abstract_state(env)

    def abstract_state(self, env: object) -> LevelBasedForagingState:
        base_env = env.unwrapped if hasattr(env, "unwrapped") else env
        return LevelBasedForagingState(
            agent_positions=tuple(
                tuple(int(value) for value in player.position)
                for player in base_env.players
            ),
            agent_levels=tuple(int(player.level) for player in base_env.players),
            foods=self._foods_from_env(base_env),
            failed_loads=frozenset(
                self._agent_index[str(agent_id)]
                for agent_id in getattr(base_env, "_last_failed_load_agents", ())
            ),
            successful_loads=frozenset(
                self._agent_index[str(agent_id)]
                for agent_id in getattr(base_env, "_last_successful_load_agents", ())
            ),
            load_attempts=frozenset(
                self._agent_index[str(agent_id)]
                for agent_id in getattr(base_env, "_last_load_attempted_agents", ())
            ),
            coop_load_violations=frozenset(
                self._agent_index[str(agent_id)]
                for agent_id in getattr(
                    base_env,
                    "_last_coop_load_violation_agents",
                    (),
                )
            ),
            step_count=int(getattr(base_env, "current_step", 0) or 0),
        )

    def safety_projection(
        self,
        state: LevelBasedForagingState,
    ) -> LevelBasedForagingState:
        return LevelBasedForagingState(
            agent_positions=state.agent_positions,
            agent_levels=state.agent_levels,
            foods=state.foods,
            failed_loads=state.failed_loads,
            successful_loads=state.successful_loads,
            load_attempts=state.load_attempts,
            coop_load_violations=state.coop_load_violations,
            step_count=0,
        )

    def local_actions(
        self,
        agent_id: str,
        state: LevelBasedForagingState,
    ) -> tuple[LocalAction, ...]:
        _ = agent_id
        if not state.foods:
            return (int(Action.NONE),)
        return self._actions

    def label(self, state: LevelBasedForagingState) -> Label:
        labels: set[str] = set()
        if state.foods:
            labels.add("food_available")
        else:
            labels.add("all_food_collected")
        if self._coop_load_ready_agents(state):
            labels.add("joint_load_ready")

        for food_row, food_col, food_level in state.foods:
            labels.add(f"food_at_{food_row}_{food_col}")
            labels.add(f"food_level_{food_level}")

        for agent_idx, position in enumerate(state.agent_positions):
            agent_id = self.agent_ids[agent_idx]
            row, col = position
            labels.add(f"{agent_id}_at_{row}_{col}")
            if agent_idx in state.failed_loads:
                labels.add(f"{agent_id}_failed_load")
            if agent_idx in state.successful_loads:
                labels.add(f"{agent_id}_successful_load")
            if agent_idx in state.load_attempts:
                labels.add(f"{agent_id}_load_attempted")
            if agent_idx not in state.coop_load_violations:
                labels.add(f"{agent_id}_coop_load_ok")

            adjacent_food = self._adjacent_food(state, position)
            if adjacent_food is None:
                continue

            _, _, food_level = adjacent_food
            labels.add(f"{agent_id}_adjacent_food")
            if state.agent_levels[agent_idx] >= food_level:
                labels.add(f"{agent_id}_can_load_solo")
            else:
                labels.add(f"{agent_id}_needs_partner")
            if any(
                other_idx != agent_idx
                and self._adjacent_food(state, other_position) == adjacent_food
                for other_idx, other_position in enumerate(state.agent_positions)
            ):
                labels.add(f"{agent_id}_partner_adjacent_same_food")

        return frozenset(labels)

    def possible_labels(self) -> tuple[str, ...]:
        labels: set[str] = {
            "food_available",
            "all_food_collected",
            "joint_load_ready",
        }
        for row in range(self.rows):
            for col in range(self.cols):
                labels.add(f"food_at_{row}_{col}")
        for food_level in range(1, self._max_food_level + 1):
            labels.add(f"food_level_{food_level}")
        for agent_id in self.agent_ids:
            for row in range(self.rows):
                for col in range(self.cols):
                    labels.add(f"{agent_id}_at_{row}_{col}")
            for family in LBF_CONTRACT_LABEL_FAMILIES:
                labels.add(f"{agent_id}_{family}")
        return tuple(sorted(labels))

    def contract_local_alphabet_by_agent(self) -> dict[str, tuple[str, ...]]:
        return level_based_foraging_contract_local_alphabet_by_agent(self.agent_ids)

    def contract_diagnostic_alphabet_by_agent(self) -> dict[str, tuple[str, ...]]:
        return level_based_foraging_contract_diagnostic_alphabet_by_agent(self.agent_ids)

    def contract_seed_formulas_by_agent(
        self,
        *,
        global_formula: str,
    ) -> dict[str, str]:
        failed_load_props = {
            f"{agent_id}_failed_load"
            for agent_id in self.agent_ids
        }
        if self._formula_atomic_props(global_formula) != failed_load_props:
            return {}

        guard = " & ".join(
            f"!{agent_id}_failed_load"
            for agent_id in self.agent_ids
        )
        return {
            agent_id: f"G(({guard}) & {agent_id}_coop_load_ok)"
            for agent_id in self.agent_ids
        }

    def joint_actions(
        self,
        state: LevelBasedForagingState,
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
        state: LevelBasedForagingState,
        joint_action: tuple[LocalAction, ...],
    ) -> frozenset[LevelBasedForagingState]:
        return frozenset(
            {self._next_state_for_joint_action(state, tuple(map(int, joint_action)))}
        )

    def transition_outcomes_for_joint_action(
        self,
        state: LevelBasedForagingState,
        joint_action: tuple[LocalAction, ...],
    ) -> tuple[AbstractTransitionOutcome[LevelBasedForagingState], ...]:
        normalized_action = tuple(int(action) for action in joint_action)
        if len(normalized_action) != self._num_agents:
            raise ValueError(
                f"Expected {self._num_agents} local actions, got "
                f"{len(normalized_action)}."
            )
        next_state, rewards = self._transition(state, normalized_action)
        env_termination = not next_state.foods
        env_truncation = bool(
            self.max_episode_steps is not None
            and next_state.step_count >= self.max_episode_steps
            and next_state.foods
        )
        return (
            AbstractTransitionOutcome(
                next_state=next_state,
                probability=1.0,
                rewards={
                    agent_id: float(rewards[agent_idx])
                    for agent_idx, agent_id in enumerate(self.agent_ids)
                },
                terminations={
                    agent_id: env_termination
                    for agent_id in self.agent_ids
                },
                truncations={
                    agent_id: env_truncation
                    for agent_id in self.agent_ids
                },
                label=self.label(next_state),
            ),
        )

    def successors_for_local_action(
        self,
        state: LevelBasedForagingState,
        agent_id: str,
        action: LocalAction,
    ) -> frozenset[LevelBasedForagingState]:
        agent_idx = self._agent_index[agent_id]
        successors: set[LevelBasedForagingState] = set()
        action_sets: list[tuple[int, ...]] = []
        for idx, other_agent_id in enumerate(self.agent_ids):
            if idx == agent_idx:
                action_sets.append((int(action),))
            else:
                action_sets.append(self.local_actions(other_agent_id, state))
        for joint_action in product(*action_sets):
            successors.update(self.successors_for_joint_action(state, joint_action))
        return frozenset(successors)

    def _next_state_for_joint_action(
        self,
        state: LevelBasedForagingState,
        joint_action: tuple[int, ...],
    ) -> LevelBasedForagingState:
        next_state, _ = self._transition(state, joint_action)
        return next_state

    def _transition(
        self,
        state: LevelBasedForagingState,
        joint_action: tuple[int, ...],
    ) -> tuple[LevelBasedForagingState, tuple[float, ...]]:
        if len(joint_action) != self._num_agents:
            raise ValueError(
                f"Expected {self._num_agents} local actions, got "
                f"{len(joint_action)}."
            )
        if not state.foods:
            terminal_state = LevelBasedForagingState(
                agent_positions=state.agent_positions,
                agent_levels=state.agent_levels,
                foods=state.foods,
                load_attempts=frozenset(),
                coop_load_violations=frozenset(),
                step_count=state.step_count,
            )
            return terminal_state, tuple(0.0 for _ in self.agent_ids)

        effective_actions = tuple(
            action
            if self._is_valid_action(state, agent_idx, action)
            else int(Action.NONE)
            for agent_idx, action in enumerate(joint_action)
        )
        next_positions = list(state.agent_positions)
        collisions: dict[tuple[int, int], list[int]] = {}
        loading_agents: set[int] = set()
        coop_ready_agents = self._coop_load_ready_agents(state)
        load_attempts = frozenset(
            agent_idx
            for agent_idx, action in enumerate(effective_actions)
            if action == int(Action.LOAD)
        )
        coop_load_violations = frozenset(
            agent_idx
            for agent_idx in coop_ready_agents
            if agent_idx not in load_attempts
        )

        for agent_idx, action in enumerate(effective_actions):
            row, col = state.agent_positions[agent_idx]
            target = (row, col)
            if action == int(Action.NORTH):
                target = (row - 1, col)
            elif action == int(Action.SOUTH):
                target = (row + 1, col)
            elif action == int(Action.WEST):
                target = (row, col - 1)
            elif action == int(Action.EAST):
                target = (row, col + 1)
            elif action == int(Action.LOAD):
                loading_agents.add(agent_idx)
            collisions.setdefault(target, []).append(agent_idx)

        for target, agent_indices in collisions.items():
            if len(agent_indices) == 1:
                next_positions[agent_indices[0]] = target

        foods = list(state.foods)
        rewards = [0.0 for _ in self.agent_ids]
        failed_loads: set[int] = set()
        successful_loads: set[int] = set()

        for food in tuple(foods):
            food_row, food_col, food_level = food
            adjacent_loaders = [
                agent_idx
                for agent_idx in loading_agents
                if self._adjacent(next_positions[agent_idx], (food_row, food_col))
            ]
            if not adjacent_loaders:
                continue
            coalition_level = sum(
                state.agent_levels[agent_idx]
                for agent_idx in adjacent_loaders
            )
            if coalition_level < food_level:
                for agent_idx in adjacent_loaders:
                    rewards[agent_idx] -= self.penalty
                failed_loads.update(adjacent_loaders)
                continue
            for agent_idx in adjacent_loaders:
                rewards[agent_idx] = float(
                    state.agent_levels[agent_idx] * food_level
                )
                if self.normalize_reward and self._food_spawned > 0:
                    rewards[agent_idx] /= float(
                        coalition_level * self._food_spawned
                    )
            successful_loads.update(adjacent_loaders)
            foods.remove(food)

        next_state = LevelBasedForagingState(
            agent_positions=tuple(next_positions),
            agent_levels=state.agent_levels,
            foods=tuple(sorted(foods)),
            failed_loads=frozenset(failed_loads),
            successful_loads=frozenset(successful_loads),
            load_attempts=load_attempts,
            coop_load_violations=coop_load_violations,
            step_count=int(state.step_count) + 1,
        )
        return next_state, tuple(rewards)

    def _is_valid_action(
        self,
        state: LevelBasedForagingState,
        agent_idx: int,
        action: int,
    ) -> bool:
        row, col = state.agent_positions[agent_idx]
        if action == int(Action.NONE):
            return True
        if action == int(Action.NORTH):
            return row > 0 and not self._food_at(state, row - 1, col)
        if action == int(Action.SOUTH):
            return row < self.rows - 1 and not self._food_at(state, row + 1, col)
        if action == int(Action.WEST):
            return col > 0 and not self._food_at(state, row, col - 1)
        if action == int(Action.EAST):
            return col < self.cols - 1 and not self._food_at(state, row, col + 1)
        if action == int(Action.LOAD):
            return self._adjacent_food(state, (row, col)) is not None
        return False

    def _adjacent_food(
        self,
        state: LevelBasedForagingState,
        position: tuple[int, int],
    ) -> tuple[int, int, int] | None:
        for food in state.foods:
            if self._adjacent(position, (food[0], food[1])):
                return food
        return None

    def _food_at(
        self,
        state: LevelBasedForagingState,
        row: int,
        col: int,
    ) -> bool:
        return any(
            food_row == row and food_col == col
            for food_row, food_col, _ in state.foods
        )

    def _coop_load_ready_agents(
        self,
        state: LevelBasedForagingState,
    ) -> frozenset[int]:
        ready: set[int] = set()
        for food_row, food_col, food_level in state.foods:
            adjacent_agents = [
                agent_idx
                for agent_idx, position in enumerate(state.agent_positions)
                if self._adjacent(position, (food_row, food_col))
            ]
            if len(adjacent_agents) < 2:
                continue
            if sum(state.agent_levels[agent_idx] for agent_idx in adjacent_agents) < food_level:
                continue
            ready.update(
                agent_idx
                for agent_idx in adjacent_agents
                if state.agent_levels[agent_idx] < food_level
            )
        return frozenset(ready)

    @staticmethod
    def _adjacent(first: tuple[int, int], second: tuple[int, int]) -> bool:
        return (
            abs(first[0] - second[0]) == 1 and first[1] == second[1]
        ) or (
            abs(first[1] - second[1]) == 1 and first[0] == second[0]
        )

    @staticmethod
    def _formula_atomic_props(formula: str) -> set[str]:
        ltl_keywords = {
            "F",
            "G",
            "M",
            "R",
            "U",
            "W",
            "X",
            "false",
            "t",
            "true",
        }
        return {
            token
            for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", str(formula))
            if token not in ltl_keywords
        }

    @staticmethod
    def _foods_from_env(env) -> tuple[tuple[int, int, int], ...]:
        return tuple(
            sorted(
                (
                    int(row),
                    int(col),
                    int(env.field[row, col]),
                )
                for row, col in zip(*env.field.nonzero())
            )
        )
