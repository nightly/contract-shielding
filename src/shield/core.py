from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from itertools import product
import os
import time
from typing import Generic, Hashable, Iterable, Mapping, Protocol, TypeVar

import numpy as np


EnvState = TypeVar("EnvState", bound=Hashable)
LocalAction = int
JointAction = tuple[LocalAction, ...]
Label = frozenset[str]


def _shield_progress_interval_states() -> int:
    raw_value = os.environ.get("CSH_SHIELD_PROGRESS_INTERVAL_STATES", "0")
    try:
        return max(int(raw_value), 0)
    except ValueError:
        return 0


def _log_shield_expansions() -> bool:
    return os.environ.get("CSH_SHIELD_EXPANSION_LOG", "").lower() in {
        "1",
        "true",
        "yes",
    }


def _log_shield_expansion_states() -> bool:
    return os.environ.get("CSH_SHIELD_EXPANSION_STATE_LOG", "").lower() in {
        "1",
        "true",
        "yes",
    }


@dataclass(frozen=True)
class AbstractTransitionOutcome(Generic[EnvState]):
    next_state: EnvState
    probability: float = 1.0
    rewards: Mapping[str, float] = field(default_factory=dict)
    terminations: Mapping[str, bool] = field(default_factory=dict)
    truncations: Mapping[str, bool] = field(default_factory=dict)
    label: Label | None = None


class AbstractionModel(Protocol[EnvState]):
    agent_ids: tuple[str, ...]

    def initial_state(self, env: object) -> EnvState: ...

    def abstract_state(self, env: object) -> EnvState: ...

    def possible_labels(self) -> tuple[str, ...]: ...

    def local_actions(self, agent_id: str, state: EnvState) -> tuple[LocalAction, ...]: ...

    def joint_actions(self, state: EnvState) -> tuple[tuple[LocalAction, ...], ...]: ...

    def successors_for_joint_action(
        self,
        state: EnvState,
        joint_action: tuple[LocalAction, ...],
    ) -> frozenset[EnvState]: ...

    def successors_for_local_action(
        self,
        state: EnvState,
        agent_id: str,
        action: LocalAction,
    ) -> frozenset[EnvState]: ...

    def transition_outcomes_for_joint_action(
        self,
        state: EnvState,
        joint_action: tuple[LocalAction, ...],
    ) -> tuple[AbstractTransitionOutcome[EnvState], ...]: ...

    def label(self, state: EnvState) -> Label: ...

def safety_projection(model: object, state: EnvState) -> EnvState:
    projector = getattr(model, "safety_projection", None)
    if projector is None:
        return state
    return projector(state)


@dataclass(frozen=True)
class DeterministicSafetyAutomaton:
    atomic_props: tuple[str, ...]
    states: frozenset[int]
    initial_state: int
    safe_states: frozenset[int]
    transition_map: dict[tuple[int, Label], int]

    def transition(self, state: int, label: Label) -> int:
        projected = frozenset(prop for prop in label if prop in self.atomic_props)
        return self.transition_map[(state, projected)]

    def valuations(self) -> tuple[Label, ...]:
        valuations: list[Label] = []
        for mask in product((False, True), repeat=len(self.atomic_props)):
            valuations.append(
                frozenset(
                    ap for ap, enabled in zip(self.atomic_props, mask, strict=True) if enabled
                )
            )
        return tuple(valuations)


@dataclass(frozen=True)
class ProductState(Generic[EnvState]):
    env_state: EnvState
    automaton_state: int


class ShieldSynthesisError(RuntimeError):
    pass


class StateSpaceLimitExceeded(ShieldSynthesisError):
    pass


class UnsafeInitialStateError(ShieldSynthesisError):
    pass


@dataclass(frozen=True)
class ReachabilityResult(Generic[EnvState]):
    initial_state: ProductState[EnvState]
    initial_states: tuple[ProductState[EnvState], ...]
    transitions: dict[ProductState[EnvState], dict[LocalAction, frozenset[ProductState[EnvState]]]]


@dataclass(frozen=True)
class JointReachabilityResult(Generic[EnvState]):
    initial_state: ProductState[EnvState]
    initial_states: tuple[ProductState[EnvState], ...]
    transitions: dict[ProductState[EnvState], dict[JointAction, frozenset[ProductState[EnvState]]]]


def initial_product_state(
    model: AbstractionModel[EnvState],
    automaton: DeterministicSafetyAutomaton,
    env_state: EnvState,
) -> ProductState[EnvState]:
    env_state = safety_projection(model, env_state)
    monitor_state = automaton.transition(automaton.initial_state, model.label(env_state))
    return ProductState(env_state=env_state, automaton_state=monitor_state)


def build_reachable_product_graph(
    model: AbstractionModel[EnvState],
    automaton: DeterministicSafetyAutomaton,
    initial_env_state: EnvState,
    agent_id: str,
    *,
    max_states: int | None = None,
    initial_product: ProductState[EnvState] | None = None,
) -> ReachabilityResult[EnvState]:
    if initial_product is None:
        initial_states = (initial_product_state(model, automaton, initial_env_state),)
    else:
        initial_states = (
            ProductState(
                env_state=safety_projection(model, initial_product.env_state),
                automaton_state=initial_product.automaton_state,
            ),
        )
    return build_reachable_product_graph_from_initial_products(
        model,
        automaton,
        initial_states,
        agent_id,
        max_states=max_states,
    )


def build_reachable_product_graph_from_initials(
    model: AbstractionModel[EnvState],
    automaton: DeterministicSafetyAutomaton,
    initial_env_states: Iterable[EnvState],
    agent_id: str,
    *,
    max_states: int | None = None,
) -> ReachabilityResult[EnvState]:
    initial_products = tuple(
        initial_product_state(model, automaton, env_state)
        for env_state in initial_env_states
    )
    return build_reachable_product_graph_from_initial_products(
        model,
        automaton,
        initial_products,
        agent_id,
        max_states=max_states,
    )


def build_reachable_product_graph_from_initial_products(
    model: AbstractionModel[EnvState],
    automaton: DeterministicSafetyAutomaton,
    initial_products: Iterable[ProductState[EnvState]],
    agent_id: str,
    *,
    max_states: int | None = None,
) -> ReachabilityResult[EnvState]:
    initial_states = tuple(
        dict.fromkeys(
            ProductState(
                env_state=safety_projection(model, product_state.env_state),
                automaton_state=product_state.automaton_state,
            )
            for product_state in initial_products
        )
    )
    if not initial_states:
        raise ValueError("At least one initial product state is required.")

    queue: deque[ProductState[EnvState]] = deque(initial_states)
    seen: set[ProductState[EnvState]] = set(initial_states)
    canonical_states: dict[ProductState[EnvState], ProductState[EnvState]] = {
        state: state for state in initial_states
    }
    transitions: dict[
        ProductState[EnvState],
        dict[LocalAction, frozenset[ProductState[EnvState]]],
    ] = {}
    progress_interval = _shield_progress_interval_states()
    next_progress_at = progress_interval
    rejecting_sinks: dict[int, ProductState[EnvState]] = {}
    rejecting_sink_env_state = initial_states[0].env_state
    started_at = time.monotonic()

    def _product_successor(raw_successor: EnvState) -> ProductState[EnvState]:
        successor = safety_projection(model, raw_successor)
        automaton_state = automaton.transition(
            state.automaton_state,
            model.label(successor),
        )
        if automaton_state in automaton.safe_states:
            candidate = ProductState(
                env_state=successor,
                automaton_state=automaton_state,
            )
            return canonical_states.setdefault(candidate, candidate)
        sink = rejecting_sinks.get(automaton_state)
        if sink is None:
            sink = ProductState(
                env_state=rejecting_sink_env_state,
                automaton_state=automaton_state,
            )
            sink = canonical_states.setdefault(sink, sink)
            rejecting_sinks[automaton_state] = sink
        return sink

    while queue:
        state = queue.popleft()
        action_map: dict[LocalAction, frozenset[ProductState[EnvState]]] = {}
        if state.automaton_state not in automaton.safe_states:
            transitions[state] = action_map
            continue
        for local_action in model.local_actions(agent_id, state.env_state):
            successors = frozenset(
                successor
                for raw_successor in model.successors_for_local_action(
                    state.env_state,
                    agent_id,
                    local_action,
                )
                for successor in (_product_successor(raw_successor),)
            )
            action_map[int(local_action)] = successors
            for successor in successors:
                if successor in seen:
                    continue
                seen.add(successor)
                if max_states is not None and len(seen) > max_states:
                    raise StateSpaceLimitExceeded(
                        "Incomplete shield synthesis: reachable product graph "
                        f"exceeded explicit max_states={max_states}. Remove the "
                        "cap for full-reachable synthesis or increase it for a "
                        "deliberately bounded debug run."
                    )
                if progress_interval and len(seen) >= next_progress_at:
                    print(
                        "[shield] local product graph: "
                        f"agent={agent_id!r} seen={len(seen)} "
                        f"queued={len(queue)} expanded={len(transitions)} "
                        f"max_states={max_states!r}",
                        flush=True,
                    )
                    next_progress_at += progress_interval
                queue.append(successor)
        transitions[state] = action_map
    if progress_interval:
        print(
            "[shield] local product graph complete: "
            f"agent={agent_id!r} seen={len(seen)} expanded={len(transitions)} "
            f"max_states={max_states!r} duration_s={time.monotonic() - started_at:.1f}",
            flush=True,
        )
    return ReachabilityResult(
        initial_state=initial_states[0],
        initial_states=initial_states,
        transitions=transitions,
    )


def build_reachable_joint_product_graph(
    model: AbstractionModel[EnvState],
    automaton: DeterministicSafetyAutomaton,
    initial_env_state: EnvState,
    *,
    max_states: int | None = None,
    initial_product: ProductState[EnvState] | None = None,
) -> JointReachabilityResult[EnvState]:
    if initial_product is None:
        initial_states = (initial_product_state(model, automaton, initial_env_state),)
    else:
        initial_states = (
            ProductState(
                env_state=safety_projection(model, initial_product.env_state),
                automaton_state=initial_product.automaton_state,
            ),
        )
    return build_reachable_joint_product_graph_from_initial_products(
        model,
        automaton,
        initial_states,
        max_states=max_states,
    )


def build_reachable_joint_product_graph_from_initials(
    model: AbstractionModel[EnvState],
    automaton: DeterministicSafetyAutomaton,
    initial_env_states: Iterable[EnvState],
    *,
    max_states: int | None = None,
) -> JointReachabilityResult[EnvState]:
    initial_products = tuple(
        initial_product_state(model, automaton, env_state)
        for env_state in initial_env_states
    )
    return build_reachable_joint_product_graph_from_initial_products(
        model,
        automaton,
        initial_products,
        max_states=max_states,
    )


def build_reachable_joint_product_graph_from_initial_products(
    model: AbstractionModel[EnvState],
    automaton: DeterministicSafetyAutomaton,
    initial_products: Iterable[ProductState[EnvState]],
    *,
    max_states: int | None = None,
) -> JointReachabilityResult[EnvState]:
    initial_states = tuple(
        dict.fromkeys(
            ProductState(
                env_state=safety_projection(model, product_state.env_state),
                automaton_state=product_state.automaton_state,
            )
            for product_state in initial_products
        )
    )
    if not initial_states:
        raise ValueError("At least one initial product state is required.")

    queue: deque[ProductState[EnvState]] = deque(initial_states)
    seen: set[ProductState[EnvState]] = set(initial_states)
    canonical_states: dict[ProductState[EnvState], ProductState[EnvState]] = {
        state: state for state in initial_states
    }
    transitions: dict[
        ProductState[EnvState],
        dict[JointAction, frozenset[ProductState[EnvState]]],
    ] = {}
    progress_interval = _shield_progress_interval_states()
    next_progress_at = progress_interval
    rejecting_sinks: dict[int, ProductState[EnvState]] = {}
    rejecting_sink_env_state = initial_states[0].env_state
    started_at = time.monotonic()

    def _product_successor(raw_successor: EnvState) -> ProductState[EnvState]:
        successor = safety_projection(model, raw_successor)
        automaton_state = automaton.transition(
            state.automaton_state,
            model.label(successor),
        )
        if automaton_state in automaton.safe_states:
            candidate = ProductState(
                env_state=successor,
                automaton_state=automaton_state,
            )
            return canonical_states.setdefault(candidate, candidate)
        sink = rejecting_sinks.get(automaton_state)
        if sink is None:
            sink = ProductState(
                env_state=rejecting_sink_env_state,
                automaton_state=automaton_state,
            )
            sink = canonical_states.setdefault(sink, sink)
            rejecting_sinks[automaton_state] = sink
        return sink

    while queue:
        state = queue.popleft()
        action_map: dict[JointAction, frozenset[ProductState[EnvState]]] = {}
        if state.automaton_state not in automaton.safe_states:
            transitions[state] = action_map
            continue
        for joint_action in model.joint_actions(state.env_state):
            normalized_joint_action = tuple(int(action) for action in joint_action)
            successors = frozenset(
                successor
                for raw_successor in model.successors_for_joint_action(
                    state.env_state,
                    normalized_joint_action,
                )
                for successor in (_product_successor(raw_successor),)
            )
            action_map[normalized_joint_action] = successors
            for successor in successors:
                if successor in seen:
                    continue
                seen.add(successor)
                if max_states is not None and len(seen) > max_states:
                    raise StateSpaceLimitExceeded(
                        "Incomplete joint shield synthesis: reachable product graph "
                        f"exceeded explicit max_states={max_states}. Remove the "
                        "cap for full-reachable synthesis or increase it for a "
                        "deliberately bounded debug run."
                    )
                if progress_interval and len(seen) >= next_progress_at:
                    print(
                        "[shield] joint product graph: "
                        f"seen={len(seen)} queued={len(queue)} "
                        f"expanded={len(transitions)} max_states={max_states!r}",
                        flush=True,
                    )
                    next_progress_at += progress_interval
                queue.append(successor)
        transitions[state] = action_map
    if progress_interval:
        print(
            "[shield] joint product graph complete: "
            f"seen={len(seen)} expanded={len(transitions)} "
            f"max_states={max_states!r} duration_s={time.monotonic() - started_at:.1f}",
            flush=True,
        )
    return JointReachabilityResult(
        initial_state=initial_states[0],
        initial_states=initial_states,
        transitions=transitions,
    )


def compute_winning_region(
    automaton: DeterministicSafetyAutomaton,
    transitions: Mapping[
        ProductState[EnvState],
        Mapping[Hashable, frozenset[ProductState[EnvState]]],
    ],
) -> frozenset[ProductState[EnvState]]:
    winning: set[ProductState[EnvState]] = {
        state
        for state in transitions
        if state.automaton_state in automaton.safe_states
    }

    changed = True
    while changed:
        changed = False
        for state in tuple(winning):
            safe_action_exists = False
            for successors in transitions[state].values():
                if successors and all(successor in winning for successor in successors):
                    safe_action_exists = True
                    break
            if not safe_action_exists:
                winning.remove(state)
                changed = True
    return frozenset(winning)


class LocalShield(Generic[EnvState]):
    def __init__(
        self,
        agent_id: str,
        model: AbstractionModel[EnvState],
        automaton: DeterministicSafetyAutomaton,
        transitions: dict[
            ProductState[EnvState],
            dict[LocalAction, frozenset[ProductState[EnvState]]],
        ],
        winning_region: frozenset[ProductState[EnvState]],
        *,
        rng: np.random.Generator | None = None,
        max_states: int | None = None,
        initial_states: Iterable[ProductState[EnvState]] | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.model = model
        self.automaton = automaton
        self.transitions = transitions
        self.winning_region = winning_region
        self.rng = rng or np.random.default_rng()
        self.max_states = max_states
        self.initial_states = tuple(initial_states or ())

    def contains(self, state: ProductState[EnvState]) -> bool:
        state = self._project_state(state)
        if state in self.winning_region:
            return True
        if state in self.transitions:
            return False
        return self._expand_from(state)

    def safe_local_actions(self, state: ProductState[EnvState]) -> tuple[LocalAction, ...]:
        state = self._project_state(state)
        if state not in self.transitions:
            self._expand_from(state)
        if state not in self.transitions:
            raise KeyError(f"Unknown product state: {state!r}")
        if state not in self.winning_region:
            return ()
        safe_actions: list[LocalAction] = []
        for action, successors in self.transitions[state].items():
            if successors and all(successor in self.winning_region for successor in successors):
                safe_actions.append(action)
        return tuple(safe_actions)

    def _project_state(self, state: ProductState[EnvState]) -> ProductState[EnvState]:
        return ProductState(
            env_state=safety_projection(self.model, state.env_state),
            automaton_state=state.automaton_state,
        )

    def _expand_from(self, state: ProductState[EnvState]) -> bool:
        projected_state = self._project_state(state)
        if projected_state in self.winning_region:
            return True
        if not hasattr(self.model, "local_actions") or not hasattr(
            self.model,
            "successors_for_local_action",
        ):
            return False
        if _log_shield_expansions():
            print(
                "[shield] expanding local shield: "
                f"agent={self.agent_id!r} monitor={projected_state.automaton_state} "
                f"known_transitions={len(self.transitions)} "
                f"known_winning={len(self.winning_region)} "
                f"max_states={self.max_states!r}",
                flush=True,
            )
            if _log_shield_expansion_states():
                print(
                    f"[shield] local expansion state: {projected_state!r}",
                    flush=True,
                )
        reachability = build_reachable_product_graph(
            self.model,
            self.automaton,
            projected_state.env_state,
            self.agent_id,
            max_states=self.max_states,
            initial_product=projected_state,
        )
        winning_region = compute_winning_region(
            self.automaton,
            reachability.transitions,
        )
        self.transitions.update(reachability.transitions)
        self.winning_region = frozenset(
            set(self.winning_region).union(winning_region)
        )
        if _log_shield_expansions():
            print(
                "[shield] expanded local shield: "
                f"agent={self.agent_id!r} added_transitions={len(reachability.transitions)} "
                f"added_winning={len(winning_region)} "
                f"total_transitions={len(self.transitions)} "
                f"total_winning={len(self.winning_region)}",
                flush=True,
            )
        return reachability.initial_state in winning_region

    def repair(
        self,
        proposed_local_action: LocalAction,
        state: ProductState[EnvState],
    ) -> LocalAction:
        safe_actions = self.safe_local_actions(state)
        if not safe_actions:
            raise UnsafeInitialStateError(f"No safe local action available at {state!r}.")

        best_score = max(int(candidate == proposed_local_action) for candidate in safe_actions)
        best_actions = [
            candidate
            for candidate in safe_actions
            if int(candidate == proposed_local_action) == best_score
        ]
        index = int(self.rng.integers(0, len(best_actions)))
        return int(best_actions[index])


class JointShield(Generic[EnvState]):
    def __init__(
        self,
        model: AbstractionModel[EnvState],
        automaton: DeterministicSafetyAutomaton,
        transitions: dict[
            ProductState[EnvState],
            dict[JointAction, frozenset[ProductState[EnvState]]],
        ],
        winning_region: frozenset[ProductState[EnvState]],
        *,
        max_states: int | None = None,
        initial_states: Iterable[ProductState[EnvState]] | None = None,
    ) -> None:
        self.model = model
        self.automaton = automaton
        self.transitions = transitions
        self.winning_region = winning_region
        self.max_states = max_states
        self.initial_states = tuple(initial_states or ())

    def contains(self, state: ProductState[EnvState]) -> bool:
        state = self._project_state(state)
        if state in self.winning_region:
            return True
        if state in self.transitions:
            return False
        return self._expand_from(state)

    def safe_joint_actions(self, state: ProductState[EnvState]) -> tuple[JointAction, ...]:
        state = self._project_state(state)
        if state not in self.transitions:
            self._expand_from(state)
        if state not in self.transitions:
            raise KeyError(f"Unknown product state: {state!r}")
        if state not in self.winning_region:
            return ()
        safe_actions: list[JointAction] = []
        for joint_action, successors in self.transitions[state].items():
            if successors and all(successor in self.winning_region for successor in successors):
                safe_actions.append(joint_action)
        return tuple(safe_actions)

    def _project_state(self, state: ProductState[EnvState]) -> ProductState[EnvState]:
        return ProductState(
            env_state=safety_projection(self.model, state.env_state),
            automaton_state=state.automaton_state,
        )

    def _expand_from(self, state: ProductState[EnvState]) -> bool:
        projected_state = self._project_state(state)
        if projected_state in self.winning_region:
            return True
        if not hasattr(self.model, "joint_actions") or not hasattr(
            self.model,
            "successors_for_joint_action",
        ):
            return False
        reachability = build_reachable_joint_product_graph(
            self.model,
            self.automaton,
            projected_state.env_state,
            max_states=self.max_states,
            initial_product=projected_state,
        )
        winning_region = compute_winning_region(
            self.automaton,
            reachability.transitions,
        )
        self.transitions.update(reachability.transitions)
        self.winning_region = frozenset(
            set(self.winning_region).union(winning_region)
        )
        return reachability.initial_state in winning_region


def synthesize_local_shield(
    model: AbstractionModel[EnvState],
    automaton: DeterministicSafetyAutomaton,
    initial_env_state: EnvState,
    agent_id: str,
    *,
    rng: np.random.Generator | None = None,
    max_states: int | None = None,
    initial_env_states: Iterable[EnvState] | None = None,
) -> LocalShield[EnvState]:
    if initial_env_states is None:
        reachability = build_reachable_product_graph(
            model,
            automaton,
            initial_env_state,
            agent_id,
            max_states=max_states,
        )
    else:
        reachability = build_reachable_product_graph_from_initials(
            model,
            automaton,
            initial_env_states,
            agent_id,
            max_states=max_states,
        )
    winning_region = compute_winning_region(automaton, reachability.transitions)
    losing_initials = tuple(
        state for state in reachability.initial_states if state not in winning_region
    )
    if losing_initials:
        raise UnsafeInitialStateError(
            "One or more initial product states are outside the winning region."
        )
    return LocalShield(
        agent_id=agent_id,
        model=model,
        automaton=automaton,
        transitions=reachability.transitions,
        winning_region=winning_region,
        rng=rng,
        max_states=max_states,
        initial_states=reachability.initial_states,
    )


def synthesize_joint_shield(
    model: AbstractionModel[EnvState],
    automaton: DeterministicSafetyAutomaton,
    initial_env_state: EnvState,
    *,
    max_states: int | None = None,
    initial_env_states: Iterable[EnvState] | None = None,
) -> JointShield[EnvState]:
    if initial_env_states is None:
        reachability = build_reachable_joint_product_graph(
            model,
            automaton,
            initial_env_state,
            max_states=max_states,
        )
    else:
        reachability = build_reachable_joint_product_graph_from_initials(
            model,
            automaton,
            initial_env_states,
            max_states=max_states,
        )
    winning_region = compute_winning_region(automaton, reachability.transitions)
    losing_initials = tuple(
        state for state in reachability.initial_states if state not in winning_region
    )
    if losing_initials:
        raise UnsafeInitialStateError(
            "One or more initial product states are outside the joint winning region."
        )
    return JointShield(
        model=model,
        automaton=automaton,
        transitions=reachability.transitions,
        winning_region=winning_region,
        max_states=max_states,
        initial_states=reachability.initial_states,
    )
