from __future__ import annotations

from collections import deque
from itertools import combinations, product
import os
import time
from typing import Iterable, Mapping, Sequence

import numpy as np

from .core import (
    AbstractionModel,
    DeterministicSafetyAutomaton,
    EnvState,
    LocalShield,
    UnsafeInitialStateError,
    safety_projection,
)
from ._contract_types import (
    AssumeGuaranteeProductState,
    AssumeGuaranteeShieldTemplate,
    CertifiedContractProfile,
    ContractCertificationError,
)


def _contract_progress_interval_states() -> int:
    raw_value = os.environ.get("CSH_CONTRACT_PROGRESS_INTERVAL_STATES", "0")
    try:
        return max(int(raw_value), 0)
    except ValueError:
        return 0


def _shield_permissiveness(shield: LocalShield[EnvState]) -> float:
    ratios: list[float] = []
    for state, actions in shield.transitions.items():
        if state not in shield.winning_region or not actions:
            continue
        safe_count = len(shield.safe_local_actions(state))
        ratios.append(float(safe_count) / float(len(actions)))
    return float(np.mean(ratios)) if ratios else 0.0


def initial_assume_guarantee_product_state(
    model: AbstractionModel[EnvState],
    automata: Sequence[DeterministicSafetyAutomaton],
    env_state: EnvState,
) -> AssumeGuaranteeProductState[EnvState]:
    env_state = safety_projection(model, env_state)
    label = model.label(env_state)
    return AssumeGuaranteeProductState(
        env_state=env_state,
        obligation_states=tuple(
            automaton.transition(automaton.initial_state, label)
            for automaton in automata
        ),
    )


def _assume_guarantee_product_state_is_safe(
    state: AssumeGuaranteeProductState[EnvState],
    automata: Sequence[DeterministicSafetyAutomaton],
) -> bool:
    return all(
        monitor_state in automata[agent_idx].safe_states
        for agent_idx, monitor_state in enumerate(state.obligation_states)
    )


def _build_assume_guarantee_product_graph(
    model: AbstractionModel[EnvState],
    automata: Sequence[DeterministicSafetyAutomaton],
    initial_env_state: EnvState,
    *,
    max_states: int | None = None,
    label_cache: dict[EnvState, frozenset[str]] | None = None,
    joint_action_cache: dict[EnvState, tuple[tuple[int, ...], ...]] | None = None,
    successor_cache: dict[tuple[EnvState, tuple[int, ...]], frozenset[EnvState]] | None = None,
    cache_successors: bool = True,
    initial_product_state: AssumeGuaranteeProductState[EnvState] | None = None,
    initial_env_states: Iterable[EnvState] | None = None,
) -> tuple[
    AssumeGuaranteeProductState[EnvState],
    tuple[AssumeGuaranteeProductState[EnvState], ...],
    dict[
        AssumeGuaranteeProductState[EnvState],
        dict[tuple[int, ...], frozenset[AssumeGuaranteeProductState[EnvState]]],
    ],
]:
    label_cache = label_cache if label_cache is not None else {}
    joint_action_cache = joint_action_cache if joint_action_cache is not None else {}
    successor_cache = (
        successor_cache
        if cache_successors and successor_cache is not None
        else ({} if cache_successors else None)
    )
    initial_env_state = safety_projection(model, initial_env_state)

    def _label_for(env_state: EnvState) -> frozenset[str]:
        label = label_cache.get(env_state)
        if label is None:
            label = model.label(env_state)
            label_cache[env_state] = label
        return label

    if initial_product_state is not None:
        initial_states = (
            AssumeGuaranteeProductState(
                env_state=safety_projection(model, initial_product_state.env_state),
                obligation_states=tuple(
                    int(monitor_state)
                    for monitor_state in initial_product_state.obligation_states
                ),
            ),
        )
    else:
        raw_initial_env_states = (
            (initial_env_state,)
            if initial_env_states is None
            else tuple(initial_env_states)
        )
        initial_states = tuple(
            dict.fromkeys(
                AssumeGuaranteeProductState(
                    env_state=safety_projection(model, raw_initial_env_state),
                    obligation_states=tuple(
                        automaton.transition(
                            automaton.initial_state,
                            _label_for(safety_projection(model, raw_initial_env_state)),
                        )
                        for automaton in automata
                    ),
                )
                for raw_initial_env_state in raw_initial_env_states
            )
        )
    if not initial_states:
        raise ValueError("At least one initial assume-guarantee product state is required.")
    initial_state = initial_states[0]
    initial_env_state = initial_state.env_state
    queue: deque[AssumeGuaranteeProductState[EnvState]] = deque(initial_states)
    seen: set[AssumeGuaranteeProductState[EnvState]] = set(initial_states)
    canonical_states: dict[
        AssumeGuaranteeProductState[EnvState],
        AssumeGuaranteeProductState[EnvState],
    ] = {state: state for state in initial_states}
    rejecting_successor_cache: dict[
        tuple[int, ...],
        AssumeGuaranteeProductState[EnvState],
    ] = {}
    transitions: dict[
        AssumeGuaranteeProductState[EnvState],
        dict[tuple[int, ...], frozenset[AssumeGuaranteeProductState[EnvState]]],
    ] = {}
    progress_interval = _contract_progress_interval_states()
    next_progress_at = progress_interval
    started_at = time.monotonic()

    while queue:
        state = queue.popleft()
        action_map: dict[
            tuple[int, ...],
            frozenset[AssumeGuaranteeProductState[EnvState]],
        ] = {}
        if not _assume_guarantee_product_state_is_safe(state, automata):
            transitions[state] = action_map
            continue
        raw_joint_actions = joint_action_cache.get(state.env_state)
        if raw_joint_actions is None:
            raw_joint_actions = tuple(
                tuple(int(action) for action in raw_joint_action)
                for raw_joint_action in model.joint_actions(state.env_state)
            )
            joint_action_cache[state.env_state] = raw_joint_actions
        for raw_joint_action in raw_joint_actions:
            joint_action = tuple(int(action) for action in raw_joint_action)
            successor_key = (state.env_state, joint_action)
            if successor_cache is None:
                env_successors = model.successors_for_joint_action(
                    state.env_state,
                    joint_action,
                )
            else:
                env_successors = successor_cache.get(successor_key)
                if env_successors is None:
                    env_successors = model.successors_for_joint_action(
                        state.env_state,
                        joint_action,
                    )
                    successor_cache[successor_key] = env_successors

            product_successors: set[AssumeGuaranteeProductState[EnvState]] = set()
            for raw_successor in env_successors:
                successor = safety_projection(model, raw_successor)
                label = _label_for(successor)
                obligation_states = tuple(
                    automaton.transition(
                        state.obligation_states[agent_idx],
                        label,
                    )
                    for agent_idx, automaton in enumerate(automata)
                )
                if all(
                    monitor_state in automata[agent_idx].safe_states
                    for agent_idx, monitor_state in enumerate(obligation_states)
                ):
                    candidate_successor = AssumeGuaranteeProductState(
                        env_state=successor,
                        obligation_states=obligation_states,
                    )
                    canonical_successor = canonical_states.setdefault(
                        candidate_successor,
                        candidate_successor,
                    )
                    product_successors.add(canonical_successor)
                    continue
                rejecting_successor = rejecting_successor_cache.get(obligation_states)
                if rejecting_successor is None:
                    rejecting_successor = AssumeGuaranteeProductState(
                        env_state=initial_env_state,
                        obligation_states=obligation_states,
                    )
                    rejecting_successor = canonical_states.setdefault(
                        rejecting_successor,
                        rejecting_successor,
                    )
                    rejecting_successor_cache[obligation_states] = rejecting_successor
                product_successors.add(rejecting_successor)
            successors = frozenset(product_successors)
            action_map[joint_action] = successors
            for successor in successors:
                if successor in seen:
                    continue
                if not _assume_guarantee_product_state_is_safe(successor, automata):
                    continue
                seen.add(successor)
                if max_states is not None and len(seen) > max_states:
                    raise ContractCertificationError(
                        "Incomplete assume-guarantee synthesis: product graph "
                        f"exceeded explicit max_states={max_states}. Remove the "
                        "cap for full-reachable synthesis or increase it for a "
                        "deliberately bounded debug run.",
                        reason="state_space_limit",
                    )
                if progress_interval and len(seen) >= next_progress_at:
                    print(
                        "[contract] assume-guarantee product graph: "
                        f"seen={len(seen)} queued={len(queue)} "
                        f"expanded={len(transitions)} max_states={max_states!r}",
                        flush=True,
                    )
                    next_progress_at += progress_interval
                queue.append(successor)
        transitions[state] = action_map
    if progress_interval:
        print(
            "[contract] assume-guarantee product graph complete: "
            f"seen={len(seen)} expanded={len(transitions)} max_states={max_states!r} "
            f"duration_s={time.monotonic() - started_at:.1f}",
            flush=True,
        )
    return initial_state, initial_states, transitions


def _compute_assume_guarantee_fixed_point(
    model: AbstractionModel[EnvState],
    automata_by_agent: Mapping[str, DeterministicSafetyAutomaton],
    transitions: Mapping[
        AssumeGuaranteeProductState[EnvState],
        Mapping[tuple[int, ...], frozenset[AssumeGuaranteeProductState[EnvState]]],
    ],
) -> tuple[
    frozenset[AssumeGuaranteeProductState[EnvState]],
    dict[AssumeGuaranteeProductState[EnvState], dict[str, tuple[int, ...]]],
]:
    agent_ids = tuple(model.agent_ids)
    automata = tuple(automata_by_agent[agent_id] for agent_id in agent_ids)
    winning: set[AssumeGuaranteeProductState[EnvState]] = {
        state
        for state in transitions
        if all(
            monitor_state in automata[agent_idx].safe_states
            for agent_idx, monitor_state in enumerate(state.obligation_states)
        )
    }
    progress_interval = _contract_progress_interval_states()
    started_at = time.monotonic()
    iteration = 0
    while True:
        iteration += 1
        next_winning: set[AssumeGuaranteeProductState[EnvState]] = set()
        next_allowed: dict[
            AssumeGuaranteeProductState[EnvState],
            dict[str, tuple[int, ...]],
        ] = {}
        for state in tuple(winning):
            state_allowed = _maximal_assume_guarantee_action_rectangle(
                model,
                agent_ids,
                state,
                transitions[state],
                winning,
            )
            if state_allowed is None:
                continue
            next_winning.add(state)
            next_allowed[state] = state_allowed
        if progress_interval:
            print(
                "[contract] assume-guarantee fixed point: "
                f"iteration={iteration} winning_in={len(winning)} "
                f"winning_out={len(next_winning)} allowed={len(next_allowed)} "
                f"duration_s={time.monotonic() - started_at:.1f}",
                flush=True,
            )
        if next_winning == winning:
            if progress_interval:
                print(
                    "[contract] assume-guarantee fixed point complete: "
                    f"iterations={iteration} winning={len(next_winning)} "
                    f"allowed={len(next_allowed)} "
                    f"duration_s={time.monotonic() - started_at:.1f}",
                    flush=True,
                )
            return frozenset(next_winning), next_allowed
        winning = next_winning


def _nonempty_action_subsets(actions: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    action_tuple = tuple(int(action) for action in actions)
    subsets: list[tuple[int, ...]] = []
    for width in range(len(action_tuple), 0, -1):
        subsets.extend(tuple(combo) for combo in combinations(action_tuple, width))
    return tuple(subsets)


_ACTION_RECTANGLE_CACHE: dict[
    tuple[tuple[int, ...], ...],
    tuple[tuple[tuple[int, ...], ...], ...],
] = {}


def _action_rectangle_score(
    candidate_sets: tuple[tuple[int, ...], ...],
) -> tuple[int, int, tuple[tuple[int, ...], ...]]:
    joint_count = 1
    for candidate_set in candidate_sets:
        joint_count *= len(candidate_set)
    return (
        sum(len(candidate_set) for candidate_set in candidate_sets),
        joint_count,
        candidate_sets,
    )


def _sorted_action_rectangles(
    actions_by_agent: Sequence[Sequence[int]],
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    action_key = tuple(
        tuple(int(action) for action in actions)
        for actions in actions_by_agent
    )
    cached = _ACTION_RECTANGLE_CACHE.get(action_key)
    if cached is not None:
        return cached

    subsets_by_agent = tuple(_nonempty_action_subsets(actions) for actions in action_key)
    if any(not subsets for subsets in subsets_by_agent):
        rectangles: tuple[tuple[tuple[int, ...], ...], ...] = tuple()
    else:
        rectangles = tuple(
            sorted(
                product(*subsets_by_agent),
                key=_action_rectangle_score,
                reverse=True,
            )
        )
    _ACTION_RECTANGLE_CACHE[action_key] = rectangles
    return rectangles


def _maximal_assume_guarantee_action_rectangle(
    model: AbstractionModel[EnvState],
    agent_ids: tuple[str, ...],
    state: AssumeGuaranteeProductState[EnvState],
    transitions: Mapping[
        tuple[int, ...],
        frozenset[AssumeGuaranteeProductState[EnvState]],
    ],
    winning: set[AssumeGuaranteeProductState[EnvState]],
) -> dict[str, tuple[int, ...]] | None:
    safe_joint_actions = {
        joint_action
        for joint_action, successors in transitions.items()
        if successors and all(successor in winning for successor in successors)
    }
    if not safe_joint_actions:
        return None

    action_rectangles = _sorted_action_rectangles(
        tuple(
            tuple(int(action) for action in model.local_actions(agent_id, state.env_state))
            for agent_id in agent_ids
        )
    )
    if not action_rectangles:
        return None

    for candidate_sets in action_rectangles:
        if all(
            tuple(int(action) for action in joint_action) in safe_joint_actions
            for joint_action in product(*candidate_sets)
        ):
            return {
                agent_id: candidate_sets[agent_idx]
                for agent_idx, agent_id in enumerate(agent_ids)
            }
    return None


def synthesize_assume_guarantee_profile_shield(
    model: AbstractionModel[EnvState],
    automata_by_agent: Mapping[str, DeterministicSafetyAutomaton],
    initial_env_state: EnvState,
    *,
    max_states: int | None = None,
    _label_cache: dict[EnvState, frozenset[str]] | None = None,
    _joint_action_cache: dict[EnvState, tuple[tuple[int, ...], ...]] | None = None,
    _successor_cache: dict[tuple[EnvState, tuple[int, ...]], frozenset[EnvState]] | None = None,
    _cache_successors: bool = True,
    initial_product_state: AssumeGuaranteeProductState[EnvState] | None = None,
    initial_env_states: Iterable[EnvState] | None = None,
) -> AssumeGuaranteeShieldTemplate[EnvState]:
    automata = tuple(automata_by_agent[agent_id] for agent_id in model.agent_ids)
    initial_state, initial_states, transitions = _build_assume_guarantee_product_graph(
        model,
        automata,
        initial_env_state,
        max_states=max_states,
        label_cache=_label_cache,
        joint_action_cache=_joint_action_cache,
        successor_cache=_successor_cache,
        cache_successors=_cache_successors,
        initial_product_state=initial_product_state,
        initial_env_states=initial_env_states,
    )
    winning_region, allowed_actions = _compute_assume_guarantee_fixed_point(
        model,
        automata_by_agent,
        transitions,
    )
    losing_initials = tuple(
        state for state in initial_states if state not in winning_region
    )
    if losing_initials:
        raise UnsafeInitialStateError(
            "One or more initial assume-guarantee product states are outside "
            "the fixed point."
        )
    return AssumeGuaranteeShieldTemplate(
        transitions=transitions,
        winning_region=winning_region,
        allowed_actions=allowed_actions,
        initial_state=initial_state,
        initial_states=initial_states,
        max_states=max_states,
    )


def _assume_guarantee_permissiveness(
    model: AbstractionModel[EnvState],
    template: AssumeGuaranteeShieldTemplate[EnvState],
) -> float:
    ratios: list[float] = []
    for state in template.winning_region:
        for agent_id in model.agent_ids:
            actions = tuple(model.local_actions(agent_id, state.env_state))
            if not actions:
                continue
            safe_count = len(template.allowed_actions[state][agent_id])
            ratios.append(float(safe_count) / float(len(actions)))
    return float(np.mean(ratios)) if ratios else 0.0


ProfileActionMaskBehavior = dict[EnvState, tuple[tuple[int, ...], ...]]


def _profile_action_mask_behavior_map(
    model: AbstractionModel[EnvState],
    certified: CertifiedContractProfile[EnvState],
) -> ProfileActionMaskBehavior | None:
    template = next(iter(certified.templates.values()), None)
    if not isinstance(template, AssumeGuaranteeShieldTemplate):
        return None

    agent_ids = tuple(str(agent_id) for agent_id in model.agent_ids)
    masks_by_env_state: ProfileActionMaskBehavior = {}
    for product_state in template.winning_region:
        state_masks = template.allowed_actions.get(product_state)
        if state_masks is None:
            return None
        mask_tuple = tuple(
            tuple(sorted({int(action) for action in state_masks.get(agent_id, ())}))
            for agent_id in agent_ids
        )
        existing = masks_by_env_state.get(product_state.env_state)
        if existing is not None and existing != mask_tuple:
            return None
        masks_by_env_state[product_state.env_state] = mask_tuple

    return masks_by_env_state
