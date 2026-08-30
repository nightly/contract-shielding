from __future__ import annotations

from itertools import chain, combinations, product
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from .automaton import SpotAutomatonBackend
from .core import (
    AbstractionModel,
    DeterministicSafetyAutomaton,
    EnvState,
    UnsafeInitialStateError,
)
from ._assume_guarantee import (
    ProfileActionMaskBehavior,
    _ACTION_RECTANGLE_CACHE,
    _action_rectangle_score,
    _assume_guarantee_permissiveness,
    _assume_guarantee_product_state_is_safe,
    _build_assume_guarantee_product_graph,
    _compute_assume_guarantee_fixed_point,
    _contract_progress_interval_states,
    _maximal_assume_guarantee_action_rectangle,
    _nonempty_action_subsets,
    _profile_action_mask_behavior_map,
    _shield_permissiveness,
    _sorted_action_rectangles,
    initial_assume_guarantee_product_state,
    synthesize_assume_guarantee_profile_shield,
)
from ._contract_candidates import (
    _FALSE_FORMULAS,
    _LTL_KEYWORDS,
    _CachedAutomatonBackend,
    _CachedSpotLTLHelper,
    _automaton_behavior_signature,
    _candidate_scopes_by_agent,
    _compile_formula,
    _enumerate_scoped_local_obligation_candidates,
    _is_fast_safety_fragment,
    _merge_seed_local_obligation_candidates,
    _model_contract_local_alphabet_by_agent,
    _model_contract_seed_formulas_by_agent,
    _normalize_contract_local_alphabet_by_agent,
    _ordered_candidate_labels,
    _prune_equivalent_local_obligation_candidates,
    _resolve_contract_local_alphabet_by_agent,
    SpotLTLHelper,
    conjunction_formula,
    enumerate_contract_local_obligation_candidates,
    enumerate_local_obligation_candidates,
    extract_atomic_props,
    formula_size,
    profile_conjunction,
)
from ._contract_types import (
    AssumeGuaranteeLocalShield,
    AssumeGuaranteeProductState,
    AssumeGuaranteeShieldTemplate,
    BanditSelectionResult,
    CertifiedContractProfile,
    ContractCandidateEnumeration,
    ContractCertificationError,
    ContractError,
    ContractLibrary,
    ContractProfile,
    ContractSynthesisConfig,
    LocalObligationCandidate,
    LocalObligationShieldTemplate,
    VotingConfig,
)


class DiscountedUCBProfileSelector:
    def __init__(
        self,
        profile_ids: Sequence[str],
        config: VotingConfig,
    ) -> None:
        if not profile_ids:
            raise ValueError("DiscountedUCBProfileSelector requires at least one profile.")
        if not (0.0 < float(config.bandit_discount) <= 1.0):
            raise ValueError("bandit_discount must be in (0, 1].")
        if float(config.bandit_exploration_coef) < 0.0:
            raise ValueError("bandit_exploration_coef must be non-negative.")
        self.profile_ids = tuple(str(profile_id) for profile_id in profile_ids)
        self.config = config
        self.visited_profile_ids: set[str] = set()
        self._discounted_counts = {
            profile_id: 0.0
            for profile_id in self.profile_ids
        }
        self._discounted_totals = {
            profile_id: 0.0
            for profile_id in self.profile_ids
        }

    def record_block(self, profile_id: str, observed_score: float) -> None:
        if profile_id not in self._discounted_counts:
            raise ValueError(f"Unknown contract profile id: {profile_id!r}")
        discount = float(self.config.bandit_discount)
        for candidate_id in self.profile_ids:
            self._discounted_counts[candidate_id] *= discount
            self._discounted_totals[candidate_id] *= discount
        self._discounted_counts[profile_id] += 1.0
        self._discounted_totals[profile_id] += float(observed_score)
        self.visited_profile_ids.add(profile_id)

    def select(self, *, current_profile_id: str) -> BanditSelectionResult:
        if current_profile_id not in self._discounted_counts:
            raise ValueError(f"Unknown contract profile id: {current_profile_id!r}")
        scores = self.arm_ucb_scores()
        unseen = [
            profile_id
            for profile_id in self.profile_ids
            if profile_id not in self.visited_profile_ids
        ]
        if unseen:
            winner = unseen[0]
            selection_margin = 0.0
        else:
            indexed = {
                profile_id: idx
                for idx, profile_id in enumerate(self.profile_ids)
            }
            ranked = sorted(
                self.profile_ids,
                key=lambda profile_id: (
                    scores[profile_id],
                    int(profile_id == current_profile_id),
                    -indexed[profile_id],
                ),
                reverse=True,
            )
            winner = ranked[0]
            selection_margin = (
                float(scores[ranked[0]] - scores[ranked[1]])
                if len(ranked) > 1
                else float(scores[ranked[0]])
            )
        return BanditSelectionResult(
            winner_profile_id=winner,
            arm_means=self.arm_means(),
            arm_counts=self.arm_counts(),
            arm_ucb_scores=scores,
            selection_margin=selection_margin,
        )

    def arm_means(self) -> dict[str, float]:
        means: dict[str, float] = {}
        for profile_id in self.profile_ids:
            count = self._discounted_counts[profile_id]
            means[profile_id] = (
                float(self._discounted_totals[profile_id] / count)
                if count > 0.0
                else 0.0
            )
        return means

    def arm_counts(self) -> dict[str, float]:
        return {
            profile_id: float(self._discounted_counts[profile_id])
            for profile_id in self.profile_ids
        }

    def arm_ucb_scores(self) -> dict[str, float]:
        means = self.arm_means()
        total_count = sum(self._discounted_counts.values())
        log_term = np.log(max(total_count, 1.0) + 1.0)
        scores: dict[str, float] = {}
        for profile_id in self.profile_ids:
            count = self._discounted_counts[profile_id]
            if count <= 0.0:
                scores[profile_id] = float("inf")
                continue
            exploration = float(self.config.bandit_exploration_coef) * np.sqrt(
                log_term / count
            )
            scores[profile_id] = float(means[profile_id] + exploration)
        return scores


def _profile_trace_fields(profile: ContractProfile) -> dict[str, Any]:
    return {
        "formulas": dict(profile.formulas),
        "active_candidates": {
            agent_id: tuple(candidate_ids)
            for agent_id, candidate_ids in profile.active_candidates.items()
        },
    }


def certify_contract_profile(
    model: AbstractionModel[EnvState],
    *,
    global_formula: str,
    profile: ContractProfile,
    initial_env_state: EnvState,
    initial_env_states: Iterable[EnvState] | None = None,
    helper: SpotLTLHelper | None = None,
    backend: SpotAutomatonBackend | None = None,
    max_states: int | None = None,
    rng: np.random.Generator | None = None,
    _label_cache: dict[EnvState, frozenset[str]] | None = None,
    _joint_action_cache: dict[EnvState, tuple[tuple[int, ...], ...]] | None = None,
    _successor_cache: dict[tuple[EnvState, tuple[int, ...]], frozenset[EnvState]] | None = None,
    _cache_successors: bool = True,
) -> CertifiedContractProfile[EnvState]:
    helper = helper or SpotLTLHelper()
    backend = backend or SpotAutomatonBackend()
    profile_formula = profile_conjunction(profile)
    if not helper.formula_implies(profile_formula, global_formula):
        witness = helper.counterexample_word(profile_formula, global_formula)
        raise ContractCertificationError(
            "Contract profile does not imply the global safety formula.",
            reason="implication",
            witness=witness,
        )

    global_automaton = _compile_formula(global_formula, backend=backend)
    automata: dict[str, DeterministicSafetyAutomaton] = {}

    for agent_id in model.agent_ids:
        formula = profile.formula_for(agent_id)
        automaton = _compile_formula(formula, backend=backend)
        automata[agent_id] = automaton

    try:
        ag_template = synthesize_assume_guarantee_profile_shield(
            model,
            automata,
            initial_env_state,
            max_states=max_states,
            _label_cache=_label_cache,
            _joint_action_cache=_joint_action_cache,
            _successor_cache=_successor_cache,
            _cache_successors=_cache_successors,
            initial_env_states=initial_env_states,
        )
    except UnsafeInitialStateError as exc:
        raise ContractCertificationError(
            "Assume-guarantee profile is not realizable from the initial state.",
            reason="local_realizability",
        ) from exc
    templates: dict[str, AssumeGuaranteeShieldTemplate[EnvState]] = {
        agent_id: ag_template
        for agent_id in model.agent_ids
    }
    permissiveness = _assume_guarantee_permissiveness(model, ag_template)

    return CertifiedContractProfile(
        profile=profile,
        global_formula=global_formula,
        profile_formula=profile_formula,
        global_automaton=global_automaton,
        automata=automata,
        templates=templates,
        permissiveness=permissiveness,
        semantics="assume_guarantee",
    )


def _profile_from_candidate_sets(
    *,
    profile_id: str,
    agent_ids: Sequence[str],
    candidate_sets: Sequence[tuple[str, ...]],
    candidates_by_id: Mapping[str, LocalObligationCandidate],
) -> ContractProfile:
    formulas: dict[str, str] = {}
    active: dict[str, tuple[str, ...]] = {}
    for agent_id, candidate_ids in zip(agent_ids, candidate_sets, strict=True):
        active[agent_id] = tuple(candidate_ids)
        formulas[agent_id] = conjunction_formula(
            [candidates_by_id[candidate_id].formula for candidate_id in candidate_ids]
        )
    return ContractProfile(
        profile_id=profile_id,
        formulas=formulas,
        active_candidates=active,
    )


def _candidate_subsets(
    candidates: Sequence[LocalObligationCandidate],
    *,
    max_active_per_agent: int,
) -> tuple[tuple[str, ...], ...]:
    ids = tuple(candidate.candidate_id for candidate in candidates)
    return _candidate_subsets_for_ids(
        ids,
        max_active_per_agent=max_active_per_agent,
    )


def _candidate_subsets_for_ids(
    candidate_ids: Sequence[str],
    *,
    max_active_per_agent: int,
) -> tuple[tuple[str, ...], ...]:
    ids = tuple(str(candidate_id) for candidate_id in candidate_ids)
    subsets: list[tuple[str, ...]] = [tuple()]
    for width in range(1, max(int(max_active_per_agent), 0) + 1):
        subsets.extend(tuple(combo) for combo in combinations(ids, width))
    return tuple(subsets)


def _iter_candidate_subsets_for_ids(
    candidate_ids: Sequence[str],
    *,
    max_active_per_agent: int,
) -> Iterator[tuple[str, ...]]:
    ids = tuple(str(candidate_id) for candidate_id in candidate_ids)
    yield tuple()
    for width in range(1, max(int(max_active_per_agent), 0) + 1):
        yield from (tuple(combo) for combo in combinations(ids, width))


def _profile_candidate_vectors(
    *,
    agent_count: int,
    subsets: Sequence[tuple[str, ...]],
) -> Iterable[tuple[tuple[str, ...], ...]]:
    return _profile_candidate_vectors_by_agent(
        subsets_by_agent=tuple(tuple(subsets) for _ in range(agent_count)),
    )


def _profile_candidate_vectors_by_agent(
    *,
    subsets_by_agent: Sequence[Sequence[tuple[str, ...]]],
) -> Iterable[tuple[tuple[str, ...], ...]]:
    agent_count = len(subsets_by_agent)
    normalized_subsets = tuple(
        tuple(tuple(str(candidate_id) for candidate_id in subset) for subset in subsets)
        for subsets in subsets_by_agent
    )
    empty = tuple(() for _ in range(agent_count))
    yielded: set[tuple[tuple[str, ...], ...]] = {empty}
    singletons = tuple(
        dict.fromkeys(
            subset
            for subsets in normalized_subsets
            for subset in subsets
            if len(subset) == 1
        )
    )

    def _yield_vector(
        agent_idx: int,
        subset: tuple[str, ...],
    ) -> tuple[tuple[str, ...], ...] | None:
        if subset not in normalized_subsets[agent_idx]:
            return None
        vector = list(empty)
        vector[agent_idx] = subset
        tuple_vector = tuple(vector)
        if tuple_vector in yielded:
            return None
        yielded.add(tuple_vector)
        return tuple_vector

    if singletons:
        first_singleton = singletons[0]
        for agent_idx in range(agent_count):
            tuple_vector = _yield_vector(agent_idx, first_singleton)
            if tuple_vector is not None:
                yield tuple_vector

    for agent_idx in range(agent_count):
        for subset in singletons[1:]:
            tuple_vector = _yield_vector(agent_idx, subset)
            if tuple_vector is not None:
                yield tuple_vector

    for tuple_vector in product(*normalized_subsets):
        if tuple_vector in yielded:
            continue
        yielded.add(tuple_vector)
        yield tuple_vector


def _lazy_profile_candidate_vectors_by_agent(
    *,
    candidate_ids_by_agent: Sequence[Sequence[str]],
    max_active_per_agent: int,
) -> Iterable[tuple[tuple[str, ...], ...]]:
    candidate_ids = tuple(
        tuple(str(candidate_id) for candidate_id in agent_candidate_ids)
        for agent_candidate_ids in candidate_ids_by_agent
    )
    prefix: list[tuple[str, ...]] = []

    def _recurse(agent_idx: int) -> Iterator[tuple[tuple[str, ...], ...]]:
        if agent_idx >= len(candidate_ids):
            yield tuple(prefix)
            return
        for subset in _iter_candidate_subsets_for_ids(
            candidate_ids[agent_idx],
            max_active_per_agent=max_active_per_agent,
        ):
            prefix.append(subset)
            yield from _recurse(agent_idx + 1)
            prefix.pop()

    yield from _recurse(0)


def _priority_profile_candidate_vectors_by_agent(
    *,
    agent_ids: Sequence[str],
    candidate_scopes: Mapping[str, Sequence[str]],
    candidates_by_id: Mapping[str, LocalObligationCandidate],
    global_formula: str,
    max_active_per_agent: int,
    use_temporal_form_heuristic: bool,
    helper: SpotLTLHelper,
) -> Iterable[tuple[tuple[str, ...], ...]]:
    global_props = set(extract_atomic_props(global_formula))
    if not global_props:
        return

    candidate_by_formula = {
        candidate.formula: candidate.candidate_id
        for candidate in candidates_by_id.values()
    }
    candidate_props_by_id = {
        candidate_id: set(extract_atomic_props(candidate.formula))
        for candidate_id, candidate in candidates_by_id.items()
    }
    guard_sets: list[tuple[str, ...]] = []
    guard_formulas: list[str] = []
    guard_sets_within_active_limit = True
    for agent_id in agent_ids:
        scoped = set(candidate_scopes[str(agent_id)])
        guards: list[str] = []
        for prop in sorted(global_props):
            for raw_formula in (f"G(!{prop})", f"G({prop})"):
                candidate_id = candidate_by_formula.get(raw_formula)
                if candidate_id is None:
                    candidate_id = candidate_by_formula.get(
                        helper.simplify(raw_formula)
                    )
                if candidate_id is not None and candidate_id in scoped:
                    guards.append(candidate_id)
                    break
        guard_set = tuple(dict.fromkeys(guards))
        if len(guard_set) > int(max_active_per_agent):
            guard_sets_within_active_limit = False
            guard_set = tuple()
        guard_sets.append(guard_set)
        guard_formulas.append(
            conjunction_formula(
                tuple(candidates_by_id[candidate_id].formula for candidate_id in guard_set)
            )
        )

    def _protocol_rank(candidate: LocalObligationCandidate) -> tuple[int, int, int]:
        prefix_rank = 0
        if use_temporal_form_heuristic:
            formula = candidate.formula
            if formula.startswith("G("):
                prefix_rank = 0
            elif formula.startswith("X("):
                prefix_rank = 1
            else:
                prefix_rank = 2
        try:
            candidate_order = int(candidate.candidate_id.removeprefix("a"))
        except ValueError:
            candidate_order = candidate.size
        return (prefix_rank, candidate_order, candidate.size)

    def _protocol_candidates(
        agent_id: str,
        *,
        excluded: set[str],
    ) -> list[LocalObligationCandidate]:
        scoped = set(candidate_scopes[str(agent_id)])
        return sorted(
            (
                candidate
                for candidate_id in scoped
                for candidate in (candidates_by_id[candidate_id],)
                if candidate_id not in excluded
                and set(extract_atomic_props(candidate.formula)) - global_props
            ),
            key=_protocol_rank,
        )

    global_cover_sets: list[tuple[str, ...]] = []
    for agent_id in agent_ids:
        scoped = set(candidate_scopes[str(agent_id)])
        scoped_candidates = tuple(
            candidates_by_id[candidate_id]
            for candidate_id in scoped
        )
        global_cover_candidates = [
            candidate
            for candidate in scoped_candidates
            if candidate.formula == global_formula
        ]
        scoped_props = {
            prop
            for candidate in scoped_candidates
            for prop in extract_atomic_props(candidate.formula)
        }
        if not global_cover_candidates and global_props <= scoped_props:
            global_cover_candidates = sorted(
                (
                    candidate
                    for candidate in scoped_candidates
                    if global_props <= candidate_props_by_id[candidate.candidate_id]
                    and helper.formula_implies(candidate.formula, global_formula)
                ),
                key=_protocol_rank,
            )
        else:
            global_cover_candidates = sorted(
                global_cover_candidates,
                key=_protocol_rank,
            )
        global_cover_sets.append(
            (global_cover_candidates[0].candidate_id,)
            if global_cover_candidates
            else tuple()
        )

    if any(global_cover_sets) and any(not cover for cover in global_cover_sets):
        balanced_vector: list[tuple[str, ...]] = []
        for agent_id, cover_set in zip(agent_ids, global_cover_sets, strict=True):
            if cover_set:
                balanced_vector.append(cover_set)
                continue
            if int(max_active_per_agent) <= 0:
                balanced_vector.append(tuple())
                continue
            candidates = _protocol_candidates(agent_id, excluded=set())
            balanced_vector.append(
                (candidates[0].candidate_id,)
                if candidates
                else tuple()
            )
        if any(balanced_vector):
            yield tuple(balanced_vector)
        if tuple(global_cover_sets) != tuple(balanced_vector):
            yield tuple(global_cover_sets)

    if not guard_sets_within_active_limit:
        return

    cover_sets: list[tuple[str, ...]] = []
    for agent_id, guard_set, guard_formula in zip(
        agent_ids,
        guard_sets,
        guard_formulas,
        strict=True,
    ):
        scoped = set(candidate_scopes[str(agent_id)])
        scoped_candidates = tuple(
            candidates_by_id[candidate_id]
            for candidate_id in scoped
            if candidate_id not in guard_set
        )
        covering_candidates = [
            candidate
            for candidate in scoped_candidates
            if candidate.formula == guard_formula
        ]
        if not covering_candidates:
            guard_props = set(extract_atomic_props(guard_formula))
            covering_candidates = sorted(
                (
                    candidate
                    for candidate in scoped_candidates
                    if guard_props <= candidate_props_by_id[candidate.candidate_id]
                    and helper.formula_implies(candidate.formula, guard_formula)
                ),
                key=_protocol_rank,
            )
        else:
            covering_candidates = sorted(covering_candidates, key=_protocol_rank)
        cover_sets.append(
            (covering_candidates[0].candidate_id,)
            if covering_candidates
            else guard_set
        )

    cover_vector = tuple(cover_sets)
    if any(cover_vector) and tuple(cover_sets) != tuple(guard_sets):
        yield cover_vector

        remaining_slots = [
            max(int(max_active_per_agent) - len(cover_set), 0)
            for cover_set in cover_sets
        ]
        for agent_idx, agent_id in enumerate(agent_ids):
            if remaining_slots[agent_idx] <= 0:
                continue
            cover_set = cover_sets[agent_idx]
            protocol_candidates = _protocol_candidates(
                agent_id,
                excluded=set(cover_set),
            )
            for candidate in protocol_candidates:
                vector = [tuple(cover) for cover in cover_sets]
                vector[agent_idx] = tuple((*cover_set, candidate.candidate_id))
                yield tuple(vector)

    base_vector = tuple(guard_sets)
    if any(base_vector):
        yield base_vector

    remaining_slots = [
        max(int(max_active_per_agent) - len(guard_set), 0)
        for guard_set in guard_sets
    ]
    if not any(remaining_slots):
        return

    for agent_idx, agent_id in enumerate(agent_ids):
        if remaining_slots[agent_idx] <= 0:
            continue
        guard_set = guard_sets[agent_idx]
        protocol_candidates = _protocol_candidates(
            agent_id,
            excluded=set(guard_set),
        )
        for candidate in protocol_candidates:
            vector = [tuple(guards) for guards in guard_sets]
            vector[agent_idx] = tuple((*guard_set, candidate.candidate_id))
            yield tuple(vector)


def build_contract_library(
    model: AbstractionModel[EnvState],
    *,
    global_formula: str,
    initial_env_state: EnvState,
    initial_env_states: Iterable[EnvState] | None = None,
    config: ContractSynthesisConfig | None = None,
    helper: SpotLTLHelper | None = None,
    backend: SpotAutomatonBackend | None = None,
    rng: np.random.Generator | None = None,
) -> ContractLibrary[EnvState]:
    config = config or ContractSynthesisConfig()
    helper = _CachedSpotLTLHelper(helper or SpotLTLHelper())
    backend = _CachedAutomatonBackend(backend or SpotAutomatonBackend())
    if not hasattr(model, "possible_labels"):
        raise ContractError("Safety model must define possible_labels().")
    agent_ids = tuple(str(agent_id) for agent_id in model.agent_ids)
    enumeration = enumerate_contract_local_obligation_candidates(
        model=model,
        global_formula=global_formula,
        config=config,
        helper=helper,
        backend=backend,
    )
    candidates = enumeration.candidates
    candidate_scopes = enumeration.candidate_scopes
    certification_trace: list[dict[str, Any]] = list(enumeration.pruning_trace)
    if config.print_candidates:
        print("Generated local-obligation candidates:")
        for candidate in candidates:
            scoped_agents = tuple(
                agent_id
                for agent_id in agent_ids
                if candidate.candidate_id in candidate_scopes[agent_id]
            )
            print(
                f"  {candidate.candidate_id}: depth={candidate.depth} "
                f"size={candidate.size} agents={scoped_agents} "
                f"formula={candidate.formula}"
            )

    empty_profile = ContractProfile(
        profile_id="profile_0000",
        formulas={agent_id: "t" for agent_id in agent_ids},
        active_candidates={agent_id: tuple() for agent_id in agent_ids},
    )
    certified_profiles: list[CertifiedContractProfile[EnvState]] = []
    profile_behaviors: dict[str, ProfileActionMaskBehavior | None] = {}
    label_cache: dict[EnvState, frozenset[str]] = {}
    joint_action_cache: dict[EnvState, tuple[tuple[int, ...], ...]] = {}
    successor_cache: dict[tuple[EnvState, tuple[int, ...]], frozenset[EnvState]] = {}

    def _certification_caches() -> tuple[
        dict[EnvState, frozenset[str]],
        dict[EnvState, tuple[tuple[int, ...], ...]],
        dict[tuple[EnvState, tuple[int, ...]], frozenset[EnvState]] | None,
    ]:
        selected_successor_cache = (
            successor_cache
            if config.cache_certification_successors
            else None
        )
        if config.reuse_certification_caches:
            return label_cache, joint_action_cache, selected_successor_cache
        return {}, {}, ({} if config.cache_certification_successors else None)

    try:
        label_cache_for_profile, joint_action_cache_for_profile, successor_cache_for_profile = (
            _certification_caches()
        )
        print(
            "[contract] certifying weakest profile: "
            f"profile_id={empty_profile.profile_id!r}",
            flush=True,
        )
        initial = certify_contract_profile(
            model,
            global_formula=global_formula,
            profile=empty_profile,
            initial_env_state=initial_env_state,
            initial_env_states=initial_env_states,
            helper=helper,
            backend=backend,
            max_states=config.max_states,
            rng=rng,
            _label_cache=label_cache_for_profile,
            _joint_action_cache=joint_action_cache_for_profile,
            _successor_cache=successor_cache_for_profile,
            _cache_successors=config.cache_certification_successors,
        )
        certified_profiles.append(initial)
        certification_trace.append(
            {
                "step": 0,
                "event": "weakest_profile_certified",
                "profile_id": initial.profile.profile_id,
            }
        )
        return ContractLibrary(
            candidates=candidates,
            initial_profile=initial,
            certified_profiles=tuple(certified_profiles),
            certification_trace=tuple(certification_trace),
            candidate_scopes=candidate_scopes,
        )
    except ContractCertificationError as exc:
        certification_trace.append(
            {
                "step": 0,
                "event": "weakest_profile_rejected",
                "reason": exc.reason,
                "witness": exc.witness,
            }
        )

    candidates_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    seen_formula_vectors: set[tuple[str, ...]] = set()
    evaluated = 0
    profile_index = 1
    yielded_vectors: set[tuple[tuple[str, ...], ...]] = set()
    profile_vectors = chain(
        _priority_profile_candidate_vectors_by_agent(
            agent_ids=agent_ids,
            candidate_scopes=candidate_scopes,
            candidates_by_id=candidates_by_id,
            global_formula=global_formula,
            max_active_per_agent=config.max_active_per_agent,
            use_temporal_form_heuristic=config.use_temporal_form_heuristic,
            helper=helper,
        ),
        _lazy_profile_candidate_vectors_by_agent(
            candidate_ids_by_agent=tuple(candidate_scopes[agent_id] for agent_id in agent_ids),
            max_active_per_agent=config.max_active_per_agent,
        ),
    )
    for selected_sets in profile_vectors:
        if selected_sets in yielded_vectors:
            continue
        yielded_vectors.add(selected_sets)
        if evaluated >= int(config.max_profiles):
            break
        if all(not selected for selected in selected_sets):
            continue
        profile = _profile_from_candidate_sets(
            profile_id=f"profile_{profile_index:04d}",
            agent_ids=agent_ids,
            candidate_sets=selected_sets,
            candidates_by_id=candidates_by_id,
        )
        profile_index += 1
        formula_vector = tuple(profile.formulas[agent_id] for agent_id in agent_ids)
        if formula_vector in seen_formula_vectors:
            continue
        seen_formula_vectors.add(formula_vector)
        evaluated += 1
        try:
            label_cache_for_profile, joint_action_cache_for_profile, successor_cache_for_profile = (
                _certification_caches()
            )
            print(
                "[contract] certifying candidate profile: "
                f"step={evaluated} profile_id={profile.profile_id!r} "
                f"active_candidates={profile.active_candidates!r}",
                flush=True,
            )
            certified = certify_contract_profile(
                model,
                global_formula=global_formula,
                profile=profile,
                initial_env_state=initial_env_state,
                initial_env_states=initial_env_states,
                helper=helper,
                backend=backend,
                max_states=config.max_states,
                rng=rng,
                _label_cache=label_cache_for_profile,
                _joint_action_cache=joint_action_cache_for_profile,
                _successor_cache=successor_cache_for_profile,
                _cache_successors=config.cache_certification_successors,
            )
        except ContractCertificationError as exc:
            certification_trace.append(
                {
                    "step": evaluated,
                    "event": "candidate_profile_rejected",
                    "profile_id": profile.profile_id,
                    "reason": exc.reason,
                    "witness": exc.witness,
                }
            )
            continue
        behavior = _profile_action_mask_behavior_map(model, certified)
        pruned_by: CertifiedContractProfile[EnvState] | None = None
        prune_reason: str | None = None
        if behavior is not None:
            for kept in certified_profiles:
                kept_behavior = profile_behaviors.get(kept.profile.profile_id)
                if kept_behavior is None:
                    continue
                if config.prune_equivalent_profiles and behavior == kept_behavior:
                    pruned_by = kept
                    prune_reason = "equivalent_action_masks"
                    break
        if pruned_by is not None:
            certification_trace.append(
                {
                    "step": evaluated,
                    "event": "candidate_profile_pruned",
                    "profile_id": certified.profile.profile_id,
                    "reason": prune_reason,
                    "kept_profile_id": pruned_by.profile.profile_id,
                    **_profile_trace_fields(certified.profile),
                    "permissiveness": certified.permissiveness,
                }
            )
            continue
        certified_profiles.append(certified)
        profile_behaviors[certified.profile.profile_id] = behavior
        certification_trace.append(
            {
                "step": evaluated,
                "event": "candidate_profile_certified",
                "profile_id": profile.profile_id,
                "permissiveness": certified.permissiveness,
            }
        )
    if not certified_profiles:
        raise ContractError(
            "No certified contract profile found within the configured candidate limits."
        )

    initial = certified_profiles[0]
    certification_trace.append(
        {
            "step": evaluated,
            "event": "initial_profile_selected",
            "profile_id": initial.profile.profile_id,
            "selection_basis": "certification_order",
            "permissiveness": initial.permissiveness,
        }
    )
    return ContractLibrary(
        candidates=candidates,
        initial_profile=initial,
        certified_profiles=tuple(certified_profiles),
        certification_trace=tuple(certification_trace),
        candidate_scopes=candidate_scopes,
    )


def serialize_local_obligation_candidates(
    candidates: Sequence[LocalObligationCandidate],
) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": candidate.candidate_id,
            "formula": candidate.formula,
            "depth": int(candidate.depth),
            "size": int(candidate.size),
        }
        for candidate in candidates
    ]


def serialize_certified_contract_profiles(
    profiles: Sequence[CertifiedContractProfile[Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "profile_id": certified.profile.profile_id,
            "formulas": dict(certified.profile.formulas),
            "active_candidates": {
                agent_id: tuple(candidate_ids)
                for agent_id, candidate_ids in certified.profile.active_candidates.items()
            },
            "profile_formula": certified.profile_formula,
            "permissiveness": float(certified.permissiveness),
        }
        for certified in profiles
    ]
