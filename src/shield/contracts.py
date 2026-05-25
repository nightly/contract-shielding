from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from itertools import chain, combinations, product
import os
import re
import shutil
import subprocess
import time
from typing import Any, Generic, Iterable, Mapping, Sequence, TypeVar

import numpy as np

from .automaton import SpotAutomatonBackend, require_spot_cli
from .core import (
    AbstractionModel,
    DeterministicSafetyAutomaton,
    EnvState,
    LocalShield,
    ProductState,
    UnsafeInitialStateError,
    safety_projection,
    synthesize_local_shield,
)


_LTL_KEYWORDS = frozenset(
    {
        "F",
        "G",
        "M",
        "R",
        "U",
        "W",
        "X",
        "f",
        "t",
        "false",
        "true",
    }
)
_FALSE_FORMULAS = frozenset({"0", "f", "false"})


def _contract_progress_interval_states() -> int:
    raw_value = os.environ.get("CSH_CONTRACT_PROGRESS_INTERVAL_STATES", "0")
    try:
        return max(int(raw_value), 0)
    except ValueError:
        return 0


class ContractError(RuntimeError):
    """Base error for contract-profile synthesis and certification."""
    pass


class ContractCertificationError(ContractError):
    """Raised when a contract profile cannot be certified."""

    def __init__(
        self,
        message: str,
        *,
        reason: str,
        witness: str | None = None,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.witness = witness


@dataclass(frozen=True)
class LocalObligationCandidate:
    """Candidate SafeLTL formula that may become one agent's local obligation."""

    candidate_id: str
    formula: str
    depth: int
    size: int


@dataclass(frozen=True)
class ContractProfile:
    """Tuple of per-agent local obligations, also called a SafeLTL contract."""

    profile_id: str
    formulas: dict[str, str]
    active_candidates: dict[str, tuple[str, ...]]

    def formula_for(self, agent_id: str) -> str:
        return self.formulas.get(agent_id, "t") or "t"


@dataclass(frozen=True)
class LocalObligationShieldTemplate(Generic[EnvState]):
    automaton: DeterministicSafetyAutomaton
    transitions: dict[ProductState[EnvState], dict[int, frozenset[ProductState[EnvState]]]]
    winning_region: frozenset[ProductState[EnvState]]


@dataclass(frozen=True)
class AssumeGuaranteeProductState(Generic[EnvState]):
    env_state: EnvState
    obligation_states: tuple[int, ...]


@dataclass(frozen=True)
class AssumeGuaranteeShieldTemplate(Generic[EnvState]):
    transitions: dict[
        AssumeGuaranteeProductState[EnvState],
        dict[tuple[int, ...], frozenset[AssumeGuaranteeProductState[EnvState]]],
    ]
    winning_region: frozenset[AssumeGuaranteeProductState[EnvState]]
    allowed_actions: dict[
        AssumeGuaranteeProductState[EnvState],
        dict[str, tuple[int, ...]],
    ]
    initial_state: AssumeGuaranteeProductState[EnvState]
    initial_states: tuple[AssumeGuaranteeProductState[EnvState], ...] = field(
        default_factory=tuple
    )
    max_states: int | None = None


class AssumeGuaranteeLocalShield(Generic[EnvState]):
    def __init__(
        self,
        *,
        agent_id: str,
        automaton: DeterministicSafetyAutomaton,
        template: AssumeGuaranteeShieldTemplate[EnvState],
        rng: np.random.Generator | None = None,
    ) -> None:
        self.agent_id = str(agent_id)
        self.automaton = automaton
        self.template = template
        self.rng = rng or np.random.default_rng()

    def contains(self, state: AssumeGuaranteeProductState[EnvState]) -> bool:
        return state in self.template.winning_region

    def safe_local_actions(
        self,
        state: AssumeGuaranteeProductState[EnvState],
    ) -> tuple[int, ...]:
        if state not in self.template.transitions:
            raise KeyError(f"Unknown assume-guarantee product state: {state!r}")
        if state not in self.template.winning_region:
            return tuple()
        return tuple(self.template.allowed_actions[state][self.agent_id])

    def repair(
        self,
        proposed_local_action: int,
        state: AssumeGuaranteeProductState[EnvState],
    ) -> int:
        safe_actions = self.safe_local_actions(state)
        if not safe_actions:
            raise UnsafeInitialStateError(f"No safe local action available at {state!r}.")
        if int(proposed_local_action) in safe_actions:
            return int(proposed_local_action)
        return int(safe_actions[int(self.rng.integers(0, len(safe_actions)))])


@dataclass(frozen=True)
class CertifiedContractProfile(Generic[EnvState]):
    """Contract profile together with its certificate artifacts and masks."""

    profile: ContractProfile
    global_formula: str
    profile_formula: str
    global_automaton: DeterministicSafetyAutomaton
    automata: dict[str, DeterministicSafetyAutomaton]
    templates: dict[
        str,
        LocalObligationShieldTemplate[EnvState] | AssumeGuaranteeShieldTemplate[EnvState],
    ]
    permissiveness: float
    semantics: str = "assume_guarantee"


@dataclass(frozen=True, init=False)
class ContractLibrary(Generic[EnvState]):
    """Finite ordered library of certified contract profiles available to learning."""

    candidates: tuple[LocalObligationCandidate, ...]
    initial_profile: CertifiedContractProfile[EnvState]
    certified_profiles: tuple[CertifiedContractProfile[EnvState], ...]
    certification_trace: tuple[dict[str, Any], ...]
    candidate_scopes: dict[str, tuple[str, ...]]

    def __init__(
        self,
        candidates: Sequence[LocalObligationCandidate],
        initial_profile: CertifiedContractProfile[EnvState],
        certified_profiles: Sequence[CertifiedContractProfile[EnvState]],
        certification_trace: Sequence[Mapping[str, Any]] | None = None,
        candidate_scopes: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        object.__setattr__(self, "candidates", tuple(candidates))
        object.__setattr__(self, "initial_profile", initial_profile)
        object.__setattr__(self, "certified_profiles", tuple(certified_profiles))
        object.__setattr__(
            self,
            "certification_trace",
            tuple(dict(event) for event in (certification_trace or ())),
        )
        object.__setattr__(
            self,
            "candidate_scopes",
            {
                str(agent_id): tuple(candidate_ids)
                for agent_id, candidate_ids in (candidate_scopes or {}).items()
            },
        )


@dataclass(frozen=True)
class ContractSynthesisConfig:
    """Configuration for bounded local-obligation search and certification."""

    depth: int = 1
    max_candidates: int = 64
    max_profiles: int = 64
    max_active_per_agent: int = 1
    max_atomic_props: int = 16
    max_states: int | None = None
    reuse_certification_caches: bool = True
    cache_certification_successors: bool = True
    max_refinement_steps: int = 8
    include_weak_until: bool = False
    print_candidates: bool = False
    candidate_labels: tuple[str, ...] | None = None
    use_model_local_alphabet: bool = True
    use_temporal_form_heuristic: bool = True
    use_model_seed_formulas: bool = False
    prune_equivalent_candidates: bool = True
    prune_equivalent_profiles: bool = True


@dataclass(frozen=True)
class ContractCandidateEnumeration:
    """Generated local-obligation candidates and their assignable agent scopes."""

    candidates: tuple[LocalObligationCandidate, ...]
    candidate_scopes: dict[str, tuple[str, ...]]
    local_alphabet_by_agent: dict[str, tuple[str, ...]] | None
    possible_labels: tuple[str, ...]
    pruning_trace: tuple[dict[str, Any], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class VotingConfig:
    warmup_episodes: int = 0
    dwell_episodes: int = 10
    bandit_discount: float = 0.95
    bandit_exploration_coef: float = 1.0


@dataclass(frozen=True)
class BanditSelectionResult:
    winner_profile_id: str
    arm_means: dict[str, float]
    arm_counts: dict[str, float]
    arm_ucb_scores: dict[str, float]
    selection_margin: float


class SpotLTLHelper:
    def __init__(
        self,
        *,
        ltlfilt_cmd: str = "ltlfilt",
        ltl2tgba_cmd: str = "ltl2tgba",
        autfilt_cmd: str = "autfilt",
        which: Any | None = None,
        run_command: Any | None = None,
    ) -> None:
        self.ltlfilt_cmd = ltlfilt_cmd
        self.ltl2tgba_cmd = ltl2tgba_cmd
        self.autfilt_cmd = autfilt_cmd
        self.which = which or shutil.which
        self.run_command = run_command

    def available(self) -> bool:
        try:
            self.require_available()
        except RuntimeError:
            return False
        return True

    def require_available(self) -> None:
        require_spot_cli(
            which=self.which,
            commands=(self.ltlfilt_cmd, self.ltl2tgba_cmd, self.autfilt_cmd),
        )

    def _run(self, command: list[str], *, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
        if self.run_command is not None:
            stdout = self.run_command(command)
            return subprocess.CompletedProcess(command, 0, stdout=str(stdout), stderr="")
        return subprocess.run(
            command,
            input=input_text,
            capture_output=True,
            text=True,
            check=False,
        )

    def simplify(self, formula: str) -> str:
        self.require_available()
        completed = self._run(
            [
                self.ltlfilt_cmd,
                "--simplify=3",
                "--full-parentheses",
                "--format=%f",
                "-f",
                formula,
            ]
        )
        simplified = completed.stdout.strip()
        return simplified if completed.returncode in {0, 1} and simplified else formula

    def is_safety(self, formula: str) -> bool:
        self.require_available()
        completed = self._run(
            [self.ltlfilt_cmd, "--count", "--safety", "-f", formula]
        )
        return completed.stdout.strip() == "1"

    def formula_implies(self, left: str, right: str) -> bool:
        self.require_available()
        if right in {"t", "true"}:
            return True
        completed = self._run(
            [self.ltlfilt_cmd, "--count", f"--imply={right}", "-f", left]
        )
        return completed.stdout.strip() == "1"

    def counterexample_word(self, left: str, right: str) -> str | None:
        self.require_available()
        first = subprocess.Popen(
            [self.ltl2tgba_cmd, "-f", f"({left}) & !({right})"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        second = subprocess.run(
            [self.autfilt_cmd, "--format=%w"],
            stdin=first.stdout,
            capture_output=True,
            text=True,
            check=False,
        )
        if first.stdout is not None:
            first.stdout.close()
        first.wait()
        word = second.stdout.strip()
        return word or None

    def rejects_word(self, formula: str, word: str | None) -> bool:
        if not word:
            return False
        self.require_available()
        completed = self._run(
            [self.ltlfilt_cmd, "--count", f"--reject-word={word}", "-f", formula]
        )
        return completed.stdout.strip() == "1"


class _CachedSpotLTLHelper:
    def __init__(self, helper: SpotLTLHelper) -> None:
        self._helper = helper
        self._simplify_cache: dict[str, str] = {}
        self._safety_cache: dict[str, bool] = {}
        self._implication_cache: dict[tuple[str, str], bool] = {}
        self._counterexample_cache: dict[tuple[str, str], str | None] = {}
        self._rejects_word_cache: dict[tuple[str, str | None], bool] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._helper, name)

    def simplify(self, formula: str) -> str:
        formula = str(formula)
        if formula not in self._simplify_cache:
            self._simplify_cache[formula] = self._helper.simplify(formula)
        return self._simplify_cache[formula]

    def is_safety(self, formula: str) -> bool:
        formula = str(formula)
        if formula not in self._safety_cache:
            self._safety_cache[formula] = bool(self._helper.is_safety(formula))
        return self._safety_cache[formula]

    def formula_implies(self, left: str, right: str) -> bool:
        key = (str(left), str(right))
        if key not in self._implication_cache:
            self._implication_cache[key] = bool(
                self._helper.formula_implies(key[0], key[1])
            )
        return self._implication_cache[key]

    def counterexample_word(self, left: str, right: str) -> str | None:
        key = (str(left), str(right))
        if key not in self._counterexample_cache:
            self._counterexample_cache[key] = self._helper.counterexample_word(
                key[0],
                key[1],
            )
        return self._counterexample_cache[key]

    def rejects_word(self, formula: str, word: str | None) -> bool:
        key = (str(formula), word)
        if key not in self._rejects_word_cache:
            self._rejects_word_cache[key] = bool(
                self._helper.rejects_word(key[0], key[1])
            )
        return self._rejects_word_cache[key]


class _CachedAutomatonBackend:
    def __init__(self, backend: SpotAutomatonBackend) -> None:
        self._backend = backend
        self._compile_cache: dict[
            tuple[str, tuple[str, ...]],
            DeterministicSafetyAutomaton,
        ] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._backend, name)

    def compile(
        self,
        formula: str,
        atomic_props: Sequence[str],
    ) -> DeterministicSafetyAutomaton:
        key = (str(formula), tuple(str(prop) for prop in atomic_props))
        if key not in self._compile_cache:
            self._compile_cache[key] = self._backend.compile(key[0], key[1])
        return self._compile_cache[key]


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


def extract_atomic_props(formula: str) -> tuple[str, ...]:
    if "{agent}" in formula:
        raise ValueError(
            "Formula templates must be specialized before atomic-proposition extraction."
        )
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", formula)
    return tuple(sorted({token for token in tokens if token not in _LTL_KEYWORDS}))


def formula_size(formula: str) -> int:
    return len(re.findall(r"[A-Za-z_][A-Za-z0-9_]*|[!&|()]", formula))


def _is_fast_safety_fragment(formula: str) -> bool:
    """Return true for formulas the bounded generator builds as plain safety."""
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", str(formula))
    return not any(token in {"F", "M", "R", "U", "W"} for token in tokens)


def conjunction_formula(formulas: Sequence[str]) -> str:
    normalized = [
        f"({formula})"
        for formula in formulas
        if formula and formula not in {"t", "true"}
    ]
    return " & ".join(normalized) if normalized else "t"


def profile_conjunction(profile: ContractProfile) -> str:
    return conjunction_formula(tuple(profile.formulas.values()))


def _ordered_candidate_labels(
    *,
    possible_labels: Sequence[str],
    global_formula: str,
    config: ContractSynthesisConfig,
) -> tuple[str, ...]:
    possible_label_set = {str(label) for label in possible_labels}
    raw_labels = config.candidate_labels or tuple(possible_labels)
    labels = tuple(
        dict.fromkeys(
            str(label)
            for label in raw_labels
            if str(label) in possible_label_set
        )
    )
    global_props = extract_atomic_props(global_formula)
    ordered = [
        prop
        for prop in global_props
        if prop in labels
    ]
    ordered.extend(label for label in labels if label not in set(ordered))
    return tuple(ordered[: max(int(config.max_atomic_props), 0)])


def _normalize_contract_local_alphabet_by_agent(
    *,
    agent_ids: Sequence[str],
    possible_labels: Sequence[str],
    local_alphabet_by_agent: Mapping[str, Sequence[str]] | None,
) -> dict[str, tuple[str, ...]] | None:
    if local_alphabet_by_agent is None:
        return None

    expected_agent_ids = tuple(str(agent_id) for agent_id in agent_ids)
    provided = {
        str(agent_id): tuple(str(label) for label in labels)
        for agent_id, labels in local_alphabet_by_agent.items()
    }
    expected_set = set(expected_agent_ids)
    provided_set = set(provided)
    missing = tuple(agent_id for agent_id in expected_agent_ids if agent_id not in provided)
    unknown = tuple(sorted(provided_set - expected_set))
    if missing or unknown:
        details = []
        if missing:
            details.append("missing keys: " + ", ".join(missing))
        if unknown:
            details.append("unknown keys: " + ", ".join(unknown))
        raise ContractError(
            "contract_local_alphabet_by_agent must provide exactly one entry per agent ("
            + "; ".join(details)
            + ")."
        )

    possible_label_set = {str(label) for label in possible_labels}
    filtered: dict[str, tuple[str, ...]] = {}
    missing_labels: dict[str, tuple[str, ...]] = {}
    for agent_id in expected_agent_ids:
        invalid = tuple(
            dict.fromkeys(
                label
                for label in provided[agent_id]
                if label not in possible_label_set
            )
        )
        if invalid:
            missing_labels[agent_id] = invalid
        labels = tuple(
            dict.fromkeys(
                label
                for label in provided[agent_id]
                if label in possible_label_set
            )
        )
        filtered[agent_id] = labels
    if missing_labels:
        details = "; ".join(
            f"{agent_id}: {', '.join(labels)}"
            for agent_id, labels in missing_labels.items()
        )
        raise ContractError(
            "contract_local_alphabet_by_agent references propositions that the safety model cannot emit: "
            + details
        )
    return filtered


def _model_contract_local_alphabet_by_agent(
    model: AbstractionModel[EnvState],
) -> Mapping[str, Sequence[str]] | None:
    local_alphabet = getattr(model, "contract_local_alphabet_by_agent", None)
    if callable(local_alphabet):
        return local_alphabet()

    return None


def _resolve_contract_local_alphabet_by_agent(
    *,
    model: AbstractionModel[EnvState],
    agent_ids: Sequence[str],
    possible_labels: Sequence[str],
    config: ContractSynthesisConfig,
) -> dict[str, tuple[str, ...]] | None:
    if config.candidate_labels is not None:
        return None
    if not config.use_model_local_alphabet:
        return None

    local_alphabet_by_agent = _model_contract_local_alphabet_by_agent(model)
    if local_alphabet_by_agent is None:
        return None
    return _normalize_contract_local_alphabet_by_agent(
        agent_ids=agent_ids,
        possible_labels=possible_labels,
        local_alphabet_by_agent=local_alphabet_by_agent,
    )


def _candidate_scopes_by_agent(
    *,
    agent_ids: Sequence[str],
    candidates: Sequence[LocalObligationCandidate],
    local_alphabet_by_agent: Mapping[str, Sequence[str]] | None,
) -> dict[str, tuple[str, ...]]:
    if local_alphabet_by_agent is None:
        candidate_ids = tuple(candidate.candidate_id for candidate in candidates)
        return {str(agent_id): candidate_ids for agent_id in agent_ids}

    allowed_props_by_agent = {
        str(agent_id): {str(label) for label in labels}
        for agent_id, labels in local_alphabet_by_agent.items()
    }
    scopes: dict[str, list[str]] = {str(agent_id): [] for agent_id in agent_ids}
    for candidate in candidates:
        candidate_props = set(extract_atomic_props(candidate.formula))
        for agent_id in agent_ids:
            if candidate_props <= allowed_props_by_agent[str(agent_id)]:
                scopes[str(agent_id)].append(candidate.candidate_id)
    return {
        str(agent_id): tuple(scopes[str(agent_id)])
        for agent_id in agent_ids
    }


def _model_contract_seed_formulas_by_agent(
    model: AbstractionModel[EnvState],
    *,
    global_formula: str,
    possible_labels: Sequence[str],
    helper: SpotLTLHelper,
) -> tuple[LocalObligationCandidate, ...]:
    seed_resolver = getattr(model, "contract_seed_formulas_by_agent", None)
    if not callable(seed_resolver):
        return tuple()
    raw_seed_formulas = seed_resolver(global_formula=global_formula)
    if not raw_seed_formulas:
        return tuple()
    possible_label_set = {str(label) for label in possible_labels}
    candidates: list[LocalObligationCandidate] = []
    seen: set[str] = set()
    for raw_formula in raw_seed_formulas.values():
        formula = helper.simplify(str(raw_formula))
        if formula in seen or formula in _FALSE_FORMULAS:
            continue
        if not set(extract_atomic_props(formula)) <= possible_label_set:
            continue
        if not helper.is_safety(formula):
            continue
        seen.add(formula)
        candidates.append(
            LocalObligationCandidate(
                candidate_id="",
                formula=formula,
                depth=0,
                size=formula_size(formula),
            )
        )
    return tuple(candidates)


def _merge_seed_local_obligation_candidates(
    candidates: Sequence[LocalObligationCandidate],
    seeds: Sequence[LocalObligationCandidate],
) -> tuple[LocalObligationCandidate, ...]:
    if not seeds:
        return tuple(candidates)
    merged: list[LocalObligationCandidate] = []
    seen: set[str] = set()
    for seed in seeds:
        if seed.formula in seen:
            continue
        seen.add(seed.formula)
        merged.append(
            replace(
                seed,
                candidate_id=f"a{len(merged):03d}",
            )
        )
    for candidate in candidates:
        if candidate.formula in seen:
            continue
        seen.add(candidate.formula)
        merged.append(
            replace(
                candidate,
                candidate_id=f"a{len(merged):03d}",
            )
        )
    return tuple(merged)


def _enumerate_scoped_local_obligation_candidates(
    *,
    possible_labels: Sequence[str],
    global_formula: str,
    config: ContractSynthesisConfig,
    local_alphabet_by_agent: Mapping[str, Sequence[str]],
    helper: SpotLTLHelper,
) -> tuple[LocalObligationCandidate, ...]:
    """Enumerate only formulas that fit at least one model-local scope."""
    max_candidates = int(config.max_candidates)
    if max_candidates <= 0:
        return tuple()

    scopes = tuple(local_alphabet_by_agent.values())
    per_scope_limit = max(1, (max_candidates + len(scopes) - 1) // len(scopes))
    scoped_candidate_lists: list[tuple[LocalObligationCandidate, ...]] = []
    for labels in scopes:
        scoped_labels = tuple(dict.fromkeys(str(label) for label in labels))
        scoped_label_set = set(scoped_labels)
        scoped_config = replace(
            config,
            candidate_labels=scoped_labels,
            max_candidates=per_scope_limit,
        )
        scoped_candidates: list[LocalObligationCandidate] = []
        for candidate in enumerate_local_obligation_candidates(
            possible_labels=possible_labels,
            global_formula=global_formula,
            config=scoped_config,
            helper=helper,
        ):
            if not set(extract_atomic_props(candidate.formula)) <= scoped_label_set:
                continue
            scoped_candidates.append(candidate)
        scoped_candidate_lists.append(tuple(scoped_candidates))

    merged: list[LocalObligationCandidate] = []
    seen_formulas: set[str] = set()
    max_scope_length = max(
        (len(candidates) for candidates in scoped_candidate_lists),
        default=0,
    )
    for index in range(max_scope_length):
        for scoped_candidates in scoped_candidate_lists:
            if index >= len(scoped_candidates):
                continue
            candidate = scoped_candidates[index]
            if candidate.formula in seen_formulas:
                continue
            seen_formulas.add(candidate.formula)
            merged.append(
                replace(
                    candidate,
                    candidate_id=f"a{len(merged):03d}",
                )
            )
            if len(merged) >= max_candidates:
                return tuple(merged)
    return tuple(merged)


def enumerate_local_obligation_candidates(
    *,
    possible_labels: Sequence[str],
    global_formula: str,
    config: ContractSynthesisConfig,
    helper: SpotLTLHelper | None = None,
) -> tuple[LocalObligationCandidate, ...]:
    helper = helper or SpotLTLHelper()
    depth = max(int(config.depth), 0)
    labels = _ordered_candidate_labels(
        possible_labels=possible_labels,
        global_formula=global_formula,
        config=config,
    )
    formulas_by_depth: dict[int, list[str]] = {
        0: [label for prop in labels for label in (prop, f"!{prop}")]
    }
    seen_formulas: set[str] = set()
    candidates: list[LocalObligationCandidate] = []
    possible_label_set = {str(label) for label in possible_labels}

    def _maybe_add(raw_formula: str, raw_depth: int) -> None:
        if len(candidates) >= int(config.max_candidates):
            return
        if _is_fast_safety_fragment(raw_formula):
            simplified = str(raw_formula)
            is_safety = True
        else:
            simplified = helper.simplify(raw_formula)
            is_safety = helper.is_safety(simplified)
        if simplified in _FALSE_FORMULAS:
            return
        if simplified in seen_formulas:
            return
        if not is_safety:
            return
        seen_formulas.add(simplified)
        candidates.append(
            LocalObligationCandidate(
                candidate_id=f"a{len(candidates):03d}",
                formula=simplified,
                depth=int(raw_depth),
                size=formula_size(simplified),
            )
        )

    if all(prop in possible_label_set for prop in extract_atomic_props(global_formula)):
        _maybe_add(global_formula, 0)

    for raw_formula in formulas_by_depth[0]:
        _maybe_add(raw_formula, 0)

    for current_depth in range(1, depth + 1):
        if len(candidates) >= int(config.max_candidates):
            break
        previous = [
            formula
            for nested_depth in range(current_depth)
            for formula in formulas_by_depth.get(nested_depth, ())
        ]
        generated: list[str] = []
        for formula in previous:
            generated.append(f"X({formula})")
            generated.append(f"G({formula})")
        for left, right in combinations(previous, 2):
            generated.append(f"({left}) & ({right})")
            generated.append(f"({left}) | ({right})")
            if config.include_weak_until:
                generated.append(f"({left}) W ({right})")
        formulas_by_depth[current_depth] = generated
        for raw_formula in generated:
            _maybe_add(raw_formula, current_depth)
            if len(candidates) >= int(config.max_candidates):
                break

    return tuple(candidates)


def enumerate_contract_local_obligation_candidates(
    model: AbstractionModel[EnvState],
    *,
    global_formula: str,
    config: ContractSynthesisConfig | None = None,
    helper: SpotLTLHelper | None = None,
    backend: SpotAutomatonBackend | None = None,
) -> ContractCandidateEnumeration:
    """Enumerate generated contract obligations using the library-builder scopes."""
    config = config or ContractSynthesisConfig()
    helper = helper or SpotLTLHelper()
    backend = backend or SpotAutomatonBackend()
    if not hasattr(model, "possible_labels"):
        raise ContractError("Safety model must define possible_labels().")
    possible_labels = tuple(str(label) for label in model.possible_labels())
    possible_label_set = set(possible_labels)
    agent_ids = tuple(str(agent_id) for agent_id in model.agent_ids)
    missing_global_props = tuple(
        prop
        for prop in extract_atomic_props(global_formula)
        if prop not in possible_label_set
    )
    if missing_global_props:
        raise ContractError(
            "Global formula references propositions that the safety model cannot emit: "
            + ", ".join(missing_global_props)
        )
    local_alphabet_by_agent = _resolve_contract_local_alphabet_by_agent(
        model=model,
        agent_ids=agent_ids,
        possible_labels=possible_labels,
        config=config,
    )
    if local_alphabet_by_agent is None:
        candidates = enumerate_local_obligation_candidates(
            possible_labels=possible_labels,
            global_formula=global_formula,
            config=config,
            helper=helper,
        )
    else:
        candidates = _enumerate_scoped_local_obligation_candidates(
            possible_labels=possible_labels,
            global_formula=global_formula,
            config=config,
            local_alphabet_by_agent=local_alphabet_by_agent,
            helper=helper,
        )
    if config.use_model_seed_formulas:
        candidates = _merge_seed_local_obligation_candidates(
            candidates,
            _model_contract_seed_formulas_by_agent(
                model,
                global_formula=global_formula,
                possible_labels=possible_labels,
                helper=helper,
            ),
        )
    pruning_trace: tuple[dict[str, Any], ...] = tuple()
    if config.prune_equivalent_candidates:
        candidates, pruning_trace = _prune_equivalent_local_obligation_candidates(
            candidates,
            backend=backend,
        )
    candidate_scopes = _candidate_scopes_by_agent(
        agent_ids=agent_ids,
        candidates=candidates,
        local_alphabet_by_agent=local_alphabet_by_agent,
    )
    return ContractCandidateEnumeration(
        candidates=tuple(candidates),
        candidate_scopes=candidate_scopes,
        local_alphabet_by_agent=local_alphabet_by_agent,
        possible_labels=possible_labels,
        pruning_trace=pruning_trace,
    )


def _compile_formula(
    formula: str,
    *,
    backend: SpotAutomatonBackend | None = None,
) -> DeterministicSafetyAutomaton:
    if formula in {"t", "true", "1"}:
        return DeterministicSafetyAutomaton(
            atomic_props=tuple(),
            states=frozenset({0}),
            initial_state=0,
            safe_states=frozenset({0}),
            transition_map={(0, frozenset()): 0},
        )
    backend = backend or SpotAutomatonBackend()
    return backend.compile(formula, extract_atomic_props(formula))


def _automaton_behavior_signature(
    automaton: DeterministicSafetyAutomaton,
) -> tuple[tuple[str, ...], tuple[tuple[bool, tuple[int, ...]], ...]]:
    """Canonicalize reachable deterministic monitor behavior from the initial state."""
    atomic_props = tuple(sorted(automaton.atomic_props))
    valuations = tuple(
        frozenset(
            prop
            for prop, enabled in zip(atomic_props, mask, strict=True)
            if enabled
        )
        for mask in product((False, True), repeat=len(atomic_props))
    )
    canonical_ids: dict[int, int] = {automaton.initial_state: 0}
    queue: deque[int] = deque([automaton.initial_state])
    state_rows: list[tuple[bool, tuple[int, ...]]] = []

    while queue:
        state = queue.popleft()
        targets: list[int] = []
        for valuation in valuations:
            target = automaton.transition(state, valuation)
            if target not in canonical_ids:
                canonical_ids[target] = len(canonical_ids)
                queue.append(target)
            targets.append(canonical_ids[target])
        state_rows.append((state in automaton.safe_states, tuple(targets)))

    return atomic_props, tuple(state_rows)


def _prune_equivalent_local_obligation_candidates(
    candidates: Sequence[LocalObligationCandidate],
    *,
    backend: SpotAutomatonBackend,
) -> tuple[tuple[LocalObligationCandidate, ...], tuple[dict[str, Any], ...]]:
    retained: list[LocalObligationCandidate] = []
    trace: list[dict[str, Any]] = []
    seen: dict[
        tuple[tuple[str, ...], tuple[tuple[bool, tuple[int, ...]], ...]],
        LocalObligationCandidate,
    ] = {}

    for candidate in candidates:
        try:
            signature = _automaton_behavior_signature(
                _compile_formula(candidate.formula, backend=backend)
            )
        except Exception:
            signature = None

        if signature is not None and signature in seen:
            kept = seen[signature]
            trace.append(
                {
                    "event": "candidate_pruned",
                    "reason": "equivalent_automaton",
                    "candidate_id": candidate.candidate_id,
                    "candidate_formula": candidate.formula,
                    "kept_candidate_id": kept.candidate_id,
                    "kept_candidate_formula": kept.formula,
                }
            )
            continue

        retained_candidate = replace(
            candidate,
            candidate_id=f"a{len(retained):03d}",
        )
        retained.append(retained_candidate)
        if signature is not None:
            seen[signature] = retained_candidate

    return tuple(retained), tuple(trace)


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
