from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, Mapping, Sequence

import numpy as np

from .core import (
    DeterministicSafetyAutomaton,
    EnvState,
    ProductState,
    UnsafeInitialStateError,
)


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
