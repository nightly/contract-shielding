from __future__ import annotations

from collections import deque
from dataclasses import replace
from itertools import combinations, product
import re
import shutil
import subprocess
from typing import Any, Mapping, Sequence

from .automaton import SpotAutomatonBackend, require_spot_cli
from .core import AbstractionModel, DeterministicSafetyAutomaton, EnvState
from ._contract_types import (
    ContractCandidateEnumeration,
    ContractError,
    ContractProfile,
    ContractSynthesisConfig,
    LocalObligationCandidate,
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

    def _run(
        self,
        command: list[str],
        *,
        input_text: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
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
    ordered = [prop for prop in global_props if prop in labels]
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
