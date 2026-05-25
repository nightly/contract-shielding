from __future__ import annotations

from itertools import product
import os
import warnings
from typing import Any, Literal, Mapping

import numpy as np
from pettingzoo import ParallelEnv

from .contracts import (
    AssumeGuaranteeLocalShield,
    AssumeGuaranteeProductState,
    AssumeGuaranteeShieldTemplate,
    CertifiedContractProfile,
    ContractLibrary,
    initial_assume_guarantee_product_state,
    synthesize_assume_guarantee_profile_shield,
)
from .core import (
    DeterministicSafetyAutomaton,
    JointAction,
    JointShield,
    Label,
    LocalShield,
    ProductState,
    AbstractionModel,
    initial_product_state,
    safety_projection,
)


AgentAutomata = Mapping[str, DeterministicSafetyAutomaton]
AgentShields = Mapping[str, LocalShield]
JointActionTable = tuple[JointAction, ...]
ContractAgentShields = Mapping[
    str,
    LocalShield | AssumeGuaranteeLocalShield,
]
ViolationPolicy = Literal["raise", "warn", "record"]


def _validate_violation_policy(policy: str) -> ViolationPolicy:
    if policy not in {"raise", "warn", "record"}:
        raise ValueError(
            "violation_policy must be one of 'raise', 'warn', or 'record', "
            f"got {policy!r}."
        )
    return policy  # type: ignore[return-value]


def _contract_undercoverage_detail_log() -> bool:
    return os.environ.get("CSH_CONTRACT_UNDERCOVERAGE_DETAIL_LOG", "").lower() in {
        "1",
        "true",
        "yes",
    }


def _allowed_actions(mask: np.ndarray | None) -> list[int] | None:
    if mask is None:
        return None
    return (
        np.flatnonzero(np.asarray(mask, dtype=bool).reshape(-1))
        .astype(int)
        .tolist()
    )


def _allowed_actions_by_agent(
    masks: Mapping[str, np.ndarray],
) -> dict[str, list[int] | None]:
    return {
        agent: _allowed_actions(mask)
        for agent, mask in masks.items()
    }


def _coerce_action_mask(
    mask: np.ndarray,
    *,
    action_dim: int,
    agent: str,
    source: str,
) -> np.ndarray:
    coerced = np.asarray(mask, dtype=bool).reshape(-1)
    expected = (int(action_dim),)
    if coerced.shape != expected:
        raise ValueError(
            f"{source} action mask for {agent!r} has shape {coerced.shape}, "
            f"expected {expected}."
        )
    return coerced


def _action_allowed(mask: np.ndarray, action: int) -> bool:
    return 0 <= int(action) < int(mask.shape[0]) and bool(mask[int(action)])


def _shield_mask_violation_message(
    *,
    wrapper_name: str,
    agent: str,
    proposed_action: int,
    cached_mask: np.ndarray,
    recomputed_mask: np.ndarray,
    current_product_state: Any,
    product_state_label: str = "product state",
) -> str:
    rejected_by: list[str] = []
    if not _action_allowed(cached_mask, proposed_action):
        rejected_by.append("cached emitted mask")
    if not _action_allowed(recomputed_mask, proposed_action):
        rejected_by.append("freshly recomputed mask")
    return (
        f"SHIELD MASK VIOLATION: {wrapper_name} rejected action before base "
        f"env.step for {agent!r}. Proposed action {int(proposed_action)}; "
        f"cached allowed actions {_allowed_actions(cached_mask)}; recomputed "
        f"allowed actions {_allowed_actions(recomputed_mask)}; rejected by "
        f"{', '.join(rejected_by)}. Current {product_state_label}: "
        f"{current_product_state!r}."
    )


def _execute_validated_shield_action(
    *,
    wrapper_name: str,
    agent: str,
    proposed_action: int,
    current_product_state: Any,
    cached_mask: np.ndarray,
    recomputed_mask: np.ndarray,
    product_state_label: str = "product state",
) -> tuple[int, bool, bool]:
    action = int(proposed_action)
    if (
        _action_allowed(cached_mask, action)
        and _action_allowed(recomputed_mask, action)
    ):
        return action, False, False

    raise ValueError(
        _shield_mask_violation_message(
            wrapper_name=wrapper_name,
            agent=agent,
            proposed_action=action,
            cached_mask=cached_mask,
            recomputed_mask=recomputed_mask,
            current_product_state=current_product_state,
            product_state_label=product_state_label,
        )
    )


def _shield_safety_violation_message(
    *,
    agent: str,
    executed_action: int,
    proposed_action: int,
    cached_action_mask: np.ndarray | None,
    recomputed_action_mask: np.ndarray | None,
    current_product_state: ProductState,
    next_product_state: ProductState,
    observed_label: Label,
    detailed: bool = False,
) -> str:
    if detailed:
        return (
            "SAFETY VIOLATION: ShieldedParallelEnv entered a rejecting monitor "
            f"state for {agent!r} after executing local action {executed_action} "
            f"(proposed {proposed_action}) from {current_product_state!r}. "
            f"Monitor transition {current_product_state.automaton_state} -> "
            f"{next_product_state.automaton_state}. Cached allowed actions "
            f"{_allowed_actions(cached_action_mask)}; recomputed allowed actions "
            f"{_allowed_actions(recomputed_action_mask)}. Observed label "
            f"{sorted(observed_label)!r}; next product state "
            f"{next_product_state!r}. This indicates the shield, safety "
            "abstraction, or environment dynamics are inconsistent."
        )
    return (
        "SAFETY VIOLATION: ShieldedParallelEnv entered a rejecting monitor "
        f"state for {agent!r} after executing local action {executed_action} "
        f"(proposed {proposed_action}); monitor "
        f"{current_product_state.automaton_state} -> "
        f"{next_product_state.automaton_state}; observed {len(observed_label)} "
        "atomic propositions. Set debug_safety_violations=True for full "
        "product-state diagnostics."
    )


def _safe_action_mask(
    shield: LocalShield | AssumeGuaranteeLocalShield,
    product_state: Any,
    action_dim: int,
    *,
    strict: bool = True,
) -> np.ndarray:
    try:
        safe_actions = shield.safe_local_actions(product_state)
    except KeyError:
        if strict:
            raise
        safe_actions = ()
    if not safe_actions:
        if not strict:
            return np.ones(int(action_dim), dtype=bool)
        raise RuntimeError(f"No safe local action available at {product_state!r}.")

    mask = np.zeros(int(action_dim), dtype=bool)
    for action in safe_actions:
        action_idx = int(action)
        if 0 <= action_idx < int(action_dim):
            mask[action_idx] = True
    if not bool(mask.any()):
        raise RuntimeError(f"Shield produced no in-range safe action at {product_state!r}.")
    return mask


def _coerce_joint_action_mask(
    mask: np.ndarray,
    *,
    joint_action_dim: int,
    source: str,
) -> np.ndarray:
    coerced = np.asarray(mask, dtype=bool).reshape(-1)
    expected = (int(joint_action_dim),)
    if coerced.shape != expected:
        raise ValueError(
            f"{source} joint_action_mask has shape {coerced.shape}, "
            f"expected {expected}."
        )
    return coerced


def _joint_shield_mask_violation_message(
    *,
    wrapper_name: str,
    proposed_joint_action: JointAction,
    proposed_joint_action_index: int | None,
    cached_mask: np.ndarray,
    recomputed_mask: np.ndarray,
    current_product_state: Any,
) -> str:
    rejected_by: list[str] = []
    if proposed_joint_action_index is None:
        rejected_by.append("joint action table")
    elif not _action_allowed(cached_mask, proposed_joint_action_index):
        rejected_by.append("cached emitted mask")
    if proposed_joint_action_index is None or not _action_allowed(
        recomputed_mask,
        proposed_joint_action_index,
    ):
        rejected_by.append("freshly recomputed mask")
    return (
        f"SHIELD MASK VIOLATION: {wrapper_name} rejected joint action before "
        f"base env.step. Proposed joint action {proposed_joint_action!r} "
        f"(index {proposed_joint_action_index!r}); cached allowed joint action "
        f"indices {_allowed_actions(cached_mask)}; recomputed allowed joint "
        f"action indices {_allowed_actions(recomputed_mask)}; rejected by "
        f"{', '.join(rejected_by)}. Current product state: "
        f"{current_product_state!r}."
    )


def _joint_shield_safety_violation_message(
    *,
    executed_joint_action: JointAction,
    proposed_joint_action: JointAction,
    cached_joint_action_mask: np.ndarray | None,
    recomputed_joint_action_mask: np.ndarray | None,
    current_product_state: ProductState,
    next_product_state: ProductState,
    observed_label: Label,
    detailed: bool = False,
) -> str:
    if detailed:
        return (
            "SAFETY VIOLATION: JointShieldedParallelEnv entered a rejecting "
            f"monitor state after executing joint action {executed_joint_action!r} "
            f"(proposed {proposed_joint_action!r}) from {current_product_state!r}. "
            f"Monitor transition {current_product_state.automaton_state} -> "
            f"{next_product_state.automaton_state}. Cached allowed joint action "
            f"indices {_allowed_actions(cached_joint_action_mask)}; recomputed "
            f"allowed joint action indices {_allowed_actions(recomputed_joint_action_mask)}. "
            f"Observed label {sorted(observed_label)!r}; next product state "
            f"{next_product_state!r}. This indicates the joint shield, safety "
            "abstraction, or environment dynamics are inconsistent."
        )
    return (
        "SAFETY VIOLATION: JointShieldedParallelEnv entered a rejecting "
        f"monitor state after executing joint action {executed_joint_action!r} "
        f"(proposed {proposed_joint_action!r}); monitor "
        f"{current_product_state.automaton_state} -> "
        f"{next_product_state.automaton_state}; observed {len(observed_label)} "
        "atomic propositions. Set debug_safety_violations=True for full "
        "product-state diagnostics."
    )


def _augment_action_mask_infos(
    infos: dict[str, dict[str, Any]],
    agents: list[str],
    *,
    masks: Mapping[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    for agent in agents:
        agent_info = infos.setdefault(agent, {})
        mask = np.asarray(masks[agent], dtype=bool)
        existing_mask = agent_info.get("action_mask")
        if existing_mask is not None:
            existing = np.asarray(existing_mask, dtype=bool).reshape(-1)
            if existing.shape != mask.shape:
                raise ValueError(
                    f"Existing action_mask for {agent!r} has shape {existing.shape}, "
                    f"expected {mask.shape}."
                )
            mask = np.logical_and(mask, existing)
        if not bool(mask.any()):
            raise RuntimeError(f"No executable safe action available for {agent!r}.")
        agent_info["action_mask"] = mask.astype(bool, copy=True)
    return infos


def _copy_infos(infos: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        agent: dict(agent_info)
        for agent, agent_info in infos.items()
    }


def _augment_safety_infos(
    infos: dict[str, dict[str, Any]],
    agents: list[str],
    *,
    safety_violations: Mapping[str, bool],
    safety_violation_counts: Mapping[str, int],
    episode_safety_violations: Mapping[str, int],
    monitor_states: Mapping[str, int],
) -> dict[str, dict[str, Any]]:
    for agent in agents:
        agent_info = infos.setdefault(agent, {})
        agent_info["safety_violation"] = bool(safety_violations[agent])
        agent_info["safety_violation_count"] = int(safety_violation_counts[agent])
        agent_info["episode_safety_violations"] = int(episode_safety_violations[agent])
        agent_info["monitor_state"] = int(monitor_states[agent])
    return infos


def _augment_contract_infos(
    infos: dict[str, dict[str, Any]],
    agents: list[str],
    *,
    profile_index: int,
    profile_changed: bool,
    certified_profile: CertifiedContractProfile,
    obligation_violations: Mapping[str, bool],
    obligation_monitor_states: Mapping[str, int],
    vote_winner: str,
    vote_margin: int,
) -> dict[str, dict[str, Any]]:
    profile = certified_profile.profile
    profile_formulas = {
        agent: str(profile.formula_for(agent))
        for agent in agents
    }
    monitor_states = {
        agent: int(obligation_monitor_states[agent])
        for agent in agents
    }
    for agent in agents:
        agent_info = infos.setdefault(agent, {})
        agent_info["contract_profile_id"] = profile.profile_id
        agent_info["contract_profile_index"] = int(profile_index)
        agent_info["contract_profile_changed"] = bool(profile_changed)
        agent_info["contract_formula"] = str(profile.formula_for(agent))
        agent_info["contract_active_candidates"] = tuple(
            profile.active_candidates.get(agent, ())
        )
        agent_info["contract_vote_winner"] = str(vote_winner)
        agent_info["contract_vote_margin"] = int(vote_margin)
        agent_info["contract_obligation_violation"] = bool(
            obligation_violations[agent]
        )
        agent_info["contract_obligation_monitor_state"] = int(
            obligation_monitor_states[agent]
        )
        agent_info["contract_profile_formulas"] = dict(profile_formulas)
        agent_info["contract_obligation_monitor_states"] = dict(monitor_states)
        agent_info["contract_semantics"] = str(
            getattr(certified_profile, "semantics", "local")
        )
        agent_info["contract_permissiveness"] = float(
            certified_profile.permissiveness
        )
    return infos


def _augment_shared_metrics(
    infos: dict[str, dict[str, Any]],
    agents: list[str],
    *,
    metrics: Mapping[str, float],
) -> dict[str, dict[str, Any]]:
    for agent in agents:
        agent_info = infos.setdefault(agent, {})
        for key, value in metrics.items():
            agent_info.setdefault(key, float(value))
    return infos


class SafetyMonitoringParallelEnv(ParallelEnv):
    metadata = {
        "name": "safety_monitor_parallel_v0",
        "is_parallelizable": True,
    }

    def __init__(
        self,
        env: ParallelEnv,
        model: AbstractionModel,
        automata: AgentAutomata,
        *,
        warn_on_violation: bool = False,
    ) -> None:
        self.env = env
        self.model = model
        self.automata = dict(automata)
        self.warn_on_violation = warn_on_violation
        self.possible_agents = list(env.possible_agents)
        self.agent_name_mapping = dict(getattr(env, "agent_name_mapping", {}))
        self.agents: list[str] = []
        self._product_states: dict[str, ProductState] = {}
        self.total_safety_violations = {
            agent: 0
            for agent in self.possible_agents
        }
        self.episode_safety_violations = {
            agent: 0
            for agent in self.possible_agents
        }

    def observation_space(self, agent: str):
        return self.env.observation_space(agent)

    def action_space(self, agent: str):
        return self.env.action_space(agent)

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        observations, infos = self.env.reset(seed=seed, options=options)
        self.agents = list(self.env.agents)
        self.episode_safety_violations = {
            agent: 0
            for agent in self.possible_agents
        }
        env_state = safety_projection(self.model, self.model.abstract_state(self.env))
        self._product_states = {
            agent: initial_product_state(self.model, self.automata[agent], env_state)
            for agent in self.possible_agents
        }
        infos = _copy_infos(infos)
        infos = _augment_safety_infos(
            infos,
            self.possible_agents,
            safety_violations={agent: False for agent in self.possible_agents},
            safety_violation_counts=self.total_safety_violations,
            episode_safety_violations=self.episode_safety_violations,
            monitor_states={
                agent: self._product_states[agent].automaton_state
                for agent in self.possible_agents
            },
        )
        return observations, infos

    def step(self, actions: dict[str, Any]):
        if not self._product_states:
            raise RuntimeError("Call reset() before step().")

        current_product_states = dict(self._product_states)
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        self.agents = list(self.env.agents)
        env_state = safety_projection(self.model, self.model.abstract_state(self.env))
        observed_label = self.model.label(env_state)

        safety_violations: dict[str, bool] = {}
        monitor_states: dict[str, int] = {}
        violating_agents: list[str] = []

        for agent in self.possible_agents:
            current_product_state = current_product_states[agent]
            automaton = self.automata[agent]
            next_monitor_state = automaton.transition(
                current_product_state.automaton_state,
                observed_label,
            )
            self._product_states[agent] = ProductState(
                env_state=env_state,
                automaton_state=next_monitor_state,
            )
            entered_rejecting = (
                current_product_state.automaton_state in automaton.safe_states
                and next_monitor_state not in automaton.safe_states
            )
            safety_violations[agent] = entered_rejecting
            monitor_states[agent] = next_monitor_state
            if entered_rejecting:
                self.total_safety_violations[agent] += 1
                self.episode_safety_violations[agent] += 1
                violating_agents.append(agent)

        if violating_agents and self.warn_on_violation:
            warnings.warn(
                "SAFETY VIOLATION: SafetyMonitoringParallelEnv entered rejecting "
                f"monitor states for {violating_agents!r} after executing local actions "
                f"{actions!r}. Observed label {sorted(observed_label)!r}; next monitor "
                f"states {monitor_states!r}.",
                RuntimeWarning,
                stacklevel=2,
            )

        infos = _copy_infos(infos)
        infos = _augment_safety_infos(
            infos,
            self.possible_agents,
            safety_violations=safety_violations,
            safety_violation_counts=self.total_safety_violations,
            episode_safety_violations=self.episode_safety_violations,
            monitor_states=monitor_states,
        )
        return observations, rewards, terminations, truncations, infos

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def state(self):
        return self.env.state()

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def __getattr__(self, item: str):
        return getattr(self.env, item)


class SafetyMetricsParallelEnv(ParallelEnv):
    metadata = {
        "name": "safety_metrics_parallel_v0",
        "is_parallelizable": True,
    }

    def __init__(self, env: ParallelEnv) -> None:
        self.env = env
        self.possible_agents = list(env.possible_agents)
        self.agent_name_mapping = dict(getattr(env, "agent_name_mapping", {}))
        self.agents: list[str] = []

    def observation_space(self, agent: str):
        return self.env.observation_space(agent)

    def action_space(self, agent: str):
        return self.env.action_space(agent)

    def _safety_metrics(self, infos: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
        num_agents = max(len(self.possible_agents), 1)
        safety_violations = [
            float(bool(infos.get(agent, {}).get("safety_violation", False)))
            for agent in self.possible_agents
        ]
        shield_interventions = [
            float(bool(infos.get(agent, {}).get("shield_intervened", False)))
            for agent in self.possible_agents
        ]
        cumulative_safety_violations = sum(
            float(infos.get(agent, {}).get("safety_violation_count", 0.0))
            for agent in self.possible_agents
        )
        episode_safety_violations = sum(
            float(infos.get(agent, {}).get("episode_safety_violations", 0.0))
            for agent in self.possible_agents
        )
        metrics = {
            "safety_violations_mean": sum(safety_violations),
            "safety_violations_agent_fraction_mean": (
                sum(safety_violations) / float(num_agents)
            ),
            "safety_violations_cumulative": cumulative_safety_violations,
            "episode_safety_violations_mean": episode_safety_violations,
            "shield_interventions_mean": sum(shield_interventions),
            "shield_interventions_agent_fraction_mean": (
                sum(shield_interventions) / float(num_agents)
            ),
        }
        if any(
            any(
                key in infos.get(agent, {})
                for key in ("load_attempted", "load_failed", "load_successful")
            )
            for agent in self.possible_agents
        ):
            load_attempts = [
                float(bool(infos.get(agent, {}).get("load_attempted", False)))
                for agent in self.possible_agents
            ]
            load_failures = [
                float(bool(infos.get(agent, {}).get("load_failed", False)))
                for agent in self.possible_agents
            ]
            load_successes = [
                float(bool(infos.get(agent, {}).get("load_successful", False)))
                for agent in self.possible_agents
            ]
            metrics.update(
                {
                    "load_attempts_mean": sum(load_attempts),
                    "load_attempts_agent_fraction_mean": (
                        sum(load_attempts) / float(num_agents)
                    ),
                    "load_failures_mean": sum(load_failures),
                    "load_failures_agent_fraction_mean": (
                        sum(load_failures) / float(num_agents)
                    ),
                    "load_successes_mean": sum(load_successes),
                    "load_successes_agent_fraction_mean": (
                        sum(load_successes) / float(num_agents)
                    ),
                }
            )
        if any(
            "contract_profile_index" in infos.get(agent, {})
            for agent in self.possible_agents
        ):
            metrics["contract_profile_index"] = float(
                next(
                    (
                        infos[agent]["contract_profile_index"]
                        for agent in self.possible_agents
                        if "contract_profile_index" in infos.get(agent, {})
                    ),
                    0.0,
                )
            )
            metrics["contract_profile_changed"] = float(
                any(
                    bool(infos.get(agent, {}).get("contract_profile_changed", False))
                    for agent in self.possible_agents
                )
            )
            metrics["contract_vote_margin"] = float(
                next(
                    (
                        infos[agent]["contract_vote_margin"]
                        for agent in self.possible_agents
                        if "contract_vote_margin" in infos.get(agent, {})
                    ),
                    0.0,
                )
            )
            metrics["contract_permissiveness"] = float(
                next(
                    (
                        infos[agent]["contract_permissiveness"]
                        for agent in self.possible_agents
                        if "contract_permissiveness" in infos.get(agent, {})
                    ),
                    0.0,
                )
            )
            metrics["contract_shield_expanded"] = float(
                any(
                    bool(infos.get(agent, {}).get("contract_shield_expanded", False))
                    for agent in self.possible_agents
                )
            )
            metrics["contract_shield_expansions_cumulative"] = float(
                next(
                    (
                        infos[agent]["contract_shield_expansions_cumulative"]
                        for agent in self.possible_agents
                        if "contract_shield_expansions_cumulative"
                        in infos.get(agent, {})
                    ),
                    0.0,
                )
            )
        return metrics

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        observations, infos = self.env.reset(seed=seed, options=options)
        self.agents = list(self.env.agents)
        infos = _copy_infos(infos)
        infos = _augment_shared_metrics(
            infos,
            self.possible_agents,
            metrics=self._safety_metrics(infos),
        )
        return observations, infos

    def step(self, actions: dict[str, Any]):
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        self.agents = list(self.env.agents)
        infos = _copy_infos(infos)
        infos = _augment_shared_metrics(
            infos,
            self.possible_agents,
            metrics=self._safety_metrics(infos),
        )
        return observations, rewards, terminations, truncations, infos

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def state(self):
        return self.env.state()

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def __getattr__(self, item: str):
        return getattr(self.env, item)


class JointShieldedParallelEnv(ParallelEnv):
    metadata = {
        "name": "joint_shielded_parallel_v0",
        "is_parallelizable": True,
    }

    def __init__(
        self,
        env: ParallelEnv,
        model: AbstractionModel,
        shield: JointShield,
        *,
        debug_safety_violations: bool = False,
        violation_policy: str = "raise",
    ) -> None:
        self.env = env
        self.model = model
        self.shield = shield
        self.debug_safety_violations = bool(debug_safety_violations)
        self.violation_policy = _validate_violation_policy(violation_policy)
        self.possible_agents = list(env.possible_agents)
        self.agent_name_mapping = dict(getattr(env, "agent_name_mapping", {}))
        self.agents: list[str] = []
        self._product_state: ProductState | None = None
        self._last_joint_action_mask: np.ndarray | None = None
        self._last_per_agent_action_masks: dict[str, np.ndarray] = {}

        action_dims = [int(self.action_space(agent).n) for agent in self.possible_agents]
        if len(set(action_dims)) != 1:
            raise ValueError(
                "JointShieldedParallelEnv requires all agents to share one "
                f"discrete action dimension, got {action_dims!r}."
            )
        self.action_dim = int(action_dims[0])
        self.joint_action_table: JointActionTable = tuple(
            tuple(int(action) for action in joint_action)
            for joint_action in product(
                range(self.action_dim),
                repeat=len(self.possible_agents),
            )
        )
        self.joint_action_indices = {
            joint_action: index
            for index, joint_action in enumerate(self.joint_action_table)
        }

    def observation_space(self, agent: str):
        return self.env.observation_space(agent)

    def action_space(self, agent: str):
        return self.env.action_space(agent)

    @property
    def joint_action_dim(self) -> int:
        return len(self.joint_action_table)

    def _current_product_state(self) -> ProductState:
        if self._product_state is None:
            raise RuntimeError("Call reset() before step().")
        return self._product_state

    def _per_agent_action_masks(
        self,
        infos: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, np.ndarray]:
        masks: dict[str, np.ndarray] = {}
        for agent in self.possible_agents:
            agent_info = infos.get(agent, {})
            raw_mask = agent_info.get("action_mask")
            if raw_mask is None:
                mask = np.ones(self.action_dim, dtype=bool)
            else:
                mask = _coerce_action_mask(
                    np.asarray(raw_mask, dtype=bool),
                    action_dim=self.action_dim,
                    agent=agent,
                    source="existing",
                )
            if not bool(mask.any()):
                raise RuntimeError(f"No executable action available for {agent!r}.")
            masks[agent] = mask
        return masks

    def _joint_action_mask(
        self,
        product_state: ProductState,
        per_agent_action_masks: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        safe_joint_actions = set(self.shield.safe_joint_actions(product_state))
        if not safe_joint_actions:
            raise RuntimeError(
                f"No safe joint action available at {product_state!r}."
            )
        mask = np.zeros(self.joint_action_dim, dtype=bool)
        for index, joint_action in enumerate(self.joint_action_table):
            if joint_action not in safe_joint_actions:
                continue
            if all(
                _action_allowed(per_agent_action_masks[agent], action)
                for agent, action in zip(
                    self.possible_agents,
                    joint_action,
                    strict=True,
                )
            ):
                mask[index] = True
        if not bool(mask.any()):
            raise RuntimeError(
                f"No executable safe joint action available at {product_state!r}."
            )
        return mask

    def _augment_joint_infos(
        self,
        infos: dict[str, dict[str, Any]],
        *,
        joint_action_mask: np.ndarray,
        proposed_joint_action: JointAction | None = None,
        executed_joint_action: JointAction | None = None,
        violation_message: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        for agent in self.possible_agents:
            agent_info = infos.setdefault(agent, {})
            agent_info["joint_action_mask"] = joint_action_mask.astype(bool, copy=True)
            agent_info["shield_intervened"] = False
            agent_info["shield_repair_failed"] = False
            if proposed_joint_action is not None:
                agent_info["shield_proposed_joint_action"] = tuple(
                    int(action) for action in proposed_joint_action
                )
            if executed_joint_action is not None:
                agent_info["shield_executed_joint_action"] = tuple(
                    int(action) for action in executed_joint_action
                )
            agent_info["shield_safety_violated"] = violation_message is not None
            if violation_message is not None:
                agent_info["shield_safety_violation"] = violation_message
        return infos

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        observations, infos = self.env.reset(seed=seed, options=options)
        self.agents = list(self.env.agents)
        env_state = safety_projection(self.model, self.model.abstract_state(self.env))
        self._product_state = initial_product_state(
            self.model,
            self.shield.automaton,
            env_state,
        )
        if not self.shield.contains(self._product_state):
            raise RuntimeError(
                "Joint shield initial state is outside the winning region."
            )
        infos = _copy_infos(infos)
        per_agent_masks = self._per_agent_action_masks(infos)
        joint_action_mask = self._joint_action_mask(
            self._product_state,
            per_agent_masks,
        )
        infos = self._augment_joint_infos(
            infos,
            joint_action_mask=joint_action_mask,
        )
        self._last_per_agent_action_masks = {
            agent: mask.astype(bool, copy=True)
            for agent, mask in per_agent_masks.items()
        }
        self._last_joint_action_mask = joint_action_mask.astype(bool, copy=True)
        return observations, infos

    def step(self, actions: dict[str, Any]):
        current_product_state = self._current_product_state()
        proposed_joint_action = tuple(
            int(actions[agent])
            for agent in self.possible_agents
        )
        proposed_joint_action_index = self.joint_action_indices.get(
            proposed_joint_action
        )
        recomputed_mask = self._joint_action_mask(
            current_product_state,
            self._last_per_agent_action_masks,
        )
        cached_mask = self._last_joint_action_mask
        if cached_mask is None:
            cached_mask = recomputed_mask
        cached_mask = _coerce_joint_action_mask(
            cached_mask,
            joint_action_dim=self.joint_action_dim,
            source="cached emitted",
        )
        recomputed_mask = _coerce_joint_action_mask(
            recomputed_mask,
            joint_action_dim=self.joint_action_dim,
            source="recomputed",
        )
        if (
            proposed_joint_action_index is None
            or not _action_allowed(cached_mask, proposed_joint_action_index)
            or not _action_allowed(recomputed_mask, proposed_joint_action_index)
        ):
            raise ValueError(
                _joint_shield_mask_violation_message(
                    wrapper_name="JointShieldedParallelEnv",
                    proposed_joint_action=proposed_joint_action,
                    proposed_joint_action_index=proposed_joint_action_index,
                    cached_mask=cached_mask,
                    recomputed_mask=recomputed_mask,
                    current_product_state=current_product_state,
                )
            )

        executed = {
            agent: int(action)
            for agent, action in zip(
                self.possible_agents,
                proposed_joint_action,
                strict=True,
            )
        }
        observations, rewards, terminations, truncations, infos = self.env.step(executed)
        self.agents = list(self.env.agents)
        env_state = safety_projection(self.model, self.model.abstract_state(self.env))
        observed_label = self.model.label(env_state)
        next_monitor_state = self.shield.automaton.transition(
            current_product_state.automaton_state,
            observed_label,
        )
        self._product_state = ProductState(
            env_state=env_state,
            automaton_state=next_monitor_state,
        )
        entered_rejecting = (
            current_product_state.automaton_state in self.shield.automaton.safe_states
            and next_monitor_state not in self.shield.automaton.safe_states
        )
        violation_message = None
        if entered_rejecting:
            violation_message = _joint_shield_safety_violation_message(
                executed_joint_action=proposed_joint_action,
                proposed_joint_action=proposed_joint_action,
                cached_joint_action_mask=cached_mask,
                recomputed_joint_action_mask=recomputed_mask,
                current_product_state=current_product_state,
                next_product_state=self._product_state,
                observed_label=observed_label,
                detailed=(
                    self.debug_safety_violations
                    or self.violation_policy == "raise"
                ),
            )
            if self.violation_policy == "raise":
                raise RuntimeError(violation_message)
            if self.violation_policy == "warn":
                warnings.warn(violation_message, RuntimeWarning, stacklevel=2)

        infos = _copy_infos(infos)
        per_agent_masks = self._per_agent_action_masks(infos)
        joint_action_mask = self._joint_action_mask(
            self._product_state,
            per_agent_masks,
        )
        infos = self._augment_joint_infos(
            infos,
            joint_action_mask=joint_action_mask,
            proposed_joint_action=proposed_joint_action,
            executed_joint_action=proposed_joint_action,
            violation_message=violation_message,
        )
        self._last_per_agent_action_masks = {
            agent: mask.astype(bool, copy=True)
            for agent, mask in per_agent_masks.items()
        }
        self._last_joint_action_mask = joint_action_mask.astype(bool, copy=True)
        return observations, rewards, terminations, truncations, infos

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def state(self):
        return self.env.state()

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def __getattr__(self, item: str):
        return getattr(self.env, item)


class ShieldedParallelEnv(ParallelEnv):
    metadata = {
        "name": "shielded_parallel_v0",
        "is_parallelizable": True,
    }

    def __init__(
        self,
        env: ParallelEnv,
        model: AbstractionModel,
        shields: AgentShields,
        *,
        repair_mode: str = "max_agreement_random",
        debug_safety_violations: bool = False,
        violation_policy: str = "raise",
    ) -> None:
        if repair_mode != "max_agreement_random":
            raise ValueError(f"Unsupported repair mode: {repair_mode}")
        self.env = env
        self.model = model
        self.shields = dict(shields)
        self.repair_mode = repair_mode
        self.debug_safety_violations = bool(debug_safety_violations)
        self.violation_policy = _validate_violation_policy(violation_policy)
        self.possible_agents = list(env.possible_agents)
        self.agent_name_mapping = dict(getattr(env, "agent_name_mapping", {}))
        self.agents: list[str] = []
        self._product_states: dict[str, ProductState] = {}
        self._last_action_masks: dict[str, np.ndarray] = {}

    def observation_space(self, agent: str):
        return self.env.observation_space(agent)

    def action_space(self, agent: str):
        return self.env.action_space(agent)

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        observations, infos = self.env.reset(seed=seed, options=options)
        self.agents = list(self.env.agents)
        env_state = safety_projection(self.model, self.model.abstract_state(self.env))
        self._product_states = {
            agent: initial_product_state(self.model, self.shields[agent].automaton, env_state)
            for agent in self.possible_agents
        }
        for agent, product_state in self._product_states.items():
            if not self.shields[agent].contains(product_state):
                raise RuntimeError(
                    f"Shield initial state for {agent!r} is outside the winning region."
                )
        masks = {
            agent: _safe_action_mask(
                self.shields[agent],
                self._product_states[agent],
                self.action_space(agent).n,
            )
            for agent in self.possible_agents
        }
        infos = _copy_infos(infos)
        infos = _augment_action_mask_infos(infos, self.possible_agents, masks=masks)
        self._last_action_masks = {
            agent: np.asarray(infos[agent]["action_mask"], dtype=bool)
            for agent in self.possible_agents
        }
        return observations, infos

    def step(self, actions: dict[str, Any]):
        if not self._product_states:
            raise RuntimeError("Call reset() before step().")

        current_product_states = dict(self._product_states)
        proposed = {
            agent: int(actions[agent])
            for agent in self.possible_agents
        }
        executed: dict[str, int] = {}
        intervened: dict[str, bool] = {}
        repair_failed: dict[str, bool] = {}
        cached_action_masks: dict[str, np.ndarray] = {}
        recomputed_action_masks: dict[str, np.ndarray] = {}

        for agent in self.possible_agents:
            proposed_action = proposed[agent]
            current_product_state = current_product_states[agent]
            action_dim = int(self.action_space(agent).n)
            recomputed_mask = _safe_action_mask(
                self.shields[agent],
                current_product_state,
                action_dim,
                strict=self.violation_policy == "raise",
            )
            cached_mask = self._last_action_masks.get(agent)
            if cached_mask is None:
                cached_mask = recomputed_mask
            cached_mask = _coerce_action_mask(
                cached_mask,
                action_dim=action_dim,
                agent=agent,
                source="cached emitted",
            )
            recomputed_mask = _coerce_action_mask(
                recomputed_mask,
                action_dim=action_dim,
                agent=agent,
                source="recomputed",
            )
            cached_action_masks[agent] = cached_mask
            recomputed_action_masks[agent] = recomputed_mask
            executed_action, did_intervene, did_fail = _execute_validated_shield_action(
                wrapper_name="ShieldedParallelEnv",
                agent=agent,
                proposed_action=proposed_action,
                current_product_state=current_product_state,
                cached_mask=cached_mask,
                recomputed_mask=recomputed_mask,
            )
            executed[agent] = executed_action
            intervened[agent] = did_intervene
            repair_failed[agent] = did_fail

        observations, rewards, terminations, truncations, infos = self.env.step(executed)
        self.agents = list(self.env.agents)
        env_state = safety_projection(self.model, self.model.abstract_state(self.env))
        observed_label = self.model.label(env_state)

        violation_messages: dict[str, str] = {}
        for agent in self.possible_agents:
            shield = self.shields[agent]
            current_product_state = current_product_states[agent]
            next_monitor_state = shield.automaton.transition(
                current_product_state.automaton_state,
                observed_label,
            )
            self._product_states[agent] = ProductState(
                env_state=env_state,
                automaton_state=next_monitor_state,
            )
            entered_rejecting = (
                current_product_state.automaton_state in shield.automaton.safe_states
                and next_monitor_state not in shield.automaton.safe_states
            )
            if entered_rejecting:
                violation_messages[agent] = _shield_safety_violation_message(
                    agent=agent,
                    executed_action=executed[agent],
                    proposed_action=proposed[agent],
                    cached_action_mask=cached_action_masks.get(agent),
                    recomputed_action_mask=recomputed_action_masks.get(agent),
                    current_product_state=current_product_state,
                    next_product_state=self._product_states[agent],
                    observed_label=observed_label,
                    detailed=(
                        self.debug_safety_violations
                        or self.violation_policy == "raise"
                    ),
                )

        if violation_messages:
            message = "\n".join(violation_messages.values())
            if self.violation_policy == "raise":
                raise RuntimeError(message)
            if self.violation_policy == "warn":
                warnings.warn(message, RuntimeWarning, stacklevel=2)

        infos = _copy_infos(infos)
        masks = {
            agent: _safe_action_mask(
                self.shields[agent],
                self._product_states[agent],
                self.action_space(agent).n,
                strict=False,
            )
            for agent in self.possible_agents
        }
        infos = _augment_action_mask_infos(infos, self.possible_agents, masks=masks)
        self._last_action_masks = {
            agent: np.asarray(infos[agent]["action_mask"], dtype=bool)
            for agent in self.possible_agents
        }
        for agent in self.possible_agents:
            agent_info = infos.setdefault(agent, {})
            agent_info["shield_intervened"] = bool(intervened[agent])
            agent_info["shield_repair_failed"] = bool(repair_failed[agent])
            agent_info["shield_proposed_action"] = int(proposed[agent])
            agent_info["shield_executed_action"] = int(executed[agent])
            agent_info["shield_safety_violated"] = agent in violation_messages
            if agent in violation_messages:
                agent_info["shield_safety_violation"] = violation_messages[agent]

        return observations, rewards, terminations, truncations, infos

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def state(self):
        return self.env.state()

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def __getattr__(self, item: str):
        return getattr(self.env, item)


class ContractShieldedParallelEnv(ParallelEnv):
    metadata = {
        "name": "contract_shielded_parallel_v0",
        "is_parallelizable": True,
    }

    def __init__(
        self,
        env: ParallelEnv,
        model: AbstractionModel,
        *,
        contract_library: ContractLibrary | None = None,
        repair_mode: str = "max_agreement_random",
        debug_safety_violations: bool = False,
        violation_policy: str = "raise",
    ) -> None:
        if repair_mode != "max_agreement_random":
            raise ValueError(f"Unsupported repair mode: {repair_mode}")
        if contract_library is None:
            raise ValueError("ContractShieldedParallelEnv requires a contract_library.")
        self.env = env
        self.model = model
        self.contract_library = contract_library
        self.repair_mode = repair_mode
        self.debug_safety_violations = bool(debug_safety_violations)
        self.violation_policy = _validate_violation_policy(violation_policy)
        self.possible_agents = list(env.possible_agents)
        self.agent_name_mapping = dict(getattr(env, "agent_name_mapping", {}))
        self.agents: list[str] = []
        self._certified_profiles = tuple(contract_library.certified_profiles)
        if not self._certified_profiles:
            raise ValueError("ContractShieldedParallelEnv requires a non-empty certified library.")
        self._profile_by_id = {
            certified.profile.profile_id: certified
            for certified in self._certified_profiles
        }
        self._profile_index_by_id = {
            certified.profile.profile_id: index
            for index, certified in enumerate(self._certified_profiles)
        }
        self._profile_shields = {
            certified.profile.profile_id: self._build_profile_shields(certified)
            for certified in self._certified_profiles
        }
        self._current_profile_id = contract_library.initial_profile.profile.profile_id
        self._has_reset = False
        self._global_product_state: ProductState | None = None
        self._contract_product_state: AssumeGuaranteeProductState | None = None
        self._last_action_masks: dict[str, np.ndarray] = {}
        self._contract_undercoverage_warning_keys: set[tuple[str, str, str]] = set()
        self._contract_shield_expansion_count = 0
        self.total_safety_violations = {
            agent: 0
            for agent in self.possible_agents
        }
        self.episode_safety_violations = {
            agent: 0
            for agent in self.possible_agents
        }
        self.total_obligation_violations = {
            agent: 0
            for agent in self.possible_agents
        }
        self.episode_obligation_violations = {
            agent: 0
            for agent in self.possible_agents
        }
        self._last_vote_winner = self._current_profile_id
        self._last_vote_margin = 0

    def observation_space(self, agent: str):
        return self.env.observation_space(agent)

    def action_space(self, agent: str):
        return self.env.action_space(agent)

    def _build_profile_shields(
        self,
        certified: CertifiedContractProfile,
    ) -> dict[str, LocalShield | AssumeGuaranteeLocalShield]:
        shields: dict[str, LocalShield | AssumeGuaranteeLocalShield] = {}
        first_template = next(iter(certified.templates.values()))
        if isinstance(first_template, AssumeGuaranteeShieldTemplate):
            for agent_id in self.model.agent_ids:
                shields[agent_id] = AssumeGuaranteeLocalShield(
                    agent_id=agent_id,
                    automaton=certified.automata[agent_id],
                    template=first_template,
                    rng=np.random.default_rng(),
                )
            return shields
        for agent_id in self.model.agent_ids:
            template = certified.templates[agent_id]
            shields[agent_id] = LocalShield(
                agent_id=agent_id,
                model=self.model,
                automaton=template.automaton,
                transitions=template.transitions,
                winning_region=template.winning_region,
                rng=np.random.default_rng(),
            )
        return shields

    def _resolve_profile_id(self, options: dict[str, Any] | None) -> str:
        options = options or {}
        raw_profile_id = options.get("contract_profile_id")
        if raw_profile_id is not None:
            profile_id = str(raw_profile_id)
            if profile_id not in self._profile_by_id:
                raise ValueError(f"Unknown contract profile id: {profile_id!r}")
            return profile_id
        if "contract_profile_index" in options:
            profile_index = int(options["contract_profile_index"])
            if profile_index < 0 or profile_index >= len(self._certified_profiles):
                raise ValueError(f"Unknown contract profile index: {profile_index}")
            return self._certified_profiles[profile_index].profile.profile_id
        return self._current_profile_id

    def _active_profile(self) -> CertifiedContractProfile:
        return self._profile_by_id[self._current_profile_id]

    def _active_shields(
        self,
    ) -> dict[str, LocalShield | AssumeGuaranteeLocalShield]:
        return self._profile_shields[self._current_profile_id]

    def _active_assume_guarantee_template(self) -> AssumeGuaranteeShieldTemplate:
        certified = self._active_profile()
        template = next(iter(certified.templates.values()))
        if not isinstance(template, AssumeGuaranteeShieldTemplate):
            raise RuntimeError("Active contract profile is not assume-guarantee certified.")
        return template

    def _replace_active_assume_guarantee_template(
        self,
        template: AssumeGuaranteeShieldTemplate,
    ) -> None:
        certified = self._active_profile()
        for agent_id in self.model.agent_ids:
            certified.templates[agent_id] = template
        for shield in self._active_shields().values():
            if isinstance(shield, AssumeGuaranteeLocalShield):
                shield.template = template

    def _expand_active_contract_profile_from(
        self,
        state: AssumeGuaranteeProductState,
        *,
        reason: str,
        observed_label: Label | None = None,
        executed_actions: Mapping[str, int] | None = None,
        current_product_state: AssumeGuaranteeProductState | None = None,
    ) -> dict[str, Any] | None:
        certified = self._active_profile()
        template = self._active_assume_guarantee_template()
        projected_state = AssumeGuaranteeProductState(
            env_state=safety_projection(self.model, state.env_state),
            obligation_states=tuple(int(value) for value in state.obligation_states),
        )
        if projected_state in template.winning_region:
            return None

        details = ""
        if _contract_undercoverage_detail_log():
            details = (
                f" actions={dict(executed_actions)!r}; "
                f"current={current_product_state!r}; "
                f"observed_label={tuple(sorted(observed_label or ()))!r}."
            )
        print(
            "[shield-undercoverage] expanding contract profile "
            f"{self._current_profile_id!r} for reason={reason!r}; "
            f"known_transitions={len(template.transitions)} "
            f"known_winning={len(template.winning_region)} "
            f"max_states={template.max_states!r}; state={projected_state!r}. "
            f"{details}",
            flush=True,
        )

        try:
            expanded = synthesize_assume_guarantee_profile_shield(
                self.model,
                certified.automata,
                projected_state.env_state,
                max_states=template.max_states,
                initial_product_state=projected_state,
            )
        except Exception:
            return None

        if projected_state not in expanded.winning_region:
            return None

        base_initial_states = template.initial_states or (template.initial_state,)
        combined_initial_states = tuple(
            dict.fromkeys((*base_initial_states, *expanded.initial_states))
        )
        combined_template = AssumeGuaranteeShieldTemplate(
            transitions={
                **template.transitions,
                **expanded.transitions,
            },
            winning_region=frozenset(
                set(template.winning_region).union(expanded.winning_region)
            ),
            allowed_actions={
                **template.allowed_actions,
                **expanded.allowed_actions,
            },
            initial_state=template.initial_state,
            initial_states=combined_initial_states,
            max_states=template.max_states,
        )
        self._replace_active_assume_guarantee_template(combined_template)
        return self._record_contract_undercoverage_event(
            reason=reason,
            state=projected_state,
            observed_label=observed_label,
            executed_actions=executed_actions,
            current_product_state=current_product_state,
        )

    def _contract_undercoverage_reason_for_step(
        self,
        *,
        current_product_state: AssumeGuaranteeProductState,
        next_product_state: AssumeGuaranteeProductState,
        executed_actions: Mapping[str, int],
    ) -> str:
        joint_action = tuple(
            int(executed_actions[agent_id])
            for agent_id in self.model.agent_ids
        )
        try:
            expected_successors = self.model.successors_for_joint_action(
                current_product_state.env_state,
                joint_action,
            )
        except Exception:
            return "uncovered_product_state"
        projected_expected = {
            safety_projection(self.model, successor)
            for successor in expected_successors
        }
        if next_product_state.env_state not in projected_expected:
            return "unexpected_successor"
        return "uncovered_product_state"

    def _record_contract_undercoverage_event(
        self,
        *,
        reason: str,
        state: AssumeGuaranteeProductState,
        observed_label: Label | None,
        executed_actions: Mapping[str, int] | None,
        current_product_state: AssumeGuaranteeProductState | None,
    ) -> dict[str, Any]:
        self._contract_shield_expansion_count += 1
        profile_id = self._current_profile_id
        event = {
            "profile_id": profile_id,
            "profile_index": int(self._profile_index_by_id[profile_id]),
            "reason": str(reason),
            "state": repr(state),
            "env_state": repr(state.env_state),
            "obligation_states": tuple(int(value) for value in state.obligation_states),
            "observed_label": tuple(sorted(observed_label or ())),
            "executed_actions": (
                {str(agent): int(action) for agent, action in executed_actions.items()}
                if executed_actions is not None
                else None
            ),
            "current_product_state": (
                repr(current_product_state)
                if current_product_state is not None
                else None
            ),
            "expansion_count": int(self._contract_shield_expansion_count),
        }
        warning_key = (profile_id, str(reason), event["state"])
        if warning_key not in self._contract_undercoverage_warning_keys:
            self._contract_undercoverage_warning_keys.add(warning_key)
            details = ""
            if _contract_undercoverage_detail_log():
                details = (
                    f" actions={event['executed_actions']!r}; "
                    f"current={event['current_product_state']!r}."
                )
            message = (
                "[shield-undercoverage] Contract shield expanded profile "
                f"{profile_id!r} for reason={reason!r}; state={event['state']}. "
                f"{details}"
                "This indicates the safety abstraction or certified start-state "
                "coverage should be refined."
            )
            print(message)
            warnings.warn(message, RuntimeWarning, stacklevel=3)
        return event

    def _augment_contract_expansion_infos(
        self,
        infos: dict[str, dict[str, Any]],
        *,
        event: Mapping[str, Any] | None,
    ) -> None:
        expanded = event is not None
        for agent in self.possible_agents:
            agent_info = infos.setdefault(agent, {})
            agent_info["contract_shield_expanded"] = expanded
            agent_info["contract_shield_expansion_reason"] = (
                str(event["reason"]) if event is not None else ""
            )
            agent_info["contract_shield_expansions_cumulative"] = int(
                self._contract_shield_expansion_count
            )
            if event is not None:
                agent_info["contract_undercoverage_event"] = dict(event)

    def _obligation_monitor_state_map(
        self,
        certified: CertifiedContractProfile,
        contract_state: AssumeGuaranteeProductState,
    ) -> dict[str, int]:
        return {
            agent: int(contract_state.obligation_states[agent_idx])
            for agent_idx, agent in enumerate(self.model.agent_ids)
        }

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        next_profile_id = self._resolve_profile_id(options)
        profile_changed = self._has_reset and next_profile_id != self._current_profile_id
        self._current_profile_id = next_profile_id
        self._last_vote_winner = str(
            (options or {}).get("contract_vote_winner", self._current_profile_id)
        )
        self._last_vote_margin = int((options or {}).get("contract_vote_margin", 0))

        observations, infos = self.env.reset(seed=seed, options=options)
        self.agents = list(self.env.agents)
        env_state = safety_projection(self.model, self.model.abstract_state(self.env))
        certified = self._active_profile()
        shields = self._active_shields()
        self.episode_safety_violations = {
            agent: 0
            for agent in self.possible_agents
        }
        self.episode_obligation_violations = {
            agent: 0
            for agent in self.possible_agents
        }
        self._global_product_state = initial_product_state(
            self.model,
            certified.global_automaton,
            env_state,
        )
        self._contract_product_state = initial_assume_guarantee_product_state(
            self.model,
            tuple(certified.automata[agent] for agent in self.model.agent_ids),
            env_state,
        )
        undercoverage_event: dict[str, Any] | None = None
        if not all(
            shields[agent].contains(self._contract_product_state)
            for agent in self.possible_agents
        ):
            undercoverage_event = self._expand_active_contract_profile_from(
                self._contract_product_state,
                reason="uncovered_initial_state",
                observed_label=self.model.label(env_state),
            )
            shields = self._active_shields()
        for agent in self.possible_agents:
            if not shields[agent].contains(self._contract_product_state):
                raise RuntimeError(
                    f"Contract shield initial state for {agent!r} is outside "
                    "the winning region."
                )
        obligation_monitor_states = self._obligation_monitor_state_map(
            certified,
            self._contract_product_state,
        )

        infos = _copy_infos(infos)
        infos = _augment_safety_infos(
            infos,
            self.possible_agents,
            safety_violations={agent: False for agent in self.possible_agents},
            safety_violation_counts=self.total_safety_violations,
            episode_safety_violations=self.episode_safety_violations,
            monitor_states={
                agent: int(self._global_product_state.automaton_state)
                for agent in self.possible_agents
            },
        )
        infos = _augment_contract_infos(
            infos,
            self.possible_agents,
            profile_index=self._profile_index_by_id[self._current_profile_id],
            profile_changed=profile_changed,
            certified_profile=certified,
            obligation_violations={agent: False for agent in self.possible_agents},
            obligation_monitor_states=obligation_monitor_states,
            vote_winner=self._last_vote_winner,
            vote_margin=self._last_vote_margin,
        )
        self._augment_contract_expansion_infos(
            infos,
            event=undercoverage_event,
        )
        masks = {
            agent: _safe_action_mask(
                shields[agent],
                self._contract_product_state,
                self.action_space(agent).n,
            )
            for agent in self.possible_agents
        }
        infos = _augment_action_mask_infos(infos, self.possible_agents, masks=masks)
        self._last_action_masks = {
            agent: np.asarray(infos[agent]["action_mask"], dtype=bool)
            for agent in self.possible_agents
        }
        self._has_reset = True
        return observations, infos

    def step(self, actions: dict[str, Any]):
        if self._global_product_state is None or self._contract_product_state is None:
            raise RuntimeError("Call reset() before step().")

        certified = self._active_profile()
        shields = self._active_shields()
        template = self._active_assume_guarantee_template()
        current_global_state = self._global_product_state
        current_contract_state = self._contract_product_state
        proposed = {
            agent: int(actions[agent])
            for agent in self.possible_agents
        }
        executed: dict[str, int] = {}
        intervened: dict[str, bool] = {}
        repair_failed: dict[str, bool] = {}
        cached_action_masks: dict[str, np.ndarray] = {}
        recomputed_action_masks: dict[str, np.ndarray] = {}

        for agent in self.possible_agents:
            proposed_action = proposed[agent]
            action_dim = int(self.action_space(agent).n)
            recomputed_mask = _safe_action_mask(
                shields[agent],
                current_contract_state,
                action_dim,
                strict=self.violation_policy == "raise",
            )
            cached_mask = self._last_action_masks.get(agent)
            if cached_mask is None:
                cached_mask = recomputed_mask
            cached_mask = _coerce_action_mask(
                cached_mask,
                action_dim=action_dim,
                agent=agent,
                source="cached emitted",
            )
            recomputed_mask = _coerce_action_mask(
                recomputed_mask,
                action_dim=action_dim,
                agent=agent,
                source="recomputed",
            )
            cached_action_masks[agent] = cached_mask
            recomputed_action_masks[agent] = recomputed_mask
            executed_action, did_intervene, did_fail = _execute_validated_shield_action(
                wrapper_name="ContractShieldedParallelEnv",
                agent=agent,
                proposed_action=proposed_action,
                current_product_state=current_contract_state,
                cached_mask=cached_mask,
                recomputed_mask=recomputed_mask,
                product_state_label="assume-guarantee product state",
            )
            executed[agent] = executed_action
            intervened[agent] = did_intervene
            repair_failed[agent] = did_fail

        observations, rewards, terminations, truncations, infos = self.env.step(executed)
        self.agents = list(self.env.agents)
        env_state = safety_projection(self.model, self.model.abstract_state(self.env))
        observed_label = self.model.label(env_state)

        next_global_monitor_state = certified.global_automaton.transition(
            current_global_state.automaton_state,
            observed_label,
        )
        self._global_product_state = ProductState(
            env_state=env_state,
            automaton_state=next_global_monitor_state,
        )
        entered_global_rejecting = (
            current_global_state.automaton_state in certified.global_automaton.safe_states
            and next_global_monitor_state not in certified.global_automaton.safe_states
        )
        safety_violations = {
            agent: bool(entered_global_rejecting)
            for agent in self.possible_agents
        }
        if entered_global_rejecting:
            for agent in self.possible_agents:
                self.total_safety_violations[agent] += 1
                self.episode_safety_violations[agent] += 1

        obligation_violations: dict[str, bool] = {}
        violation_messages: dict[str, str] = {}
        next_contract_state = AssumeGuaranteeProductState(
            env_state=env_state,
            obligation_states=tuple(
                certified.automata[agent].transition(
                    current_contract_state.obligation_states[agent_idx],
                    observed_label,
                )
                for agent_idx, agent in enumerate(self.model.agent_ids)
            ),
        )
        self._contract_product_state = next_contract_state
        undercoverage_event: dict[str, Any] | None = None
        if next_contract_state not in template.winning_region:
            undercoverage_event = self._expand_active_contract_profile_from(
                next_contract_state,
                reason=self._contract_undercoverage_reason_for_step(
                    current_product_state=current_contract_state,
                    next_product_state=next_contract_state,
                    executed_actions=executed,
                ),
                observed_label=observed_label,
                executed_actions=executed,
                current_product_state=current_contract_state,
            )
            if undercoverage_event is not None:
                certified = self._active_profile()
                shields = self._active_shields()
                template = self._active_assume_guarantee_template()
        obligation_monitor_states = self._obligation_monitor_state_map(
            certified,
            next_contract_state,
        )
        for agent_idx, agent in enumerate(self.model.agent_ids):
            automaton = certified.automata[agent]
            current_monitor_state = current_contract_state.obligation_states[agent_idx]
            next_obligation_monitor_state = next_contract_state.obligation_states[
                agent_idx
            ]
            entered_obligation_rejecting = (
                current_monitor_state in automaton.safe_states
                and next_obligation_monitor_state not in automaton.safe_states
            )
            obligation_violations[agent] = entered_obligation_rejecting
            if entered_obligation_rejecting:
                self.total_obligation_violations[agent] += 1
                self.episode_obligation_violations[agent] += 1
            if entered_obligation_rejecting:
                violation_messages[agent] = (
                    "CONTRACT OBLIGATION VIOLATION: ContractShieldedParallelEnv reached "
                    f"a rejecting local-obligation monitor state for {agent!r} "
                    f"after executing local action {executed[agent]} "
                    f"(proposed {proposed[agent]}) from {current_contract_state!r}. "
                    f"Monitor transition {current_monitor_state} -> "
                    f"{next_obligation_monitor_state}. Cached allowed actions "
                    f"{_allowed_actions(cached_action_masks.get(agent))}; "
                    f"recomputed allowed actions "
                    f"{_allowed_actions(recomputed_action_masks.get(agent))}. "
                    f"Observed label {sorted(observed_label)!r}; next product state "
                    f"{next_contract_state!r}."
                )
        entered_contract_rejecting = (
            current_contract_state in template.winning_region
            and next_contract_state not in template.winning_region
        )
        if entered_contract_rejecting and not violation_messages:
            cached_allowed_actions = _allowed_actions_by_agent(cached_action_masks)
            recomputed_allowed_actions = _allowed_actions_by_agent(
                recomputed_action_masks
            )
            violation_messages["__contract__"] = (
                "CONTRACT FIXED-POINT VIOLATION: ContractShieldedParallelEnv reached "
                "an assume-guarantee product state outside the certified fixed point "
                f"after executing actions {executed!r}. Cached allowed actions "
                f"{cached_allowed_actions!r}; "
                f"recomputed allowed actions "
                f"{recomputed_allowed_actions!r}. "
                f"Observed label "
                f"{sorted(observed_label)!r}; next product state "
                f"{next_contract_state!r}."
            )

        if entered_global_rejecting:
            cached_allowed_actions = _allowed_actions_by_agent(cached_action_masks)
            recomputed_allowed_actions = _allowed_actions_by_agent(
                recomputed_action_masks
            )
            violation_messages.setdefault(
                "__global__",
                "SAFETY VIOLATION: ContractShieldedParallelEnv reached a rejecting "
                f"global safety monitor state after executing actions {executed!r}. "
                f"Global monitor transition {current_global_state.automaton_state} -> "
                f"{next_global_monitor_state}. Cached allowed actions "
                f"{cached_allowed_actions!r}; recomputed allowed actions "
                f"{recomputed_allowed_actions!r}. "
                f"Observed label {sorted(observed_label)!r}; next global product "
                f"state {self._global_product_state!r}; next contract product "
                f"state {next_contract_state!r}.",
            )
        if violation_messages:
            message = "\n".join(violation_messages.values())
            if self.violation_policy == "raise":
                raise RuntimeError(message)
            if self.violation_policy == "warn":
                warnings.warn(message, RuntimeWarning, stacklevel=2)

        infos = _copy_infos(infos)
        infos = _augment_safety_infos(
            infos,
            self.possible_agents,
            safety_violations=safety_violations,
            safety_violation_counts=self.total_safety_violations,
            episode_safety_violations=self.episode_safety_violations,
            monitor_states={
                agent: next_global_monitor_state
                for agent in self.possible_agents
            },
        )
        infos = _augment_contract_infos(
            infos,
            self.possible_agents,
            profile_index=self._profile_index_by_id[self._current_profile_id],
            profile_changed=False,
            certified_profile=certified,
            obligation_violations=obligation_violations,
            obligation_monitor_states=obligation_monitor_states,
            vote_winner=self._last_vote_winner,
            vote_margin=self._last_vote_margin,
        )
        self._augment_contract_expansion_infos(
            infos,
            event=undercoverage_event,
        )
        masks = {
            agent: _safe_action_mask(
                shields[agent],
                next_contract_state,
                self.action_space(agent).n,
                strict=not bool(
                    terminations.get(agent, False) or truncations.get(agent, False)
                ),
            )
            for agent in self.possible_agents
        }
        infos = _augment_action_mask_infos(infos, self.possible_agents, masks=masks)
        self._last_action_masks = {
            agent: np.asarray(infos[agent]["action_mask"], dtype=bool)
            for agent in self.possible_agents
        }
        for agent in self.possible_agents:
            agent_info = infos.setdefault(agent, {})
            agent_info["shield_intervened"] = bool(intervened[agent])
            agent_info["shield_repair_failed"] = bool(repair_failed[agent])
            agent_info["shield_proposed_action"] = int(proposed[agent])
            agent_info["shield_executed_action"] = int(executed[agent])
            agent_info["shield_safety_violated"] = bool(entered_global_rejecting)
            if entered_global_rejecting:
                agent_info["shield_safety_violation"] = violation_messages["__global__"]

        return observations, rewards, terminations, truncations, infos

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def state(self):
        return self.env.state()

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def __getattr__(self, item: str):
        return getattr(self.env, item)
