"""Hand-authored Level-Based Foraging contract profiles for debug runs."""

from __future__ import annotations

from src.experiments import CandidateContractProfileSpec


def _no_failed_loads_clause(agent_ids: tuple[str, ...]) -> str:
    return " & ".join(f"!{agent_id}_failed_load" for agent_id in agent_ids)


def _local_no_failed_loads(agent_id: str) -> str:
    return f"G(!{agent_id}_failed_load)"


def _symmetric_coop_load_contract(
    agent_ids: tuple[str, ...],
) -> dict[str, str]:
    no_failed_loads = _no_failed_loads_clause(agent_ids)
    return {
        agent_id: f"G(({no_failed_loads}) & {agent_id}_coop_load_ok)"
        for agent_id in agent_ids
    }


def candidate_contract_profiles(
    *,
    formula_name: str,
    global_formula: str,
    agent_ids: tuple[str, ...],
) -> tuple[CandidateContractProfileSpec, ...]:
    del global_formula
    if formula_name != "avoid_failed_loads":
        return tuple()
    baseline = {
        agent_id: _local_no_failed_loads(agent_id)
        for agent_id in agent_ids
    }
    return (
        CandidateContractProfileSpec(
            profile_id="baseline_no_failed_loads",
            formulas=baseline,
            description="Each agent carries its own failed-load avoidance guard.",
            tags=("baseline",),
        ),
        CandidateContractProfileSpec(
            profile_id="symmetric_coop_load_ready",
            formulas=_symmetric_coop_load_contract(agent_ids),
            description=(
                "All agents preserve failed-load safety and commit to loading "
                "when cooperative loading is ready."
            ),
            tags=("promising", "coop-load", "symmetric"),
        ),
    )
