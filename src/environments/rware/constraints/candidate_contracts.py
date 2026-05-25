"""Hand-authored RWARE contract profiles for debug runs."""

from __future__ import annotations

from src.experiments import CandidateContractProfileSpec


def _queue_yield_baseline(agent_ids: tuple[str, ...]) -> dict[str, str]:
    return {
        agent_id: f"G({agent_id}_queue_yield_ok)"
        for agent_id in agent_ids
    }


def candidate_contract_profiles(
    *,
    formula_name: str,
    global_formula: str,
    agent_ids: tuple[str, ...],
) -> tuple[CandidateContractProfileSpec, ...]:
    del global_formula
    if formula_name != "queue_yield_protocol":
        return tuple()
    baseline = _queue_yield_baseline(agent_ids)
    return (
        CandidateContractProfileSpec(
            profile_id="baseline_queue_yield",
            formulas=baseline,
            description="Each robot obeys the queue-yield protocol.",
            tags=("baseline",),
        ),
    )
