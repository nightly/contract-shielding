"""Hand-authored Connector contract profiles for debug runs."""

from __future__ import annotations

from src.experiments import CandidateContractProfileSpec


def _reservation_baseline(agent_ids: tuple[str, ...]) -> dict[str, str]:
    return {
        agent_id: f"G({agent_id}_respects_reservations_ok)"
        for agent_id in agent_ids
    }


def candidate_contract_profiles(
    *,
    formula_name: str,
    global_formula: str,
    agent_ids: tuple[str, ...],
) -> tuple[CandidateContractProfileSpec, ...]:
    del global_formula
    if formula_name != "respect_reserved_routes":
        return tuple()
    first_agent = agent_ids[0]
    last_agent = agent_ids[-1]
    baseline = _reservation_baseline(agent_ids)
    return (
        CandidateContractProfileSpec(
            profile_id="baseline_reservation_respect",
            formulas=baseline,
            description="Each agent maintains its own reservation-respect guarantee.",
            tags=("baseline",),
        ),
        CandidateContractProfileSpec(
            profile_id="promising_teammate_reservation_priority",
            formulas={
                **baseline,
                last_agent: (
                    f"({baseline[last_agent]}) & "
                    f"(G({last_agent}_respects_{first_agent}_reservation_ok))"
                ),
            },
            description="The trailing agent explicitly respects the lead agent reservation.",
            tags=("promising", "reservation"),
        ),
        CandidateContractProfileSpec(
            profile_id="nonpromising_avoid_connection",
            formulas={
                **baseline,
                first_agent: f"({baseline[first_agent]}) & (G(!{first_agent}_connected))",
            },
            description="A deliberately reward-hostile profile that avoids completion.",
            tags=("nonpromising", "conservative"),
        ),
    )
