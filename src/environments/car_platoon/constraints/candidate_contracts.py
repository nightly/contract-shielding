"""Hand-authored Car Platoon contract profiles for debug runs."""

from __future__ import annotations

from src.experiments import CandidateContractProfileSpec


def _gap_baseline(agent_ids: tuple[str, ...]) -> dict[str, str]:
    return {
        agent_id: f"G({agent_id}_gap_safe)"
        for agent_id in agent_ids
    }


def candidate_contract_profiles(
    *,
    formula_name: str,
    global_formula: str,
    agent_ids: tuple[str, ...],
) -> tuple[CandidateContractProfileSpec, ...]:
    del global_formula
    if formula_name != "maintain_safe_gap":
        return tuple()
    last_agent = agent_ids[-1]
    baseline = _gap_baseline(agent_ids)
    return (
        CandidateContractProfileSpec(
            profile_id="baseline_safe_gap",
            formulas=baseline,
            description="Each controlled follower maintains its own safe gap.",
            tags=("baseline",),
        ),
        CandidateContractProfileSpec(
            profile_id="promising_next_conservative_follow",
            formulas={
                **baseline,
                last_agent: (
                    f"({baseline[last_agent]}) & "
                    f"(X({last_agent}_conservative_follow_ok))"
                ),
            },
            description="The rear controlled follower commits to conservative following next.",
            tags=("promising", "platoon"),
        ),
        CandidateContractProfileSpec(
            profile_id="nonpromising_current_conservative_follow",
            formulas={
                **baseline,
                last_agent: (
                    f"({baseline[last_agent]}) & "
                    f"({last_agent}_conservative_follow_ok)"
                ),
            },
            description="A conservative current-state-only following obligation.",
            tags=("nonpromising", "conservative"),
        ),
    )
