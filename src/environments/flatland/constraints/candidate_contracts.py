"""Hand-authored Flatland contract profiles for debug runs."""

from __future__ import annotations

from src.experiments import CandidateContractProfileSpec


def _global_no_deadlock(agent_ids: tuple[str, ...]) -> str:
    clauses = " & ".join(f"!{agent_id}_deadlocked" for agent_id in agent_ids)
    return f"G({clauses})"


def candidate_contract_profiles(
    *,
    formula_name: str,
    global_formula: str,
    agent_ids: tuple[str, ...],
) -> tuple[CandidateContractProfileSpec, ...]:
    del global_formula
    if formula_name != "avoid_deadlocks":
        return tuple()
    first_agent = agent_ids[0]
    last_agent = agent_ids[-1]
    global_obligation = _global_no_deadlock(agent_ids)
    baseline = {agent_id: "t" for agent_id in agent_ids}
    baseline[last_agent] = global_obligation
    return (
        CandidateContractProfileSpec(
            profile_id="baseline_no_deadlock",
            formulas=baseline,
            description="One profile owner carries the full global no-deadlock obligation.",
            tags=("baseline",),
        ),
        CandidateContractProfileSpec(
            profile_id="promising_yield_to_priority_train",
            formulas={
                **baseline,
                last_agent: (
                    f"({global_obligation}) & "
                    f"(X({last_agent}_yields_to_{first_agent}_ok))"
                ),
            },
            description="The opposing train yields on the next decision point.",
            tags=("promising", "yield"),
        ),
        CandidateContractProfileSpec(
            profile_id="nonpromising_hold_priority_train",
            formulas={
                **baseline,
                first_agent: f"G(!{first_agent}_moving)",
            },
            description="A conservative profile that keeps the first train from moving.",
            tags=("nonpromising", "conservative"),
        ),
    )
