"""Hand-authored CookingZoo contract profiles for debug runs."""

from __future__ import annotations

from src.experiments import CandidateContractProfileSpec


def _etiquette_formula(agent_id: str) -> str:
    return " & ".join(
        (
            f"G(!{agent_id}_bad_delivery_attempt)",
            f"G(!{agent_id}_holding_teammate_only_ingredient)",
            f"G(!{agent_id}_blocking_delivery_access)",
            f"G(!{agent_id}_blocking_cutboard_access)",
        )
    )


def _etiquette_baseline(agent_ids: tuple[str, ...]) -> dict[str, str]:
    return {
        agent_id: _etiquette_formula(agent_id)
        for agent_id in agent_ids
    }


def candidate_contract_profiles(
    *,
    formula_name: str,
    global_formula: str,
    agent_ids: tuple[str, ...],
) -> tuple[CandidateContractProfileSpec, ...]:
    del global_formula
    if formula_name != "kitchen_etiquette":
        return tuple()
    first_player = agent_ids[0]
    baseline = _etiquette_baseline(agent_ids)
    return (
        CandidateContractProfileSpec(
            profile_id="baseline_kitchen_etiquette",
            formulas=baseline,
            description="Both players obey the broad kitchen-etiquette formula.",
            tags=("baseline",),
        ),
        CandidateContractProfileSpec(
            profile_id="promising_delivery_station_clearance",
            formulas={
                **baseline,
                first_player: (
                    f"({baseline[first_player]}) & "
                    f"(G({first_player}_delivery_station_ok))"
                ),
            },
            description="Player 0 keeps delivery station access protocol-safe.",
            tags=("promising", "station-clearance"),
        ),
        CandidateContractProfileSpec(
            profile_id="promising_cutboard_station_clearance",
            formulas={
                **baseline,
                first_player: (
                    f"({baseline[first_player]}) & "
                    f"(G({first_player}_cutboard_station_ok))"
                ),
            },
            description="Player 0 keeps cutboard access protocol-safe.",
            tags=("promising", "station-clearance"),
        ),
        CandidateContractProfileSpec(
            profile_id="nonpromising_full_station_clearance",
            formulas={
                **baseline,
                first_player: (
                    f"({baseline[first_player]}) & "
                    f"(G({first_player}_station_clearance_ok))"
                ),
            },
            description="A conservative profile that requires all station-clearance checks.",
            tags=("nonpromising", "conservative"),
        ),
    )
