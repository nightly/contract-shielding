"""Hand-authored Pressure Plate contract profiles for debug runs."""

from __future__ import annotations

from src.experiments import CandidateContractProfileSpec


def _runner_open_door_formula(runner: str = "agent_2") -> str:
    return f"G(({runner}_after_door_0 -> door_0_open) & ({runner}_after_door_1 -> door_1_open))"


def _runner_strict_crossing_formula(runner: str = "agent_2") -> str:
    return (
        f"G((({runner}_crossing_door_0 | {runner}_in_door_0) -> door_0_open) & "
        f"(({runner}_crossing_door_1 | {runner}_in_door_1) -> door_1_open))"
    )


def _runner_return_routes_formula(runner: str = "agent_2") -> str:
    return (
        f"G(((!goal_achieved & ({runner}_in_door_0 | {runner}_after_door_0)) -> "
        f"{runner}_can_return_before_door_0) & "
        f"((!goal_achieved & ({runner}_in_door_1 | {runner}_after_door_1)) -> "
        f"{runner}_can_return_before_door_1))"
    )


def candidate_contract_profiles(
    *,
    formula_name: str,
    global_formula: str,
    agent_ids: tuple[str, ...],
) -> tuple[CandidateContractProfileSpec, ...]:
    del global_formula
    runner = "agent_2" if "agent_2" in agent_ids else agent_ids[-1]
    if formula_name == "runner_requires_return_routes":
        global_obligation = _runner_return_routes_formula(runner)
        holder_formulas = {agent_id: "t" for agent_id in agent_ids}
        holder_formulas[runner] = global_obligation
        if "agent_0" in holder_formulas:
            holder_formulas["agent_0"] = (
                f"G((!goal_achieved & ({runner}_waiting_at_door_0 | "
                f"{runner}_in_door_0 | {runner}_after_door_0)) -> X(door_0_open))"
            )
        if "agent_1" in holder_formulas:
            holder_formulas["agent_1"] = (
                f"G((!goal_achieved & ({runner}_waiting_at_door_1 | "
                f"{runner}_in_door_1 | {runner}_after_door_1)) -> X(door_1_open))"
            )
        return (
            CandidateContractProfileSpec(
                profile_id="holders_preserve_return_routes",
                formulas=holder_formulas,
                description=(
                    "Door holders keep the route back to the outside of each "
                    "door available until the goal is reached."
                ),
                tags=("promising", "return-route", "door-phase"),
            ),
        )

    if formula_name == "runner_crosses_open_doors_strict":
        global_obligation = _runner_strict_crossing_formula(runner)
        holder_phase_formulas = {agent_id: "t" for agent_id in agent_ids}
        holder_phase_formulas[runner] = global_obligation
        if "agent_0" in holder_phase_formulas:
            holder_phase_formulas["agent_0"] = (
                f"G(({runner}_waiting_at_door_0 | {runner}_in_door_0) -> X(door_0_open))"
            )
        if "agent_1" in holder_phase_formulas:
            holder_phase_formulas["agent_1"] = (
                f"G(({runner}_waiting_at_door_1 | {runner}_in_door_1) -> X(door_1_open))"
            )
        return (
            CandidateContractProfileSpec(
                profile_id="phase_holders_keep_crossing_doors_open",
                formulas=holder_phase_formulas,
                description=(
                    "Door holders keep the relevant door open while the runner is "
                    "waiting at or occupying that doorway."
                ),
                tags=("promising", "strict-crossing", "door-phase"),
            ),
        )

    if formula_name != "runner_requires_open_doors":
        return tuple()
    global_obligation = _runner_open_door_formula(runner)
    baseline = {agent_id: "t" for agent_id in agent_ids}
    baseline[runner] = global_obligation
    return (
        CandidateContractProfileSpec(
            profile_id="baseline_runner_open_doors",
            formulas=baseline,
            description="The runner carries the global open-door-before-crossing obligation.",
            tags=("baseline",),
        ),
        CandidateContractProfileSpec(
            profile_id="promising_runner_not_waiting_door_0",
            formulas={
                **baseline,
                runner: f"({global_obligation}) & (!{runner}_waiting_at_door_0)",
            },
            description="The runner avoids the first-door waiting phase at reset.",
            tags=("promising", "door-phase"),
        ),
        CandidateContractProfileSpec(
            profile_id="nonpromising_runner_never_crosses",
            formulas={
                **baseline,
                runner: f"G(!{runner}_after_door_0) & G(!{runner}_after_door_1)",
            },
            description="A conservative profile that satisfies safety by preventing crossing.",
            tags=("nonpromising", "conservative"),
        ),
    )
