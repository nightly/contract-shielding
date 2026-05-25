"""LTL safety specifications for the pressure plate environment."""


GLOBAL_SAFETY_FORMULAS = {
    "no_goal_before_first_door": "!F({agent}_at_goal & !door_0_open)",
    "goal_runner_waits_for_first_door": "(!agent_2_at_goal) W door_0_open",
    "goal_runner_requires_open_doors": "G(({agent}_after_door_0 -> door_0_open) & ({agent}_after_door_1 -> door_1_open))",
    "runner_requires_open_doors": "G((agent_2_after_door_0 -> door_0_open) & (agent_2_after_door_1 -> door_1_open))",
    "runner_crosses_open_doors_strict": (
        "G(((agent_2_crossing_door_0 | agent_2_in_door_0) -> door_0_open) & "
        "((agent_2_crossing_door_1 | agent_2_in_door_1) -> door_1_open))"
    ),
    "runner_requires_return_routes": (
        "G(((!goal_achieved & (agent_2_in_door_0 | agent_2_after_door_0)) -> "
        "agent_2_can_return_before_door_0) & "
        "((!goal_achieved & (agent_2_in_door_1 | agent_2_after_door_1)) -> "
        "agent_2_can_return_before_door_1))"
    ),
    "door_0_requires_plate_0": "G(door_0_open -> plate_0_pressed)",
    "all_agents_not_at_goal": "G(!{agent}_at_goal)",
}
