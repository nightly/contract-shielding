"""LTL safety specifications for the RWARE environment."""


GLOBAL_SAFETY_FORMULAS = {
    "all_agents_avoid_goal": "G(!{agent}_at_goal)",
    "requested_shelf_stays_requested_until_pickup": "G(requested_shelf_at_2_1 -> X(requested_shelf_at_2_1 | agent_0_carrying_requested | agent_1_carrying_requested))",
    "queue_priority_for_requested_carriers": "G(!({agent}_in_queue_zone & !{agent}_carrying_requested & {agent}_blocking_requested_carrier))",
    "queue_yield_protocol": "G({agent}_queue_yield_ok)",
    "queue_zone_only_when_loaded": "G({agent}_in_queue_zone -> {agent}_loaded)",
}
