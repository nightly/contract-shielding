"""LTL safety specifications for the Cooking Zoo environment."""


GLOBAL_SAFETY_FORMULAS = {
    "kitchen_etiquette": "G(!({agent}_bad_delivery_attempt | {agent}_holding_teammate_only_ingredient | {agent}_blocking_delivery_access | {agent}_blocking_cutboard_access))",
    "no_bad_delivery": "G(!{agent}_bad_delivery_attempt)",
    "no_teammate_only_ingredient": "G(!{agent}_holding_teammate_only_ingredient)",
    "no_station_blocking": "G(!({agent}_blocking_delivery_access | {agent}_blocking_cutboard_access))",
}
