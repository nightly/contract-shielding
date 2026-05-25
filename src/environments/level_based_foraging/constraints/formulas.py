"""LTL safety specifications for the Level-Based Foraging environment."""


GLOBAL_SAFETY_FORMULAS = {
    "avoid_failed_loads": "G(!{agent}_failed_load)",
}
