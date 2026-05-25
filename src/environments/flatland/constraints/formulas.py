"""LTL safety specifications for the Flatland RailwayEnv environment."""


GLOBAL_SAFETY_FORMULAS = {
    "avoid_deadlocks": "G(!{agent}_deadlocked)",
}
