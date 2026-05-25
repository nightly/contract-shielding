"""LTL safety specifications for the car platoon environment."""


GLOBAL_SAFETY_FORMULAS = {
    "maintain_safe_gap": "G({agent}_gap_safe)",
    "maintain_safe_gap_decomposed": "G(!({agent}_crashed | {agent}_too_far))",
    "avoid_collision": "G(!{agent}_crashed)",
    "avoid_falling_behind": "G(!{agent}_too_far)",
}
