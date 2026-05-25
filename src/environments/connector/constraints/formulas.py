"""LTL safety specifications for the Connector environment."""


GLOBAL_SAFETY_FORMULAS = {
    "avoid_blocking": "G(!{agent}_blocked)",
    "avoid_reserved_route_blocking": "G(!{agent}_reserved_route_blocked)",
    "preserve_reserved_route_clearance": "G({agent}_reserved_route_clear_ok)",
    "respect_reserved_routes": "G({agent}_respects_reservations_ok)",
}
