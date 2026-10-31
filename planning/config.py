"""Configuration for the advisory planner."""

import os


def narrative_planner_enabled() -> bool:
    """Read the feature flag without importing or changing application wiring."""
    return os.getenv("NARRATIVE_PLANNER_ENABLED", "true").strip().lower() in {
        "1", "true", "yes", "on"
    }
