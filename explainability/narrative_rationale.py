def narrative_rationale(state):
    """
    Returns the professional narrative context for AetherScribe audits.
    """
    return (
        f"Orchestration calibrated for {state.genre.upper()} constraints "
        f"at Phase: {state.phase.upper()} | Narrative Wavefront Tension: {state.tension:.2f}."
    )
