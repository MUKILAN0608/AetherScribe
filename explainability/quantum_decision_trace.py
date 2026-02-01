def quantum_trace(agent, action, probability):
    return (
        f"{agent} selected '{action}' "
        f"with measured quantum probability {probability:.3f}."
    )
