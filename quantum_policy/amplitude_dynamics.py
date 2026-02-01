import numpy as np

def stabilize(probabilities):
    """
    Prevents numerical collapse.
    """
    eps = 1e-6
    probs = np.clip(probabilities, eps, 1.0)
    return probs / probs.sum()
