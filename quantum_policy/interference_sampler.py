import numpy as np

def sample(probabilities):
    """
    Samples according to quantum measurement outcomes.
    """
    return np.random.choice(len(probabilities), p=probabilities)
