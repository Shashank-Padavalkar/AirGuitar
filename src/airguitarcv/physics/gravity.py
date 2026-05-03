import numpy as np

def get_gravity_vector(strength: float = 9.8) -> np.ndarray:
    """Returns a 2D gravity vector (x, y) assuming y points down."""
    # Since y increases downwards in image coordinates
    return np.array([0.0, strength])
