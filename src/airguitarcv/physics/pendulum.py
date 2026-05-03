import numpy as np

def calculate_pendulum_angular_accel(
    angle: float, 
    pivot_pos: np.ndarray, 
    com_pos: np.ndarray, 
    gravity_strength: float, 
    stiffness: float
) -> float:
    """
    Calculate angular acceleration for a pendulum.
    Assumes angle 0 is pointing right (1, 0), PI/2 is pointing down (0, 1).
    """
    # Distance from pivot to center of mass
    r = np.linalg.norm(com_pos - pivot_pos)
    if r < 1e-4:
        return 0.0
        
    # The angle of the pendulum from the vertical downward direction
    # Vector pointing down is (0, 1)
    # The pendulum is pointing at `angle`.
    # Vector from pivot to com is (cos(angle), sin(angle))
    
    # Angle relative to downward vertical
    theta = angle - (np.pi / 2.0)
    
    # Angular acceleration: alpha = -(g/r) * sin(theta) * stiffness
    alpha = -(gravity_strength / r) * np.sin(theta) * stiffness
    return alpha
