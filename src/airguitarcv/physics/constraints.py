from dataclasses import dataclass
import numpy as np
from enum import Enum

class GrabZone(Enum):
    UPPER = "upper"
    LOWER = "lower"

@dataclass
class GrabConstraint:
    zone: GrabZone
    local_offset: float # 0.0 is neck, 1.0 is bottom
    hand_pos: np.ndarray # Current position of the grabbing hand
