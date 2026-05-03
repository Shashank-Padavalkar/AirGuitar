import cv2
import numpy as np
from airguitarcv.physics.rigid_body import RigidBody
from airguitarcv.constants import COLOR_CYAN, COLOR_YELLOW, COLOR_BLUE

class Hologram:
    def __init__(self):
        self.body_radius = 80
        self.neck_thickness = 20

    def draw(self, frame: np.ndarray, guitar: RigidBody) -> np.ndarray:
        """Render the simulated rigid body."""
        overlay = frame.copy()
        
        neck_pos, body_pos = guitar.get_points()
        
        pt1 = (int(neck_pos[0]), int(neck_pos[1]))
        pt2 = (int(body_pos[0]), int(body_pos[1]))
        
        # Draw neck
        cv2.line(overlay, pt1, pt2, COLOR_CYAN, self.neck_thickness)
        
        # Draw body
        cv2.circle(overlay, pt2, self.body_radius, COLOR_YELLOW, -1)
        
        # Draw grip zones for debug (optional)
        # We can calculate them based on length and angle
        
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        return frame
