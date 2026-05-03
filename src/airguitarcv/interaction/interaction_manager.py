import numpy as np
from typing import Dict, Optional, Tuple
from airguitarcv.config import AppConfig
from airguitarcv.constants import HandState
from airguitarcv.interaction.hand_tracker import HandResult
from airguitarcv.physics.rigid_body import RigidBody
from airguitarcv.physics.constraints import GrabConstraint, GrabZone
import mediapipe as mp

class InteractionManager:
    def __init__(self, config: AppConfig):
        self.config = config
        self.mp_hands = mp.solutions.hands
        
    def process_interactions(
        self, 
        hands: Dict[str, HandResult], 
        hand_states: Dict[str, HandState], 
        guitar: RigidBody,
        w: int,
        h: int
    ):
        """Update guitar constraints based on hand positions and states."""
        
        neck_pos, body_pos = guitar.get_points()
        
        # Define grip zones
        upper_grip_pos = guitar.position + (body_pos - neck_pos) * self.config.interaction.upper_grip_offset
        lower_grip_pos = guitar.position + (body_pos - neck_pos) * self.config.interaction.lower_grip_offset
        
        for hand_id, hand in hands.items():
            state = hand_states.get(hand_id, HandState.UNKNOWN)
            
            # Use middle MCP as the "grab center" and convert to pixel coordinates
            norm_center = hand.landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
            hand_center = np.array([norm_center[0] * w, norm_center[1] * h])
            
            # Check if this hand is currently grabbing
            is_grabbing = (hand_id in guitar.constraints)
            
            if state == HandState.CLOSED_FIST:
                if not is_grabbing:
                    # Try to grab
                    dist_to_upper = np.linalg.norm(hand_center - upper_grip_pos)
                    dist_to_lower = np.linalg.norm(hand_center - lower_grip_pos)
                    
                    # Grab radius in pixel coords
                    grab_radius = self.config.interaction.grab_radius * w
                    
                    if dist_to_upper < grab_radius:
                        # Grab upper
                        guitar.add_constraint(hand_id, GrabConstraint(
                            zone=GrabZone.UPPER,
                            local_offset=self.config.interaction.upper_grip_offset,
                            hand_pos=hand_center
                        ))
                    elif dist_to_lower < grab_radius:
                        # Grab lower
                        guitar.add_constraint(hand_id, GrabConstraint(
                            zone=GrabZone.LOWER,
                            local_offset=self.config.interaction.lower_grip_offset,
                            hand_pos=hand_center
                        ))
            elif state == HandState.OPEN_HAND:
                if is_grabbing:
                    # Release
                    guitar.remove_constraint(hand_id)
