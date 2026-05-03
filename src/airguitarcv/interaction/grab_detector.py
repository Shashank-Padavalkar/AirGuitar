import numpy as np
import mediapipe as mp
from airguitarcv.constants import HandState
from airguitarcv.interaction.hand_tracker import HandResult

class GrabDetector:
    def __init__(self, grab_strength_threshold=0.8):
        self.mp_hands = mp.solutions.hands
        self.threshold = grab_strength_threshold
        
    def detect(self, hand: HandResult) -> HandState:
        """
        Classifies hand state into OPEN_HAND, HALF_CLOSED, or CLOSED_FIST.
        Uses distance from fingertips to palm.
        """
        lms = hand.landmarks
        
        wrist = lms[self.mp_hands.HandLandmark.WRIST]
        tips = [
            lms[self.mp_hands.HandLandmark.INDEX_FINGER_TIP],
            lms[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
            lms[self.mp_hands.HandLandmark.RING_FINGER_TIP],
            lms[self.mp_hands.HandLandmark.PINKY_TIP]
        ]
        mcps = [
            lms[self.mp_hands.HandLandmark.INDEX_FINGER_MCP],
            lms[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP],
            lms[self.mp_hands.HandLandmark.RING_FINGER_MCP],
            lms[self.mp_hands.HandLandmark.PINKY_MCP]
        ]
        
        # Calculate distance from each tip to wrist
        open_dists = [np.linalg.norm(t - wrist) for t in tips]
        # Calculate distance from MCP to wrist (reference for open hand)
        ref_dists = [np.linalg.norm(m - wrist) for m in mcps]
        
        # Ratio of tip distance to MCP distance
        # If tip is closer to wrist than MCP, finger is curled
        ratios = [d_tip / d_mcp for d_tip, d_mcp in zip(open_dists, ref_dists)]
        avg_ratio = sum(ratios) / len(ratios)
        
        # Heuristics:
        # avg_ratio > 1.8 -> OPEN
        # avg_ratio < 1.0 -> CLOSED
        
        if avg_ratio < 1.0:
            return HandState.CLOSED_FIST
        elif avg_ratio < 1.6:
            return HandState.HALF_CLOSED
        else:
            return HandState.OPEN_HAND
