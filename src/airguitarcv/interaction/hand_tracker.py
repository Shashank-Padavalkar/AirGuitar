import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass

@dataclass
class HandResult:
    landmarks: Dict[int, np.ndarray] # Landmark ID to [x, y, z] normalized
    handedness: str # "Left" or "Right"
    mp_landmarks: Any # For drawing

class HandTracker:
    def __init__(self, min_det_conf=0.5, min_track_conf=0.5, max_hands=2):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=min_det_conf,
            min_tracking_confidence=min_track_conf
        )
        
    def process(self, frame: np.ndarray) -> Dict[str, HandResult]:
        """Returns a dict mapping hand_id ('Left', 'Right') to HandResult."""
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False
        results = self.hands.process(img_rgb)
        
        tracked_hands = {}
        if results.multi_hand_landmarks and results.multi_handedness:
            for idx, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
                # Note: MediaPipe handedness is flipped when using front-facing camera unless mirrored.
                # Since we mirror the frame in pipeline.py before passing it here, 'Left' means user's Left hand.
                label = handedness.classification[0].label
                
                landmarks_dict = {}
                for lm_idx, lm in enumerate(hand_landmarks.landmark):
                    landmarks_dict[lm_idx] = np.array([lm.x, lm.y, lm.z])
                    
                tracked_hands[label] = HandResult(
                    landmarks=landmarks_dict,
                    handedness=label,
                    mp_landmarks=hand_landmarks
                )
                
        return tracked_hands
        
    def close(self):
        self.hands.close()
