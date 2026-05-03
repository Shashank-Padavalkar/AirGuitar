from typing import Tuple
from airguitarcv.detectors.pose_detector import PoseResult
from airguitarcv.config import ClassifierConfig
from airguitarcv.logger import logger
import mediapipe as mp

class GuitarPoseClassifier:
    def __init__(self, config: ClassifierConfig):
        self.config = config
        self.mp_pose = mp.solutions.pose
        
    def classify(self, pose_result: PoseResult) -> Tuple[bool, float]:
        """
        Heuristic classifier for guitar pose.
        Returns: (is_guitar_pose, confidence_score)
        """
        if not pose_result.landmarks:
            return False, 0.0
            
        landmarks = pose_result.landmarks
        
        # Get required landmarks
        required_idx = [
            self.mp_pose.PoseLandmark.LEFT_SHOULDER.value,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
            self.mp_pose.PoseLandmark.LEFT_ELBOW.value,
            self.mp_pose.PoseLandmark.RIGHT_ELBOW.value,
            self.mp_pose.PoseLandmark.LEFT_WRIST.value,
            self.mp_pose.PoseLandmark.RIGHT_WRIST.value,
            self.mp_pose.PoseLandmark.LEFT_HIP.value,
            self.mp_pose.PoseLandmark.RIGHT_HIP.value
        ]
        
        # Check visibility
        for idx in required_idx:
            if idx not in landmarks or landmarks[idx].visibility < 0.5:
                return False, 0.0
                
        l_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        r_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        l_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
        r_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        l_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        r_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        
        # Heuristics for guitar stance (Assuming right-handed player)
        # 1. Left hand (fret hand) is generally extended outwards and higher than hips
        # 2. Right hand (strum hand) is lower, near the torso
        # 3. Both wrists above hips
        
        wrists_above_hips = (l_wrist.y < l_hip.y) and (r_wrist.y < r_hip.y)
        
        # Note: y=0 is top, y=1 is bottom
        y_center = (l_shoulder.y + r_shoulder.y) / 2
        hip_y_center = (l_hip.y + r_hip.y) / 2
        
        l_wrist_valid = l_wrist.y < hip_y_center
        r_wrist_valid = r_wrist.y < hip_y_center
        
        score = 0.0
        
        if wrists_above_hips:
            score += 0.4
            
        # Camera mirrored, so left side of person is positive x (right side of image)
        if l_wrist.x > l_shoulder.x: 
            score += 0.3
            
        # Right wrist near center
        if abs(r_wrist.x - l_shoulder.x) < 0.3: 
            score += 0.3
            
        is_guitar = score >= self.config.confidence_threshold
        
        return is_guitar, score
