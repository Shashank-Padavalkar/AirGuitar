import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from airguitarcv.detectors.base_detector import BaseDetector
from airguitarcv.config import PoseConfig

@dataclass
class Landmark:
    x: float
    y: float
    z: float
    visibility: float

@dataclass
class PoseResult:
    landmarks: Optional[Dict[int, Landmark]]
    mp_results: Optional[Any] # MediaPipe raw results for drawing

class PoseDetector(BaseDetector):
    def __init__(self, config: PoseConfig):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
            model_complexity=config.model_complexity
        )
        
    def process(self, image: np.ndarray) -> PoseResult:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.pose.process(image_rgb)
        
        landmarks_dict = None
        if results.pose_landmarks:
            landmarks_dict = {}
            for idx, lm in enumerate(results.pose_landmarks.landmark):
                landmarks_dict[idx] = Landmark(
                    x=lm.x, y=lm.y, z=lm.z, visibility=lm.visibility
                )
                
        return PoseResult(landmarks=landmarks_dict, mp_results=results)

    def close(self):
        self.pose.close()
