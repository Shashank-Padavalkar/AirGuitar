import cv2
import time
import numpy as np
import mediapipe as mp
from airguitarcv.config import AppConfig
from airguitarcv.logger import logger
from airguitarcv.constants import SystemState
from airguitarcv.detectors.pose_detector import PoseDetector
from airguitarcv.recognition.guitar_pose_classifier import GuitarPoseClassifier
from airguitarcv.render.overlay_renderer import OverlayRenderer
from airguitarcv.interaction.hand_tracker import HandTracker
from airguitarcv.interaction.grab_detector import GrabDetector
from airguitarcv.interaction.interaction_manager import InteractionManager
from airguitarcv.physics.rigid_body import RigidBody

class AirGuitarPipeline:
    def __init__(self, config: AppConfig):
        self.config = config
        self.state = SystemState.INITIALIZING
        
        # Initialize modules
        self.pose_detector = PoseDetector(config.pose)
        self.classifier = GuitarPoseClassifier(config.classifier)
        self.renderer = OverlayRenderer()
        
        # Phase 2 modules
        self.hand_tracker = HandTracker(
            min_det_conf=config.hand.min_detection_confidence,
            min_track_conf=config.hand.min_tracking_confidence,
            max_hands=config.hand.max_num_hands
        )
        self.grab_detector = GrabDetector(config.interaction.grab_strength)
        self.interaction_manager = InteractionManager(config)
        self.guitar = None
        
        # Initialize camera
        self.cap = cv2.VideoCapture(self.config.camera.device_id)
        if not self.cap.isOpened():
            logger.error(f"Failed to open camera {self.config.camera.device_id}")
            raise RuntimeError("Camera initialization failed")
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.config.camera.fps)
        
        logger.info("Pipeline initialized successfully")
        
    def run(self):
        prev_time = time.time()
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Failed to grab frame")
                break
                
            frame = cv2.flip(frame, 1) # Mirror image
            h, w, _ = frame.shape
            
            # 1. Detection
            pose_result = self.pose_detector.process(frame)
            hands = self.hand_tracker.process(frame)
            
            # 2. Recognition
            hand_states = {}
            for hid, hand in hands.items():
                hand_states[hid] = self.grab_detector.detect(hand)

            is_guitar_pose = False
            if pose_result.landmarks:
                self.state = SystemState.POSE_DETECTED
                is_guitar_pose, pose_confidence = self.classifier.classify(pose_result)
                
                if is_guitar_pose:
                    self.state = SystemState.GUITAR_POSE_VALID
            else:
                self.state = SystemState.NO_POSE
                
            # Physics Time step
            curr_time = time.time()
            dt = curr_time - prev_time if prev_time > 0 else 0.033
            prev_time = curr_time
                
            # Spawn guitar if valid pose and no guitar exists
            if is_guitar_pose and self.guitar is None and pose_result.landmarks:
                l_shoulder = pose_result.landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
                r_hip = pose_result.landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value]
                spawn_pos = np.array([l_shoulder.x * w, l_shoulder.y * h])
                length = np.linalg.norm(np.array([r_hip.x * w, r_hip.y * h]) - spawn_pos)
                self.guitar = RigidBody(position=spawn_pos, length=length)
                self.guitar.com_offset = self.config.physics.mass_distribution[2] # Example use
                
            # Interaction & Physics Update
            if self.guitar:
                self.interaction_manager.process_interactions(hands, hand_states, self.guitar, w, h)
                
                # We need pixel coords for hands
                hand_pixel_positions = {}
                for hid, hand in hands.items():
                    norm_pos = hand.landmarks[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_MCP]
                    hand_pixel_positions[hid] = np.array([norm_pos[0] * w, norm_pos[1] * h])
                    
                self.guitar.update(hand_pixel_positions, dt, self.config.physics)
                
                # Check out of bounds / despawn
                g_pos = self.guitar.position
                if g_pos[1] > h + self.guitar.length:
                    self.guitar = None # Despawn
            
            # 3. Render
            fps = 1 / dt if dt > 0 else 0
            
            render_frame = self.renderer.draw(
                frame=frame,
                pose_result=pose_result,
                state=self.state,
                is_guitar_pose=is_guitar_pose,
                fps=fps,
                hands=hands,
                hand_states=hand_states,
                guitar=self.guitar
            )
            
            # 4. Display
            cv2.imshow("AirGuitarCV", render_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def cleanup(self):
        logger.info("Cleaning up resources...")
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        if hasattr(self, 'pose_detector') and self.pose_detector:
            self.pose_detector.close()
        if hasattr(self, 'hand_tracker') and self.hand_tracker:
            self.hand_tracker.close()
