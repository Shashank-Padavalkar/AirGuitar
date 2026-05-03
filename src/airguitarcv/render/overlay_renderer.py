import cv2
import numpy as np
import mediapipe as mp
from typing import Dict
from airguitarcv.constants import SystemState, HandState, COLOR_GREEN, COLOR_RED, COLOR_WHITE, COLOR_YELLOW, COLOR_CYAN, COLOR_BLUE
from airguitarcv.detectors.pose_detector import PoseResult
from airguitarcv.interaction.hand_tracker import HandResult
from airguitarcv.physics.rigid_body import RigidBody
from airguitarcv.render.hologram import Hologram

class OverlayRenderer:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.hologram = Hologram()

    def draw(
        self, 
        frame: np.ndarray, 
        pose_result: PoseResult, 
        state: SystemState, 
        is_guitar_pose: bool, 
        fps: float,
        hands: Dict[str, HandResult],
        hand_states: Dict[str, HandState],
        guitar: RigidBody,
        show_debug: bool = True
    ) -> np.ndarray:
        render_frame = frame.copy()
        h, w, _ = render_frame.shape
        
        # Draw pose skeleton
        if pose_result.mp_results and pose_result.mp_results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                render_frame,
                pose_result.mp_results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
        # Draw hands and hand states
        for hand_id, hand in hands.items():
            if hand.mp_landmarks:
                self.mp_drawing.draw_landmarks(
                    render_frame,
                    hand.mp_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
            
            # Display hand state text near wrist
            if hand_id in hand_states:
                state_text = hand_states[hand_id].name
                wrist = hand.landmarks[self.mp_hands.HandLandmark.WRIST]
                px, py = int(wrist[0] * w), int(wrist[1] * h)
                cv2.putText(render_frame, f"{hand_id}: {state_text}", (px, py - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2)
            
        # Render physics guitar
        if guitar:
            render_frame = self.hologram.draw(render_frame, guitar)
            
            if show_debug:
                neck_pos, body_pos = guitar.get_points()
                cv2.circle(render_frame, (int(neck_pos[0]), int(neck_pos[1])), 10, COLOR_RED, -1)
                cv2.circle(render_frame, (int(body_pos[0]), int(body_pos[1])), 10, COLOR_BLUE, -1)
                
                com_dir = (body_pos - neck_pos) * guitar.com_offset
                com_pos = neck_pos + com_dir
                cv2.circle(render_frame, (int(com_pos[0]), int(com_pos[1])), 8, COLOR_GREEN, -1)

        # Draw status text
        status_text = "NO GUITAR POSE"
        status_color = COLOR_RED
        
        if state == SystemState.GUITAR_POSE_VALID:
            status_text = "GUITAR POSE DETECTED"
            status_color = COLOR_GREEN
            
        cv2.putText(render_frame, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)
        cv2.putText(render_frame, f"FPS: {fps:.1f}", (w - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_WHITE, 2)
        
        return render_frame
