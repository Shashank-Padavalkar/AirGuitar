import numpy as np
from typing import Dict
from airguitarcv.physics.constraints import GrabConstraint, GrabZone
from airguitarcv.physics.pendulum import calculate_pendulum_angular_accel

class RigidBody:
    def __init__(self, position: np.ndarray, length: float):
        self.position = np.array(position, dtype=float) # Neck position
        self.angle = 0.0 # Radians
        self.velocity = np.zeros(2)
        self.angular_velocity = 0.0
        
        self.length = length
        self.com_offset = 0.7 # Center of mass is 70% down the guitar
        
        self.constraints: Dict[str, GrabConstraint] = {}
        
    def add_constraint(self, hand_id: str, constraint: GrabConstraint):
        self.constraints[hand_id] = constraint
        
    def remove_constraint(self, hand_id: str):
        if hand_id in self.constraints:
            del self.constraints[hand_id]
            
    def update(self, hand_positions: Dict[str, np.ndarray], dt: float, config: 'PhysicsConfig'):
        # Update constraint hand positions
        for hand_id, pos in hand_positions.items():
            if hand_id in self.constraints:
                self.constraints[hand_id].hand_pos = np.array(pos, dtype=float)

        num_constraints = len(self.constraints)
        
        if num_constraints == 2:
            # Two hand grab - fully constrained
            c_list = list(self.constraints.values())
            c1, c2 = c_list[0], c_list[1]
            
            # Ensure c1 is upper, c2 is lower
            if c1.local_offset > c2.local_offset:
                c1, c2 = c2, c1
                
            p1, p2 = c1.hand_pos, c2.hand_pos
            
            # Calculate angle
            delta = p2 - p1
            target_angle = np.arctan2(delta[1], delta[0])
            
            # Smooth angle transition
            self.angle = self._lerp_angle(self.angle, target_angle, 1.0 - config.smoothing_factor)
            
            # Calculate neck position
            dir_vec = np.array([np.cos(self.angle), np.sin(self.angle)])
            target_neck = p1 - dir_vec * (self.length * c1.local_offset)
            
            self.position = self._lerp_pos(self.position, target_neck, 1.0 - config.smoothing_factor)
            
            self.velocity = np.zeros(2)
            self.angular_velocity = 0.0
            
        elif num_constraints == 1:
            # One hand grab - pendulum
            c = list(self.constraints.values())[0]
            pivot = c.hand_pos
            
            # Center of mass position
            dir_vec = np.array([np.cos(self.angle), np.sin(self.angle)])
            com_pos = self.position + dir_vec * (self.length * self.com_offset)
            
            # Calculate pendulum acceleration
            alpha = calculate_pendulum_angular_accel(
                angle=self.angle,
                pivot_pos=pivot,
                com_pos=com_pos,
                gravity_strength=config.gravity_strength,
                stiffness=config.pendulum_stiffness
            )
            
            self.angular_velocity += alpha * dt
            self.angular_velocity *= config.damping
            self.angle += self.angular_velocity * dt
            
            # Maintain pivot constraint
            dir_vec = np.array([np.cos(self.angle), np.sin(self.angle)])
            self.position = pivot - dir_vec * (self.length * c.local_offset)
            
            self.velocity = np.zeros(2)
            
        else:
            # Free fall
            self.velocity[1] += config.gravity_strength * dt
            self.position += self.velocity * dt
            
            self.angular_velocity *= config.damping
            self.angle += self.angular_velocity * dt
            
    def _lerp_angle(self, a: float, b: float, t: float) -> float:
        # Shortest path interpolation for angles
        diff = (b - a + np.pi) % (2 * np.pi) - np.pi
        return a + diff * t
        
    def _lerp_pos(self, a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
        return a + (b - a) * t

    def get_points(self):
        """Returns (neck_pos, body_end_pos)"""
        dir_vec = np.array([np.cos(self.angle), np.sin(self.angle)])
        end_pos = self.position + dir_vec * self.length
        return self.position, end_pos
