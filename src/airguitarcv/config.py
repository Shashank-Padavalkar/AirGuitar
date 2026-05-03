import yaml
from dataclasses import dataclass
from pathlib import Path

@dataclass
class CameraConfig:
    device_id: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30

@dataclass
class PoseConfig:
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    model_complexity: int = 1

@dataclass
class HandConfig:
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    max_num_hands: int = 2

@dataclass
class ClassifierConfig:
    confidence_threshold: float = 0.6

@dataclass
class InteractionConfig:
    grab_radius: float = 0.15 # Normalized screen distance
    grab_strength: float = 0.8
    upper_grip_offset: float = 0.1 # 10% from neck
    lower_grip_offset: float = 0.8 # 20% from bottom is 0.8 from neck

@dataclass
class PhysicsConfig:
    gravity_strength: float = 9.8
    mass_distribution: list = None
    damping: float = 0.95
    pendulum_stiffness: float = 0.8
    release_timeout: float = 2.0
    fade_duration: float = 1.0
    smoothing_factor: float = 0.2
    
    def __post_init__(self):
        if self.mass_distribution is None:
            self.mass_distribution = [0.2, 0.4, 0.4]

@dataclass
class AppConfig:
    camera: CameraConfig
    pose: PoseConfig
    hand: HandConfig
    classifier: ClassifierConfig
    interaction: InteractionConfig
    physics: PhysicsConfig

def load_config(config_path: str) -> AppConfig:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    with open(path, "r") as f:
        data = yaml.safe_load(f)
        
    return AppConfig(
        camera=CameraConfig(**data.get("camera", {})),
        pose=PoseConfig(**data.get("pose", {})),
        hand=HandConfig(**data.get("hand", {})),
        classifier=ClassifierConfig(**data.get("classifier", {})),
        interaction=InteractionConfig(**data.get("interaction", {})),
        physics=PhysicsConfig(**data.get("physics", {})),
    )
