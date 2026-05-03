from enum import Enum, auto

class SystemState(Enum):
    INITIALIZING = auto()
    NO_POSE = auto()
    POSE_DETECTED = auto()
    GUITAR_POSE_VALID = auto()
    ERROR = auto()

class HandState(Enum):
    UNKNOWN = auto()
    OPEN_HAND = auto()
    HALF_CLOSED = auto()
    CLOSED_FIST = auto()

# Colors (BGR for OpenCV)
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_CYAN = (255, 255, 0)
