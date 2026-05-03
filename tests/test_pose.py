import pytest
from airguitarcv.detectors.pose_detector import PoseDetector
from airguitarcv.config import PoseConfig

def test_pose_detector_init():
    config = PoseConfig()
    detector = PoseDetector(config)
    assert detector is not None
    detector.close()
