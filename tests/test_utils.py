import pytest
from airguitarcv.utils.geometry import calculate_angle

def test_calculate_angle():
    a = (0, 1)
    b = (0, 0)
    c = (1, 0)
    angle = calculate_angle(a, b, c)
    assert abs(angle - 90.0) < 1e-5
