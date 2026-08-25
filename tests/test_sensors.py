import numpy as np
import pytest

from src.sensors import quaternion_to_roll_pitch


def test_quaternion_pitch_matches_known_five_degree_y_rotation() -> None:
    half_angle = np.deg2rad(5.0) / 2.0
    quaternion_wxyz = np.array([np.cos(half_angle), 0.0, np.sin(half_angle), 0.0])

    roll, pitch = quaternion_to_roll_pitch(quaternion_wxyz)

    assert np.rad2deg(roll) == pytest.approx(0.0, abs=1e-12)
    assert np.rad2deg(pitch) == pytest.approx(5.0, abs=1e-12)
