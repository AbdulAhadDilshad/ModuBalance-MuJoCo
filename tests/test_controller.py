import math

import pytest

from src.controller import LocalObservation, PDConnectorController


def observation(pitch: float, angular_velocity: float = 0.0) -> LocalObservation:
    return LocalObservation(0.0, 0.0, pitch, 0.0, angular_velocity, 0.0, 0.0)


def test_pd_controller_uses_local_pitch_error() -> None:
    controller = PDConnectorController(kp=10.0, kd=2.0, max_torque=20.0)
    result = controller.compute_action(observation(math.radians(5.0), angular_velocity=0.1))
    assert result.error == pytest.approx(-math.radians(5.0))
    assert result.torque == pytest.approx(10.0 * -math.radians(5.0) - 0.2)
    assert not result.saturated


def test_pd_controller_saturates_symmetrically() -> None:
    controller = PDConnectorController(kp=100.0, kd=0.0, max_torque=3.0)
    assert controller.compute_action(observation(1.0)).torque == -3.0
    positive = controller.compute_action(observation(-1.0))
    assert positive.torque == 3.0
    assert positive.saturated


def test_pd_controller_rejects_invalid_limits() -> None:
    with pytest.raises(ValueError):
        PDConnectorController(kp=1.0, kd=1.0, max_torque=0.0)

