import numpy as np
import pytest

from src.evaluation import settling_time


def test_settling_time_requires_continuous_dwell() -> None:
    time = np.arange(0.0, 4.1, 0.1)
    pitch = np.full_like(time, 3.0)
    pitch[(time >= 1.0) & (time < 1.5)] = 1.0
    pitch[time >= 2.0] = 1.0
    result = settling_time(time, pitch, 0.0, 4.0, tolerance_deg=2.0, dwell_time=1.0)
    assert result == pytest.approx(2.0)


def test_settling_time_returns_none_without_dwell() -> None:
    time = np.arange(0.0, 2.1, 0.1)
    pitch = np.full_like(time, 2.1)
    assert settling_time(time, pitch, 0.0, 2.0, 2.0, 1.0) is None

