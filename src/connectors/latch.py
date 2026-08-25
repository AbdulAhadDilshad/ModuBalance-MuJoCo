"""Dwell/hysteresis controller for runtime MuJoCo weld latches."""

from __future__ import annotations

import mujoco
import numpy as np


LATCH_OPEN = 0
LATCH_LOCKED = 1


class MagneticLatchController:
    def __init__(
        self,
        equality_id: int,
        angle_threshold: float,
        unlock_angle: float,
        angular_velocity_threshold: float,
        unlock_angular_velocity: float,
        minimum_stable_time: float,
    ) -> None:
        self.equality_id = equality_id
        self.angle_threshold = angle_threshold
        self.unlock_angle = unlock_angle
        self.angular_velocity_threshold = angular_velocity_threshold
        self.unlock_angular_velocity = unlock_angular_velocity
        self.minimum_stable_time = minimum_stable_time
        self.state = LATCH_OPEN
        self._stable_since: float | None = None
        self.transitions = 0

    def update(self, data: mujoco.MjData, angle: float, angular_velocity: float, time: float) -> int:
        if self.state == LATCH_OPEN:
            stable = abs(angle) <= self.angle_threshold and abs(angular_velocity) <= self.angular_velocity_threshold
            self._stable_since = time if stable and self._stable_since is None else self._stable_since
            if not stable:
                self._stable_since = None
            if self._stable_since is not None and time - self._stable_since >= self.minimum_stable_time:
                self.state = LATCH_LOCKED
                self.transitions += 1
                data.eq_active[self.equality_id] = 1
        elif abs(angle) >= self.unlock_angle or abs(angular_velocity) >= self.unlock_angular_velocity:
            self.state = LATCH_OPEN
            self.transitions += 1
            self._stable_since = None
            data.eq_active[self.equality_id] = 0
        return self.state

