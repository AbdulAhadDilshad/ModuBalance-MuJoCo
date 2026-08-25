"""Time-based physical disturbances; these never signal the controller."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class PayloadDisturbance:
    start_time: float
    equivalent_mass: float
    offset_x: float
    gravity: float = 9.81

    def apply(self, data: mujoco.MjData, body_id: int, site_id: int) -> bool:
        """Apply the equivalent payload force at its actual off-centre MJCF site."""
        data.xfrc_applied[body_id, :] = 0.0
        if data.time + 1e-12 < self.start_time:
            return False
        downward_force = self.equivalent_mass * self.gravity
        force = np.array([0.0, 0.0, -downward_force], dtype=float)
        # xfrc_applied torque is about the body's center of mass. Derive the lever
        # arm from MuJoCo world positions so it remains correct while Cube B pitches.
        lever_arm = np.asarray(data.site_xpos[site_id] - data.xipos[body_id], dtype=float)
        data.xfrc_applied[body_id, :3] = force
        data.xfrc_applied[body_id, 3:] = np.cross(lever_arm, force)
        return True


@dataclass(frozen=True)
class SmoothBaseMotion:
    start_time: float
    target_velocity: float
    ramp_duration: float

    def target(self, time: float) -> tuple[float, float]:
        """Position and velocity target with a half-cosine acceleration ramp."""
        elapsed = max(0.0, time - self.start_time)
        if elapsed <= 0.0:
            return 0.0, 0.0
        if elapsed < self.ramp_duration:
            phase = np.pi * elapsed / self.ramp_duration
            velocity = 0.5 * self.target_velocity * (1.0 - np.cos(phase))
            position = 0.5 * self.target_velocity * (
                elapsed - self.ramp_duration * np.sin(phase) / np.pi
            )
            return float(position), float(velocity)
        ramp_distance = 0.5 * self.target_velocity * self.ramp_duration
        position = ramp_distance + self.target_velocity * (elapsed - self.ramp_duration)
        return float(position), float(self.target_velocity)
