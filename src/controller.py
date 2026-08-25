"""Replaceable connector-control policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class LocalObservation:
    """Measurements available locally at the Cube B connector."""

    joint_angle: float
    joint_velocity: float
    module_pitch: float
    module_roll: float
    module_angular_velocity_y: float
    measured_joint_torque: float
    connector_force: float


@dataclass(frozen=True)
class ControlResult:
    torque: float
    error: float
    saturated: bool


class ConnectorPolicy(Protocol):
    """Interface future adaptive, MPC, RL, or magnetic policies can implement."""

    def compute_action(self, observation: LocalObservation) -> ControlResult:
        """Compute one connector command using local measurements only."""


class PDConnectorController:
    """Local pitch PD controller with symmetric torque saturation."""

    def __init__(self, kp: float, kd: float, max_torque: float, desired_pitch: float = 0.0) -> None:
        if kp < 0 or kd < 0 or max_torque <= 0:
            raise ValueError("kp and kd must be non-negative; max_torque must be positive")
        self.kp = float(kp)
        self.kd = float(kd)
        self.max_torque = float(max_torque)
        self.desired_pitch = float(desired_pitch)

    def compute_action(self, observation: LocalObservation) -> ControlResult:
        error = self.desired_pitch - observation.module_pitch
        raw_torque = self.kp * error - self.kd * observation.module_angular_velocity_y
        torque = float(np.clip(raw_torque, -self.max_torque, self.max_torque))
        return ControlResult(torque=torque, error=error, saturated=not np.isclose(raw_torque, torque))


class OffConnectorController:
    """Scientific passive baseline: never supplies corrective torque."""

    def __init__(self, desired_pitch: float = 0.0) -> None:
        self.desired_pitch = float(desired_pitch)

    def compute_action(self, observation: LocalObservation) -> ControlResult:
        return ControlResult(
            torque=0.0,
            error=self.desired_pitch - observation.module_pitch,
            saturated=False,
        )

