"""Phase-1-compatible ideal torque connector baseline."""

from __future__ import annotations

import mujoco
import numpy as np

from .base import ConnectorMeasurement


class IdealTorqueConnector:
    def __init__(
        self, name: str, actuator_id: int, max_torque: float, roll_actuator_id: int | None = None
    ) -> None:
        self.name = name
        self.actuator_id = actuator_id
        self.roll_actuator_id = roll_actuator_id
        self.max_torque = float(max_torque)
        self._last = ConnectorMeasurement(0.0, 0.0, np.zeros(4), np.zeros(4), 0)

    def apply(self, model: mujoco.MjModel, data: mujoco.MjData, command: object) -> ConnectorMeasurement:
        requested = np.atleast_1d(np.asarray(command, dtype=float))
        raw_pitch = float(requested[0])
        raw_roll = float(requested[1]) if requested.size > 1 else 0.0
        pitch_torque = float(np.clip(raw_pitch, -self.max_torque, self.max_torque))
        roll_torque = float(np.clip(raw_roll, -self.max_torque, self.max_torque))
        data.ctrl[self.actuator_id] = pitch_torque
        if self.roll_actuator_id is not None:
            data.ctrl[self.roll_actuator_id] = roll_torque
        normalized = np.array(
            [abs(pitch_torque), abs(pitch_torque), abs(roll_torque), abs(roll_torque)]
        ) / self.max_torque
        self._last = ConnectorMeasurement(
            force=0.0,
            torque=float(np.hypot(pitch_torque, roll_torque)),
            commands=normalized,
            element_forces=np.zeros(4),
            saturated_elements=int(not np.isclose(raw_pitch, pitch_torque))
            + int(not np.isclose(raw_roll, roll_torque)),
        )
        return self._last

    def observe(self) -> ConnectorMeasurement:
        return self._last
