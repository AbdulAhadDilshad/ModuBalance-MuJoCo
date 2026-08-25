"""Bounded four-element magnetic connector using actual MuJoCo site forces."""

from __future__ import annotations

import mujoco
import numpy as np

from .base import ConnectorMeasurement


def allocate_four_magnet_commands(
    nominal: float,
    pitch: float,
    pitch_rate: float,
    roll: float,
    roll_rate: float,
    pitch_gain: float,
    pitch_rate_gain: float,
    roll_gain: float,
    roll_rate_gain: float,
) -> tuple[np.ndarray, int]:
    """Map pitch/roll correction into front/rear and left/right differentials."""
    pitch_delta = -(pitch_gain * pitch + pitch_rate_gain * pitch_rate)
    roll_delta = roll_gain * roll + roll_rate_gain * roll_rate
    raw = np.array(
        [
            nominal + pitch_delta + roll_delta,  # front-left
            nominal + pitch_delta - roll_delta,  # front-right
            nominal - pitch_delta + roll_delta,  # rear-left
            nominal - pitch_delta - roll_delta,  # rear-right
        ],
        dtype=float,
    )
    commands = np.clip(raw, 0.0, 1.0)
    return commands, int(np.count_nonzero(~np.isclose(raw, commands)))


class MagneticConnector:
    """Four attraction pairs with F=u*Fmax/(1+(d/d0)^2)."""

    def __init__(
        self,
        name: str,
        lower_body_id: int,
        upper_body_id: int,
        lower_site_ids: list[int],
        upper_site_ids: list[int],
        maximum_force: float,
        distance_scale: float,
    ) -> None:
        if len(lower_site_ids) != 4 or len(upper_site_ids) != 4:
            raise ValueError("MagneticConnector requires exactly four lower and four upper sites")
        if maximum_force <= 0 or distance_scale <= 0:
            raise ValueError("Magnetic force and distance scale must be positive")
        self.name = name
        self.lower_body_id = lower_body_id
        self.upper_body_id = upper_body_id
        self.lower_site_ids = lower_site_ids
        self.upper_site_ids = upper_site_ids
        self.maximum_force = float(maximum_force)
        self.distance_scale = float(distance_scale)
        self._last = ConnectorMeasurement(0.0, 0.0, np.zeros(4), np.zeros(4), 0)

    def apply(self, model: mujoco.MjModel, data: mujoco.MjData, command: object) -> ConnectorMeasurement:
        raw_commands = np.asarray(command, dtype=float)
        commands = np.clip(raw_commands, 0.0, 1.0)
        forces = np.zeros(4, dtype=float)
        separations = np.zeros(4, dtype=float)
        upper_torque = np.zeros(3, dtype=float)
        zero_torque = np.zeros(3, dtype=float)
        for index, (lower_site, upper_site) in enumerate(zip(self.lower_site_ids, self.upper_site_ids)):
            lower_point = np.asarray(data.site_xpos[lower_site], dtype=float).copy()
            upper_point = np.asarray(data.site_xpos[upper_site], dtype=float).copy()
            displacement = lower_point - upper_point
            distance = float(np.linalg.norm(displacement))
            separations[index] = distance
            direction = displacement / distance if distance > 1e-9 else np.array([0.0, 0.0, -1.0])
            attenuation = 1.0 / (1.0 + (distance / self.distance_scale) ** 2)
            magnitude = float(commands[index] * self.maximum_force * attenuation)
            force_on_upper = magnitude * direction
            mujoco.mj_applyFT(
                model, data, force_on_upper, zero_torque, upper_point, self.upper_body_id, data.qfrc_applied
            )
            mujoco.mj_applyFT(
                model, data, -force_on_upper, zero_torque, lower_point, self.lower_body_id, data.qfrc_applied
            )
            forces[index] = magnitude
            upper_torque += np.cross(upper_point - data.xipos[self.upper_body_id], force_on_upper)
        self._last = ConnectorMeasurement(
            force=float(np.sum(forces)),
            torque=float(np.linalg.norm(upper_torque)),
            commands=commands.copy(),
            element_forces=forces,
            saturated_elements=int(np.count_nonzero((commands <= 1e-12) | (commands >= 1.0 - 1e-12))),
            max_separation=float(np.max(separations)),
        )
        return self._last

    def observe(self) -> ConnectorMeasurement:
        return self._last
