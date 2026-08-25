"""Named MuJoCo sensor access and quaternion conversion."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .controller import LocalObservation


def quaternion_to_roll_pitch(quaternion_wxyz: np.ndarray) -> tuple[float, float]:
    """Return intrinsic XYZ roll and pitch in radians from MuJoCo's wxyz quaternion."""
    w, x, y, z = (float(v) for v in quaternion_wxyz)
    sin_roll_cos_pitch = 2.0 * (w * x + y * z)
    cos_roll_cos_pitch = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sin_roll_cos_pitch, cos_roll_cos_pitch)
    sin_pitch = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sin_pitch)
    return float(roll), float(pitch)


class SensorError(RuntimeError):
    pass


@dataclass
class LocalSensorSuite:
    """Resolve sensors by name once, then build only a local observation."""

    model: mujoco.MjModel

    def __post_init__(self) -> None:
        required = (
            "hinge_position",
            "hinge_velocity",
            "cube_b_orientation",
            "cube_b_angular_velocity",
            "balance_actuator_force",
            "connector_force",
            "connector_torque",
            "platform_position",
            "platform_velocity",
        )
        missing = [name for name in required if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name) < 0]
        if missing:
            raise SensorError(f"MJCF is missing required sensors: {', '.join(missing)}")
        self._slices = {name: self._sensor_slice(name) for name in required}

    def _sensor_slice(self, name: str) -> slice:
        sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        start = int(self.model.sensor_adr[sensor_id])
        return slice(start, start + int(self.model.sensor_dim[sensor_id]))

    def values(self, data: mujoco.MjData, name: str) -> np.ndarray:
        return np.asarray(data.sensordata[self._slices[name]], dtype=float)

    def scalar(self, data: mujoco.MjData, name: str) -> float:
        return float(self.values(data, name)[0])

    def orientation(self, data: mujoco.MjData) -> tuple[float, float]:
        return quaternion_to_roll_pitch(self.values(data, "cube_b_orientation"))

    def observe(self, data: mujoco.MjData) -> LocalObservation:
        roll, pitch = self.orientation(data)
        angular_velocity = self.values(data, "cube_b_angular_velocity")
        connector_force = float(np.linalg.norm(self.values(data, "connector_force")))
        return LocalObservation(
            joint_angle=self.scalar(data, "hinge_position"),
            joint_velocity=self.scalar(data, "hinge_velocity"),
            module_pitch=pitch,
            module_roll=roll,
            module_angular_velocity_y=float(angular_velocity[1]),
            measured_joint_torque=self.scalar(data, "balance_actuator_force"),
            connector_force=connector_force,
        )

