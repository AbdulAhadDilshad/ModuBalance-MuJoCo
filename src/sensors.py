"""Named MuJoCo sensor access and quaternion conversion."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .controller import LocalObservation
from .agent import Phase2LocalObservation


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


class Phase2SensorSuite:
    """Named-sensor boundary for A/B/C; policies never receive MjData."""

    def __init__(self, model: mujoco.MjModel, rng: np.random.Generator, noise: dict[str, float]) -> None:
        self.model = model
        self.rng = rng
        self.orientation_noise = np.deg2rad(float(noise["orientation_std_deg"]))
        self.rate_noise = float(noise["angular_velocity_std"])
        self.force_noise = float(noise["force_std"])
        names = [
            *(f"orientation_{module}" for module in "ABC"),
            *(f"angular_velocity_{module}" for module in "ABC"),
            "position_AB", "velocity_AB", "position_BC", "velocity_BC",
            "force_AB", "torque_AB", "force_BC", "torque_BC",
            "phase2_platform_position", "phase2_platform_velocity",
        ]
        self._slices: dict[str, slice] = {}
        for name in names:
            sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            if sensor_id < 0:
                raise SensorError(f"Phase-2 MJCF is missing sensor '{name}'")
            start = int(model.sensor_adr[sensor_id])
            self._slices[name] = slice(start, start + int(model.sensor_dim[sensor_id]))

    def values(self, data: mujoco.MjData, name: str) -> np.ndarray:
        return np.asarray(data.sensordata[self._slices[name]], dtype=float)

    def scalar(self, data: mujoco.MjData, name: str) -> float:
        return float(self.values(data, name)[0])

    def observe_module(
        self,
        data: mujoco.MjData,
        module_id: str,
        connector_force: float,
        connector_torque: float,
        connected: bool,
    ) -> Phase2LocalObservation:
        roll, pitch = quaternion_to_roll_pitch(self.values(data, f"orientation_{module_id}"))
        gyro = self.values(data, f"angular_velocity_{module_id}")
        pitch += float(self.rng.normal(0.0, self.orientation_noise))
        roll += float(self.rng.normal(0.0, self.orientation_noise))
        if module_id == "A":
            joint_angle, joint_velocity = 0.0, 0.0
        elif module_id == "B":
            joint_angle = self.scalar(data, "position_AB")
            joint_velocity = self.scalar(data, "velocity_AB")
        else:
            joint_angle = self.scalar(data, "position_BC")
            joint_velocity = self.scalar(data, "velocity_BC")
        return Phase2LocalObservation(
            module_id=module_id,
            local_pitch=pitch,
            local_roll=roll,
            local_pitch_rate=float(gyro[1] + self.rng.normal(0.0, self.rate_noise)),
            local_roll_rate=float(gyro[0] + self.rng.normal(0.0, self.rate_noise)),
            local_joint_angle=joint_angle,
            local_joint_velocity=joint_velocity,
            local_connector_force=max(0.0, connector_force + float(self.rng.normal(0.0, self.force_noise))),
            local_connector_torque=connector_torque,
            connector_connected=connected,
        )
