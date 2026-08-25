"""Local intelligent-module abstraction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .communication import AgentMessage
from .controller import ConnectorPolicy, ControlResult, LocalObservation


class IntelligentModuleAgent:
    """Cube B's local agent, deliberately isolated from MuJoCo global state."""

    def __init__(self, connector: ConnectorPolicy) -> None:
        self.connector = connector

    def act(self, observation: LocalObservation) -> ControlResult:
        return self.connector.compute_action(observation)


@dataclass(frozen=True)
class Phase2LocalObservation:
    """The complete information boundary for one Phase-2 distributed agent."""

    module_id: str
    local_pitch: float
    local_roll: float
    local_pitch_rate: float
    local_roll_rate: float
    local_joint_angle: float
    local_joint_velocity: float
    local_connector_force: float
    local_connector_torque: float
    connector_connected: bool


@dataclass(frozen=True)
class ModuleAction:
    ideal_torque: float
    pitch_signal: float
    roll_signal: float
    saturated: bool


class DistributedModuleAgent:
    """Local controller that can consume only observation + explicit messages."""

    def __init__(
        self,
        module_id: str,
        kp: float,
        kd: float,
        max_torque: float,
        neighbour_weight: float,
    ) -> None:
        self.module_id = module_id
        self.kp = float(kp)
        self.kd = float(kd)
        self.max_torque = float(max_torque)
        self.neighbour_weight = float(neighbour_weight)

    @staticmethod
    def message_payload(observation: Phase2LocalObservation) -> dict[str, float]:
        return {
            "pitch_error": -observation.local_pitch,
            "angular_velocity": observation.local_pitch_rate,
            "connector_load": observation.local_connector_force,
        }

    def act(
        self, observation: Phase2LocalObservation, messages: tuple[AgentMessage, ...]
    ) -> ModuleAction:
        local_raw = -self.kp * observation.local_pitch - self.kd * observation.local_pitch_rate
        neighbour_raw = 0.0
        if messages:
            pitch_errors = np.asarray([message.payload["pitch_error"] for message in messages])
            angular_velocities = np.asarray([message.payload["angular_velocity"] for message in messages])
            neighbour_raw = self.neighbour_weight * (
                self.kp * float(np.mean(pitch_errors)) - self.kd * float(np.mean(angular_velocities))
            )
        raw = local_raw + neighbour_raw if observation.connector_connected else 0.0
        torque = float(np.clip(raw, -self.max_torque, self.max_torque))
        return ModuleAction(
            ideal_torque=torque,
            pitch_signal=observation.local_pitch + self.neighbour_weight * (-neighbour_raw / max(self.kp, 1e-9)),
            roll_signal=observation.local_roll,
            saturated=not np.isclose(raw, torque),
        )
