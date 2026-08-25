"""Connector protocol data shared by ideal and magnetic implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import mujoco
import numpy as np


@dataclass(frozen=True)
class ConnectorMeasurement:
    force: float
    torque: float
    commands: np.ndarray
    element_forces: np.ndarray
    saturated_elements: int = 0


class Connector(Protocol):
    name: str

    def observe(self) -> ConnectorMeasurement:
        ...

    def apply(self, model: mujoco.MjModel, data: mujoco.MjData, command: object) -> ConnectorMeasurement:
        ...

