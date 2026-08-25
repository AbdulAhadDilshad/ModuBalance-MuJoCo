"""Replaceable ideal, magnetic, and latch connector models."""

from .base import ConnectorMeasurement
from .ideal import IdealTorqueConnector
from .latch import LATCH_LOCKED, LATCH_OPEN, MagneticLatchController
from .magnetic import MagneticConnector, allocate_four_magnet_commands

__all__ = [
    "ConnectorMeasurement",
    "IdealTorqueConnector",
    "MagneticConnector",
    "MagneticLatchController",
    "LATCH_OPEN",
    "LATCH_LOCKED",
    "allocate_four_magnet_commands",
]

