"""Local intelligent-module abstraction."""

from __future__ import annotations

from .controller import ConnectorPolicy, ControlResult, LocalObservation


class IntelligentModuleAgent:
    """Cube B's local agent, deliberately isolated from MuJoCo global state."""

    def __init__(self, connector: ConnectorPolicy) -> None:
        self.connector = connector

    def act(self, observation: LocalObservation) -> ControlResult:
        return self.connector.compute_action(observation)

