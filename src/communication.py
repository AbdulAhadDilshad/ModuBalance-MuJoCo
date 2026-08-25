"""Explicit, restricted communication for distributed module agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class AgentMessage:
    source: str
    target: str
    delivery_time: float
    payload: dict[str, float]


class CommunicationChannel:
    """Nearest-neighbour channel with delay, dropout, and message noise.

    The channel owns no simulator reference. It can only transport explicit compact
    dictionaries between links supplied by the active topology.
    """

    def __init__(
        self,
        links: set[frozenset[str]],
        delay_ms: float = 0.0,
        packet_drop_probability: float = 0.0,
        noise_std: float = 0.0,
        seed: int = 0,
        enabled: bool = True,
    ) -> None:
        if delay_ms < 0 or not 0.0 <= packet_drop_probability <= 1.0 or noise_std < 0:
            raise ValueError("Invalid communication delay, dropout probability, or noise")
        self.links = set(links)
        self.delay_s = delay_ms / 1000.0
        self.packet_drop_probability = packet_drop_probability
        self.noise_std = noise_std
        self.enabled = enabled
        self._rng = np.random.default_rng(seed)
        self._queue: list[AgentMessage] = []

    def update_links(self, links: set[frozenset[str]]) -> None:
        self.links = set(links)
        self._queue = [message for message in self._queue if frozenset((message.source, message.target)) in self.links]

    def neighbours(self, module_id: str) -> tuple[str, ...]:
        neighbours: list[str] = []
        for link in self.links:
            if module_id in link:
                neighbours.extend(node for node in link if node != module_id)
        return tuple(sorted(neighbours))

    def publish(self, source: str, payload: Mapping[str, float], time: float) -> None:
        if not self.enabled:
            return
        for target in self.neighbours(source):
            if self._rng.random() < self.packet_drop_probability:
                continue
            noisy = {
                key: float(value + self._rng.normal(0.0, self.noise_std))
                for key, value in payload.items()
            }
            self._queue.append(AgentMessage(source, target, time + self.delay_s, noisy))

    def receive(self, target: str, time: float) -> tuple[AgentMessage, ...]:
        delivered = [message for message in self._queue if message.target == target and message.delivery_time <= time + 1e-12]
        self._queue = [message for message in self._queue if message not in delivered]
        latest_by_source: dict[str, AgentMessage] = {}
        for message in delivered:
            latest_by_source[message.source] = message
        return tuple(latest_by_source[source] for source in sorted(latest_by_source))

