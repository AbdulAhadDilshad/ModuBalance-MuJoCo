"""Active module/connector topology, separate from distributed policy logic."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ActiveTopology:
    modules: dict[str, bool] = field(default_factory=lambda: {"A": True, "B": True, "C": True})
    connectors: dict[str, bool] = field(default_factory=lambda: {"AB": True, "BC": True})
    c_offset_x: float = 0.0

    def communication_links(self) -> set[frozenset[str]]:
        links: set[frozenset[str]] = set()
        if self.connectors["AB"] and self.modules["A"] and self.modules["B"]:
            links.add(frozenset(("A", "B")))
        if self.connectors["BC"] and self.modules["B"] and self.modules["C"]:
            links.add(frozenset(("B", "C")))
        return links

    def remove_c(self) -> None:
        self.modules["C"] = False
        self.connectors["BC"] = False

    def add_c(self) -> None:
        self.modules["C"] = True
        self.connectors["BC"] = True

    def reposition_c(self, offset_x: float) -> None:
        self.c_offset_x = float(offset_x)

