"""Dependency-light experiment CSV logging."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


class ExperimentLogger:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def append(self, **values: Any) -> None:
        self.records.append(values)

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not self.records:
            raise ValueError("Cannot save an empty experiment log")
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.records[0]))
            writer.writeheader()
            writer.writerows(self.records)
        return destination


def load_csv(path: str | Path) -> dict[str, list[float]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Experiment data not found: {source}")
    columns: dict[str, list[float]] = {}
    with source.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for key, value in row.items():
                columns.setdefault(key, []).append(float(value))
    if not columns:
        raise ValueError(f"Experiment data is empty: {source}")
    return columns

