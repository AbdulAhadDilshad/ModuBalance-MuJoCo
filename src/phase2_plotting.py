"""Headless-safe Phase-2 diagnostic and comparison plotting."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def _events(ax: plt.Axes, config: dict[str, Any]) -> None:
    ax.axvline(float(config["payload"]["start_time"]), color="#d95f02", linestyle="--", label="Payload")
    ax.axvline(float(config["base_motion"]["start_time"]), color="#7570b3", linestyle="--", label="Base motion")


def generate_phase2_plots(
    records: list[dict[str, Any]], config: dict[str, Any], output_dir: str | Path
) -> list[Path]:
    output = Path(output_dir)
    time = np.asarray([row["time"] for row in records])
    paths: list[Path] = []

    for axis_name in ("pitch", "roll"):
        fig, ax = plt.subplots(figsize=(10, 4.7))
        for module, color in zip("ABC", ("#1b6ca8", "#e76f51", "#2a9d8f")):
            ax.plot(time, [row[f"module_{module}_{axis_name}"] for row in records], label=f"Module {module}", color=color)
        tolerance = float(config["evaluation"]["balance_tolerance_deg" if axis_name == "pitch" else "roll_tolerance_deg"])
        ax.axhline(tolerance, color="gray", linestyle=":", label="Tolerance")
        ax.axhline(-tolerance, color="gray", linestyle=":")
        _events(ax, config)
        ax.set(xlabel="Time (s)", ylabel=f"{axis_name.title()} (deg)", title=f"Three-module {axis_name} response")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
        paths.append(_save(fig, output / f"module_{axis_name}.png"))

    fig, ax = plt.subplots(figsize=(10, 4.7))
    for connector in ("AB", "BC"):
        ax.plot(time, [row[f"connector_{connector}_force"] for row in records], label=f"{connector} force")
    _events(ax, config)
    ax.set(xlabel="Time (s)", ylabel="Force (N)", title="Connector force")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    paths.append(_save(fig, output / "connector_force.png"))

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for axis, connector in zip(axes, ("AB", "BC")):
        for magnet in range(1, 5):
            axis.plot(time, [row[f"{connector}_m{magnet}_command"] for row in records], label=f"M{magnet}")
        axis.set(ylabel="Command", title=f"Connector {connector} magnets")
        axis.grid(alpha=0.25)
        axis.legend(ncol=4, loc="best")
    axes[-1].set_xlabel("Time (s)")
    paths.append(_save(fig, output / "magnet_commands.png"))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.step(time, [row["connector_AB_latch_state"] for row in records], where="post", label="AB latch")
    ax.step(time, [row["connector_BC_latch_state"] for row in records], where="post", label="BC latch")
    ax.step(time, [row["topology_module_C_active"] for row in records], where="post", label="Module C active")
    ax.set(xlabel="Time (s)", ylabel="Binary state", title="Latch and topology state", ylim=(-0.1, 1.1))
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    paths.append(_save(fig, output / "latch_topology.png"))
    return paths


def comparison_plot(
    series: dict[str, list[dict[str, Any]]],
    field: str,
    output_path: str | Path,
    ylabel: str,
    title: str,
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, records in series.items():
        ax.plot([row["time"] for row in records], [row[field] for row in records], label=label)
    ax.set(xlabel="Time (s)", ylabel=ylabel, title=title)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    return _save(fig, Path(output_path))

