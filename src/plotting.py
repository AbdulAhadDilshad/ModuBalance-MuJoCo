"""Automatic experiment and baseline-comparison plots."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _arrays(records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    return {key: np.asarray([row[key] for row in records], dtype=float) for key in records[0]}


def _events(axis: plt.Axes, payload_time: float, base_time: float) -> None:
    axis.axvline(payload_time, color="#d95f02", linestyle="--", linewidth=1.3, label="Payload disturbance")
    axis.axvline(base_time, color="#7570b3", linestyle="--", linewidth=1.3, label="Base motion begins")


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def generate_experiment_plots(
    records: list[dict[str, Any]], config: dict[str, Any], output_dir: str | Path
) -> list[Path]:
    if not records:
        raise ValueError("Cannot plot an empty experiment")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data = _arrays(records)
    time = data["time"]
    payload_time = float(config["payload"]["start_time"])
    base_time = float(config["base_motion"]["start_time"])
    tolerance = float(config["evaluation"]["balance_tolerance_deg"])
    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(time, data["pitch_deg"], color="#1b6ca8", linewidth=1.5, label="Cube B pitch")
    ax.axhline(tolerance, color="gray", linestyle="--", label=f"+/-{tolerance:g} deg tolerance")
    ax.axhline(-tolerance, color="gray", linestyle="--")
    _events(ax, payload_time, base_time)
    ax.set(xlabel="Time (s)", ylabel="Pitch (deg)", title="Cube B pitch response")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    paths.append(output / "pitch_response.png")
    _save(fig, paths[-1])

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(time, data["controller_torque_command"], label="Commanded", linewidth=1.3)
    ax.plot(time, data["actuator_torque_applied"], label="Measured/applied", linewidth=1.0, alpha=0.75)
    _events(ax, payload_time, base_time)
    ax.set(xlabel="Time (s)", ylabel="Torque (N m)", title="Ideal connector control torque")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    paths.append(output / "controller_torque.png")
    _save(fig, paths[-1])

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(time, data["connector_force"], color="#2a9d8f", label="Connector force magnitude")
    ax2 = ax.twinx()
    ax2.plot(time, data["connector_torque_y"], color="#e76f51", alpha=0.8, label="Connector torque Y")
    _events(ax, payload_time, base_time)
    ax.set(xlabel="Time (s)", ylabel="Force (N)", title="Local connector measurements")
    ax2.set_ylabel("Torque Y (N m)")
    ax.grid(alpha=0.25)
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], loc="best")
    paths.append(output / "connector_measurements.png")
    _save(fig, paths[-1])

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(time, data["base_position"], label="Position (m)")
    ax.plot(time, data["base_velocity"], label="Velocity (m/s)")
    ax.plot(time, data["base_target_velocity"], linestyle="--", label="Target velocity (m/s)")
    _events(ax, payload_time, base_time)
    ax.set(xlabel="Time (s)", title="Movable foundation response")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    paths.append(output / "base_motion.png")
    _save(fig, paths[-1])

    fig, ax = plt.subplots(figsize=(10, 4.5))
    angle_line = ax.plot(time, data["joint_angle_deg"], color="#264653", label="Joint angle")
    ax2 = ax.twinx()
    velocity_line = ax2.plot(
        time, data["joint_velocity"], color="#e9c46a", alpha=0.85, label="Joint velocity"
    )
    _events(ax, payload_time, base_time)
    ax.set(xlabel="Time (s)", ylabel="Joint angle (deg)", title="Balance hinge response")
    ax2.set_ylabel("Joint velocity (rad/s)")
    ax.grid(alpha=0.25)
    lines = angle_line + velocity_line + ax.get_lines()[1:]
    ax.legend(lines, [line.get_label() for line in lines], loc="best")
    paths.append(output / "joint_response.png")
    _save(fig, paths[-1])
    return paths


def generate_comparison_plot(
    off_records: list[dict[str, Any]],
    pd_records: list[dict[str, Any]],
    config: dict[str, Any],
    output_path: str | Path,
) -> Path:
    off = _arrays(off_records)
    pd = _arrays(pd_records)
    tolerance = float(config["evaluation"]["balance_tolerance_deg"])
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.plot(off["time"], off["pitch_deg"], color="#d1495b", linewidth=1.25, label="Controller OFF")
    ax.plot(pd["time"], pd["pitch_deg"], color="#00798c", linewidth=1.5, label="Intelligent PD ON")
    ax.axhline(tolerance, color="gray", linestyle="--", label=f"+/-{tolerance:g} deg tolerance")
    ax.axhline(-tolerance, color="gray", linestyle="--")
    _events(ax, float(config["payload"]["start_time"]), float(config["base_motion"]["start_time"]))
    ax.set(xlabel="Time (s)", ylabel="Pitch (deg)", title="Scientific baseline: local intelligent control contribution")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _save(fig, destination)
    return destination
