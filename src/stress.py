"""Stress-test design, aggregation, and operating-envelope plots."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def one_factor_cases(stress: dict[str, Any]) -> list[dict[str, Any]]:
    nominal = dict(stress["nominal"])
    mapping = {
        "payload": "payloads",
        "base_velocity": "base_velocities",
        "disturbance_force": "disturbance_forces",
        "magnet_strength_scale": "magnet_strength_scales",
        "friction": "frictions",
        "sensor_noise_deg": "sensor_noise_degrees",
        "communication_delay_ms": "communication_delays_ms",
        "packet_drop_probability": "packet_drop_probabilities",
    }
    cases = [{"sweep_parameter": "nominal", **nominal}]
    for parameter, config_key in mapping.items():
        for value in stress[config_key]:
            case = dict(nominal)
            case[parameter] = value
            case["sweep_parameter"] = parameter
            cases.append(case)
    return cases


def _group(rows: list[dict[str, Any]], parameter: str) -> tuple[list[float], list[float], list[float]]:
    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["sweep_parameter"] == parameter:
            grouped[float(row[parameter])].append(row)
    x = sorted(grouped)
    success = [100.0 * np.mean([item["recovery_success"] for item in grouped[value]]) for value in x]
    peak = [float(np.mean([item["peak_pitch"] for item in grouped[value]])) for value in x]
    return x, success, peak


def generate_stress_plots(rows: list[dict[str, Any]], output_dir: str | Path) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    plots: list[Path] = []
    specifications = [
        ("payload", "Payload (kg)", "payload_vs_success_rate.png"),
        ("base_velocity", "Base velocity (m/s)", "velocity_vs_success_rate.png"),
        ("magnet_strength_scale", "Magnet strength scale", "magnet_strength_vs_success_rate.png"),
        ("friction", "Friction coefficient", "friction_vs_success_rate.png"),
        ("sensor_noise_deg", "Orientation noise std (deg)", "sensor_noise_vs_success_rate.png"),
    ]
    for parameter, xlabel, filename in specifications:
        x, success, _ = _group(rows, parameter)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(x, success, marker="o")
        ax.set(xlabel=xlabel, ylabel="Recovery success rate (%)", ylim=(-5, 105))
        ax.grid(alpha=0.3)
        path = output / filename
        fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig); plots.append(path)

    x, _, peak = _group(rows, "payload")
    fig, ax = plt.subplots(figsize=(7, 4.5)); ax.plot(x, peak, marker="o")
    ax.set(xlabel="Payload (kg)", ylabel="Mean peak pitch (deg)"); ax.grid(alpha=0.3)
    path = output / "payload_vs_peak_pitch.png"; fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig); plots.append(path)

    grouped: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        if row["sweep_parameter"] == "communication_delay_ms" and row["settling_time"] is not None:
            grouped[float(row["communication_delay_ms"])].append(float(row["settling_time"]))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    delays = sorted(grouped); ax.plot(delays, [np.mean(grouped[value]) for value in delays], marker="o")
    ax.set(xlabel="Communication delay (ms)", ylabel="Mean payload settling time (s)"); ax.grid(alpha=0.3)
    path = output / "communication_delay_vs_settling_time.png"; fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig); plots.append(path)

    heatmap_rows = [row for row in rows if row["sweep_parameter"] == "payload_velocity_heatmap"]
    payloads = sorted({float(row["payload"]) for row in heatmap_rows})
    velocities = sorted({float(row["base_velocity"]) for row in heatmap_rows})
    if payloads and velocities:
        matrix = np.zeros((len(payloads), len(velocities)))
        for i, payload in enumerate(payloads):
            for j, velocity in enumerate(velocities):
                values = [row["recovery_success"] for row in heatmap_rows if float(row["payload"]) == payload and float(row["base_velocity"]) == velocity]
                matrix[i, j] = np.mean(values) if values else np.nan
        fig, ax = plt.subplots(figsize=(8, 5)); image = ax.imshow(matrix, origin="lower", aspect="auto", vmin=0, vmax=1, cmap="RdYlGn")
        ax.set_xticks(range(len(velocities)), velocities); ax.set_yticks(range(len(payloads)), payloads)
        ax.set(xlabel="Base velocity (m/s)", ylabel="Payload (kg)", title="Payload/base-velocity recovery envelope")
        fig.colorbar(image, ax=ax, label="Recovery fraction")
        path = output / "payload_velocity_success_heatmap.png"; fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig); plots.append(path)
    return plots

