"""Quantitative evaluation for three-module, magnetic, and topology experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _multi_axis_settling_time(
    time: np.ndarray,
    pitch: np.ndarray,
    roll: np.ndarray,
    start_time: float,
    end_time: float,
    pitch_tolerance: float,
    roll_tolerance: float,
    dwell: float,
) -> float | None:
    for index in np.flatnonzero((time >= start_time) & (time + dwell <= end_time + 1e-9)):
        end_index = int(np.searchsorted(time, time[index] + dwell, side="left"))
        if end_index >= len(time):
            break
        if (
            np.all(np.max(np.abs(pitch[index : end_index + 1]), axis=1) <= pitch_tolerance)
            and np.all(np.max(np.abs(roll[index : end_index + 1]), axis=1) <= roll_tolerance)
        ):
            return float(time[index] - start_time)
    return None


def evaluate_phase2(records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    if not records:
        raise ValueError("Cannot evaluate an empty Phase-2 run")
    time = np.asarray([row["time"] for row in records], dtype=float)
    pitch = np.column_stack(
        [[row[f"module_{module}_pitch"] for row in records] for module in "ABC"]
    ).astype(float)
    roll = np.column_stack(
        [[row[f"module_{module}_roll"] for row in records] for module in "ABC"]
    ).astype(float)
    connector_force = np.column_stack(
        [[row[f"connector_{connector}_force"] for row in records] for connector in ("AB", "BC")]
    ).astype(float)
    connector_torque = np.column_stack(
        [[row[f"connector_{connector}_torque"] for row in records] for connector in ("AB", "BC")]
    ).astype(float)
    saturated = np.asarray([row["saturated_element_count"] for row in records], dtype=float)
    effort = np.asarray([row["control_effort"] for row in records], dtype=float)
    evaluation = config["evaluation"]
    payload_time = float(config["payload"]["start_time"])
    base_time = float(config["base_motion"]["start_time"])
    duration = float(time[-1])
    payload_settling = _multi_axis_settling_time(
        time,
        pitch,
        roll,
        payload_time,
        min(base_time, duration),
        float(evaluation["balance_tolerance_deg"]),
        float(evaluation["roll_tolerance_deg"]),
        float(evaluation["settling_dwell_time"]),
    ) if duration >= payload_time + float(evaluation["settling_dwell_time"]) else None
    base_settling = _multi_axis_settling_time(
        time,
        pitch,
        roll,
        base_time,
        duration,
        float(evaluation["balance_tolerance_deg"]),
        float(evaluation["roll_tolerance_deg"]),
        float(evaluation["settling_dwell_time"]),
    ) if duration >= base_time + float(evaluation["settling_dwell_time"]) else None
    max_pitch = float(np.max(np.abs(pitch)))
    max_roll = float(np.max(np.abs(roll)))
    report = {
        "peak_pitch_deg": max_pitch,
        "peak_roll_deg": max_roll,
        "settling_time_payload_s": payload_settling,
        "settling_time_base_s": base_settling,
        "rms_pitch_deg": float(np.sqrt(np.mean(np.square(pitch)))),
        "rms_roll_deg": float(np.sqrt(np.mean(np.square(roll)))),
        "max_connector_force_n": float(np.max(connector_force)),
        "max_connector_torque_nm": float(np.max(connector_torque)),
        "max_magnet_command": float(
            max(row[f"{connector}_m{magnet}_command"] for row in records for connector in ("AB", "BC") for magnet in range(1, 5))
        ),
        "saturation_fraction": float(np.sum(saturated) / max(1.0, len(records) * 8.0)),
        "latch_transitions": int(records[-1]["latch_transition_count"]),
        "control_effort": float(np.trapezoid(effort, time)),
        "detachment_failure": bool(any(row["detachment_failure"] for row in records)),
    }
    report["recovery_success"] = bool(
        max_pitch <= float(evaluation["failure_max_pitch_deg"])
        and max_roll <= float(evaluation["failure_max_roll_deg"])
        and not report["detachment_failure"]
        and (payload_settling is not None or duration < payload_time + float(evaluation["settling_dwell_time"]))
        and (base_settling is not None or duration < base_time + float(evaluation["settling_dwell_time"]))
    )
    return report


def save_phase2_report(report: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return destination


def print_phase2_summary(report: dict[str, Any]) -> None:
    print("===============================")
    print("ModuBalance Phase-2 Results")
    print("===============================")
    print(f"Peak pitch / roll: {report['peak_pitch_deg']:.3f} / {report['peak_roll_deg']:.3f} deg")
    print(f"RMS pitch / roll: {report['rms_pitch_deg']:.3f} / {report['rms_roll_deg']:.3f} deg")
    print(f"Payload settling: {report['settling_time_payload_s']}")
    print(f"Base settling: {report['settling_time_base_s']}")
    print(f"Maximum connector force: {report['max_connector_force_n']:.3f} N")
    print(f"Maximum connector torque: {report['max_connector_torque_nm']:.3f} N m")
    print(f"Magnet saturation fraction: {report['saturation_fraction']:.4f}")
    print(f"Latch transitions: {report['latch_transitions']}")
    print(f"Recovery: {'PASS' if report['recovery_success'] else 'FAIL'}")

