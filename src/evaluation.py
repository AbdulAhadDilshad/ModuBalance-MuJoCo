"""Quantitative, measurement-derived feasibility evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def settling_time(
    time: np.ndarray,
    pitch_deg: np.ndarray,
    event_time: float,
    phase_end: float,
    tolerance_deg: float,
    dwell_time: float,
) -> float | None:
    """First post-event time that begins a continuous in-band dwell interval."""
    candidates = np.flatnonzero((time >= event_time) & (time + dwell_time <= phase_end + 1e-9))
    absolute_pitch = np.abs(pitch_deg)
    for start_index in candidates:
        end_time = time[start_index] + dwell_time
        end_index = int(np.searchsorted(time, end_time, side="left"))
        if end_index < len(time) and np.all(absolute_pitch[start_index : end_index + 1] <= tolerance_deg):
            return float(time[start_index] - event_time)
    return None


def _phase_mask(time: np.ndarray, start: float, end: float, include_end: bool = False) -> np.ndarray:
    return (time >= start) & (time <= end if include_end else time < end)


def evaluate_experiment(
    records: list[dict[str, Any]],
    config: dict[str, Any],
    controller_mode: str,
) -> dict[str, Any]:
    if not records:
        raise ValueError("Cannot evaluate an empty experiment")
    time = np.asarray([row["time"] for row in records], dtype=float)
    pitch = np.asarray([row["pitch_deg"] for row in records], dtype=float)
    torque = np.asarray([row["commanded_joint_torque"] for row in records], dtype=float)
    saturated = np.asarray([row["actuator_saturated"] for row in records], dtype=bool)

    payload_time = float(config["payload"]["start_time"])
    base_time = float(config["base_motion"]["start_time"])
    duration = float(time[-1])
    tolerance = float(config["evaluation"]["balance_tolerance_deg"])
    dwell = float(config["evaluation"]["settling_dwell_time"])

    initial_mask = _phase_mask(time, 0.0, min(payload_time, duration + 1e-9))
    payload_mask = _phase_mask(time, payload_time, min(base_time, duration + 1e-9))
    base_mask = _phase_mask(time, base_time, duration, include_end=True)

    def peak(mask: np.ndarray) -> float:
        return float(np.max(np.abs(pitch[mask]))) if np.any(mask) else float("nan")

    def rms(mask: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(pitch[mask])))) if np.any(mask) else float("nan")

    initial_settling = settling_time(time, pitch, 0.0, min(payload_time, duration), tolerance, dwell)
    payload_settling = (
        settling_time(time, pitch, payload_time, min(base_time, duration), tolerance, dwell)
        if duration >= payload_time + dwell
        else None
    )
    base_settling = (
        settling_time(time, pitch, base_time, duration, tolerance, dwell)
        if duration >= base_time + dwell
        else None
    )

    report: dict[str, Any] = {
        "controller": controller_mode,
        "maximum_absolute_pitch_before_payload_deg": peak(initial_mask),
        "maximum_absolute_pitch_after_payload_deg": peak(payload_mask),
        "maximum_absolute_pitch_during_base_motion_deg": peak(base_mask),
        "final_pitch_error_deg": float(abs(pitch[-1])),
        "settling_time_initial_s": initial_settling,
        "settling_time_payload_s": payload_settling,
        "settling_time_base_s": base_settling,
        "rms_pitch_error_deg": rms(np.ones_like(time, dtype=bool)),
        "rms_pitch_during_base_motion_deg": rms(base_mask),
        "maximum_control_torque_nm": float(np.max(np.abs(torque))),
        "actuator_saturated": bool(np.any(saturated)),
        "initial_recovery_pass": initial_settling is not None,
        "payload_recovery_pass": payload_settling is not None,
        "base_recovery_pass": base_settling is not None,
    }
    report["overall_feasibility_pass"] = bool(
        report["initial_recovery_pass"]
        and report["payload_recovery_pass"]
        and report["base_recovery_pass"]
    )
    return report


def save_report(report: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return destination


def _format_time(value: float | None) -> str:
    return "not settled" if value is None else f"{value:.3f} s"


def print_summary(report: dict[str, Any]) -> None:
    status = lambda passed: "PASS" if passed else "FAIL"
    print("==============================")
    print("ModuBalance Experiment Results")
    print("==============================")
    print(f"Controller: {str(report['controller']).upper()}")
    print("\nInitial disturbance:")
    print(f"Peak pitch: {report['maximum_absolute_pitch_before_payload_deg']:.3f} deg")
    print(f"Settling time: {_format_time(report['settling_time_initial_s'])}")
    print(f"Status: {status(report['initial_recovery_pass'])}")
    print("\nPayload disturbance:")
    print(f"Peak pitch: {report['maximum_absolute_pitch_after_payload_deg']:.3f} deg")
    print(f"Settling time: {_format_time(report['settling_time_payload_s'])}")
    print(f"Status: {status(report['payload_recovery_pass'])}")
    print("\nMoving-base disturbance:")
    print(f"Peak pitch: {report['maximum_absolute_pitch_during_base_motion_deg']:.3f} deg")
    print(f"RMS pitch: {report['rms_pitch_during_base_motion_deg']:.3f} deg")
    print(f"Status: {status(report['base_recovery_pass'])}")
    print(f"\nFinal pitch error: {report['final_pitch_error_deg']:.3f} deg")
    print(f"Overall RMS pitch error: {report['rms_pitch_error_deg']:.3f} deg")
    print(f"Maximum control torque: {report['maximum_control_torque_nm']:.3f} N m")
    print(f"Actuator saturated: {'YES' if report['actuator_saturated'] else 'NO'}")
    print(f"\nOverall feasibility result: {status(report['overall_feasibility_pass'])}")

