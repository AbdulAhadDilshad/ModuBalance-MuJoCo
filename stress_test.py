"""Reproducible Monte Carlo stress testing for distributed magnetic control."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from pathlib import Path

import yaml

from src.configuration import load_config
from src.phase2_simulation import run_phase2_experiment
from src.stress import generate_stress_plots, one_factor_cases


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase-2 magnetic connector stress tests.")
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()
    if args.trials < 1:
        raise SystemExit("--trials must be at least one")
    base = load_config(ROOT / "config" / "phase2.yaml")
    with (ROOT / "config" / "stress.yaml").open("r", encoding="utf-8") as handle:
        stress = yaml.safe_load(handle)
    cases = one_factor_cases(stress)
    # A single deterministic grid supplies the requested 2-D operating envelope.
    for payload in stress["payloads"]:
        for velocity in stress["base_velocities"]:
            case = dict(stress["nominal"])
            case.update(payload=payload, base_velocity=velocity, sweep_parameter="payload_velocity_heatmap")
            cases.append(case)
    rows = []
    nominal_force = float(base["magnetic"]["maximum_force_per_magnet"])
    run_number = 0
    for case in cases:
        case_trials = args.trials if case["sweep_parameter"] != "payload_velocity_heatmap" else 1
        for trial in range(case_trials):
            config = deepcopy(base)
            config["simulation"]["duration"] = float(stress["duration"])
            config["payload"]["start_time"] = float(stress["payload_start_time"])
            config["base_motion"]["start_time"] = float(stress["base_start_time"])
            config["payload"]["equivalent_mass"] = float(case["payload"])
            config["base_motion"]["target_velocity"] = float(case["base_velocity"])
            config["disturbance"]["force_x"] = float(case["disturbance_force"])
            config["disturbance"]["start_time"] = 4.0
            config["magnetic"]["maximum_force_per_magnet"] = nominal_force * float(case["magnet_strength_scale"])
            config["friction"] = float(case["friction"])
            config["sensor_noise"]["orientation_std_deg"] = float(case["sensor_noise_deg"])
            config["sensor_noise"]["angular_velocity_std"] = float(case["sensor_noise_deg"]) * 0.02
            config["communication"]["delay_ms"] = float(case["communication_delay_ms"])
            config["communication"]["packet_drop_probability"] = float(case["packet_drop_probability"])
            seed = int(base["simulation"]["seed"]) + run_number * 100 + trial
            result = run_phase2_experiment(
                config,
                ROOT / "models" / "three_module_robot.xml",
                ROOT / "results" / "stress" / "runs",
                experiment="magnetic",
                controller_mode="distributed",
                communication_mode="nearest_neighbor",
                connector_type="magnetic",
                latch_enabled=False,
                headless=True,
                save_artifacts=False,
                print_results=False,
                seed=seed,
                verify_geometry=False,
            )
            report = result.report
            rows.append({
                "seed": seed,
                "sweep_parameter": case["sweep_parameter"],
                "payload": case["payload"],
                "velocity": case["base_velocity"],
                "base_velocity": case["base_velocity"],
                "disturbance": case["disturbance_force"],
                "disturbance_force": case["disturbance_force"],
                "magnet_strength": case["magnet_strength_scale"],
                "magnet_strength_scale": case["magnet_strength_scale"],
                "friction": case["friction"],
                "sensor_noise": case["sensor_noise_deg"],
                "sensor_noise_deg": case["sensor_noise_deg"],
                "communication_delay": case["communication_delay_ms"],
                "communication_delay_ms": case["communication_delay_ms"],
                "packet_loss": case["packet_drop_probability"],
                "packet_drop_probability": case["packet_drop_probability"],
                "peak_pitch": report["peak_pitch_deg"],
                "peak_roll": report["peak_roll_deg"],
                "settling_time": report["settling_time_payload_s"],
                "rms_error": report["rms_pitch_deg"],
                "rms_roll": report["rms_roll_deg"],
                "max_connector_force": report["max_connector_force_n"],
                "max_connector_torque": report["max_connector_torque_nm"],
                "max_magnet_command": report["max_magnet_command"],
                "saturation_fraction": report["saturation_fraction"],
                "latch_cycles": report["latch_transitions"],
                "control_effort": report["control_effort"],
                "detachment_failure": report["detachment_failure"],
                "recovery_success": report["recovery_success"],
            })
        run_number += 1
        print(f"Completed stress case {run_number}/{len(cases)}: {case['sweep_parameter']}")
    output = ROOT / "results" / "stress"
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "stress_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    plots = generate_stress_plots(rows, output)
    successes = sum(bool(row["recovery_success"]) for row in rows)
    print(f"Stress results: {csv_path} ({successes}/{len(rows)} successful runs)")
    for path in plots:
        print(f"  {path}")


if __name__ == "__main__":
    main()
