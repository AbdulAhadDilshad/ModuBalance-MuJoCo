"""Headless equivalent-payload operating-envelope sweep."""

from __future__ import annotations

import argparse
import csv
import os
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.configuration import load_config
from src.simulation import run_experiment


DEFAULT_MASSES = (0.10, 0.25, 0.50, 0.75, 1.00, 1.50)


def _plot(rows: list[dict[str, object]], key: str, ylabel: str, filename: str) -> Path:
    masses = [float(row["payload_mass_kg"]) for row in rows]
    values = [float("nan") if row[key] is None else float(row[key]) for row in rows]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(masses, values, marker="o", linewidth=1.7)
    ax.set(xlabel="Equivalent payload mass (kg)", ylabel=ylabel)
    ax.grid(alpha=0.3)
    destination = ROOT / "results" / filename
    fig.tight_layout()
    fig.savefig(destination, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep equivalent off-centre payload masses.")
    parser.add_argument("--masses", type=float, nargs="+", default=DEFAULT_MASSES)
    args = parser.parse_args()
    if any(mass < 0 for mass in args.masses):
        raise SystemExit("Payload masses cannot be negative")
    base_config = load_config(ROOT / "config" / "default.yaml")
    rows: list[dict[str, object]] = []
    for mass in args.masses:
        config = deepcopy(base_config)
        config["payload"]["equivalent_mass"] = mass
        result = run_experiment(
            config,
            ROOT / "models" / "modular_balance.xml",
            ROOT / "results" / "sweep_runs" / f"payload_{mass:.2f}_kg",
            controller_mode="pd",
            headless=True,
            print_results=False,
        )
        report = result.report
        row = {
            "payload_mass_kg": mass,
            "peak_pitch_deg": report["maximum_absolute_pitch_after_payload_deg"],
            "settling_time_s": report["settling_time_payload_s"],
            "rms_pitch_deg": report["rms_pitch_error_deg"],
            "maximum_torque_nm": report["maximum_control_torque_nm"],
            "saturation": report["actuator_saturated"],
            "recovery_pass": report["payload_recovery_pass"],
        }
        rows.append(row)
        print(
            f"payload={mass:.2f} kg  peak={float(row['peak_pitch_deg']):.3f} deg  "
            f"settling={row['settling_time_s']}  pass={row['recovery_pass']}"
        )

    output_csv = ROOT / "results" / "payload_sweep.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plots = (
        _plot(rows, "peak_pitch_deg", "Peak |pitch| after payload (deg)", "payload_vs_peak_pitch.png"),
        _plot(rows, "settling_time_s", "Payload settling time (s)", "payload_vs_settling_time.png"),
        _plot(rows, "rms_pitch_deg", "Whole-run RMS pitch error (deg)", "payload_vs_rms_error.png"),
    )
    print(f"Sweep data: {output_csv}")
    print("Sweep plots: " + ", ".join(str(path) for path in plots))


if __name__ == "__main__":
    main()
