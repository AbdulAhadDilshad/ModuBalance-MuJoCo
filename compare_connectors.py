"""Compare ideal torque, four-magnet, and four-magnet+latch connectors."""

from __future__ import annotations

import csv
from copy import deepcopy
from pathlib import Path

from src.configuration import load_config
from src.phase2_plotting import comparison_plot
from src.phase2_simulation import run_phase2_experiment


ROOT = Path(__file__).resolve().parent


def main() -> None:
    config = load_config(ROOT / "config" / "phase2.yaml")
    cases = {
        "Ideal torque": ("ideal", False),
        "Four magnets": ("magnetic", False),
        "Four magnets + latch": ("magnetic", True),
    }
    records = {}
    rows = []
    for label, (connector, latch) in cases.items():
        result = run_phase2_experiment(
            deepcopy(config),
            ROOT / "models" / "three_module_robot.xml",
            ROOT / "results" / "comparisons" / label.lower().replace(" ", "_").replace("+", "plus"),
            experiment="magnetic" if connector == "magnetic" else "three_module",
            controller_mode="distributed",
            communication_mode="nearest_neighbor",
            connector_type=connector,
            latch_enabled=latch,
            headless=True,
            print_results=False,
        )
        records[label] = result.records
        rows.append({"connector": label, **result.report})
    output = ROOT / "results" / "comparisons"
    comparison_plot(records, "module_C_pitch", output / "ideal_vs_magnetic_pitch.png", "Module C pitch (deg)", "Ideal vs magnetic connector")
    comparison_plot(records, "module_C_roll", output / "ideal_vs_magnetic_roll.png", "Module C roll (deg)", "Ideal vs magnetic connector roll")
    with (output / "connector_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            f"{row['connector']}: peak pitch={row['peak_pitch_deg']:.3f} deg, "
            f"RMS={row['rms_pitch_deg']:.3f} deg, effort={row['control_effort']:.3f}, "
            f"success={row['recovery_success']}"
        )
    print(f"Connector comparison results: {output}")


if __name__ == "__main__":
    main()

