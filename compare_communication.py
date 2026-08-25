"""Compare centralized, nearest-neighbour, local-only, and delayed control."""

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
        "Centralized": ("centralized", "nearest_neighbor"),
        "Distributed nearest": ("distributed", "nearest_neighbor"),
        "Distributed local only": ("distributed", "none"),
        "Distributed 50 ms delay": ("distributed", "delayed"),
    }
    records = {}
    rows = []
    output = ROOT / "results" / "comparisons" / "communication"
    for label, (controller, communication) in cases.items():
        result = run_phase2_experiment(
            deepcopy(config),
            ROOT / "models" / "three_module_robot.xml",
            output / label.lower().replace(" ", "_"),
            experiment="three_module",
            controller_mode=controller,
            communication_mode=communication,
            connector_type="ideal",
            headless=True,
            print_results=False,
        )
        records[label] = result.records
        rows.append({"strategy": label, **result.report})
    comparison_plot(records, "module_C_pitch", output / "communication_pitch.png", "Module C pitch (deg)", "Communication strategy comparison")
    comparison_plot(records, "module_C_roll", output / "communication_roll.png", "Module C roll (deg)", "Communication strategy roll comparison")
    with (output / "communication_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            f"{row['strategy']}: RMS pitch={row['rms_pitch_deg']:.3f} deg, "
            f"settling={row['settling_time_payload_s']}, success={row['recovery_success']}"
        )
    print(f"Communication comparison results: {output}")


if __name__ == "__main__":
    main()

