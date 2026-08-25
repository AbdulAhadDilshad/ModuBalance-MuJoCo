"""Run or load PD and passive baselines, then plot them together."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.configuration import load_config
from src.logger import load_csv
from src.plotting import generate_comparison_plot
from src.simulation import run_experiment


ROOT = Path(__file__).resolve().parent


def _records_from_columns(columns: dict[str, list[float]]) -> list[dict[str, Any]]:
    keys = list(columns)
    return [{key: columns[key][index] for key in keys} for index in range(len(columns[keys[0]]))]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare controller OFF with intelligent PD control.")
    parser.add_argument("--rerun", action="store_true", help="Force both headless experiments to run again.")
    args = parser.parse_args()
    config = load_config(ROOT / "config" / "default.yaml")
    records: dict[str, list[dict[str, Any]]] = {}
    for mode in ("off", "pd"):
        csv_path = ROOT / "results" / mode / "experiment.csv"
        if args.rerun or not csv_path.is_file():
            result = run_experiment(
                config,
                ROOT / "models" / "modular_balance.xml",
                ROOT / "results" / mode,
                controller_mode=mode,
                headless=True,
                print_results=False,
            )
            records[mode] = result.records
        else:
            records[mode] = _records_from_columns(load_csv(csv_path))
    destination = generate_comparison_plot(
        records["off"],
        records["pd"],
        config,
        ROOT / "results" / "comparison" / "pitch_pd_vs_off.png",
    )
    off_peak = max(abs(row["pitch_deg"]) for row in records["off"])
    pd_peak = max(abs(row["pitch_deg"]) for row in records["pd"])
    print(f"Controller OFF peak |pitch|: {off_peak:.3f} deg")
    print(f"Intelligent PD peak |pitch|: {pd_peak:.3f} deg")
    print(f"Comparison plot: {destination}")


if __name__ == "__main__":
    main()
