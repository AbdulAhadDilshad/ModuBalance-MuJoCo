"""Command-line entry point for one ModuBalance experiment."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from src.configuration import load_config
from src.simulation import run_experiment


PROJECT_ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the ModuBalance-MuJoCo feasibility experiment.")
    parser.add_argument("--headless", action="store_true", help="Run without the interactive MuJoCo viewer.")
    parser.add_argument("--controller", choices=("pd", "off"), default="pd", help="Connector controller mode.")
    parser.add_argument("--duration", type=float, help="Override simulation duration in seconds.")
    parser.add_argument("--payload-mass", type=float, help="Override equivalent payload mass in kilograms.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "default.yaml")
    parser.add_argument("--output-dir", type=Path, help="Override the mode-specific output directory.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = deepcopy(load_config(args.config))
    if args.duration is not None:
        if args.duration <= 0:
            raise SystemExit("--duration must be positive")
        config["simulation"]["duration"] = args.duration
    if args.payload_mass is not None:
        if args.payload_mass < 0:
            raise SystemExit("--payload-mass cannot be negative")
        config["payload"]["equivalent_mass"] = args.payload_mass
    output = args.output_dir or PROJECT_ROOT / "results" / args.controller
    run_experiment(
        config=config,
        model_path=PROJECT_ROOT / "models" / "modular_balance.xml",
        output_dir=output,
        controller_mode=args.controller,
        headless=args.headless,
    )


if __name__ == "__main__":
    main()

