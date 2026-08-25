"""Command-line entry point for one ModuBalance experiment."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from src.configuration import load_config
from src.phase2_simulation import run_phase2_experiment
from src.simulation import run_experiment


PROJECT_ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the ModuBalance-MuJoCo feasibility experiment.")
    parser.add_argument("--headless", action="store_true", help="Run without the interactive MuJoCo viewer.")
    parser.add_argument(
        "--experiment",
        choices=(
            "phase1", "three_module", "magnetic", "remove_module", "add_module",
            "reposition_module", "two_leg", "wheeled",
        ),
        default="phase1",
        help="Research experiment to run.",
    )
    parser.add_argument(
        "--controller",
        choices=("pd", "off", "distributed", "centralized", "passive"),
        help="Phase-1 or Phase-2 controller mode.",
    )
    parser.add_argument("--mode", choices=("distributed", "centralized", "passive"), help="Alias for Phase-2 --controller.")
    parser.add_argument("--communication", choices=("nearest_neighbor", "none", "delayed"), default="nearest_neighbor")
    parser.add_argument("--connector", choices=("ideal", "magnetic"), help="Override the experiment's connector model.")
    parser.add_argument("--latch", choices=("on", "off"), default="off", help="Enable magnetic latch policy.")
    parser.add_argument("--duration", type=float, help="Override simulation duration in seconds.")
    parser.add_argument("--payload-mass", type=float, help="Override equivalent payload mass in kilograms.")
    parser.add_argument("--config", type=Path, help="Override the experiment configuration file.")
    parser.add_argument("--output-dir", type=Path, help="Override the mode-specific output directory.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    experiment = args.experiment
    if args.mode is not None and experiment == "phase1":
        experiment = "three_module"
    controller = args.mode or args.controller or ("pd" if experiment == "phase1" else "distributed")
    config_path = args.config or PROJECT_ROOT / "config" / ("default.yaml" if experiment == "phase1" else "phase2.yaml")
    config = deepcopy(load_config(config_path))
    if args.duration is not None:
        if args.duration <= 0:
            raise SystemExit("--duration must be positive")
        config["simulation"]["duration"] = args.duration
    if args.payload_mass is not None:
        if args.payload_mass < 0:
            raise SystemExit("--payload-mass cannot be negative")
        config["payload"]["equivalent_mass"] = args.payload_mass
    if experiment == "phase1":
        phase1_controller = "off" if controller in {"off", "passive"} else "pd"
        if controller not in {"pd", "off", "passive"}:
            raise SystemExit("Phase 1 supports --controller pd or off")
        output = args.output_dir or PROJECT_ROOT / "results" / phase1_controller
        run_experiment(
            config=config,
            model_path=PROJECT_ROOT / "models" / "modular_balance.xml",
            output_dir=output,
            controller_mode=phase1_controller,
            headless=args.headless,
        )
        return

    if experiment in {"two_leg", "wheeled"}:
        raise SystemExit(f"The {experiment} prototype runner is installed in a later Phase-2 milestone.")
    phase2_controller = "passive" if controller == "off" else controller
    if phase2_controller not in {"distributed", "centralized", "passive"}:
        raise SystemExit("Phase 2 supports distributed, centralized, or passive control")
    connector = args.connector or ("magnetic" if experiment == "magnetic" else "ideal")
    suffix = f"{phase2_controller}_{args.communication}_{connector}"
    if args.latch == "on":
        suffix += "_latch"
    output = args.output_dir or PROJECT_ROOT / "results" / "phase2" / experiment / suffix
    run_phase2_experiment(
        config=config,
        model_path=PROJECT_ROOT / "models" / "three_module_robot.xml",
        output_dir=output,
        experiment=experiment,
        controller_mode=phase2_controller,
        communication_mode=args.communication,
        connector_type=connector,
        latch_enabled=args.latch == "on",
        headless=args.headless,
    )


if __name__ == "__main__":
    main()
