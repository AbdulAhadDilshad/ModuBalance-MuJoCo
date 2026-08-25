"""Configuration loading and lightweight validation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")
    config = deepcopy(loaded)
    required_sections = ("simulation", "controller", "payload", "base_motion", "evaluation")
    missing = [section for section in required_sections if section not in config]
    if missing:
        raise ValueError(f"Missing configuration sections: {', '.join(missing)}")
    if "model" not in config and "modules" not in config:
        raise ValueError("Configuration must contain either a Phase-1 'model' or Phase-2 'modules' section")
    if float(config["simulation"]["timestep"]) <= 0:
        raise ValueError("simulation.timestep must be positive")
    if int(config["simulation"]["control_decimation"]) < 1:
        raise ValueError("simulation.control_decimation must be at least one")
    return config
