from pathlib import Path

import mujoco
import yaml

from src.stress import one_factor_cases


ROOT = Path(__file__).resolve().parents[1]


def test_phase2h_models_compile_with_named_control_objects() -> None:
    expected = {
        "two_leg_modular_robot.xml": ("stand_pitch", "stand_pitch_motor"),
        "wheeled_modular_robot.xml": ("wheeled_pitch", "wheeled_pitch_motor"),
    }
    for filename, (joint, actuator) in expected.items():
        model = mujoco.MjModel.from_xml_path(str(ROOT / "models" / filename))
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint) >= 0
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator) >= 0


def test_stress_design_covers_every_requested_one_factor_value() -> None:
    with (ROOT / "config" / "stress.yaml").open("r", encoding="utf-8") as handle:
        stress = yaml.safe_load(handle)
    cases = one_factor_cases(stress)
    assert {case["payload"] for case in cases if case["sweep_parameter"] == "payload"} == set(stress["payloads"])
    assert {case["base_velocity"] for case in cases if case["sweep_parameter"] == "base_velocity"} == set(stress["base_velocities"])
    assert {case["magnet_strength_scale"] for case in cases if case["sweep_parameter"] == "magnet_strength_scale"} == set(stress["magnet_strength_scales"])
