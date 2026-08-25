from pathlib import Path

import mujoco
import pytest

from src.simulation import verify_model_geometry


ROOT = Path(__file__).resolve().parents[1]


def test_neutral_model_is_a_connected_vertical_stack() -> None:
    model = mujoco.MjModel.from_xml_path(str(ROOT / "models" / "modular_balance.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    geometry = verify_model_geometry(model, data)

    assert geometry["gap_platform_cube_a"] == pytest.approx(0.0, abs=1e-9)
    assert geometry["gap_cube_a_cube_b"] == pytest.approx(0.0, abs=1e-9)
    assert geometry["gap_cube_b_payload"] == pytest.approx(0.002, abs=1e-9)
    assert geometry["hinge_z"] == pytest.approx(geometry["cube_a_top_z"], abs=1e-9)
    assert geometry["hinge_z"] == pytest.approx(geometry["cube_b_bottom_z"], abs=1e-9)

