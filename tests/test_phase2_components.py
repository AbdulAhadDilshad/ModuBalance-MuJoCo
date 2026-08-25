from pathlib import Path

import mujoco
import numpy as np
import pytest

from src.communication import CommunicationChannel
from src.connectors import LATCH_LOCKED, MagneticLatchController, allocate_four_magnet_commands
from src.phase2_simulation import verify_three_module_geometry
from src.topology import ActiveTopology


ROOT = Path(__file__).resolve().parents[1]


def test_nearest_neighbour_channel_cannot_send_a_to_c() -> None:
    topology = ActiveTopology()
    channel = CommunicationChannel(topology.communication_links(), seed=1)
    assert channel.neighbours("A") == ("B",)
    assert channel.neighbours("B") == ("A", "C")
    assert channel.neighbours("C") == ("B",)
    channel.publish("A", {"pitch_error": 1.0}, time=0.0)
    assert channel.receive("C", time=1.0) == ()


def test_magnet_allocator_creates_pitch_and_roll_differentials() -> None:
    commands, saturated = allocate_four_magnet_commands(
        nominal=0.5,
        pitch=0.05,
        pitch_rate=0.0,
        roll=0.03,
        roll_rate=0.0,
        pitch_gain=2.0,
        pitch_rate_gain=0.0,
        roll_gain=2.0,
        roll_rate_gain=0.0,
    )
    front = np.mean(commands[:2])
    rear = np.mean(commands[2:])
    left = np.mean(commands[[0, 2]])
    right = np.mean(commands[[1, 3]])
    assert rear > front  # positive pitch demands a negative correcting moment
    assert left > right  # positive roll demands stronger +Y magnets
    assert saturated == 0


def test_phase2_model_has_connected_stack_and_four_magnet_pairs() -> None:
    model = mujoco.MjModel.from_xml_path(str(ROOT / "models" / "three_module_robot.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    gaps = verify_three_module_geometry(model, data)
    assert gaps["A_B"] == pytest.approx(0.0, abs=1e-9)
    assert gaps["B_C"] == pytest.approx(0.0, abs=1e-9)
    for connector in ("AB", "BC"):
        for side in ("lower", "upper"):
            assert all(
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{connector}_{side}_m{i}") >= 0
                for i in range(1, 5)
            )


def test_latch_requires_stable_dwell() -> None:
    model = mujoco.MjModel.from_xml_path(str(ROOT / "models" / "three_module_robot.xml"))
    data = mujoco.MjData(model)
    equality_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "latch_AB")
    latch = MagneticLatchController(equality_id, 0.02, 0.05, 0.1, 0.3, 0.5)
    assert latch.update(data, 0.0, 0.0, 0.0) != LATCH_LOCKED
    assert latch.update(data, 0.0, 0.0, 0.49) != LATCH_LOCKED
    assert latch.update(data, 0.0, 0.0, 0.50) == LATCH_LOCKED
    assert bool(data.eq_active[equality_id])
