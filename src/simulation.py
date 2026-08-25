"""MuJoCo experiment orchestration, independent of rendering rate."""

from __future__ import annotations

import time as wall_time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from .agent import IntelligentModuleAgent
from .controller import OffConnectorController, PDConnectorController
from .disturbances import PayloadDisturbance, SmoothBaseMotion
from .evaluation import evaluate_experiment, print_summary, save_report
from .logger import ExperimentLogger
from .plotting import generate_experiment_plots
from .sensors import LocalSensorSuite


@dataclass(frozen=True)
class ExperimentResult:
    records: list[dict[str, Any]]
    report: dict[str, Any]
    output_dir: Path
    csv_path: Path
    plot_paths: list[Path]


def _named_id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise RuntimeError(f"Required MuJoCo object not found: {name}")
    return int(object_id)


def _make_agent(config: dict[str, Any], controller_mode: str) -> IntelligentModuleAgent:
    desired = np.deg2rad(float(config["controller"]["desired_pitch_deg"]))
    if controller_mode == "pd":
        policy = PDConnectorController(
            kp=float(config["controller"]["kp"]),
            kd=float(config["controller"]["kd"]),
            max_torque=float(config["controller"]["max_torque"]),
            desired_pitch=float(desired),
        )
    elif controller_mode == "off":
        policy = OffConnectorController(desired_pitch=float(desired))
    else:
        raise ValueError(f"Unknown controller mode '{controller_mode}'; expected 'pd' or 'off'")
    return IntelligentModuleAgent(policy)


def _configure_model(model: mujoco.MjModel, config: dict[str, Any]) -> None:
    model.opt.timestep = float(config["simulation"]["timestep"])
    model.opt.gravity[:] = np.asarray(config["simulation"]["gravity"], dtype=float)
    motor_id = _named_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "balance_motor")
    max_torque = float(config["controller"]["max_torque"])
    model.actuator_ctrlrange[motor_id, :] = (-max_torque, max_torque)
    model.actuator_forcerange[motor_id, :] = (-max_torque, max_torque)


def _box_vertical_bounds(
    model: mujoco.MjModel, data: mujoco.MjData, geom_name: str
) -> tuple[float, float]:
    """Compute a box geom's world-Z bounds from its actual pose and half-extents."""
    geom_id = _named_id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_BOX):
        raise ValueError(f"Geometry verification requires box geom '{geom_name}'")
    rotation = np.asarray(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
    half_extent_z = float(np.dot(np.abs(rotation[2, :]), model.geom_size[geom_id, :3]))
    center_z = float(data.geom_xpos[geom_id, 2])
    return center_z - half_extent_z, center_z + half_extent_z


def verify_model_geometry(
    model: mujoco.MjModel, data: mujoco.MjData, warning_threshold: float = 0.005
) -> dict[str, float]:
    """Print and return neutral-pose world geometry interfaces using MuJoCo data."""
    platform_bottom, platform_top = _box_vertical_bounds(model, data, "platform_geom")
    cube_a_bottom, cube_a_top = _box_vertical_bounds(model, data, "cube_a_geom")
    cube_b_bottom, cube_b_top = _box_vertical_bounds(model, data, "cube_b_geom")
    payload_bottom, payload_top = _box_vertical_bounds(model, data, "payload_visual")

    hinge_id = _named_id(model, mujoco.mjtObj.mjOBJ_JOINT, "balance_hinge")
    hinge_body_id = int(model.jnt_bodyid[hinge_id])
    body_rotation = np.asarray(data.xmat[hinge_body_id], dtype=float).reshape(3, 3)
    hinge_world = data.xpos[hinge_body_id] + body_rotation @ model.jnt_pos[hinge_id]

    values = {
        "platform_bottom_z": platform_bottom,
        "platform_top_z": platform_top,
        "cube_a_bottom_z": cube_a_bottom,
        "cube_a_top_z": cube_a_top,
        "cube_b_bottom_z": cube_b_bottom,
        "cube_b_top_z": cube_b_top,
        "payload_bottom_z": payload_bottom,
        "payload_top_z": payload_top,
        "hinge_z": float(hinge_world[2]),
        "gap_platform_cube_a": cube_a_bottom - platform_top,
        "gap_cube_a_cube_b": cube_b_bottom - cube_a_top,
        "gap_cube_b_payload": payload_bottom - cube_b_top,
    }
    print("==============================")
    print("Neutral Model Geometry Check")
    print("==============================")
    print(f"Platform bottom/top Z: {platform_bottom:.6f} / {platform_top:.6f} m")
    print(f"Cube A bottom/top Z:   {cube_a_bottom:.6f} / {cube_a_top:.6f} m")
    print(f"Cube B bottom/top Z:   {cube_b_bottom:.6f} / {cube_b_top:.6f} m")
    print(f"Payload bottom/top Z:  {payload_bottom:.6f} / {payload_top:.6f} m")
    print(f"Hinge world Z:         {values['hinge_z']:.6f} m")
    print(f"Gap platform->Cube A:  {values['gap_platform_cube_a']:.6f} m")
    print(f"Gap Cube A->Cube B:    {values['gap_cube_a_cube_b']:.6f} m")
    print(f"Gap Cube B->payload:   {values['gap_cube_b_payload']:.6f} m")
    for label, key in (
        ("platform->Cube A", "gap_platform_cube_a"),
        ("Cube A->Cube B", "gap_cube_a_cube_b"),
        ("Cube B->payload", "gap_cube_b_payload"),
    ):
        if values[key] > warning_threshold:
            print(f"WARNING: {label} gap {values[key]:.6f} m exceeds {warning_threshold:.3f} m")
    return values


def run_experiment(
    config: dict[str, Any],
    model_path: str | Path,
    output_dir: str | Path,
    controller_mode: str = "pd",
    headless: bool = True,
    print_results: bool = True,
) -> ExperimentResult:
    """Run one deterministic experiment and save data, report, and plots."""
    model_file = Path(model_path).resolve()
    if not model_file.is_file():
        raise FileNotFoundError(f"MuJoCo model not found: {model_file}")
    model = mujoco.MjModel.from_xml_path(str(model_file))
    _configure_model(model, config)
    data = mujoco.MjData(model)

    # Verify the intended stacked assembly in its neutral pose before applying
    # the deliberate five-degree initial pitch disturbance.
    mujoco.mj_forward(model, data)
    verify_model_geometry(model, data)

    hinge_joint_id = _named_id(model, mujoco.mjtObj.mjOBJ_JOINT, "balance_hinge")
    hinge_qpos_address = int(model.jnt_qposadr[hinge_joint_id])
    data.qpos[hinge_qpos_address] = np.deg2rad(float(config["model"]["initial_pitch_deg"]))
    mujoco.mj_forward(model, data)

    cube_b_id = _named_id(model, mujoco.mjtObj.mjOBJ_BODY, "cube_b")
    payload_site_id = _named_id(model, mujoco.mjtObj.mjOBJ_SITE, "payload_site")
    payload_geom_id = _named_id(model, mujoco.mjtObj.mjOBJ_GEOM, "payload_visual")
    configured_offset = float(config["payload"]["offset_x"])
    site_offset = float(model.site_pos[payload_site_id, 0])
    if not np.isclose(site_offset, configured_offset, atol=1e-9):
        raise ValueError(
            f"payload.offset_x ({configured_offset}) must match payload_site local X ({site_offset})"
        )
    balance_actuator_id = _named_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "balance_motor")
    base_actuator_id = _named_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "platform_position_servo")
    sensors = LocalSensorSuite(model)
    agent = _make_agent(config, controller_mode)
    payload = PayloadDisturbance(
        start_time=float(config["payload"]["start_time"]),
        equivalent_mass=float(config["payload"]["equivalent_mass"]),
        offset_x=float(config["payload"]["offset_x"]),
        gravity=abs(float(config["simulation"]["gravity"][2])),
    )
    base_motion = SmoothBaseMotion(
        start_time=float(config["base_motion"]["start_time"]),
        target_velocity=float(config["base_motion"]["target_velocity"]),
        ramp_duration=float(config["base_motion"]["ramp_duration"]),
    )

    duration = float(config["simulation"]["duration"])
    control_decimation = int(config["simulation"]["control_decimation"])
    logger = ExperimentLogger()
    last_action = agent.act(sensors.observe(data))
    data.ctrl[balance_actuator_id] = last_action.torque

    def step_once(step_index: int) -> None:
        nonlocal last_action
        payload_active = payload.apply(data, cube_b_id, payload_site_id)
        model.geom_rgba[payload_geom_id, 3] = 0.90 if payload_active else 0.28
        target_position, target_velocity = base_motion.target(float(data.time))
        data.ctrl[base_actuator_id] = target_position
        if step_index % control_decimation == 0:
            last_action = agent.act(sensors.observe(data))
            data.ctrl[balance_actuator_id] = last_action.torque
        mujoco.mj_step(model, data)
        observation = sensors.observe(data)
        connector_torque = sensors.values(data, "connector_torque")
        logger.append(
            time=float(data.time),
            pitch_deg=float(np.rad2deg(observation.module_pitch)),
            roll_deg=float(np.rad2deg(observation.module_roll)),
            joint_angle=observation.joint_angle,
            joint_angle_deg=float(np.rad2deg(observation.joint_angle)),
            joint_velocity=observation.joint_velocity,
            controller_torque_command=last_action.torque,
            actuator_torque_applied=observation.measured_joint_torque,
            connector_force=observation.connector_force,
            connector_torque_y=float(connector_torque[1]),
            base_position=sensors.scalar(data, "platform_position"),
            base_velocity=sensors.scalar(data, "platform_velocity"),
            base_target_position=target_position,
            base_target_velocity=target_velocity,
            payload_active=int(payload_active),
            controller_error_rad=last_action.error,
            controller_error_deg=float(np.rad2deg(last_action.error)),
            controller_output=last_action.torque,
            actuator_saturated=int(last_action.saturated),
        )
        if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
            raise FloatingPointError(f"Simulation became non-finite at t={data.time:.4f} s")

    if headless:
        step_index = 0
        while data.time < duration - 0.5 * model.opt.timestep:
            step_once(step_index)
            step_index += 1
    else:
        try:
            from mujoco import viewer as mj_viewer

            start_wall = wall_time.perf_counter()
            step_index = 0
            with mj_viewer.launch_passive(model, data) as viewer:
                while viewer.is_running() and data.time < duration - 0.5 * model.opt.timestep:
                    step_once(step_index)
                    step_index += 1
                    viewer.sync()
                    ahead = start_wall + data.time - wall_time.perf_counter()
                    if ahead > 0:
                        wall_time.sleep(min(ahead, model.opt.timestep))
        except (ImportError, RuntimeError) as exc:
            raise RuntimeError(
                "MuJoCo viewer could not start. Use --headless on machines without a display."
            ) from exc

    output = Path(output_dir)
    csv_path = logger.save(output / "experiment.csv")
    report = evaluate_experiment(logger.records, config, controller_mode)
    save_report(report, output / "evaluation.json")
    plot_paths = generate_experiment_plots(logger.records, config, output)
    if print_results:
        print_summary(report)
        print("\nResults saved:")
        print(f"  {csv_path}")
        for plot_path in plot_paths:
            print(f"  {plot_path}")
    return ExperimentResult(logger.records, report, output, csv_path, plot_paths)
