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

    hinge_joint_id = _named_id(model, mujoco.mjtObj.mjOBJ_JOINT, "balance_hinge")
    hinge_qpos_address = int(model.jnt_qposadr[hinge_joint_id])
    data.qpos[hinge_qpos_address] = np.deg2rad(float(config["model"]["initial_pitch_deg"]))
    mujoco.mj_forward(model, data)

    cube_b_id = _named_id(model, mujoco.mjtObj.mjOBJ_BODY, "cube_b")
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
        payload_active = payload.apply(data, cube_b_id)
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
            joint_angle_rad=observation.joint_angle,
            joint_angle_deg=float(np.rad2deg(observation.joint_angle)),
            joint_angular_velocity=observation.joint_velocity,
            commanded_joint_torque=last_action.torque,
            applied_joint_torque=observation.measured_joint_torque,
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
        print(f"\nData: {csv_path}")
        print(f"Plots: {len(plot_paths)} files in {output}")
    return ExperimentResult(logger.records, report, output, csv_path, plot_paths)
