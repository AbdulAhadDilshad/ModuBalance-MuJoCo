"""Three-agent Phase-2 simulation with communication, connectors, latches, and topology events."""

from __future__ import annotations

import time as wall_time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from .agent import DistributedModuleAgent, ModuleAction, Phase2LocalObservation
from .communication import CommunicationChannel
from .connectors import (
    LATCH_OPEN,
    IdealTorqueConnector,
    MagneticConnector,
    MagneticLatchController,
    allocate_four_magnet_commands,
)
from .disturbances import PayloadDisturbance, SmoothBaseMotion
from .logger import ExperimentLogger
from .phase2_evaluation import evaluate_phase2, print_phase2_summary, save_phase2_report
from .phase2_plotting import generate_phase2_plots
from .sensors import Phase2SensorSuite
from .topology import ActiveTopology


@dataclass(frozen=True)
class Phase2RunResult:
    records: list[dict[str, Any]]
    report: dict[str, Any]
    output_dir: Path
    csv_path: Path | None
    plot_paths: list[Path]


def _id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    value = mujoco.mj_name2id(model, kind, name)
    if value < 0:
        raise RuntimeError(f"Required Phase-2 MuJoCo object not found: {name}")
    return int(value)


def _box_bounds(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> tuple[float, float]:
    geom_id = _id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    rotation = np.asarray(data.geom_xmat[geom_id]).reshape(3, 3)
    half_z = float(np.dot(np.abs(rotation[2]), model.geom_size[geom_id, :3]))
    center = float(data.geom_xpos[geom_id, 2])
    return center - half_z, center + half_z


def verify_three_module_geometry(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, float]:
    bounds = {
        "platform": _box_bounds(model, data, "platform_geom"),
        "A": _box_bounds(model, data, "cube_a_geom"),
        "B": _box_bounds(model, data, "cube_b_geom"),
        "C": _box_bounds(model, data, "cube_c_geom"),
        "payload": _box_bounds(model, data, "phase2_payload_visual"),
    }
    gaps = {
        "platform_A": bounds["A"][0] - bounds["platform"][1],
        "A_B": bounds["B"][0] - bounds["A"][1],
        "B_C": bounds["C"][0] - bounds["B"][1],
        "C_payload": bounds["payload"][0] - bounds["C"][1],
    }
    print("Phase-2 neutral geometry gaps (m): " + ", ".join(f"{key}={value:.6f}" for key, value in gaps.items()))
    if any(value > 0.005 for value in gaps.values()):
        raise RuntimeError(f"Phase-2 model contains an excessive visible gap: {gaps}")
    return gaps


def _centralized_actions(
    observations: dict[str, Phase2LocalObservation], config: dict[str, Any]
) -> dict[str, ModuleAction]:
    kp = float(config["controller"]["kp"])
    kd = float(config["controller"]["kd"])
    limit = float(config["controller"]["max_torque"])
    b, c = observations["B"], observations["C"]
    ab_pitch = -kp * (0.65 * b.local_pitch + 0.35 * c.local_pitch) - kd * (
        0.65 * b.local_pitch_rate + 0.35 * c.local_pitch_rate
    )
    bc_pitch = -0.85 * kp * c.local_pitch - kd * c.local_pitch_rate
    ab_roll = -kp * (0.65 * b.local_roll + 0.35 * c.local_roll) - kd * (
        0.65 * b.local_roll_rate + 0.35 * c.local_roll_rate
    )
    bc_roll = -0.85 * kp * c.local_roll - kd * c.local_roll_rate
    def action(pitch: float, roll: float, obs: Phase2LocalObservation) -> ModuleAction:
        return ModuleAction(
            ideal_torque=float(np.clip(pitch, -limit, limit)),
            pitch_signal=obs.local_pitch,
            roll_signal=obs.local_roll,
            saturated=abs(pitch) > limit or abs(roll) > limit,
        )
    return {"A": action(0.0, 0.0, observations["A"]), "B": action(ab_pitch, ab_roll, b), "C": action(bc_pitch, bc_roll, c)}


def run_phase2_experiment(
    config: dict[str, Any],
    model_path: str | Path,
    output_dir: str | Path,
    experiment: str = "three_module",
    controller_mode: str = "distributed",
    communication_mode: str = "nearest_neighbor",
    connector_type: str = "ideal",
    latch_enabled: bool = False,
    headless: bool = True,
    save_artifacts: bool = True,
    print_results: bool = True,
    seed: int | None = None,
    verify_geometry: bool = True,
) -> Phase2RunResult:
    model = mujoco.MjModel.from_xml_path(str(Path(model_path).resolve()))
    data = mujoco.MjData(model)
    model.opt.timestep = float(config["simulation"]["timestep"])
    model.geom_friction[:, 0] = float(config["friction"])
    mujoco.mj_forward(model, data)
    if verify_geometry:
        verify_three_module_geometry(model, data)

    # Joint-specific addresses avoid relying on numeric ordering.
    initial_angles = {
        "joint_AB": float(config["modules"]["initial_joint_ab_deg"]),
        "joint_BC": float(config["modules"]["initial_joint_bc_deg"]),
        "joint_AB_roll": float(config["modules"]["initial_roll_ab_deg"]),
        "joint_BC_roll": float(config["modules"]["initial_roll_bc_deg"]),
    }
    for joint_name, angle_deg in initial_angles.items():
        joint_id = _id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        data.qpos[int(model.jnt_qposadr[joint_id])] = np.deg2rad(angle_deg)
    mujoco.mj_forward(model, data)

    run_seed = int(config["simulation"]["seed"] if seed is None else seed)
    rng = np.random.default_rng(run_seed)
    sensors = Phase2SensorSuite(model, rng, config["sensor_noise"])
    topology = ActiveTopology()
    if experiment == "add_module":
        topology.remove_c()

    comm = config["communication"]
    channel_enabled = communication_mode != "none" and bool(comm["enabled"])
    delay_ms = float(comm["delay_ms"])
    if communication_mode == "delayed" and delay_ms <= 0:
        delay_ms = 50.0
    channel = CommunicationChannel(
        topology.communication_links(),
        delay_ms=delay_ms,
        packet_drop_probability=float(comm["packet_drop_probability"]),
        noise_std=float(comm["noise_std"]),
        seed=run_seed + 1,
        enabled=channel_enabled,
    )
    agents = {
        module: DistributedModuleAgent(
            module,
            kp=float(config["controller"]["kp"]),
            kd=float(config["controller"]["kd"]),
            max_torque=float(config["controller"]["max_torque"]),
            neighbour_weight=float(config["controller"]["neighbour_weight"]),
        )
        for module in "ABC"
    }

    body_ids = {module: _id(model, mujoco.mjtObj.mjOBJ_BODY, f"cube_{module.lower()}") for module in "ABC"}
    motor_ids = {name: _id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in (
        "motor_AB", "motor_AB_roll", "motor_BC", "motor_BC_roll", "platform_position_servo"
    )}
    max_torque = float(config["controller"]["max_torque"])
    if connector_type == "ideal":
        connectors: dict[str, Any] = {
            "AB": IdealTorqueConnector("AB", motor_ids["motor_AB"], max_torque, motor_ids["motor_AB_roll"]),
            "BC": IdealTorqueConnector("BC", motor_ids["motor_BC"], max_torque, motor_ids["motor_BC_roll"]),
        }
    elif connector_type == "magnetic":
        magnetic = config["magnetic"]
        connectors = {}
        for name, lower, upper in (("AB", "A", "B"), ("BC", "B", "C")):
            connectors[name] = MagneticConnector(
                name,
                body_ids[lower],
                body_ids[upper],
                [_id(model, mujoco.mjtObj.mjOBJ_SITE, f"{name}_lower_m{i}") for i in range(1, 5)],
                [_id(model, mujoco.mjtObj.mjOBJ_SITE, f"{name}_upper_m{i}") for i in range(1, 5)],
                maximum_force=float(magnetic["maximum_force_per_magnet"]),
                distance_scale=float(magnetic["distance_scale"]),
            )
        for motor_id in (motor_ids["motor_AB"], motor_ids["motor_AB_roll"], motor_ids["motor_BC"], motor_ids["motor_BC_roll"]):
            data.ctrl[motor_id] = 0.0
    else:
        raise ValueError("connector_type must be 'ideal' or 'magnetic'")

    latch_cfg = config["latch"]
    latches = {
        name: MagneticLatchController(
            _id(model, mujoco.mjtObj.mjOBJ_EQUALITY, f"latch_{name}"),
            np.deg2rad(float(latch_cfg["angle_threshold_deg"])),
            np.deg2rad(float(latch_cfg["unlock_angle_deg"])),
            float(latch_cfg["angular_velocity_threshold"]),
            float(latch_cfg["unlock_angular_velocity"]),
            float(latch_cfg["minimum_stable_time"]),
        )
        for name in ("AB", "BC")
    }
    data.eq_active[:] = 0

    payload_site = _id(model, mujoco.mjtObj.mjOBJ_SITE, "phase2_payload_site")
    payload_geom = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "phase2_payload_visual")
    payload = PayloadDisturbance(
        start_time=float(config["payload"]["start_time"]),
        equivalent_mass=float(config["payload"]["equivalent_mass"]),
        offset_x=0.06,
    )
    base_motion = SmoothBaseMotion(
        start_time=float(config["base_motion"]["start_time"]),
        target_velocity=float(config["base_motion"]["target_velocity"]),
        ramp_duration=float(config["base_motion"]["ramp_duration"]),
    )

    c_body, c_geom = body_ids["C"], _id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_c_geom")
    c_mass, c_inertia, c_rgba = float(model.body_mass[c_body]), model.body_inertia[c_body].copy(), model.geom_rgba[c_geom].copy()
    phase2_site_names = ["imu_C", "BC_upper_m1", "BC_upper_m2", "BC_upper_m3", "BC_upper_m4", "phase2_payload_site"]
    c_site_ids = [_id(model, mujoco.mjtObj.mjOBJ_SITE, name) for name in phase2_site_names]
    c_original_site_positions = {site: model.site_pos[site].copy() for site in c_site_ids}
    c_original_geom_position = model.geom_pos[c_geom].copy()
    payload_original_geom_position = model.geom_pos[payload_geom].copy()

    def set_c_active(active: bool) -> None:
        model.body_mass[c_body] = c_mass if active else max(0.02, c_mass * 0.01)
        model.body_inertia[c_body] = c_inertia if active else c_inertia * 0.01
        model.geom_rgba[c_geom] = c_rgba
        model.geom_rgba[c_geom, 3] = c_rgba[3] if active else 0.08
        model.geom_rgba[payload_geom, 3] = 0.28 if active else 0.03
        mujoco.mj_setConst(model, data)

    if experiment == "add_module":
        set_c_active(False)

    reconfigured = False
    connector_measurements = {name: connector.observe() for name, connector in connectors.items()}
    observations: dict[str, Phase2LocalObservation] = {}
    actions = {module: ModuleAction(0.0, 0.0, 0.0, False) for module in "ABC"}
    magnet_commands = {name: np.full(4, float(config["magnetic"]["nominal_command"])) for name in ("AB", "BC")}
    received_counts = {module: 0 for module in "ABC"}
    logger = ExperimentLogger()
    duration = float(config["simulation"]["duration"])
    decimation = int(config["simulation"]["control_decimation"])

    def handle_reconfiguration() -> None:
        nonlocal reconfigured
        if reconfigured or data.time < float(config["reconfiguration"]["event_time"]):
            return
        if experiment == "remove_module":
            topology.remove_c()
            set_c_active(False)
            data.eq_active[latches["BC"].equality_id] = 0
            ab_joint = _id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint_AB")
            data.qvel[int(model.jnt_dofadr[ab_joint])] += float(
                config["reconfiguration"]["removal_recoil_velocity"]
            )
        elif experiment == "add_module":
            topology.add_c()
            set_c_active(True)
            bc_joint = _id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint_BC")
            data.qpos[int(model.jnt_qposadr[bc_joint])] = np.deg2rad(
                float(config["reconfiguration"]["add_alignment_error_deg"])
            )
            data.qvel[int(model.jnt_dofadr[bc_joint])] = 0.0
            mujoco.mj_forward(model, data)
        elif experiment == "reposition_module":
            offset = float(config["reconfiguration"]["reposition_offset_x"])
            topology.reposition_c(offset)
            model.body_ipos[c_body, 0] += offset
            model.geom_pos[c_geom] = c_original_geom_position + np.array([offset, 0.0, 0.0])
            model.geom_pos[payload_geom] = payload_original_geom_position + np.array([offset, 0.0, 0.0])
            for site in c_site_ids:
                model.site_pos[site] = c_original_site_positions[site] + np.array([offset, 0.0, 0.0])
            mujoco.mj_setConst(model, data)
            bc_joint = _id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint_BC")
            data.qpos[int(model.jnt_qposadr[bc_joint])] += np.deg2rad(
                float(config["reconfiguration"]["reposition_alignment_deg"])
            )
            data.qvel[int(model.jnt_dofadr[bc_joint])] = 0.0
            mujoco.mj_forward(model, data)
        reconfigured = experiment in {"remove_module", "add_module", "reposition_module"}
        channel.update_links(topology.communication_links())

    def update_control() -> None:
        nonlocal observations, actions, magnet_commands
        observations = {
            "A": sensors.observe_module(data, "A", connector_measurements["AB"].force, connector_measurements["AB"].torque, topology.connectors["AB"]),
            "B": sensors.observe_module(data, "B", connector_measurements["AB"].force, connector_measurements["AB"].torque, topology.connectors["AB"]),
            "C": sensors.observe_module(data, "C", connector_measurements["BC"].force, connector_measurements["BC"].torque, topology.connectors["BC"]),
        }
        for module in "ABC":
            if topology.modules[module]:
                channel.publish(module, agents[module].message_payload(observations[module]), float(data.time))
        messages = {module: channel.receive(module, float(data.time)) for module in "ABC"}
        received_counts.update({module: len(messages[module]) for module in "ABC"})
        if controller_mode == "centralized":
            actions = _centralized_actions(observations, config)
        elif controller_mode == "distributed":
            actions = {module: agents[module].act(observations[module], messages[module]) for module in "ABC"}
        elif controller_mode == "passive":
            actions = {module: ModuleAction(0.0, 0.0, 0.0, False) for module in "ABC"}
        else:
            raise ValueError("controller_mode must be centralized, distributed, or passive")

        ab_pitch = actions["B"].ideal_torque
        bc_pitch = actions["C"].ideal_torque
        ab_roll = -float(config["controller"]["kp"]) * observations["B"].local_roll - float(config["controller"]["kd"]) * observations["B"].local_roll_rate
        bc_roll = -float(config["controller"]["kp"]) * observations["C"].local_roll - float(config["controller"]["kd"]) * observations["C"].local_roll_rate
        if connector_type == "ideal":
            if topology.connectors["AB"]:
                connector_measurements["AB"] = connectors["AB"].apply(model, data, [ab_pitch, ab_roll])
            else:
                connector_measurements["AB"] = connectors["AB"].apply(model, data, [0.0, 0.0])
            if topology.connectors["BC"]:
                connector_measurements["BC"] = connectors["BC"].apply(model, data, [bc_pitch, bc_roll])
            else:
                connector_measurements["BC"] = connectors["BC"].apply(model, data, [0.0, 0.0])
        else:
            magnetic = config["magnetic"]
            for connector_name, module in (("AB", "B"), ("BC", "C")):
                obs = observations[module]
                commands, _ = allocate_four_magnet_commands(
                    float(magnetic["nominal_command"]),
                    obs.local_pitch,
                    obs.local_pitch_rate,
                    obs.local_roll,
                    obs.local_roll_rate,
                    float(magnetic["pitch_gain"]),
                    float(magnetic["pitch_rate_gain"]),
                    float(magnetic["roll_gain"]),
                    float(magnetic["roll_rate_gain"]),
                )
                if not topology.connectors[connector_name] or controller_mode == "passive":
                    commands[:] = 0.0
                if latch_enabled and latches[connector_name].state != LATCH_OPEN:
                    commands *= float(latch_cfg["locked_holding_scale"])
                magnet_commands[connector_name] = commands

        if latch_enabled:
            for connector_name, module in (("AB", "B"), ("BC", "C")):
                if topology.connectors[connector_name]:
                    latches[connector_name].update(
                        data,
                        observations[module].local_joint_angle,
                        observations[module].local_joint_velocity,
                        float(data.time),
                    )

    update_control()

    def step_once(step_index: int) -> None:
        nonlocal connector_measurements
        handle_reconfiguration()
        data.qfrc_applied[:] = 0.0
        payload_active = payload.apply(data, body_ids["C"], payload_site) if topology.modules["C"] else False
        model.geom_rgba[payload_geom, 3] = 0.90 if payload_active else (0.28 if topology.modules["C"] else 0.03)
        if topology.c_offset_x != 0.0:
            # Equivalent gravity moment of Cube C's shifted COM about the BC hinge.
            offset_torque = np.array([0.0, topology.c_offset_x * c_mass * 9.81, 0.0])
            mujoco.mj_applyFT(
                model,
                data,
                np.zeros(3),
                offset_torque,
                data.xipos[body_ids["C"]],
                body_ids["C"],
                data.qfrc_applied,
            )
        disturbance = config["disturbance"]
        if float(disturbance["start_time"]) <= data.time < float(disturbance["start_time"]) + float(disturbance["duration"]):
            data.xfrc_applied[body_ids["C"], 0] += float(disturbance["force_x"])
        target_position, target_velocity = base_motion.target(float(data.time))
        data.ctrl[motor_ids["platform_position_servo"]] = target_position
        if step_index % decimation == 0:
            update_control()
        if connector_type == "magnetic":
            for name in ("AB", "BC"):
                if topology.connectors[name]:
                    connector_measurements[name] = connectors[name].apply(model, data, magnet_commands[name])
                else:
                    connector_measurements[name] = connectors[name].apply(model, data, np.zeros(4))
        mujoco.mj_step(model, data)
        if not observations:
            return
        row: dict[str, Any] = {
            "time": float(data.time),
            "payload_active": int(payload_active),
            "base_position": sensors.scalar(data, "phase2_platform_position"),
            "base_velocity": sensors.scalar(data, "phase2_platform_velocity"),
            "base_target_velocity": target_velocity,
            "topology_module_C_active": int(topology.modules["C"]),
            "topology_connector_AB_active": int(topology.connectors["AB"]),
            "topology_connector_BC_active": int(topology.connectors["BC"]),
            "topology_c_offset_x": topology.c_offset_x,
            "reconfiguration_event_applied": int(reconfigured),
            "connector_AB_force": connector_measurements["AB"].force,
            "connector_AB_torque": connector_measurements["AB"].torque,
            "connector_AB_latch_state": latches["AB"].state,
            "connector_BC_force": connector_measurements["BC"].force,
            "connector_BC_torque": connector_measurements["BC"].torque,
            "connector_BC_latch_state": latches["BC"].state,
            "latch_transition_count": latches["AB"].transitions + latches["BC"].transitions,
            "saturated_element_count": connector_measurements["AB"].saturated_elements + connector_measurements["BC"].saturated_elements,
            "control_effort": float(np.sum(np.abs(connector_measurements["AB"].commands)) + np.sum(np.abs(connector_measurements["BC"].commands))),
            "max_connector_separation": max(
                connector_measurements["AB"].max_separation,
                connector_measurements["BC"].max_separation,
            ),
            "detachment_failure": int(
                any(
                    topology.connectors[name]
                    and connector_measurements[name].max_separation
                    > float(config["evaluation"]["failure_connector_separation_m"])
                    for name in ("AB", "BC")
                )
            ),
            "communication_messages_A": received_counts["A"],
            "communication_messages_B": received_counts["B"],
            "communication_messages_C": received_counts["C"],
        }
        for module in "ABC":
            obs = observations[module]
            row.update({
                f"module_{module}_pitch": float(np.rad2deg(obs.local_pitch)),
                f"module_{module}_roll": float(np.rad2deg(obs.local_roll)),
                f"module_{module}_pitch_rate": obs.local_pitch_rate,
                f"module_{module}_roll_rate": obs.local_roll_rate,
                f"module_{module}_joint_angle": obs.local_joint_angle,
                f"module_{module}_joint_velocity": obs.local_joint_velocity,
                f"module_{module}_action": actions[module].ideal_torque,
            })
        for name in ("AB", "BC"):
            measurement = connector_measurements[name]
            for magnet in range(4):
                row[f"{name}_m{magnet + 1}_command"] = float(measurement.commands[magnet])
                row[f"{name}_m{magnet + 1}_force"] = float(measurement.element_forces[magnet])
        logger.append(**row)

    if headless:
        index = 0
        while data.time < duration - 0.5 * model.opt.timestep:
            step_once(index)
            index += 1
    else:
        from mujoco import viewer as mj_viewer
        start_wall = wall_time.perf_counter()
        index = 0
        with mj_viewer.launch_passive(model, data) as viewer:
            while viewer.is_running() and data.time < duration - 0.5 * model.opt.timestep:
                step_once(index)
                index += 1
                viewer.sync()
                ahead = start_wall + data.time - wall_time.perf_counter()
                if ahead > 0:
                    wall_time.sleep(min(ahead, model.opt.timestep))

    report = evaluate_phase2(logger.records, config)
    output = Path(output_dir)
    csv_path: Path | None = None
    plot_paths: list[Path] = []
    if save_artifacts:
        csv_path = logger.save(output / "experiment.csv")
        save_phase2_report(report, output / "evaluation.json")
        plot_paths = generate_phase2_plots(logger.records, config, output)
    if print_results:
        print_phase2_summary(report)
        if csv_path is not None:
            print("Results saved:")
            print(f"  {csv_path}")
            for path in plot_paths:
                print(f"  {path}")
    return Phase2RunResult(logger.records, report, output, csv_path, plot_paths)
