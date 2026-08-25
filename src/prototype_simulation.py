"""Standing-only two-leg and slow wheeled Phase-2H feasibility runners."""

from __future__ import annotations

import csv
import json
import time as wall_time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from .disturbances import SmoothBaseMotion
from .sensors import quaternion_to_roll_pitch


def _id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    value = mujoco.mj_name2id(model, kind, name)
    if value < 0:
        raise RuntimeError(f"Prototype model is missing {name}")
    return int(value)


def run_prototype(
    kind: str,
    config: dict[str, Any],
    model_path: str | Path,
    output_dir: str | Path,
    headless: bool = True,
) -> dict[str, Any]:
    model = mujoco.MjModel.from_xml_path(str(Path(model_path).resolve()))
    data = mujoco.MjData(model)
    model.opt.timestep = float(config["simulation"]["timestep"])
    if kind == "two_leg":
        pitch_joint, roll_joint = "stand_pitch", "stand_roll"
        pitch_motor, roll_motor = "stand_pitch_motor", "stand_roll_motor"
        orientation_sensor, gyro_sensor = "prototype_orientation", "prototype_gyro"
        body_name, payload_site_name = "robot_body", "prototype_payload_site"
        base_servo = None
    elif kind == "wheeled":
        pitch_joint, roll_joint = "wheeled_pitch", None
        pitch_motor, roll_motor = "wheeled_pitch_motor", None
        orientation_sensor, gyro_sensor = "wheeled_orientation", "wheeled_gyro"
        body_name, payload_site_name = "wheeled_body", "wheeled_payload_site"
        base_servo = _id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wheel_base_servo")
    else:
        raise ValueError("Prototype kind must be two_leg or wheeled")
    pitch_joint_id = _id(model, mujoco.mjtObj.mjOBJ_JOINT, pitch_joint)
    data.qpos[int(model.jnt_qposadr[pitch_joint_id])] = np.deg2rad(float(config["initial"]["pitch_deg"]))
    if roll_joint is not None:
        roll_joint_id = _id(model, mujoco.mjtObj.mjOBJ_JOINT, roll_joint)
        data.qpos[int(model.jnt_qposadr[roll_joint_id])] = np.deg2rad(float(config["initial"]["roll_deg"]))
    mujoco.mj_forward(model, data)
    pitch_motor_id = _id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, pitch_motor)
    roll_motor_id = _id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, roll_motor) if roll_motor else None
    body_id = _id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    site_id = _id(model, mujoco.mjtObj.mjOBJ_SITE, payload_site_name)
    orientation_id = _id(model, mujoco.mjtObj.mjOBJ_SENSOR, orientation_sensor)
    gyro_id = _id(model, mujoco.mjtObj.mjOBJ_SENSOR, gyro_sensor)
    orientation_slice = slice(int(model.sensor_adr[orientation_id]), int(model.sensor_adr[orientation_id] + model.sensor_dim[orientation_id]))
    gyro_slice = slice(int(model.sensor_adr[gyro_id]), int(model.sensor_adr[gyro_id] + model.sensor_dim[gyro_id]))
    base_motion = SmoothBaseMotion(**config["base_motion"]) if kind == "wheeled" else None
    records: list[dict[str, Any]] = []
    kp, kd, limit = (float(config["controller"][key]) for key in ("kp", "kd", "max_torque"))

    def step() -> None:
        data.xfrc_applied[body_id] = 0.0
        roll, pitch = quaternion_to_roll_pitch(np.asarray(data.sensordata[orientation_slice]))
        gyro = np.asarray(data.sensordata[gyro_slice])
        pitch_torque = float(np.clip(-kp * pitch - kd * gyro[1], -limit, limit))
        roll_torque = float(np.clip(-kp * roll - kd * gyro[0], -limit, limit))
        data.ctrl[pitch_motor_id] = pitch_torque
        if roll_motor_id is not None:
            data.ctrl[roll_motor_id] = roll_torque
        payload_active = data.time >= float(config["payload"]["start_time"])
        if payload_active:
            force = np.array([0.0, 0.0, -float(config["payload"]["equivalent_mass"]) * 9.81])
            lever = data.site_xpos[site_id] - data.xipos[body_id]
            data.xfrc_applied[body_id, :3] += force
            data.xfrc_applied[body_id, 3:] += np.cross(lever, force)
        disturbance = config["disturbance"]
        if float(disturbance["start_time"]) <= data.time < float(disturbance["start_time"]) + float(disturbance["duration"]):
            data.xfrc_applied[body_id, 0] += float(disturbance["force_x"])
            data.xfrc_applied[body_id, 1] += float(disturbance["force_y"])
        target_velocity = 0.0
        if base_motion is not None and base_servo is not None:
            target_position, target_velocity = base_motion.target(float(data.time))
            data.ctrl[base_servo] = target_position
        mujoco.mj_step(model, data)
        roll, pitch = quaternion_to_roll_pitch(np.asarray(data.sensordata[orientation_slice]))
        base_position = float(data.qpos[0]) if kind == "wheeled" else 0.0
        base_velocity = float(data.qvel[0]) if kind == "wheeled" else 0.0
        records.append({
            "time": float(data.time), "body_pitch": float(np.rad2deg(pitch)), "body_roll": float(np.rad2deg(roll)),
            "pitch_torque": pitch_torque, "roll_torque": roll_torque, "payload_active": int(payload_active),
            "base_position": base_position, "base_velocity": base_velocity, "base_target_velocity": target_velocity,
            "foot_contact_count": int(data.ncon), "com_x": float(data.subtree_com[body_id, 0]), "com_y": float(data.subtree_com[body_id, 1]),
        })

    duration = float(config["simulation"]["duration"])
    if headless:
        while data.time < duration - model.opt.timestep / 2:
            step()
    else:
        from mujoco import viewer as mj_viewer
        start = wall_time.perf_counter()
        with mj_viewer.launch_passive(model, data) as viewer:
            while viewer.is_running() and data.time < duration - model.opt.timestep / 2:
                step(); viewer.sync()
                ahead = start + data.time - wall_time.perf_counter()
                if ahead > 0: wall_time.sleep(min(ahead, model.opt.timestep))
    pitch = np.asarray([row["body_pitch"] for row in records]); roll = np.asarray([row["body_roll"] for row in records])
    report = {
        "peak_pitch_deg": float(np.max(np.abs(pitch))), "peak_roll_deg": float(np.max(np.abs(roll))),
        "rms_pitch_deg": float(np.sqrt(np.mean(pitch ** 2))), "rms_roll_deg": float(np.sqrt(np.mean(roll ** 2))),
        "minimum_foot_contacts": int(min(row["foot_contact_count"] for row in records)),
        "recovery_success": bool(max(np.max(np.abs(pitch)), np.max(np.abs(roll))) <= float(config["evaluation"]["failure_angle_deg"])),
    }
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    with (output / "experiment.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    with (output / "evaluation.json").open("w", encoding="utf-8") as handle: json.dump(report, handle, indent=2)
    fig, ax = plt.subplots(figsize=(10, 4.8)); ax.plot([r["time"] for r in records], pitch, label="Pitch"); ax.plot([r["time"] for r in records], roll, label="Roll")
    ax.axhline(float(config["evaluation"]["tolerance_deg"]), color="gray", linestyle=":"); ax.axhline(-float(config["evaluation"]["tolerance_deg"]), color="gray", linestyle=":")
    ax.set(xlabel="Time (s)", ylabel="Angle (deg)", title=f"{kind.replace('_', ' ').title()} attitude"); ax.grid(alpha=0.3); ax.legend(); fig.tight_layout(); fig.savefig(output / "body_attitude.png", dpi=160); plt.close(fig)
    print(f"{kind}: peak pitch={report['peak_pitch_deg']:.3f}, peak roll={report['peak_roll_deg']:.3f}, recovery={'PASS' if report['recovery_success'] else 'FAIL'}")
    print(f"Results saved: {output}")
    return report
