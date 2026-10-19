"""EMU v5018 P0B: exploratory robot.xml kinematic swing-foot envelope.

Grounded in six-joint limits and ball-foot positions from CPU MuJoCo FK.
A fixed torso and nominal support leg are NOT a dynamically stable gait.
The servo-speed bounding is an optimistic *joint angle* envelope, not
torque-limited achievable clearance. Do not use these numbers directly
to choose training obstacle height without dynamic trials.

Run: python scripts/v5018_clearance_envelope.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np

from emu.contracts import ROBOT_XML

SEED = 733
JOINTS = ("r_hip_roll", "r_hip_pitch", "r_knee")
WINDOWS_S = (.08, .16, .32, .60)
SERVO_NO_LOAD_SPEED_RAD_S = 4.40  # robot.xml's stated nominal value
SAMPLES = 2500
MIN_ADVANCE_M = .025


def main():
    model = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    right = model.geom("r_foot_ball").id
    left = model.geom("l_foot_ball").id
    baseline_r = np.asarray(d.geom_xpos[right]).copy()
    baseline_l = np.asarray(d.geom_xpos[left]).copy()
    joints = [model.joint(name) for name in JOINTS]
    ranges = np.asarray([model.jnt_range[j.id] for j in joints])
    addresses = [int(model.jnt_qposadr[j.id]) for j in joints]
    neutral = np.asarray([d.qpos[a] for a in addresses], dtype=float)
    rng = np.random.default_rng(SEED)
    reports = []
    for seconds in WINDOWS_S:
        angle_limit = SERVO_NO_LOAD_SPEED_RAD_S * seconds
        lower = np.maximum(ranges[:, 0], neutral - angle_limit)
        upper = np.minimum(ranges[:, 1], neutral + angle_limit)
        best = None
        viable_count = 0
        peak = -float("inf")
        for sample_idx in range(SAMPLES):
            angles = rng.uniform(lower, upper)
            for addr, angle in zip(addresses, angles):
                d.qpos[addr] = angle
            mujoco.mj_forward(model, d)
            p = np.asarray(d.geom_xpos[right])
            foot_forward = float(p[0] - baseline_r[0])
            foot_lift = float(p[2] - d.geom_xpos[left, 2])
            lateral_shift = float(p[1] - baseline_r[1])
            if foot_forward >= MIN_ADVANCE_M:
                viable_count += 1
                if foot_lift > peak:
                    peak = foot_lift
                    best = {
                        "sample_idx": sample_idx,
                        "foot_lift_relative_to_nominal_support_m": foot_lift,
                        "forward_advance_m": foot_forward,
                        "lateral_shift_m": lateral_shift,
                        "r_joint_angles_deg": np.degrees(angles).tolist(),
                    }
        reports.append({
            "window_s": seconds, "joint_angle_change_max_rad": angle_limit,
            "samples": SAMPLES, "samples_reaching_25mm_forward": viable_count,
            "highest_lift_while_advancing_25mm_m": peak if best else None,
            "best_kinematic_sample": best,
        })
    payload = {
        "model": "robot.xml: current six-servo STS3215-like simulation",
        "mujoco": mujoco.__version__,
        "seed": SEED,
        "neutral_right_ball_xyz_m": baseline_r.tolist(),
        "neutral_left_ball_xyz_m": baseline_l.tolist(),
        "joint_limits_rad": {name: limits.tolist() for name, limits in zip(JOINTS, ranges)},
        "estimated_unloaded_joint_speed_limit_rad_s": SERVO_NO_LOAD_SPEED_RAD_S,
        "time_windows": reports,
        "limitations": [
            "Torso fixed, left foot nominal, no balance constraint and no collision checking.",
            "Passive shin dynamics held at neutral; self contacts of rods ignored by baseline XML.",
            "The joint speed limit is optimistic; servo torque, latency, backlash and control lag ignored.",
            "Stall/rated torque alone cannot prove dynamically achievable clearance.",
            "Random sampling gives observed lower bounds on kinematic max, not certified maxima.",
            "Any training obstacle heights require real walking contact/clearance trials.",
        ],
    }
    out = Path("v5018_clearance_envelope.json")
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"windows": reports, "result_file": str(out)}, indent=2))


if __name__ == "__main__":
    main()
