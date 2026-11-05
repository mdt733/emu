#!/usr/bin/env python3
"""Read-only real-MuJoCo swing-foot reach envelope with the other foot grounded.

This is a KINEMATIC necessary-condition probe, not a dynamic stair certificate.
The real XML owns segment geometry, joint ranges, and foot radius; no joints
are extended, motors strengthened, collision forces disabled or actor altered.
MuJoCo's current contact semantics (thin non-foot rods disabled) are reported.

Run: python scripts/probe_stair_kinematics.py [--output /path/to/result.json]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from emu.contracts import ROBOT_XML


def measure_reach(*, pitch_step_deg=5, knee_step_deg=5):
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
    data = mujoco.MjData(model)
    if model.opt.timestep <= 0 or model.nv < 12:
        raise RuntimeError("Unexpected EMU robot physics")
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    balls = {
        side: mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_foot_ball")
        for side in ("r", "l")
    }
    joints = {
        side: {
            kind: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                   f"{side}_{kind}")
            for kind in ("hip_roll", "hip_pitch", "knee")
        } for side in balls
    }
    if any(v < 0 for v in balls.values()) or any(
            ident < 0 for group in joints.values() for ident in group.values()):
        raise RuntimeError("Missing robot joint or foot geometry")
    if not math.isclose(model.geom_size[balls["r"], 0], .018, abs_tol=1e-5):
        raise RuntimeError("Unexpected foot contact radius")
    limits = {}
    for side in balls:
        limits[side] = {}
        for name, joint_id in joints[side].items():
            lo, hi = np.rad2deg(model.jnt_range[joint_id])
            limits[side][name] = (float(lo), float(hi))
    heights_m = (.01, .02, .025, .03, .035, .04)
    margin_m = .002
    # A forward relative lead foot by at least 20mm is required.
    minimum_forward_m = .02
    max_lateral_m = .25
    results = []

    # Repeat with both possible stance legs; joints of the planted leg
    # remain neutral, while its physical sphere centre is put on the floor.
    # Only the whole torso free-joint Z is translated to establish contact.
    for stance_side, swing_side in (("l", "r"), ("r", "l")):
        stance = balls[stance_side]
        swing = balls[swing_side]
        hip = joints[swing_side]["hip_pitch"]
        knee = joints[swing_side]["knee"]
        roll = joints[swing_side]["hip_roll"]
        hip_lo, hip_hi = limits[swing_side]["hip_pitch"]
        knee_lo, knee_hi = limits[swing_side]["knee"]
        roll_lo, roll_hi = limits[swing_side]["hip_roll"]
        # Include both exact joint limits and zero; no illegal extrapolation.
        hips = np.unique(np.r_[np.arange(hip_lo, hip_hi + 1e-6,
                                        pitch_step_deg), hip_hi, 0.])
        knees = np.unique(np.r_[np.arange(knee_lo, knee_hi + 1e-6,
                                         knee_step_deg), knee_hi, 0.])
        rolls = sorted(set(float(np.clip(d,roll_lo,roll_hi))
                           for d in (-15., 0., 15.)))
        candidates = []
        for hp in hips:
            for kn in knees:
                for rl in rolls:
                    data.qpos[:] = model.qpos0
                    data.qvel[:] = 0.
                    data.qpos[model.jnt_qposadr[hip]] = math.radians(float(hp))
                    data.qpos[model.jnt_qposadr[knee]] = math.radians(float(kn))
                    data.qpos[model.jnt_qposadr[roll]] = math.radians(float(rl))
                    mujoco.mj_forward(model, data)
                    data.qpos[2] += float(
                        model.geom_size[stance, 0] - data.geom_xpos[stance, 2])
                    mujoco.mj_forward(model, data)
                    r = data.geom_xpos[swing] - data.geom_xpos[stance]
                    lateral = abs(float(r[1]))
                    forward = float(r[0])
                    height = float(
                        data.geom_xpos[swing, 2] - model.geom_size[swing, 0])
                    if (forward >= minimum_forward_m and lateral <= max_lateral_m):
                        com = data.subtree_com[torso_id]
                        com_offset = float(np.linalg.norm(
                            com[:2] - data.geom_xpos[stance, :2]))
                        candidates.append((
                            forward,height,com_offset,float(hp),float(kn),
                            float(rl),float(data.xpos[torso_id,2])))
        for rise in heights_m:
            qualified = [
                x for x in candidates if x[1] >= rise + margin_m]
            best = max(qualified, key=lambda x: (x[0],-x[2])) if qualified else None
            results.append(dict(
                stance=stance_side, swing=swing_side,
                riser_cm=100.*rise,
                collision_free_dynamic_climb_proven=False,
                candidate_poses=len(qualified),
                max_forward_reach_m=round(best[0],5) if best else None,
                swing_bottom_height_m=round(best[1],5) if best else None,
                com_to_stance_foot_xy_m=round(best[2],5) if best else None,
                swing_hip_pitch_deg=round(best[3],2) if best else None,
                swing_knee_deg=round(best[4],2) if best else None,
                swing_hip_roll_deg=round(best[5],2) if best else None,
                torso_height_m=round(best[6],5) if best else None))
    rods = [
        name for name in ("r_thigh_rod","l_thigh_rod",
                          "r_shin_upper","l_shin_upper",
                          "r_shin_lower_rod","l_shin_lower_rod")
        if (lambda gid: gid >= 0 and model.geom_contype[gid] == 0 and
            model.geom_conaffinity[gid] == 0)(
            mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,name))]
    return {
        "schema": "emu_v5051_static_supported_swing_reach_v1",
        "source_robot_xml": str(ROBOT_XML),
        "model_mass_kg": round(float(model.body_mass.sum()), 6),
        "foot_radius_m": float(model.geom_size[balls["r"],0]),
        "minimum_forward_reach_m": minimum_forward_m,
        "riser_margin_m": margin_m,
        "max_lateral_m": max_lateral_m,
        "sample_pitch_step_deg": pitch_step_deg,
        "sample_knee_step_deg": knee_step_deg,
        "limits_deg": limits,
        "noncontact_rods": rods,
        "warning": (
            "Static real-MuJoCo FK reach is a necessary condition only; "
            "no stance load, torque-speed limit, active impulse transfer, "
            "riser/shin nonpenetration or dynamic recovery is certified."
        ),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        help="Optional JSON destination, outside the checkout")
    args = parser.parse_args()
    result = measure_reach()
    for row in result["results"]:
        print(
            f"{row['stance']} stance / {row['swing']} swing "
            f"actual riser {row['riser_cm']:g}cm: "
            f"static reach {row['max_forward_reach_m']}m, "
            f"poses={row['candidate_poses']} "
            f"(NOT dynamic certification)", flush=True)
    if args.output:
        destination = args.output.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2) + "\n",
                               encoding="utf-8")
        print(f"Evidence: {destination}")


if __name__ == "__main__":
    main()
