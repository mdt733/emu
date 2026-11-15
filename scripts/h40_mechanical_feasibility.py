"""Source-pinned H40 servo/spring feasibility probe; no policy training.

Four paired cases isolate passive knee/roll stiffness from motor force limits.
The swing trial is ZERO-G and airborne (no stance/contact). The neutral
flat-ground settle is a limited support proxy, NOT a gait or stairs exam.
Only the unchanged 1x/1x case represents the champion's rated physics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from emu.contracts import ACTUATED_JOINTS, ROBOT_XML, SIM_DT
from emu.environment import TARGET_SPAN
from emu.physical_variants import SERVO_RATED_CONTINUOUS_TORQUE_NM

# The same 4 arms, poses and timings: change exactly two physical variables.
ARMS = ((1.0, 1.0), (0.25, 1.0), (1.0, 2.0), (0.25, 2.0))
KNEE_ANGLES_DEG = (-55, -40, -30, -25, -20, 0, 30, 45, 90)
ROLL_ANGLES_DEG = (-25, -20, 0, 20, 25)
SWING_TARGET_DEG = -30.0


def _joint(model, name):
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if i < 0:
        raise ValueError(f"Missing H40 joint: {name}")
    return int(model.jnt_qposadr[i]), int(model.jnt_dofadr[i])


def _body(model, name):
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if i < 0:
        raise ValueError(f"Missing H40 body: {name}")
    return i


def _model(xml, spring_scale, torque_scale):
    # Modifications exist ONLY in this new, unshared in-memory MjModel.
    model = mujoco.MjModel.from_xml_string(xml)
    assert model.nu == 6 and model.ntendon == 2
    if not np.allclose(model.actuator_forcerange,
                       [[-SERVO_RATED_CONTINUOUS_TORQUE_NM,
                         SERVO_RATED_CONTINUOUS_TORQUE_NM]] * 6, atol=1e-6):
        raise RuntimeError("Unexpected champion actuator limits")
    model.tendon_stiffness[:] *= spring_scale
    for side in "rl":
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                              f"{side}_hip_roll")
        model.jnt_stiffness[j] *= spring_scale
    model.actuator_forcerange[:] *= torque_scale
    return model


def _static(model, joint, angles):
    d = mujoco.MjData(model)
    pos, dof = _joint(model, joint)
    result = []
    for deg in angles:
        mujoco.mj_resetData(model, d)
        d.qpos[pos] = math.radians(deg)
        mujoco.mj_forward(model, d)
        result.append(dict(angle_deg=deg,
                           passive_torque_nm=round(float(d.qfrc_passive[dof]), 6)))
    return result


def _simulate(model, *, swing):
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    pos, dof = _joint(model, "r_knee")
    knee_act = ACTUATED_JOINTS.index("r_knee")
    torso = _body(model, "torso")
    foot = _body(model, "r_foot")
    if swing:
        # No terrain or gravity can mask knee-vs-servo resistance here.
        model.opt.gravity[:] = 0.0
        data.qpos[2] = 1.0  # floating free root, well above the floor
        data.ctrl[knee_act] = math.radians(SWING_TARGET_DEG)
    mujoco.mj_forward(model, data)
    starting_relative_z = float(data.xpos[foot, 2] - data.xpos[torso, 2])
    peak_rise = 0.0
    min_knee = float(data.qpos[pos])
    max_abs_force = 0.0
    sat_frames = 0
    min_torso_z = float(data.xpos[torso, 2])
    grounded_frames = 0
    steps = round(0.4 / SIM_DT)
    limit = float(model.actuator_forcerange[knee_act, 1])
    for _ in range(steps):
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all():
            raise RuntimeError("Non-finite feasibility simulation")
        min_knee = min(min_knee, float(data.qpos[pos]))
        peak_rise = max(peak_rise,
                        float(data.xpos[foot, 2] - data.xpos[torso, 2])
                        - starting_relative_z)
        max_abs_force = max(max_abs_force, abs(float(data.actuator_force[knee_act])))
        sat_frames += abs(float(data.actuator_force[knee_act])) >= 0.98 * limit
        min_torso_z = min(min_torso_z, float(data.xpos[torso, 2]))
        grounded_frames += any(
            data.contact[i].geom1 == mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, "r_foot_ball")
            or data.contact[i].geom2 == mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, "r_foot_ball")
            for i in range(data.ncon))
    return dict(
        final_knee_deg=round(math.degrees(float(data.qpos[pos])), 3),
        deepest_knee_deg=round(math.degrees(min_knee), 3),
        max_relative_foot_rise_mm=round(peak_rise * 1000, 3),
        max_abs_knee_actuator_nm=round(max_abs_force, 5),
        knee_at_cap_fraction=round(sat_frames / steps, 4),
        min_torso_z_m=round(min_torso_z, 5),
        right_foot_contact_fraction=round(grounded_frames / steps, 4),
    )


def run(output: Path):
    output = output.expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("Choose a new or empty results directory")
    xml_path = Path(ROBOT_XML)
    xml = xml_path.read_text(encoding="utf-8")
    data = dict(
        schema="h40_mechanical_feasibility_v1",
        source_generated_h40_sha256=hashlib.sha256(xml.encode()).hexdigest(),
        source_xml=str(xml_path),
        rated_servo_nm=SERVO_RATED_CONTINUOUS_TORQUE_NM,
        protocol=dict(swing="0.4s gravity-free floating right-knee -30deg target",
                      settle="0.4s neutral target on flat MuJoCo floor; no policy",
                      pose_sweep="mj_forward zero-velocity passive qfrc at isolated joint angles",
                      change="knee tendon + hip-roll spring scale; or actuator force-range scale",
                      important="2x torque is a diagnostic only, not hardware-valid",
                      limits="No fixed prism riser, inverse kinematics, loaded swing, sensor or policy in this experiment"),
        cases=[])
    for spring_scale, torque_scale in ARMS:
        model = _model(xml, spring_scale, torque_scale)
        data["cases"].append(dict(
            case=("hardware_valid" if (spring_scale, torque_scale) == (1.0, 1.0)
                  else "diagnostic_only"),
            knee_and_roll_spring_scale=spring_scale,
            actuator_torque_scale=torque_scale,
            knee_passive_sweep=_static(model, "r_knee", KNEE_ANGLES_DEG),
            roll_passive_sweep=_static(model, "r_hip_roll", ROLL_ANGLES_DEG),
            swing=_simulate(model, swing=True),
            neutral_settle=_simulate(_model(xml, spring_scale, torque_scale), swing=False),
        ))
    output.mkdir(parents=True, exist_ok=True)
    (output / "mechanics.json").write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = [
        "# H40 mechanical feasibility — isolated diagnosis", "",
        f"Generated champion XML SHA256: `{data['source_generated_h40_sha256']}`", "",
        "The 1×/1× model is hardware-valid *within existing sim assumptions*; "
        "all other cases are untrained counterfactuals. "
        "The swing is unloaded zero-g, and the neutral settle is **not** trained balance.", "",
        "| Spring | Torque | Knee spring at −30° (Nm) | Swing knee min | Foot rise (mm) | Swing cap % | Settle min torso (m) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in data["cases"]:
        knee = next(x for x in c["knee_passive_sweep"] if x["angle_deg"] == -30)
        s, t = c["swing"], c["neutral_settle"]
        lines.append(
            f"| {c['knee_and_roll_spring_scale']:.2f}× | {c['actuator_torque_scale']:.0f}× | "
            f"{knee['passive_torque_nm']:+.3f} | {s['deepest_knee_deg']:.1f}° | "
            f"{s['max_relative_foot_rise_mm']:.1f} | {100*s['knee_at_cap_fraction']:.1f} | "
            f"{t['min_torso_z_m']:.3f} |")
    lines += ["", "## Interpretation guardrails", "",
              "- Spring improvement in an unloaded swing does not prove a climb.",
              "- Neutral settling is only a same-initial-state support proxy: it cannot certify stance or gait.",
              "- 2× torque is never a valid hardware or training configuration.",
              "- The next gate is matched loaded foot clearance and frozen-policy actuator requests.",
              ""]
    (output / "mechanics.md").write_text("\n".join(lines), encoding="utf-8")
    return output / "mechanics.json", output / "mechanics.md"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    for path in run(args.output):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
