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
from xml.etree import ElementTree as ET

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


# Temporary spring candidates are NOT production models. The 30/30 mm original
# causes overlapping anchor sites at -90 degrees, so candidate geometry may
# deliberately use 35/25 mm offsets. Targets are *neutral knee torque*, not
# actuator gains or changes to the frozen policy's [-33.2,+33.2] target span.
SPRING_CANDIDATES = (
    ("30_30_same_neutral_soft", .030, .030, 150., .520606),
    ("35_25_long_preload", .035, .025, 80., .350),
    ("35_25_gentle_low_rate", .035, .025, 150., .350),
    ("35_25_same_neutral_support", .035, .025, 150., .520606),
    ("35_25_gentler_low_rate", .035, .025, 150., .250),
    ("35_25_medium_rate", .035, .025, 300., .350),
)
DESIGN_ANGLES_DEG = (-90, -85, -70, -55, -40, -30, -20, 0, 30, 60, 90)


def _spring_candidate(xml, thigh_arm_m, shin_arm_m, stiffness_n_m,
                      neutral_support_nm):
    """In-memory candidate; preserve the generated H40 XML and actor.

    A tendon with 30/30mm anchors collapses to zero length at -90deg.
    Asymmetric 35/25mm anchors retain a 10mm separation at that limit.
    Design-only -90deg joint travel is NOT approved hardware clearance.
    """
    if not (0.015 <= thigh_arm_m <= .050 and
            0.015 <= shin_arm_m <= .050 and
            10. <= stiffness_n_m <= 500. and
            0. <= neutral_support_nm <= .6):
        raise ValueError("Candidate spring outside research envelope")
    l_neutral = math.hypot(thigh_arm_m, shin_arm_m)
    lever_neutral = thigh_arm_m * shin_arm_m / l_neutral
    springlength = l_neutral + neutral_support_nm / (stiffness_n_m * lever_neutral)
    root = ET.fromstring(xml)
    for side in "rl":
        knee_body = root.find(f".//body[@name='{side}_shin']")
        thigh_site = root.find(f".//site[@name='{side}_knee_strut_thigh']")
        shin_site = root.find(f".//site[@name='{side}_knee_strut_shin']")
        tendon = root.find(f".//spatial[@name='{side}_knee_support']")
        knee = root.find(f".//joint[@name='{side}_knee']")
        if any(el is None for el in (knee_body, thigh_site, shin_site, tendon, knee)):
            raise RuntimeError("Generated H40 missing spring links")
        knee_xyz = [float(v) for v in knee_body.get("pos").split()]
        a = thigh_arm_m / math.sqrt(2.0)
        b = shin_arm_m / math.sqrt(2.0)
        thigh_site.set("pos", f"{knee_xyz[0]+a:.9f} 0 {knee_xyz[2]+a:.9f}")
        shin_site.set("pos", f"{b:.9f} 0 {-b:.9f}")
        tendon.set("stiffness", f"{stiffness_n_m:.9f}")
        tendon.set("springlength", f"{springlength:.9f}")
        knee.set("range", "-90 95")  # research geometry only; no NN changes
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    return model, springlength



def _spring_force_curve(model, angles, rest_m):
    """Use measured MuJoCo tendon geometry and actual passive joint force."""
    result = _static(model, "r_knee", angles)
    data = mujoco.MjData(model)
    qpos, _ = _joint(model, "r_knee")
    tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON,
                            "r_knee_support")
    if tid < 0:
        raise RuntimeError("Missing right H40 strut")
    for row in result:
        mujoco.mj_resetData(model, data)
        data.qpos[qpos] = math.radians(row["angle_deg"])
        mujoco.mj_forward(model, data)
        length = float(data.ten_length[tid])
        row["attachment_separation_mm"] = round(1000. * length, 4)
        row["spring_compression_force_n"] = round(
            float(model.tendon_stiffness[tid]) * (rest_m - length), 4)
    return result


def spring_design_study(xml):
    """Passive torque measured with mj_forward, not a guessed knee lever."""
    base = mujoco.MjModel.from_xml_string(xml)
    root = ET.fromstring(xml)
    baseline_ref = float(root.find(
        ".//spatial[@name='r_knee_support']").get("springlength"))
    baseline_curve = _spring_force_curve(
        base, [x for x in DESIGN_ANGLES_DEG if x >= -55], baseline_ref)
    baseline_neutral = next(
        v["passive_torque_nm"] for v in baseline_curve if v["angle_deg"] == 0)
    rows = [dict(name="unchanged_champion", hardware_valid=True,
                 stiffness_n_m=float(base.tendon_stiffness[0]),
                 rest_length_mm=baseline_ref * 1000.,
                 thigh_arm_mm=30, shin_arm_mm=30,
                 neutral_support_nm=baseline_neutral,
                 max_passive_torque_nm=max(
                     v["passive_torque_nm"] for v in baseline_curve),
                 static=baseline_curve)]
    for name, a, b, k, tau0 in SPRING_CANDIDATES:
        model, rest = _spring_candidate(xml, a, b, k, tau0)
        # At -90 equal 30/30mm anchors coincide exactly, so MuJoCo has
        # no well-defined spatial tendon direction. Never sample that singularity.
        angles = (DESIGN_ANGLES_DEG if abs(a - b) > 1e-8
                  else [x for x in DESIGN_ANGLES_DEG if x > -90])
        curve = _spring_force_curve(model, angles, rest)
        normal = next(x["passive_torque_nm"] for x in curve if x["angle_deg"] == 0)
        if not math.isclose(normal, tau0, rel_tol=.005, abs_tol=.003):
            raise RuntimeError(f"MuJoCo spring force mismatch: {name}: {normal}")
        rows.append(dict(name=name, hardware_valid=False,
                         stiffness_n_m=k, rest_length_mm=rest * 1000.,
                         thigh_arm_mm=a * 1000., shin_arm_mm=b * 1000.,
                         neutral_support_nm=tau0,
                         max_passive_torque_nm=max(x["passive_torque_nm"]
                                                   for x in curve),
                         min_attachment_separation_mm=abs(a - b) * 1000.,
                         static=curve,
                         neutral_settle=_simulate(
                             _spring_candidate(xml, a, b, k, tau0)[0],
                             swing=False)))
    return dict(
        statement="research-only MuJoCo static sweeps, not buildable spring designs",
        candidate_joint_range_deg=[-90, 95],
        warning=("Original equal 30/30mm tendon sites coincide at -90deg. "
                 "The real foot also folds into the hip/chassis in this nominal geometry. "
                 "Neither physical clearance nor actuator reach is proven."),
        engineering=("MuJoCo tendon springlength is an effective reference separation, "
                     "NOT evidence that a compression cartridge of that length fits "
                     "between 10-60mm-spaced mounting points. A long spring and "
                     "preload require a distinct real guided linkage/rocker."),
        rows=rows)

def run(output: Path):
    output = output.expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("Choose a new or empty results directory")
    xml_path = Path(ROBOT_XML)
    xml = xml_path.read_text(encoding="utf-8")
    data = dict(
        schema="h40_mechanical_feasibility_v2",
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
    data["spring_design"] = spring_design_study(xml)
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
    lines += ["", "## Spring redesign (research models only)", "",
              "Reduced rate + preload is calibrated to neutral torque. "
              "A long, remote/rocker-mounted spring may be required.",
              "The -90° joint stop is ONLY used for candidate force-angle "
              "sweeps; a physical foot/torso collision remains unresolved.", "",
              "| Candidate | Effective rate N/m | Rest ref mm | Knee at 0° | Knee at -55° | Peak sampled knee torque | Neutral settle torso m |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for candidate in data["spring_design"]["rows"]:
        lookup = {v["angle_deg"]: v["passive_torque_nm"]
                  for v in candidate["static"]}
        settled = candidate.get("neutral_settle", {}).get("min_torso_z_m")
        lines.append(
            f"| {candidate['name']} | {candidate['stiffness_n_m']:.1f} | "
            f"{candidate['rest_length_mm'] if candidate['rest_length_mm'] is not None else float('nan'):.1f} | "
            f"{lookup[0]:+.3f} | {lookup[-55]:+.3f} | "
            f"{candidate.get('max_passive_torque_nm', float('nan')):.3f} | "
            f"{settled if settled is not None else float('nan'):.3f} |")
    lines += ["", "## Spring-force growth (neutral to deep flexion)", "",
              "| Candidate | Force at 0° | Force at −55° | Increase |",
              "|---|---:|---:|---:|"]
    for candidate in data["spring_design"]["rows"]:
        force = {v["angle_deg"]: v["spring_compression_force_n"]
                 for v in candidate["static"]}
        gain = 100. * (force[-55] / force[0] - 1.) if force[0] else float("nan")
        lines.append(f"| {candidate['name']} | {force[0]:.2f} N | "
                     f"{force[-55]:.2f} N | {gain:+.1f}% |")
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
