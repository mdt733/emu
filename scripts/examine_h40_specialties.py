"""Independent H40 specialist physical exams: long true-prism terrain and birth recovery.

Research-only frozen-actor diagnostics; no training, teacher reward or PPO updates.
Command/transition/control are separately covered by emu.physical_capability_exam.
The terrain matrix is a **crossing** test, not a stationary-balance or route-planning
test. The birth-kick test measures recovery after an actual *initial velocity*,
not a timed applied push. All limits and gate values are provisional.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from hashlib import sha256
import io
import json
import math
from pathlib import Path
import platform
import time
import zipfile

import numpy as np

from emu.capability_focus import FOCUS_NAMES
from emu.contracts import CONTROL_DT
from emu.physical_capability_exam import Scenario, score_first_life, wilson_95

SCHEMA = "emu_h40_specialist_physical_v1"
STEP_METRES = (.010, .025, .050, .065, .080)
SLOPE_DEGREES = (3., 6., 10., 15.)
KICK_MPS = (.10, .25, .40, .60)
RECOVERY_FINAL_SECONDS = 5
RECOVERY_STABLE_SECONDS = 2
RECOVERY_SPEED_MPS = .10
RECOVERY_YAW_RAD_S = .15
RECOVERY_P95_TILT_DEG = 20.
RECOVERY_FOOT_SUPPORT = .75
RECOVERY_MAX_BAD_CONTACTS = .05


@dataclass(frozen=True)
class Challenge:
    world: int
    specialty: str
    kind: str
    level: int
    replicate: int
    vx: float = 0.
    vy: float = 0.
    yaw: float = 0.
    step_m: float = 0.
    grade_deg: float = 0.
    birth_kick_mps: float = 0.


def challenges(mode: str, *, worlds: int, repeats: int) -> list[Challenge]:
    """Fixed signed-grid protocol; no teacher random-difficulty relabelling."""
    if mode not in ("terrain", "recovery") or worlds < 2 or repeats < 1:
        raise ValueError("invalid specialist exam settings")
    specs: list[tuple[str, str, int, float, float, float]] = []
    if mode == "terrain":
        # A zero-height seam makes a true-physics forward control.
        specs.append(("command", "flat_control", 0, 0., 0., 0.))
        for sign, name in ((1, "uphill"), (-1, "downhill")):
            for i, value in enumerate(SLOPE_DEGREES, 1):
                specs.append(("slope", name, i, 0., sign * value, 0.))
        for sign, name in ((1, "step_up"), (-1, "step_down")):
            for i, value in enumerate(STEP_METRES, 1):
                specs.append(("step", name, i, sign * value, 0., 0.))
    else:
        specs.append(("resilience", "no_kick_control", 0, 0., 0., 0.))
        for sign, name in ((1, "forward_birth_kick"), (-1, "backward_birth_kick")):
            for i, value in enumerate(KICK_MPS, 1):
                specs.append(("resilience", name, i, 0., 0., sign * value))
    if len(specs) * repeats > worlds:
        raise ValueError(
            f"{mode} requires at least {len(specs) * repeats} worlds "
            f"for {repeats} repeats per difficulty, got {worlds}")
    rows = [
        Challenge(len_placeholder, specialty, kind, level, rep,
                  vx=.25 if mode == "terrain" else 0.,
                  step_m=step, grade_deg=grade, birth_kick_mps=kick)
        for specialty, kind, level, step, grade, kick in specs
        for rep in range(repeats)
        for len_placeholder in (0,)
    ]
    rows = [Challenge(i, **{k: v for k, v in asdict(row).items()
                             if k != "world"}) for i, row in enumerate(rows)]
    rows.extend(Challenge(i, "padding", "padding", 0, 0)
                for i in range(len(rows), worlds))
    return rows


def score_terrain(rows: list[Challenge], measured: dict,
                  *, horizon: int) -> list[dict]:
    """Exact measured geometry, independent two-foot support, then crossing.

    A torso crossing the lip is never sufficient. A step-up/down has to
    demonstrate five frames of top support for each foot and five upright
    frames 60cm beyond the seam. Lane/heading discipline stays a separate
    stronger qualification, not a hidden terrain requirement.
    """
    required = ("actual_lip_m", "actual_grade_deg", "survived",
                "both_feet_top_five", "traversed_upright_5frame",
                "traversed_5frame", "completed_destination", "fall_frame",
                "first_life_frames", "best_forward_progress_m",
                "speed_mae_mps", "yaw_mae_rad_s", "mean_support")
    arrays = {}
    for name in required:
        if name not in measured:
            raise ValueError(f"missing MJX physical evidence: {name}")
        arrays[name] = np.asarray(measured[name])
        if arrays[name].shape != (len(rows),):
            raise ValueError(f"wrong shape for {name}")
    result = []
    for tr in rows:
        if tr.specialty == "padding":
            continue
        i = tr.world
        lip = float(arrays["actual_lip_m"][i])
        grade = float(arrays["actual_grade_deg"][i])
        if not np.isfinite(lip) or abs(lip-tr.step_m) > .0005:
            raise RuntimeError(f"{tr.kind} world {i}: actual lip mismatch")
        if not np.isfinite(grade) or abs(grade-tr.grade_deg) > .30:
            raise RuntimeError(f"{tr.kind} world {i}: actual grade mismatch")
        if not all(math.isfinite(float(arrays[field][i]))
                   for field in ("best_forward_progress_m", "speed_mae_mps",
                                 "yaw_mae_rad_s", "mean_support")):
            raise FloatingPointError(f"{tr.kind} world {i}: nonfinite score")
        survived = bool(arrays["survived"][i])
        both = bool(arrays["both_feet_top_five"][i])
        upright_crossed = both and bool(arrays["traversed_upright_5frame"][i])
        aligned_crossed = upright_crossed and bool(arrays["traversed_5frame"][i])
        if not 0 <= int(arrays["first_life_frames"][i]) <= horizon:
            raise ValueError("first-life frame count outside exam")
        # Mission and durability are independent; require both for a PASS.
        # No strict speed-MAE gate on steps: a safe wind-up or brief pause
        # is legitimate when the physical course is actually completed.
        mission = survived and upright_crossed
        result.append(dict(
            **asdict(tr), measured_lip_m=lip, measured_grade_deg=grade,
            physical_each_foot_top_five=both,
            physical_upright_crossed=upright_crossed,
            physically_aligned_crossed=aligned_crossed,
            strict_destination_complete=bool(arrays["completed_destination"][i]),
            survived_full_horizon=survived, first_life_frames=int(
                arrays["first_life_frames"][i]),
            fall_frame=int(arrays["fall_frame"][i]),
            max_forward_progress_m=float(arrays["best_forward_progress_m"][i]),
            vx_mae_mps=float(arrays["speed_mae_mps"][i]),
            yaw_mae_rad_s=float(arrays["yaw_mae_rad_s"][i]),
            mean_support=float(arrays["mean_support"][i]),
            physical_mission_pass=mission,
            failure_reasons=",".join(
                reason for fails, reason in (
                    (not survived, "first_life_ended"),
                    (not both, "both_feet_not_supported_on_destination"),
                    (not upright_crossed, "not_physically_traversed"))
                if fails)))
    return result


def _rolling_mean(x: np.ndarray, width: int) -> np.ndarray:
    if x.size < width or width < 1:
        return np.empty((0,), dtype=float)
    return np.convolve(x, np.ones(width) / width, mode="valid")


def score_recovery(traces: dict, trials: list[Challenge],
                   *, horizon: int) -> list[dict]:
    """First-life velocity-kick recovery without absolute-position holding."""
    if horizon < 10 * 50 or horizon % 50:
        raise ValueError("Recovery exam needs >=10 full seconds")
    scenario = Scenario("birth_velocity_recovery", horizon // 50,
                        ((horizon // 50, 0., 0., 0.),), "balance")
    base_rows = score_first_life(traces, scenario)
    initial_vel = np.asarray(traces["root_qvel"])
    if initial_vel.shape != (horizon, len(trials), 6):
        raise ValueError("missing actual initial MJX root qvel")
    actual_xy = np.asarray(traces["physical_position_xy_after_step"])
    initial_xy = np.asarray(traces["position_xy"])
    headings = np.asarray(traces["physical_heading_rad_after_step"])
    initial_heading = np.asarray(traces["heading_rad"])
    detail = []
    window = RECOVERY_STABLE_SECONDS * 50
    last = RECOVERY_FINAL_SECONDS * 50
    for tr in trials:
        if tr.specialty == "padding":
            continue
        i = tr.world
        row = base_rows[i]
        n = int(row["first_life_frames"])
        xy = np.concatenate((initial_xy[0, i:i+1],
                             actual_xy[:n, i]), axis=0)
        heading = np.unwrap(np.concatenate((
            initial_heading[0, i:i+1], headings[:n, i])))
        speed = np.linalg.norm(np.diff(xy, axis=0), axis=1) / CONTROL_DT
        yaw = np.abs(np.diff(heading)) / CONTROL_DT
        recovery_speed = _rolling_mean(speed, window)
        recovery_yaw = _rolling_mean(yaw, window)
        candidates = np.flatnonzero(
            (recovery_speed <= RECOVERY_SPEED_MPS) &
            (recovery_yaw <= RECOVERY_YAW_RAD_S))
        time_to_recovery = (
            float((int(candidates[0]) + window) * CONTROL_DT)
            if len(candidates) else None)
        full = bool(row["survived_full_horizon"])
        final_speed = float(np.mean(speed[-last:])) if full else None
        final_yaw = float(np.mean(yaw[-last:])) if full else None
        failures = []
        if not full:
            failures.append("first_life_ended")
        if row["p95_tilt_deg"] > RECOVERY_P95_TILT_DEG:
            failures.append("torso_tilt")
        if row["contact_fraction"] < RECOVERY_FOOT_SUPPORT:
            failures.append("insufficient_foot_contact")
        if row["bad_contact_fraction"] > RECOVERY_MAX_BAD_CONTACTS:
            failures.append("bad_contacts")
        if final_speed is None or final_speed > RECOVERY_SPEED_MPS:
            failures.append("not_settled_linear_final5s")
        if final_yaw is None or final_yaw > RECOVERY_YAW_RAD_S:
            failures.append("not_settled_yaw_final5s")
        if time_to_recovery is None:
            failures.append("no_sustained_2s_recovery")
        detail.append(dict(
            **asdict(tr),
            measured_initial_world_x_velocity_mps=float(initial_vel[0, i, 0]),
            survived_full_horizon=full,
            first_life_frames=n,
            p95_tilt_deg=row["p95_tilt_deg"],
            contact_fraction=row["contact_fraction"],
            bad_contact_fraction=row["bad_contact_fraction"],
            max_radius_m=row["max_radius_m"],
            final_drift_m=row["final_drift_m"],
            final5s_mean_horizontal_speed_mps=final_speed,
            final5s_mean_abs_yaw_rad_s=final_yaw,
            time_to_sustained_recovery_s=time_to_recovery,
            positive_motor_shaft_work_j=row["positive_motor_shaft_work_j"],
            physical_mission_pass=not failures,
            failure_reasons=",".join(failures)))
    return detail


def summarize(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Per-direction/per-severity Wilson statistics and tested-only AUC."""
    groups: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["kind"], row["level"]), []).append(row)
    curves = []
    for (kind, level), subset in sorted(groups.items()):
        n = len(subset)
        pass_count = sum(bool(r["physical_mission_pass"]) for r in subset)
        survive_count = sum(bool(r["survived_full_horizon"]) for r in subset)
        lo, hi = wilson_95(pass_count, n)
        v = subset[0]
        severity = (abs(float(v["birth_kick_mps"])) if v["specialty"] == "resilience"
                    else abs(float(v["step_m"])) if v["specialty"] == "step"
                    else abs(float(v["grade_deg"])) if v["specialty"] == "slope"
                    else 0.)
        unit = ("m/s" if v["specialty"] == "resilience" else
                "m" if v["specialty"] == "step" else
                "degrees" if v["specialty"] == "slope" else "none")
        curves.append(dict(
            specialty=v["specialty"], kind=kind, level=level,
            severity=severity, severity_unit=unit, trials=n,
            physical_mission_passes=pass_count, full_survivals=survive_count,
            physical_pass_fraction=pass_count/n,
            wilson95_lo=lo, wilson95_hi=hi))
    envelopes = []
    for kind in sorted({x["kind"] for x in curves}):
        group = sorted((x for x in curves if x["kind"] == kind),
                       key=lambda x: x["severity"])
        auc = (float(np.trapezoid(
            [x["physical_pass_fraction"] for x in group],
            [x["severity"] for x in group]))
            if len(group) >= 2 else None)
        envelopes.append(dict(
            kind=kind, severity_unit=group[0]["severity_unit"],
            measured_levels=len(group), first_severity=group[0]["severity"],
            last_severity=group[-1]["severity"],
            tested_range_auc=auc,
            auc_interpretation="raw integral over measured severity only"
            if auc is not None else "not measured: fewer than 2 levels"))
    return curves, envelopes


def _csv(rows: list[dict]) -> bytes:
    if not rows:
        raise ValueError("Cannot serialize an empty exam")
    b = io.StringIO()
    writer = csv.DictWriter(b, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return b.getvalue().encode("utf-8")


def archive_results(path: Path, manifest: dict, rows: list[dict],
                    curves: list[dict], envelopes: list[dict],
                    traces: dict | None = None) -> None:
    path = Path(path).expanduser().resolve()
    if path.exists():
        raise FileExistsError(f"Never overwrite an exam: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = dict(manifest=manifest, curves=curves, envelopes=envelopes)
    md = [
        "# EMU H40 independent physical specialist exam",
        "",
        "Frozen actor, true MJX prism contact, first life only.",
        "No reward-based success or composite champion score.",
        "Pass thresholds remain provisional.",
        "",
        "| Physical specialty | Challenge | Severity | Survived | Mission pass |",
        "|---|---|---:|---:|---:|",
    ]
    for r in curves:
        md.append(f'| {r["specialty"]} | {r["kind"]} | '
                  f'{r["severity"]:g} {r["severity_unit"]} | '
                  f'{r["full_survivals"]}/{r["trials"]} | '
                  f'{r["physical_mission_passes"]}/{r["trials"]} |')
    md.extend(("", "A single measured difficulty has no AUC.", ""))
    partial = path.with_suffix(path.suffix+".partial")
    try:
        with zipfile.ZipFile(partial, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("manifest.json", json.dumps(manifest, indent=2)+"\n")
            z.writestr("summary.json", json.dumps(summary, indent=2)+"\n")
            z.writestr("summary.md", "\n".join(md))
            z.writestr("worlds.csv", _csv(rows))
            z.writestr("curves.csv", _csv(curves))
            z.writestr("envelopes.csv", _csv(envelopes))
            for name, payload in sorted((traces or {}).items()):
                if not name.startswith("traces/") or not name.endswith(".npz"):
                    raise ValueError("Invalid diagnostic trace filename")
                z.writestr(name, payload)
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


def run(*, policy: Path, output: Path, mode: str = "terrain",
        worlds: int | None = None, repeats: int | None = None,
        horizon: int | None = None, seed: int = 1701,
        device: str = "cuda") -> dict:
    if mode not in ("terrain", "recovery") or seed < 0 or device not in ("cpu", "cuda"):
        raise ValueError("invalid specialist mode/seed/device")
    if worlds is None:
        worlds = 192 if mode == "terrain" else 24
    if repeats is None:
        repeats = 8 if mode == "terrain" else 2
    if horizon is None:
        horizon = 600 if mode == "terrain" else 1000
    if not 2 <= worlds <= 1024 or horizon < 250 or horizon > 1500:
        raise ValueError("invalid worlds/horizon for frozen speciality exam")
    if mode == "recovery" and horizon % 50:
        raise ValueError("recovery horizon must be a whole number of seconds")
    if mode == "recovery" and worlds > 64:
        raise ValueError("full physical recovery traces limited to <=64 worlds")
    trials = challenges(mode, worlds=worlds, repeats=repeats)
    policy = Path(policy).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if output.suffix.lower() != ".zip":
        raise ValueError("use an independent .zip output")
    if output.exists():
        raise FileExistsError(f"Refusing overwrite: {output}")
    if not policy.is_file() or policy.suffix != ".emupolicy":
        raise FileNotFoundError("Use an exported .emupolicy")
    if output == policy or output in policy.parents or policy in output.parents:
        raise ValueError("Exam output must be independent of policy")

    from emu.contracts import ROBOT_XML, OBS_SCHEMA_TERRAIN, sensor_observation_contract
    from emu.environment import EmuMJXEnv
    from emu.policy_export import load_exported_actor_for_exam
    from emu.ppo import make_trainer, configure_jax, require_backend
    from emu.physical_variants import selected_profile
    from emu.training_config import (
        PPO_LR, ROLLOUT_STEPS, PPO_ENTROPY_COST, PPO_VALUE_COST)
    actor_params, norm, actor = load_exported_actor_for_exam(policy)
    if (actor["observation_schema"] != OBS_SCHEMA_TERRAIN or
            actor["training"].get("terrain_carpet_prisms") is not True or
            not actor["array_only"]):
        raise ValueError("Need the source-matched 360-input real-prism H40 actor")
    configure_jax(device)
    import jax
    import jax.numpy as jnp
    import jaxlib
    require_backend(device == "cpu")
    train, control, sensor = (actor["training"], actor["control"],
                              actor["sensor_settings"])
    vi = control.get("velocity_i") or {}
    col = lambda name: np.asarray([getattr(t, name) for t in trials],
                                  dtype=np.float32)
    env = EmuMJXEnv(
        str(ROBOT_XML), num_envs=worlds, seed=seed,
        max_episode_steps=horizon+50, push_max_force_n=.7,
        observation_schema=OBS_SCHEMA_TERRAIN,
        terrain_step_max_m=.08, unified_capability_focus=True,
        terrain_carpet=True, terrain_carpet_prisms=True,
        solver_iterations=train.get("solver_iterations"),
        solver_ls_iterations=train.get("solver_ls_iterations"),
        academy_stream=(mode == "terrain"),
        academy_height_cap_m=.15, terrain_lesson="coherent-careful",
        attitude_profile=train["attitude_profile"],
        pitch_command_max_deg=float(train["pitch_command_max_deg"]),
        phase_speed_gain=float(control["phase_speed_gain"]),
        science_phase_lowcmd_scale=float(train.get("science_phase_lowcmd_scale", 1.)),
        velocity_i_ramp_s=float(vi.get("ramp_s", .4)),
        velocity_i_multiplier=float(vi.get("multiplier", 1.)),
        sensor_noise_scale=float(sensor.get("sensor_noise_scale", 0.)),
        exam_fixture_steps_m=col("step_m"),
        exam_fixture_grades_deg=col("grade_deg"),
        exam_fixture_kicks_mps=col("birth_kick_mps"),
        **{k: v for k, v in sensor.items() if k != "sensor_noise_scale"})
    expected = sensor_observation_contract(
        sensor, schema=OBS_SCHEMA_TERRAIN, array_only=True)
    if env.sensor_contract != expected:
        raise RuntimeError("source actor sensor mismatch")
    trainer = make_trainer(
        env, seed=seed, lr=PPO_LR, rollout=ROLLOUT_STEPS,
        epochs=1, minibatches=1,
        entropy_cost=PPO_ENTROPY_COST, value_cost=PPO_VALUE_COST)
    teacher = env.params_array(dict(
        specialist_mix=0., athletic_rehearsal_frac=0.,
        regime_frac_steady=1., regime_frac_transition=0.,
        regime_frac_resilience=0.,
        probe_cmd_vx=0., probe_cmd_vy=0., probe_cmd_yaw=0.,
        probe_push_n=0., probe_tilt_rad=0.,
        probe_linvel_mps=0., probe_angvel_rad_s=0.,
        probe_joint_rad=0., terrain_exposure=0., slope_exposure=0.,
        academy_episode_probability=0.))
    started = time.perf_counter()
    print(f"[specialties] {mode}: {worlds} worlds, {horizon/50:g}s, "
          f"{len([t for t in trials if t.specialty != 'padding'])} assigned",
          flush=True)
    trace_files = {}
    if mode == "terrain":
        commands = np.stack((col("vx"), col("vy"), col("yaw")), axis=1)
        score_mask = np.array(
            [t.specialty != "padding" for t in trials], dtype=bool)
        measured = jax.device_get(trainer["evaluate_matrix"](
            actor_params, teacher,
            jnp.asarray(norm["mean"]), jnp.asarray(norm["std"]),
            jnp.asarray(commands), jnp.asarray(score_mask),
            jax.random.PRNGKey(seed), horizon))
        rows = score_terrain(trials, measured, horizon=horizon)
    else:
        zero = jnp.zeros((horizon, 3), dtype=jnp.float32)
        traces = jax.device_get(trainer["evaluate_trace"](
            actor_params, teacher,
            jnp.asarray(norm["mean"]), jnp.asarray(norm["std"]),
            jax.random.PRNGKey(seed), horizon, 0.,
            command_schedule=zero))
        rows = score_recovery(traces, trials, horizon=horizon)
        # A representative full first-life record of one recoverer and one
        # failure is helpful, but not a restartable simulation.
        from emu.physical_capability_exam import (
            capture_representative_traces)
        base = score_first_life(
            traces, Scenario("birth_velocity_recovery", horizon//50,
                             ((horizon//50, 0., 0., 0.),), "balance"))
        trace_files = capture_representative_traces(
            traces, base,
            Scenario("birth_velocity_recovery", horizon//50,
                     ((horizon//50, 0., 0., 0.),), "balance"))
        del traces
    curves, envelopes = summarize(rows)
    root = Path(__file__).resolve().parents[1]
    source_files = ("scripts/examine_h40_specialties.py",
                    "emu/physical_capability_exam.py",
                    "emu/ppo.py", "emu/environment.py",
                    "emu/carpet_prism_mjx.py", "emu/terrain_carpet_mjx.py",
                    "emu/robot.xml", "emu/policy_export.py",
                    "emu/physical_variants.py", "emu/tof_geometry.py",
                    "emu/contracts.py", "emu/training_config.py",
                    "emu/domain_randomization.py", "emu/capability_focus.py")
    manifest = dict(
        schema=SCHEMA, status="PROVISIONAL_NOT_A_CHAMPION_RANKING",
        protocol="signed exact seam challenges" if mode == "terrain"
                 else "signed world-x birth-velocity recovery",
        mode=mode, control_dt_s=CONTROL_DT, horizon_frames=horizon,
        horizon_seconds=horizon*CONTROL_DT, seed=seed,
        worlds=worlds, assigned_worlds=len(rows), repeats=repeats,
        actor_artifact_sha256=actor["artifact_sha256"],
        actor_source=actor["source"], robot_model_sha256=actor["robot_model_sha256"],
        physical_profile=selected_profile(),
        source_sha256={f: sha256((root/f).read_bytes()).hexdigest()
                       for f in source_files},
        sensor_sha256=env.sensor_contract["sha256"],
        runtime=dict(python=platform.python_version(),
                     jax=jax.__version__, jaxlib=jaxlib.__version__,
                     mujoco=env.mujoco.mj_versionString()),
        simulator="MJX 8 true convex triangle prisms / 4 source 1m tiles",
        device=jax.default_backend(), wall_seconds=time.perf_counter()-started,
        scenarios=sorted({r["kind"] for r in rows}),
        first_life_only=True, no_optimizer_updates=True,
        no_teacher_reward_success=True,
        terrain_gate="each foot 5 top-support frames + 5 upright frames >=60cm beyond seam, full horizon survived",
        terrain_limit="exam-only source fixture hardcap: 15 degrees, 8cm; not physical ability frontier",
        recovery_gate=dict(
            duration_s=horizon*CONTROL_DT,
            final_segment_s=RECOVERY_FINAL_SECONDS,
            stable_window_s=RECOVERY_STABLE_SECONDS,
            final_speed_mps=RECOVERY_SPEED_MPS,
            final_yaw_rad_s=RECOVERY_YAW_RAD_S,
            p95_tilt_deg=RECOVERY_P95_TILT_DEG,
            foot_support=RECOVERY_FOOT_SUPPORT,
            max_bad_contact_fraction=RECOVERY_MAX_BAD_CONTACTS),
        exclusions=("No oblique/side slopes or lateral pushes in this source fixture; "
                    "no multi-step sequence and no viewer parity proof; "
                    "command/control/transitions retain the separate 30s/60s v1 exam; "
                    "no electrochemical motor energy measurement"))
    archive_results(output, manifest, rows, curves, envelopes, trace_files)
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("terrain", "recovery"),
                        default="terrain")
    parser.add_argument("--policy", type=Path,
                        default=Path("showcase/champion.emupolicy"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worlds", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    m = run(**vars(args))
    print(f"[specialties] archived {args.output}: {m['assigned_worlds']} "
          f"worlds, {m['status']}", flush=True)


if __name__ == "__main__":
    main()
