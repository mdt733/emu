"""EMU H40 compact mid-training physical pulse: one 2048-world MJX GPU scan.

The six independent capability groups share a single frozen actor, exact true-
prism physics and one fixed episode length. No frame traces, separate per-
specialist GPU compiles, training rewards, optimizer updates or rankings.
This is a quick trend screen, NOT the 60s standing/endurance certification.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import platform
import tempfile
import time
import zipfile

import numpy as np

from emu.contracts import CONTROL_DT
from emu.physical_capability_exam import wilson_95

SCHEMA = "emu_h40_compact_specialist_pulse_v1"
DEFAULT_WORLDS = 2048
DEFAULT_HORIZON = 600  # 12 seconds; same JAX compile for every specialty
# Exactly 2048 independent first-life worlds. Scale down only for diagnostics.
PER_CASE_WORLDS = {
    "command": 48, "control": 64, "transition": 128,
    "slope": 48, "step": 32, "resilience": 32,
}
RECOVERY_MAX_SPEED = .10
RECOVERY_MAX_YAW = .15


@dataclass(frozen=True)
class Case:
    specialty: str
    kind: str
    level: int
    severity: float
    severity_unit: str
    vx: float = 0.
    vy: float = 0.
    yaw: float = 0.
    step_m: float = 0.
    grade_deg: float = 0.
    birth_kick_mps: float = 0.
    program: int = 0  # 0=fixed operator command, 1=forward/stop/reverse


@dataclass(frozen=True)
class Trial:
    world: int
    case: Case
    replicate: int


def registered_cases() -> tuple[Case, ...]:
    """46 physical tasks, fixed order and immutable difficulty definition."""
    cases = [
        Case("command", "forward", 1, .15, "m/s", vx=.15),
        Case("command", "forward", 2, .25, "m/s", vx=.25),
        Case("command", "reverse", 1, .15, "m/s", vx=-.15),
        Case("command", "reverse", 2, .25, "m/s", vx=-.25),
        Case("command", "strafe_left", 1, .15, "m/s", vy=.15),
        Case("command", "strafe_right", 1, .15, "m/s", vy=-.15),
        Case("command", "yaw_left", 1, .4, "rad/s", yaw=.4),
        Case("command", "yaw_right", 1, .4, "rad/s", yaw=-.4),
        Case("control", "zero_command_balance", 0, 0., "none"),
        Case("control", "forward_heading", 1, .25, "m/s", vx=.25),
        Case("control", "combined_left", 1, .25, "rad/s",
             vx=.2, vy=.1, yaw=.25),
        Case("control", "combined_right", 1, .25, "rad/s",
             vx=.2, vy=-.1, yaw=-.25),
        Case("transition", "forward_stop_reverse", 1, .25, "m/s",
             vx=.25, program=1),
        Case("transition", "reverse_stop_forward", 1, .25, "m/s",
             vx=-.25, program=1),
    ]
    for sign, name in ((1, "uphill"), (-1, "downhill")):
        for i, grade in enumerate((3., 6., 10., 15.), 1):
            cases.append(Case("slope", name, i, grade, "degrees",
                              vx=.25, grade_deg=sign*grade))
    for sign, name in ((1, "step_up"), (-1, "step_down")):
        for i, height in enumerate((.005, .01, .02, .03, .04, .05, .065, .08), 1):
            cases.append(Case("step", name, i, height, "m",
                              vx=.25, step_m=sign*height))
    for sign, name in ((1, "kick_forward"), (-1, "kick_backward")):
        for i, kick in enumerate((.10, .25, .40, .60), 1):
            cases.append(Case("resilience", name, i, kick, "m/s",
                              birth_kick_mps=sign*kick))
    assert len(cases) == 46
    return tuple(cases)


def allocate(worlds: int = DEFAULT_WORLDS) -> list[Trial]:
    """Full dense 2048 grid; small smoke still covers every named case."""
    cases = registered_cases()
    if worlds < len(cases) or worlds > 4096:
        raise ValueError(f"worlds must be {len(cases)}..4096")
    weights = np.asarray([PER_CASE_WORLDS[c.specialty] for c in cases],
                         dtype=np.float64)
    if int(weights.sum()) != DEFAULT_WORLDS:
        raise AssertionError("canonical allocation no longer equals 2048")
    targets = weights * worlds / DEFAULT_WORLDS
    counts = np.maximum(1, np.floor(targets).astype(int))
    while counts.sum() > worlds:
        candidates = np.flatnonzero(counts > 1)
        if not len(candidates):
            raise ValueError("no capacity for all cases")
        i = min(candidates, key=lambda k: (
            targets[k]-counts[k], -int(k)))
        counts[i] -= 1
    for _ in range(int(worlds-counts.sum())):
        i = max(range(len(cases)), key=lambda k: (
            targets[k]-counts[k], -k))
        counts[i] += 1
    trials = [Trial(world, case, rep)
              for world, (case, rep) in enumerate(
                  (case, rep) for case, count in zip(cases, counts)
                  for rep in range(int(count)))]
    if len(trials) != worlds or any(
            sum(t.case == case for t in trials) < 1 for case in cases):
        raise AssertionError("missing/duplicated physical exam allocation")
    return trials


def _columns(trials: list[Trial], name: str) -> np.ndarray:
    return np.asarray([getattr(t.case, name) for t in trials],
                      dtype=np.float32)


def score(trials: list[Trial], physical: dict, horizon: int) -> list[dict]:
    """Pure host scoring over first-life scalars; no reward-derived success."""
    needed = (
        "actual_lip_m", "actual_grade_deg", "survived", "fall_frame",
        "first_life_frames", "both_feet_top_five",
        "traversed_upright_5frame", "traversed_5frame",
        "best_forward_progress_m", "speed_mae_mps",
        "lateral_mae_mps", "yaw_mae_rad_s", "mean_support",
        "mean_horizontal_speed_mps", "mean_abs_yaw_rate_rad_s",
        "bad_contact_fraction", "minimum_gravity_z",
        "mean_stance_slip_proxy", "positive_servo_shaft_work_j",
        "final2s_horizontal_speed_mps", "final2s_abs_yaw_rate_rad_s",
        "longest_stable_streak_frames", "initial_world_x_velocity_mps",
        "transition_initial_vx_mps", "transition_stop_speed_mps",
        "transition_final_vx_mps",
    )
    arrays = {}
    for field in needed:
        if field not in physical:
            raise ValueError(f"physical matrix missing {field}")
        arr = np.asarray(physical[field])
        if arr.shape != (len(trials),):
            raise ValueError(f"{field} is not a [world] compact scalar")
        if not np.isfinite(arr.astype(float)).all():
            raise FloatingPointError(f"nonfinite MJX evidence: {field}")
        arrays[field] = arr

    result = []
    for t in trials:
        c, i = t.case, t.world
        metric = lambda name: float(arrays[name][i])
        if abs(metric("actual_lip_m")-c.step_m) > .0005:
            raise ValueError(f"world {i}: physical step fixture mismatch")
        if abs(metric("actual_grade_deg")-c.grade_deg) > .30:
            raise ValueError(f"world {i}: physical grade fixture mismatch")
        if not 1 <= int(arrays["first_life_frames"][i]) <= horizon:
            raise ValueError(f"world {i}: invalid first-life frame count")
        if c.specialty == "resilience" and abs(
                metric("initial_world_x_velocity_mps")-c.birth_kick_mps) > .05:
            raise ValueError(f"world {i}: physical kick was not delivered")

        survived = bool(arrays["survived"][i])
        failures = [] if survived else ["first_life_ended"]
        if c.specialty in ("slope", "step"):
            # The same source-geometry top-foot proof used by the old
            # matrix: each foot independently supported five frames and
            # body upright for five frames 60cm beyond the real seam.
            if not bool(arrays["both_feet_top_five"][i]):
                failures.append("both_feet_not_supported_on_destination")
            if not bool(arrays["traversed_upright_5frame"][i]):
                failures.append("no_supported_physical_traversal")
        elif c.specialty == "resilience":
            if metric("final2s_horizontal_speed_mps") > RECOVERY_MAX_SPEED:
                failures.append("not_settled_linear")
            if metric("final2s_abs_yaw_rate_rad_s") > RECOVERY_MAX_YAW:
                failures.append("not_settled_yaw")
            if metric("longest_stable_streak_frames") < 50:
                failures.append("no_continuous_1s_recovery")
            if metric("minimum_gravity_z") < .75:
                failures.append("unsafe_tilt")
        elif c.specialty == "transition":
            sign = 1. if c.vx > 0 else -1.
            if sign*metric("transition_initial_vx_mps") < .125:
                failures.append("initial_direction")
            if metric("transition_stop_speed_mps") > .08:
                failures.append("never_stopped")
            if -sign*metric("transition_final_vx_mps") < .10:
                failures.append("never_reversed")
            if metric("yaw_mae_rad_s") > .20:
                failures.append("yaw_bias")
        elif c.kind == "zero_command_balance":
            if metric("mean_horizontal_speed_mps") > .10:
                failures.append("unwanted_translation")
            if metric("mean_abs_yaw_rate_rad_s") > .15:
                failures.append("unwanted_rotation")
        else:
            vx_limit = .08 if c.kind == "forward_heading" else .12
            yaw_limit = .12 if c.kind == "forward_heading" else .20
            if metric("speed_mae_mps") > vx_limit:
                failures.append("vx_tracking")
            if metric("lateral_mae_mps") > .10:
                failures.append("vy_tracking")
            if metric("yaw_mae_rad_s") > yaw_limit:
                failures.append("yaw_tracking")
        if metric("mean_support") < .50:
            failures.append("insufficient_foot_contact")
        if metric("bad_contact_fraction") > .10:
            failures.append("bad_contact")
        if c.specialty == "command" and c.kind == "forward" and c.level == 2:
            if not bool(arrays["traversed_upright_5frame"][i]):
                failures.append("flat_control_not_traversed")

        row = dict(world=i, specialty=c.specialty, kind=c.kind,
                   level=c.level, severity=c.severity,
                   unit=c.severity_unit, replicate=t.replicate,
                   command_vx=c.vx, command_vy=c.vy, command_yaw=c.yaw,
                   step_m=c.step_m, grade_deg=c.grade_deg,
                   initial_kick_mps=c.birth_kick_mps,
                   program=c.program, survived=survived,
                   first_life_frames=int(arrays["first_life_frames"][i]),
                   physical_mission_pass=not failures,
                   failure_reasons=",".join(failures))
        row.update({key: metric(key) for key in needed
                    if key not in ("survived", "first_life_frames")})
        result.append(row)
    return result


def summarize(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """Compact capability headline, exact difficulty curves and tested AUC."""
    groups = {}
    curves = {}
    for r in rows:
        groups.setdefault(r["specialty"], []).append(r)
        curves.setdefault((r["specialty"], r["kind"], r["level"]), []).append(r)
    def summary_of(name, group):
        n = len(group)
        passes = sum(r["physical_mission_pass"] for r in group)
        survived = sum(r["survived"] for r in group)
        lo, hi = wilson_95(passes, n)
        return dict(name=name, attempted=n, survived=survived,
                    passed=passes, pass_fraction=passes/n,
                    wilson95_lo=lo, wilson95_hi=hi)
    headline = [dict(specialty=k, **summary_of(k, v))
                for k, v in groups.items()]
    curve_rows = []
    for (specialty, kind, level), group in sorted(curves.items()):
        c = group[0]
        curve_rows.append(dict(specialty=specialty, kind=kind,
                               level=level, severity=c["severity"],
                               unit=c["unit"], **summary_of(kind, group)))
    envelopes = []
    for kind in sorted({(r["specialty"], r["kind"]) for r in curve_rows}):
        group = sorted((r for r in curve_rows
                        if (r["specialty"], r["kind"]) == kind),
                       key=lambda x: x["severity"])
        area = (float(np.trapezoid(
            [r["pass_fraction"] for r in group],
            [r["severity"] for r in group]))
            if len(group) >= 2 else None)
        envelopes.append(dict(
            specialty=kind[0], kind=kind[1],
            unit=group[0]["unit"], levels=len(group),
            range_min=group[0]["severity"], range_max=group[-1]["severity"],
            tested_range_auc=area))
    return headline, curve_rows, envelopes


def _csv(rows: list[dict]) -> bytes:
    if not rows:
        raise ValueError("empty specialist results")
    b = io.StringIO()
    w = csv.DictWriter(b, fieldnames=tuple(rows[0]))
    w.writeheader()
    w.writerows(rows)
    return b.getvalue().encode("utf-8")


def write_zip(output: Path, manifest: dict, rows: list[dict],
              headline: list[dict], curves: list[dict],
              envelopes: list[dict]) -> None:
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Will not overwrite a physical exam: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    md = ["# EMU compact physical pulse", "",
          "Fast 2048-world frozen physical evidence; not endurance certification.",
          "No composite ranking, optimizer update, or reset-life score.", "",
          "| Specialty | Passed | Survived | Trials |",
          "|---|---:|---:|---:|"]
    for r in headline:
        md.append(f"| {r['specialty']} | {r['passed']} | "
                  f"{r['survived']} | {r['attempted']} |")
    md.extend(("", "Each severity and direction is recorded separately.",
               "The station-keeping, long endurance, timed pushes and stairs "
               "are separate examinations.", ""))
    # One-row format allows quick concatenation across immutable
    # checkpoints without parsing 2048 worlds or six nested JSON groups.
    pulse = dict(actor_step=manifest["actor_step"],
                 seed=manifest["seed"], worlds=manifest["worlds"],
                 horizon_frames=manifest["horizon_frames"],
                 mjx_compute_wall_s=manifest["mjx_compute_wall_s"])
    for line in headline:
        key = line["specialty"]
        pulse[key+"_pass_fraction"] = line["pass_fraction"]
        pulse[key+"_survival_fraction"] = line["survived"]/line["attempted"]
    partial = output.with_name(output.name+".partial")
    try:
        with zipfile.ZipFile(partial, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("manifest.json", json.dumps(manifest, indent=2)+"\n")
            z.writestr("summary.json", json.dumps(dict(
                headlines=headline, curves=curves, envelopes=envelopes),
                indent=2)+"\n")
            z.writestr("summary.md", "\n".join(md))
            z.writestr("worlds.csv", _csv(rows))
            z.writestr("curves.csv", _csv(curves))
            z.writestr("pulse.csv", _csv([pulse]))
        partial.replace(output)
    finally:
        partial.unlink(missing_ok=True)


def run(*, policy: Path, output: Path, worlds: int = DEFAULT_WORLDS,
        horizon: int = DEFAULT_HORIZON, seed: int = 1701,
        device: str = "cuda") -> dict:
    if device not in ("cpu", "cuda") or not 300 <= horizon <= 1500 or seed < 0:
        raise ValueError("invalid pulse device/horizon/seed")
    trials = allocate(worlds)
    policy = Path(policy).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if output.suffix != ".zip" or output.exists():
        raise ValueError("Output must be a fresh .zip")
    if not policy.is_file() or policy.suffix not in (".emupolicy", ".pkl"):
        raise FileNotFoundError("Use a frozen exported actor or complete .pkl training snapshot")

    from emu.contracts import ROBOT_XML, OBS_SCHEMA_TERRAIN, sensor_observation_contract
    from emu.environment import EmuMJXEnv
    from emu.policy_export import (
        build_bundle, write_bundle, load_exported_actor_for_exam)
    from emu.ppo import make_trainer, configure_jax, require_backend
    from emu.physical_variants import selected_profile
    from emu.training_config import (
        PPO_LR, ROLLOUT_STEPS, PPO_ENTROPY_COST, PPO_VALUE_COST)
    # Materialize a snapshot-derived .emupolicy in a temporary file,
    # preserving the existing strict source/actor authentication loader.
    with tempfile.TemporaryDirectory(prefix="emu-pulse-") as temporary:
        frozen_path = policy
        if policy.suffix == ".pkl":
            frozen_path = Path(temporary) / "snapshot.emupolicy"
            write_bundle(build_bundle(policy), frozen_path)
        params, norm, actor = load_exported_actor_for_exam(frozen_path)
    if (actor["observation_schema"] != OBS_SCHEMA_TERRAIN or
            actor["training"].get("terrain_carpet_prisms") is not True or
            not actor["array_only"]):
        raise ValueError("Frozen pulse requires source-matched terrain360 true-prism H40")
    configure_jax(device)
    import jax
    import jax.numpy as jnp
    import jaxlib
    # Persistent XLA cache survives process restarts. The same 2048x600
    # compiled kernel can be reused for many source-matched checkpoints;
    # JAX keys cache entries by program, jaxlib and GPU configuration.
    cache_dir = Path(os.environ.get(
        "JAX_COMPILATION_CACHE_DIR", "~/.cache/emu/jax-pulse")).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache_dir.resolve()))
    require_backend(device == "cpu")
    train, control, sensor = (actor["training"], actor["control"],
                              actor["sensor_settings"])
    vi = control.get("velocity_i") or {}
    env = EmuMJXEnv(
        str(ROBOT_XML), num_envs=worlds, seed=seed,
        max_episode_steps=horizon+50, push_max_force_n=.7,
        observation_schema=OBS_SCHEMA_TERRAIN,
        terrain_step_max_m=.08, unified_capability_focus=True,
        terrain_carpet=True, terrain_carpet_prisms=True,
        solver_iterations=train.get("solver_iterations"),
        solver_ls_iterations=train.get("solver_ls_iterations"),
        academy_stream=True, academy_height_cap_m=.15,
        terrain_lesson="coherent-careful",
        attitude_profile=train["attitude_profile"],
        pitch_command_max_deg=float(train["pitch_command_max_deg"]),
        phase_speed_gain=float(control["phase_speed_gain"]),
        science_phase_lowcmd_scale=float(
            train.get("science_phase_lowcmd_scale", 1.)),
        velocity_i_ramp_s=float(vi.get("ramp_s", .4)),
        velocity_i_multiplier=float(vi.get("multiplier", 1.)),
        sensor_noise_scale=float(sensor.get("sensor_noise_scale", 0.)),
        exam_fixture_steps_m=_columns(trials, "step_m"),
        exam_fixture_grades_deg=_columns(trials, "grade_deg"),
        exam_fixture_kicks_mps=_columns(trials, "birth_kick_mps"),
        **{k: v for k, v in sensor.items() if k != "sensor_noise_scale"})
    expected = sensor_observation_contract(
        sensor, schema=OBS_SCHEMA_TERRAIN, array_only=True)
    if env.sensor_contract != expected:
        raise RuntimeError("frozen actor/physical sensor mismatch")
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
    commands = np.stack([_columns(trials, key)
                         for key in ("vx", "vy", "yaw")], axis=1)
    target = np.array([
        t.case.specialty in ("slope", "step") or (
            t.case.specialty == "command" and t.case.kind == "forward"
            and t.case.level == 2)
        for t in trials], dtype=bool)
    programs = np.asarray([t.case.program for t in trials], np.int32)
    print(f"[pulse] {worlds} MJX worlds / {horizon} frames / six specialties; "
          f"source actor step {actor['source']['step']:,}", flush=True)
    start = time.perf_counter()
    measured = jax.device_get(trainer["evaluate_matrix"](
        params, teacher, jnp.asarray(norm["mean"]),
        jnp.asarray(norm["std"]), jnp.asarray(commands),
        jnp.asarray(target), jax.random.PRNGKey(seed), horizon,
        jnp.asarray(programs)))
    wall = time.perf_counter()-start
    rows = score(trials, measured, horizon)
    headline, curves, envelopes = summarize(rows)
    root = Path(__file__).resolve().parents[1]
    owned = ("scripts/examine_h40_specialties.py", "emu/ppo.py",
             "emu/environment.py", "emu/contracts.py", "emu/robot.xml",
             "emu/carpet_prism_mjx.py", "emu/terrain_carpet_mjx.py",
             "emu/physical_variants.py", "emu/policy_export.py",
             "emu/domain_randomization.py", "emu/tof_geometry.py")
    manifest = dict(
        schema=SCHEMA, verdict="PROVISIONAL_COMPACT_SCREEN_NOT_CERTIFICATION",
        source_sha256={x: sha256((root/x).read_bytes()).hexdigest() for x in owned},
        actor_sha256=actor["source"]["actor_sha256"],
        artifact_sha256=actor["artifact_sha256"],
        actor_step=actor["source"]["step"], actor_source=actor["source"],
        robot_model_sha256=actor["robot_model_sha256"],
        sensor_sha256=env.sensor_contract["sha256"],
        physical_profile=selected_profile(),
        runtime=dict(python=platform.python_version(),
                     jax=jax.__version__, jaxlib=jaxlib.__version__,
                     mujoco=env.mujoco.mj_versionString()),
        worlds=worlds, assigned_worlds=len(rows), trials_per_case={
            f"{c.specialty}/{c.kind}/{c.level}":
            sum(t.case == c for t in trials) for c in registered_cases()},
        horizon_frames=horizon, horizon_s=horizon*CONTROL_DT,
        seed=seed, backend=jax.default_backend(),
        mjx_compute_wall_s=round(wall, 3),
        measured_world_steps_per_s=round(worlds*horizon/max(wall, .001), 1),
        true_physics="eight convex prisms on four 1m source tiles",
        training_updates=0, first_life_only=True, gpu_trace_export=False,
        persistent_jax_compilation_cache=True,
        recovery="initial signed velocity kick, final 2s quiet, 1s stable streak",
        transitions="same-world velocity commands change after 1/3 and 2/3 of horizon",
        caveats="No 60s endurance, station keeping, timed forces or sequential stairs")
    write_zip(output, manifest, rows, headline, curves, envelopes)
    for s in headline:
        print(f"[pulse] {s['specialty']:<11} "
              f"mission {s['passed']:>3}/{s['attempted']:<3} "
              f"survived {s['survived']:>3}/{s['attempted']:<3}")
    print(f"[pulse] {output} ({wall:.1f}s MJX call)", flush=True)
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy", type=Path,
                   default=Path("showcase/champion.emupolicy"),
                   help="Frozen .emupolicy or complete .pkl checkpoint snapshot")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--worlds", type=int, default=DEFAULT_WORLDS)
    p.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    p.add_argument("--seed", type=int, default=1701)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = p.parse_args(argv)
    run(**vars(args))


if __name__ == "__main__":
    main()
