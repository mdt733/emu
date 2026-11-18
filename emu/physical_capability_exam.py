"""EMU Physical Capability Exam v1: read-only, source-pinned flat-world qualification.

Uses PPO's existing *frozen* evaluate_trace path, never its optimizer. This
module deliberately does not certify terrain, viewer parity, or another source
profile. Historical matrix and prism exam results are not leaderboards here.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from hashlib import sha256
import io
import json
import math
from pathlib import Path
import time
import zipfile

import numpy as np

from .contracts import CONTROL_DT

SCHEMA = "emu_physical_capability_v1"
# Proposed gates. Do not label an actor a champion until both reference policies
# calibrate these thresholds and the MJX/browser parity experiment is complete.
GATES_REVISION = "provisional-1"
STAND_MAX_RADIUS_M = 0.20
STAND_MAX_HEADING_RAD = 0.35
STAND_P95_TILT_DEG = 20.0
MIN_FOOT_SUPPORT_FRACTION = 0.75
MAX_BAD_CONTACT_FRACTION = 0.05


@dataclass(frozen=True)
class Scenario:
    name: str
    seconds: int
    segments: tuple[tuple[int, float, float, float], ...]
    category: str

    def command_schedule(self, frames: int | None = None) -> np.ndarray:
        total = self.seconds * 50
        if frames is not None and frames != total:
            raise ValueError("Scenario duration must match its fixed protocol")
        schedule = np.concatenate([
            np.tile(np.array([vx, vy, yaw], dtype=np.float32),
                    (duration * 50, 1))
            for duration, vx, vy, yaw in self.segments
        ])
        if schedule.shape != (total, 3):
            raise ValueError(f"{self.name}: invalid segment durations")
        return schedule


SCENARIOS = (
    Scenario("stand_zero_60s", 60, ((60, 0., 0., 0.),), "balance"),
    Scenario("forward_025_30s", 30, ((30, .25, 0., 0.),), "locomotion"),
    Scenario("reverse_020_30s", 30, ((30, -.20, 0., 0.),), "locomotion"),
    Scenario("left_015_30s", 30, ((30, 0., .15, 0.),), "locomotion"),
    Scenario("right_015_30s", 30, ((30, 0., -.15, 0.),), "locomotion"),
    Scenario("yaw_left_050_30s", 30, ((30, 0., 0., .5),), "locomotion"),
    Scenario("yaw_right_050_30s", 30, ((30, 0., 0., -.5),), "locomotion"),
    Scenario("combined_30s", 30, ((30, .20, .10, .3),), "locomotion"),
    Scenario("forward_stop_reverse_30s", 30,
             ((10, .25, 0., 0.), (10, 0., 0., 0.),
              (10, -.20, 0., 0.)), "transitions"),
)
SCENARIOS_BY_NAME = {s.name: s for s in SCENARIOS}


def _safe_mean(a: np.ndarray) -> float:
    return float(np.mean(a)) if a.size else 0.0


def score_first_life(trace: dict, scenario: Scenario,
                     *, dt: float = CONTROL_DT) -> list[dict]:
    """Score actual post-physics states; first done ends observation forever.

    The source environment auto-resets internally for PPO. Everything after
    the first done is explicitly excluded; no second life can rescue a fall.
    Per-frame commands are ORIGINAL operator commands, not I-boosted demands.
    """
    done = np.asarray(trace["done"], dtype=bool)
    fell = np.asarray(trace["fell"], dtype=bool)
    if done.ndim != 2 or fell.shape != done.shape:
        raise ValueError("done/fell must both be [frames, worlds]")
    frames, worlds = done.shape
    if frames != scenario.seconds * 50:
        raise ValueError("unexpected exam horizon")
    if not np.isclose(dt, .02):
        raise ValueError("source control dt differs from v1 protocol")
    required = ("position_xy", "heading_rad",
                "physical_position_xy_after_step",
                "physical_heading_rad_after_step", "command",
                "terrain_gravity_z", "terrain_bad_contact",
                "floor_contact", "servo_shaft_drive_power_w")
    for name in required:
        if name not in trace or np.asarray(trace[name]).shape[0] != frames:
            raise ValueError(f"missing/misaligned physical trace {name}")
    output = []
    for world in range(worlds):
        terminal = np.flatnonzero(done[:, world])
        n = int(terminal[0] + 1) if len(terminal) else frames
        ended = bool(len(terminal))
        fell_first = bool(np.any(fell[:n, world]))
        positions = np.concatenate((
            np.asarray(trace["position_xy"][0, world:world+1], dtype=float),
            np.asarray(trace["physical_position_xy_after_step"][:n, world],
                       dtype=float)), axis=0)
        headings = np.unwrap(np.concatenate((
            np.asarray(trace["heading_rad"][0, world:world+1], dtype=float),
            np.asarray(trace["physical_heading_rad_after_step"][:n, world],
                       dtype=float))))
        command = np.asarray(trace["command"][:n, world], dtype=float)
        if command.shape != (n, 3):
            raise ValueError("malformed requested command history")
        dx = np.diff(positions, axis=0) / dt
        theta = headings[:-1]
        c, s = np.cos(theta), np.sin(theta)
        actual_vx = c * dx[:, 0] + s * dx[:, 1]
        actual_vy = -s * dx[:, 0] + c * dx[:, 1]
        actual_yaw = np.diff(headings) / dt

        # Integrate the OPERATOR's desired SE(2) path from the same birth
        # pose. Using the observed heading here would hide yaw drift.
        desired_heading = headings[0] + np.concatenate((
            np.zeros(1), np.cumsum(command[:, 2] * dt)))
        cd, sd = np.cos(desired_heading[:-1]), np.sin(desired_heading[:-1])
        desired_delta = dt * np.stack((
            cd * command[:, 0] - sd * command[:, 1],
            sd * command[:, 0] + cd * command[:, 1]), axis=1)
        target = positions[0] + np.concatenate((
            np.zeros((1, 2)), np.cumsum(desired_delta, axis=0)), axis=0)
        path_error = np.linalg.norm(positions - target, axis=1)
        displacement = np.linalg.norm(positions - positions[0], axis=1)
        grav = np.clip(np.asarray(
            trace["terrain_gravity_z"][:n, world], dtype=float), -1., 1.)
        tilt = np.degrees(np.arccos(grav))
        contacts = np.asarray(trace["floor_contact"][:n, world], dtype=bool)
        support = np.any(contacts.reshape((n, -1)), axis=1)
        bad = np.asarray(trace["terrain_bad_contact"][:n, world], dtype=float)
        drive = np.asarray(
            trace["servo_shaft_drive_power_w"][:n, world], dtype=float)
        drive_energy = float(np.sum(np.maximum(drive, 0.)) * dt)
        valid = all(np.isfinite(x).all() for x in (
            positions, headings, command, tilt, drive, actual_vx, actual_vy,
            actual_yaw, bad))
        reason = ("nonfinite" if not valid else
                  "fall" if fell_first else
                  "other_terminal" if ended else "horizon_survived")
        survived = bool(not ended and valid)
        # Both surviving and bounded are required. A moving robot is not a
        # standing robot simply because its head remains above the ground.
        balance_ok = (survived and
                      float(np.max(displacement)) <= STAND_MAX_RADIUS_M and
                      float(np.max(np.abs(headings - headings[0]))) <=
                      STAND_MAX_HEADING_RAD and
                      float(np.percentile(tilt, 95)) <= STAND_P95_TILT_DEG and
                      _safe_mean(support) >= MIN_FOOT_SUPPORT_FRACTION and
                      _safe_mean(bad > .5) <= MAX_BAD_CONTACT_FRACTION)
        row = dict(
            scenario=scenario.name, category=scenario.category, world=world,
            commanded_seconds=scenario.seconds,
            first_life_frames=n, time_to_first_fall_s=(
                round(float((np.flatnonzero(fell[:n, world])[0]+1)*dt), 3)
                if fell_first else None),
            observed_seconds=round(n * dt, 3),
            survived_full_horizon=survived, terminal_reason=reason,
            terminal_frame=(n if ended else None),
            max_radius_m=float(np.max(displacement)),
            final_drift_m=float(displacement[-1]),
            max_heading_drift_rad=float(np.max(np.abs(
                headings - headings[0]))),
            p95_tilt_deg=float(np.percentile(tilt, 95)),
            peak_tilt_deg=float(np.max(tilt)),
            contact_fraction=_safe_mean(support),
            bad_contact_fraction=_safe_mean(bad > .5),
            vx_mae_mps=_safe_mean(np.abs(actual_vx - command[:, 0])),
            vy_mae_mps=_safe_mean(np.abs(actual_vy - command[:, 1])),
            yaw_mae_rad_s=_safe_mean(np.abs(actual_yaw - command[:, 2])),
            achieved_vx_mps=_safe_mean(actual_vx),
            achieved_vy_mps=_safe_mean(actual_vy),
            achieved_yaw_rad_s=_safe_mean(actual_yaw),
            mean_trajectory_error_m=_safe_mean(path_error),
            final_trajectory_error_m=float(path_error[-1]),
            positive_motor_shaft_work_j=drive_energy,
            positive_motor_shaft_power_w=drive_energy / (n * dt),
            bounded_standing_pass=(
                balance_ok if scenario.category == "balance" else None),
        )
        if scenario.name == "forward_stop_reverse_30s":
            # A stop is not proved by a small *average* velocity across
            # forward and backward sections. Judge the actual stop interval.
            a, b = 10 * 50, min(n, 20 * 50)
            row["stop_interval_mean_speed_mps"] = (
                _safe_mean(np.hypot(actual_vx[a:b], actual_vy[a:b]))
                if b > a else None)
        output.append(row)
    return output


def summarize(rows: list[dict], *, expected_worlds: int) -> list[dict]:
    grouped = {}
    for row in rows:
        grouped.setdefault(row["scenario"], []).append(row)
    output = []
    for name, group in grouped.items():
        if len(group) != expected_worlds:
            raise ValueError(f"{name}: incomplete world count")
        survival = sum(r["survived_full_horizon"] for r in group)
        output.append(dict(
            scenario=name, category=group[0]["category"],
            trials=len(group), full_survivals=survival,
            full_survival_fraction=survival / len(group),
            bounded_standing_passes=(
                sum(bool(r["bounded_standing_pass"]) for r in group)
                if group[0]["category"] == "balance" else None),
            mean_first_life_s=_safe_mean(np.asarray(
                [r["observed_seconds"] for r in group])),
            mean_vx_mae_mps=_safe_mean(np.asarray(
                [r["vx_mae_mps"] for r in group])),
            mean_vy_mae_mps=_safe_mean(np.asarray(
                [r["vy_mae_mps"] for r in group])),
            mean_yaw_mae_rad_s=_safe_mean(np.asarray(
                [r["yaw_mae_rad_s"] for r in group])),
            mean_final_trajectory_error_m=_safe_mean(np.asarray(
                [r["final_trajectory_error_m"] for r in group])),
            mean_positive_motor_shaft_work_j=_safe_mean(np.asarray(
                [r["positive_motor_shaft_work_j"] for r in group])),
        ))
    return output


def _csv_bytes(rows: list[dict]) -> bytes:
    if not rows:
        raise ValueError("empty results")
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def save_evidence(path: Path, manifest: dict, rows: list[dict],
                  summary: list[dict]) -> None:
    """One atomic ZIP; no results produced until all scenarios succeed."""
    path = Path(path).expanduser().resolve()
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"manifest": manifest, "scenarios": summary}
    md = ["# EMU physical capability exam v1", "",
          "Frozen, first-life MJX evaluation. No training; no reset credit.",
          "All thresholds provisional until reference-robot calibration.", "",
          "| Scenario | Full survival | Standing gate | Mean vx MAE | Mean trajectory error |",
          "|---|---:|---:|---:|---:|"]
    for s in summary:
        gate = (str(s["bounded_standing_passes"]) + "/" + str(s["trials"])
                if s["bounded_standing_passes"] is not None else "—")
        md.append(f'| {s["scenario"]} | {s["full_survivals"]}/{s["trials"]} '
                  f'| {gate} | {s["mean_vx_mae_mps"]:.3f} m/s '
                  f'| {s["mean_final_trajectory_error_m"]:.3f} m |')
    md += ["", "Only scenarios actually present in this ZIP were measured.",
           "No composite champion ranking is assigned.", ""]
    partial = path.with_name(path.name + ".partial")
    try:
        with zipfile.ZipFile(partial, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("manifest.json", json.dumps(
                manifest, indent=2, sort_keys=True) + "\n")
            archive.writestr("summary.json", json.dumps(record, indent=2) + "\n")
            archive.writestr("summary.md", "\n".join(md))
            archive.writestr("worlds.csv", _csv_bytes(rows))
            archive.writestr("capabilities.csv", _csv_bytes(summary))
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


def run(*, policy: Path, output: Path, worlds: int = 8, seed: int = 1701,
        device: str = "cuda", selected: tuple[str, ...] | None = None) -> dict:
    """Frozen actor, source-authenticated model, true MJX first-life traces."""
    if not 2 <= worlds <= 32 or seed < 0 or device not in ("cpu", "cuda"):
        raise ValueError("v1 supports 2..32 worlds, a nonnegative seed, cpu/cuda")
    scenarios = tuple(SCENARIOS_BY_NAME[x] for x in selected) if selected else SCENARIOS
    if len({x.name for x in scenarios}) != len(scenarios):
        raise ValueError("duplicate scenario")
    policy = Path(policy).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if not policy.is_file() or policy.suffix != ".emupolicy":
        raise FileNotFoundError("Need a frozen .emupolicy")
    if output == policy or output in policy.parents or policy in output.parents:
        raise ValueError("Result archive must be independent of actor")
    if output.exists():
        raise FileExistsError("Results archive already exists")

    from .contracts import ROBOT_XML, OBS_SCHEMA_TERRAIN, sensor_observation_contract
    from .policy_export import load_exported_actor_for_exam
    from .ppo import make_trainer, configure_jax, require_backend
    from .environment import EmuMJXEnv
    from .training_config import (
        PPO_LR, ROLLOUT_STEPS, PPO_ENTROPY_COST, PPO_VALUE_COST)
    params, norm, actor = load_exported_actor_for_exam(policy)
    if (actor["observation_schema"] != OBS_SCHEMA_TERRAIN or
            actor["training"].get("terrain_carpet_prisms") is not True or
            not actor["array_only"]):
        raise ValueError("Frozen v1 requires matching true-prism H40 actor")
    configure_jax(device)
    import jax
    import jax.numpy as jnp
    require_backend(device == "cpu")
    train, control, sensor = (actor["training"], actor["control"],
                              actor["sensor_settings"])
    vi = control.get("velocity_i") or {}
    # A long test MUST NOT timeout at the original 700-frame training horizon.
    # Academy goal completion is disabled; only a fall or other genuine
    # terminal should end the robot's first life within the exam.
    max_frames = max(s.seconds * 50 for s in scenarios)
    env = EmuMJXEnv(
        str(ROBOT_XML), num_envs=worlds, seed=seed,
        max_episode_steps=max_frames + 50, push_max_force_n=.7,
        observation_schema=OBS_SCHEMA_TERRAIN,
        terrain_step_max_m=.08, unified_capability_focus=True,
        terrain_carpet=True, terrain_carpet_prisms=True,
        solver_iterations=train.get("solver_iterations"),
        solver_ls_iterations=train.get("solver_ls_iterations"),
        academy_stream=False, terrain_lesson="coherent-careful",
        attitude_profile=train["attitude_profile"],
        pitch_command_max_deg=float(train["pitch_command_max_deg"]),
        phase_speed_gain=float(control["phase_speed_gain"]),
        science_phase_lowcmd_scale=float(train.get(
            "science_phase_lowcmd_scale", 1.)),
        velocity_i_ramp_s=float(vi.get("ramp_s", .4)),
        velocity_i_multiplier=float(vi.get("multiplier", 1.)),
        sensor_noise_scale=float(sensor.get("sensor_noise_scale", 0.)),
        **{k: v for k, v in sensor.items() if k != "sensor_noise_scale"},
    )
    expected = sensor_observation_contract(
        sensor, schema=OBS_SCHEMA_TERRAIN, array_only=True)
    if env.sensor_contract != expected:
        raise RuntimeError("sensor contract mismatch")
    trainer = make_trainer(
        env, seed=seed, lr=PPO_LR, rollout=ROLLOUT_STEPS,
        epochs=1, minibatches=1,
        entropy_cost=PPO_ENTROPY_COST, value_cost=PPO_VALUE_COST)
    teacher = env.params_array(dict(
        specialist_mix=0., athletic_rehearsal_frac=0.,
        regime_frac_steady=1., regime_frac_transition=0.,
        regime_frac_resilience=0., probe_push_n=0.,
        probe_tilt_rad=0., probe_linvel_mps=0.,
        probe_angvel_rad_s=0., probe_joint_rad=0.,
        terrain_exposure=0., slope_exposure=0.,
        academy_episode_probability=0.))
    rows = []
    started = time.perf_counter()
    for idx, scenario in enumerate(scenarios):
        print(f"[physical-v1] {scenario.name}: {scenario.seconds}s × "
              f"{worlds} worlds, seed={seed+idx}", flush=True)
        trace = jax.device_get(trainer["evaluate_trace"](
            params, teacher, jnp.asarray(norm["mean"]),
            jnp.asarray(norm["std"]), jax.random.PRNGKey(seed+idx),
            scenario.seconds * 50, 0.,
            command_schedule=jnp.asarray(scenario.command_schedule())))
        rows.extend(score_first_life(trace, scenario))
        del trace
    summary = summarize(rows, expected_worlds=worlds)
    import inspect
    # Hash *all* core scoring and physical inference sources, as well as
    # the verified actor/model identity. An altered examiner cannot silently
    # share a comparison key with an earlier build.
    source_paths = ("emu/physical_capability_exam.py", "emu/ppo.py",
                    "emu/environment.py", "emu/contracts.py",
                    "emu/terrain_carpet_mjx.py", "emu/carpet_prism_mjx.py",
                    "emu/physical_variants.py")
    root = Path(__file__).resolve().parents[1]
    source_hashes = {p: sha256((root / p).read_bytes()).hexdigest()
                     for p in source_paths}
    manifest = dict(
        schema=SCHEMA, gates_revision=GATES_REVISION,
        status="UNCALIBRATED_NOT_A_CHAMPION_RANKING",
        actor_artifact_sha256=actor["artifact_sha256"],
        actor_weights_sha256=actor["source"]["actor_sha256"],
        actor_training_source=actor["source"],
        robot_model_sha256=actor["robot_model_sha256"],
        physical_profile="h40",
        sensor_contract_sha256=env.sensor_contract["sha256"],
        source_files_sha256=source_hashes,
        compiled_solver_iterations=int(env.mj_model.opt.iterations),
        compiled_solver_ls_iterations=int(env.mj_model.opt.ls_iterations),
        simulator="MuJoCo MJX, actual 8 convex prism meshes / 4 source tiles",
        benchmark_terrain="level floor, zero step and slope exposure",
        control_dt_s=CONTROL_DT,
        worlds=worlds, seed=seed, device=jax.default_backend(),
        scenarios=[x.name for x in scenarios],
        scenario_seed_rule="base seed + canonical scenario index",
        duration_s_by_scenario={x.name: x.seconds for x in scenarios},
        no_optimizer_updates=True, first_life_only=True,
        terminal_resets_discarded=True,
        exam_timeout_frames=max_frames+50,
        motor_work="positive servo shaft drive work, not battery energy",
        wall_seconds=round(time.perf_counter()-started, 3),
        gate_values=dict(
            stand_radius_m=STAND_MAX_RADIUS_M,
            heading_rad=STAND_MAX_HEADING_RAD,
            p95_tilt_deg=STAND_P95_TILT_DEG,
            contact_fraction=MIN_FOOT_SUPPORT_FRACTION,
            bad_contact_fraction=MAX_BAD_CONTACT_FRACTION))
    save_evidence(output, manifest, rows, summary)
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path,
                        default=Path("showcase/champion.emupolicy"))
    parser.add_argument("--output", type=Path, required=True,
                        help="New ZIP path; never overwritten")
    parser.add_argument("--worlds", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--scenario", action="append",
                        choices=tuple(SCENARIOS_BY_NAME),
                        help="May be supplied repeatedly; defaults to all")
    args = parser.parse_args(argv)
    result = run(policy=args.policy, output=args.output, worlds=args.worlds,
                 seed=args.seed, device=args.device,
                 selected=tuple(args.scenario) if args.scenario else None)
    print(f"[physical-v1] wrote {args.output}; {result['status']}", flush=True)


if __name__ == "__main__":
    main()
