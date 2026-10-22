"""Winner-refork campaign: one 312-input lineage, five controlled challenges.

General gait, turning, recovery, efficiency and a real height-growing obstacle
specialist compete at 40/60/80M (and optional 100M). One proven winner becomes
the sole parent for all five next-round policies. Frozen exams compare EVERY
policy on the same general physics and same progressively taller real steps.
Never modifies historical campaign5020 or the source in main.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
from dataclasses import asdict, replace
from pathlib import Path

from .experiment_io import read_json as _read, save_json as _save, sha256_file as _sha
from .checkpoint_io import (checkpoint_step, git_provenance,
                            resolve_target_checkpoint,
                            run_training, update_boundary_at_or_after)
from .contracts import OBS_SCHEMA_PREVIEW, source_repo_root
from .scoring import graded_step_score
from .training_config import (TrainingConfig, foundation_config,
                              training_sensor_contract, verify_existing_phase_config)

VERSION = "v5020-winner-refork-4"
STAGES = ("walk40", "walk60", "walk80", "terrain100")
TARGET = {"walk40": 40_000_000, "walk60": 60_000_000,
          "walk80": 80_000_000, "terrain100": 100_000_000}
PROBE_STEPS = 2_000_000
WALK_ARMS = ("general", "turns", "recovery", "efficiency", "steps")
# Identical fixed evaluation heights for every policy, even those which
# only trained on flat surfaces. 150mm is the existing MJX box's geometry
# ceiling; the earned training frontier is NOT capped at any of these steps.
EXAM_HEIGHTS_MM = (5, 10, 20, 40, 80, 150)
EXAM_WEIGHTS = (.10, .15, .25, .25, .15, .10)
TERRAIN_SAFETY_M = .150
# Deliberately milder than the original v5020 mature jump, which failed
# catastrophically at 21.36M. These are envelopes, not target achievements.
WALK_LEVELS = {
    "walk40": dict(vx_hi_max=.40, vx_lo_min=-.12, vy_max_mps=.14,
                   yaw_max_rad_s=.90, phase_speed_gain=.75, push_max_force_n=.70,
                   combo_mix_frac=0., domain_terrain_radius=.16,
                   difficulty_base_step=.0014, pitch_command_max_deg=8.),
    "walk60": dict(vx_hi_max=.45, vx_lo_min=-.15, vy_max_mps=.17,
                   yaw_max_rad_s=1.10, phase_speed_gain=.90, push_max_force_n=1.0,
                   combo_mix_frac=.20, domain_terrain_radius=.18,
                   difficulty_base_step=.0016, pitch_command_max_deg=10.),
    "walk80": dict(vx_hi_max=.50, vx_lo_min=-.18, vy_max_mps=.20,
                   yaw_max_rad_s=1.30, phase_speed_gain=1.0, push_max_force_n=1.5,
                   combo_mix_frac=.40, domain_terrain_radius=.20,
                   difficulty_base_step=.0018, pitch_command_max_deg=12.),
}
# General-locomotion gates increase with maturity. They are preliminary,
# explicitly conservative safeguards, not certified physical performance.
FLOORS = {
    "walk40": (.60, 0., 0.),
    "walk60": (.70, .03, .06),
    "walk80": (.80, .05, .10),
    "terrain100": (.80, .05, .10),
}


def arms(stage: str) -> tuple[str, ...]:
    if stage not in STAGES:
        raise ValueError(f"unknown generation {stage}")
    return WALK_ARMS


def plan(envs: int = 2048) -> dict:
    return {
        "version": VERSION, "observation_schema": OBS_SCHEMA_PREVIEW,
        "proposed_envs": envs, "source": "existing schema-9 foundation (explicit checkpoint)",
        "independent_step_exam_mm": list(EXAM_HEIGHTS_MM),
        "earned_step_height_ceiling_mm": int(TERRAIN_SAFETY_M*1000),
        "generations": [
            {"stage": s, "target_steps_per_actor": TARGET[s],
             "arms": list(arms(s)), "probe_steps_per_arm": PROBE_STEPS,
             "winner_count": 1, "physical_step_training": "steps" in arms(s)}
            for s in STAGES],
        "safety": ("probe each new curriculum for 2M, reject collapse/absurd spring energy, "
                   "screen ALL contenders at 5/10/20/40/80/150mm, including flat controls; "
                   "only promote a qualified general walker, regardless of step score"),
        "code_branch": "feature/v5020-coldstart-terrain",
    }


def config(run: Path, *, stage: str, arm: str, target: int,
           seed: int, envs: int, device: str) -> TrainingConfig:
    if stage not in STAGES or arm not in arms(stage):
        raise ValueError(f"unsupported refork stage/arm {stage}/{arm}")
    base = foundation_config(run_dir=run, seed=seed, device=device, envs=envs)
    profile = WALK_LEVELS["walk80" if stage == "terrain100" else stage]
    cfg = replace(base, **profile, run_dir=str(run), steps=int(target),
                  observation_schema=OBS_SCHEMA_PREVIEW,
                  attitude_profile="v5015", forward_preview_masked=False,
                  domain_randomization=True, domain_terrain_mode="contact",
                  domain_hardware_radius=0.,
                  freeze_obs_norm_after_steps=0, allow_rollback=True,
                  open_frontier=False, capability_field_mode="shadow",
                  field_max_influence=0.,
                  field_influence_start_steps=40_000_000,
                  field_influence_full_steps=80_000_000,
                  terrain_height_safety_m=0.,
                  terrain_adaptive_flat_control=False,
                  terrain_random_placement=False,
                  terrain_progress_weight=0.)
    if stage in ("walk60", "walk80", "terrain100"):
        cfg = replace(cfg, capability_field_mode="graph",
                      field_max_influence=.03 if stage == "walk60" else .06)
    if arm == "turns":
        cfg = replace(cfg, combo_mix_frac={
            "walk40": .15, "walk60": .35,
            "walk80": .55, "terrain100": .55}[stage])
    elif arm == "recovery":
        cfg = replace(cfg, push_max_force_n={
            "walk40": .90, "walk60": 1.20,
            "walk80": 1.80, "terrain100": 1.80}[stage])
    elif arm == "efficiency":
        cfg = replace(cfg, motor_guidance_weight={
            "walk40": .012, "walk60": .020,
            "walk80": .030, "terrain100": .030}[stage])
    elif arm == "steps":
        # Physical step height is earned continuously from REAL edge
        # crossings with fall-rate retreat and evidence-count gating;
        # 150mm is a safety ceiling, not a target or a curriculum stage.
        # Flat & non-forward-command births are retained for general gait.
        # Keep gentle steps on level ground first, decoupled from slopes.
        cfg = replace(cfg,domain_terrain_radius=0.,
                      terrain_random_placement=True,
                      terrain_height_initial_m=.003,
                      terrain_height_safety_m=TERRAIN_SAFETY_M,
                      terrain_height_exposure=.60,
                      terrain_height_edge_fraction=.35,
                      terrain_height_growth_m=.001,
                      terrain_progress_weight=.035)
    return cfg


def _validated_step(path: Path, cfg: TrainingConfig) -> int:
    return checkpoint_step(path, expected_observation_schema=OBS_SCHEMA_PREVIEW,
                           expected_sensor_contract=training_sensor_contract(cfg))


def _foundation(path: Path, *, envs: int) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"missing frozen schema-9 foundation: {path}")
    source = TrainingConfig.load(path.parent.parent / "training_config.json")
    if source.envs != envs or source.observation_schema != OBS_SCHEMA_PREVIEW:
        raise ValueError("foundation env-count or 312-input sensor schema differs")
    step = _validated_step(path, source)
    if not 12_000_000 <= step <= 22_000_000:
        raise ValueError(f"expected a saved 12–22M foundation snapshot, got {step:,}")
    return {"path": str(path.resolve()), "sha256": _sha(path), "step": step}


def _context(root: Path, *, foundation: dict, seed: int, envs: int,
             screen_envs: int, horizon: int, provenance: dict) -> None:
    repo = source_repo_root()
    if root == Path.home().resolve() or (repo and (root == repo or repo in root.parents)):
        raise ValueError("campaign root must be outside Git and outside the home root")
    if not provenance.get("available") or provenance.get("dirty") or not provenance.get("commit"):
        raise RuntimeError("refork requires a clean, identifiable Git revision")
    if provenance.get("branch") != "feature/v5020-coldstart-terrain":
        raise RuntimeError("refork work must run on feature/v5020-coldstart-terrain")
    expected = {"version": VERSION, "foundation": foundation, "seed": seed,
                "envs": envs, "screen_envs": screen_envs, "horizon": horizon,
                "source_commit": provenance["commit"]}
    location = root / "refork_context.json"
    if location.exists():
        if _read(location) != expected:
            raise RuntimeError("campaign identity changed; keep the original options/ref or use a NEW root")
    else:
        extras = {p.name for p in root.iterdir()} - {".campaign.lock"}
        if extras:
            raise RuntimeError(f"refusing unrecognised output directory: {sorted(extras)}")
        _save(location, expected)


def _phase(run: Path, cfg: TrainingConfig, *, parent: Path,
           provenance: dict) -> Path:
    contract = training_sensor_contract(cfg)
    parent_step = _validated_step(parent, cfg)
    if parent_step >= cfg.steps:
        raise ValueError(f"{run.name} target {cfg.steps:,} <= parent {parent_step:,}")
    spec = {"version": VERSION, "parent": {"path": str(parent.resolve()),
            "sha256": _sha(parent), "step": parent_step},
            "config": {k: v for k, v in asdict(cfg).items()
                       if k not in ("device", "resume")}}
    spec_file = run / "stage_spec.json"
    if spec_file.exists():
        if _read(spec_file) != spec:
            raise RuntimeError(f"changed phase parent/config: {run}")
    else:
        if run.exists() and any(run.iterdir()):
            raise RuntimeError(f"phase directory not empty: {run}")
        _save(spec_file, spec)
    verify_existing_phase_config(run, cfg)
    meta = run / "run_meta.json"
    if meta.exists() and (_read(meta).get("source_provenance") or {}).get("git_commit") != provenance["commit"]:
        raise RuntimeError(f"training source changed in {run}")
    complete, partial, boundary = resolve_target_checkpoint(
        run, target_steps=cfg.steps, envs=cfg.envs, label=str(run),
        expected_observation_schema=OBS_SCHEMA_PREVIEW,
        expected_sensor_contract=contract)
    if complete is None:
        resume = "auto" if partial is not None else str(parent)
        print(f"[refork] training {run}: {_validated_step(partial or parent,cfg):,} -> {boundary:,}", flush=True)
        run_training(replace(cfg, resume=resume), provenance=provenance)
        complete, partial, _ = resolve_target_checkpoint(
            run, target_steps=cfg.steps, envs=cfg.envs, label=str(run),
            expected_observation_schema=OBS_SCHEMA_PREVIEW,
            expected_sensor_contract=contract)
        if complete is None:
            raise RuntimeError(f"phase stopped without validated final: {run}")
    if _validated_step(complete, cfg) != boundary:
        raise RuntimeError("final checkpoint not at exact PPO boundary")
    return complete


def probe_health(probe_run: Path) -> dict:
    """Audit a whole short continuation, then judge its latest trajectory.

    A recovering learner is not rejected for one early fall-rate spike.
    Extreme finite physics is different: ANY logged physically absurd
    energy/power sample invalidates the probe even if its last rows look
    healthy. The recent window still vetoes persistent collapse.
    These are research screening thresholds, not hardware certifications.
    """
    with (probe_run / "metrics.csv").open(newline="", encoding="utf-8") as fd:
        rows = list(csv.DictReader(fd))
    if not rows:
        raise RuntimeError(f"no actual rollout metrics for probe {probe_run}")

    required = ("fall_rate", "ep_len_mean", "spring_potential_energy_j")
    # Current telemetry includes these; older source-pinned research traces
    # may omit them. Do not manufacture a zero for missing optional data.
    optional = ("servo_shaft_abs_power_w", "servo_over_rated_frac",
                "spring_potential_energy_peak_j", "servo_shaft_abs_power_peak_w",
                "nan_grad_skips_total")
    def samples(key: str, *, mandatory: bool = True) -> list[float] | None:
        if not mandatory and key not in rows[0]:
            return None
        data = []
        for n, row in enumerate(rows):
            try:
                value = float(row[key])
            except (KeyError, ValueError, TypeError) as exc:
                raise ValueError(f"missing/non-numeric {key} in probe row {n}") from exc
            if not math.isfinite(value):
                raise ValueError(f"nonfinite {key} in probe row {n}")
            data.append(value)
        return data

    readings = {k: samples(k) for k in required}
    readings.update({k: samples(k, mandatory=False) for k in optional})
    falls = readings["fall_rate"]
    lengths = readings["ep_len_mean"]
    spring = readings["spring_potential_energy_j"]
    if any(not 0. <= v <= 1. for v in falls):
        raise ValueError("fall_rate outside physical [0,1] range")
    if any(v < 0. for v in lengths + spring):
        raise ValueError("negative physical episode length or spring energy")
    for key in ("servo_shaft_abs_power_w", "spring_potential_energy_peak_j",
                "servo_shaft_abs_power_peak_w"):
        if readings[key] is not None and any(v < 0. for v in readings[key]):
            raise ValueError(f"negative physical measurement {key}")
    if readings["servo_over_rated_frac"] is not None and any(
            not 0. <= v <= 1. for v in readings["servo_over_rated_frac"]):
        raise ValueError("servo overload fraction outside [0,1]")

    # If completed-episode counts exist, distinguish a sample with no
    # completed episodes from a measured healthy zero-fall cohort.
    episodes = samples("episodes", mandatory=False)
    if episodes is not None and any(x < 0. for x in episodes):
        raise ValueError("negative completed episode count")
    if episodes is not None and sum(episodes) < 8.:
        raise ValueError("insufficient completed episodes to qualify a probe")

    # One contiguous recent window evaluates the final learned policy.
    # Prior severe windows are retained as *warnings* if subsequently
    # recovered, never silently erased by truncation to the last 3 rows.
    window = max(3, min(12, math.ceil(len(rows) * .20)))
    recent_fall = falls[-window:]
    recent_len = lengths[-window:]
    def avg(values: list[float]) -> float:
        return sum(values) / len(values)

    # Pool by completed episodes rather than averaging ratios from tiny
    # and huge cohorts as if they carried equal evidence. Old historical
    # traces without episode counts retain their original row-mean method.
    recent_episodes = None if episodes is None else sum(episodes[-window:])
    if recent_episodes is not None:
        if recent_episodes < 8.:
            raise ValueError("insufficient recent completed episodes to qualify a probe")
        fall = sum(v*n for v,n in zip(recent_fall,episodes[-window:])) / recent_episodes
        length = sum(v*n for v,n in zip(recent_len,episodes[-window:])) / recent_episodes
    else:
        fall = avg(recent_fall)
        length = avg(recent_len)
    spring_peak = readings["spring_potential_energy_peak_j"]
    max_spring = max(max(spring), max(spring_peak) if spring_peak else 0.)
    severe_windows = sum(
        avg(falls[i:i + 3]) > .92 and avg(lengths[i:i + 3]) < 70.
        for i in range(max(0, len(rows) - 2))
    )
    reasons = []
    warnings = []
    if fall > .92:
        reasons.append(f"probe fall rate {fall:.1%} >92%")
    if length < 70.:
        reasons.append(f"probe mean episode length {length:.1f} <70 frames")
    if max_spring > 50.:
        reasons.append(f"unphysical mean spring potential {max_spring:.1f}J >50J")
    # Deliberately permissive failsafe for an aggregate six-actuator
    # ABSOLUTE mechanical-power diagnostic, not a battery-power estimate.
    # Typical healthy research screens are single-digit watts; 1000W is
    # orders of magnitude beyond a plausible reading for this robot.
    power_candidates = []
    if readings["servo_shaft_abs_power_w"] is not None:
        power_candidates += readings["servo_shaft_abs_power_w"]
    if readings["servo_shaft_abs_power_peak_w"] is not None:
        power_candidates += readings["servo_shaft_abs_power_peak_w"]
    max_power = max(power_candidates) if power_candidates else None
    if max_power is not None and max_power > 1000.:
        reasons.append(f"unphysical mean shaft absolute power {max_power:.1f}W >1000W")
    if severe_windows and not reasons:
        warnings.append(f"{severe_windows} earlier severely collapsed 3-row windows; later recovery observed")
    skips = readings["nan_grad_skips_total"]
    if skips is not None and max(skips) > 0.:
        warnings.append(f"PPO has reported skipped nonfinite gradients (max {max(skips):g}); inspect the trace")

    return {"eligible": not reasons, "reasons": reasons,
            "fall_rate": round(fall, 5), "mean_episode_frames": round(length, 2),
            "max_logged_spring_j": round(max_spring, 3),
            "max_logged_shaft_abs_power_w": None if max_power is None else round(max_power, 3),
            "rows_checked": len(rows), "recent_window_rows": min(len(rows),window),
            "worst_3row_fall_rate": round(max(
                (avg(falls[i:i+3]) for i in range(max(1, len(rows)-2))), default=0.), 5),
            "severe_collapse_windows": int(severe_windows),
            "total_completed_episodes": None if episodes is None else int(sum(episodes)),
            "recent_completed_episodes": None if recent_episodes is None else int(recent_episodes),
            "warnings": warnings}


def _score(evidence: dict, *, stage: str, name: str) -> dict:
    def val(key: str) -> float:
        x = float(evidence[key])
        if not math.isfinite(x):
            raise ValueError(f"nonfinite {key}")
        return x
    try:
        core = val("screen_core_survival")
        hard = (val("screen_resilience100_survival") +
                val("screen_robust70_survival")) / 2
        growth = (val("screen_growth140_survival") +
                  val("screen_growthcombo140_survival")) / 2
        slope = (val("screen_contact_slope_survival") +
                 val("screen_contact_harder_survival")) / 2
        gait = (val("screen_robust70_step_evidence") +
                val("screen_contact_slope_step_evidence")) / 2
        motion = sum(val(f"screen_{p}_{axis}_mae")
                     for p in ("robust70", "growth140", "contact_slope")
                     for axis in ("vx", "vy", "yaw")) / 9
        height = sum(val(f"screen_{p}_height_shortfall_m")
                     for p in ("core", "robust70", "contact_slope")) / 3
        power = (val("power_robust_shaft_abs_power_w") +
                 val("power_contact_slope_shaft_abs_power_w")) / 2
        overload = (val("power_robust_over_rated_frac") +
                    val("power_contact_slope_over_rated_frac")) / 2
        # Shared measurement doctrine rejects upper-frontier mixes,
        # insufficient physical attempts and impossible passage>contact.
        step_score,step_trials=graded_step_score(
            evidence, EXAM_HEIGHTS_MM, EXAM_WEIGHTS)
        crossings=[t.crossing_rate for t in step_trials]
        contact=[t.contact_rate for t in step_trials]
    except (KeyError, ValueError, TypeError) as exc:
        return {"arm": name, "eligible": False, "score": None,
                "reasons": [f"invalid held-out evidence: {exc}"]}
    core_min, hard_min, slope_min = FLOORS[stage]
    reasons = []
    if core < core_min:
        reasons.append(f"core survival {core:.2f} < {core_min:.2f}")
    if hard < hard_min:
        reasons.append(f"recovery survival {hard:.2f} < {hard_min:.2f}")
    if slope < slope_min:
        reasons.append(f"contact-slope survival {slope:.2f} < {slope_min:.2f}")
    if gait < .10:
        reasons.append("insufficient stepping evidence")
    if height > .055:
        reasons.append("chassis height shortfall >55mm")
    # Every arm can earn course points, including the existing control.
    # True crossing influences selection but cannot bypass survival gates.
    # Graded heights add discrimination as the earnable frontier grows.
    rank = (.28 * core + .16 * hard + .16 * slope + .12 * growth +
            .10 * gait + .25 * step_score -
            .035 * min(1., motion / .65) -
            .025 * min(1., height / .055) -
            .025 * min(2., power / 15.) -
            .025 * min(1., overload))
    return {"arm": name, "eligible": not reasons,
            "score": round(rank, 6), "reasons": reasons,
            "core": round(core, 5), "hard": round(hard, 5),
            "slope": round(slope, 5), "growth": round(growth, 5),
            "gait": round(gait, 5), "motion_mae": round(motion, 5),
            "height_shortfall_m": round(height, 5),
            "shaft_abs_power_w": round(power, 5),
            "overrated_frac": round(overload, 5),
            "graded_step_score": round(step_score, 5),
            "step_attempts": {str(t.height_mm): t.attempts for t in step_trials},
            "step_crossings": {str(mm): round(x, 5)
                               for mm, x in zip(EXAM_HEIGHTS_MM, crossings)},
            "step_contacts": {str(mm): round(x, 5)
                              for mm, x in zip(EXAM_HEIGHTS_MM, contact)}}


def _exam_many(checkpoints: list[Path], *, stage: str, root: Path,
               screen_envs: int, horizon: int, device: str) -> dict[str, dict]:
    """One frozen MJX examiner build, many source-pinned policies.

    Every parent/challenger receives identical seeds, world geometry,
    sensor contract and physical-height tests. Constructing fresh MJX
    worlds per policy needlessly re-JITs large functions on the 8GB GPU;
    one shared batch does the exact same examinations more efficiently.
    """
    from .evaluation import evaluate_runs
    if not checkpoints:
        return {}
    sources=[Path(p).expanduser().resolve() for p in checkpoints]
    kwargs={"envs":screen_envs,"horizon":horizon,"device":device,
            "eval_pitch_deg":8.,"key_by_path":True,
            "write_screen_files":False,
            "v5020_physical_exam":True,
            "physical_step_heights_mm":EXAM_HEIGHTS_MM,
            "eval_phase_speed_gain":1.0,"course_root":root}
    return evaluate_runs(sources,**kwargs)


def _exam(ck: Path, *, stage: str, root: Path, screen_envs: int,
          horizon: int, device: str) -> dict:
    """Single-policy compatibility wrapper for manual examination."""
    source=Path(ck).expanduser().resolve()
    return _exam_many([source],stage=stage,root=root,screen_envs=screen_envs,
                      horizon=horizon,device=device)[str(source)]


def _scoreboard(root: Path, stage: str, candidates: dict, *,
                parent: Path, screen_envs: int, horizon: int,
                device: str) -> dict:
    target = TARGET[stage]
    location = root / stage / "scoreboard.json"
    if location.exists():
        board = _read(location)
        expected = {key: value["sha256"] if value else None
                    for key, value in candidates.items()}
        if (board.get("version") != VERSION or board.get("target") != target
            or board.get("parent_sha256") != _sha(parent)
            or board.get("candidates") != expected):
            raise RuntimeError("scoreboard provenance changed; do not recompute or mix data")
        return board
    # Frozen exam construction/compilation happens ONCE per generation.
    # It is read-only: neither PPO nor the teacher ever sees these scores.
    paths=[Path(parent).resolve()] + [
        Path(candidates[name]["path"]).resolve()
        for name in arms(stage) if candidates[name] is not None]
    examined=_exam_many(paths,stage=stage,root=root,screen_envs=screen_envs,
                        horizon=horizon,device=device)
    parent_rating=_score(examined[str(Path(parent).resolve())],stage=stage,name="incumbent")
    results=[]
    for name in arms(stage):
        saved=candidates[name]
        if saved is None:
            results.append({"arm":name,"eligible":False,"score":None,
                            "reasons":["probe rejected"]})
            continue
        evidence=examined[str(Path(saved["path"]).resolve())]
        score=_score(evidence,stage=stage,name=name)
        score["checkpoint_sha256"]=saved["sha256"]
        results.append(score)
    eligible = sorted((r for r in results if r["eligible"]),
                      key=lambda r: (-r["score"], r["arm"]))
    # Do not mistake strict, maturity-dependent CERTIFICATION floors for a
    # physical safety limit. A coherent, finite, improving walker can
    # earn another bounded generation while still failing the certification
    # needed for public champion publication. Unhealthy 2M probes never
    # arrive here, and corrupt/incomplete exam evidence has no numeric rank.
    provisional = sorted((
        r for r in results
        if not r["eligible"] and r.get("score") is not None
        and r["core"] >= .20 and r["gait"] >= .04
        and r["hard"] >= 0. and r["slope"] >= 0.
        and r["height_shortfall_m"] <= .09
    ), key=lambda r: (-r["score"], r["arm"]))
    if eligible:
        best = eligible[0]
        if parent_rating["eligible"] and (
            best["score"] < parent_rating["score"] - .005
            or best["core"] < parent_rating["core"] - .10
        ):
            winner = "incumbent"
            qualification = "incumbent-qualified"
            reason = "Qualified incumbent retained over weaker challengers; keep learning next generation"
        else:
            winner = best["arm"]
            qualification = "certified"
            reason = "Best qualifying frozen policy"
    elif provisional:
        best = provisional[0]
        # This permits *research continuation*, NEVER a public promotion.
        # If an already-qualified incumbent exists, don't replace it with
        # an unqualified descendant just to extend the run.
        if parent_rating["eligible"]:
            winner = "incumbent"
            qualification = "incumbent-qualified"
            reason = "Qualified incumbent retained; no challenger qualified, but training may continue"
        else:
            winner = best["arm"]
            qualification = "provisional"
            reason = ("Provisional research continuation: coherent held-out gait "
                      "but strict certification floors not yet satisfied")
    else:
        winner = None
        qualification = "none"
        reason = "No viable contender with valid frozen evidence; preserve incumbent and STOP"
    board = {"version": VERSION, "stage": stage, "target": target,
             "parent_sha256": _sha(parent),
             "candidates": {k: v["sha256"] if v else None for k, v in candidates.items()},
             "incumbent": parent_rating, "rows": results,
             "winner": winner, "qualification": qualification,
             "reason": reason}
    _save(location, board)
    report = root / stage / "scoreboard.md"
    report.write_text(
        f"# {stage} — winner-refork held-out screening\n\n"
        f"Parent SHA-256: {_sha(parent)}\n\n"
        f"Next research parent: {winner or 'NONE'}\n\n"
        f"Qualification: {qualification.upper()} (provisional is NOT a champion)\n\n"
        f"Verdict: {reason}\n\n"
        "| Arm | Eligible | Score | Core | Recovery | Contact slopes | Graded step score | Crossings by mm | Reasons |\n"
        "|---|---|---:|---:|---:|---:|---:|---|---|\n" +
        "".join(f"| {r['arm']} | {r['eligible']} | {r['score']} | "
                f"{r.get('core','—')} | {r.get('hard','—')} | "
                f"{r.get('slope','—')} | {r.get('graded_step_score','—')} | "
                f"{r.get('step_crossings','—')} | "
                f"{', '.join(r['reasons'])} |\n" for r in results),
        encoding="utf-8")
    return board


def _run_candidate(root: Path, *, stage: str, arm: str, parent: Path,
                   seed: int, envs: int, device: str, provenance: dict):
    out = root / stage / arm
    pstep = _validated_step(parent, config(out/"probe",stage=stage,arm=arm,
                                               target=TARGET[stage],seed=seed,envs=envs,
                                               device=device))
    probe_target = update_boundary_at_or_after(pstep + PROBE_STEPS, envs)
    probe_run = out / "probe"
    # A 2M probe spans only ~16 PPO updates at 4096 worlds. The normal
    # 10-update logging cadence leaves too few points to distinguish
    # collapse from recovery. Log every update ONLY for short probes.
    probe_cfg = replace(
        config(probe_run,stage=stage,arm=arm,target=probe_target,
               seed=seed,envs=envs,device=device),
        log_every_updates=1)
    probe_ck = _phase(probe_run, probe_cfg, parent=parent, provenance=provenance)
    health = probe_health(probe_run)
    _save(out/"probe_health.json",health)
    print(f"[refork] {stage}/{arm}: probe {health}", flush=True)
    if not health["eligible"]:
        return None
    finish_run = out / "finish"
    finish_cfg = config(finish_run,stage=stage,arm=arm,target=TARGET[stage],
                        seed=seed,envs=envs,device=device)
    final = _phase(finish_run, finish_cfg, parent=probe_ck, provenance=provenance)
    return {"path": str(final.resolve()), "sha256": _sha(final),
            "step": _validated_step(final, finish_cfg), "probe": health}


def _terrain_entry_ok(board: dict) -> bool:
    winning = next((r for r in board["rows"] if r["arm"] == board["winner"]), None)
    return bool(winning and winning["eligible"] and
                winning["core"] >= .80 and winning["hard"] >= .05 and
                winning["slope"] >= .10 and winning["gait"] >= .10)


def run(root: Path, *, foundation: Path, seed: int, envs: int,
        screen_envs: int, horizon: int, device: str, stop_after: str) -> None:
    if envs < 8 or envs % 8 or screen_envs < 8 or horizon < 100:
        raise ValueError("invalid world count or exam horizon")
    root = root.expanduser().resolve()
    source = git_provenance()
    starting = _foundation(foundation.expanduser().resolve(),envs=envs)
    root.mkdir(parents=True, exist_ok=True)
    with (root/".campaign.lock").open("a+") as fd:
        try:
            fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another refork campaign is running") from exc
        _context(root,foundation=starting,seed=seed,envs=envs,
                 screen_envs=screen_envs,horizon=horizon,provenance=source)
        ancestor = Path(starting["path"])
        for stage in STAGES:
            if stage == "terrain100":
                prior = _read(root/"walk80"/"scoreboard.json")
                if not _terrain_entry_ok(prior):
                    raise RuntimeError("80M winner not qualified for 100M continuation; STOP")
            print(f"\n[refork] {stage}: five challengers from {_validated_step(ancestor,config(root/stage/arms(stage)[0]/'probe',stage=stage,arm=arms(stage)[0],target=TARGET[stage],seed=seed,envs=envs,device=device)):,}",flush=True)
            candidates = {}
            for name in arms(stage):
                # On resume, completed rejected probes are re-read and not retrained.
                candidates[name] = _run_candidate(root,stage=stage,arm=name,
                   parent=ancestor,seed=seed,envs=envs,device=device,provenance=source)
            board = _scoreboard(root,stage,candidates,parent=ancestor,
                                screen_envs=screen_envs,horizon=horizon,device=device)
            print(f"[refork] {stage}: {board['reason']} ({board['winner']}; {board['qualification']})",flush=True)
            if not board["winner"]:
                raise RuntimeError(f"{stage} produced no viable research continuation. "
                                   f"Kept parent {ancestor}; see {root/stage/'scoreboard.md'}")
            chosen = ({"path":str(ancestor.resolve()),"sha256":_sha(ancestor),
                       "step":_validated_step(ancestor,config(root/stage/arms(stage)[0]/'probe',
                         stage=stage,arm=arms(stage)[0],target=TARGET[stage],seed=seed,
                         envs=envs,device=device))}
                      if board["winner"] == "incumbent"
                      else candidates[board["winner"]])
            next_ck = Path(chosen["path"])
            selection_file = root / stage / "selection.json"
            selection = {"version":VERSION,"winner":board["winner"],
                         "qualification":board["qualification"],
                         "parent_sha256":_sha(ancestor),"checkpoint":chosen,
                         "scoreboard_sha256":_sha(root/stage/"scoreboard.json")}
            if selection_file.exists():
                if _read(selection_file) != selection:
                    raise RuntimeError("winner selection changed after promotion")
            else:
                _save(selection_file,selection)
            ancestor = next_ck
            if stage == stop_after:
                name = ("HELD-OUT QUALIFIED research winner"
                        if board["qualification"] in ("certified", "incumbent-qualified")
                        else "PROVISIONAL research parent (not certified)")
                print(f"[refork] stopped after {stage}. {name}: {ancestor}",flush=True)
                return


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="312-input five-way winner/refork: general, turns, recovery, efficiency, earned-height steps")
    p.add_argument("--root",type=Path,default=Path.home()/"robot_lab"/"emu_runs"/"v5020_refork")
    p.add_argument("--foundation",type=Path,help="exact existing ~20M schema-9 final.pkl; no new foundation trained")
    p.add_argument("--seed",type=int,default=733)
    p.add_argument("--envs",type=int,default=2048)
    p.add_argument("--screen-envs",type=int,default=32)
    p.add_argument("--screen-horizon",type=int,default=300)
    p.add_argument("--device",choices=("cuda","cpu","auto","gpu"),default="cuda")
    p.add_argument("--stop-after",choices=STAGES,default="walk40")
    mode=p.add_mutually_exclusive_group()
    mode.add_argument("--run",action="store_true",help="real foreground GPU training")
    mode.add_argument("--status",action="store_true",help="read-only persisted winners and scoreboards")
    a=p.parse_args(argv)
    if a.status:
        print(json.dumps({s:{"scoreboard":_read(a.root/s/"scoreboard.json")
                                 if (a.root/s/"scoreboard.json").is_file() else None,
                             "selection":_read(a.root/s/"selection.json")
                                 if (a.root/s/"selection.json").is_file() else None}
                          for s in STAGES},indent=2))
        return 0
    if not a.run:
        print(json.dumps(plan(a.envs),indent=2))
        return 0
    if not a.foundation:
        p.error("--run requires --foundation (the exact existing 312-input 20M checkpoint)")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false")
    run(a.root,foundation=a.foundation,seed=a.seed,envs=a.envs,
        screen_envs=a.screen_envs,horizon=a.screen_horizon,
        device=a.device,stop_after=a.stop_after)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
