"""Winner-refork campaign: one 312-input lineage, four controlled challengers per generation.

Use the proven main-style locomotion baseline and promote exactly one qualified
policy at 40/60/80M. Only after an 80M locomotion qualification can a separate
5mm physical-step generation run. Never modifies the historical campaign5020.
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

from .campaign5020 import _read, _save, _sha
from .checkpoint_io import (checkpoint_step, git_provenance,
                            resolve_target_checkpoint,
                            run_training, update_boundary_at_or_after)
from .contracts import OBS_SCHEMA_PREVIEW, source_repo_root
from .training_config import (TrainingConfig, foundation_config,
                              training_sensor_contract, verify_existing_phase_config)

VERSION = "v5020-winner-refork-1"
STAGES = ("walk40", "walk60", "walk80", "terrain100")
TARGET = {"walk40": 40_000_000, "walk60": 60_000_000,
          "walk80": 80_000_000, "terrain100": 100_000_000}
PROBE_STEPS = 2_000_000
WALK_ARMS = ("general", "turns", "recovery", "efficiency")
TERRAIN_ARMS = ("flat_control", "step3", "step5", "step5_efficiency")
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
    return TERRAIN_ARMS if stage == "terrain100" else WALK_ARMS


def plan(envs: int = 2048) -> dict:
    return {
        "version": VERSION, "observation_schema": OBS_SCHEMA_PREVIEW,
        "proposed_envs": envs, "source": "existing schema-9 foundation (explicit checkpoint)",
        "generations": [
            {"stage": s, "target_steps_per_actor": TARGET[s],
             "arms": list(arms(s)), "probe_steps_per_arm": PROBE_STEPS,
             "winner_count": 1, "physical_step_training": s == "terrain100"}
            for s in STAGES],
        "safety": ("probe each new curriculum for 2M, reject collapse/absurd spring energy, "
                   "screen all survivors in matched held-out physics, keep only qualified winner; "
                   "terrain100 requires an eligible walk80 winner"),
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
    if stage != "terrain100":
        if arm == "turns":
            cfg = replace(cfg, combo_mix_frac={
                "walk40": .15, "walk60": .35, "walk80": .55}[stage])
        elif arm == "recovery":
            cfg = replace(cfg, push_max_force_n={
                "walk40": .90, "walk60": 1.20, "walk80": 1.80}[stage])
        elif arm == "efficiency":
            cfg = replace(cfg, motor_guidance_weight={
                "walk40": .012, "walk60": .020, "walk80": .030}[stage])
    else:
        # Identical static MJX model topology for all four contenders.
        # In control, the mocap box remains off-course (x=5m).
        cfg = replace(cfg, terrain_adaptive_flat_control=True,
                      terrain_random_placement=True,
                      terrain_height_initial_m=.003,
                      terrain_height_exposure=.25,
                      terrain_height_edge_fraction=.35,
                      terrain_height_growth_m=.0005,
                      terrain_height_safety_m=0. if arm == "flat_control"
                      else .003 if arm == "step3" else .005)
        if arm == "step5_efficiency":
            cfg = replace(cfg, motor_guidance_weight=.02)
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
    if not 19_000_000 <= step <= 22_000_000:
        raise ValueError(f"expected ~20M original foundation, got {step:,}")
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
    """Reject gross on-policy collapse before paying for a full 20M generation."""
    with (probe_run / "metrics.csv").open(newline="", encoding="utf-8") as fd:
        rows = list(csv.DictReader(fd))
    if not rows:
        raise RuntimeError(f"no actual rollout metrics for probe {probe_run}")
    recent = rows[-3:]
    def readings(name: str) -> list[float]:
        vals = [float(r[name]) for r in recent]
        if not all(math.isfinite(v) for v in vals):
            raise ValueError(f"nonfinite {name} in {probe_run}")
        return vals
    fall = sum(readings("fall_rate")) / len(recent)
    length = sum(readings("ep_len_mean")) / len(recent)
    max_spring = max(readings("spring_potential_energy_j"))
    reasons = []
    if fall > .92:
        reasons.append(f"probe fall rate {fall:.1%} >92%")
    if length < 70.:
        reasons.append(f"probe mean episode length {length:.1f} <70 frames")
    if max_spring > 50.:
        reasons.append(f"unphysical mean spring potential {max_spring:.1f}J >50J")
    return {"eligible": not reasons, "reasons": reasons,
            "fall_rate": round(fall, 5), "mean_episode_frames": round(length, 2),
            "max_logged_spring_j": round(max_spring, 3)}


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
        crossing = val("course_5mm_crossing_rate") if stage == "terrain100" else None
        attempts = val("course_5mm_attempt_count") if stage == "terrain100" else None
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
    if stage == "terrain100" and (attempts < 1 or crossing < .01):
        reasons.append("no verified 5mm physical box crossing")
    rank = (.33 * core + .20 * hard + .20 * slope + .12 * growth +
            .10 * gait - .035 * min(1., motion / .65) -
            .025 * min(1., height / .055) -
            .025 * min(2., power / 15.) -
            .025 * min(1., overload) +
            (.12 * crossing if crossing is not None else 0.))
    return {"arm": name, "eligible": not reasons,
            "score": round(rank, 6), "reasons": reasons,
            "core": round(core, 5), "hard": round(hard, 5),
            "slope": round(slope, 5), "growth": round(growth, 5),
            "gait": round(gait, 5), "motion_mae": round(motion, 5),
            "height_shortfall_m": round(height, 5),
            "shaft_abs_power_w": round(power, 5),
            "overrated_frac": round(overload, 5),
            "crossing_5mm": None if crossing is None else round(crossing, 5)}


def _exam(ck: Path, *, stage: str, root: Path, screen_envs: int,
          horizon: int, device: str) -> dict:
    from .evaluation import evaluate_runs
    # Checkpoint-path evaluation reads its owner run configuration and exact
    # schema-9 sensor contract; all candidates face identical examiner worlds.
    kwargs = {"envs": screen_envs, "horizon": horizon, "device": device,
              "eval_pitch_deg": 8., "key_by_path": True,
              "write_screen_files": False}
    if stage == "terrain100":
        kwargs.update(v5020_physical_exam=True, physical_step_height_mm=5,
                      course_root=root)
    else:
        kwargs["v5020_locomotion_exam"] = True
    result = evaluate_runs([ck], **kwargs)
    return result[str(ck.resolve())]


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
    # The parent is examined on the SAME difficulty/seed as contenders.
    parent_evidence = _exam(parent, stage=stage, root=root,
                            screen_envs=screen_envs, horizon=horizon, device=device)
    parent_rating = _score(parent_evidence, stage=stage, name="incumbent")
    results = []
    for name in arms(stage):
        saved = candidates[name]
        if saved is None:
            results.append({"arm": name, "eligible": False, "score": None,
                            "reasons": ["probe rejected"]})
            continue
        evidence = _exam(Path(saved["path"]), stage=stage, root=root,
                         screen_envs=screen_envs, horizon=horizon, device=device)
        score = _score(evidence, stage=stage, name=name)
        score["checkpoint_sha256"] = saved["sha256"]
        results.append(score)
    eligible = sorted((r for r in results if r["eligible"]),
                      key=lambda r: (-r["score"], r["arm"]))
    if not eligible:
        winner = None
        reason = "No eligible candidate; retain incumbent and STOP"
    elif (parent_rating["eligible"] and
          (eligible[0]["score"] < parent_rating["score"] - .005 or
           eligible[0]["core"] < parent_rating["core"] - .10)):
        winner = None
        reason = "Every candidate regressed against frozen incumbent; STOP"
    else:
        winner = eligible[0]["arm"]
        reason = "Promoted best qualifying frozen policy"
    board = {"version": VERSION, "stage": stage, "target": target,
             "parent_sha256": _sha(parent),
             "candidates": {k: v["sha256"] if v else None for k, v in candidates.items()},
             "incumbent": parent_rating, "rows": results,
             "winner": winner, "reason": reason}
    _save(location, board)
    report = root / stage / "scoreboard.md"
    report.write_text(
        f"# {stage} — winner-refork held-out screening\n\n"
        f"Parent SHA-256: {_sha(parent)}\n\nWinner: {winner or 'NONE'}\n\n"
        f"Verdict: {reason}\n\n"
        "| Arm | Eligible | Score | Core | Recovery | Contact slopes | 5mm crossing | Reasons |\n"
        "|---|---|---:|---:|---:|---:|---:|---|\n" +
        "".join(f"| {r['arm']} | {r['eligible']} | {r['score']} | "
                f"{r.get('core','—')} | {r.get('hard','—')} | "
                f"{r.get('slope','—')} | {r.get('crossing_5mm','—')} | "
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
    probe_cfg = config(probe_run,stage=stage,arm=arm,target=probe_target,
                       seed=seed,envs=envs,device=device)
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
                    raise RuntimeError("80M winner not qualified for physical boxes; STOP")
            print(f"\n[refork] {stage}: four challengers from {_validated_step(ancestor,config(root/stage/arms(stage)[0]/'probe',stage=stage,arm=arms(stage)[0],target=TARGET[stage],seed=seed,envs=envs,device=device)):,}",flush=True)
            candidates = {}
            for name in arms(stage):
                # On resume, completed rejected probes are re-read and not retrained.
                candidates[name] = _run_candidate(root,stage=stage,arm=name,
                   parent=ancestor,seed=seed,envs=envs,device=device,provenance=source)
            board = _scoreboard(root,stage,candidates,parent=ancestor,
                                screen_envs=screen_envs,horizon=horizon,device=device)
            print(f"[refork] {stage}: {board['reason']} ({board['winner']})",flush=True)
            if not board["winner"]:
                raise RuntimeError(f"{stage} produced no safely promotable winner. "
                                   f"Kept parent {ancestor}; see {root/stage/'scoreboard.md'}")
            chosen = candidates[board["winner"]]
            next_ck = Path(chosen["path"])
            selection_file = root / stage / "selection.json"
            selection = {"version":VERSION,"winner":board["winner"],
                         "parent_sha256":_sha(ancestor),"checkpoint":chosen,
                         "scoreboard_sha256":_sha(root/stage/"scoreboard.json")}
            if selection_file.exists():
                if _read(selection_file) != selection:
                    raise RuntimeError("winner selection changed after promotion")
            else:
                _save(selection_file,selection)
            ancestor = next_ck
            if stage == stop_after:
                print(f"[refork] stopped after {stage}. Champion: {ancestor}",flush=True)
                return


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="312-input four-way winner/refork walk40 -> walk60 -> walk80 -> guarded terrain100")
    p.add_argument("--root",type=Path,default=Path.home()/"robot_lab"/"emu_runs"/"v5020_refork")
    p.add_argument("--foundation",type=Path,help="exact existing ~20M schema-9 final.pkl; no new foundation trained")
    p.add_argument("--seed",type=int,default=733)
    p.add_argument("--envs",type=int,default=2048)
    p.add_argument("--screen-envs",type=int,default=32)
    p.add_argument("--screen-horizon",type=int,default=300)
    p.add_argument("--device",choices=("cuda","cpu","auto","gpu"),default="cuda")
    p.add_argument("--stop-after",choices=STAGES,default="walk80")
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
