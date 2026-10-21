"""v5020 312-input successive-halving training laboratory.

Shared true-contact-slope foundation -> four concurrent hypotheses (trained
sequentially to fit one 8GB GPU) -> two finalists -> one long-run champion.
Every candidate inherits the EXACT common 20M learner. Frozen independent
tests, not on-policy PPO rewards, determine promotions. No main/showcase writes.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from dataclasses import asdict, replace
from pathlib import Path

from .branch_lab import _read, _save
from .checkpoint_io import (
    git_provenance, resolve_target_checkpoint, run_training, checkpoint_step,
)
from .contracts import OBS_SCHEMA_PREVIEW, source_repo_root
from .training_config import (
    foundation_config, mature_config, training_sensor_contract,
    verify_existing_phase_config,
)

VERSION = "v5020-terrain-lab-1"
FOUNDATION = 20_000_000
ROUNDS = {"round1": 80_000_000, "round2": 120_000_000, "round3": 160_000_000}
SLOTS = {"round2": 2, "round3": 1}
ARMS = {
    "balanced": {"hypothesis": "reference 312-input real slope + varied 3–30mm step course", "knobs": {}},
    "slope_recovery": {"hypothesis": "stronger real slopes and servo-aware recovery", "knobs": {
        "domain_terrain_radius": .55, "motor_guidance_weight": .02}},
    "step_focus": {"hypothesis": "more obstacle births and light supported-progress coaching", "knobs": {
        "terrain_height_exposure": .75, "terrain_progress_weight": .025}},
    "efficient": {"hypothesis": "same physical course, heavier servo-efficiency teaching", "knobs": {
        "motor_guidance_weight": .05}},
}


def plan(seed: int, envs: int, screen_envs: int, horizon: int) -> dict:
    total=FOUNDATION+len(ARMS)*(ROUNDS["round1"]-FOUNDATION)
    total+=SLOTS["round2"]*(ROUNDS["round2"]-ROUNDS["round1"])
    total+=SLOTS["round3"]*(ROUNDS["round3"]-ROUNDS["round2"])
    return {
        "version": VERSION, "seed": seed, "train_worlds": envs,
        "screen_worlds": screen_envs, "screen_horizon": horizon,
        "observation_contract": OBS_SCHEMA_PREVIEW,
        "foundation_steps": FOUNDATION, "round_targets_absolute_steps": ROUNDS,
        "arms": ARMS, "finalists": SLOTS,
        "total_environment_steps_all_arms": total,
        "promotions": "held-out core/recovery/true contact slopes and real 20mm step outcomes",
        "safety": "one foreground GPU job, validated ancestor checkpoints, pinned source, stop if insufficient eligible finalists",
    }


def _sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as fd:
        for block in iter(lambda: fd.read(1<<20), b""):
            h.update(block)
    return h.hexdigest()


def config(run: Path, *, stage: str, arm: str | None,
           seed: int, envs: int, device: str, resume: str = ""):
    if stage == "foundation":
        cfg=foundation_config(run_dir=run,seed=seed,envs=envs,device=device)
        return replace(cfg, steps=FOUNDATION, resume=resume,
                       observation_schema=OBS_SCHEMA_PREVIEW,
                       attitude_profile="v5015",pitch_command_max_deg=8.0,
                       domain_randomization=True,domain_terrain_mode="contact",
                       domain_terrain_radius=.16,freeze_obs_norm_after_steps=0)
    if stage not in ROUNDS or arm not in ARMS:
        raise ValueError(f"invalid v5020 stage/arm {stage}/{arm}")
    target=ROUNDS[stage]
    cfg=mature_config("graph",run_dir=run,steps=target,
                      seed=seed,envs=envs,device=device,resume=None)
    # Keep all teaching channels live; 312 actor's physical sensor contract
    # (including masking=false) remains IDENTICAL to foundation.
    cfg=replace(cfg,resume=resume,
        observation_schema=OBS_SCHEMA_PREVIEW,forward_preview_masked=False,
        attitude_profile="v5015",
        pitch_command_max_deg={"round1":15.0,"round2":20.0,"round3":25.0}[stage],
        field_max_influence=.10,
        freeze_obs_norm_after_steps=0,
        domain_randomization=True,domain_terrain_mode="contact",
        domain_terrain_radius={"round1":.30,"round2":.50,"round3":.65}[stage],
        domain_hardware_radius={"round1":0.,"round2":.10,"round3":.20}[stage],
        terrain_box_height_mm=-1,terrain_adaptive_flat_control=False,
        terrain_random_placement=True,
        terrain_height_safety_m={"round1":.030,"round2":.050,"round3":.060}[stage],
        terrain_height_initial_m=.003,
        terrain_height_exposure=.55,terrain_height_edge_fraction=.35,
        terrain_height_growth_m=.001,
        terrain_progress_weight=0.)
    return replace(cfg,**ARMS[arm]["knobs"])


def _completed(run: Path, target: int, envs: int, cfg):
    ck,partial,boundary=resolve_target_checkpoint(
        run,target_steps=target,envs=envs,label=str(run),
        expected_observation_schema=OBS_SCHEMA_PREVIEW,
        expected_sensor_contract=training_sensor_contract(cfg))
    if ck is not None and checkpoint_step(ck,
        expected_observation_schema=OBS_SCHEMA_PREVIEW,
        expected_sensor_contract=training_sensor_contract(cfg)) != boundary:
        raise RuntimeError("incorrect target checkpoint boundary")
    return ck,partial


def _context(root: Path, *, seed: int, envs: int, provenance: dict):
    repo=source_repo_root()
    if root == Path.home().resolve() or (repo and (root == repo or repo in root.parents)):
        raise RuntimeError("lab outputs must be outside the Git checkout and home root")
    if not provenance.get("available") or not provenance.get("commit") or provenance.get("dirty"):
        raise RuntimeError("v5020 lab requires a clean, identifiable Git commit")
    wanted={"version":VERSION,"seed":seed,"envs":envs,
            "git_commit":provenance["commit"]}
    path=root/"lab_context.json"
    if path.exists():
        if _read(path)!=wanted:
            raise RuntimeError("campaign source/seed/world count changed; use the pinned commit or NEW root")
    else:
        _save(path,wanted)


def _stage_run(root: Path, *, stage: str, arm: str | None, parent: Path | None,
               seed: int, envs: int, device: str, provenance: dict) -> Path:
    run=(root/"foundation20" if stage=="foundation" else root/stage/arm)
    target=FOUNDATION if stage=="foundation" else ROUNDS[stage]
    cfg=config(run,stage=stage,arm=arm,seed=seed,envs=envs,device=device)
    contract=training_sensor_contract(cfg)
    ancestor={"path":str(parent.resolve()),"sha256":_sha(parent),
              "step":checkpoint_step(parent,expected_observation_schema=OBS_SCHEMA_PREVIEW,
                                     expected_sensor_contract=contract)} if parent else None
    if ancestor and ancestor["step"]>=target:
        raise RuntimeError("parent is already at or beyond target")
    spec={"stage":stage,"arm":arm,"target":target,
          "parent":ancestor,"version":VERSION,
          "source_commit":provenance["commit"],
          "config":{k:v for k,v in asdict(cfg).items() if k not in ("device","resume")}}
    path=run/"lab_spec.json"
    if path.is_file():
        if _read(path)!=spec:
            raise RuntimeError(f"phase ancestor or configuration drifted: {run}")
    else:
        if run.exists() and any(run.iterdir()):
            raise RuntimeError(f"unrecognised nonempty phase folder: {run}")
        _save(path,spec)
    verify_existing_phase_config(run,cfg)
    meta=run/"run_meta.json"
    if meta.is_file():
        actual=_read(meta).get("source_provenance",{}).get("git_commit")
        if actual!=provenance["commit"]:
            raise RuntimeError(f"training provenance changed for {run}")
    ck,partial=_completed(run,target,envs,cfg)
    if ck is None:
        how="auto" if partial else str(parent) if parent else ""
        print(f"[v5020 lab] {stage}/{arm or 'shared'}: {checkpoint_step(partial or parent) if (partial or parent) else 0:,} -> {target:,}",flush=True)
        run_training(replace(cfg,resume=how),provenance=provenance)
        ck,partial=_completed(run,target,envs,cfg)
        if ck is None:
            raise RuntimeError(f"phase incomplete: {run}, partial checkpoint={partial}")
    print(f"[v5020 lab] validated {stage}/{arm or 'shared'}: {ck}",flush=True)
    return ck


def _score_arm(data: dict, name: str) -> dict:
    def val(key: str) -> float:
        x=data.get(key)
        if x is None or not isinstance(x,(int,float)) or not math.isfinite(float(x)):
            raise ValueError(f"missing/nonfinite held-out metric {key}")
        return float(x)
    try:
        core=val("screen_core_survival")
        hard=(val("screen_resilience100_survival")+val("screen_robust70_survival"))/2
        growth=(val("screen_growth140_survival")+val("screen_growthcombo140_survival"))/2
        slope=(val("screen_contact_slope_survival")+val("screen_contact_harder_survival"))/2
        step=val("course_20mm_crossing_rate")
        touched=val("course_20mm_contact_rate")
        attempts=val("course_20mm_attempt_count")
        motion=sum(val(f"screen_{p}_{axis}_mae") for p in ("robust70","growth140","contact_slope")
                   for axis in ("vx","vy","yaw"))/9
        height=sum(val(f"screen_{p}_height_shortfall_m")
                   for p in ("core","robust70","contact_slope"))/3
        gait=(val("screen_robust70_step_evidence")+val("screen_contact_slope_step_evidence"))/2
    except ValueError as exc:
        return {"arm":name,"eligible":False,"score":None,"reasons":[str(exc)]}
    # No power shortcut lets a standing/falling actor outrank a walking
    # robot. Exercise four independent physical objectives first.
    rank=(.21*core+.22*hard+.16*growth+.19*slope+.16*step+
          .04*touched+.02*gait-.05*min(1.,motion/.65)
          -.05*min(1.,height/.07))
    reasons=[]
    if core<.25: reasons.append("core survival <25%")
    if hard<.04: reasons.append("resilience survival <4%")
    if slope<.04: reasons.append("true-contact slope survival <4%")
    if attempts<1: reasons.append("no eligible real step-course births; cannot rank obstacle performance")
    if height>.065: reasons.append("mean chassis height shortfall >65mm")
    return {"arm":name,"eligible":not reasons,"reasons":reasons,
            "score":round(rank,6),"core":round(core,5),
            "hard_survival":round(hard,5),"growth_survival":round(growth,5),
            "gentle_slope_survival":round(slope,5),"real_slope_survival":round(slope,5),
            "course_20mm_crossing":round(step,5),"course_20mm_contact":round(touched,5),
            "course_20mm_attempts":int(attempts),
            "motion_mae":round(motion,5),"height_shortfall_m":round(height,5),
            "step_evidence":round(gait,5)}


def _score_is_current(root: Path, stage: str, *, envs: int,
                      screen_envs: int, horizon: int, seed: int, device: str,
                      names: list[str]) -> bool:
    file=root/f"{stage}_scoreboard.json"
    if not file.is_file() or not (root/f"{stage}_scorecard.md").exists():
        return False
    board=_read(file)
    if (board.get("round")!=stage or board.get("target")!=ROUNDS[stage]
        or board.get("screen_envs")!=screen_envs or board.get("screen_horizon")!=horizon
        or board.get("score_version")!=VERSION):
        return False
    if {row.get("arm") for row in board.get("results",[])}!=set(names):
        return False
    for row in board["results"]:
        run=root/stage/row["arm"]
        cfg=config(run,stage=stage,arm=row["arm"],seed=seed,envs=envs,device=device)
        ck,_=_completed(run,ROUNDS[stage],envs,cfg)
        if ck is None or _sha(ck)!=row.get("checkpoint_sha256"):
            return False
    return True


def _screen(root: Path, stage: str, names: list[str], *,
            seed: int, envs: int, screen_envs: int,
            horizon: int, device: str):
    if _score_is_current(root,stage,envs=envs,screen_envs=screen_envs,
                         horizon=horizon,seed=seed,device=device,names=names):
        print(f"[v5020 lab] reusing verified {stage} held-out scorecard")
        return _read(root/f"{stage}_scoreboard.json")
    from .evaluation import evaluate_runs
    rows=[]
    for name in names:
        run=root/stage/name
        cfg=config(run,stage=stage,arm=name,seed=seed,envs=envs,device=device)
        ck,_=_completed(run,ROUNDS[stage],envs,cfg)
        if ck is None:
            raise RuntimeError(f"no validated {stage}/{name} to evaluate")
        print(f"[v5020 lab] independent held-out exam: {stage}/{name}",flush=True)
        evidence=evaluate_runs([run],envs=screen_envs,horizon=horizon,
            device=device,eval_pitch_deg=15.0,
            v5020_physical_exam=True,course_root=root)[name]
        row=_score_arm(evidence,name)
        row["checkpoint_sha256"]=_sha(ck)
        rows.append(row)
    if stage=="round1":
        baseline=next(r for r in rows if r["arm"]=="balanced")
        if baseline.get("hard_survival") is not None:
            for row in rows:
                if row["arm"]!="balanced" and row.get("hard_survival") is not None:
                    if row["hard_survival"]<baseline["hard_survival"]-.10:
                        row["eligible"]=False
                        row["reasons"].append("resilience 10pp below balanced baseline")
    rows.sort(key=lambda r:(not r["eligible"],-float(r["score"]) if r["score"] is not None else float("inf"),r["arm"]))
    board={"round":stage,"target":ROUNDS[stage],"screen_pitch_deg":15.,
           "screen_envs":screen_envs,"screen_horizon":horizon,
           "score_version":VERSION,"results":rows}
    _save(root/f"{stage}_scoreboard.json",board)
    # The historical renderer says "v5015" and knows nothing about
    # physical crossings. Give v5020 a truthful human-readable scoreboard.
    lines=[f"# EMU v5020 {stage} — independent policy tournament", "",
           f"Frozen policies at {ROUNDS[stage]:,} absolute steps. All arms "
           f"examined with {screen_envs} worlds and {horizon} frames per test.",
           "",
           "Real 20mm step crossings require collidable foot contact, "
           "real supported passage in the original episode; rewards do not count.",
           "The 20mm exam is a held-out probe, not a claim of maximum height.",
           "",
           "| Arm | Eligible | Score | Core | Recovery | True slopes | 20mm contact | 20mm crossing | Reasons |",
           "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        lines.append(
            f"| {row['arm']} | {row['eligible']} | {row['score']} | "
            f"{row.get('core','—')} | {row.get('hard_survival','—')} | "
            f"{row.get('real_slope_survival','—')} | "
            f"{row.get('course_20mm_contact','—')} | "
            f"{row.get('course_20mm_crossing','—')} | "
            f"{', '.join(row['reasons'])} |")
    scorecard=root/f"{stage}_scorecard.md"
    temp=scorecard.with_name(scorecard.name+'.tmp')
    temp.write_text("\n".join(lines)+"\n",encoding="utf-8")
    temp.replace(scorecard)
    for row in rows:
        print(f"[v5020 lab] {stage} {row['arm']:15} eligible={row['eligible']} score={row['score']} reasons={row['reasons']}",flush=True)
    return board


def _choose(root: Path, from_stage: str, to_stage: str, *,
            board: dict, previous: list[str]) -> list[str]:
    slots=SLOTS[to_stage]
    eligible=[row for row in board["results"] if row.get("eligible") and
              row.get("score") is not None and row["arm"] in previous]
    if len(eligible)<slots:
        raise RuntimeError(f"Only {len(eligible)}/{slots} eligible {from_stage} finalists. STOP: inspect results.md; no arbitrary winner")
    selected=sorted(r["arm"] for r in eligible[:slots])
    lock=root/f"{to_stage}_selection.json"
    data={"selected":selected,"from_round":from_stage,
          "scoreboard_sha256":_sha(root/f"{from_stage}_scoreboard.json")}
    if lock.exists():
        if _read(lock)!=data:
            raise RuntimeError(f"{to_stage} locked selection/scoreboard changed")
    else:
        _save(lock,data)
    return selected


def _report(root: Path):
    from .results import write_report
    try:
        for x in write_report(root):
            print(f"[v5020 lab] report: {x}",flush=True)
    except (OSError,ValueError,TypeError,KeyError) as exc:
        # A reporting failure is visible, but doesn't poison proven training.
        print(f"[v5020 lab] WARNING: offline report generation failed: {exc}",flush=True)


def run(root: Path, *, seed: int, envs: int, screen_envs: int,
        horizon: int, device: str, stop_after: str):
    root=root.expanduser().resolve()
    if envs<8 or envs%8 or screen_envs<4 or horizon<20:
        raise ValueError("envs must be >=8 and a multiple of 8, screen-envs >=4, horizon >=20")
    source=git_provenance()
    root.mkdir(parents=True,exist_ok=True)
    # Linux fcntl prevents two terminal invocations clobbering run state.
    with (root/".campaign.lock").open("a+") as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another campaign already owns {root}") from exc
        _context(root,seed=seed,envs=envs,provenance=source)
        # No hidden dependency on the one-off, 64-world smoke checkpoint.
        parent=_stage_run(root,stage="foundation",arm=None,parent=None,
                          seed=seed,envs=envs,device=device,provenance=source)
        _report(root)
        if stop_after=="foundation":
            return
        prior_names=list(ARMS)
        prior="foundation"
        for stage in ROUNDS:
            if stage=="round1":
                names=prior_names
            else:
                old=_read(root/f"{prior}_scoreboard.json")
                names=_choose(root,prior,stage,board=old,previous=prior_names)
            for name in names:
                ancestor=parent if prior=="foundation" else root/prior/name/"checkpoints"/"final.pkl"
                _stage_run(root,stage=stage,arm=name,parent=ancestor,
                           seed=seed,envs=envs,device=device,provenance=source)
                _report(root)
            _screen(root,stage,names,seed=seed,envs=envs,
                    screen_envs=screen_envs,horizon=horizon,device=device)
            _report(root)
            print(f"[v5020 lab] {stage} complete at {ROUNDS[stage]:,} policy steps",flush=True)
            if stage==stop_after:
                return
            prior=stage
            prior_names=names


def main(argv=None) -> int:
    parser=argparse.ArgumentParser(description="EMU v5020: 20M root -> 4x80M -> 2x120M -> 1x160M")
    parser.add_argument("--root",type=Path,default=Path.home()/"robot_lab"/"emu_runs"/"v5020_lab")
    parser.add_argument("--seed",type=int,default=733)
    parser.add_argument("--envs",type=int,default=512)
    parser.add_argument("--screen-envs",type=int,default=32)
    parser.add_argument("--screen-horizon",type=int,default=300)
    parser.add_argument("--device",choices=("cuda","cpu","auto","gpu"),default="cuda")
    parser.add_argument("--stop-after",choices=("foundation",*ROUNDS),default="round3")
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument("--run",action="store_true",help="execute campaign in foreground; default is plan only")
    mode.add_argument("--status",action="store_true",help="inspect only the persisted campaign context and scoreboard")
    args=parser.parse_args(argv)
    if args.status:
        print(json.dumps({
            "context":_read(args.root/"lab_context.json") if (args.root/"lab_context.json").exists() else None,
            "scores":{r:_read(args.root/f"{r}_scoreboard.json") for r in ROUNDS
                      if (args.root/f"{r}_scoreboard.json").exists()},
        },indent=2,sort_keys=True))
        return 0
    if not args.run:
        print(json.dumps(plan(args.seed,args.envs,args.screen_envs,args.screen_horizon),indent=2,sort_keys=True))
        return 0
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false")
    run(args.root,seed=args.seed,envs=args.envs,
        screen_envs=args.screen_envs,horizon=args.screen_horizon,
        device=args.device,stop_after=args.stop_after)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
