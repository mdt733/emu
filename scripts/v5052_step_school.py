"""v5052: controlled real-PPO step-learning campaign, not frozen-policy tuning.

One 40M cold/common parent (20M flat then early low-step Academy), followed
by seven INDEPENDENT 40M continuations to 80M. All 320M steps are PPO updates.
A separate source-pinned eight-scenario true-prism exam measures genuine
partial and completed stair contact missions. No policy architecture changes.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import csv
import json
import os
from pathlib import Path
import sys

from emu.checkpoint_io import load_checkpoint, update_boundary_at_or_after
from emu.training_config import TrainingConfig, training_sensor_contract
from scripts.v5051_train_campaign import (
    source_sha, validate_checkout, digest, run)

PARENT_STEPS = 40_000_000
TARGET_STEPS = 80_000_000
DEFAULT_ENVS = 2048
SOURCE = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path.home() / "emu-data/runs/v5052-step-school-s735"

# Physically identical baseline until 40M; Academy begins just after the
# established 20M balance/walking nursery so every arm inherits some attempts.
# Six specialist learner and one uninterrupted 360-input actor throughout.
COMMON = (
    "--profile","sim2real","--terrain-actor","--unified-capabilities",
    "--terrain-carpet","--carpet-prisms","--academy-stream",
    "--terrain-step-max-m",".08","--terrain-lesson","coherent-careful",
    "--science-step-approach-m",".22",
    "--growth-recipe","main-mature","--foundation-steps","20000000",
    "--fixed-eval-every-steps","40000000",
    "--solver-iterations","2","--solver-ls-iterations","4",
    "--academy-intro-steps","20000000",
    "--academy-start-steps","40000000",
    "--academy-ramp-steps","15000000",
    "--academy-intro-share",".20","--academy-max-share",".60",
    "--athletic-rehearsal-share",".15","--retention-share",".15",
    "--temporal-vx-reward-share",".25",
)

@dataclass(frozen=True)
class Arm:
    name: str
    hypothesis: str
    overrides: tuple[str,...] = ()

ARMS = (
    Arm("00_control","Same Academy training schedule and mature velocity tracking"),
    Arm("01_goal","Reward real supported Academy forward progress; velocity unchanged",
        ("--science-step-goal-weight",".30")),
    Arm("02_relief","Relax Academy-only forward velocity matching; no added progress reward",
        ("--science-step-velocity-relief",".80")),
    Arm("03_goal_relief","Goal progress plus temporary velocity relief during step manoeuvre",
        ("--science-step-goal-weight",".30",
         "--science-step-velocity-relief",".80")),
    Arm("04_contact_teacher","Goal/relief plus privileged *reward-side* true prism foothold teaching",
        ("--science-step-goal-weight",".30",
         "--science-step-velocity-relief",".80",
         "--science-foothold-bonus","3",
         "--science-supported-cross-bonus","5",
         "--science-bilateral-support-bonus","3")),
    Arm("05_phase_freedom","Goal/relief plus slow low-command phase and NO phase-sync lift payment",
        ("--science-step-goal-weight",".30",
         "--science-step-velocity-relief",".80",
         "--science-phase-lowcmd-scale",".25",
         "--science-phase-sync-strength","0")),
    Arm("06_dense_school","Goal/relief plus denser real step births (tests exposure interactions)",
        ("--science-step-goal-weight",".30",
         "--science-step-velocity-relief",".80",
         "--academy-max-share",".75",
         "--academy-ramp-steps","10000000")),
)

# One held-out physics seed initially. Increasing the *requested* lip frontier
# does not make a fixed-height step: always score measured signed terrain.
# Evaluate slow AND moderate approaches to distinguish timing from clearance.
EXAMS = (
    ("stand","flat",0.0,0.0),
    ("fast","flat",0.70,0.0),
    ("lip05_slow","step",0.12,0.5),
    ("lip15_slow","step",0.12,1.5),
    ("lip30_slow","step",0.12,3.0),
    ("lip05_move","step",0.30,0.5),
    ("lip15_move","step",0.30,1.5),
    ("lip30_move","step",0.30,3.0),
)


def select_arms(names: str):
    if names in ("all",""):
        return ARMS
    values=names.split(",")
    lookup={a.name:a for a in ARMS}
    if not values or len(set(values)) != len(values) or any(x not in lookup for x in values):
        raise ValueError("Unknown/duplicate arm; choices: "+", ".join(lookup))
    return tuple(lookup[x] for x in values)


def command(root, *, seed, envs, arm=None, resume=None, check=False):
    steps=PARENT_STEPS if arm is None else TARGET_STEPS
    directory=root/"parent" if arm is None else root/"arms"/arm.name
    cmd=[sys.executable,"-m","emu.cli","train","--run-dir",str(directory),
         "--seed",str(seed),"--envs",str(envs),
         "--device","cuda","--steps",str(steps),*COMMON]
    if arm:
        cmd.extend(arm.overrides)
    if resume is not None:
        cmd.extend(("--resume",str(resume)))
    if check:
        cmd.append("--check-resume-only")
    return cmd


def contract(root, seed, envs, sha, parent=None):
    expected=dict(schema="v5052_stair_learning_v1",
                  source_sha=sha,seed=seed,envs=envs,
                  parent_steps=PARENT_STEPS,target_steps=TARGET_STEPS,
                  common=list(COMMON),
                  arms=[dict(name=a.name,overrides=list(a.overrides))
                        for a in ARMS])
    path=root/"STUDY_CONTRACT.json"
    if path.exists():
        if json.loads(path.read_text())!=expected:
            raise RuntimeError("Study/source/treatment plan changed; use fresh root")
    else:
        if root.exists() and any(root.iterdir()):
            raise RuntimeError("Nonempty root without study contract; refusing mixed data")
        root.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(expected,indent=2)+"\n")
    if parent is not None:
        lock=root/"PARENT_IDENTITY.json"
        wanted=dict(source_sha=sha,sha256=parent[1],boundary=parent[2])
        if lock.exists():
            if json.loads(lock.read_text())!=wanted:
                raise RuntimeError("Research parent changed after forks were defined")
        else:
            lock.write_text(json.dumps(wanted,indent=2)+"\n")


def parent_identity(root,seed,envs):
    directory=root/"parent"
    checkpoint=directory/"checkpoints/final.pkl"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Complete 40M parent before forking: {checkpoint}")
    cfg=TrainingConfig.load(directory/"training_config.json")
    if (cfg.seed,cfg.envs,cfg.steps)!=(seed,envs,PARENT_STEPS):
        raise RuntimeError("Parent seed/world-count/age mismatch")
    payload=load_checkpoint(
        checkpoint,validate=True,
        expected_observation_schema=cfg.observation_schema,
        expected_sensor_contract=training_sensor_contract(cfg))
    boundary=update_boundary_at_or_after(PARENT_STEPS,envs)
    if int(payload["meta"]["step"])!=boundary:
        raise RuntimeError("Parent PPO update boundary is not 40M")
    stored=(payload["meta"].get("source_provenance") or {}).get("git_commit")
    if stored!=source_sha():
        raise RuntimeError("Parent source does not match active Git commit")
    return checkpoint,digest(checkpoint),boundary


def train_one(root,seed,envs,arm,checkpoint):
    directory=root/"parent" if arm is None else root/"arms"/arm.name
    resume="auto" if (directory/"training_config.json").is_file() else checkpoint
    # Baseline first run passes no resume; every fork passes the SAME
    # immutable parent. Interrupted arms always resume their own history.
    cmd=command(root,seed=seed,envs=envs,arm=arm,resume=resume)
    run(command(root,seed=seed,envs=envs,arm=arm,resume=resume,check=True))
    run(cmd)


def physical_protocol(root,worlds,horizon):
    wanted=dict(schema="v5052_physical_v1",worlds=worlds,horizon=horizon,
                scenarios=[dict(label=label,mode=mode,speed=speed,lip_cm=lip)
                           for label,mode,speed,lip in EXAMS])
    path=root/"PHYSICAL_PROTOCOL.json"
    if path.is_file():
        if json.loads(path.read_text())!=wanted:
            raise RuntimeError("Physical exam protocol changed; use a new study root")
    else:
        path.write_text(json.dumps(wanted,indent=2)+"\n")
    return wanted


def physical_exams(root,arm,seeds,device,worlds,horizon):
    final=root/"arms"/arm.name/"checkpoints/final.pkl"
    if not final.is_file():
        raise FileNotFoundError("Incomplete PPO arm, cannot examine: "+str(final))
    from emu.prism_exam import evaluate_prism_policy
    policy=root/"exports"/(arm.name+".emupolicy")
    policy.parent.mkdir(parents=True,exist_ok=True)
    if not policy.exists():
        run([sys.executable,"-m","emu.cli","export-policy",str(final),str(policy)])
    identity=digest(policy)
    physical_protocol(root,worlds,horizon)
    for seed in seeds:
        for label,mode,speed,lip in EXAMS:
            directory=root/"exams"/arm.name/f"s{seed}-{label}"
            result=directory/"prism_exam.json"
            if result.is_file():
                meta=json.loads(result.read_text())
                p=meta["protocol"]
                if (meta["source_actor_sha256"]!=identity or
                        p["seed"]!=seed or p["worlds"]!=worlds or
                        p["horizon"]!=horizon or meta["mode"]!=mode or
                        abs(p["requested_vx_mps"]-speed)>1e-7 or
                        abs(p["requested_lip_frontier_cm"]-lip)>1e-7):
                    raise RuntimeError("Exam changed policy or protocol: "+str(result))
                continue
            if directory.exists():
                raise RuntimeError("Incomplete exam: move aside "+str(directory))
            print(f"STEP SCIENCE exam {arm.name}: {label} seed={seed}",flush=True)
            evaluate_prism_policy(
                policy=policy,output=directory,worlds=worlds,horizon=horizon,
                seed=seed,device=device,mode=mode,speed=speed,
                lip_cm=lip,grade_deg=0.)


def forensic(root,arms,source=None):
    """Diagnose *where* step traversal stalls; do not invent attempts.

    Accept older v5051 exam directories as input without changing or
    overwriting them. Counts are conditional on REAL verified top contacts,
    NOT success divided by requested/attempted obstacles.
    """
    data_root=Path(source).expanduser().resolve() if source else root
    result=[]
    for arm in arms:
        directory=data_root/"exams"/arm.name
        files=sorted(directory.glob("*/prism_exam.json"))
        if not files:
            continue
        events=[]
        by_verdict=Counter()
        worlds=0
        fast_vx=[]
        fast_alive=[]
        stand_footfalls=[]
        examined=0
        for path in files:
            d=json.loads(path.read_text())
            speed=float(d["protocol"]["requested_vx_mps"])
            mode=d["mode"]
            examined+=1
            if mode=="step":
                events+=list(d.get("mission_autopsy",{}).get("events",[]))
                worlds+=len(d["worlds"])
            elif mode=="flat" and speed>.5:
                fast_vx += [float(w["achieved_vx_mps"]) for w in d["worlds"]
                            if w.get("achieved_vx_mps") is not None]
                fast_alive += [bool(w["survived"]) for w in d["worlds"]]
            elif mode=="flat" and speed==0.:
                stand_footfalls += [float(w["footfall_rate_hz"]) for w in d["worlds"]
                                    if w.get("footfall_rate_hz") is not None]
        for e in events:
            by_verdict[e["verdict"]]+=1
        up=[e for e in events if e["signed_actual_height_cm"]>=.15]
        def at_least(values,threshold):
            return sum(v>=threshold for v in values)
        result.append(dict(
            arm=arm.name,exams=examined,step_worlds=worlds,
            eligible_contacts=len(events),
            up_contact_ge_0_15cm=len(up),
            up_contact_ge_1cm=sum(e["signed_actual_height_cm"]>=1 for e in events),
            up_contact_ge_2cm=sum(e["signed_actual_height_cm"]>=2 for e in events),
            up_contact_ge_3cm=sum(e["signed_actual_height_cm"]>=3 for e in events),
            bilateral=sum(bool(e["both_feet_upper_support"]) for e in events),
            progress_ge_15cm=at_least([e["post_contact_progress_m"] for e in events],.15),
            progress_ge_30cm=at_least([e["post_contact_progress_m"] for e in events],.30),
            progress_ge_50cm=at_least([e["post_contact_progress_m"] for e in events],.50),
            complete=sum(bool(e["complete"]) for e in events),
            failure_breakdown=dict(by_verdict),
            fast_vx=round(sum(fast_vx)/len(fast_vx),4) if fast_vx else None,
            fast_survival=round(sum(fast_alive)/len(fast_alive),4) if fast_alive else None,
            still_footfalls_hz=(round(sum(stand_footfalls)/len(stand_footfalls),3)
                                if stand_footfalls else None)))
    # Printing forensic mode reads the prior study but never touches it.
    if source:
        print(json.dumps(result,indent=2))
    else:
        out=root/"STEP_FAILURE_ANALYSIS.json"
        out.write_text(json.dumps(dict(arms=result),indent=2)+"\n")
        cols=["Arm","Top contacts","UP ≥1cm","UP ≥2cm",
              "Bilateral","Progress ≥15cm","Progress ≥30cm",
              "Complete","Fast vx","Fast survival","Zero-cmd footfalls/s"]
        md=["# v5052: physical climb failure-stage autopsy","",
            "A completed mission needs five-frame bilateral top support and "
            "50cm heading-controlled follow-through. Positive foothold "
            "contact alone is not a stair climb. The denominator is "
            "supported contacts, not total step attempts or exact-height "
            "obstacle encounters. Step size is MEASURED, never nominal.","",
            "| "+" | ".join(cols)+" |",
            "| "+" | ".join("---" for _ in cols)+" |"]
        for x in result:
            md.append("| "+" | ".join(map(str,[
                x["arm"],x["eligible_contacts"],x["up_contact_ge_1cm"],
                x["up_contact_ge_2cm"],x["bilateral"],
                x["progress_ge_15cm"],x["progress_ge_30cm"],
                x["complete"],x["fast_vx"],x["fast_survival"],
                x["still_footfalls_hz"]]))+" |")
        md+=["","## Failure verdict counts",""]
        for x in result:
            md.append(f"**{x['arm']}**: "+json.dumps(x["failure_breakdown"]))
        (root/"STEP_FAILURE_ANALYSIS.md").write_text("\n".join(md)+"\n")
        print(f"Report: {root/'STEP_FAILURE_ANALYSIS.md'}",flush=True)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage",required=True,choices=("plan","parent","arms","exams","report","all","forensic"))
    ap.add_argument("--root",type=Path,default=DEFAULT_ROOT)
    ap.add_argument("--seed",type=int,default=735)
    ap.add_argument("--envs",type=int,default=DEFAULT_ENVS)
    ap.add_argument("--only",default="all",help="comma-separated named arms")
    ap.add_argument("--exam-worlds",type=int,default=8)
    ap.add_argument("--exam-horizon",type=int,default=650)
    ap.add_argument("--exam-seeds",default="1701")
    ap.add_argument("--exam-device",choices=("cpu","cuda"),default="cuda")
    ap.add_argument("--source-exams",type=Path,default=None,
                    help="forensic only: previous v5051 study root, read-only")
    a=ap.parse_args()
    if not 0<=a.seed<2**31 or not 8<=a.envs<=4096:
        ap.error("invalid seed or GPU world count")
    if not 2<=a.exam_worlds<=32 or not 100<=a.exam_horizon<=650:
        ap.error("invalid examination geometry")
    try:
        seeds=tuple(int(x) for x in a.exam_seeds.split(","))
        selected=select_arms(a.only)
    except ValueError as ex:
        ap.error(str(ex))
    if (not 1<=len(seeds)<=4 or len(set(seeds))!=len(seeds)
            or any(not 0<=x<2**31 for x in seeds)):
        ap.error("exam seeds must be 1-4 distinct nonnegative integers")
    root=a.root.expanduser().resolve()
    if root==SOURCE or SOURCE in root.parents or root==Path.home():
        ap.error("Study root must be outside checkout and home root")
    if a.stage=="plan":
        print(f"v5052: {PARENT_STEPS//1_000_000}M cold/common parent + "
              f"{len(selected)} x {(TARGET_STEPS-PARENT_STEPS)//1_000_000}M "
              f"independent PPO branches = "
              f"{(PARENT_STEPS+len(selected)*(TARGET_STEPS-PARENT_STEPS))//1_000_000}M")
        for arm in selected:
            print(arm.name,arm.hypothesis,arm.overrides)
            print("  PPO:",*command(root,seed=a.seed,envs=a.envs,arm=arm,
                                    resume=root/"parent/checkpoints/final.pkl"))
        print(f"Physical exam: {len(EXAMS)} scenarios × "
              f"{len(seeds)} seed(s) × {a.exam_worlds} worlds/arm")
        return
    if a.stage=="forensic":
        # Prior v5051 contains different arm IDs; discover their actual
        # saved names rather than pretending they are v5052 treatments.
        if a.source_exams is not None and a.only=="all":
            previous=Path(a.source_exams).expanduser().resolve()/"exams"
            if not previous.is_dir():
                ap.error("--source-exams has no exams/ directory")
            selected=tuple(Arm(p.name,"historical v5051") for p in sorted(
                previous.iterdir()) if p.is_dir())
        forensic(root,selected,a.source_exams)
        return
    validate_checkout()
    sha=source_sha()
    contract(root,a.seed,a.envs,sha)
    if a.stage in ("parent","all"):
        train_one(root,a.seed,a.envs,None,None)
    if a.stage in ("arms","exams","report","all"):
        parent=parent_identity(root,a.seed,a.envs)
        contract(root,a.seed,a.envs,sha,parent)
    if a.stage in ("arms","all"):
        for arm in selected:
            print(f"v5052 PPO: {arm.name} at {PARENT_STEPS:,}->{TARGET_STEPS:,}",flush=True)
            train_one(root,a.seed,a.envs,arm,parent[0])
    if a.stage in ("exams","all"):
        for arm in selected:
            physical_exams(root,arm,seeds,a.exam_device,a.exam_worlds,
                           a.exam_horizon)
    if a.stage in ("report","exams","all"):
        forensic(root,selected)


if __name__=="__main__":
    main()
