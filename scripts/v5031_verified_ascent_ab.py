#!/usr/bin/env python3
"""v5031 matched 80M B -> 100M B-control / B+verified-ascent study.

PLAN ONLY unless --run. The original v5030 80M checkpoint is never altered.
This new source adds ONLY an opt-in reward-only per-episode height high-water
mark; actor schema, network, PPO optimizer, sensor contract and physical model
remain the same. The cross-SHA fork is deliberate and recorded. Tests and
both 2048-CUDA PPO preflights are mandatory before either training arm.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import campaign_support as base

BRANCH = "feature/v5031-verified-ascent"
PARENT_SHA = "a7be11b587f0912e642e100b22ec6d617f548f5b"
SEED = 747
ENVS = 2048
START_M, TARGET_M = 80, 100
CASES = (
    {"name": "B80_control", "reallocated": 0.0, "recipe": "main", "schedule": "balanced",
     "from_steps": START_M*1_000_000, "steps": TARGET_M*1_000_000},
    {"name": "C80_verified_ascent3", "reallocated": 3.0, "recipe": "main", "schedule": "balanced",
     "from_steps": START_M*1_000_000, "steps": TARGET_M*1_000_000},
)
ANCESTOR_DEFAULT = (
    Path.home() / "robot_lab/emu_runs/v5030_ab_learning_horizon_s747/"
    "at80M/B_halfsecond25/checkpoints/final.pkl"
)
ROOT_DEFAULT = Path.home() / "robot_lab/emu_runs/v5031_verified_ascent_s747"


def json_at(path):
    return json.loads(path.read_text())


def hashfile(path):
    digest=hashlib.sha256()
    with path.open("rb") as file:
        for piece in iter(lambda: file.read(1024*1024), b""):
            digest.update(piece)
    return digest.hexdigest()


def checked_source():
    sha=base.git("rev-parse","HEAD")
    if base.git("branch","--show-current")!=BRANCH or base.git("status","--porcelain"):
        raise RuntimeError("v5031 must run on its clean unmodified feature branch")
    return sha


def parent_check(path):
    if not path.is_file() or path.name!="final.pkl":
        raise RuntimeError("Missing completed original B 80M final checkpoint: "+str(path))
    run=path.parent.parent
    if run.name!="B_halfsecond25":
        raise RuntimeError("Only the original 80M B policy may seed this study")
    meta=json_at(run/"run_meta.json")
    cfg=json_at(run/"training_config.json")
    manifest=json_at(run.parent/"campaign_manifest.json")
    if (meta.get("source_provenance") or {}).get("git_commit")!=PARENT_SHA:
        raise RuntimeError("Original B 80M source mismatch")
    if manifest.get("source_sha")!=PARENT_SHA or manifest.get("target_M")!=80:
        raise RuntimeError("Original B 80M campaign mismatch")
    for key,expected in (
        ("seed",SEED),("envs",ENVS),("temporal_vx_reward_share",.5),
        ("terrain_lesson","coherent-careful"),
        ("growth_recipe","main-mature"),("athletic_rehearsal_frac",.25),
        ("foundation_steps",20_000_000),("force_program","gentle"),
    ):
        if cfg.get(key)!=expected:
            raise RuntimeError(f"Original B80 {key} differs: {cfg.get(key)!r} != {expected!r}")
    if "verified_ascent_reallocation" in cfg and cfg["verified_ascent_reallocation"]!=0:
        raise RuntimeError("Ancestor already has a v5031 reward treatment")
    return {"ancestor":str(path),"sha256":hashfile(path),"source_sha":PARENT_SHA}


def train_command(root, case, resume):
    return [
        sys.executable,"-m","emu.cli","train",
        "--run-dir",str(root/case["name"]),"--profile","sim2real",
        "--device","cuda","--seed",str(SEED),"--envs",str(ENVS),
        "--steps",str(TARGET_M*1_000_000),"--terrain-actor",
        "--terrain-step-max-m",".08","--unified-capabilities",
        "--foundation-steps","20000000","--foundation-mix","main",
        "--teaching-schedule","balanced","--growth-recipe","main-mature",
        "--terrain-lesson","coherent-careful","--force-program","gentle",
        "--athletic-rehearsal-share",".25","--temporal-vx-reward-share",".5",
        "--verified-ascent-reallocation",str(case["reallocated"]),
        "--fixed-eval-every-steps","20000000",
        "--resume",str(resume),
    ]


def smoke(share, device, full):
    command=[
        sys.executable,"scripts/smoke_unified_focus.py",
        "--device",device,"--envs",str(ENVS if full else 24),
        "--terrain-max-m",".08","--growth-recipe","main-mature",
        "--terrain-lesson","coherent-careful","--force-program","gentle",
        "--fast-rehearsal-share",".25","--temporal-vx-reward-share",".5",
        "--verified-ascent-reallocation",str(share)]
    if full:
        command.append("--full-update")
    return command


def exams():
    return {
        "speed_ladder": ["--speed-ladder","--worlds","16","--horizon","700",
                         "--speed",".15","--ladder-speeds",
                         ".1",".2",".3",".4",".5",".6",".8","1.0"],
        "terrain_matrix": ["--terrain-matrix","--worlds","16","--horizon","350",
                           "--speed",".15","--matrix-grades-deg","5","8","12","14.9",
                           "--matrix-heights-cm","2","2.5","3","3.5"],
    }


def brief(root, source_sha):
    documents={}
    for case in CASES:
        docs={}
        for suite in exams():
            path=root/"physical_exams"/case["name"]/suite/"unified_exam.json"
            d=json_at(path)
            if d["source_commit"]!=source_sha:
                raise RuntimeError("Exam source mismatch")
            docs[suite]=d
        documents[case["name"]]=docs
    a,b=(c["name"] for c in CASES)
    for suite in exams():
        if documents[a][suite]["protocol"]!=documents[b][suite]["protocol"]:
            raise RuntimeError("Different physical exam fixtures in "+suite)
        if [r["case"] for r in documents[a][suite]["results"]] != [
                r["case"] for r in documents[b][suite]["results"]]:
            raise RuntimeError("Different exam cases in "+suite)
    s=["# EMU v5031 — verified ascent A/B physical report","",
       f"Frozen source: `{source_sha}`  ",
       f"Common ancestor: B at 80M, original source `{PARENT_SHA}`  ",
       "Both compared at 100M, same seed and same physical fixtures.","",
       "## Supported uphill and downhill crossings","",
       "| Height | Control UP | Transfer UP | Control DOWN | Transfer DOWN |",
       "|---|---:|---:|---:|---:|"]
    d1=documents[a]["terrain_matrix"]["results"]
    d2=documents[b]["terrain_matrix"]["results"]
    for ra,rb in zip(d1,d2):
        if ra["case"].get("kind")!="step":
            continue
        def crossings(r, direction):
            w=[x for x in r["worlds"] if x.get("direction")==direction]
            return f"{sum(x.get('crossing_pass') is True for x in w)}/{len(w)}"
        s.append("| "+ra["case"]["name"]+" | "+
                 " | ".join([crossings(ra,"step-up"),crossings(rb,"step-up"),
                             crossings(ra,"step-down"),crossings(rb,"step-down")])+" |")
    s+=["","## Actual speed and 700-frame survival","",
        "| Requested | Control actual | Transfer actual | Control survived | Transfer survived |",
        "|---|---:|---:|---:|---:|"]
    for ra,rb in zip(documents[a]["speed_ladder"]["results"],
                     documents[b]["speed_ladder"]["results"]):
        if not ra["case"]["name"].startswith("flat_vx"):
            continue
        def mean(r):
            values=[w["achieved_mean_vx_mps"] for w in r["worlds"]
                    if isinstance(w.get("achieved_mean_vx_mps"),(int,float))]
            return sum(values)/len(values) if values else None
        x,y=mean(ra),mean(rb)
        fmt=lambda v: "—" if v is None else f"{v:.3f}"
        s.append("| "+ra["case"]["name"]+" | "+
                 " | ".join((fmt(x),fmt(y),str(ra["survived"])+"/16",
                            str(rb["survived"])+"/16"))+" |")
    s+=["","## Graduation gate","",
        "- Prefer supported **3cm UP** success, not staying upright at the lip.",
        "- Preserve real 0.6m/s tracking and 0.8m/s survival; require lower net heading drift.",
        "- Compare 12°+ incline missions, falls and shaft J/m before promotion.",
        "- No 1m/s claim without sustained walking: a brief speed spike after a fall does not count.",
        "- This is one seed, so an apparent success must be repeated before the final Pinnacle.",""]
    path=root/"physical_comparison_report.md"
    path.write_text("\n".join(s),encoding="utf-8")
    return path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",action="store_true",help="Mandatory CPU + 2048 CUDA gates, two sequential continuations and exams")
    p.add_argument("--skip-smoke",action="store_true",help="Only after an exact-SHA acceptance stamp exists")
    p.add_argument("--ancestor",type=Path,default=ANCESTOR_DEFAULT)
    p.add_argument("--root",type=Path,default=ROOT_DEFAULT)
    args=p.parse_args()
    sha=checked_source()
    root=args.root.expanduser().resolve()
    parent=args.ancestor.expanduser().resolve()
    if root==base.ROOT or base.ROOT in root.parents:
        p.error("Results must be outside the Git working tree")
    if root==parent.parent.parent or parent.parent.parent in root.parents:
        p.error("Results must not overwrite the original B80 campaign")
    ancestor=parent_check(parent)
    manifest={"version":"v5031_verified_ascent_ab_v1","source_sha":sha,
              "branch":BRANCH,"seed":SEED,"envs":ENVS,
              "control_share":.5,"reallocated_credits":[0.,3.],
              "from_steps":START_M*1_000_000,"target_steps":TARGET_M*1_000_000,
              "ancestor":ancestor,"cases":list(CASES),
              "explicit_compatible_cross_source_fork":True}
    print("V5031 80->100M SAME B80 POLICY A/B; only verified transfer differs",flush=True)
    print("SOURCE",sha,"PARENT",PARENT_SHA,flush=True)
    for c in CASES:
        print(c["name"]," ".join(train_command(root,c,parent)),flush=True)
    if not args.run:
        print("PLAN ONLY. Add --run to test and train.",flush=True)
        return 0
    root.mkdir(parents=True,exist_ok=True)
    mf=root/"campaign_manifest.json"
    if mf.is_file():
        existing=json_at(mf)
        if existing!=manifest:
            differing=[key for key in sorted(set(existing)|set(manifest))
                       if existing.get(key)!=manifest.get(key)]
            raise RuntimeError(
                "Experiment manifest or ancestor checkpoint changed; "
                "mismatching fields: "+", ".join(differing))
    elif any(root.iterdir()):
        raise RuntimeError("Nonempty output without manifest")
    else:
        mf.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    env={**os.environ,"PYTHONPATH":str(base.ROOT),
         "XLA_PYTHON_CLIENT_PREALLOCATE":"false",
         "EMU_UNIFIED_CONTINUOUS":"cold-growth-approved"}
    stamp=root/"preflight_pass.json"
    acceptance={"source_sha":sha,"parent_sha256":ancestor["sha256"],
                "cpu":True,"full_cuda_2048":True,"budgets":[0.,3.]}
    if args.skip_smoke:
        if not stamp.is_file() or json_at(stamp)!=acceptance:
            raise RuntimeError("No matching preflight stamp: run again without --skip-smoke")
    else:
        checked_source()
        base.execute([sys.executable,"-m","pytest","-q","tests/test_v5031_verified_ascent.py"],
                     root/"logs"/"verified_helper_tests.log",env)
        for c in CASES:
            checked_source()
            base.execute(smoke(c["reallocated"],"cpu",False),
                         root/"logs"/f"cpu_{c['name']}.log",env)
            checked_source()
            base.execute(smoke(c["reallocated"],"cuda",True),
                         root/"logs"/f"cuda2048_{c['name']}.log",env)
        stamp.write_text(json.dumps(acceptance,indent=2,sort_keys=True)+"\n")
    base.collect_runtime(root)
    try:
        for c in CASES:
            checked_source()
            stage=root/c["name"]
            final=stage/"checkpoints/final.pkl"
            if final.is_file():
                meta=json_at(stage/"run_meta.json")
                if (meta.get("source_provenance") or {}).get("git_commit")!=sha:
                    raise RuntimeError("Existing checkpoint has different source "+str(final))
                print("Completed arm preserved",c["name"],flush=True)
                continue
            if stage.exists() and any(stage.iterdir()):
                if not base.has_checkpoint(stage):
                    raise RuntimeError("Partial run without checkpoint: "+str(stage))
                resume="auto"
            else:
                resume=parent
            base.execute(train_command(root,c,resume),
                         root/"logs"/(c["name"]+".log"),env,
                         campaign=(root,CASES))
            if not final.is_file():
                raise RuntimeError("No completed 100M checkpoint for "+c["name"])
        for c in CASES:
            checked_source()
            actor=root/(c["name"]+".emupolicy")
            if not actor.is_file():
                base.execute([sys.executable,"-m","emu.cli","export-policy",
                              str(root/c["name"]/"checkpoints/final.pkl"),str(actor)],
                             root/"logs"/(c["name"]+"_export.log"),env)
            for suite,flags in exams().items():
                dest=root/"physical_exams"/c["name"]/suite
                if (dest/"unified_exam.json").is_file() and (dest/"unified_exam.md").is_file():
                    print("Completed exam preserved",c["name"],suite,flush=True)
                    continue
                if dest.exists() and any(dest.iterdir()):
                    raise RuntimeError("Partial exam exists: "+str(dest))
                base.execute([sys.executable,"-m","emu.cli","capability-exam",
                              "--policy",str(actor),"--output",str(dest),
                              "--device","cpu",*flags],
                             root/"logs"/(c["name"]+"_"+suite+".log"),env)
    finally:
        base.summary(root,CASES,sha)
    compact=brief(root,sha)
    with zipfile.ZipFile(root/"v5031_verified_ascent_physical_comparison.zip","w",
                         compression=zipfile.ZIP_DEFLATED) as z:
        for filename in ("campaign_manifest.json","preflight_pass.json",
                         "overnight_summary.md","overnight_summary.csv",
                         "physical_comparison_report.md"):
            f=root/filename
            if f.is_file():
                z.write(f,arcname=filename)
        for c in CASES:
            for suite in exams():
                for filename in ("unified_exam.json","unified_exam.md"):
                    path=root/"physical_exams"/c["name"]/suite/filename
                    z.write(path,arcname=f"{c['name']}/{suite}/{filename}")
    print("COMPLETE: review",compact,flush=True)
    print("SHARE:",root/"v5031_verified_ascent_physical_comparison.zip",flush=True)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
