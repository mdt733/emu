#!/usr/bin/env python3
"""Source-pinned controlled v5030 25-frame velocity reward A/B.

New source/new checkpoint shape -> FRESH shared 20M gait foundation.
Then SAME source, SAME 20M network/optimizer/normalizer/governor, SAME
seed, episodes, sensor contract and 6-focus teacher to 60M cumulative.
A: original reward (share 0). B: only difference is 25-frame forward
tracking (share 0.5), with instantaneous overspeed/yaw/contacts retained.
No v5029 checkpoint transplant, no edits to main or finished branches.

Plan by default; --run gates CPU & 2048-world CUDA shape acceptance.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import campaign_support as base

BRANCH="feature/v5030-reward-lab"
ROOT_DEFAULT=Path.home()/"robot_lab/emu_runs/v5030_temporal_reward_s747"
SEED=747
ENVS=2048
FOUNDATION=20_000_000
TARGET=60_000_000
CASE=[
    dict(name="foundation20",steps=FOUNDATION,from_steps=0,
         recipe="main",schedule="balanced",temporal=0.0),
    dict(name="A_original25",steps=TARGET,from_steps=FOUNDATION,
         recipe="main",schedule="balanced",temporal=0.0),
    dict(name="B_halfsecond25",steps=TARGET,from_steps=FOUNDATION,
         recipe="main",schedule="balanced",temporal=0.5),
]

def checked_sha():
    sha=base.git("rev-parse","HEAD")
    if base.git("branch","--show-current")!=BRANCH or base.git("status","--porcelain"):
        raise RuntimeError("v5030 must run from an unchanged clean feature branch")
    return sha


def training_command(root,case,*,resume=None):
    cmd=[
        sys.executable,"-m","emu.cli","train",
        "--run-dir",str(root/case["name"]),
        "--profile","sim2real","--device","cuda",
        "--seed",str(SEED),"--envs",str(ENVS),"--steps",str(case["steps"]),
        "--terrain-actor","--terrain-step-max-m",".08",
        "--unified-capabilities","--foundation-steps",str(FOUNDATION),
        "--foundation-mix","main","--teaching-schedule","balanced",
        "--growth-recipe","main-mature","--terrain-lesson","coherent-careful",
        "--force-program","gentle",
        "--athletic-rehearsal-share",".25",
        "--temporal-vx-reward-share",str(case["temporal"]),
        "--fixed-eval-every-steps","20000000"]
    if resume:
        cmd.extend(("--resume",str(resume)))
    return cmd


def smoke_command(backend,share,full=False):
    cmd=[
        sys.executable,"scripts/smoke_unified_focus.py",
        "--device",backend,"--envs",str(2048 if full else 24),
        "--terrain-max-m",".08",
        "--growth-recipe","main-mature",
        "--terrain-lesson","coherent-careful",
        "--force-program","gentle",
        "--fast-rehearsal-share",".25",
        "--temporal-vx-reward-share",str(share)]
    if full:
        cmd.append("--full-update")
    return cmd


def provenance(root,sha):
    return {"version":"v5030_temporal_reward_ab_v1",
            "source_sha":sha,"branch":BRANCH,
            "seed":SEED,"envs":ENVS,"common_foundation":FOUNDATION,
            "two_arm_target":TARGET,
            "causal_history_frames":25,"control_dt_s":.02,
            "only_treatment":"temporal_vx_reward_share",
            "cases":CASE,"no_v5029_checkpoint_transfer":True}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run",action="store_true",
                    help="CPU/CUDA preflight, shared cold foundation then matched 20->60M arms")
    ap.add_argument("--skip-smoke",action="store_true",
                    help="resume after exact source/2048 CUDA preflight; never bypass first gate")
    ap.add_argument("--root",type=Path,default=ROOT_DEFAULT)
    args=ap.parse_args()
    sha=checked_sha()
    root=args.root.expanduser().resolve()
    if root==base.ROOT or base.ROOT in root.parents:
        ap.error("output must be outside source checkout")
    manifest=provenance(root,sha)
    print("V5030 CONTROLLED REWARD STUDY",sha,flush=True)
    print("Cold20 + two identical 20M->60M continuation arms; "
          "only treatment is 0 vs 0.5 velocity smoothing",flush=True)
    for c in CASE:
        print(c["name"]+": "+" ".join(training_command(root,c)),flush=True)
    if not args.run:
        print("PLAN ONLY; add --run",flush=True)
        return 0
    root.mkdir(parents=True,exist_ok=True)
    mf=root/"campaign_manifest.json"
    if mf.is_file():
        if json.loads(mf.read_text())!=manifest:
            raise RuntimeError("Different experiment/checkpoint source: use a new root")
    elif any(root.iterdir()):
        raise RuntimeError("Refusing to overwrite unmanifested output")
    else:
        mf.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    env={**os.environ,
         "PYTHONPATH":str(base.ROOT),
         "XLA_PYTHON_CLIENT_PREALLOCATE":"false",
         "EMU_UNIFIED_CONTINUOUS":"cold-growth-approved"}
    pre=root/"preflight_pass.json"
    expected={"sha":sha,"cpu":True,"full_cuda_2048":True,
              "shares_tested":[0.0,0.5]}
    if args.skip_smoke:
        if not pre.is_file() or json.loads(pre.read_text())!=expected:
            raise RuntimeError("No source-matched 2048-world preflight: omit --skip-smoke")
    else:
        for share in (0.,.5):
            checked_sha()
            base.execute(smoke_command("cpu",share),
                root/"logs"/f"cpu_{share:g}.log",env)
            checked_sha()
            base.execute(smoke_command("cuda",share,full=True),
                root/"logs"/f"cuda2048_{share:g}.log",env)
        pre.write_text(json.dumps(expected,indent=2,sort_keys=True)+"\n")
    base.collect_runtime(root)
    try:
        for i,c in enumerate(CASE):
            checked_sha()
            folder=root/c["name"]
            final=folder/"checkpoints/final.pkl"
            if final.is_file():
                print("Completed branch preserved:",final,flush=True)
                continue
            resume=(
                "auto" if base.has_checkpoint(folder) else
                (root/CASE[0]["name"]/"checkpoints/final.pkl") if i else None
            )
            if i and not (root/CASE[0]["name"]/"checkpoints/final.pkl").is_file():
                raise RuntimeError("Both arms must inherit the SAME validated 20M final checkpoint")
            cmd=training_command(root,c,resume=resume)
            base.execute(cmd,root/"logs"/(c["name"]+".log"),env,
                         campaign=(root,CASE))
            if not final.is_file():
                raise RuntimeError("No final validation for "+c["name"])
        print("A/B training complete; exporting and testing actual policies",flush=True)
        for c in CASE[1:]:
            checked_sha()
            policy=root/(c["name"]+".emupolicy")
            if not policy.is_file():
                base.execute([sys.executable,"-m","emu.cli","export-policy",
                    str(root/c["name"]/"checkpoints/final.pkl"),str(policy)],
                    root/"logs"/(c["name"]+"_export.log"),env)
            exams={
                "speed_ladder":["--speed-ladder","--worlds","16",
                    "--horizon","700","--speed",".15","--ladder-speeds",
                    ".1",".2",".3",".4",".5",".6",".8","1.0"],
                "terrain_matrix":["--terrain-matrix","--worlds","16",
                    "--horizon","350","--speed",".15",
                    "--matrix-grades-deg","5","8","12","14.9",
                    "--matrix-heights-cm","2","2.5","3","3.5"]}
            for label,flags in exams.items():
                folder=root/"physical_exams"/c["name"]/label
                if (folder/"unified_exam.json").is_file() and (
                      folder/"unified_exam.md").is_file():
                    print("Matched exam already present:",folder,flush=True)
                    continue
                if folder.exists() and any(folder.iterdir()):
                    raise RuntimeError("Partial exam output; keep for investigation "+str(folder))
                base.execute([sys.executable,"-m","emu.cli","capability-exam",
                             "--policy",str(policy),"--output",str(folder),
                             "--device","cpu",*flags],
                             root/"logs"/(c["name"]+"_"+label+".log"),env)
    finally:
        base.summary(root,CASE,sha)
        extra=root/"v5030_temporal_physical_comparison.zip"
        with zipfile.ZipFile(extra,"w",compression=zipfile.ZIP_DEFLATED) as z:
            for fn in ("campaign_manifest.json","overnight_results.zip",
                       "overnight_summary.md","overnight_summary.csv"):
                path=root/fn
                if path.is_file():z.write(path,arcname=fn)
            for c in CASE[1:]:
                for label in ("speed_ladder","terrain_matrix"):
                    for fn in ("unified_exam.json","unified_exam.md"):
                        file=root/"physical_exams"/c["name"]/label/fn
                        if file.is_file():z.write(
                            file,arcname=f"{c['name']}/{label}/{fn}")
        print("SHAREABLE A/B DATA:",extra,flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
