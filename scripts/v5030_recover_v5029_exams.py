#!/usr/bin/env python3
"""Recover v5029 post-training physical exams WITHOUT rerunning any PPO.

Run from the *separate* v5030 reward-lab checkout, keeping original v5029
source, final checkpoint and incomplete first exam directory untouched.
The sole examiner behavior needed for the speed ladder is its explicit
fast_flat_ladder safety envelope: ordinary schedule caps are unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

from campaign_support import execute

EXAM_BRANCH="feature/v5030-reward-lab"
TRAIN_SHA="8288dc0b02c2b97f9a915cd14c0b65958e6b77c4"
RUN_ROOT=Path.home()/"robot_lab/emu_runs/v5029_speed_restoration_s737"
SOURCE_CHECKOUT=Path.home()/"emu-v5029"


def git(root: Path,*args: str) -> str:
    return subprocess.check_output(
        ["git","-C",str(root),*args],text=True,stderr=subprocess.STDOUT
    ).strip()


def sha256(path: Path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()


def source_check():
    here=Path(__file__).resolve().parents[1]
    current=git(here,"rev-parse","HEAD")
    if git(here,"branch","--show-current")!=EXAM_BRANCH or git(here,"status","--porcelain"):
        raise RuntimeError("Use an unmodified v5030 reward-lab checkout")
    if not SOURCE_CHECKOUT.is_dir() or git(SOURCE_CHECKOUT,"rev-parse","HEAD")!=TRAIN_SHA:
        raise RuntimeError("Original ~/emu-v5029 source SHA mismatch/missing. DO NOT modify it")
    if not (here/"emu"/"physical_exam.py").is_file():
        raise RuntimeError("Not in the active examiner checkout")
    return here,current


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root",type=Path,default=RUN_ROOT)
    ap.add_argument("--run",action="store_true",help="run the read-only exams; default prints plan")
    args=ap.parse_args()
    here,exam_sha=source_check()
    root=args.root.expanduser().resolve()
    checkpoint=root/"cold120_speed_terrain/checkpoints/final.pkl"
    policy=root/"physical_policy.emupolicy"
    if not checkpoint.is_file() or not policy.is_file():
        raise FileNotFoundError("Missing completed v5029 final.pkl or physical_policy.emupolicy")
    output=root/"recovered_exams_v5030"
    actor_sha=sha256(policy)
    tasks={
        "speed_ladder": [
            "--speed-ladder","--worlds","16","--horizon","700","--speed",".15",
            "--ladder-speeds",".1",".2",".3",".4",".5",".6",".8","1.0",
        ],
        "terrain_matrix": [
            "--terrain-matrix","--worlds","16","--horizon","350","--speed",".15",
            "--matrix-grades-deg","5","8","12","14.9",
            "--matrix-heights-cm","2","2.5","3","3.5",
        ],
    }
    print("TRAINED POLICY: exact frozen v5029 source",TRAIN_SHA,flush=True)
    print("EXAMINER: separately pinned v5030 source",exam_sha,flush=True)
    print("POLICY SHA256:",actor_sha,flush=True)
    print("No training. No modifications to source or checkpoint.",flush=True)
    print("NEW exam results:",output,flush=True)
    if not args.run:
        print("PLAN ONLY; add --run",flush=True)
        return 0
    output.mkdir(parents=True,exist_ok=True)
    manifest={
        "status":"posthoc_read_only_exam",
        "training_source_sha":TRAIN_SHA,
        "exam_source_sha":exam_sha,
        "actor_sha256":actor_sha,
        "checkpoint_path":str(checkpoint),
        "original_incomplete_speed_exam":str(root/"physical_exams/speed_ladder"),
        "note":"v5029 speed ladder stopped before 0.8m/s because generic make_schedule capped at 0.6. No training was interrupted.",
        "protocols":tasks,
    }
    manifest_file=output/"recovery_manifest.json"
    if manifest_file.is_file():
        if json.loads(manifest_file.read_text())!=manifest:
            raise RuntimeError("Cannot mix different source or policy into an existing exam directory")
    elif any(output.iterdir()):
        raise RuntimeError("Unexpected existing exam outputs without a matching manifest")
    else:
        manifest_file.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    env={**os.environ,"PYTHONPATH":str(here),"XLA_PYTHON_CLIENT_PREALLOCATE":"false"}
    for name,flags in tasks.items():
        folder=output/name
        data=folder/"unified_exam.json"
        md=folder/"unified_exam.md"
        if data.is_file() and md.is_file():
            report=json.loads(data.read_text())
            if (report["source_commit"]!=exam_sha
                or report["actor_artifact_sha256"]!=json.loads(data.read_text())["actor_artifact_sha256"]):
                raise RuntimeError("Existing exam source mismatch; manual review required")
            print("COMPLETE; not rerunning:",folder,flush=True)
            continue
        if folder.exists() and any(folder.iterdir()):
            raise RuntimeError("Incomplete exam output, preserving it for investigation: "+str(folder))
        cmd=[
            sys.executable,"-m","emu.cli","capability-exam",
            "--policy",str(policy),"--output",str(folder),
            "--device","cpu",*flags,
        ]
        print("READ-ONLY EXAM:",name,flush=True)
        execute(cmd,output/(name+".log"),env)
        report=json.loads(data.read_text())
        if report["source_commit"]!=exam_sha or report["source_training_step"]<120_000_000:
            raise RuntimeError("Completed exam has unexpected source or training-age provenance")
        if len(report["results"])!=len(
                [c for c in report["results"]]):
            raise RuntimeError("Invalid exam document")
        print("Exam complete:",name,len(report["results"]),"conditions",flush=True)
    # The final package includes the original 120M training report and all
    # independent per-world results. No rewriting the old overnight ZIP.
    bundle=output/"v5029_recovered_physical_results.zip"
    with zipfile.ZipFile(bundle,"w",compression=zipfile.ZIP_DEFLATED) as z:
        files=[manifest_file,
               root/"overnight_results.zip",
               root/"cold120_speed_terrain/results.md",
               root/"cold120_speed_terrain/results_data.json",
               root/"cold120_speed_terrain/fixed_eval.csv"]
        for name in tasks:
            files.extend([output/name/"unified_exam.json",
                          output/name/"unified_exam.md"])
        for file in files:
            if file.is_file():
                z.write(file,arcname=(
                    "training/"+file.name if file.parent!=output and
                    file.parent.name not in tasks else
                    str(file.relative_to(output))))
    print("RECOVERED BOTH PHYSICAL EXAMS:",output,flush=True)
    print("SEND THIS ZIP:",bundle,flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
