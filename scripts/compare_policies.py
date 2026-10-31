#!/usr/bin/env python3
"""Read-only, source-matched comparison: archived 232 actor vs new 360 Pinnacle.

Both actors run the SAME real long two-slab collision world, commands,
time horizons, seeds and scoring. Main uses its genuine 232-input
sensor contract; the candidate uses its genuine 360-input 4x4 array.
No observation conversion, checkpoint writing or release promotion.
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

from campaign_support import execute

REPO=Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATE=REPO/"showcase/champion.emupolicy"
DEFAULT_OUT=Path.home()/"robot_lab/emu_runs/v5032_main_physical_comparison"
CASES={
    "speed_ladder": [
        "--speed-ladder","--worlds","16","--horizon","700","--speed",".15",
        "--ladder-speeds",".1",".2",".3",".4",".5",".6",".8","1.0"],
    "terrain_matrix": [
        "--terrain-matrix","--worlds","16","--horizon","350","--speed",".15",
        "--matrix-grades-deg","5","8","12","14.9",
        "--matrix-heights-cm","2","2.5","3","3.5"],
}

def git(*args):
    return subprocess.check_output(
        ["git",*args],cwd=REPO,text=True,stderr=subprocess.STDOUT).strip()

def file_hash(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda:f.read(1<<20),b""):
            h.update(part)
    return h.hexdigest()

def summary_rows(report,name,kind):
    rows=[]
    for condition in report["results"]:
        p=condition["case"]
        samples=condition["worlds"]
        speed=[w.get("achieved_mean_vx_mps") for w in samples
               if w.get("achieved_mean_vx_mps") is not None]
        bias=[w.get("tracking_signed_bias_yaw_rad_s") for w in samples
              if w.get("tracking_signed_bias_yaw_rad_s") is not None]
        up=[w for w in samples if w.get("direction")=="step-up"]
        down=[w for w in samples if w.get("direction")=="step-down"]
        rows.append(dict(
            policy=name,exam=kind,condition=p["name"],
            requested_vx_mps=p.get("vx",report["protocol"]["speed"]),
            mean_achieved_vx_mps=round(sum(speed)/len(speed),4) if speed else "",
            mean_abs_yaw_bias_rad_s=round(sum(map(abs,bias))/len(bias),4) if bias else "",
            alive=condition["survived"],worlds=len(samples),
            mission=condition["mission_passed"],
            strict=condition["passed"],
            supported_step_up=f"{sum(w['crossing_pass'] is True for w in up)}/{len(up)}" if up else "",
            supported_step_down=f"{sum(w['crossing_pass'] is True for w in down)}/{len(down)}" if down else "",
            slope_stable=condition["stable"] if p["kind"]=="slope" else "",
        ))
    return rows


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run",action="store_true",help="execute read-only matched exams")
    ap.add_argument("--candidate",type=Path,default=DEFAULT_CANDIDATE,
                    help="exported 360-input actor; defaults to v5032 candidate")
    ap.add_argument("--baseline",type=Path,required=True,
                    help="old 232-input actor exported from rollback SHA 2ec9e9d (required); new main is 360")
    ap.add_argument("--output-root",type=Path,default=DEFAULT_OUT)
    args=ap.parse_args()
    if git("branch","--show-current") not in ("main","release/v5032-pinnacle-candidate") or git("status","--porcelain"):
        raise RuntimeError("Run from clean main or the exact v5032 release checkout only")
    source=git("rev-parse","HEAD")
    root=args.output_root.expanduser().resolve()
    candidates={"main_232":args.baseline.expanduser().resolve(),
                "candidate_360":args.candidate.expanduser().resolve()}
    if candidates["main_232"]==candidates["candidate_360"]:
        raise RuntimeError("Baseline and candidate must be different actors")
    if root==REPO or REPO in root.parents:
        raise RuntimeError("Results belong outside the source checkout")
    # Verify both actors with one loader, including actual robot-model SHA,
    # motor mapping and the ORIGINAL sensor observation contract.
    from emu.viewer_server import load_exported_actor_for_exam
    from emu.contracts import OBS_SCHEMA_SENSOR,OBS_SCHEMA_TERRAIN
    actors={}
    for label,path in candidates.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        _,_,info=load_exported_actor_for_exam(path)
        expected=OBS_SCHEMA_SENSOR if label=="main_232" else OBS_SCHEMA_TERRAIN
        if info["observation_schema"]!=expected or info["array_only"] != (label=="candidate_360"):
            raise RuntimeError(f"{label}: wrong original actor/sensor meaning")
        actors[label]={"path":str(path),"sha256":file_hash(path),
                       "policy_observation_schema":expected,
                       "training_step":info["source"]["step"],
                       "original_training_source":info["source"].get("git_commit")}
    manifest={"schema":"emu_cross_actor_real_physical_v1",
              "exam_source_commit":source,
              "fixture":"long real two-slab support, common 32m half-length",
              "actors":actors,"cases":CASES,
              "note":"same real collision, actual commands/seeds/scoring; unequal authentic sensors by design"}
    print("PINNACLE COMPARISON / SHA:",source,flush=True)
    print("ARCHIVED 232:",actors["main_232"],flush=True)
    print("CURRENT 360:",actors["candidate_360"],flush=True)
    print("EXAMS: 16 worlds, 700-frame speed ladder; 350-frame slope + paired step matrix")
    if not args.run:
        print("PLAN ONLY: append --run; no files written")
        return 0
    root.mkdir(parents=True,exist_ok=True)
    mpath=root/"manifest.json"
    if mpath.is_file() and json.loads(mpath.read_text())!=manifest:
        raise RuntimeError("Existing results belong to a different policy or source; use new --output-root")
    if not mpath.is_file():
        if any(root.iterdir()):
            raise RuntimeError("Refusing existing output without matching manifest")
        mpath.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    env={**os.environ,"PYTHONPATH":str(REPO),"XLA_PYTHON_CLIENT_PREALLOCATE":"false"}
    for label,path in candidates.items():
        for kind,flags in CASES.items():
            folder=root/label/kind
            jsonpath=folder/"unified_exam.json"
            if jsonpath.is_file() and (folder/"unified_exam.md").is_file():
                report=json.loads(jsonpath.read_text())
                if (report["source_commit"]!=source or
                    report["actor_artifact_sha256"]!=actors[label]["sha256"]):
                    raise RuntimeError("Existing exam source or actor changed; refuse reuse")
                print("Already complete:",folder,flush=True)
                continue
            if folder.is_dir() and any(folder.iterdir()):
                raise RuntimeError("Preserving partial exam; select new output root: "+str(folder))
            command=[sys.executable,"-m","emu.cli","capability-exam",
                     "--policy",str(path),"--output",str(folder),"--device","cpu"]
            if label=="main_232":
                command+=["--legacy-main-baseline"]
            command+=flags
            execute(command,root/(label+"_"+kind+".log"),env)
            report=json.loads(jsonpath.read_text())
            if (report["source_commit"]!=source or
                report["actor_artifact_sha256"]!=actors[label]["sha256"]):
                raise RuntimeError("Completed exam provenance mismatch")
    rows=[]
    for label in candidates:
        for kind in CASES:
            report=json.loads((root/label/kind/"unified_exam.json").read_text())
            if len(report["results"])!=(8 if kind=="speed_ladder" else 25):
                raise RuntimeError("Missing physical conditions in "+label+"/"+kind)
            rows.extend(summary_rows(report,label,kind))
    with (root/"physical_comparison.csv").open("w",newline="") as fp:
        out=csv.DictWriter(fp,fieldnames=list(rows[0]))
        out.writeheader();out.writerows(rows)
    md=["# Pinnacle physical comparison: same challenges, authentic actor sensors",
        "",f"Examiner SHA: `{source}`","",
        "Both policies ran the SAME real dual-slab contact fixtures and seeds.",
        "Archived actor is 232-input tripod/IMU/flow; released Pinnacle is 360-input 4x4.",
        "A physical challenge baseline is valid, but different actual sensory",
        "visibility is part of the capabilities being compared, not a bug.","",
        "| Exam | Condition | Policy | Achieved vx | Survived | Mission | Step UP | Step DOWN |",
        "|---|---|---|---:|---:|---:|---:|---:|"]
    for x in rows:
        md.append(f"| {x['exam']} | {x['condition']} | {x['policy']} | "
                  f"{x['mean_achieved_vx_mps']} | {x['alive']}/{x['worlds']} | "
                  f"{x['mission']}/{x['worlds']} | {x['supported_step_up']} | {x['supported_step_down']} |")
    (root/"physical_comparison.md").write_text("\n".join(md)+"\n")
    zip_path=root/"archived232_vs_pinnacle360_physical_comparison.zip"
    with zipfile.ZipFile(zip_path,"w",compression=zipfile.ZIP_DEFLATED) as z:
        for file in [mpath,root/"physical_comparison.csv",root/"physical_comparison.md"]:
            z.write(file,arcname=file.name)
        for label in candidates:
            for kind in CASES:
                for filename in ("unified_exam.json","unified_exam.md"):
                    file=root/label/kind/filename
                    z.write(file,arcname=str(file.relative_to(root)))
    print("COMPARISON:",root/"physical_comparison.md",flush=True)
    print("SHARE:",zip_path,flush=True)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
