#!/usr/bin/env python3
"""v5029: one clean 120M cold-start, 360-input speed+terrain experiment.

No ancestor transplant, no edits to main or v5028. A 20M main-style nursery
hands over to the six specialists PLUS a genuine 25% fast-flat rehearsal.
Both actual-command and actual-speed evidence are collected; physical tests
are read-only and never promote a curriculum frontier.

Plan only by default. --run tests CPU, full 2048 CUDA update, then trains.
Same source/manifest may --run --skip-smoke to resume after interruption.
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

HERE=Path(__file__).resolve().parents[1]
BRANCH="feature/v5029-speed-restoration"
SEED=737
STEPS=120_000_000
FOUNDATION=20_000_000
WORLDS=2048
REHEARSAL=.25
DEFAULT_ROOT=Path.home()/"robot_lab/emu_runs/v5029_speed_restoration_s737"
CASE=dict(name="cold120_speed_terrain",recipe="main",schedule="balanced",
          steps=STEPS,from_steps=0)


def source_guard(sha):
    if (base.git("branch","--show-current")!=BRANCH or
            base.git("rev-parse","HEAD")!=sha or
            base.git("status","--porcelain")):
        raise RuntimeError("Source-pinned campaign requires original clean branch and SHA")


def training_command(root, *, resume=False):
    cmd=[
        sys.executable,"-m","emu.cli","train",
        "--run-dir",str(root/CASE["name"]),
        "--steps",str(STEPS),"--envs",str(WORLDS),
        "--seed",str(SEED),"--device","cuda",
        "--profile","sim2real","--terrain-actor",
        "--terrain-step-max-m",".08","--unified-capabilities",
        "--foundation-steps",str(FOUNDATION),"--foundation-mix","main",
        "--teaching-schedule","balanced","--growth-recipe","main-mature",
        "--athletic-rehearsal-share",str(REHEARSAL),
        "--terrain-lesson","coherent-careful","--force-program","gentle",
        "--retention-share","0","--fixed-eval-every-steps","20000000"]
    if resume:
        cmd.extend(("--resume","auto"))
    return cmd


def check_python_source(env):
    child="""
import importlib.util, pathlib,sys
root=pathlib.Path(sys.argv[1]).resolve()
for module in ('emu.cli','emu.environment','emu.training','emu.run'):
    spec=importlib.util.find_spec(module)
    actual=pathlib.Path(spec.origin).resolve() if spec else None
    expected=root.joinpath(*module.split('.')).with_suffix('.py')
    if actual!=expected:
        raise SystemExit('WRONG EMU IMPORT '+module+': '+str(actual))
print('Pinned EMU Python source verified: '+str(root),flush=True)
"""
    subprocess.run([sys.executable,"-c",child,str(HERE)],
                   cwd=HERE,env=env,check=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run",action="store_true",help="preflight and launch foreground GPU train")
    ap.add_argument("--skip-smoke",action="store_true",
                    help="resume only after same-SHA 2048 CUDA preflight stamp")
    ap.add_argument("--root",type=Path,default=DEFAULT_ROOT)
    args=ap.parse_args()
    sha=base.git("rev-parse","HEAD")
    source_guard(sha)
    root=args.root.expanduser().resolve()
    if root==HERE or HERE in root.parents:
        ap.error("results directory must be OUTSIDE source checkout")
    manifest={
        "version":"v5029-speed-restoration-1","branch":BRANCH,"source_sha":sha,
        "seed":SEED,"envs":WORLDS,"target_steps":STEPS,
        "foundation_steps":FOUNDATION,"athletic_rehearsal_share":REHEARSAL,
        "one_shared_policy":True,"actual_cmd_thresholds_mps":[.3,.5,.8],
        "terrain":"coherent-careful","floor_half_length_m":32.,
        "no_previous_checkpoints":True}
    print("EMU v5029",BRANCH,sha,flush=True)
    print("120M cold steps; first 20M main-style gait, then six specialists "
          "plus 25% fast-flat gait rehearsal",flush=True)
    print("2048 CUDA worlds; persistent run",root,flush=True)
    print("Command:",' '.join(training_command(root)),flush=True)
    if not args.run:
        print("PLAN ONLY: append --run to start. No files written.",flush=True)
        return 0
    root.mkdir(parents=True,exist_ok=True)
    manifest_file=root/"campaign_manifest.json"
    if manifest_file.is_file():
        if json.loads(manifest_file.read_text())!=manifest:
            ap.error("existing campaign uses different source/config; choose a new --root")
    elif any(root.iterdir()):
        ap.error("refusing nonempty root with no v5029 manifest")
    else:
        manifest_file.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    old_path=os.environ.get("PYTHONPATH","")
    env={**os.environ,
         "PYTHONPATH":str(HERE)+(os.pathsep+old_path if old_path else ""),
         "XLA_PYTHON_CLIENT_PREALLOCATE":"false",
         "EMU_UNIFIED_CONTINUOUS":"cold-growth-approved"}
    check_python_source(env)
    base.execute(training_command(root)+["--plan"],
                 root/"logs/plan.log",env)
    stamp=root/"preflight_pass.json"
    def expected_stamp():
        return {"sha":sha,"cuda_worlds":WORLDS,
                "athletic_rehearsal_share":REHEARSAL,
                "floor_half_length_m":32.,"cli_plan_passed":True}
    try:
        if args.skip_smoke:
            current=json.loads(stamp.read_text()) if stamp.is_file() else None
            if current!=expected_stamp():
                ap.error("no matching CPU + real 2048 CUDA preflight; omit --skip-smoke")
        else:
            source_guard(sha)
            base.execute([sys.executable,"scripts/smoke_unified_focus.py",
                          "--device","cpu","--envs","24",
                          "--terrain-max-m",".08",
                          "--terrain-lesson","coherent-careful",
                          "--growth-recipe","main-mature",
                          "--fast-rehearsal-share",str(REHEARSAL),
                          "--force-program","gentle"],
                         root/"logs/preflight_cpu.log",env)
            source_guard(sha)
            base.execute([sys.executable,"scripts/smoke_unified_focus.py",
                          "--device","cuda","--envs",str(WORLDS),
                          "--full-update","--terrain-max-m",".08",
                          "--terrain-lesson","coherent-careful",
                          "--growth-recipe","main-mature",
                          "--fast-rehearsal-share",str(REHEARSAL),
                          "--force-program","gentle"],
                         root/"logs/preflight_2048_cuda.log",env)
            stamp.write_text(json.dumps(expected_stamp(),indent=2,sort_keys=True)+"\n")
        base.collect_runtime(root)
        source_guard(sha)
        run=root/CASE["name"]
        if run.exists() and any(run.iterdir()) and not (run/"training_config.json").is_file():
            raise RuntimeError("Incomplete/foreign run directory: "+str(run))
        base.execute(training_command(root,resume=base.has_checkpoint(run)),
                     root/"logs/train.log",env,
                     campaign=(root,[CASE]))
        ck=run/"checkpoints/final.pkl"
        if not ck.is_file():
            raise RuntimeError("Missing validated final checkpoint: "+str(ck))
        policy=root/"physical_policy.emupolicy"
        if not policy.is_file():
            base.execute([sys.executable,"-m","emu.cli","export-policy",
                          str(ck),str(policy)],root/"logs/export.log",env)
        exams=[
            ("speed_ladder",["--speed-ladder","--worlds","16",
                             "--horizon","700","--speed",".15",
                             "--ladder-speeds",".1",".2",".3",".4",
                             ".5",".6",".8","1.0"]),
            ("terrain_matrix",["--terrain-matrix","--worlds","16",
                               "--horizon","350","--speed",".15",
                               "--matrix-grades-deg","5","8","12","14.9",
                               "--matrix-heights-cm","2","2.5","3","3.5"]),
        ]
        for name,flags in exams:
            source_guard(sha)
            output=root/"physical_exams"/name
            if (output/"unified_exam.json").is_file() and (output/"unified_exam.md").is_file():
                print("Existing finished physical exam:",output,flush=True)
                continue
            if output.exists() and any(output.iterdir()):
                raise RuntimeError("Incomplete exam directory: "+str(output))
            base.execute([sys.executable,"-m","emu.cli","capability-exam",
                          "--policy",str(policy),"--output",str(output),
                          "--device","cpu",*flags],
                         root/"logs"/("exam_"+name+".log"),env)
        print("All frozen exams finished. No previous model replaced.",flush=True)
    except KeyboardInterrupt:
        print("Interrupted: restart SAME SHA using --run --skip-smoke",file=sys.stderr)
        return 130
    finally:
        base.summary(root,[CASE],sha)
        bundle=root/"overnight_results.zip"
        with zipfile.ZipFile(bundle,"a",compression=zipfile.ZIP_DEFLATED) as z:
            for name in ("speed_ladder","terrain_matrix"):
                for fn in ("unified_exam.json","unified_exam.md"):
                    path=root/"physical_exams"/name/fn
                    if path.is_file():
                        z.write(path,arcname="physical_exams/"+name+"/"+fn)
        print("CAMPAIGN BUNDLE:",bundle,flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
