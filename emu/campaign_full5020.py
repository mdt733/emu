"""Fresh v5020 research: ONE source-pinned foundation -> five-way winner/refork.

This is thin orchestration over the existing validated campaign5020._stage_run
and campaign_refork. It contains no new PPO code and never alters an older
experiment root, source checkout or public showcase.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from .campaign5020 import _context, _stage_run, _report
from .campaign_refork import run as refork_run, plan as refork_plan, STAGES
from .checkpoint_io import git_provenance
from .contracts import source_repo_root
from .results_refork import write_refork_report


def preflight() -> None:
    """The real schema-9 JAX/MJX course and frozen exam contract, in a CHILD.

    Keep XLA's compiled models/dependencies out of the parent launcher: the
    actual long-run CUDA trainer and examiner each own their GPU resources.
    """
    repo=source_repo_root()
    if repo is None:
        raise RuntimeError("cannot identify source checkout for MJX preflight")
    cmd=[sys.executable,"-m","pytest","-q",
         "tests/test_v5020_coldstart.py::test_three_world_312_real_step_frozen_exams_compile"]
    print("[v5020 full] physical JAX/MJX examiner preflight...",flush=True)
    subprocess.run(cmd,cwd=repo,check=True)
    print("[v5020 full] physical examiner smoke passed",flush=True)


def plan(*, envs:int, seed:int, screen_envs:int, horizon:int, stop_after:str)->dict:
    return {"version":"v5020-full-1","foundation_steps":20_000_000,
            "foundation":"cold-start rolling_tof_312, gentle collidable slopes",
            "refork":refork_plan(envs),
            "start_seed":seed,"worlds":envs,"screen_worlds":screen_envs,
            "screen_horizon":horizon,"stop_after":stop_after,
            "single_foreground_process":True,"no_resume_recompile_after_completed_stage":True,
            "outputs":["foundation20/checkpoints/final.pkl",
                       "results.md","results_data.json","results_bundle.zip",
                       "refork/results.md","refork/results_data.json",
                       "refork/results_bundle.zip","refork/walk40/scoreboard.md"]}


def run(root:Path,*,envs:int,seed:int,screen_envs:int,horizon:int,
        device:str,stop_after:str,skip_preflight:bool=False)->None:
    if stop_after not in STAGES:
        raise ValueError(f"unsupported full campaign stop: {stop_after}")
    # Source identity and root are locked BEFORE any expensive preflight work.
    root=root.expanduser().resolve()
    source=git_provenance()
    _context(root,seed=seed,envs=envs,provenance=source)
    if not skip_preflight:
        preflight()
    parent=_stage_run(root,stage="foundation",arm=None,parent=None,
                      seed=seed,envs=envs,device=device,provenance=source)
    _report(root)
    print(f"[v5020 full] 312-input foundation VALIDATED at {parent}",flush=True)
    try:
        refork_run(root/"refork",foundation=parent,seed=seed,envs=envs,
                   screen_envs=screen_envs,horizon=horizon,
                   device=device,stop_after=stop_after)
    finally:
        # Include partial/probe evidence even after a failed or disqualified
        # generation. Reporting cannot rewrite neural weights or scores.
        try:
            for dest in write_refork_report(root/"refork"):
                print(f"[v5020 full] research evidence: {dest}",flush=True)
        except (OSError,ValueError,TypeError,KeyError,AttributeError) as exc:
            print(f"[v5020 full] WARNING: refork report generation failed: {exc}",
                  flush=True)


def main(argv=None)->int:
    p=argparse.ArgumentParser(description="EMU v5020 full 312-input 20M root + five-way winner-refork")
    p.add_argument("--root",type=Path,
                   default=Path.home()/"robot_lab"/"emu_runs"/"v5020_full_4096")
    p.add_argument("--seed",type=int,default=733)
    p.add_argument("--envs",type=int,default=4096)
    p.add_argument("--screen-envs",type=int,default=32)
    p.add_argument("--screen-horizon",type=int,default=300)
    p.add_argument("--device",choices=("cuda","cpu","auto","gpu"),default="cuda")
    p.add_argument("--stop-after",choices=STAGES,default="walk80")
    p.add_argument("--skip-preflight",action="store_true",help="only for a previously checked pinned source")
    mode=p.add_mutually_exclusive_group()
    mode.add_argument("--run",action="store_true",help="start/resume all stages in the foreground")
    mode.add_argument("--preflight",action="store_true",help="compile physical promotion exam without creating a run")
    mode.add_argument("--status",action="store_true",help="read persisted summaries without touching training")
    a=p.parse_args(argv)
    if a.status:
        context=a.root.expanduser()/"lab_context.json"
        refork=a.root.expanduser()/"refork"
        progress={}
        for stage in STAGES:
            selection=refork/stage/"selection.json"
            progress[stage]=(json.loads(selection.read_text())
                             if selection.exists() else None)
        print(json.dumps({"root":str(a.root),"source":json.loads(context.read_text())
                          if context.exists() else None,"generations":progress},
                         indent=2,sort_keys=True))
        return 0
    if not a.run and not a.preflight:
        print(json.dumps(plan(envs=a.envs,seed=a.seed,screen_envs=a.screen_envs,
                              horizon=a.screen_horizon,stop_after=a.stop_after),
                         indent=2,sort_keys=True))
        return 0
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false")
    if a.preflight:
        preflight()
        return 0
    run(a.root,envs=a.envs,seed=a.seed,screen_envs=a.screen_envs,
        horizon=a.screen_horizon,device=a.device,stop_after=a.stop_after,
        skip_preflight=a.skip_preflight)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
