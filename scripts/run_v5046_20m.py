#!/usr/bin/env python3
"""Source-pinned cold-start 20M real-MJX walking experiment, never main.

Run 'plan' before start. 'start' runs real CUDA/PPO smoke with the exact
solver, then the full production train/checkpoint/evaluator CLI. 'resume'
continues only this source/run config via --resume auto, without repeating
GPU smoke. Every mode uses its own working tree, not an old editable install.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
STEPS = 20_000_000


def git(*argv: str) -> str:
    return subprocess.check_output(["git", *argv], cwd=ROOT, text=True).strip()


def train_cmd(args, mode: str, run_dir: Path) -> list[str]:
    cmd = [
        sys.executable, "-u", "-m", "emu.cli", "train",
        "--run-dir", str(run_dir), "--steps", str(STEPS),
        "--seed", str(args.seed), "--envs", "2048",
        "--device", "cuda", "--profile", "sim2real",
        "--pitch-deg", "8", "--terrain-actor",
        "--terrain-step-max-m", ".08",
        "--unified-capabilities", "--terrain-carpet", "--carpet-prisms",
        "--carpet-transition-cap-deg", "3",
        "--foundation-steps", "10000000",
        "--foundation-mix", "main",
        "--growth-recipe", "main-mature",
        "--teaching-schedule", "terrain-early",
        "--terrain-lesson", "coherent-careful",
        "--force-program", "gentle",
        "--athletic-rehearsal-share", ".10",
        "--retention-share", ".15",
        "--academy-stream",
        "--academy-intro-steps", "60000000",
        "--academy-intro-share", ".10",
        "--academy-start-steps", "80000000",
        "--academy-ramp-steps", "20000000",
        "--academy-max-share", ".50",
        "--academy-height-cap-m", ".15",
        "--academy-course-episode-frames", "1200",
        "--fixed-eval-every-steps", "10000000",
        "--solver-iterations", str(args.solver_iterations),
        "--solver-ls-iterations", str(args.solver_ls_iterations),
    ]
    if mode in ("resume", "verify"):
        cmd += ["--resume", "auto"]
    if mode == "verify":
        cmd += ["--check-resume-only"]
    if mode == "plan":
        cmd += ["--plan"]
    return cmd


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("plan", "start", "resume", "verify"))
    p.add_argument("--seed", type=int, default=735)
    p.add_argument("--solver-iterations", type=int, default=2)
    p.add_argument("--solver-ls-iterations", type=int, default=4)
    p.add_argument("--run-dir", default="",
                   help="default ~/emu-data/runs/v5046_20m_s<seed>_solver<N>_ls<N>")
    p.add_argument("--no-gpu-gate", action="store_true",
                   help="skip pre-training 2048-world PPO smoke (start only, riskier)")
    a = p.parse_args()
    if not 0 <= a.seed < 2**31:
        p.error("seed must be in 0..2147483647")
    if not (1 <= a.solver_iterations <= 100 and
            1 <= a.solver_ls_iterations <= 100):
        p.error("both solver limits must be 1..100")
    root = (Path(a.run_dir) if a.run_dir else
            Path("~/emu-data/runs") / (
                f"v5046_20m_s{a.seed}_solver{a.solver_iterations}_"
                f"ls{a.solver_ls_iterations}"))
    root = root.expanduser().resolve()
    if root == ROOT or ROOT in root.parents or root in ROOT.parents:
        p.error("training output must live outside the source checkout")
    branch = git("branch", "--show-current")
    source_sha = git("rev-parse", "HEAD")
    if branch != "experiment/v5046-75cm-four-tile-midpoint":
        p.error("20M experiment must run on PR #35 experiment branch")
    if git("status", "--porcelain", "--untracked-files=no"):
        p.error("tracked source dirty: commit/stash changes before training")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env["EMU_UNIFIED_CONTINUOUS"] = "cold-growth-approved"
    cmd = train_cmd(a, a.mode, root)
    print(f"EMU 20M WALK TRIAL | source={source_sha} | branch={branch}",
          flush=True)
    print(f"solver={a.solver_iterations}/{a.solver_ls_iterations}; "
          f"seed={a.seed}; envs=2048; requested_steps={STEPS:,}; "
          f"foundation=10M; run_dir={root}", flush=True)
    print("Command: "+" ".join(cmd[3:]), flush=True)
    if a.mode == "plan":
        return subprocess.call(cmd, cwd=ROOT, env=env)
    if a.mode == "start":
        if root.exists() and any(root.iterdir()):
            raise RuntimeError(f"Training directory already nonempty: {root}; "
                               "never overwrite; use resume after a checkpoint")
        if not a.no_gpu_gate:
            # Exact physical trainer + solver caps, checked before any new
            # training artifact is created. Import directly from checkout.
            smoke=[
                sys.executable,"-u",str(ROOT/"scripts/smoke_gpu_carpet.py"),
                "--device","cuda","--worlds","2048","--frames","2",
                "--seed",str(a.seed),"--ppo","--prisms","--full-update",
                "--benchmark-updates","2",
                "--solver-iterations",str(a.solver_iterations),
                "--solver-ls-iterations",str(a.solver_ls_iterations)]
            print("CUDA FULL PPO GATE: "+" ".join(smoke[2:]), flush=True)
            code=subprocess.call(smoke,cwd=ROOT,env=env)
            if code:
                print("GPU gate FAILED: 20M learner NOT started", flush=True)
                return code
        # Another process cannot silently reuse the path after gate.
        if root.exists() and any(root.iterdir()):
            raise RuntimeError("Run directory changed during GPU gate: "+str(root))
    elif not (root/"training_config.json").is_file():
        raise RuntimeError("No existing 20M configuration for "+a.mode+
                           " under "+str(root))
    return subprocess.call(cmd,cwd=ROOT,env=env)


if __name__=="__main__":
    raise SystemExit(main())
