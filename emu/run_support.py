"""Stable checkpoint handoff, provenance and training subprocess runner.

Training is only configured here; all learning happens inside training.py.
Experiments and ladders should import these primitives, not old experiment
recipe modules. Validated checkpoints and console logging retain their
original semantics.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .paths import VERSION, source_repo_root
from .training_config import FOUNDATION_STEPS, ROLLOUT_STEPS, TrainingConfig
from .checkpoints import find_resume, find_valid_resume
from .checkpoint_io import load_checkpoint

LOG = f"[emu_{VERSION}]"

def update_boundary_at_or_after(target_steps: int, envs: int) -> int:
    """First complete PPO-update boundary at or beyond an environment-step target."""
    batch = int(envs) * int(ROLLOUT_STEPS)
    if batch <= 0:
        raise ValueError(f"invalid PPO batch size: envs={envs} rollout={ROLLOUT_STEPS}")
    target = int(target_steps)
    return ((target + batch - 1) // batch) * batch


def foundation_handoff_step(envs: int) -> int:
    return update_boundary_at_or_after(FOUNDATION_STEPS, envs)


def checkpoint_step(path: Path) -> int:
    return int(load_checkpoint(path, validate=True)['meta']['step'])


def promote_checkpoint(src: Path, dst: Path) -> Path:
    """Give an exact validated handoff checkpoint its canonical final.pkl name."""
    if src.resolve() == dst.resolve():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    return dst


def resolve_target_checkpoint(run_dir: Path, *, target_steps: int, envs: int, label: str) -> tuple[Path | None, Path | None, int]:
    """Return (complete_final, resumable_partial, expected_update_boundary).

    Completion and resumability are deliberately separate concepts. A partial
    latest/snapshot checkpoint may continue a run, but only a current-schema
    checkpoint on the exact target update boundary counts as complete.
    """
    expected = update_boundary_at_or_after(target_steps, envs)
    raw_resume = find_resume(run_dir)
    resume = find_valid_resume(run_dir)
    if raw_resume is not None and resume is None:
        raise RuntimeError(
            f"{label} has checkpoints but none are valid current-schema rewind points under {run_dir/'checkpoints'}"
        )
    if resume is None:
        return None, None, expected
    step = checkpoint_step(resume)
    if step > expected:
        raise RuntimeError(
            f"{label} checkpoint is beyond the expected target boundary: step={step:,} expected={expected:,}; "
            "use the original env-count/config or start a clean run"
        )
    if step == expected:
        final = promote_checkpoint(resume, run_dir / 'checkpoints' / 'final.pkl')
        return final, None, expected
    return None, resume, expected


def resolve_foundation_handoff(foundation: Path, *, envs: int) -> tuple[Path | None, Path | None, int]:
    """Foundation-specific name for the shared-ancestor completion contract."""
    return resolve_target_checkpoint(
        foundation, target_steps=FOUNDATION_STEPS, envs=envs, label='foundation'
    )


def git_provenance() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=source_repo_root(), text=True, stderr=subprocess.DEVNULL).strip()
    try:
        status = git("status", "--porcelain")
        return {
            "available": True,
            "commit": git("rev-parse", "HEAD"),
            "branch": git("branch", "--show-current"),
            "describe": git("describe", "--always", "--dirty"),
            "dirty": bool(status),
        }
    except Exception:
        return {"available": False, "commit": "", "branch": "", "describe": "", "dirty": False}



def run_training(config: TrainingConfig, *, provenance: dict[str, Any]) -> None:
    run_dir = Path(config.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "training_config.json"
    config.save(config_path)
    cmd = [sys.executable, "-c", "from emu.training import main; raise SystemExit(main())", "--config", str(config_path)]
    env = dict(os.environ)
    env.update({
        "EMU_SOURCE_VERSION": VERSION,
        "EMU_GIT_COMMIT": str(provenance.get("commit", "")),
        "EMU_GIT_BRANCH": str(provenance.get("branch", "")),
        "EMU_GIT_DESCRIBE": str(provenance.get("describe", "")),
        "EMU_GIT_DIRTY": "true" if provenance.get("dirty") else "false",
    })
    with (run_dir / "console.log").open("a", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line); log.flush()
            t = line.strip()
            if t.startswith(LOG + " step=") or "fixed-eval step=" in t or "AUTO-RECOVERY" in t or "training complete" in t or "Traceback" in t:
                print("    " + t[:260], flush=True)
        rc = proc.wait()
    if rc:
        raise RuntimeError(f"training failed for {run_dir.name}; see {run_dir/'console.log'}")


