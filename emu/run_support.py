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
import uuid
from pathlib import Path
from typing import Any

from .paths import VERSION, source_repo_root
from .training_config import FOUNDATION_STEPS, ROLLOUT_STEPS, TrainingConfig
from .checkpoints import find_resume, find_valid_resume
from .checkpoint_io import (load_checkpoint, validate_resume_payload, find_newest_valid_snapshot)

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
    temporary = dst.with_name(f'.{dst.name}.{uuid.uuid4().hex}.tmp')
    try:
        try:
            os.link(src, temporary)
        except OSError:
            shutil.copy2(src, temporary)
        os.replace(temporary, dst)
    finally:
        temporary.unlink(missing_ok=True)
    return dst


def finish_if_target_reached(
    run_dir: Path, resume_path: Path, resumed_step: int,
    target_steps: int, envs: int,
) -> bool:
    """Never execute another PPO update after a completed resumed run."""
    expected = update_boundary_at_or_after(target_steps, envs)
    if resumed_step > expected:
        raise RuntimeError(f'resumed step {resumed_step:,} exceeds target boundary {expected:,}')
    if resumed_step < expected:
        return False
    ck_dir = run_dir.resolve() / 'checkpoints'
    if resume_path.resolve().parent != ck_dir:
        raise RuntimeError('already-complete checkpoint must belong to this run directory')
    promote_checkpoint(resume_path, ck_dir / 'final.pkl')
    return True


def resume_training_lineage(
    run_dir: Path, *, resume: str, target_steps: int, envs: int,
) -> tuple[dict[str, Any] | None, int, bool]:
    """Load exact resume state; identify completed runs without a new PPO update.

    Only auto-resume may fall back to a valid rotating snapshot. External forks
    inherit a recovery snapshot only when their own run has none. MJX physical
    episodes restart separately; this restores *learning* state.
    """
    if not resume:
        return None, 0, False
    is_auto = resume.lower() == 'auto'
    resume_path = (run_dir / 'checkpoints' / 'latest.pkl') if is_auto else Path(resume)
    if not resume_path.exists():
        raise RuntimeError(f"--resume requested but checkpoint does not exist: {resume_path}")
    try:
        resumed_payload = load_checkpoint(resume_path, validate=False)
        try:
            validate_resume_payload(resumed_payload, resume_path)
        except Exception as primary_exc:
            if not is_auto:
                raise
            fallback = find_newest_valid_snapshot(run_dir / 'checkpoints')
            if fallback is None:
                raise RuntimeError(f"primary resume checkpoint invalid ({primary_exc}); no rotating snapshot is available") from primary_exc
            resumed_payload = load_checkpoint(fallback)
            resume_path = fallback
            print(f"{LOG} --resume: primary checkpoint invalid; falling back to validated rotating snapshot {fallback}", flush=True)
        resumed_step = int(resumed_payload['meta']['step'])
        print(f"{LOG} --resume: loaded exact checkpoint {resume_path} (step={resumed_step:,})", flush=True)
        if finish_if_target_reached(run_dir, resume_path, resumed_step, target_steps, envs):
            print(f"{LOG} already complete at step={resumed_step:,}; "
                  "published final.pkl without any new PPO update", flush=True)
            return resumed_payload, resumed_step, True
    except Exception as exc:
        raise RuntimeError(f"--resume failed for {resume_path}: {exc}") from exc

    # Preserve stronger run-local rollback candidates across external forks.
    src_ckpt_dir = resume_path.parent
    dst_ckpt_dir = run_dir / 'checkpoints'
    if src_ckpt_dir.resolve() != dst_ckpt_dir.resolve():
        src_recovery = src_ckpt_dir / 'recovery.pkl'
        dst_recovery = dst_ckpt_dir / 'recovery.pkl'
        if src_recovery.exists() and not dst_recovery.exists():
            dst_ckpt_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_recovery, dst_recovery)
            print(f"{LOG} --resume: seeded recovery.pkl from the source run ({src_recovery}).", flush=True)
    return resumed_payload, resumed_step, False


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


