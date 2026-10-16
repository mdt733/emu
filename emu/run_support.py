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
from typing import Any, Callable
from dataclasses import dataclass

from .contracts import VERSION, source_repo_root
from .training_config import FOUNDATION_STEPS, ROLLOUT_STEPS, TrainingConfig
from .checkpoint_io import find_resume, find_valid_resume
from .checkpoint_io import load_checkpoint, payload_train_state_is_finite

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
    if is_auto:
        # A power loss can remove or corrupt latest.pkl while an existing
        # final or rotating snapshot remains valid. Use the SAME canonical
        # discovery order as the experiment launcher.
        preferred = find_resume(run_dir)
        if preferred is None:
            raise RuntimeError(f'--resume auto found no checkpoint under {run_dir / "checkpoints"}')
        resume_path = find_valid_resume(run_dir)
        if resume_path is None:
            raise RuntimeError(
                f'--resume auto found checkpoints but no valid current-schema rewind point under '
                f'{run_dir / "checkpoints"}')
        if resume_path != preferred:
            print(f'{LOG} --resume: preferred {preferred.name} unusable; '
                  f'falling back to validated {resume_path}', flush=True)
    else:
        resume_path = Path(resume)
        if not resume_path.exists():
            raise RuntimeError(f'--resume requested but checkpoint does not exist: {resume_path}')
    try:
        resumed_payload = load_checkpoint(resume_path, validate=True)
        resumed_step = int(resumed_payload['meta']['step'])
        print(f'{LOG} --resume: loaded exact checkpoint {resume_path} (step={resumed_step:,})', flush=True)
        if finish_if_target_reached(run_dir, resume_path, resumed_step, target_steps, envs):
            print(f'{LOG} already complete at step={resumed_step:,}; '
                  'published final.pkl without any new PPO update', flush=True)
            return resumed_payload, resumed_step, True
    except Exception as exc:
        raise RuntimeError(f'--resume failed for {resume_path}: {exc}') from exc

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


@dataclass(frozen=True)
class RecoveredLearner:
    """One successful temporal rewind; no partial result escapes to training."""
    state: Any
    field: Any
    obs_mean: Any
    obs_var: Any
    obs_count: float
    lr_scale: float


def try_restore_recovery(
    recovery_path: Path, *, state: Any, governor: Any,
    health: Any, env: Any, step: int, report: dict[str, Any],
    obs_mean: Any, obs_var: Any, obs_count: float,
    task_params_for_field: Callable[[Any], dict[str, Any]],
    log_prefix: str,
) -> RecoveredLearner | None:
    """Restore EVERY learning coordinate and reset the physical worlds.

    Failed/missing/nonfinite candidates leave the collapse streak intact.
    A successful restore uses fold_in(env.key0, absolute_step), deliberately
    avoiding both the collapsed episode buffers and the original reset luck.
    Only the 6 explicit values returned here are rebound by training.py.
    """
    if not recovery_path.exists():
        print(f"{log_prefix} AUTO-RECOVERY triggered after {int(report['stepping_collapse_streak'])} "
              "collapsed updates, but recovery.pkl does not exist yet -- curriculum retreat continues.", flush=True)
        return None

    # Restoring physical worlds can fail after the teacher was rewound.
    # Keep snapshots so a failed recovery cannot leave a mixed learner.
    old_governor = old_health = None
    try:
        import jax
        import jax.numpy as jnp
        from .capability_field import CapabilityField

        payload = load_checkpoint(recovery_path)
        if not payload_train_state_is_finite(payload):
            print(f"{log_prefix} AUTO-RECOVERY triggered but {recovery_path.name} contains non-finite "
                  "params -- skipping recovery, continuing with ordinary retreat only.", flush=True)
            return None

        saved = payload['state']
        restored_state = state._replace(
            params=jax.device_put(saved['params']),
            opt_state=jax.device_put(saved['opt_state']))
        restored_field = CapabilityField.from_state_dict(payload['field_state'])
        old_governor, old_health = governor.state_dict(), health.state_dict()
        governor.load_state_dict(payload['curriculum_state'])
        health.load_state_dict(payload['health_state'])
        if payload.get('obs_norm') is not None:
            obs_mean = jnp.asarray(payload['obs_norm']['mean'], dtype=jnp.float32)
            obs_var = jnp.asarray(payload['obs_norm']['var'], dtype=jnp.float32)
            obs_count = float(payload['obs_norm']['count'])
        lr_scale = float(payload['meta']['lr_scale'])

        # The restored field, not the stale collapsed field, must choose the
        # reset task. Keep JAX RNG derivation and reset order unchanged.
        fresh_key = jax.random.fold_in(env.key0, step)
        fresh_params = env.params_array(task_params_for_field(restored_field))
        fresh_env_state = env.reset_jit(fresh_key, fresh_params)
        restored_state = restored_state._replace(
            env_state=fresh_env_state, obs=env.obs(fresh_env_state))
        best_step = payload.get('meta', {}).get('step', '?')
        print(f"{log_prefix} AUTO-RECOVERY: stepping-collapse sustained for "
              f"{int(report['stepping_collapse_streak'])} updates despite repeated retreats -- "
              f"rolled back network + curriculum to the {payload.get('meta', {}).get('recovery_tier', 'recovery')} snapshot from "
              f"step={best_step} (competence={payload.get('meta', {}).get('competence', '?')}). "
              f"Every env RESET with a fresh RNG derived from step={step:,} (not the run's "
              "original seed) -- a genuinely different attempt, not a replay of the same luck "
              "that led here, and no env inherits whatever physical state the collapsed policy "
              "left it in.", flush=True)
        health.note_recovery_performed()
        return RecoveredLearner(
            restored_state, restored_field, obs_mean, obs_var,
            obs_count, lr_scale)
    except Exception as exc:
        if old_governor is not None:
            try:
                governor.load_state_dict(old_governor)
                health.load_state_dict(old_health)
            except Exception as undo_exc:
                # Continuing would mix the new teacher with old PPO weights.
                raise RuntimeError(
                    'AUTO-RECOVERY failed and could not undo curriculum/health rewind; '
                    'stopping to prevent inconsistent learning state') from undo_exc
        print(f"{log_prefix} AUTO-RECOVERY triggered but failed to load {recovery_path.name} ({exc!r}) "
              "-- continuing with ordinary retreat only.", flush=True)
        return None


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


