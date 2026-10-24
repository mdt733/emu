"""Current EMU checkpoint contract and persistence.

A checkpoint is a temporal rewind point, not a loose bag of compatible fields.
It owns exactly the policy/optimizer state, observation normalizer, curriculum,
training-health state, capability field, and provenance required to continue the
same training process.
"""
from __future__ import annotations

import hashlib
import os
import pickle
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from .curriculum import FIELD_VERSION
from .contracts import (ACTION_DIM, OBS_DIM, OBS_SCHEMA_LEGACY, OBS_SCHEMA_SENSOR, OBS_SCHEMA_TERRAIN,
                        SENSOR_OBS_DIM, TERRAIN_OBS_DIM, sensor_observation_contract)
from .contracts import ROBOT_XML, VERSION
from .ppo import PPO_HIDDEN_SIZES, expected_param_shapes

CHECKPOINT_SCHEMA = 7  # Historical Pitch15 pickle format, never renumber.
SENSOR_CHECKPOINT_SCHEMA = 8
TERRAIN_CHECKPOINT_SCHEMA = 9
ROBOT_MODEL_SHA256 = hashlib.sha256(ROBOT_XML.read_bytes()).hexdigest()


def source_provenance_from_env() -> dict[str, Any]:
    """Source identity supplied by the experiment runner."""
    dirty_raw = os.environ.get('EMU_GIT_DIRTY', '')
    dirty = dirty_raw.lower() in ('1', 'true', 'yes', 'dirty') if dirty_raw else None
    return {
        'version': os.environ.get('EMU_SOURCE_VERSION', VERSION),
        'git_commit': os.environ.get('EMU_GIT_COMMIT', ''),
        'git_branch': os.environ.get('EMU_GIT_BRANCH', ''),
        'git_describe': os.environ.get('EMU_GIT_DESCRIBE', ''),
        'git_dirty': dirty,
    }


def payload_train_state_is_finite(payload: dict[str, Any]) -> bool:
    try:
        import jax
        leaves = jax.tree_util.tree_leaves((payload['state']['params'], payload['state']['opt_state']))
        return all(bool(np.all(np.isfinite(np.asarray(leaf)))) for leaf in leaves)
    except Exception:
        return False


def _shape_tree(value: Any):
    if isinstance(value, dict):
        return {k: _shape_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_shape_tree(v) for v in value]
    return tuple(np.asarray(value).shape)


def _validate_policy_param_shapes(params: Any, where: str = '', *, obs_dim: int = OBS_DIM) -> None:
    expected = expected_param_shapes(obs_dim=obs_dim)
    actual = _shape_tree(params)
    if actual != expected:
        raise ValueError(
            f"checkpoint policy parameter-shape mismatch{where}: expected={expected} got={actual}"
        )


def _validate_observation_meta(meta: dict[str, Any], where: str = '',
                               *, expected_observation_schema: str = OBS_SCHEMA_LEGACY,
                               expected_sensor_contract: dict | None = None) -> int:
    """Fail closed on width, meaning, and acquisition parameters.

    A legacy 7 checkpoint has no sensor metadata; it can ONLY mean Pitch15.
    A v5016 8 checkpoint requires the full independently regenerated physical
    sensor contract and explicit caller opt-in before even inspecting weights.
    """
    schema = meta.get('observation_schema', OBS_SCHEMA_LEGACY)
    disk_schema = meta.get('checkpoint_schema', -1)
    if expected_observation_schema == OBS_SCHEMA_LEGACY:
        if disk_schema != CHECKPOINT_SCHEMA or schema != OBS_SCHEMA_LEGACY:
            raise ValueError(f"checkpoint observation schema mismatch{where}: expected legacy Pitch15 schema 7")
        if 'sensor_contract' in meta:
            raise ValueError(f"checkpoint{where} has sensor metadata inside legacy schema 7")
        return OBS_DIM
    schema_versions = {OBS_SCHEMA_SENSOR:(SENSOR_CHECKPOINT_SCHEMA,SENSOR_OBS_DIM),
                       OBS_SCHEMA_TERRAIN:(TERRAIN_CHECKPOINT_SCHEMA,TERRAIN_OBS_DIM)}
    if expected_observation_schema not in schema_versions:
        raise ValueError(f"unknown expected checkpoint observation schema {expected_observation_schema!r}")
    expected_version, expected_dim = schema_versions[expected_observation_schema]
    if disk_schema != expected_version or schema != expected_observation_schema:
        raise ValueError(f"checkpoint observation schema mismatch{where}: expected {expected_observation_schema} schema {expected_version}")
    supplied = meta.get('sensor_contract')
    if not isinstance(supplied, dict) or not isinstance(supplied.get('settings'), dict):
        raise ValueError(f"checkpoint{where} missing complete sensor contract")
    canonical = sensor_observation_contract(supplied['settings'],schema=expected_observation_schema)
    if supplied != canonical:
        raise ValueError(f"checkpoint{where} sensor contract fingerprint/geometry mismatch")
    if expected_sensor_contract is None or expected_sensor_contract != canonical:
        raise ValueError(f"checkpoint{where} sensor contract differs from requested run")
    return expected_dim


def _validate_rewind_meta(meta: dict[str, Any], where: str = '') -> None:
    for key in ('step', 'lr_scale'):
        if key not in meta:
            raise ValueError(f"checkpoint{where} missing required rewind metadata: {key}")
    try:
        step = int(meta['step'])
        lr_scale = float(meta['lr_scale'])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"checkpoint{where} has invalid rewind metadata") from exc
    if step < 0 or not np.isfinite(lr_scale) or lr_scale <= 0.0:
        raise ValueError(f"checkpoint{where} has invalid rewind metadata: step={step} lr_scale={lr_scale}")


def save_checkpoint(path: Path, state: Any, meta: dict[str, Any], obs_norm: dict[str, Any], *,
                    curriculum_state: dict[str, Any], health_state: dict[str, Any],
                    field_state: dict[str, Any], require_finite_state: bool = False) -> dict[str, float]:
    """Write one complete current-schema rewind point."""
    import jax

    path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(meta)
    _validate_rewind_meta(meta, f" at {path}")
    meta.setdefault('source_provenance', source_provenance_from_env())

    t0 = time.perf_counter()
    # PPOState also owns live MJX env state, current observations and RNG. Those
    # are process-local rollout state, not rewind state: resume intentionally
    # starts fresh episodes. Transfer/persist only the learnable PPO state.
    host_params, host_opt_state = jax.device_get((state.params, state.opt_state))
    host_state = {'params': host_params, 'opt_state': host_opt_state}
    host_norm = jax.device_get(obs_norm)
    t1 = time.perf_counter()

    if require_finite_state:
        try:
            finite = all(bool(np.all(np.isfinite(np.asarray(leaf))))
                         for leaf in jax.tree_util.tree_leaves((host_params, host_opt_state)))
        except Exception:
            finite = False
        if not finite:
            return {'device_get_s': t1 - t0, 'pickle_write_s': 0.0, 'total_s': t1 - t0,
                    'bytes': 0.0, 'saved': 0.0}

    meta.setdefault('checkpoint_schema', CHECKPOINT_SCHEMA)
    meta.setdefault('field_schema', FIELD_VERSION)
    meta.setdefault('obs_dim', int(OBS_DIM))
    meta.setdefault('action_dim', int(ACTION_DIM))
    meta.setdefault('policy_hidden_sizes', list(PPO_HIDDEN_SIZES))
    meta.setdefault('robot_model_sha256', ROBOT_MODEL_SHA256)
    meta.setdefault('created_unix', float(time.time()))
    schema = meta.get('observation_schema', OBS_SCHEMA_LEGACY)
    if schema in (OBS_SCHEMA_SENSOR,OBS_SCHEMA_TERRAIN):
        dimension = _validate_observation_meta(
            meta, f" at {path}", expected_observation_schema=schema,
            expected_sensor_contract=meta.get('sensor_contract'))
    else:
        dimension = _validate_observation_meta(meta, f" at {path}")
    if meta.get('obs_dim') != dimension or meta.get('action_dim') != ACTION_DIM:
        raise ValueError('checkpoint observation/action width mismatch at save')
    _validate_policy_param_shapes(host_params, f" at {path}", obs_dim=dimension)
    if (np.asarray(host_norm['mean']).shape != (dimension,) or
            np.asarray(host_norm['var']).shape != (dimension,)):
        raise ValueError('checkpoint normalizer width mismatch at save')
    payload = {
        'state': host_state,
        'obs_norm': host_norm,
        'curriculum_state': curriculum_state,
        'health_state': health_state,
        'field_state': field_state,
        'meta': meta,
    }
    # Never expose partially written latest.pkl to a restart or the live viewer.
    # os.replace is atomic on the destination filesystem; temporary files are
    # created beside the final path to guarantee that property.
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile('wb', dir=path.parent,
                                         prefix=f'.{path.name}.', suffix='.tmp',
                                         delete=False) as f:
            temp_path = Path(f.name)
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    t2 = time.perf_counter()
    try:
        size_bytes = float(path.stat().st_size)
    except OSError:
        size_bytes = 0.0
    return {
        'device_get_s': t1 - t0,
        'pickle_write_s': t2 - t1,
        'total_s': t2 - t0,
        'bytes': size_bytes,
        'saved': 1.0,
    }


def load_checkpoint(path: Path, *, validate: bool = True,
                    expected_observation_schema: str = OBS_SCHEMA_LEGACY,
                    expected_sensor_contract: dict | None = None) -> dict[str, Any]:
    with path.open('rb') as f:
        payload = pickle.load(f)
    return (validate_resume_payload(
        payload, path, expected_observation_schema=expected_observation_schema,
        expected_sensor_contract=expected_sensor_contract) if validate else payload)


def validate_resume_payload(payload: Any, path: Path | None = None, *,
                            expected_observation_schema: str = OBS_SCHEMA_LEGACY,
                            expected_sensor_contract: dict | None = None) -> dict[str, Any]:
    """Validate the exact-resume contract; never reinterpret a partial checkpoint."""
    where = f" at {path}" if path is not None else ""
    if not isinstance(payload, dict):
        raise ValueError(f"resume checkpoint{where} is not a dict payload")
    required = {'state', 'obs_norm', 'curriculum_state', 'health_state', 'field_state', 'meta'}
    if set(payload) != required:
        raise ValueError(f"resume checkpoint{where} payload keys mismatch: expected={sorted(required)} got={sorted(payload)}")
    meta = payload.get('meta')
    if not isinstance(meta, dict):
        raise ValueError(f"resume checkpoint{where} has no meta contract")
    obs_dim = _validate_observation_meta(
        meta, where, expected_observation_schema=expected_observation_schema,
        expected_sensor_contract=expected_sensor_contract)
    if int(meta.get('field_schema', -1)) != FIELD_VERSION:
        raise ValueError(f"field schema mismatch{where}: checkpoint={meta.get('field_schema')} current={FIELD_VERSION}")
    if meta.get('obs_dim') != obs_dim or meta.get('action_dim') != ACTION_DIM:
        raise ValueError(f"checkpoint observation/action contract mismatch{where}: checkpoint={meta.get('obs_dim')}/{meta.get('action_dim')} current={obs_dim}/{ACTION_DIM}")
    if tuple(meta.get('policy_hidden_sizes', ())) != tuple(PPO_HIDDEN_SIZES):
        raise ValueError(f"checkpoint policy architecture mismatch{where}: checkpoint={meta.get('policy_hidden_sizes')} current={list(PPO_HIDDEN_SIZES)}")
    if meta.get('robot_model_sha256') != ROBOT_MODEL_SHA256:
        raise ValueError(f"checkpoint robot.xml hash mismatch{where}: checkpoint={meta.get('robot_model_sha256')} current={ROBOT_MODEL_SHA256}")
    _validate_rewind_meta(meta, where)
    state = payload.get('state')
    if not isinstance(state, dict) or set(state) != {'params', 'opt_state'}:
        got = sorted(state) if isinstance(state, dict) else type(state).__name__
        raise ValueError(f"resume checkpoint{where} PPO state contract mismatch: {got}")
    _validate_policy_param_shapes(state['params'], where, obs_dim=obs_dim)
    norm = payload.get('obs_norm')
    if not isinstance(norm, dict) or set(norm) != {'mean', 'var', 'count'}:
        got = sorted(norm) if isinstance(norm, dict) else type(norm).__name__
        raise ValueError(f"resume checkpoint{where} observation-normalizer contract mismatch: {got}")
    try:
        mean = np.asarray(norm['mean']); var = np.asarray(norm['var']); count = float(norm['count'])
        if mean.shape != (obs_dim,) or var.shape != (obs_dim,):
            raise ValueError(f"shape={mean.shape}/{var.shape}")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(var)) or np.any(var <= 0.0) or not np.isfinite(count) or count <= 0.0:
            raise ValueError('non-finite/non-positive values')
    except Exception as exc:
        raise ValueError(f"resume checkpoint{where} invalid observation-normalizer state: {exc}") from exc
    for key in ('state', 'obs_norm', 'curriculum_state', 'health_state', 'field_state'):
        if payload.get(key) is None:
            raise ValueError(f"resume checkpoint{where} missing required {key}")
    if not isinstance(payload['curriculum_state'], dict) or not isinstance(payload['curriculum_state'].get('params'), dict):
        raise ValueError(f"resume checkpoint{where} missing curriculum parameter state")
    if not isinstance(payload['health_state'], dict):
        raise ValueError(f"resume checkpoint{where} missing training-health state")
    if not isinstance(payload['field_state'], dict):
        raise ValueError(f"resume checkpoint{where} missing capability field state")
    if not payload_train_state_is_finite(payload):
        raise ValueError(f"resume checkpoint{where} contains non-finite policy/optimiser state")
    return payload


def _snapshot_path(checkpoints_dir: Path, step: int) -> Path:
    return checkpoints_dir / f'snapshot_step_{step:012d}.pkl'


def save_rotating_snapshot(checkpoints_dir: Path, state: Any, meta: dict[str, Any],
                           obs_norm: dict[str, Any], *, curriculum_state: dict[str, Any],
                           health_state: dict[str, Any], field_state: dict[str, Any],
                           step: int, keep: int) -> dict[str, float]:
    if keep <= 0:
        return {'device_get_s': 0.0, 'pickle_write_s': 0.0, 'total_s': 0.0, 'bytes': 0.0, 'saved': 0.0}
    timing = save_checkpoint(
        _snapshot_path(checkpoints_dir, step), state, meta, obs_norm,
        curriculum_state=curriculum_state, health_state=health_state, field_state=field_state,
        require_finite_state=True)
    # Never rotate away known-good snapshots when this update failed finite checks.
    if not timing.get('saved', 0.0):
        return timing
    existing = sorted(checkpoints_dir.glob('snapshot_step_*.pkl'))
    for stale in existing[:-keep]:
        try:
            stale.unlink()
        except OSError:
            pass
    return timing


def find_newest_valid_snapshot(checkpoints_dir: Path) -> Path | None:
    for candidate in sorted(checkpoints_dir.glob('snapshot_step_*.pkl'), reverse=True):
        try:
            load_checkpoint(candidate, validate=True)
            return candidate
        except Exception:
            continue
    return None


# Resume discovery
from pathlib import Path


def resume_candidates(run_dir: Path) -> list[Path]:
    """Canonical continuation order for one run, strongest semantic file first."""
    ck = Path(run_dir) / "checkpoints"
    # Both may exist when a process was interrupted between updating latest
    # and promoting final. The most recently committed file wins.
    out = sorted([ck / "final.pkl", ck / "latest.pkl"],
                 key=lambda p: p.stat().st_mtime_ns if p.exists() else -1,
                 reverse=True)
    out.extend(sorted(ck.glob("snapshot_step_*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True))
    return [path for path in out if path.exists()]


def find_resume(run_dir: Path) -> Path | None:
    """Best checkpoint path from which to continue one run; payload is not opened."""
    candidates = resume_candidates(run_dir)
    return candidates[0] if candidates else None


def find_valid_resume(run_dir: Path, *,
                      expected_observation_schema: str = OBS_SCHEMA_LEGACY,
                      expected_sensor_contract: dict | None = None) -> Path | None:
    """Best current-schema continuation checkpoint, skipping corrupt/incompatible files."""

    for path in resume_candidates(run_dir):
        try:
            load_checkpoint(path, validate=True,
                            expected_observation_schema=expected_observation_schema,
                            expected_sensor_contract=expected_sensor_contract)
            return path
        except Exception:
            continue
    return None


def find_evaluation(root: Path) -> Path | None:
    """Best policy checkpoint for read-only viewing/evaluation across experiment arms."""
    candidates: list[tuple[int, float, Path]] = []
    for run in ("graph", "shadow", "foundation20"):
        ck = Path(root) / run / "checkpoints"
        for priority, name in enumerate(("final.pkl", "best.pkl", "latest.pkl")):
            path = ck / name
            if path.exists():
                candidates.append((priority, -path.stat().st_mtime, path))
    return min(candidates, key=lambda item: (item[0], item[1]))[2] if candidates else None


# Exact run boundaries, validated checkpoint handoffs, recovery and subprocess launch.
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


def checkpoint_step(path: Path, *,
                    expected_observation_schema: str = OBS_SCHEMA_LEGACY,
                    expected_sensor_contract: dict | None = None) -> int:
    return int(load_checkpoint(path, validate=True,
        expected_observation_schema=expected_observation_schema,
        expected_sensor_contract=expected_sensor_contract)['meta']['step'])


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
    expected_observation_schema: str = OBS_SCHEMA_LEGACY,
    expected_sensor_contract: dict | None = None,
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
        resume_path = find_valid_resume(
            run_dir, expected_observation_schema=expected_observation_schema,
            expected_sensor_contract=expected_sensor_contract)
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
        resumed_payload = load_checkpoint(
            resume_path, validate=True,
            expected_observation_schema=expected_observation_schema,
            expected_sensor_contract=expected_sensor_contract)
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
        from .curriculum import CapabilityField

        payload = load_checkpoint(
            recovery_path,
            expected_observation_schema=getattr(env, 'observation_schema', OBS_SCHEMA_LEGACY),
            expected_sensor_contract=getattr(env, 'sensor_contract', None))
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


def resolve_target_checkpoint(run_dir: Path, *, target_steps: int, envs: int, label: str,
                              expected_observation_schema: str = OBS_SCHEMA_LEGACY,
                              expected_sensor_contract: dict | None = None) -> tuple[Path | None, Path | None, int]:
    """Return (complete_final, resumable_partial, expected_update_boundary).

    Completion and resumability are deliberately separate concepts. A partial
    latest/snapshot checkpoint may continue a run, but only a current-schema
    checkpoint on the exact target update boundary counts as complete.
    """
    expected = update_boundary_at_or_after(target_steps, envs)
    raw_resume = find_resume(run_dir)
    resume = find_valid_resume(run_dir,
        expected_observation_schema=expected_observation_schema,
        expected_sensor_contract=expected_sensor_contract)
    if raw_resume is not None and resume is None:
        raise RuntimeError(
            f"{label} has checkpoints but none are valid current-schema rewind points under {run_dir/'checkpoints'}"
        )
    if resume is None:
        return None, None, expected
    step = checkpoint_step(resume,
        expected_observation_schema=expected_observation_schema,
        expected_sensor_contract=expected_sensor_contract)
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
