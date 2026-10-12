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
import time
from pathlib import Path
from typing import Any

import numpy as np

from .capability_field import FIELD_VERSION
from .contracts import ACTION_DIM, OBS_DIM
from .paths import ROBOT_XML, VERSION
from .ppo import PPO_HIDDEN_SIZES, expected_param_shapes

CHECKPOINT_SCHEMA = 7
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


def _validate_policy_param_shapes(params: Any, where: str = '') -> None:
    expected = expected_param_shapes()
    actual = _shape_tree(params)
    if actual != expected:
        raise ValueError(
            f"checkpoint policy parameter-shape mismatch{where}: expected={expected} got={actual}"
        )


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
    payload = {
        'state': host_state,
        'obs_norm': host_norm,
        'curriculum_state': curriculum_state,
        'health_state': health_state,
        'field_state': field_state,
        'meta': meta,
    }
    with path.open('wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
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


def load_checkpoint(path: Path, *, validate: bool = True) -> dict[str, Any]:
    with path.open('rb') as f:
        payload = pickle.load(f)
    return validate_resume_payload(payload, path) if validate else payload


def validate_resume_payload(payload: Any, path: Path | None = None) -> dict[str, Any]:
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
    if int(meta.get('checkpoint_schema', -1)) != CHECKPOINT_SCHEMA:
        raise ValueError(f"checkpoint schema mismatch{where}: checkpoint={meta.get('checkpoint_schema')} current={CHECKPOINT_SCHEMA}")
    if int(meta.get('field_schema', -1)) != FIELD_VERSION:
        raise ValueError(f"field schema mismatch{where}: checkpoint={meta.get('field_schema')} current={FIELD_VERSION}")
    if meta.get('obs_dim') != OBS_DIM or meta.get('action_dim') != ACTION_DIM:
        raise ValueError(f"checkpoint observation/action contract mismatch{where}: checkpoint={meta.get('obs_dim')}/{meta.get('action_dim')} current={OBS_DIM}/{ACTION_DIM}")
    if tuple(meta.get('policy_hidden_sizes', ())) != tuple(PPO_HIDDEN_SIZES):
        raise ValueError(f"checkpoint policy architecture mismatch{where}: checkpoint={meta.get('policy_hidden_sizes')} current={list(PPO_HIDDEN_SIZES)}")
    if meta.get('robot_model_sha256') != ROBOT_MODEL_SHA256:
        raise ValueError(f"checkpoint robot.xml hash mismatch{where}: checkpoint={meta.get('robot_model_sha256')} current={ROBOT_MODEL_SHA256}")
    _validate_rewind_meta(meta, where)
    state = payload.get('state')
    if not isinstance(state, dict) or set(state) != {'params', 'opt_state'}:
        got = sorted(state) if isinstance(state, dict) else type(state).__name__
        raise ValueError(f"resume checkpoint{where} PPO state contract mismatch: {got}")
    _validate_policy_param_shapes(state['params'], where)
    norm = payload.get('obs_norm')
    if not isinstance(norm, dict) or set(norm) != {'mean', 'var', 'count'}:
        got = sorted(norm) if isinstance(norm, dict) else type(norm).__name__
        raise ValueError(f"resume checkpoint{where} observation-normalizer contract mismatch: {got}")
    try:
        mean = np.asarray(norm['mean']); var = np.asarray(norm['var']); count = float(norm['count'])
        if mean.shape != (OBS_DIM,) or var.shape != (OBS_DIM,):
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
        curriculum_state=curriculum_state, health_state=health_state, field_state=field_state)
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
