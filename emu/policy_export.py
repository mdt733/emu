#!/usr/bin/env python3
"""Export an EMU checkpoint to the browser-native .emupolicy format.

The output is JSON despite the custom extension. It contains only deployable
actor state + observation normalization + control contract + robot XML, plus
small provenance/evaluation metadata when the checkpoint belongs to a v5013
capability-teacher tournament.

Nothing in the format requires Python at playback time.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .checkpoint_io import load_checkpoint
from .contracts import (
    ACTION_DIM,
    ACTUATED_JOINTS,
    BASE_OBS_DIM,
    BASE_OBS_FIELDS,
    CONTROL_DT,
    PITCH_CMD_MAX_RAD,
    CONTROL_SUBSTEPS,
    HISTORY_LEN,
    OBS_DIM,
    SIM_DT,
)
from .environment import TARGET_SPAN
from .contracts import ROBOT_XML, VERSION
from .ppo import PPO_HIDDEN_SIZES, PPO_STD_MAX, PPO_STD_MIN, build_obs_norm_mask

FORMAT = "emu-policy"
SCHEMA = 1
EVAL_KEYS = (
    "final_step", "broad", "screen_hard_recovery", "screen_growth_headroom",
    "recovery", "worst_direction", "calibration_mae", "competent_nodes",
    "known_nodes", "competent_edges", "known_edges", "frontier_vx_hi",
    "frontier_vx_lo",
)


def _f32_blob(a) -> dict:
    arr = np.asarray(a, dtype='<f4', order='C')
    return {
        "dtype": "float32-le",
        "shape": list(arr.shape),
        "data_b64": base64.b64encode(arr.tobytes(order='C')).decode('ascii'),
    }


def _scalar(d: dict, key: str, default: float) -> float:
    try:
        v = float(d.get(key, default))
        return v if np.isfinite(v) else float(default)
    except Exception:
        return float(default)


def _json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return default


def _find_tournament_root(checkpoint: Path, explicit: Path | None = None) -> Path | None:
    if explicit is not None:
        root = explicit.expanduser().resolve()
        return root if (root / 'tournament_summary.csv').exists() else None
    for parent in checkpoint.resolve().parents:
        if (parent / 'tournament_summary.csv').exists() and (parent / 'tournament_decisions.json').exists():
            return parent
    return None


def _number(text: Any) -> float | int | None:
    if text is None or text == '':
        return None
    try:
        x = float(text)
        if not np.isfinite(x):
            return None
        if x.is_integer():
            return int(x)
        return x
    except Exception:
        return None


def _tournament_metadata(checkpoint: Path, run_name: str, run_cfg: dict,
                         explicit_root: Path | None = None) -> tuple[dict | None, dict | None]:
    root = _find_tournament_root(checkpoint, explicit_root)
    if root is None:
        return None, None

    decisions = _json(root / 'tournament_decisions.json', {}) or {}
    rows: dict[str, dict] = {}
    try:
        with (root / 'tournament_summary.csv').open(newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                if row.get('name'):
                    rows[row['name']] = row
    except Exception:
        rows = {}

    seed = int(run_cfg.get('seed', 0) or 0)
    candidates = [run_name]
    if seed:
        candidates.insert(0, f'seed{seed}:{run_name}')
    # The tournament summary prefixes confirmation arms but not the main seed.
    match_name = next((name for name in candidates if name in rows), None)
    if match_name is None:
        match_name = next((name for name in rows if name.endswith(':' + run_name)), None)
    row = rows.get(match_name) if match_name else None

    main_seed = int(decisions.get('seed', 0) or 0)
    confirm_seed = int(decisions.get('confirm_seed', 0) or 0)
    stages = decisions.get('stages') or {}
    selected_dose = ((stages.get('A_dose') or {}).get('selected'))
    challenger = ((stages.get('final120') or {}).get('challenger'))
    confirmation = stages.get('confirmation') or {}

    if seed == confirm_seed and run_name.startswith('Z0_confirm_'):
        role = 'confirmation control'
    elif seed == confirm_seed and run_name.startswith('Z1_confirm_'):
        role = 'confirmation challenger'
    elif run_name == 'A0_shadow':
        role = 'seed control'
    elif run_name == challenger:
        role = 'final challenger'
    elif run_name == selected_dose:
        role = 'selected dose'
    else:
        role = 'tournament arm'

    evaluation = None
    if row is not None:
        evaluation = {
            'summary_name': match_name,
            'seed': seed or None,
            'role': role,
        }
        for key in EVAL_KEYS:
            evaluation[key] = _number(row.get(key))

    tournament = {
        'root_name': root.name,
        'main_seed': main_seed or None,
        'confirm_seed': confirm_seed or None,
        'selected_dose': selected_dose,
        'challenger': challenger,
        'confirmation_status': confirmation.get('status'),
        'seed733_utility_delta': _number(confirmation.get('seed733_utility_delta')),
        'confirm_utility_delta': _number(confirmation.get('confirm_utility_delta')),
    }
    return evaluation, tournament


def _training_demo_nodes(field_state: dict, limits: dict, *, count: int = 96) -> list[dict]:
    """Small, self-assessed command sampler; never advertise as held-out proof.

    These are evidenced *nodes*, not guaranteed-safe command transitions.
    Filtering requires at least three effective event samples plus a decent
    posterior success estimate. No training state enters the browser actor.
    """
    from .capability_field import CapabilityField, MIN_EFFECTIVE_EVIDENCE
    f = CapabilityField.from_state_dict(field_state)
    n, p = f.node_evidence(), f.node_competence()
    good = (n >= MIN_EFFECTIVE_EVIDENCE) & (p >= 0.70)
    ranked=[]
    for xyz in np.argwhere(good):
        ix,iy,iz = (int(x) for x in xyz)
        vx, vy, yaw = (float(v) for v in f.command_for_index(ix,iy,iz))
        if not (limits['vx_min'] <= vx <= limits['vx_max'] and
                abs(vy) <= limits['vy_abs'] and abs(yaw) <= limits['yaw_abs']):
            continue
        ranked.append((float(p[ix,iy,iz]) + 0.025 * min(float(n[ix,iy,iz]), 15),
                       {'vx': vx, 'vy': vy, 'yaw': yaw,
                        'predicted': round(float(p[ix,iy,iz]), 3),
                        'effective_events': round(float(n[ix,iy,iz]), 2)}))
    ranked.sort(key=lambda pair: pair[0], reverse=True)
    return [row for _,row in ranked[:count]]


def build_bundle(checkpoint: Path, *, tournament_root: Path | None = None,
                 label: str | None = None) -> dict:
    checkpoint = checkpoint.expanduser().resolve()
    payload = load_checkpoint(checkpoint)
    run_meta = {}
    run_meta_path = checkpoint.parent.parent / 'run_meta.json'
    if run_meta_path.exists():
        run_meta = _json(run_meta_path, {}) or {}
    run_cfg = run_meta.get('config') or {}
    params = payload['state']['params']
    pi = params['pi']
    norm = payload['obs_norm']
    meta = payload['meta']
    curriculum = payload.get('curriculum_state') or {}
    cp = curriculum.get('params') or {}

    if len(pi) != 3:
        raise ValueError(f"viewer schema 1 expects a 3-layer actor, got {len(pi)} layers")

    obs_mean = np.asarray(norm['mean'], dtype=np.float32)
    obs_std = np.sqrt(np.asarray(norm['var'], dtype=np.float32))
    if obs_mean.shape != (OBS_DIM,) or obs_std.shape != (OBS_DIM,):
        raise ValueError(f"unexpected observation normalizer shape {obs_mean.shape}/{obs_std.shape}")

    actor_layers = []
    actor_hash = hashlib.sha256()
    for layer in pi:
        w = np.asarray(layer['w'], dtype='<f4', order='C')
        b = np.asarray(layer['b'], dtype='<f4', order='C')
        actor_hash.update(w.tobytes(order='C'))
        actor_hash.update(b.tobytes(order='C'))
        actor_layers.append({"w": _f32_blob(w), "b": _f32_blob(b)})

    log_std = np.asarray(params.get('log_std', np.zeros((ACTION_DIM,), np.float32)), dtype=np.float32)
    robot_xml = ROBOT_XML.read_text(encoding='utf-8')

    vx_hi = max(0.06, _scalar(cp, 'frontier_vx_hi', 0.60))
    vx_lo = min(-0.02, _scalar(cp, 'frontier_vx_lo', -0.20))
    vy = max(0.05, _scalar(cp, 'frontier_vy', 0.22))
    yaw = max(0.20, _scalar(cp, 'frontier_yaw', 1.50))
    posture_d = np.clip(_scalar(cp, 'posture_difficulty', 1.0), 0.0, 1.0)

    run_name = str(meta.get('run_name') or checkpoint.parent.parent.name)
    evaluation, tournament = _tournament_metadata(checkpoint, run_name, run_cfg, tournament_root)
    source_prov = meta.get('source_provenance') or {}

    profile = str(run_cfg.get('attitude_profile') or 'v5014')
    configured_pitch_deg = _scalar(run_cfg, 'pitch_command_max_deg', 8.0214)
    policy_pitch_limit = (float(PITCH_CMD_MAX_RAD) if profile != 'v5015'
                          else float(np.deg2rad(np.clip(configured_pitch_deg, 0.0, 45.0))))
    command_limits = {
        'vx_min': float(vx_lo), 'vx_max': float(vx_hi),
        'vy_abs': float(vy), 'yaw_abs': float(yaw),
        'pitch_abs_rad': policy_pitch_limit,
    }
    source_version = str(source_prov.get('version') or '')
    bundle = {
        "format": FORMAT,
        "schema": SCHEMA,
        "created_by": f"EMU {VERSION}",
        "source": {
            "checkpoint": checkpoint.name,
            "step": int(meta.get('step', 0)),
            "run_name": run_name,
            "display_label": label or run_name,
            "seed": int(run_cfg.get('seed', 0) or 0),
            "checkpoint_schema": int(meta.get('checkpoint_schema', -1)),
            "version": str(source_prov.get('version') or VERSION),
            "git_commit": str(source_prov.get('git_commit') or ''),
            "robot_model_sha256": str(meta.get('robot_model_sha256') or ''),
            "actor_sha256": actor_hash.hexdigest(),
        },
        "training": {
            "attitude_profile": profile,
            "pitch_command_max_deg": configured_pitch_deg if profile == 'v5015' else 8.0214,
            "capability_field_mode": run_cfg.get('capability_field_mode'),
            "field_max_influence": _number(run_cfg.get('field_max_influence')),
            "field_teach_nodes": run_cfg.get('field_teach_nodes'),
            "field_teach_edges": run_cfg.get('field_teach_edges'),
            "field_teach_recovery": run_cfg.get('field_teach_recovery'),
            "field_evidence_from_taught": run_cfg.get('field_evidence_from_taught'),
            "field_influence_start_steps": _number(run_cfg.get('field_influence_start_steps')),
            "field_influence_full_steps": _number(run_cfg.get('field_influence_full_steps')),
            "field_influence_fade_start_steps": _number(run_cfg.get('field_influence_fade_start_steps')),
            "field_influence_zero_steps": _number(run_cfg.get('field_influence_zero_steps')),
        },
        "policy": {
            "obs_dim": OBS_DIM,
            "base_obs_dim": BASE_OBS_DIM,
            "history_len": HISTORY_LEN,
            "action_dim": ACTION_DIM,
            "hidden_sizes": list(PPO_HIDDEN_SIZES),
            "activation": "tanh",
            "output": "deterministic_mean_clipped",
            "layers": actor_layers,
            "log_std": _f32_blob(log_std),
            "std_clip": [PPO_STD_MIN, PPO_STD_MAX],
        },
        "observation": {
            "base_fields": list(BASE_OBS_FIELDS),
            "history_order": "oldest_to_newest",
            "mean": _f32_blob(obs_mean),
            "std": _f32_blob(obs_std),
            "normalize_mask": _f32_blob(build_obs_norm_mask()),
        },
        "control": {
            "sim_dt": SIM_DT,
            "control_substeps": CONTROL_SUBSTEPS,
            "control_dt": CONTROL_DT,
            "actuated_joints": list(ACTUATED_JOINTS),
            "target_span_rad": [float(x) for x in np.asarray(TARGET_SPAN)],
            "one_control_frame_actuation_delay": True,
            "neutral_root_z_m": 0.2584,
            "default_commands": {
                "vx": 0.04, "vy": 0.0, "yaw": 0.0, "height": 0.0,
                "pitch": 0.0, "roll": 0.0, "step_height": 0.012,
                "phase_rate_hz": 0.0, "sensor_scale": 0.0,
            },
            "phase_speed_gain": _scalar(run_cfg, 'phase_speed_gain', 0.75),
            "phase_speed_reference_mps": _scalar(run_cfg, 'vx_hi_max', 0.60),
            "sensor_noise_scale": 1.5,
            "phase_base_hz": 1.025,
            "phase_rate_max_hz": 2.6,
            "command_limits": {
                **command_limits,
                "height_abs_m": 0.018, "roll_abs_rad": 0.055,
                "step_height_min_m": 0.008, "step_height_max_m": 0.042,
            },
        },
        "features": {
            "pitch_command_actor_channel": True,
            "pitch_tracking_trained": profile == 'v5015' or source_version.startswith('v5014'),
            "note": ("v5015: commanded chassis attitude is separated from whole-robot support geometry, "
                     "with crouch-aware reward and structural tracking."
                     if profile == 'v5015' else
                     "v5014: gravity-relative chassis pitch is a soft operator tracking objective."),
        },
        "robot_xml": robot_xml,
    }
    bundle['demo'] = {
        'source': 'training_capability_field',
        'not_held_out': True,
        'node_candidates': _training_demo_nodes(payload['field_state'], command_limits),
    }
    if evaluation is not None:
        bundle['evaluation'] = evaluation
    if tournament is not None:
        bundle['tournament'] = tournament
    return bundle


def write_bundle(bundle: dict, out: Path, *, pretty: bool = False) -> None:
    out = out.expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(bundle, indent=2 if pretty else None,
                      separators=None if pretty else (',', ':'))
    out.write_text(text, encoding='utf-8')


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Export an EMU .pkl checkpoint for the local browser viewer")
    ap.add_argument('checkpoint', type=Path)
    ap.add_argument('output', type=Path, nargs='?')
    ap.add_argument('--pretty', action='store_true', help='indent JSON (larger file, easier to inspect)')
    ap.add_argument('--tournament-root', type=Path, help='optional tournament root containing summary/decisions')
    ap.add_argument('--label', help='human-friendly viewer label')
    ns = ap.parse_args(argv)
    out = ns.output or ns.checkpoint.with_suffix('.emupolicy')
    bundle = build_bundle(ns.checkpoint, tournament_root=ns.tournament_root, label=ns.label)
    write_bundle(bundle, out, pretty=ns.pretty)
    size = out.stat().st_size
    ev = bundle.get('evaluation') or {}
    suffix = f" broad={ev.get('broad'):.3f}" if isinstance(ev.get('broad'), (int, float)) else ''
    print(f"Wrote {out} ({size/1024:.1f} KiB) step={bundle['source']['step']:,} actor={bundle['source']['actor_sha256'][:12]}{suffix}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
