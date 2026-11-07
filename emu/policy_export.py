"""Portable schema-3 actor policy export, contract verification and normalizer.

This is separate from the HTTP server and can run read-only on CPU.
"""
from __future__ import annotations
import json
from pathlib import Path

# Browser-native actor export; never writes trainer state or PPO checkpoints.
import argparse
import base64
import csv
import hashlib
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
    OBS_SCHEMA_LEGACY, OBS_SCHEMA_SENSOR, OBS_SCHEMA_TERRAIN,
    SENSOR_OBS_DIM, SENSOR_BASE_OBS_DIM, SENSOR_OBS_FIELDS,
    TERRAIN_OBS_DIM,TERRAIN_BASE_OBS_DIM,TERRAIN_OBS_FIELDS,
    sensor_observation_contract,
    SIM_DT,
)
from .environment import TARGET_SPAN, NEUTRAL_ROOT_Z
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


def load_exported_actor_for_exam(path: Path) -> tuple[dict, dict, dict]:
    """Decode an immutable browser actor for the SAME frozen MJX examiner.

    A .emupolicy is NOT a schema-7 training checkpoint; it contains only the
    actor and observation normalizer. Refuse silent model/obs/servo changes.
    The examiner must never infer new physical parameters from a filename.
    """
    path = Path(path).expanduser().resolve()
    bundle = json.loads(path.read_text(encoding='utf-8'))
    policy, observation, control = (bundle.get('policy') or {},
                                    bundle.get('observation') or {},
                                    bundle.get('control') or {})
    source = bundle.get('source') or {}
    training = bundle.get('training') or {}
    if bundle.get('format') != FORMAT or bundle.get('schema') not in (1,2,3):
        raise ValueError('not an EMU schema-1/2/3 exported actor')
    schema = bundle['schema']
    obs_dim = TERRAIN_OBS_DIM if schema==3 else SENSOR_OBS_DIM if schema==2 else OBS_DIM
    base_dim = TERRAIN_BASE_OBS_DIM if schema==3 else SENSOR_BASE_OBS_DIM if schema==2 else BASE_OBS_DIM
    fields = TERRAIN_OBS_FIELDS if schema==3 else SENSOR_OBS_FIELDS if schema==2 else BASE_OBS_FIELDS
    actor_schema = OBS_SCHEMA_TERRAIN if schema==3 else OBS_SCHEMA_SENSOR if schema==2 else OBS_SCHEMA_LEGACY
    if schema in (2,3):
        # The public schema-2 bundle must agree with the REAL train-time
        # sensor timing, geometry and contract hash; size alone is unsafe.
        provided = observation.get('sensor_contract')
        if not isinstance(provided,dict):
            raise ValueError('missing schema-2 sensor contract')
        # Both archived 360-tripod and the new 360-array-only contracts
        # may be examined, but never interpreted as each other.
        array_only = (schema==3 and
            provided.get('sensor_model_revision') ==
                'forward_tof4x4_array_plane_fit_no_tripod_2')
        expected = sensor_observation_contract(
            provided.get('settings'),schema=actor_schema,array_only=array_only)
        if provided != expected:
            raise ValueError('schema-2 sensor contract hash/geometry mismatch')
    if (policy.get('obs_dim'), policy.get('base_obs_dim'),
        policy.get('history_len'), policy.get('action_dim')) != (
            obs_dim, base_dim, HISTORY_LEN, ACTION_DIM):
        raise ValueError('export actor observations/actions differ from the current contract')
    if (policy.get('hidden_sizes') != list(PPO_HIDDEN_SIZES) or
        policy.get('activation') != 'tanh' or
        policy.get('output') != 'deterministic_mean_clipped'):
        raise ValueError('export policy network differs from current actor')
    if (observation.get('base_fields') != list(fields) or
        observation.get('history_order') != 'oldest_to_newest'):
        raise ValueError('export policy channel names/history order differ')
    if (control.get('actuated_joints') != list(ACTUATED_JOINTS) or
        control.get('one_control_frame_actuation_delay') is not True or
        control.get('control_substeps') != CONTROL_SUBSTEPS or
        not np.isclose(control.get('control_dt', -1), CONTROL_DT, atol=1e-9) or
        not np.isclose(control.get('sim_dt', -1), SIM_DT, atol=1e-9) or
        not np.allclose(control.get('target_span_rad', []), TARGET_SPAN, atol=1e-6)):
        raise ValueError('export control/servo delivery does not match current simulator')
    xml_bytes = ROBOT_XML.read_bytes()
    model_hash = hashlib.sha256(xml_bytes).hexdigest()
    if (source.get('robot_model_sha256') != model_hash or
        bundle.get('robot_xml') != xml_bytes.decode('utf-8')):
        raise ValueError('export robot XML differs from this MJX model; refuse a false baseline')
    expected_ckpt_schema=9 if schema==3 else 8 if schema==2 else 7
    if source.get('checkpoint_schema') != expected_ckpt_schema or not isinstance(source.get('step'), int):
        raise ValueError(f'export provenance is not a known schema-{expected_ckpt_schema} training lineage')
    if training.get('attitude_profile') not in ('v5014','v5015'):
        raise ValueError('missing supported training attitude profile')
    pitch = float(training.get('pitch_command_max_deg', float('nan')))
    velocity_i=control.get('velocity_i') or {}
    if not isinstance(velocity_i,dict):
        raise ValueError('invalid velocity-I controller metadata')
    ramp_i=float(velocity_i.get('ramp_s',.4))
    mult_i=float(velocity_i.get('multiplier',1.))
    if (not np.isfinite(ramp_i) or not .1<=ramp_i<=5. or
            not np.isfinite(mult_i) or not 1.<=mult_i<=3. or
            (mult_i>1. and schema!=3)):
        raise ValueError('unsupported velocity-I controller or schema')
    phase_gain = float(control.get('phase_speed_gain', float('nan')))
    if not (np.isfinite(pitch) and 0 < pitch <= 45 and
            np.isfinite(phase_gain) and phase_gain >= 0):
        raise ValueError('missing or invalid exported task/phase configuration')

    def decode(blob, shape, name):
        if not isinstance(blob, dict) or blob.get('dtype') != 'float32-le' or blob.get('shape') != list(shape):
            raise ValueError(f'exported {name} blob shape/dtype mismatch')
        try:
            raw = base64.b64decode(blob['data_b64'], validate=True)
        except (KeyError, ValueError) as exc:
            raise ValueError(f'invalid exported {name} base64') from exc
        if len(raw) != int(np.prod(shape)) * 4:
            raise ValueError(f'exported {name} binary length mismatch')
        out = np.frombuffer(raw, dtype='<f4').reshape(shape).copy()
        if not np.isfinite(out).all():
            raise ValueError(f'nonfinite exported {name}')
        return out

    layer_sizes = (obs_dim, *PPO_HIDDEN_SIZES, ACTION_DIM)
    specs = policy.get('layers')
    if not isinstance(specs, list) or len(specs) != 3:
        raise ValueError('export must contain exactly three actor layers')
    layers = []
    actor_hash = hashlib.sha256()
    for i, spec in enumerate(specs):
        w = decode(spec.get('w'), (layer_sizes[i], layer_sizes[i+1]), f'actor[{i}].w')
        b = decode(spec.get('b'), (layer_sizes[i+1],), f'actor[{i}].b')
        actor_hash.update(w.tobytes(order='C'))
        actor_hash.update(b.tobytes(order='C'))
        layers.append({'w': w, 'b': b})
    if actor_hash.hexdigest() != source.get('actor_sha256'):
        raise ValueError('exported actor hash does not match declared source')
    mean = decode(observation.get('mean'), (obs_dim,), 'normalizer mean')
    std = decode(observation.get('std'), (obs_dim,), 'normalizer std')
    mask = decode(observation.get('normalize_mask'), (obs_dim,), 'normalizer mask')
    if np.any(std <= 0) or not np.array_equal(mask, build_obs_norm_mask(actor_schema)):
        raise ValueError('export observation standard deviations/mask incompatible')
    return {'pi': layers}, {'mean': mean, 'std': std}, {
        'source': source, 'training': training, 'control': control,
        'artifact_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'robot_model_sha256': model_hash,
        'observation_schema': actor_schema,
        'sensor_settings': dict(observation['sensor_contract']['settings']) if schema in (2,3) else {},
        'array_only': bool(schema==3 and
            observation['sensor_contract'].get('sensor_model_revision') ==
                'forward_tof4x4_array_plane_fit_no_tripod_2'),
        'obs_dim': obs_dim,
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
    # Legacy policy bundles remain schema 1. Sensor checkpoints explicitly
    # cross the versioned schema-8 gate before any browser export takes place.
    unverified = load_checkpoint(checkpoint, validate=False)
    meta_hint = unverified.get('meta') or {}
    actor_schema = meta_hint.get('observation_schema',OBS_SCHEMA_LEGACY)
    is_sensor = actor_schema in (OBS_SCHEMA_SENSOR,OBS_SCHEMA_TERRAIN)
    if is_sensor:
        contract = meta_hint.get('sensor_contract')
        payload = load_checkpoint(
            checkpoint, expected_observation_schema=actor_schema,
            expected_sensor_contract=contract)
        if actor_schema == OBS_SCHEMA_TERRAIN:
            obs_dim,base_dim,obs_fields = TERRAIN_OBS_DIM,TERRAIN_BASE_OBS_DIM,TERRAIN_OBS_FIELDS
        else:
            obs_dim,base_dim,obs_fields = SENSOR_OBS_DIM,SENSOR_BASE_OBS_DIM,SENSOR_OBS_FIELDS
    else:
        payload = load_checkpoint(checkpoint)
        contract = None
        obs_dim, base_dim = OBS_DIM, BASE_OBS_DIM
        obs_fields = BASE_OBS_FIELDS
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
    if obs_mean.shape != (obs_dim,) or obs_std.shape != (obs_dim,):
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
        "schema": 3 if actor_schema==OBS_SCHEMA_TERRAIN else 2 if is_sensor else SCHEMA,
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
            # Provenance, NOT a new actor channel. A frozen policy should
            # identify the actual collision engine on which it trained.
            # Missing keys in historical bundles preserve old semantics.
            "terrain_carpet": bool(run_cfg.get('terrain_carpet', False)),
            "terrain_carpet_prisms": bool(run_cfg.get('terrain_carpet_prisms', False)),
            # Physical solver identity must travel WITH the frozen actor;
            # the independent exam must not silently revert to XML defaults.
            "solver_iterations": run_cfg.get('solver_iterations'),
            "solver_ls_iterations": run_cfg.get('solver_ls_iterations'),
            "stillness_birth_frac": _scalar(run_cfg, "stillness_birth_frac", 0.),
            # v5051 phase and teaching settings are *experiment provenance*.
            # Phase clock dynamics must also be reproduced at inference.
            "science_phase_lowcmd_scale": _scalar(
                run_cfg, "science_phase_lowcmd_scale", 1.),
            "science_gate_mode": run_cfg.get("science_gate_mode","legacy"),
            "science_phase_sync_strength": _scalar(
                run_cfg, "science_phase_sync_strength",1.),
            "science_crouch_allowance_m": _scalar(
                run_cfg, "science_crouch_allowance_m",0.),
            "science_stride_efficiency_weight": _scalar(
                run_cfg, "science_stride_efficiency_weight",0.),
            "science_supported_cross_bonus": _scalar(
                run_cfg, "science_supported_cross_bonus",0.),
            "science_foothold_bonus": _scalar(
                run_cfg, "science_foothold_bonus",0.),
            "science_bilateral_support_bonus": _scalar(
                run_cfg, "science_bilateral_support_bonus",0.),
            "science_sensor_teacher_bonus": _scalar(
                run_cfg, "science_sensor_teacher_bonus",0.),
            "science_followthrough_bonus": _scalar(
                run_cfg, "science_followthrough_bonus",0.),
            # v5052 deployment provenance; 360 actor remains unchanged.
            "science_step_goal_weight": _scalar(
                run_cfg, "science_step_goal_weight",0.),
            "science_step_velocity_relief": _scalar(
                run_cfg, "science_step_velocity_relief",0.),
            "science_step_approach_m": _scalar(
                run_cfg, "science_step_approach_m",0.),
            "carpet_transition_cap_deg": _scalar(
                run_cfg, 'carpet_transition_cap_deg', 3.0),
            "academy_stream": bool(run_cfg.get('academy_stream', False)),
            "academy_height_cap_m": _scalar(
                run_cfg, 'academy_height_cap_m', .150),
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
            "obs_dim": obs_dim,
            "base_obs_dim": base_dim,
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
            "base_fields": list(obs_fields),
            "sensor_contract": contract if is_sensor else None,
            "history_order": "oldest_to_newest",
            "mean": _f32_blob(obs_mean),
            "std": _f32_blob(obs_std),
            "normalize_mask": _f32_blob(build_obs_norm_mask(
                actor_schema)),
        },
        "control": {
            "sim_dt": SIM_DT,
            "control_substeps": CONTROL_SUBSTEPS,
            "control_dt": CONTROL_DT,
            "actuated_joints": list(ACTUATED_JOINTS),
            "target_span_rad": [float(x) for x in np.asarray(TARGET_SPAN)],
            "one_control_frame_actuation_delay": True,
            "neutral_root_z_m": NEUTRAL_ROOT_Z,
            "default_commands": {
                "vx": 0.04, "vy": 0.0, "yaw": 0.0, "height": 0.0,
                "pitch": 0.0, "roll": 0.0, "step_height": 0.012,
                "phase_rate_hz": 0.0, "sensor_scale": 0.0,
            },
            "phase_speed_gain": _scalar(run_cfg, 'phase_speed_gain', 0.75),
            "velocity_i": {
                "ramp_s": _scalar(run_cfg,'velocity_i_ramp_s',.4),
                "multiplier": _scalar(run_cfg,'velocity_i_multiplier',1.),
                "algorithm": "sensor_relative_shortfall_bounded_v1",
            },
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


def export_main(argv=None) -> int:
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
