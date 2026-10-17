"""Build and serve EMU's browser-local MuJoCo viewer.

Optional loopback-only API serves validated, immutable policy snapshots. The
trainer never communicates with the browser; physics and inference stay local.
"""
from __future__ import annotations

import http.server
import json
import mimetypes
import os
import shutil
import socketserver
import subprocess
import webbrowser
from pathlib import Path
from urllib.parse import urlsplit



# Read-only live checkpoint watcher, bound to the same local viewer server.
from dataclasses import dataclass
import json
from pathlib import Path
from threading import Lock


@dataclass(frozen=True)
class Snapshot:
    path: Path
    signature: tuple[str, int, int]
    phase: str


def _run_priority(relative_parts: tuple[str, ...]) -> tuple[int, str]:
    """Later ladder rounds outrank their parent, independent of runner version."""
    first = relative_parts[0] if relative_parts else ''
    if first.startswith('round') and first[5:].isdigit():
        return (1000 + int(first[5:]), '/'.join(relative_parts))
    if first == 'foundation20':
        return (500, '/'.join(relative_parts))
    # Historical 00_foundation/10_adapt/40_generalist directories, or any
    # user-defined numbered stage. No dependency on a retired forge recipe.
    if len(first) >= 3 and first[:2].isdigit() and first[2] == '_':
        return (int(first[:2]) + 500, '/'.join(relative_parts))
    return (100, '/'.join(relative_parts))


def candidates(root: Path) -> list[tuple[str, Path]]:
    """Newest logical ladder stages first, regardless of recipe filenames.

    Only existing checkpoint directories are visited; this also recognises
    arbitrary arm names and history without importing any training runners.
    """
    root = Path(root)
    dirs = [p for p in root.rglob('checkpoints') if p.is_dir()]
    dirs.sort(key=lambda ck: _run_priority(ck.parent.relative_to(root).parts),
              reverse=True)
    result: list[tuple[str, Path]] = []
    for ck in dirs:
        rel = ck.parent.relative_to(root)
        phase = '/'.join(rel.parts) if rel.parts else 'training'
        result.extend(((phase, ck / 'latest.pkl'), (phase, ck / 'final.pkl')))
        result.extend((phase, p) for p in sorted(
            ck.glob('snapshot_step_*.pkl'), reverse=True))
    result.append(('complete', root / 'final.pkl'))
    return result


def inspect_candidate(phase: str, path: Path) -> Snapshot | None:
    try:
        stat = path.stat()
        if not path.is_file() or stat.st_size < 1024:
            return None
        return Snapshot(path, (str(path.resolve()), stat.st_size, stat.st_mtime_ns), phase)
    except OSError:
        return None


class LivePolicyWatcher:
    """Cache policy JSON by completed-checkpoint signature, not by request."""
    def __init__(self, root: Path):
        self.root = Path(root).expanduser().resolve()
        self._lock = Lock()
        self._signature: tuple | None = None
        self._text: str | None = None
        self._revision: str | None = None
        self._phase: str | None = None
        self._step: int | None = None
        self._error = ''

    def _refresh(self) -> None:
        # Avoid pickle loads on each browser poll. Failed writes can be retried
        # next poll; last validated actor stays available throughout.
        with self._lock:
            seen = set()
            for phase, path in candidates(self.root):
                snap = inspect_candidate(phase, path)
                if snap is None or snap.signature in seen:
                    continue
                seen.add(snap.signature)
                if snap.signature == self._signature:
                    self._error = ''
                    return
                try:
                    bundle = build_bundle(path)
                    # Ensure a writer didn't replace the file while exporting.
                    if inspect_candidate(phase, path) != snap:
                        self._error = 'Checkpoint changed while being read; retrying.'
                        return
                    bundle.setdefault('source', {})['phase'] = phase
                    body = json.dumps(bundle, separators=(',', ':'))
                    source = bundle['source']
                    self._signature = snap.signature
                    self._text = body
                    self._phase = phase
                    self._step = int(source['step'])
                    self._revision = f"{source['step']}:{source['actor_sha256']}"
                    self._error = ''
                    return
                except Exception as exc:
                    self._error = f'Newest candidate not usable yet ({type(exc).__name__}); retaining last valid policy.'
                    # Do not serve an older phase as new while a newer checkpoint
                    # is being written, but allow fallback at initial startup.
                    if self._text is not None:
                        return
            if self._text is None and not self._error:
                self._error = 'No validated checkpoints yet. Viewer remains ready.'

    def status(self) -> dict:
        self._refresh()
        return {'enabled': True, 'available': self._text is not None,
                'revision': self._revision, 'step': self._step,
                'phase': self._phase, 'note': self._error}

    def policy(self) -> str | None:
        self._refresh()
        return self._text

def viewer_source() -> Path:
    packaged = Path(__file__).resolve().parent / 'viewer_src'
    if (packaged / 'package.json').exists():
        return packaged
    raise RuntimeError('No browser viewer source included in this EMU installation')


def viewer_dist() -> Path:
    packaged = Path(__file__).resolve().parent / 'webviewer_dist'
    return packaged if packaged.exists() else viewer_source() / 'dist'


def build() -> int:
    src = viewer_source()
    if not (src / 'package.json').exists():
        raise RuntimeError(f'viewer source not found: {src}')
    if shutil.which('npm') is None:
        raise RuntimeError('Node/npm are required to build the viewer')
    # Canonical build MUST consume the committed lockfile. npm install may
    # silently resolve newer dependency versions between Git clones.
    lock = src / 'package-lock.json'
    if not lock.is_file():
        raise RuntimeError(
            f'Viewer lockfile missing: {lock}. In emu/viewer_src/, run npm install '
            'to generate package-lock.json, verify the build and commit the '
            'lockfile before distributing this Git checkout.'
        )
    subprocess.check_call(['npm', 'ci', '--no-audit', '--no-fund'], cwd=src)
    subprocess.check_call(['npm', 'run', 'build'], cwd=src)
    dist = src / 'dist'
    # This single package-owned source is never regenerated from a mirror.
    packaged = Path(__file__).resolve().parent / 'webviewer_dist'
    if packaged.exists():
        shutil.rmtree(packaged)
    shutil.copytree(dist, packaged)
    print(f'viewer built -> {packaged}')
    return 0


def make_handler(dist: Path, *, watcher=None, policy: Path | None = None):
    """Fixed local API, no user-supplied filesystem paths or directory browsing."""
    mimetypes.add_type('application/wasm', '.wasm')
    # A manually selected brain stays fixed for this entire viewer session.
    # Only the separate live watcher selects new validated policy snapshots.
    initial_policy = policy.read_bytes() if policy is not None else None

    class Handler(http.server.SimpleHTTPRequestHandler):
        extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                          '.wasm': 'application/wasm'}

        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(dist), **kw)

        def _send(self, content: bytes, mime: str, *, status: int = 200):
            self.send_response(status)
            self.send_header('Content-Type', mime)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Length', str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def _json(self, data: dict, *, status=200):
            self._send(json.dumps(data).encode('utf-8'), 'application/json; charset=utf-8', status=status)

        def do_GET(self):
            route = urlsplit(self.path).path
            if route == '/api/viewer-config':
                return self._json({'live': watcher is not None,
                                   'initial_policy': initial_policy is not None})
            if route == '/api/live/status':
                if watcher is None:
                    return self._json({'enabled': False}, status=404)
                return self._json(watcher.status())
            if route == '/api/live/policy':
                if watcher is None:
                    return self._json({'error': 'live mode disabled'}, status=404)
                body = watcher.policy()
                if body is None:
                    return self._json({'error': 'no complete policy yet'}, status=404)
                return self._send(body.encode('utf-8'), 'application/json; charset=utf-8')
            if route == '/api/initial-policy':
                if initial_policy is None:
                    return self._json({'error': 'no initial policy'}, status=404)
                return self._send(initial_policy, 'application/json; charset=utf-8')
            return super().do_GET()

        def list_directory(self, path):
            self.send_error(403, 'Directory listings disabled')
            return None
    return Handler


def serve(*, port: int = 8765, policy: Path | None = None,
          live_root: Path | None = None, open_browser: bool = True) -> int:
    dist = viewer_dist()
    if not (dist / 'index.html').exists():
        print('Viewer build missing; invoking npm build...', flush=True)
        build()
        dist = viewer_dist()
    # The viewer is a separate read-only CPU process: do not let checkpoint
    # validation initialize or preallocate the trainer's CUDA accelerator.
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    watcher = LivePolicyWatcher(live_root) if live_root is not None else None
    policy = policy.expanduser().resolve() if policy is not None else None
    if policy is not None and not policy.is_file():
        raise FileNotFoundError(f'Policy not found: {policy}')
    # Loopback deliberately: this endpoint contains training-derived weights and
    # has no authentication. Use a separate secured service for remote clients.
    handler = make_handler(dist, watcher=watcher, policy=policy)
    class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True
        allow_reuse_address = True
    with Server(('127.0.0.1', int(port)), handler) as httpd:
        url = f'http://127.0.0.1:{port}/'
        print(f'EMU viewer: {url}', flush=True)
        if watcher:
            print(f'Live viewer watching completed checkpoints under {watcher.root}', flush=True)
        print('Training stays on GPU; browser runs the robot and neural inference.', flush=True)
        if open_browser:
            try: webbrowser.open(url)
            except Exception: pass
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


# Browser-native actor export; never writes trainer state or PPO checkpoints.
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
    if bundle.get('format') != FORMAT or bundle.get('schema') != SCHEMA:
        raise ValueError('not an EMU schema-1 exported actor')
    if (policy.get('obs_dim'), policy.get('base_obs_dim'),
        policy.get('history_len'), policy.get('action_dim')) != (
            OBS_DIM, BASE_OBS_DIM, HISTORY_LEN, ACTION_DIM):
        raise ValueError('export actor observations/actions differ from the current contract')
    if (policy.get('hidden_sizes') != list(PPO_HIDDEN_SIZES) or
        policy.get('activation') != 'tanh' or
        policy.get('output') != 'deterministic_mean_clipped'):
        raise ValueError('export policy network differs from current actor')
    if (observation.get('base_fields') != list(BASE_OBS_FIELDS) or
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
    if source.get('checkpoint_schema') != 7 or not isinstance(source.get('step'), int):
        raise ValueError('export provenance is not a known schema-7 training lineage')
    if training.get('attitude_profile') not in ('v5014','v5015'):
        raise ValueError('missing supported training attitude profile')
    pitch = float(training.get('pitch_command_max_deg', float('nan')))
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

    layer_sizes = (OBS_DIM, *PPO_HIDDEN_SIZES, ACTION_DIM)
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
    mean = decode(observation.get('mean'), (OBS_DIM,), 'normalizer mean')
    std = decode(observation.get('std'), (OBS_DIM,), 'normalizer std')
    mask = decode(observation.get('normalize_mask'), (OBS_DIM,), 'normalizer mask')
    if np.any(std <= 0) or not np.array_equal(mask, build_obs_norm_mask()):
        raise ValueError('export observation standard deviations/mask incompatible')
    return {'pi': layers}, {'mean': mean, 'std': std}, {
        'source': source, 'training': training, 'control': control,
        'artifact_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'robot_model_sha256': model_hash,
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
    from .curriculum import CapabilityField, MIN_EFFECTIVE_EVIDENCE
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


# Local read-only capability-field viewer, separate from the trained actor feed.
import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from .contracts import seed_root

from .curriculum import FIELD_VERSION
from .contracts import VERSION

HTML = r'''<!doctype html><meta charset="utf-8"><title>EMU capability field</title>
<style>
:root{color-scheme:dark}body{font-family:system-ui,sans-serif;margin:20px;background:#111;color:#ddd;max-width:1180px}.bar{display:flex;gap:14px;align-items:center;flex-wrap:wrap}label{font-size:14px}select,input{font-size:16px}canvas{margin-top:14px;border:1px solid #555;max-width:100%;image-rendering:pixelated;background:#000;cursor:crosshair}.muted{color:#999;font-size:13px}.stats{font-family:ui-monospace,monospace;white-space:pre-wrap;margin-top:10px;line-height:1.45}.run{margin:8px 0;color:#bbb;font-family:ui-monospace,monospace;font-size:13px}.grid{display:grid;grid-template-columns:minmax(0,2fr) minmax(250px,1fr);gap:18px}@media(max-width:800px){.grid{grid-template-columns:1fr}}.legend{display:flex;gap:10px;align-items:center;font-size:12px;color:#aaa;margin-top:8px}.sw{width:80px;height:10px;background:linear-gradient(90deg,#7a1414,#d8a52c,#32b86b)}
</style>
<h2>EMU live capability field</h2>
<div class="run" id="run">waiting for active field…</div>
<div class="bar">
<label>Layer <select id="layer">
<option value="node_competence">command competence</option>
<option value="node_lower">conservative competence</option>
<option value="node_confidence">confidence</option>
<option value="node_quality">command quality</option>
<option value="node_learning">learning / forgetting</option>
<option value="node_staleness">staleness</option>
<option value="node_evidence">effective trials</option>
<option value="recovery_frontier">integrated recovery frontier</option>
<option value="edge_coverage">maneuver coverage</option>
<option value="edge_competence">maneuver competence</option>
<option value="edge_rate">maneuver rate</option>
</select></label>
<label>Yaw slice <input id="yaw" type="range" min="0" max="24" value="12"><span id="yawLabel"></span></label>
<label><input id="auto" type="checkbox" checked> auto refresh</label>
</div>
<div class="legend"><span>low</span><span class="sw"></span><span>high</span><span>dark = unknown</span></div>
<div class="grid"><div><canvas id="c" width="840" height="600"></canvas></div><div><div class="stats" id="picked">click a voxel for details</div></div></div>
<div class="stats" id="stats"></div>
<div class="muted">x = vx, y = vy, slider = yaw rate. Held-out evaluator outcomes are deliberately not painted back into this training-derived map.</div>
<script>
const c=document.getElementById('c'),ctx=c.getContext('2d'),layer=document.getElementById('layer'),yaw=document.getElementById('yaw'),yl=document.getElementById('yawLabel'),stats=document.getElementById('stats'),picked=document.getElementById('picked'),runEl=document.getElementById('run');let D=null,SEL=null;
function clamp(v,a=0,b=1){return Math.max(a,Math.min(b,v))}
function heat(v,known=true){if(!known)return '#070707';v=clamp(v);let r,g,b;if(v<.5){r=170+Math.round(120*v);g=35+Math.round(330*v);b=35}else{r=290-Math.round(420*(v-.5));g=200+Math.round(80*(v-.5));b=55+Math.round(60*(v-.5))}return `rgb(${clamp(r,0,255)},${clamp(g,0,255)},${clamp(b,0,255)})`}
function diverge(v){v=clamp((v+1)/2);return heat(v,true)}
function at(a,ix,iy,iz){const ny=D.vy.length,nz=D.yaw.length;return a[(ix*ny+iy)*nz+iz]}
function draw(){if(!D||D.waiting)return;const zi=clamp(+yaw.value,0,D.yaw.length-1),nx=D.vx.length,ny=D.vy.length,w=c.width/nx,h=c.height/ny,L=layer.value;yl.textContent=` ${D.yaw[zi].toFixed(2)} rad/s`;ctx.clearRect(0,0,c.width,c.height);let max=Math.max(1,D.max_node_evidence||1);if(L==='edge_rate')max=Math.max(.01,D.max_edge_rate||.01);for(let ix=0;ix<nx;ix++)for(let iy=0;iy<ny;iy++){let v=0,known=true,arr=null;if(L==='node_competence'){v=at(D.node_competence,ix,iy,zi);known=at(D.node_evidence,ix,iy,zi)>=D.min_evidence}else if(L==='node_lower'){v=at(D.node_lower,ix,iy,zi);known=at(D.node_evidence,ix,iy,zi)>=D.min_evidence}else if(L==='node_confidence'){v=at(D.node_confidence,ix,iy,zi);known=at(D.node_evidence,ix,iy,zi)>0}else if(L==='node_quality'){v=at(D.node_quality,ix,iy,zi);known=at(D.node_evidence,ix,iy,zi)>0}else if(L==='node_learning'){v=at(D.node_learning,ix,iy,zi);known=at(D.node_evidence,ix,iy,zi)>0}else if(L==='node_staleness'){v=at(D.node_staleness,ix,iy,zi);known=at(D.node_evidence,ix,iy,zi)>0}else if(L==='node_evidence'){v=Math.log1p(at(D.node_evidence,ix,iy,zi))/Math.log1p(max);known=at(D.node_evidence,ix,iy,zi)>0}else if(L==='recovery_frontier'){v=at(D.recovery_frontier,ix,iy,zi);known=at(D.recovery_known,ix,iy,zi)>0}else if(L==='edge_coverage'){v=at(D.edge_coverage,ix,iy,zi)/26;known=at(D.edge_coverage,ix,iy,zi)>0}else if(L==='edge_competence'){v=at(D.edge_competence_node,ix,iy,zi);known=at(D.edge_coverage,ix,iy,zi)>0}else{v=at(D.edge_rate_node,ix,iy,zi)/max;known=at(D.edge_coverage,ix,iy,zi)>0}ctx.fillStyle=(L==='node_learning'&&known)?diverge(clamp(v,-1,1)):heat(v,known);ctx.fillRect(ix*w,c.height-(iy+1)*h,Math.ceil(w),Math.ceil(h));if(SEL&&SEL.ix===ix&&SEL.iy===iy&&SEL.iz===zi){ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.strokeRect(ix*w+1,c.height-(iy+1)*h+1,Math.max(1,w-2),Math.max(1,h-2));}}ctx.fillStyle='#ddd';ctx.font='13px system-ui';ctx.fillText(`vx ${D.vx[0].toFixed(2)} → ${D.vx[nx-1].toFixed(2)} m/s`,10,18);ctx.save();ctx.translate(15,c.height-10);ctx.rotate(-Math.PI/2);ctx.fillText(`vy ${D.vy[0].toFixed(2)} → ${D.vy[ny-1].toFixed(2)} m/s`,0,0);ctx.restore();runEl.textContent=`active: ${D.run_name}   field schema ${D.field_schema}`;renderPicked();stats.textContent=`updated: ${D.mtime_text}\nknown/mastered/frontier nodes: ${D.known_nodes} / ${D.competent_nodes} / ${D.frontier_nodes}\nknown/mastered recovery cells: ${D.known_recovery} / ${D.competent_recovery}\nknown/mastered maneuver edges: ${D.known_edges} / ${D.competent_edges}\nevents node/recovery/edge: ${D.node_events} / ${D.recovery_events} / ${D.edge_events}`}
function renderPicked(){if(!D||!SEL){picked.textContent='click a voxel for details';return}const {ix,iy,iz}=SEL;const e=at(D.node_evidence,ix,iy,iz);picked.textContent=`vx    ${D.vx[ix].toFixed(3)} m/s\nvy    ${D.vy[iy].toFixed(3)} m/s\nyaw   ${D.yaw[iz].toFixed(3)} rad/s\n\ncompetence   ${(100*at(D.node_competence,ix,iy,iz)).toFixed(1)}%\nconservative ${(100*at(D.node_lower,ix,iy,iz)).toFixed(1)}%\nconfidence   ${(100*at(D.node_confidence,ix,iy,iz)).toFixed(1)}%\nquality      ${(100*at(D.node_quality,ix,iy,iz)).toFixed(1)}%\neff trials   ${e.toFixed(1)}\ntrend        ${(100*at(D.node_learning,ix,iy,iz)).toFixed(1)} pp\nstaleness    ${(100*at(D.node_staleness,ix,iy,iz)).toFixed(1)}%\n\nrecovery frontier ${(100*at(D.recovery_frontier,ix,iy,iz)).toFixed(0)}% severity\nmaneuver dirs     ${at(D.edge_coverage,ix,iy,iz).toFixed(0)} / 26\nmaneuver competence ${(100*at(D.edge_competence_node,ix,iy,iz)).toFixed(1)}%\nmaneuver rate     ${at(D.edge_rate_node,ix,iy,iz).toFixed(2)} norm/s`}
c.addEventListener('click',ev=>{if(!D||D.waiting)return;const r=c.getBoundingClientRect(),x=(ev.clientX-r.left)*c.width/r.width,y=(ev.clientY-r.top)*c.height/r.height;SEL={ix:clamp(Math.floor(x/(c.width/D.vx.length)),0,D.vx.length-1),iy:clamp(D.vy.length-1-Math.floor(y/(c.height/D.vy.length)),0,D.vy.length-1),iz:+yaw.value};draw()});
async function load(){try{const r=await fetch('/data',{cache:'no-store'}),ct=r.headers.get('content-type')||'';if(!r.ok){throw new Error(`HTTP ${r.status}: ${(await r.text()).slice(0,160)}`)}if(!ct.includes('application/json'))throw new Error(`expected JSON, got ${ct}`);const d=await r.json();if(d.waiting){D=d;runEl.textContent=d.message||'waiting…';stats.textContent=`searching: ${d.search_root||''}`;return}D=d;yaw.max=D.yaw.length-1;if(+yaw.value>D.yaw.length-1)yaw.value=Math.floor(D.yaw.length/2);if(SEL)SEL.iz=+yaw.value;draw()}catch(e){stats.textContent='capability viewer error: '+e.message}}
layer.addEventListener('change',draw);yaw.addEventListener('input',()=>{if(SEL)SEL.iz=+yaw.value;draw()});load();setInterval(()=>{if(document.getElementById('auto').checked)load()},2000);
</script>'''


def payload(path: Path):
    last = None
    for _ in range(3):
        try:
            with np.load(path) as d:
                vx, vy, yaw = [np.asarray(d[k], float) for k in ('vx', 'vy', 'yaw')]
                nc = np.asarray(d['node_competence'], float)
                ne = np.asarray(d['node_evidence'], float)
                schema = int(np.asarray(d['field_schema']).ravel()[0])
                if schema != FIELD_VERSION:
                    raise ValueError(f'field schema {schema} is not current schema {FIELD_VERSION}')
                nl = np.asarray(d['node_lower'], float)
                nq = np.asarray(d['node_quality'], float)
                conf = np.asarray(d['node_confidence'], float)
                slow = np.asarray(d['node_slow'], float)
                step = int(np.asarray(d['step']).ravel()[0])
                half = float(np.asarray(d['half_life_steps']).ravel()[0])
                last_node = np.asarray(d['node_last_step'], float)
                rc = np.asarray(d['recovery_competence'], float)
                re = np.asarray(d['recovery_evidence'], float)
                rl = np.asarray(d['recovery_lower'], float)
                ec = np.asarray(d['edge_competence'], float)
                ee = np.asarray(d['edge_evidence'], float)
                el = np.asarray(d['edge_lower'], float)
                er = np.asarray(d['edge_rate'], float)
            break
        except Exception as exc:
            last = exc
            time.sleep(0.04)
    else:
        raise last

    min_evidence = 3.0
    strong = 8.0
    node_master = (ne >= strong) & (nl >= .80)
    frontier = (ne >= min_evidence) & (~node_master) & (nc >= .35)
    node_learning = np.clip(nc - slow, -1.0, 1.0)
    node_stale = np.clip(np.maximum(0.0, step-last_node) / max(half, 1.0), 0.0, 2.0) / 2.0

    rec_ok = (re >= strong) & (rl >= .70)
    bins = (np.arange(rc.shape[-1], dtype=float) + .5) / max(1, rc.shape[-1])
    recovery_frontier = np.max(np.where(rec_ok, bins, 0.0), axis=-1)
    recovery_known = np.sum(re >= 2.0, axis=-1)

    edge_known_mask = ee >= 2.0
    edge_master_mask = (ee >= strong) & (el >= .70)
    edge_coverage = np.sum(edge_known_mask, axis=-1)
    edge_comp_node = np.sum(np.where(edge_known_mask, ec, 0.0), axis=-1) / np.maximum(edge_coverage, 1)
    edge_rate_node = np.max(np.where(edge_known_mask, er, 0.0), axis=-1)

    mt = path.stat().st_mtime
    summary_path = path.parent / 'capability_field_summary.json'
    summ = {}
    try:
        if summary_path.exists(): summ = json.loads(summary_path.read_text(encoding='utf-8'))
    except Exception:
        pass
    return {
        'waiting': False, 'field_schema': schema,
        'run_name': path.parent.name, 'path': str(path),
        'vx': vx.tolist(), 'vy': vy.tolist(), 'yaw': yaw.tolist(),
        'node_competence': nc.ravel().tolist(), 'node_lower': nl.ravel().tolist(),
        'node_evidence': ne.ravel().tolist(), 'node_quality': nq.ravel().tolist(),
        'node_confidence': conf.ravel().tolist(), 'node_learning': node_learning.ravel().tolist(),
        'node_staleness': node_stale.ravel().tolist(),
        'recovery_frontier': recovery_frontier.ravel().tolist(), 'recovery_known': recovery_known.ravel().tolist(),
        'edge_coverage': edge_coverage.ravel().tolist(), 'edge_competence_node': edge_comp_node.ravel().tolist(),
        'edge_rate_node': edge_rate_node.ravel().tolist(),
        'max_node_evidence': float(np.max(ne)) if ne.size else 1.0,
        'max_edge_rate': float(np.max(edge_rate_node)) if edge_rate_node.size else 0.0,
        'min_evidence': min_evidence,
        'known_nodes': int(np.sum(ne >= min_evidence)), 'competent_nodes': int(np.sum(node_master)),
        'frontier_nodes': int(np.sum(frontier)),
        'known_recovery': int(np.sum(re >= 2.0)),
        'competent_recovery': int(np.sum(rec_ok)),
        'known_edges': int(np.sum(edge_known_mask)), 'competent_edges': int(np.sum(edge_master_mask)),
        'node_events': int(float(summ.get('field_node_events_total', 0))),
        'recovery_events': int(float(summ.get('field_recovery_events_total', 0))),
        'edge_events': int(float(summ.get('field_edge_events_total', 0))),
        'mtime': mt, 'mtime_text': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mt)),
    }


def field_main(argv=None):
    ap = argparse.ArgumentParser(description=f'EMU {VERSION} live capability-field viewer')
    group = ap.add_mutually_exclusive_group()
    group.add_argument('--run-dir', default='', help='watch one exact run directory')
    group.add_argument('--run-root', default='', help='auto-follow newest field under <run-root>')
    ap.add_argument('--seed', type=int, default=733)
    ap.add_argument('--port', type=int, default=8765)
    args = ap.parse_args(argv)

    exact = Path(args.run_dir).expanduser().resolve() if args.run_dir else None
    root = Path(args.run_root).expanduser().resolve() if args.run_root else seed_root(args.seed)

    def find_npz():
        if exact is not None:
            p = exact / 'capability_field_latest.npz'
            if not p.exists(): p = exact / 'capability_field_final.npz'
            return p if p.exists() else None
        if not root.exists(): return None
        candidates = [p for pat in ('*/capability_field_latest.npz','*/capability_field_final.npz') for p in root.glob(pat) if p.is_file()]
        return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None

    class H(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def send_bytes(self, status, content_type, body):
            self.send_response(status); self.send_header('Content-Type', content_type); self.send_header('Cache-Control','no-store'); self.send_header('Content-Length',str(len(body))); self.end_headers(); self.wfile.write(body)
        def send_json(self, obj, status=200): self.send_bytes(status,'application/json; charset=utf-8',json.dumps(obj,separators=(',',':')).encode())
        def do_GET(self):
            req=urlparse(self.path).path
            if req=='/': return self.send_bytes(200,'text/html; charset=utf-8',HTML.encode())
            if req=='/data':
                npz=find_npz()
                if npz is None:
                    where=str(exact if exact is not None else root)
                    return self.send_json({'waiting':True,'message':'waiting for first capability-field snapshot…','search_root':where})
                try: return self.send_json(payload(npz))
                except Exception as exc: return self.send_json({'waiting':True,'message':f'field exists but is temporarily unreadable: {exc}','search_root':str(npz)},503)
            return self.send_json({'error':'not found','path':req},404)

    srv=ThreadingHTTPServer(('127.0.0.1',args.port),H)
    mode=f'exact run {exact}' if exact is not None else f'auto-follow newest field under {root}'
    print(f'EMU {VERSION} capability viewer: http://127.0.0.1:{args.port}/\n{mode}')
    srv.serve_forever()

