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
from threading import Lock


@dataclass(frozen=True)
class Snapshot:
    path: Path
    signature: tuple[str, int, int]
    phase: str


def candidates(root: Path) -> list[tuple[str, Path]]:
    """Return the newest validated-looking checkpoint candidates first.

    The live watcher checks source and actor validity before publishing.
    No retired ladder/tournament ordering is used.
    """
    root = Path(root)
    dirs = [p for p in root.rglob('checkpoints') if p.is_dir()]
    result: list[tuple[str, Path]] = []
    for ck in dirs:
        rel = ck.parent.relative_to(root)
        phase = '/'.join(rel.parts) if rel.parts else 'training'
        result.extend(((phase, ck / 'latest.pkl'), (phase, ck / 'final.pkl')))
        result.extend((phase, p) for p in sorted(
            ck.glob('snapshot_step_*.pkl'), reverse=True))
    result.append(('complete', root / 'final.pkl'))
    # Watch the newest on-disk checkpoint regardless of directory naming.
    # Atomic writer signatures protect against torn or overwritten saves.
    result.sort(
        key=lambda candidate: (
            candidate[1].stat().st_mtime_ns if candidate[1].is_file() else -1,
            candidate[1].name == 'latest.pkl',
        ),
        reverse=True)
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
                    from .policy_export import build_bundle
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


def read_terrain_training_snapshot(root: Path) -> dict:
    """Read-only live frontier evidence from real metrics.csv.

    Called through the loopback-only viewer endpoint. The trainer is never
    contacted or modified and no checkpoint/optimizer state is accessed.
    """
    from .results import _last_finite_metrics
    root=Path(root).expanduser().resolve()
    metrics=root/'metrics.csv'
    if not metrics.is_file():
        # Some launchers watch a parent containing multiple source-pinned runs.
        candidates=[p for p in root.glob('*/metrics.csv') if p.is_file()]
        if not candidates:
            raise FileNotFoundError('No metrics.csv under the live training run')
        metrics=max(candidates,key=lambda p:p.stat().st_mtime)
    row=_last_finite_metrics(metrics)
    if not row:
        raise ValueError('No complete training metrics yet')
    run_dir=metrics.parent
    try:
        cfg=json.loads((run_dir/'training_config.json').read_text(encoding='utf-8'))
        if not isinstance(cfg,dict):cfg={}
    except (OSError,ValueError):
        cfg={}
    try:
        meta=json.loads((run_dir/'run_meta.json').read_text(encoding='utf-8'))
        if not isinstance(meta,dict):meta={}
    except (OSError,ValueError):
        meta={}
    if not (cfg.get('terrain_carpet') and cfg.get('terrain_carpet_prisms')):
        raise ValueError(
            'Current viewer only accepts real 2x2 / eight-prism MJX sources. '
            'Open historical box-terrain runs from their own source revision.')
    return {
        'schema':'emu_terrain_live_v1',
        'training_step':int(float(row['step'])),
        'run_name':run_dir.name,
        'latest_metrics':row,
        'training_config':{
            'seed':cfg.get('seed',735),
            'terrain_carpet':bool(cfg.get('terrain_carpet',False)),
            'terrain_carpet_prisms':bool(cfg.get('terrain_carpet_prisms',False)),
            'academy_stream':bool(cfg.get('academy_stream',False)),
            'refined_school':bool(cfg.get('refined_school',False)),
            'foundation_steps':cfg.get('foundation_steps',20_000_000),
            'academy_intro_steps':cfg.get('academy_intro_steps',60_000_000),
            'academy_start_steps':cfg.get('academy_start_steps',80_000_000),
            'academy_ramp_steps':cfg.get('academy_ramp_steps',20_000_000),
            'carpet_transition_cap_deg':cfg.get('carpet_transition_cap_deg',3.),
            # Actual configured physical push ceiling, distinct from the
            # teacher's independently growing frontier_push_n target.
            'push_max_force_n':cfg.get('push_max_force_n',0.35),
        },
        'source_sha':(meta.get('source_provenance') or {}).get('git_commit'),
        'physics_mode':'GPU: eight deformable convex MJX prisms, four 1m tiles; '
                       'browser spectator: 225 mutable CPU MuJoCo 1m hfields',
        'warnings':[
            'Browser specialists reconstruct signed training tile seams and '
            'births with REAL hfield contacts; this is not a byte-for-byte '
            'replay of GPU prism collisions, solver, RNG or domain randomisation.',
            'Curriculum frontiers are teaching targets, not measured capability.'
        ]
    }


def sensor_lab_robot_xml() -> str:
    """Use the exact trainer h40 morphology; no viewer-only geometry."""
    from .physical_variants import generate_xml
    source = (Path(__file__).resolve().parent / 'robot.xml').read_text(encoding='utf-8')
    return generate_xml(source, 'h40')


def make_sensor_lab_bundle() -> bytes:
    """A tiny untrained physical fixture, independently of any old champion.

    No checkpoint, fake zero-weight neural network, numpy or showcase policy.
    All actual robot parts and ToF directions are generated from current
    trainer source. Browser simulation never calls the neural actor here.
    """
    import hashlib
    from .contracts import (ACTUATED_JOINTS, TERRAIN_OBS_FIELDS,
                            OBS_SCHEMA_TERRAIN, sensor_observation_contract)

    model = sensor_lab_robot_xml()
    identity = sensor_observation_contract(
        schema=OBS_SCHEMA_TERRAIN, array_only=True)
    b = {
        'format': 'emu-policy',
        'schema': 3,
        'sensor_lab': True,
        'source': {
            'checkpoint_schema': 9,
            'run_name': 'H40 SENSOR LAB — UNTRAINED',
            'step': 0,
            'actor_sha256': 'not-applicable-sensor-lab',
            'robot_model_sha256': hashlib.sha256(model.encode()).hexdigest(),
        },
        'policy': {
            'obs_dim': 360, 'base_obs_dim': 90, 'history_len': 4,
            'action_dim': 6, 'layers': [],
        },
        'observation': {
            'base_fields': list(TERRAIN_OBS_FIELDS),
            'sensor_contract': identity,
        },
        'control': {
            'sim_dt': .005, 'control_substeps': 4, 'control_dt': .020,
            'neutral_root_z_m': .4,
            'actuated_joints': list(ACTUATED_JOINTS),
            'target_span_rad': [.38,.48,.58,.38,.48,.58],
            'one_control_frame_actuation_delay': True,
            'sensor_noise_scale': 1.5,
            'phase_base_hz': 1.025,
            'phase_speed_gain': .75,
            'phase_rate_max_hz': 2.6,
            'default_commands': {
                'vx': 0., 'vy': 0., 'yaw': 0., 'height': 0.,
                'pitch': 0., 'roll': 0., 'step_height': .012,
            },
            'command_limits': {
                'vx_min': -.4, 'vx_max': .65, 'vy_abs': .3,
                'yaw_abs': 1.5, 'pitch_abs_rad': .14,
                'height_abs_m': .018, 'roll_abs_rad': .055,
                'step_height_min_m': .008, 'step_height_max_m': .042,
            },
        },
        'robot_xml': model,
    }
    return json.dumps(b, separators=(',', ':')).encode('utf-8')


def make_handler(dist: Path, *, watcher=None, policy: Path | None = None,
                 sensor_lab: bool = False):
    """Fixed local API, no user-supplied filesystem paths or directory browsing."""
    mimetypes.add_type('application/wasm', '.wasm')
    # A manually selected brain stays fixed for this entire viewer session.
    # Only the separate live watcher selects new validated policy snapshots.
    initial_policy = (make_sensor_lab_bundle() if sensor_lab
                      else policy.read_bytes() if policy is not None else None)

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
                                   'sensor_lab': sensor_lab,
                                   'initial_policy': initial_policy is not None})
            if route == '/api/live/status':
                if watcher is None:
                    return self._json({'enabled': False}, status=404)
                return self._json(watcher.status())
            if route == '/api/live/terrain':
                if watcher is None:
                    return self._json({'error':'live training mode disabled'},status=404)
                try:
                    return self._json(read_terrain_training_snapshot(watcher.root))
                except (OSError,ValueError,TypeError) as exc:
                    return self._json({'error':type(exc).__name__,'note':str(exc)},status=503)
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
          live_root: Path | None = None, open_browser: bool = True,
          sensor_lab: bool = False) -> int:
    if sensor_lab and (policy is not None or live_root is not None):
        raise ValueError('Sensor lab does not accept a trained policy or live-run watcher')
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
    handler = make_handler(dist, watcher=watcher, policy=policy, sensor_lab=sensor_lab)
    class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True
        allow_reuse_address = True
    with Server(('127.0.0.1', int(port)), handler) as httpd:
        url = f'http://127.0.0.1:{port}/'
        print(f'EMU viewer: {url}', flush=True)
        if watcher:
            print(f'Live viewer watching completed checkpoints under {watcher.root}', flush=True)
        print(('SENSOR LAB: 40cm h40 robot in browser CPU MuJoCo; '
               'NO trained policy, NO PPO, no CUDA required.' if sensor_lab else
               'Training stays on GPU; browser runs the robot and neural inference.'),
              flush=True)
        if open_browser:
            try: webbrowser.open(url)
            except Exception: pass
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0

# Lazily import heavy checkpoint/export dependencies only for live playback.
# The standalone CPU sensor laboratory requires only Python stdlib + npm.
