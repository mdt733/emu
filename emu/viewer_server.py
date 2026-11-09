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
        'physics_mode':'8 convex triangular-prism MJX collisions '
                       '(four 75cm tiles); browser terrain is illustrative nine boxes',
        'warnings':[
            'Browser nine-box terrain is illustrative, not the 8 actual MJX '
            'convex-prism colliders, normals, tiles or terrain recycler.',
            'Curriculum frontiers are teaching targets, not measured capability.'
        ]
    }


def sensor_lab_robot_xml() -> str:
    """Generate the immutable h40 mechanical body with trainer-style foot contacts.

    Source robot.xml is never edited. The floor is disabled by the viewer's
    fixture builder; 4-bit prism contacts are accepted only by the feet.
    """
    from xml.etree import ElementTree as ET
    from .physical_variants import generate_xml

    source = (Path(__file__).resolve().parent / 'robot.xml').read_text(encoding='utf-8')
    root = ET.fromstring(generate_xml(source, 'h40'))
    for name in ('r_foot_ball', 'l_foot_ball'):
        geom = root.find(f".//geom[@name='{name}']")
        if geom is None:
            raise RuntimeError(f'Sensor lab missing physical foot geom {name}')
        geom.set('conaffinity', str(int(geom.get('conaffinity', '1')) | 4))

    # v5057 compensated for longer-leg COM by moving the 300g torso box
    # forward, leaving its REAR face ~19mm ahead of the x=0 hip pivots
    # in h40. The dynamics are linked through the torso body, but the
    # screenshot correctly shows a physically impossible open-air joint.
    #
    # Draw the two rear chassis-to-hip mounting rails plus their crossbar
    # only in this untrained sensor laboratory. These are massless,
    # collisionless VISUAL fixtures, not an assertion about finished
    # hardware mass/inertia or a change to v5058 training physics.
    torso = root.find(".//body[@name='torso']")
    box = root.find(".//geom[@name='torso_box']")
    if torso is None or box is None:
        raise RuntimeError('Sensor lab missing torso and chassis geometry')
    box_x = float(box.get('pos', '0 0 0').split()[0])
    half_x = float(box.get('size', '0.05 0.09 0.025').split()[0])
    rear_x = box_x - half_x
    # Overlap the box by 20mm; terminate 8mm behind the hip axis.
    front_x = rear_x + .020
    common = {'type': 'capsule', 'size': '.011', 'mass': '0',
              'contype': '0', 'conaffinity': '0',
              'rgba': '0.7 0.19 0.19 1'}
    for side, y in (('r', -.08), ('l', .08)):
        ET.SubElement(torso, 'geom', {
            **common, 'name': f'sensor_lab_{side}_hip_mount',
            'fromto': f'-0.008 {y:.3f} 0 {front_x:.8f} {y:.3f} 0',
        })
    ET.SubElement(torso, 'geom', {
        **common, 'name': 'sensor_lab_hip_mount_crossbar',
        'fromto': '-0.008 -0.080 0 -0.008 0.080 0',
    })
    return ET.tostring(root, encoding='unicode')


def make_sensor_lab_bundle() -> bytes:
    """Untrained h40 MJCF for CPU sensor calibration, never a policy export.

    Reuse the public bundle's validated observation *shape*, but disable its
    trained actor in the browser. This mode cannot be exported or resumed.
    """
    import hashlib
    from .physical_variants import terrain_tof_pitch_deg

    template = Path(__file__).resolve().parents[1] / 'showcase' / 'champion.emupolicy'
    b = json.loads(template.read_text(encoding='utf-8'))
    model = sensor_lab_robot_xml()
    b['robot_xml'] = model
    b['sensor_lab'] = True
    b['control']['neutral_root_z_m'] = .4
    sensor = b['observation']['sensor_contract']
    sensor['tof4x4_pitch_from_horizontal_deg'] = list(terrain_tof_pitch_deg('h40'))
    identity = {k:v for k,v in sensor.items() if k != 'sha256'}
    sensor['sha256'] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    b['source'].update(run_name='40 cm sensor laboratory — NO TRAINED ACTOR',
                       step=0, actor_sha256='not-applicable-sensor-lab',
                       robot_model_sha256=hashlib.sha256(model.encode()).hexdigest())
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
