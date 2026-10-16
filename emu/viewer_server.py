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
