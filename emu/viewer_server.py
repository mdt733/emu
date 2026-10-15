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
                                   'initial_policy': policy is not None})
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
                if policy is None:
                    return self._json({'error': 'no initial policy'}, status=404)
                try:
                    body = policy.read_bytes()
                except OSError:
                    return self._json({'error': 'policy no longer readable'}, status=404)
                return self._send(body, 'application/json; charset=utf-8')
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
    from .live_viewer import LivePolicyWatcher
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
