"""Only maintained browser actor/export and local-serving parity contracts."""
from __future__ import annotations
import base64
import json
import shutil
import subprocess
from pathlib import Path
import numpy as np
import pytest
from emu.contracts import BASE_OBS_FIELDS, OBS_DIM, ACTION_DIM
from emu.ppo import PPO_HIDDEN_SIZES, build_obs_norm_mask

ROOT=Path(__file__).resolve().parents[1]

def blob(value: np.ndarray) -> dict:
    array = np.asarray(value, dtype='<f4', order='C')
    return {'dtype': 'float32-le', 'shape': list(array.shape),
            'data_b64': base64.b64encode(array.tobytes()).decode('ascii')}
def reference_action(obs: np.ndarray, layers: list[tuple], mean: np.ndarray,
                     std: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # Independent NumPy oracle for the deployed deterministic actor.
    x = np.where(mask > 0.5, np.clip((obs - mean) / np.maximum(std, 1e-8),
                                    -10.0, 10.0), obs).astype(np.float32)
    for weights, bias in layers[:-1]:
        x = np.tanh(x @ weights + bias).astype(np.float32)
    weights, bias = layers[-1]
    return np.clip(x @ weights + bias, -1.0, 1.0)


@pytest.mark.skipif(shutil.which('node') is None, reason='Node.js needed for browser parity')
def test_exported_actor_matches_browser_inference(tmp_path):
    rng = np.random.default_rng(733)
    dims = (OBS_DIM, *PPO_HIDDEN_SIZES, ACTION_DIM)
    layers = []
    for size_in, size_out in zip(dims[:-1], dims[1:]):
        weights = (rng.standard_normal((size_in, size_out)).astype(np.float32)
                   * np.float32(.6 / np.sqrt(size_in)))
        bias = (rng.standard_normal(size_out).astype(np.float32)
                * np.float32(.08))
        layers.append((weights, bias))

    mean = rng.normal(0, .15, OBS_DIM).astype(np.float32)
    std = rng.uniform(.15, 2.0, OBS_DIM).astype(np.float32)
    mask = build_obs_norm_mask()
    observations = [
        np.zeros(OBS_DIM, dtype=np.float32),
        rng.normal(0, 1, OBS_DIM).astype(np.float32),
        np.where(np.arange(OBS_DIM) % 2 == 0, 50.0, -50.0).astype(np.float32),
        np.linspace(-.7, .9, OBS_DIM, dtype=np.float32),
    ]
    bundle = {
        'format': 'emu-policy', 'schema': 1, 'robot_xml': '<mujoco/>',
        'policy': {'obs_dim': OBS_DIM, 'base_obs_dim': len(BASE_OBS_FIELDS),
                   'history_len': 4, 'action_dim': ACTION_DIM,
                   'layers': [{'w': blob(w), 'b': blob(b)} for w, b in layers]},
        'observation': {'base_fields': list(BASE_OBS_FIELDS),
                        'mean': blob(mean), 'std': blob(std),
                        'normalize_mask': blob(mask)},
        'control': {},
    }
    js = r"""
import fs from 'node:fs';
import { PolicyRunner, parsePolicy } from './emu/viewer_src/src/policy.js';
const request = JSON.parse(fs.readFileSync(0, 'utf8'));
const runner = new PolicyRunner(parsePolicy(JSON.stringify(request.bundle)));
const actions = request.observations.map(
    obs => Array.from(runner.infer(Float32Array.from(obs))));
process.stdout.write(JSON.stringify(actions));
"""
    result = subprocess.run(
        ['node', '--input-type=module', '--eval', js],
        cwd=ROOT, input=json.dumps({
            'bundle': bundle, 'observations': [o.tolist() for o in observations],
        }), text=True, capture_output=True, check=True, timeout=25,
    )
    browser_actions = np.asarray(json.loads(result.stdout), dtype=np.float32)
    assert browser_actions.shape == (len(observations), ACTION_DIM)
    for observed, browser in zip(observations, browser_actions):
        expected = reference_action(observed, layers, mean, std, mask)
        np.testing.assert_allclose(browser, expected, rtol=1e-4, atol=2e-5)

    # The real local HTTP server serves ONE immutable manual-policy snapshot,
    # with explicit wasm MIME and no live feed unless requested.
    from http.client import HTTPConnection
    from http.server import ThreadingHTTPServer
    from threading import Thread
    from emu.viewer_server import make_handler

    viewer_dir = tmp_path / 'viewer_dist'
    viewer_dir.mkdir()
    (viewer_dir / 'index.html').write_text('viewer', encoding='utf-8')
    (viewer_dir / 'engine.wasm').write_bytes(b'wasm')
    manual = tmp_path / 'manual.emupolicy'
    first_bytes = json.dumps(bundle).encode('utf-8')
    manual.write_bytes(first_bytes)
    server = ThreadingHTTPServer(
        ('127.0.0.1', 0), make_handler(viewer_dir, policy=manual))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def get_local(path):
        connection = HTTPConnection('127.0.0.1', server.server_port, timeout=5)
        try:
            connection.request('GET', path)
            response = connection.getresponse()
            return response.status, {key.lower(): value for key, value in response.getheaders()}, response.read()
        finally:
            connection.close()

    try:
        status, headers, data = get_local('/api/initial-policy')
        assert status == 200 and data == first_bytes
        assert headers['cache-control'] == 'no-store'
        manual.write_text('changed on disk', encoding='utf-8')
        status, _, data = get_local('/api/initial-policy')
        assert status == 200 and data == first_bytes
        status, _, data = get_local('/api/viewer-config')
        assert status == 200 and json.loads(data) == {
            'live': False, 'initial_policy': True}
        status, headers, data = get_local('/engine.wasm')
        assert status == 200 and headers['content-type'] == 'application/wasm'
        assert data == b'wasm'
        status, _, _ = get_local('/api/live/status')
        assert status == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

