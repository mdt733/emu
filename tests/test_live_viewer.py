"""Read-only viewer feed: checkpoint identity, failure isolation, HTTP contract."""
from __future__ import annotations

import json
from pathlib import Path
from threading import Thread
from http.server import HTTPServer
from types import SimpleNamespace
from urllib.request import urlopen

import numpy as np

from emu.live_viewer import LivePolicyWatcher, candidates
from emu.viewer_server import make_handler
from emu.capability_field import CapabilityField
from emu.policy_export import _training_demo_nodes


def _checkpoint(path, length=1536):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'P' * length)
    return path


def test_live_picks_newest_phase_and_does_not_reload_same_file(tmp_path, monkeypatch):
    import emu.policy_export as export
    ck1 = _checkpoint(tmp_path / '00_foundation' / 'checkpoints' / 'final.pkl')
    calls = []
    def fake_export(path):
        calls.append(path)
        return {'source': {'step': 20_054_016 if path == ck1 else 60_030_976,
                           'actor_sha256': 'a' if path == ck1 else 'b'}}
    monkeypatch.setattr(export, 'build_bundle', fake_export)
    watcher = LivePolicyWatcher(tmp_path)
    assert watcher.status()['step'] == 20_054_016
    assert watcher.policy() is not None
    assert len(calls) == 1
    ck2 = _checkpoint(tmp_path / '10_adapt' / 'checkpoints' / 'latest.pkl')
    assert watcher.status()['step'] == 60_030_976
    assert len(calls) == 2
    assert watcher.status()['phase'] == '10_adapt'
    # A broken newer checkpoint leaves the last good actor available.
    ck2.write_bytes(b'F' * 1600)
    def broken(path):
        if path == ck2: raise ValueError('unfinished write')
        return fake_export(path)
    monkeypatch.setattr(export, 'build_bundle', broken)
    assert watcher.status()['available'] is True
    assert watcher.status()['step'] == 60_030_976
    assert 'retaining last valid' in watcher.status()['note']


def test_candidate_order_uses_later_existing_phases_first(tmp_path):
    _checkpoint(tmp_path / '00_foundation' / 'checkpoints' / 'latest.pkl')
    _checkpoint(tmp_path / '40_generalist' / 'checkpoints' / 'final.pkl')
    _checkpoint(tmp_path / 'round1' / 'baseline' / 'checkpoints' / 'latest.pkl')
    _checkpoint(tmp_path / 'round3' / 'pitch15' / 'checkpoints' / 'final.pkl')
    names = [name for name, _ in candidates(tmp_path)]
    assert names.index('round3/pitch15') < names.index('round1/baseline')
    assert names.index('round1/baseline') < names.index('40_generalist')
    assert names.index('40_generalist') < names.index('00_foundation')
    assert names.index('complete') > names.index('00_foundation')


def test_trainer_field_evidence_yields_exported_demo_nodes():
    field = CapabilityField()
    q = field.command_indices(np.asarray([0.13]), np.asarray([0.0]), np.asarray([0.0]))
    at = tuple(int(a[0]) for a in q)
    field.node_success[at] = 14
    field.node_failure[at] = 0
    limits = {'vx_min': -0.20, 'vx_max': 0.60, 'vy_abs':0.25, 'yaw_abs': 1.5}
    result = _training_demo_nodes(field.state_dict(), limits)
    assert len(result) == 1
    assert result[0]['predicted'] > .80
    assert result[0]['effective_events'] == 14
    field.node_success[at] = 0
    assert not _training_demo_nodes(field.state_dict(), limits)


def test_static_server_json_config_and_wasm_mime(tmp_path):
    (tmp_path / 'index.html').write_text('EMU', encoding='utf8')
    (tmp_path / 'module.wasm').write_bytes(b'\x00asm')
    server = HTTPServer(('127.0.0.1', 0), make_handler(tmp_path))
    th = Thread(target=server.serve_forever, daemon=True); th.start()
    try:
        base = 'http://127.0.0.1:' + str(server.server_port)
        with urlopen(base + '/api/viewer-config') as res:
            assert json.loads(res.read()) == {'live':False,'initial_policy':False}
        with urlopen(base + '/module.wasm') as res:
            assert res.headers['Content-Type'] == 'application/wasm'
            assert res.read() == b'\x00asm'
    finally:
        server.shutdown(); server.server_close(); th.join(timeout=2)


def test_checkpoint_replacement_is_atomic_even_if_pickle_fails(tmp_path, monkeypatch):
    from emu import checkpoint_io as ck
    path = tmp_path / 'latest.pkl'
    state = SimpleNamespace(params={'pi': [np.ones((2,), dtype=np.float32)]},
                            opt_state={'x':np.ones((1,), dtype=np.float32)})
    meta = {'step': 12, 'lr_scale':1.0}
    norm = {'mean':np.zeros(220,dtype=np.float32),
            'var':np.ones(220,dtype=np.float32),'count':1.0}
    opts = {'curriculum_state':{},'health_state':{},'field_state':{}}
    ck.save_checkpoint(path, state, meta, norm, **opts)
    before = path.read_bytes()
    def fail_dump(obj, file, protocol):
        file.write(b'half written')
        raise OSError('disk write aborted')
    monkeypatch.setattr(ck.pickle, 'dump', fail_dump)
    import pytest
    with pytest.raises(OSError, match='disk write aborted'):
        ck.save_checkpoint(path, state, meta, norm, **opts)
    assert path.read_bytes() == before
    assert not list(tmp_path.glob('*.tmp'))
