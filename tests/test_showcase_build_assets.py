"""Ensure an HTML SPA fallback cannot be deployed as a static showcase."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_verify():
    spec = importlib.util.spec_from_file_location('verify_showcase_dist', ROOT / 'scripts' / 'verify_showcase_dist.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.verify


def test_static_build_requires_real_policy_and_wasm(tmp_path):
    verify = _load_verify()
    (tmp_path / 'index.html').write_text('<!doctype html>')
    with pytest.raises(FileNotFoundError, match='Static manifest missing'):
        verify(tmp_path)
    (tmp_path / 'showcase.json').write_text(json.dumps({
        'schema':1, 'mode':'showcase', 'policy':'policies/champion.emupolicy',
        'label':'Untrained preview', 'trained':False,
    }))
    with pytest.raises(FileNotFoundError, match='Static policy missing'):
        verify(tmp_path)
    (tmp_path / 'policies').mkdir()
    (tmp_path / 'policies/champion.emupolicy').write_text(json.dumps({'format':'emu-policy','schema':1}))
    with pytest.raises(FileNotFoundError, match='MuJoCo WASM asset missing'):
        verify(tmp_path)
    (tmp_path / 'assets').mkdir()
    (tmp_path / 'assets/mujoco-test.wasm').write_bytes(b'\x00asm')
    assert verify(tmp_path)['trained'] is False
