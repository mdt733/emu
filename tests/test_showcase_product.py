"""Deployment tests intentionally require no Node registry access or GPU."""
from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _module():
    spec = importlib.util.spec_from_file_location('prepare_showcase_test', ROOT / 'scripts' / 'prepare_showcase.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tree(tmp_path):
    (tmp_path / 'emu' / 'viewer_src').mkdir(parents=True)
    (tmp_path / 'showcase').mkdir()
    shutil.copyfile(ROOT / 'emu' / 'viewer_src' / 'demo_zero.emupolicy', tmp_path / 'emu' / 'viewer_src' / 'demo_zero.emupolicy')
    return tmp_path / 'emu' / 'viewer_src' / 'public'


def test_preview_build_is_explicitly_untrained(tmp_path):
    out = _tree(tmp_path)
    mod = _module(); mod.ROOT = tmp_path
    manifest = mod.prepare(output_dir=out)
    assert manifest['trained'] is False and manifest['autoplay'] is True
    assert (out / 'policies' / 'champion.emupolicy').is_file()
    assert json.loads((out / 'showcase.json').read_text()) == manifest
    with pytest.raises(ValueError, match='No champion'):
        mod.prepare(output_dir=out, require_champion=True)


def test_champion_must_be_export_and_nonzero(tmp_path):
    out = _tree(tmp_path)
    mod = _module(); mod.ROOT = tmp_path
    champion = tmp_path / 'showcase' / 'champion.emupolicy'
    shutil.copyfile(tmp_path / 'emu' / 'viewer_src' / 'demo_zero.emupolicy', champion)
    with pytest.raises(ValueError, match='zero-step'):
        mod.prepare(output_dir=out)
    policy = json.loads(champion.read_text())
    policy['source']['step'] = 400_000_000
    champion.write_text(json.dumps(policy))
    assert mod.prepare(output_dir=out)['trained'] is True


def test_viewer_uses_one_package_owned_source():
    from emu.viewer_server import viewer_source
    assert viewer_source().resolve() == (ROOT / 'emu' / 'viewer_src').resolve()
    assert not (ROOT / 'viewer').exists(), 'duplicate viewer source must not return'


def test_viewer_required_ui_elements_and_drop_overlay_exit():
    html = (ROOT / 'emu' / 'viewer_src' / 'index.html').read_text()
    main = (ROOT / 'emu' / 'viewer_src' / 'src' / 'main.js').read_text()
    for element in ('settingsDrawer', 'settingsOpen', 'settingsClose', 'drawerBackdrop'):
        assert f'id="{element}"' in html
    assert 'dropOverlay.hidden = true;' in main


def test_release_manifest_reports_exact_frontend_commit(tmp_path):
    out = _tree(tmp_path)
    mod = _module()
    mod.ROOT = tmp_path
    revision = 'f' * 40
    manifest = mod.prepare(output_dir=out, source_commit=revision)
    assert manifest['source_commit'] == revision
    assert json.loads((out / 'showcase.json').read_text())['source_commit'] == revision
    with pytest.raises(ValueError, match='source_commit'):
        mod.prepare(output_dir=out, source_commit='not-a-git-revision')


# --- showcase build assets ---
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
    with pytest.raises(ValueError, match='revision mismatch'):
        verify(tmp_path, expected_source_commit='a' * 40)
    manifest = json.loads((tmp_path / 'showcase.json').read_text())
    manifest['source_commit'] = 'a' * 40
    (tmp_path / 'showcase.json').write_text(json.dumps(manifest))
    assert verify(tmp_path, expected_source_commit='a' * 40)['source_commit'] == 'a' * 40


# --- viewer build lockfile ---
from pathlib import Path

import pytest

from emu import viewer_server


def test_viewer_build_requires_lockfile(monkeypatch, tmp_path):
    src = tmp_path / 'viewer'
    src.mkdir()
    (src / 'package.json').write_text('{}')
    monkeypatch.setattr(viewer_server, 'viewer_source', lambda: src)
    monkeypatch.setattr(viewer_server.shutil, 'which', lambda _: '/usr/bin/npm')
    with pytest.raises(RuntimeError, match='lockfile missing'):
        viewer_server.build()


def test_viewer_build_uses_npm_ci_not_install(monkeypatch, tmp_path):
    src = tmp_path / 'viewer'
    src.mkdir()
    (src / 'package.json').write_text('{}')
    (src / 'package-lock.json').write_text('{}')
    (src / 'dist').mkdir()
    (src / 'dist' / 'index.html').write_text('<html></html>')
    monkeypatch.setattr(viewer_server, 'viewer_source', lambda: src)
    monkeypatch.setattr(viewer_server.shutil, 'which', lambda _: '/usr/bin/npm')
    commands = []
    monkeypatch.setattr(viewer_server.subprocess, 'check_call', lambda args, cwd: commands.append((args, cwd)))
    fake_pkg = tmp_path / 'package'
    fake_pkg.mkdir()
    monkeypatch.setattr(viewer_server, '__file__', str(fake_pkg / 'viewer_server.py'))
    assert viewer_server.build() == 0
    assert [c[0][:2] for c in commands] == [['npm', 'ci'], ['npm', 'run']]
    assert (fake_pkg / 'webviewer_dist' / 'index.html').is_file()
    assert not (fake_pkg / 'viewer_src').exists()  # build never mirrors source trees


# --- viewer source ---
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def test_one_canonical_viewer_source_no_tracked_mirror():
    tracked = subprocess.check_output(
        ['git', 'ls-files', '-z', '--', 'viewer/', 'emu/viewer_src/'],
        cwd=ROOT,
    ).decode('utf-8').split(chr(0))
    assert not [p for p in tracked if p.startswith('viewer/')], (
        'Old top-level viewer source was accidentally reintroduced')
    canonical = {p.removeprefix('emu/viewer_src/') for p in tracked
                 if p.startswith('emu/viewer_src/')}
    for required in ('package-lock.json', 'package.json', 'index.html',
                     'src/main.js', 'src/policy.js', 'src/sim.js',
                     'demo_zero.emupolicy'):
        assert required in canonical, f'Canonical browser file missing: {required}'
    assert not (ROOT / 'viewer').exists()
