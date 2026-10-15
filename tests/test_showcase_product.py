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
    (tmp_path / 'viewer').mkdir()
    (tmp_path / 'showcase').mkdir()
    shutil.copyfile(ROOT / 'viewer' / 'demo_zero.emupolicy', tmp_path / 'viewer' / 'demo_zero.emupolicy')
    return tmp_path / 'viewer' / 'public'


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
    shutil.copyfile(tmp_path / 'viewer' / 'demo_zero.emupolicy', champion)
    with pytest.raises(ValueError, match='zero-step'):
        mod.prepare(output_dir=out)
    policy = json.loads(champion.read_text())
    policy['source']['step'] = 400_000_000
    champion.write_text(json.dumps(policy))
    assert mod.prepare(output_dir=out)['trained'] is True


def test_viewer_and_package_source_copies_are_identical():
    paths = ['index.html', 'package.json', 'src/main.js', 'src/style.css',
             'src/showcase.js', 'tests/showcase.test.mjs']
    for rel in paths:
        assert (ROOT / 'viewer' / rel).read_bytes() == (ROOT / 'emu' / 'viewer_src' / rel).read_bytes()


def test_viewer_required_ui_elements_and_drop_overlay_exit():
    html = (ROOT / 'viewer' / 'index.html').read_text()
    main = (ROOT / 'viewer' / 'src' / 'main.js').read_text()
    for element in ('settingsDrawer', 'settingsOpen', 'settingsClose', 'drawerBackdrop'):
        assert f'id="{element}"' in html
    assert 'dropOverlay.hidden = true;' in main
