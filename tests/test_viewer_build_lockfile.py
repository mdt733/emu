"""The real browser build must not silently select different npm dependencies."""
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
