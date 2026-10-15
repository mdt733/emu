"""The browser has exactly one tracked source, owned by the Python package."""
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
