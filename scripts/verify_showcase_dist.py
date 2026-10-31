#!/usr/bin/env python3
"""Fail release builds if Vite omitted static policy assets (stdlib-only)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify(dist: Path, *, expected_source_commit: str = '') -> dict:
    index = dist / 'index.html'
    manifest_path = dist / 'showcase.json'
    if not index.is_file():
        raise FileNotFoundError(f'Viewer HTML missing: {index}')
    if not manifest_path.is_file():
        raise FileNotFoundError(f'Static manifest missing: {manifest_path}; run python scripts/prepare_showcase.py before building')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if expected_source_commit and manifest.get('source_commit') != expected_source_commit:
        raise ValueError(f'Static viewer revision mismatch: expected {expected_source_commit}, found {manifest.get("source_commit")!r}')
    if manifest.get('trained') is not True:
        raise ValueError('Public showcase must contain a trained approved policy')
    policy = manifest.get('policy')
    if manifest.get('schema') != 1 or manifest.get('mode') != 'showcase' or policy != 'policies/champion.emupolicy':
        raise ValueError(f'Invalid showcase manifest: {manifest_path}')
    policy_path = dist / policy
    if not policy_path.is_file() or policy_path.stat().st_size == 0:
        raise FileNotFoundError(f'Static policy missing: {policy_path}')
    exported = json.loads(policy_path.read_text(encoding='utf-8'))
    from verify_showcase_release import validate_browser_bundle, verify_approved_release
    validate_browser_bundle(exported, require_trained=True)
    if (ROOT / 'showcase' / 'release_approval.json').is_file():
        verify_approved_release(ROOT / 'showcase' / 'champion.emupolicy',
                                ROOT / 'showcase' / 'release_approval.json')
    # Static manifest must resolve to the *exact* approved local asset.
    if policy_path.read_bytes() != (ROOT / 'showcase' / 'champion.emupolicy').read_bytes():
        raise ValueError('Published asset differs from reviewed champion')
    if exported.get('source', {}).get('step', 0) <= 0 and manifest.get('trained'):
        raise ValueError('Trained showcase cannot contain zero-step actor')
    if not list((dist / 'assets').glob('*.wasm')):
        raise FileNotFoundError(f'MuJoCo WASM asset missing: {dist / "assets"}')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expect-source-commit', default='',
                        help='Require an exact Git commit fingerprint in the published manifest')
    args = parser.parse_args()
    data = verify(ROOT / 'emu' / 'viewer_src' / 'dist', expected_source_commit=args.expect_source_commit)
    print(f"Showcase build verified: {data['label']} (trained={data['trained']})")
