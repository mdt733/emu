#!/usr/bin/env python3
"""Fail release builds if Vite omitted static policy assets (stdlib-only)."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify(dist: Path) -> dict:
    index = dist / 'index.html'
    manifest_path = dist / 'showcase.json'
    if not index.is_file():
        raise FileNotFoundError(f'Viewer HTML missing: {index}')
    if not manifest_path.is_file():
        raise FileNotFoundError(f'Static manifest missing: {manifest_path}; run python scripts/prepare_showcase.py before building')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    policy = manifest.get('policy')
    if manifest.get('schema') != 1 or manifest.get('mode') != 'showcase' or policy != 'policies/champion.emupolicy':
        raise ValueError(f'Invalid showcase manifest: {manifest_path}')
    policy_path = dist / policy
    if not policy_path.is_file() or policy_path.stat().st_size == 0:
        raise FileNotFoundError(f'Static policy missing: {policy_path}')
    exported = json.loads(policy_path.read_text(encoding='utf-8'))
    if exported.get('format') != 'emu-policy' or exported.get('schema') != 1:
        raise ValueError(f'Invalid exported policy: {policy_path}')
    if not list((dist / 'assets').glob('*.wasm')):
        raise FileNotFoundError(f'MuJoCo WASM asset missing: {dist / "assets"}')
    return manifest


if __name__ == '__main__':
    data = verify(ROOT / 'viewer' / 'dist')
    print(f"Showcase build verified: {data['label']} (trained={data['trained']})")
