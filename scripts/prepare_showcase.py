#!/usr/bin/env python3
"""Prepare static-only browser assets for GitHub Pages, without training deps.

Does not alter training/checkpoints or the committed source policy. This output is
build-only and is ignored by Git; only showcase/champion.emupolicy is committed.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def prepare(*, output_dir: Path, policy_path: Path | None = None,
            source_url: str = '', source_commit: str = '', require_champion: bool = False) -> dict:
    champion = ROOT / 'showcase' / 'champion.emupolicy'
    # Release builds always require a real trained policy. The old
    # zero-action fallback was misleading and had no publishing use.
    selected = policy_path.expanduser().resolve() if policy_path is not None else champion
    if not selected.is_file():
        raise FileNotFoundError(f'Trained policy not found: {selected}')
    trained = True

    # The public site bundles only a trained, validated browser-native
    # .emupolicy. Python checkpoints are never served.
    from verify_showcase_release import validate_browser_bundle
    bundle = json.loads(selected.read_text(encoding='utf-8'))
    validate_browser_bundle(bundle, require_trained=trained)
    source = bundle.get('source') or {}
    if trained and int(source.get('step') or 0) <= 0:
        raise ValueError('Refusing to publish a zero-step policy as a trained champion')
    label = str(source.get('run_name') or 'Selected EMU champion')[:100]
    if source_commit and not re.fullmatch(r'[0-9a-f]{40}', source_commit):
        raise ValueError('source_commit must be a complete lowercase Git commit SHA')
    manifest = {
        'schema': 1,
        'mode': 'showcase',
        'policy': 'policies/champion.emupolicy',
        'trained': trained,
        'autoplay': True,
        'label': label,
        'source_url': source_url if source_url.startswith('https://github.com/') else '',
        'source_commit': source_commit,
    }

    # Write a complete immutable set into a temporary staging folder, then
    # replace the public directory. Never remove an arbitrary --output-dir.
    output_dir = output_dir.expanduser().resolve()
    if not (output_dir.name == 'public' and output_dir.parent == (ROOT / 'emu' / 'viewer_src').resolve()):
        raise ValueError('Output must be this checkout\'s emu/viewer_src/public directory')
    staging = output_dir.parent / '.showcase-build-tmp'
    if staging.exists():
        shutil.rmtree(staging)
    (staging / 'policies').mkdir(parents=True)
    shutil.copyfile(selected, staging / 'policies' / 'champion.emupolicy')
    (staging / 'showcase.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    (staging / '.nojekyll').write_text('', encoding='utf-8')
    if output_dir.exists():
        # Keep this operation safe when a developer has unrelated content.
        unexpected = set(os.listdir(output_dir)) - {'policies', 'showcase.json', '.nojekyll'}
        if unexpected:
            raise ValueError(f'Refusing to replace public/ containing unknown files: {sorted(unexpected)}')
        shutil.rmtree(output_dir)
    staging.replace(output_dir)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', type=Path, default=None, help='Optional trained exported policy (otherwise showcase/champion.emupolicy)')
    parser.add_argument('--source-url', default='', help='Public GitHub repository URL')
    parser.add_argument('--source-commit', default='', help='Full Git commit used to build this static frontend')
    parser.add_argument('--require-champion', action='store_true', help='Require a trained champion (always enforced; retained for CI compatibility)')
    args = parser.parse_args()
    manifest = prepare(
        output_dir=ROOT / 'emu' / 'viewer_src' / 'public',
        policy_path=args.policy,
        source_url=args.source_url,
        source_commit=args.source_commit,
        require_champion=args.require_champion,
    )
    print(f"Static EMU showcase ready: {manifest['label']} (trained policy)")


if __name__ == '__main__':
    main()
