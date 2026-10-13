#!/usr/bin/env python3
"""Prepare static-only browser assets for GitHub Pages, without training deps.

Does not alter training/checkpoints or the committed source policy. This output is
build-only and is ignored by Git; only showcase/champion.emupolicy is committed.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def prepare(*, output_dir: Path, policy_path: Path | None = None,
            source_url: str = '', require_champion: bool = False) -> dict:
    champion = ROOT / 'showcase' / 'champion.emupolicy'
    fixture = ROOT / 'viewer' / 'demo_zero.emupolicy'
    if policy_path is not None:
        selected = policy_path.expanduser().resolve()
        trained = selected != fixture.resolve()
    else:
        trained = champion.is_file()
        selected = champion if trained else fixture
    if require_champion and not trained:
        raise ValueError('No champion policy: refusing to publish an untrained preview')
    if not selected.is_file():
        raise FileNotFoundError(f'Policy not found: {selected}')

    # Validate the deployed file, not only its extension. Require an export
    # from this viewer's six-actuator 55x4 contract.
    bundle = json.loads(selected.read_text(encoding='utf-8'))
    policy = bundle.get('policy') or {}
    if (bundle.get('format') != 'emu-policy' or bundle.get('schema') != 1
            or policy.get('obs_dim') != 220 or policy.get('action_dim') != 6
            or policy.get('history_len') != 4 or policy.get('base_obs_dim') != 55
            or len(policy.get('layers', [])) != 3
            or not isinstance(bundle.get('robot_xml'), str)):
        raise ValueError(f'Incompatible EMU browser policy: {selected}')
    source = bundle.get('source') or {}
    if trained and int(source.get('step') or 0) <= 0:
        raise ValueError('Refusing to publish a zero-step policy as a trained champion')
    label = str(source.get('run_name') or ('Selected EMU champion' if trained else 'Zero-action test fixture'))[:100]
    manifest = {
        'schema': 1,
        'mode': 'showcase',
        'policy': 'policies/champion.emupolicy',
        'trained': trained,
        'autoplay': True,
        'label': label,
        'source_url': source_url if source_url.startswith('https://github.com/') else '',
    }

    # Write a complete immutable set into a temporary staging folder, then
    # replace the public directory. Never remove an arbitrary --output-dir.
    output_dir = output_dir.expanduser().resolve()
    if not (output_dir.name == 'public' and output_dir.parent == (ROOT / 'viewer').resolve()):
        raise ValueError('Output must be this checkout\'s viewer/public directory')
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
    parser.add_argument('--policy', type=Path, default=None, help='Optional exported policy (otherwise showcase/champion.emupolicy or zero fixture)')
    parser.add_argument('--source-url', default='', help='Public GitHub repository URL')
    parser.add_argument('--require-champion', action='store_true', help='Refuse to publish the untrained fixture')
    args = parser.parse_args()
    manifest = prepare(
        output_dir=ROOT / 'viewer' / 'public',
        policy_path=args.policy,
        source_url=args.source_url,
        require_champion=args.require_champion,
    )
    print(f"Static EMU showcase ready: {manifest['label']} ({'trained policy' if manifest['trained'] else 'UNTRAINED preview'})")


if __name__ == '__main__':
    main()
