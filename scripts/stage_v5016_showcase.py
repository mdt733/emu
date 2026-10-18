#!/usr/bin/env python3
"""Prepare (but NEVER publish or merge) the selected v5016 Pinnacle actor.

Run in a CLEAN, separate feature-branch release checkout after the campaign
finishes. Default = read-only preflight. --stage explicitly replaces the
feature branch's tracked showcase/champion.emupolicy and writes the matching
reviewed-asset pin. No GPU, trainer changes, or automatic commits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

from verify_showcase_release import (
    ROOT, approval_for_bundle, validate_browser_bundle, verify_approved_release,
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as inp:
        for part in iter(lambda: inp.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def _read(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'not a JSON object: {path}')
    return value


def selected_candidate(root: Path, *, accept_budget_stop: bool = False) -> tuple[Path, dict, dict]:
    root = root.expanduser().resolve()
    campaign = _read(root / 'campaign.json')
    winner = _read(root / 'champion.json')
    if campaign.get('version') != 'v5016-pinnacle-1' or campaign.get('schema') != 1:
        raise ValueError('not a recognized v5016 Pinnacle campaign')
    if not campaign.get('finished') and not accept_budget_stop:
        raise ValueError('campaign not marked finished; use --accept-budget-stop ONLY after GPU campaign exits')
    if campaign.get('champion') != winner:
        raise ValueError('champion.json differs from authoritative campaign ledger')
    if campaign.get('foundation', {}).get('sensor_sha256') != campaign.get('sensor_sha256'):
        raise ValueError('foundation and campaign sensor contracts differ')
    filename = Path(winner['path']).expanduser().resolve()
    if filename.name not in ('best.pkl', 'final.pkl'):
        raise ValueError('selected champion must be best.pkl or final.pkl')
    foundation_root = Path(campaign['foundation']['run_dir']).expanduser().resolve()
    if not (root in filename.parents or foundation_root in filename.parents):
        raise ValueError('champion path is outside this Pinnacle campaign/foundation')
    if digest(filename) != winner.get('sha256'):
        raise ValueError('champion.pkl no longer matches pinned SHA-256')
    if winner.get('sensor_sha256') != campaign.get('sensor_sha256'):
        raise ValueError('winner observation contract does not match campaign')
    valid_sources = {campaign.get('source_commit')}
    valid_sources.update(v.get('previous_commit') for v in campaign.get('source_migrations', []))
    if (winner.get('source') and winner['source'] not in valid_sources
            and filename.parent.parent != foundation_root):
        raise ValueError('selected model source revision disagrees with campaign')

    # Require actual, independently evaluated promotion evidence. Neither the
    # largest training step nor a guessed filename can choose the public actor.
    boards = sorted((root / 'scoreboards').glob('*.json'))
    eligible = []
    for path in boards:
        board = _read(path)
        for row in board.get('rows', []):
            policy = row.get('policy') or {}
            rating = row.get('rating') or {}
            if policy.get('sha256') == winner['sha256'] and rating.get('eligible') is True:
                eligible.append((path.name, rating.get('score')))
    if not eligible:
        raise ValueError('chosen champion has no eligible independently scored tournament checkpoint')
    return filename, campaign, {'winner':winner, 'eligible_scoreboards':eligible}


def _require_checkout() -> None:
    r = subprocess.run(['git', 'branch', '--show-current'], cwd=ROOT,
                       check=True, text=True, capture_output=True)
    if r.stdout.strip() != 'feature/v5016-sim2real':
        raise RuntimeError('stage release only in feature/v5016-sim2real; never main')
    changes = subprocess.run(['git', 'status', '--porcelain'], cwd=ROOT,
                             check=True, text=True, capture_output=True)
    if changes.stdout.strip():
        raise RuntimeError('release checkout must be clean (use a separate checkout; not your running training tree)')


def _atomic(path: Path, contents: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile('wb', dir=path.parent, prefix='.' + path.name,
                                         suffix='.tmp', delete=False) as out:
            temp = Path(out.name)
            out.write(contents)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temp, path)
        temp = None
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path.home()/'robot_lab/emu_runs/v5016_pinnacle')
    p.add_argument('--stage', action='store_true',
                   help='write champion bundle and approval pin on THIS FEATURE BRANCH (no commit/merge)')
    p.add_argument('--accept-budget-stop', action='store_true',
                   help='explicitly accept a soft-budget exit that is not marked finished')
    args = p.parse_args(argv)

    filename, campaign, evidence = selected_candidate(
        args.root, accept_budget_stop=args.accept_budget_stop)
    from emu.checkpoint_io import load_checkpoint
    from emu.contracts import OBS_SCHEMA_SENSOR
    from emu.viewer_server import build_bundle
    # Trust the canonical immutable contract; do not skip checkpoint validation.
    payload = load_checkpoint(
        filename, expected_observation_schema=OBS_SCHEMA_SENSOR,
        expected_sensor_contract=load_checkpoint(
            filename, validate=False)['meta']['sensor_contract'])
    meta = payload['meta']
    if meta['sensor_contract']['sha256'] != campaign['sensor_sha256']:
        raise ValueError('checkpoint sensor fingerprint disagrees with Pinnacle campaign')
    if int(meta['step']) != evidence['winner']['step']:
        raise ValueError('winner step does not match checkpoint')
    run_source = (meta.get('source_provenance') or {}).get('git_commit')
    if run_source != evidence['winner'].get('source'):
        raise ValueError('checkpoint Git training provenance differs from selected policy')
    valid_sources = {campaign['source_commit']}
    valid_sources.update(v.get('previous_commit') for v in campaign.get('source_migrations', []))
    if run_source not in valid_sources and filename.parent.parent != Path(campaign['foundation']['run_dir']).resolve():
        raise ValueError('champion did not originate from the Pinnacle source revision')

    bundle = build_bundle(filename, label=f"EMU v5016 Pinnacle · {meta['step']:,} steps")
    identity = validate_browser_bundle(bundle)
    if identity['schema'] != 2 or identity['obs_dim'] != 232:
        raise RuntimeError('expected 232-input schema-8 browser actor')

    selected_score = max((float(score) for _, score in evidence['eligible_scoreboards']
                          if score is not None), default=float('-inf'))
    print(f"Selected: {filename}")
    print(f"Checkpoint: {evidence['winner']['sha256'][:20]} · {meta['step']:,} steps")
    print(f"Evidence: {evidence['eligible_scoreboards']} (best score {selected_score:.4f})")
    print(f"Browser: schema={identity['schema']}, obs={identity['obs_dim']}, actor={identity['actor_sha256'][:20]}")
    if not args.stage:
        print("PREVIEW ONLY; no files changed. After the campaign, use --stage in a clean feature-branch checkout.")
        return 0

    _require_checkout()
    out = ROOT/'showcase/champion.emupolicy'
    approval = ROOT/'showcase/release_approval.json'
    if approval.exists():
        raise RuntimeError('release_approval.json already exists; do not silently overwrite an approved champion')

    serialized = (json.dumps(bundle, separators=(',', ':'), ensure_ascii=True) + '\n').encode('utf-8')
    # New approved asset is pinned by SHA256 and the Git blob identity. Both
    # will enter a human-reviewed PR together, followed by a manual Pages job.
    _atomic(out, serialized)
    pin = approval_for_bundle(bundle, out, checkpoint_sha256=evidence['winner']['sha256'])
    _atomic(approval, (json.dumps(pin, sort_keys=True, indent=2) + '\n').encode('utf-8'))
    verify_approved_release(out, approval)
    print(f"STAGED ONLY: {out.relative_to(ROOT)}, {approval.relative_to(ROOT)}")
    print("Review Git diff; run Python/browser CI and a static preview; commit on feature branch and merge ONLY when approved.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
