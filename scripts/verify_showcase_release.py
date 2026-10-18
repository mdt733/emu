#!/usr/bin/env python3
"""Validate immutable, explicitly reviewed public EMU browser release assets.

Only Python stdlib is needed; can run in the no-CUDA GitHub Pages builder.
The legacy 120M pin stays in the deployment workflow until a reviewed
v5016 release_approval.json is committed with a new champion.emupolicy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
REVISION = "tripod_tof_flow_async_v1"
SENSOR_FIELDS = ("ground_forward_grade_rad",
                 "ground_sideways_grade_rad",
                 "ground_visibility_confidence")


def _blob(blob: dict, shape: list[int], name: str) -> None:
    if not isinstance(blob, dict) or blob.get('dtype') != 'float32-le' or blob.get('shape') != shape:
        raise ValueError(f"invalid {name} shape/type")
    # Validate the size of exported weights, not only the JSON metadata.
    import base64
    raw = base64.b64decode(blob.get('data_b64', ''), validate=True)
    from math import prod
    if len(raw) != 4 * prod(shape):
        raise ValueError(f"invalid {name} binary length")


def validate_browser_bundle(bundle: dict, *, require_trained: bool = True) -> dict:
    if not isinstance(bundle, dict) or bundle.get('format') != 'emu-policy':
        raise ValueError('not an EMU browser policy')
    schema = bundle.get('schema')
    if schema not in (1, 2):
        raise ValueError('unsupported browser policy schema')
    dim = 220 if schema == 1 else 232
    width = dim // 4
    policy = bundle.get('policy') or {}
    source = bundle.get('source') or {}
    obs = bundle.get('observation') or {}
    if (policy.get('obs_dim') != dim or policy.get('base_obs_dim') != width
            or policy.get('history_len') != 4 or policy.get('action_dim') != 6):
        raise ValueError(f'actor shape mismatch for browser schema {schema}')
    if not isinstance(bundle.get('robot_xml'), str) or not bundle['robot_xml'].strip():
        raise ValueError('browser bundle has no robot model')
    fields = obs.get('base_fields')
    if not isinstance(fields, list) or len(fields) != width or len(set(fields)) != width:
        raise ValueError('invalid browser observation field order')
    layers = policy.get('layers')
    h = policy.get('hidden_sizes')
    if not isinstance(h, list) or len(h) != 2 or not all(isinstance(v, int) and 1 <= v <= 2048 for v in h):
        raise ValueError('invalid browser neural architecture')
    if not isinstance(layers, list) or len(layers) != 3:
        raise ValueError('expected three actor layers')
    for i, (layer, ni, no) in enumerate(zip(layers, (dim, h[0], h[1]), (h[0], h[1], 6))):
        _blob(layer.get('w') or {}, [ni, no], f'actor layer {i} weights')
        _blob(layer.get('b') or {}, [no], f'actor layer {i} bias')
    for key in ('mean', 'std', 'normalize_mask'):
        _blob(obs.get(key) or {}, [dim], 'observation ' + key)
    if require_trained and not (isinstance(source.get('step'), int) and source['step'] > 0):
        raise ValueError('refusing untrained policy')
    if schema == 1:
        if source.get('checkpoint_schema') not in (None, 7):
            raise ValueError('legacy browser policy cannot carry schema-8 training')
    else:
        if source.get('checkpoint_schema') != 8:
            raise ValueError('sensor browser policy requires checkpoint schema 8')
        c = obs.get('sensor_contract')
        if not isinstance(c, dict) or c.get('schema') != 'sim2real_232' or c.get('sensor_model_revision') != REVISION:
            raise ValueError('missing or unsupported v5016 sensor contract')
        if c.get('history_len') != 4 or c.get('action_dim') != 6 or c.get('base_fields') != fields:
            raise ValueError('sensor contract fields/history differ from policy')
        if tuple(fields[-3:]) != SENSOR_FIELDS:
            raise ValueError('sensor observation grade field order changed')
        if (c.get('tripod_mounts_body_m') != [[.035, -.05, -.025], [.035, .05, -.025], [-.04, 0, -.025]]
                or c.get('tripod_beams_body') != [[0, 0, -1]] * 3
                or c.get('tof_range_limits_m') != [.035, .6]):
            raise ValueError('unapproved sensor geometry')
        if not isinstance(c.get('settings'), dict) or not HEX64.fullmatch(str(c.get('sha256') or '')):
            raise ValueError('invalid sensor contract fingerprint')
        core = {k: v for k, v in c.items() if k != 'sha256'}
        digest = hashlib.sha256(json.dumps(core, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        if digest != c['sha256']:
            raise ValueError('sensor contract hash mismatch')
    return {'schema': schema, 'obs_dim': dim, 'checkpoint_schema': source.get('checkpoint_schema'),
            'step': source.get('step'), 'actor_sha256': source.get('actor_sha256'),
            'training_source_commit': source.get('git_commit'),
            'sensor_contract_sha256': (obs.get('sensor_contract') or {}).get('sha256') if schema == 2 else None}


def approval_for_bundle(bundle: dict, policy_file: Path, *, checkpoint_sha256: str) -> dict:
    identity = validate_browser_bundle(bundle)
    if identity['schema'] != 2 or not HEX64.fullmatch(checkpoint_sha256):
        raise ValueError('approval generation only accepts trained schema-8 actor with checkpoint hash')
    if not HEX64.fullmatch(str(identity['actor_sha256'] or '')):
        raise ValueError('actor SHA256 missing')
    if not HEX40.fullmatch(str(identity['training_source_commit'] or '')):
        raise ValueError('checkpoint training source commit missing')
    blob = subprocess.run(['git', 'hash-object', str(policy_file)],
                          check=True, text=True, capture_output=True).stdout.strip()
    if not HEX40.fullmatch(blob):
        raise ValueError('Git asset hash unavailable')
    return {'release_schema': 1, 'policy_file': 'showcase/champion.emupolicy',
            'policy_sha256': hashlib.sha256(policy_file.read_bytes()).hexdigest(),
            'policy_git_blob': blob,
            'checkpoint_sha256': checkpoint_sha256, **identity}


def verify_approved_release(policy_path: Path, approval_path: Path) -> dict:
    approval = json.loads(approval_path.read_text(encoding='utf-8'))
    expected = {'release_schema', 'policy_file', 'policy_sha256', 'policy_git_blob',
                'checkpoint_sha256', 'schema', 'obs_dim', 'checkpoint_schema',
                'step', 'actor_sha256', 'training_source_commit',
                'sensor_contract_sha256'}
    if set(approval) != expected or approval['release_schema'] != 1:
        raise ValueError('unrecognised approval contract; review release format')
    if approval['policy_file'] != 'showcase/champion.emupolicy':
        raise ValueError('release approval points outside champion path')
    if not HEX64.fullmatch(str(approval['checkpoint_sha256'])):
        raise ValueError('missing exact checkpoint evidence')
    if hashlib.sha256(policy_path.read_bytes()).hexdigest() != approval['policy_sha256']:
        raise ValueError('approved browser policy bytes have changed')
    identity = validate_browser_bundle(json.loads(policy_path.read_text(encoding='utf-8')))
    if identity['schema'] != 2 or identity['checkpoint_schema'] != 8 or identity['obs_dim'] != 232:
        raise ValueError('v5016 approval cannot downgrade to 220-input legacy model')
    for field, value in identity.items():
        if approval.get(field) != value:
            raise ValueError(f'approved release {field} disagrees with bundled policy')
    actual_blob = subprocess.run(['git', 'hash-object', str(policy_path)],
                                 check=True, text=True, capture_output=True).stdout.strip()
    if approval['policy_git_blob'] != actual_blob:
        raise ValueError('approved Git blob changed')
    return approval


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy', type=Path, default=ROOT / 'showcase' / 'champion.emupolicy')
    p.add_argument('--approval', type=Path, default=ROOT / 'showcase' / 'release_approval.json')
    args = p.parse_args()
    if not args.approval.is_file():
        raise SystemExit('No v5016 release_approval.json yet: legacy showcase pin remains active')
    a = verify_approved_release(args.policy, args.approval)
    print(f"Approved v5016 browser policy: {a['step']:,} steps, {a['obs_dim']} channels, asset {a['policy_sha256'][:16]}")


if __name__ == '__main__':
    main()
