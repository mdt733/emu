#!/usr/bin/env python3
"""Validate immutable, explicitly reviewed public EMU browser release assets.

Only Python stdlib is needed; can run in the no-CUDA GitHub Pages builder.
Only a reviewed, trained 360-input physical h40 actor can be released.
Historical release protocols remain accessible in Git history.
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
PHYSICAL_TERRAIN_REVISION = "vl53l8cx_array_plane_square45_below30_v3"
SENSOR_SETTING_KEYS = frozenset({
    'tof_period_frames', 'flow_period_frames', 'tof_delay_frames',
    'flow_delay_frames', 'tof_dropout_prob', 'flow_dropout_prob',
    'gyro_bias_std_rad_s', 'sensor_noise_scale',
})
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
    if schema != 3:
        raise ValueError('Only the current 360-input physical actor can publish')
    dim = 360
    width = dim // 4
    policy = bundle.get('policy') or {}
    source = bundle.get('source') or {}
    obs = bundle.get('observation') or {}
    controller = bundle.get('control') or {}
    trim = controller.get('velocity_i') or {}
    if not isinstance(trim,dict):
        raise ValueError('invalid controller metadata')
    if float(trim.get('multiplier',1.)) != 1.:
        # Inference-only browser does not yet implement velocity-I.
        # Prevent a misleading published "champion" that behaves
        # differently from its actual controller-assisted training.
        raise ValueError('velocity-I actor cannot publish until browser controller parity is implemented')
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
    if source.get('checkpoint_schema') != 9:
        raise ValueError('Only schema-9 360-input training can publish')
    # A public 360 actor must be the physical, ARRAY-ONLY 4x4 sensor
    # lineage. A self-consistent hash alone is not proof of correct
    # extrinsics: every geometry/estimator constant must match the
    # independently implemented browser observation contract.
    if source.get('checkpoint_schema') != 9:
        raise ValueError('360-input release requires checkpoint schema 9')
    c = obs.get('sensor_contract')
    if not isinstance(c, dict):
        raise ValueError('missing 360-input sensor contract')
    expected_fields = (
        SENSOR_FIELDS +
        tuple(f'tof4x4_range_{row}{col}_m'
              for row in range(4) for col in range(4)) +
        tuple(f'tof4x4_valid_{row}{col}'
              for row in range(4) for col in range(4))
    )
    if (c.get('schema') != 'terrain_360' or
            c.get('sensor_model_revision') != PHYSICAL_TERRAIN_REVISION or
            c.get('base_fields') != fields or
            c.get('history_len') != 4 or c.get('action_dim') != 6 or
            c.get('control_dt') != 0.02 or
            tuple(fields[55:]) != expected_fields):
        raise ValueError('unsupported 360-input observation contract')
    common = {
        'schema', 'sensor_model_revision', 'base_fields', 'history_len',
        'action_dim', 'control_dt', 'settings', 'sha256',
        'tof4x4_mount_body_m', 'tof4x4_range_limits_m', 'tof4x4_occlusion',
        'tof4x4_validity', 'array_ground_estimator',
        'array_ground_max_fit_residual_m', 'array_ground_min_valid_cells',
    }
    geometry = {
        'tof4x4_fov_horizontal_deg', 'tof4x4_fov_vertical_deg',
        'tof4x4_tilt_down_deg', 'tof4x4_mount_yaw_deg',
        'tof4x4_rays_body', 'tof4x4_hardware_zone_ids',
    }
    if set(c) != common | geometry:
        raise ValueError('unexpected physical 360-input sensor fields')
    if (c.get('tof4x4_range_limits_m') != [.04, 1.5] or
            c.get('tof4x4_occlusion') !=
                'nearest ray-sphere foot hit masks terrain return' or
            c.get('tof4x4_validity') !=
                'one bit per cell, delayed, zero on dropout' or
            c.get('array_ground_estimator') !=
                'weighted_4x4_plane_residual_8mm_v1' or
            c.get('array_ground_max_fit_residual_m') != .008 or
            c.get('array_ground_min_valid_cells') != 6):
        raise ValueError('unapproved physical 360-input estimator')
    # Independently recalculate all 16 physical image-plane rays and
    # confirm ST's lens-inverted raw zone order.
    import math
    half = math.tan(math.radians(45.) / 2.)
    cs = math.cos(math.radians(70.))
    sn = math.sin(math.radians(70.))
    rays = []
    for row in range(4):
        for col in range(4):
            u = (col - 1.5) * half / 2.
            v = (1.5 - row) * half / 2.
            x, y, z = cs + sn * v, u, -sn + cs * v
            norm = math.sqrt(x*x + y*y + z*z)
            rays.append([x/norm, y/norm, z/norm])
    actual = c.get('tof4x4_rays_body')
    if (c.get('tof4x4_mount_body_m') != [0, 0, -.03] or
            c.get('tof4x4_fov_horizontal_deg') != 45 or
            c.get('tof4x4_fov_vertical_deg') != 45 or
            c.get('tof4x4_tilt_down_deg') != 70 or
            c.get('tof4x4_mount_yaw_deg') != 0 or
            c.get('tof4x4_hardware_zone_ids') !=
                [3,2,1,0,7,6,5,4,11,10,9,8,15,14,13,12] or
            not isinstance(actual, list) or len(actual) != 16 or
            any(not isinstance(a,list) or len(a)!=3 or
                any(type(n) not in (int,float) or
                    not math.isfinite(n) or abs(n-b)>1e-6
                    for n,b in zip(a,expect))
                for a,expect in zip(actual,rays))):
        raise ValueError('unapproved VL53L8CX physical rays, zone order or mounting')
    settings = c.get('settings')
    if not isinstance(settings, dict) or set(settings) != SENSOR_SETTING_KEYS:
        raise ValueError('invalid 360-input sensor timing/noise settings')
    import math
    for key in ('tof_period_frames', 'flow_period_frames'):
        if type(settings[key]) is not int or not 1 <= settings[key] <= 20:
            raise ValueError('invalid sensor sampling period')
    for key in ('tof_delay_frames', 'flow_delay_frames'):
        if type(settings[key]) is not int or not 0 <= settings[key] < 4:
            raise ValueError('invalid sensor frame delay')
    for key in ('tof_dropout_prob', 'flow_dropout_prob'):
        value = settings[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('invalid sensor dropout probability')
    for key in ('gyro_bias_std_rad_s', 'sensor_noise_scale'):
        value = settings[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError('invalid sensor noise scale')
    if not HEX64.fullmatch(str(c.get('sha256') or '')):
        raise ValueError('missing 360-input sensor fingerprint')
    digest = hashlib.sha256(json.dumps(
        {key: value for key, value in c.items() if key != 'sha256'},
        sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if digest != c['sha256']:
        raise ValueError('360-input sensor fingerprint mismatch')
    return {'schema': schema, 'obs_dim': dim, 'checkpoint_schema': source.get('checkpoint_schema'),
            'step': source.get('step'), 'actor_sha256': source.get('actor_sha256'),
            'training_source_commit': source.get('git_commit'),
            'sensor_contract_sha256': obs['sensor_contract']['sha256']}


def approval_for_bundle(bundle: dict, policy_file: Path, *, checkpoint_sha256: str) -> dict:
    identity = validate_browser_bundle(bundle)
    if identity['schema'] != 3 or not HEX64.fullmatch(checkpoint_sha256):
        raise ValueError('approval requires a trained 360 actor and retained checkpoint hash')
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
    """One reviewed trained policy, one complete retained checkpoint hash.

    The owner can replace main's policy after independent physical exams,
    but no historical/inference-only release exceptions survive here.
    """
    approval = json.loads(approval_path.read_text(encoding='utf-8'))
    fields = {'release_schema', 'policy_file', 'policy_sha256', 'policy_git_blob',
              'checkpoint_sha256', 'schema', 'obs_dim', 'checkpoint_schema',
              'step', 'actor_sha256', 'training_source_commit',
              'sensor_contract_sha256'}
    if (set(approval) != fields or approval.get('release_schema') != 1 or
            not HEX64.fullmatch(str(approval.get('checkpoint_sha256') or '')) or
            approval.get('policy_file') != 'showcase/champion.emupolicy'):
        raise ValueError('Current release requires a source-pinned trained actor and checkpoint')
    policy_bytes = policy_path.read_bytes()
    if hashlib.sha256(policy_bytes).hexdigest() != approval['policy_sha256']:
        raise ValueError('Approved policy bytes changed')
    identity = validate_browser_bundle(json.loads(policy_bytes))
    if (identity['schema'],identity['checkpoint_schema'],identity['obs_dim']) != (3,9,360):
        raise ValueError('Release must be the current physical 360 actor')
    for field,value in identity.items():
        if approval.get(field) != value:
            raise ValueError(f'Approved {field} differs from actor')
    actual = subprocess.run(['git','hash-object',str(policy_path)],
                            check=True,text=True,capture_output=True).stdout.strip()
    if actual != approval['policy_git_blob']:
        raise ValueError('Approved Git policy blob changed')
    return approval


def verify_showcase_dist(dist: Path, *, expected_source_commit: str = '') -> dict:
    """Verify the built static site contains the EXACT approved public actor.

    Used by both the ordinary browser CI and the protected Pages release.
    It does not need JAX or a training checkpoint.
    """
    dist = dist.resolve()
    approved = verify_approved_release(
        ROOT / 'showcase' / 'champion.emupolicy',
        ROOT / 'showcase' / 'release_approval.json')
    index = dist / 'index.html'
    policy = dist / 'policies' / 'champion.emupolicy'
    meta = dist / 'showcase.json'
    if not all(x.is_file() for x in (index, policy, meta)):
        raise ValueError('built site missing index, showcase manifest or policy')
    source_bytes = (ROOT / 'showcase' / 'champion.emupolicy').read_bytes()
    if policy.read_bytes() != source_bytes:
        raise ValueError('bundled actor is not byte-identical to approved asset')
    manifest = json.loads(meta.read_text(encoding='utf-8'))
    if (manifest.get('schema') != 1 or
            manifest.get('policy') != 'policies/champion.emupolicy' or
            manifest.get('trained') is not True or
            manifest.get('mode') != 'showcase'):
        raise ValueError('built manifest does not identify approved trained actor')
    if expected_source_commit and (
            not HEX40.fullmatch(expected_source_commit) or
            manifest.get('source_commit') != expected_source_commit):
        raise ValueError('built source commit does not match approved checkout')
    if manifest.get('label') != json.loads(source_bytes)['source']['run_name']:
        raise ValueError('built label does not identify the released actor')
    html = index.read_text(encoding='utf-8')
    assets = dist / 'assets'
    if '<html' not in html or '<script' not in html or not list(assets.rglob('*.js')):
        raise ValueError('missing built browser entry point / JavaScript assets')
    # MuJoCo packages may emit WASM as a standalone asset or inline it
    # in the JS bundle; the static dist must at least contain the
    # source-reviewed JS and matching playable policy. The actual
    # wasm runtime is also checked in browser CI / user viewer.
    if not approved['policy_sha256'] == hashlib.sha256(policy.read_bytes()).hexdigest():
        raise ValueError('built actor SHA differs from reviewed release')
    return {'label': manifest['label'], 'trained': True,
            'sha256': approved['policy_sha256']}



def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy', type=Path, default=ROOT / 'showcase' / 'champion.emupolicy')
    p.add_argument('--approval', type=Path, default=ROOT / 'showcase' / 'release_approval.json')
    p.add_argument('--verify-dist', action='store_true',
                   help='Validate built static HTML, WASM, manifest and exact approved policy')
    p.add_argument('--expect-source-commit', default='',
                   help='Expected Git SHA in the built viewer manifest')
    args = p.parse_args()
    if args.verify_dist:
        data = verify_showcase_dist(
            ROOT / 'emu' / 'viewer_src' / 'dist',
            expected_source_commit=args.expect_source_commit)
        print(f"Showcase build verified: {data['label']} (trained={data['trained']})")
        return
    if args.expect_source_commit:
        p.error('--expect-source-commit requires --verify-dist')
    if not args.approval.is_file():
        raise SystemExit('No release approval: public champion remains unchanged')
    a = verify_approved_release(args.policy, args.approval)
    print(f"Approved browser policy: {a['step']:,} steps, {a['obs_dim']} channels, asset {a['policy_sha256'][:16]}")


if __name__ == '__main__':
    main()
