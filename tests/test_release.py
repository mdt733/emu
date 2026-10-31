"""Public release invariants for the 232-input Pinnacle and future 360 actor.

Standalone release approval must match the independently implemented sensor
geometry and exact Git policy blob; training/CI never auto-publishes a model.
"""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import pytest

from emu.contracts import (
    OBS_SCHEMA_TERRAIN, SENSOR_SETTINGS_DEFAULTS, TERRAIN_OBS_FIELDS,
    sensor_observation_contract,
)
from scripts.verify_showcase_release import (
    approval_for_bundle, validate_browser_bundle, verify_approved_release,
)

ROOT = Path(__file__).resolve().parents[1]


def _zero_blob(shape):
    size = 4
    for item in shape:
        size *= item
    return {
        'dtype': 'float32-le',
        'shape': list(shape),
        'data_b64': base64.b64encode(bytes(size)).decode('ascii'),
    }


def _candidate360():
    contract = sensor_observation_contract(
        dict(SENSOR_SETTINGS_DEFAULTS), schema=OBS_SCHEMA_TERRAIN,
        array_only=True)
    widths = [360, 2, 2, 6]
    layers = [
        {'w': _zero_blob([ni, no]), 'b': _zero_blob([no])}
        for ni, no in zip(widths[:-1], widths[1:])
    ]
    return {
        'format': 'emu-policy', 'schema': 3, 'robot_xml': '<mujoco/>',
        'source': {
            'step': 60_000_000, 'checkpoint_schema': 9,
            'actor_sha256': 'a' * 64, 'git_commit': 'b' * 40,
        },
        'policy': {
            'obs_dim': 360, 'base_obs_dim': 90, 'history_len': 4,
            'action_dim': 6, 'hidden_sizes': [2, 2], 'layers': layers,
        },
        'observation': {
            'base_fields': list(TERRAIN_OBS_FIELDS),
            'sensor_contract': contract,
            'mean': _zero_blob([360]), 'std': _zero_blob([360]),
            'normalize_mask': _zero_blob([360]),
        },
        'control': {},
    }


def test_existing_pinnacle_release_still_matches_exact_approval():
    approved = verify_approved_release(
        ROOT / 'showcase/champion.emupolicy',
        ROOT / 'showcase/release_approval.json',
    )
    assert approved['schema'] == 2
    assert approved['obs_dim'] == 232
    assert approved['checkpoint_schema'] == 8


def test_array_only_360_actor_can_be_reviewed_without_publishing(tmp_path):
    candidate = _candidate360()
    identity = validate_browser_bundle(candidate)
    assert identity['checkpoint_schema'] == 9
    assert identity['obs_dim'] == 360
    assert identity['sensor_contract_sha256'] == (
        candidate['observation']['sensor_contract']['sha256'])

    policy = tmp_path / 'candidate.emupolicy'
    policy.write_text(json.dumps(candidate), encoding='utf-8')
    approval = approval_for_bundle(
        candidate, policy, checkpoint_sha256='c' * 64)
    approval_file = tmp_path / 'reviewed_approval.json'
    approval_file.write_text(json.dumps(approval), encoding='utf-8')

    assert verify_approved_release(policy, approval_file) == approval
    # Even after approval, a change of policy bytes cannot pass the release gate.
    policy.write_text(json.dumps(candidate, indent=2), encoding='utf-8')
    with pytest.raises(ValueError, match='policy bytes'):
        verify_approved_release(policy, approval_file)


def _rehash_contract(candidate):
    contract = candidate['observation']['sensor_contract']
    body = {key: value for key, value in contract.items() if key != 'sha256'}
    contract['sha256'] = hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


@pytest.mark.parametrize('change', [
    'tripod_reintroduced', 'ray_extrinsics', 'estimator', 'field_order',
    'checkpoint_schema', 'settings', 'tampered_fingerprint', 'untrained',
])
def test_array_only_release_rejects_mutated_physical_identity(change):
    candidate = _candidate360()
    contract = candidate['observation']['sensor_contract']
    if change == 'tripod_reintroduced':
        contract['tripod_mounts_body_m'] = [[0, 0, 0]]
        _rehash_contract(candidate)
    elif change == 'ray_extrinsics':
        contract['tof4x4_pitch_from_horizontal_deg'][0] += 1
        _rehash_contract(candidate)
    elif change == 'estimator':
        contract['array_ground_estimator'] = 'unreviewed_algorithm'
        _rehash_contract(candidate)
    elif change == 'field_order':
        fields = candidate['observation']['base_fields']
        fields[59], fields[60] = fields[60], fields[59]
        contract['base_fields'] = fields.copy()
        _rehash_contract(candidate)
    elif change == 'checkpoint_schema':
        candidate['source']['checkpoint_schema'] = 8
    elif change == 'settings':
        contract['settings']['tof_period_frames'] = 50
        _rehash_contract(candidate)
    elif change == 'tampered_fingerprint':
        contract['sha256'] = '0' * 64
    elif change == 'untrained':
        candidate['source']['step'] = 0
    with pytest.raises(ValueError):
        validate_browser_bundle(candidate)
