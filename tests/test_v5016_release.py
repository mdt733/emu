"""Release-gate contracts: no GitHub Pages promotion without a real, pinned actor."""
from __future__ import annotations

import base64
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from verify_showcase_release import (
    validate_browser_bundle, approval_for_bundle, verify_approved_release,
)
from stage_v5016_showcase import selected_candidate


def _f32(arr):
    arr = np.asarray(arr, dtype='<f4', order='C')
    return {'dtype': 'float32-le', 'shape': list(arr.shape),
            'data_b64': base64.b64encode(arr.tobytes()).decode('ascii')}


def sensor_fixture():
    from emu.contracts import sensor_observation_contract, SENSOR_OBS_FIELDS
    c = sensor_observation_contract()
    layers = [
        {'w': _f32(np.zeros((232, 3))), 'b': _f32(np.zeros(3))},
        {'w': _f32(np.zeros((3, 4))), 'b': _f32(np.zeros(4))},
        {'w': _f32(np.zeros((4, 6))), 'b': _f32(np.zeros(6))},
    ]
    return {'format': 'emu-policy', 'schema': 2,
            'source': {'step': 300_000_000, 'checkpoint_schema': 8,
                       'git_commit': 'a' * 40, 'actor_sha256': 'b' * 64},
            'robot_xml': '<mujoco/>',
            'policy': {'obs_dim': 232, 'base_obs_dim': 58, 'history_len': 4,
                       'action_dim': 6, 'hidden_sizes': [3, 4], 'layers': layers},
            'observation': {'base_fields': list(SENSOR_OBS_FIELDS),
                            'sensor_contract': c,
                            'mean': _f32(np.zeros(232)),
                            'std': _f32(np.ones(232)),
                            'normalize_mask': _f32(np.ones(232))}}


def test_release_verifier_understands_real_sensor_contract_and_refuses_mixed_schema():
    bundle = sensor_fixture()
    assert validate_browser_bundle(bundle)['obs_dim'] == 232
    bundle['policy']['obs_dim'] = 220
    with pytest.raises(ValueError, match='shape mismatch'):
        validate_browser_bundle(bundle)
    bundle = sensor_fixture()
    bundle['observation']['sensor_contract']['settings']['tof_delay_frames'] = 2
    with pytest.raises(ValueError, match='sensor contract hash'):
        validate_browser_bundle(bundle)
    bundle = sensor_fixture()
    bundle['source']['checkpoint_schema'] = 7
    with pytest.raises(ValueError, match='checkpoint schema 8'):
        validate_browser_bundle(bundle)
    bundle = sensor_fixture()
    bundle['policy']['layers'][0]['w']['data_b64'] = ''
    with pytest.raises(ValueError, match='binary length'):
        validate_browser_bundle(bundle)


def test_exact_approval_pin_checks_policy_bytes_and_checkpoint_provenance(tmp_path):
    bundle = sensor_fixture()
    policy = tmp_path / 'champion.emupolicy'
    policy.write_text(json.dumps(bundle), encoding='utf-8')
    approval = tmp_path / 'release_approval.json'
    approval.write_text(json.dumps(approval_for_bundle(
        bundle, policy, checkpoint_sha256='f' * 64)), encoding='utf-8')
    assert verify_approved_release(policy, approval)['step'] == 300_000_000
    modified = sensor_fixture()
    modified['source']['step'] += 1
    policy.write_text(json.dumps(modified), encoding='utf-8')
    with pytest.raises(ValueError, match='bytes have changed'):
        verify_approved_release(policy, approval)
    policy.write_text(json.dumps(bundle), encoding='utf-8')
    wrong = json.loads(approval.read_text())
    wrong['schema'] = 1
    approval.write_text(json.dumps(wrong), encoding='utf-8')
    with pytest.raises(ValueError, match='schema disagrees'):
        verify_approved_release(policy, approval)


def test_campaign_winner_preflight_requires_evidence_and_finished_or_explicit_budget_stop(tmp_path):
    root = tmp_path / 'pinnacle'
    (root/'scoreboards').mkdir(parents=True)
    (root/'polish'/'300m'/'checkpoints').mkdir(parents=True)
    checkpoint = root/'polish'/'300m'/'checkpoints'/'best.pkl'
    checkpoint.write_bytes(b'test bytes (no pickle required for selection preflight)')
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    winner = {'path': str(checkpoint), 'run_dir': str(checkpoint.parent.parent),
              'sha256': digest, 'step': 300_000_000,
              'source': 'a'*40, 'sensor_sha256': 'b'*64}
    froot = tmp_path/'foundation'
    campaign = {'schema': 1, 'version': 'v5016-pinnacle-1',
                'source_commit': 'a'*40, 'sensor_sha256': 'b'*64,
                'foundation': {'run_dir':str(froot), 'sensor_sha256':'b'*64},
                'champion': winner, 'finished':False}
    (root/'campaign.json').write_text(json.dumps(campaign))
    (root/'champion.json').write_text(json.dumps(winner))
    with pytest.raises(ValueError, match='not marked finished'):
        selected_candidate(root)
    with pytest.raises(ValueError, match='no eligible independently scored'):
        selected_candidate(root, accept_budget_stop=True)
    (root/'scoreboards'/'round2.json').write_text(json.dumps(
        {'rows':[{'name':'r2_grace','policy':winner,
                  'rating':{'eligible':True,'score':.501}}]}))
    chosen, _, evidence=selected_candidate(root, accept_budget_stop=True)
    assert chosen==checkpoint
    assert evidence['eligible_scoreboards'][0][0]=='round2.json'
    checkpoint.write_bytes(b'tampered')
    with pytest.raises(ValueError, match='pinned SHA-256'):
        selected_candidate(root, accept_budget_stop=True)
