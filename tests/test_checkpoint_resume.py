import pickle
import tempfile
import unittest
from pathlib import Path
from typing import NamedTuple, Any

import numpy as np

from emu.capability_field import CapabilityField
from emu.checkpoint_io import save_checkpoint, validate_resume_payload
from emu.contracts import OBS_DIM
from emu.ppo import expected_param_shapes


class FakeState(NamedTuple):
    params: Any
    opt_state: Any


class FakeFullState(NamedTuple):
    params: Any
    opt_state: Any
    env_state: Any
    obs: Any
    key: Any


def fake_policy_params(fill: float = 0.0):
    def build(spec):
        if isinstance(spec, dict):
            return {k: build(v) for k, v in spec.items()}
        if isinstance(spec, list):
            return [build(v) for v in spec]
        return np.full(spec, fill, dtype=np.float32)
    return build(expected_param_shapes())


class CheckpointResumeTests(unittest.TestCase):
    def test_exact_resume_roundtrip_contract(self):
        field = CapabilityField()
        params = fake_policy_params()
        params['pi'][0]['w'][0, 0] = 1.25
        params['vf'][-1]['b'][0] = -0.5
        state = FakeState(
            params=params,
            opt_state=(np.asarray([3.0, 4.0], np.float32), {'count': np.asarray(17, np.int32)}),
        )
        norm = {'mean': np.linspace(-.2, .2, OBS_DIM, dtype=np.float32),
                'var': np.linspace(.8, 1.2, OBS_DIM, dtype=np.float32), 'count': 1234.0}
        curriculum = {'params': {'steady_difficulty': .25}, 'marker': [1, 2, 3]}
        health = {'kl_stress_history': [.2, .1], 'kl_stress_chronic': False,
                  'stepping_collapse_streak': 7}
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'roundtrip.pkl'
            save_checkpoint(
                path, state, {'step': 123456, 'lr_scale': 0.5}, norm,
                curriculum_state=curriculum, health_state=health,
                field_state=field.state_dict())
            with path.open('rb') as f:
                loaded = pickle.load(f)
            validate_resume_payload(loaded, path)

        np.testing.assert_array_equal(loaded['state']['params']['pi'][0]['w'], state.params['pi'][0]['w'])
        np.testing.assert_array_equal(loaded['state']['params']['vf'][-1]['b'], state.params['vf'][-1]['b'])
        np.testing.assert_array_equal(loaded['state']['opt_state'][0], state.opt_state[0])
        self.assertEqual(int(loaded['state']['opt_state'][1]['count']), 17)
        self.assertEqual(set(loaded['state']), {'params', 'opt_state'})
        np.testing.assert_array_equal(loaded['obs_norm']['mean'], norm['mean'])
        np.testing.assert_array_equal(loaded['obs_norm']['var'], norm['var'])
        self.assertEqual(float(loaded['obs_norm']['count']), norm['count'])
        self.assertEqual(loaded['curriculum_state']['marker'], curriculum['marker'])
        self.assertEqual(loaded['health_state'], health)
        restored = CapabilityField.from_state_dict(loaded['field_state'])
        self.assertEqual(restored.fingerprint(), field.fingerprint())

    def test_checkpoint_strips_live_rollout_state(self):
        field = CapabilityField()
        state = FakeFullState(
            params=fake_policy_params(1.0), opt_state=({'m': np.asarray([0.0], np.float32)},),
            env_state=np.ones((1024,), np.float32), obs=np.ones((32,), np.float32), key=np.asarray([1, 2], np.uint32),
        )
        norm = {'mean': np.zeros((OBS_DIM,), np.float32), 'var': np.ones((OBS_DIM,), np.float32), 'count': 1.0}
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'compact.pkl'
            save_checkpoint(path, state, {'step': 1, 'lr_scale': .25}, norm,
                            curriculum_state={'params': {}}, health_state={}, field_state=field.state_dict())
            with path.open('rb') as f:
                loaded = pickle.load(f)
        self.assertEqual(set(loaded['state']), {'params', 'opt_state'})

    def test_resume_rejects_nonfinite_optimizer_state_and_wrong_policy_architecture(self):
        field = CapabilityField()
        state = FakeState(params=fake_policy_params(1.0), opt_state=(np.asarray([0.0], np.float32),))
        norm = {'mean': np.zeros((OBS_DIM,), np.float32), 'var': np.ones((OBS_DIM,), np.float32), 'count': 1.0}
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'strict.pkl'
            save_checkpoint(path, state, {'step': 1, 'lr_scale': .25}, norm,
                            curriculum_state={'params': {}}, health_state={}, field_state=field.state_dict())
            with path.open('rb') as f:
                payload = pickle.load(f)
            bad_opt = pickle.loads(pickle.dumps(payload))
            bad_opt['state']['opt_state'] = (np.asarray([np.nan], np.float32),)
            with self.assertRaisesRegex(ValueError, 'policy/optimiser'):
                validate_resume_payload(bad_opt)
            bad_arch = pickle.loads(pickle.dumps(payload))
            bad_arch['meta']['policy_hidden_sizes'] = [128, 128]
            with self.assertRaisesRegex(ValueError, 'policy architecture'):
                validate_resume_payload(bad_arch)
            bad_shape = pickle.loads(pickle.dumps(payload))
            bad_shape['state']['params']['pi'][0]['w'] = np.zeros((OBS_DIM, 128), np.float32)
            with self.assertRaisesRegex(ValueError, 'parameter-shape'):
                validate_resume_payload(bad_shape)

    def test_resume_requires_meta(self):
        with self.assertRaises(ValueError):
            validate_resume_payload({'state': FakeState(params={}, opt_state=())})


def test_resume_rejects_extra_top_level_state():
    from emu.checkpoint_io import CHECKPOINT_SCHEMA, ROBOT_MODEL_SHA256
    from emu.capability_field import FIELD_VERSION
    from emu.contracts import ACTION_DIM
    payload = {
        'state': {'params': {'p': np.asarray([1.0], np.float32)}, 'opt_state': ()},
        'obs_norm': {'mean': np.zeros((OBS_DIM,), np.float32), 'var': np.ones((OBS_DIM,), np.float32), 'count': 1.0}, 'curriculum_state': {'params': {}}, 'health_state': {},
        'field_state': {},
        'meta': {'checkpoint_schema': CHECKPOINT_SCHEMA, 'field_schema': FIELD_VERSION,
                 'obs_dim': OBS_DIM, 'action_dim': ACTION_DIM,
                 'robot_model_sha256': ROBOT_MODEL_SHA256, 'step': 1, 'lr_scale': 0.5},
        'obsolete_extra': {},
    }
    try:
        validate_resume_payload(payload)
    except ValueError:
        return
    raise AssertionError('extra top-level checkpoint state was accepted')


def test_snapshot_fallback_skips_incompatible_newer_checkpoint():
    from emu.checkpoint_io import find_newest_valid_snapshot
    field = CapabilityField()
    state = FakeState(params=fake_policy_params(1.0), opt_state=())
    norm = {'mean': np.zeros((OBS_DIM,), np.float32), 'var': np.ones((OBS_DIM,), np.float32), 'count': 1.0}
    curriculum = {'params': {'steady_difficulty': 0.0}}
    health = {}
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        older = root / 'snapshot_step_000000000100.pkl'
        newer = root / 'snapshot_step_000000000200.pkl'
        for path in (older, newer):
            save_checkpoint(path, state, {'step': 100, 'lr_scale': 0.5}, norm, curriculum_state=curriculum,
                            health_state=health, field_state=field.state_dict())
        with newer.open('rb') as f:
            bad = pickle.load(f)
        bad['meta']['checkpoint_schema'] = -1
        with newer.open('wb') as f:
            pickle.dump(bad, f)
        assert find_newest_valid_snapshot(root) == older


if __name__ == '__main__':
    unittest.main()
