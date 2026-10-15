"""Five essential learning contracts: reward, JAX, config, resume, atomic save."""
import os
import pickle
import tempfile
import unittest
from pathlib import Path
from typing import NamedTuple, Any
import numpy as np
import pytest
from emu.scoring import weighted_training_terms
from emu.training_config import TrainingConfig, mature_config
from emu.capability_field import CapabilityField
from emu.checkpoint_io import save_checkpoint, validate_resume_payload
from emu.contracts import OBS_DIM
from emu.ppo import expected_param_shapes

def fake_policy_params(fill: float = 0.0):
    def build(spec):
        if isinstance(spec, dict):
            return {k: build(v) for k, v in spec.items()}
        if isinstance(spec, list):
            return [build(v) for v in spec]
        return np.full(spec, fill, dtype=np.float32)
    return build(expected_param_shapes())
class FakeState(NamedTuple):
    params: Any
    opt_state: Any

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

def test_weighted_reward_is_bitwise_equivalent_to_original_float32_expression():
    rng = np.random.default_rng(733)
    inputs = [rng.uniform(0, 1, 512).astype(np.float32) for _ in range(9)]
    alive, locomotion, posture, contact, anti_slip, clearance, smoothness, safety, gate = inputs
    original = (0.58*alive + 1.55*locomotion + 0.68*posture*gate +
                0.50*contact*gate + 0.18*anti_slip + 0.16*clearance +
                0.12*smoothness + 0.17*safety)
    actual, parts = weighted_training_terms(
        alive=alive, locomotion=locomotion, posture=posture, contact=contact,
        anti_slip=anti_slip, clearance=clearance, smoothness=smoothness,
        joint_safety=safety, stepping_gate=gate)
    assert set(parts) == {'alive', 'locomotion', 'posture', 'contact',
                          'anti_slip', 'clearance', 'smoothness', 'joint_safety'}
    np.testing.assert_array_equal(actual, original)
    np.testing.assert_array_equal(actual, sum(parts.values()))

def test_training_reward_and_terminal_payment_jit_compile_in_jax():
    import jax
    import jax.numpy as jnp
    from emu.scoring import training_reward, apply_early_fall_penalty

    m = dict(
        terminal=jnp.array([0., 1.]), vx=jnp.array([.2, .4]),
        cmd_vx=jnp.array([.2, .3]), vy=jnp.zeros(2), cmd_vy=jnp.zeros(2),
        yaw_rate=jnp.zeros(2), cmd_yaw=jnp.zeros(2),
        body_pitch=jnp.zeros(2), cmd_pitch=jnp.zeros(2),
        gravity_z=jnp.ones(2), height=jnp.full(2, .25),
        target_height=jnp.full(2, .25), height_error=jnp.zeros(2),
        attitude_quality=jnp.ones(2), foot_mid_cone=jnp.ones(2),
        bad_contact=jnp.zeros(2), support=jnp.ones(2),
        stance_slip=jnp.zeros(2), swing_clearance=jnp.array([.02, .0]),
        target_step_height=jnp.full(2, .04),
        organic_step_shape=jnp.zeros(2), action_delta_norm=jnp.zeros(2),
        joint_limit_pressure=jnp.zeros(2), scissor_pressure=jnp.zeros(2),
        age_s=jnp.array([1.0, 1.0]), ramp_frac=jnp.ones(2),
        stepping_activity_so_far=jnp.array([.2, 0.]),
    )
    @jax.jit
    def one_step(values):
        raw, parts = training_reward(values, jnp.zeros(2), attitude_profile='v5015', xp=jnp)
        out, fall = apply_early_fall_penalty(raw, values['terminal'] > .5,
                                             jnp.array([10., 10.]), 700, xp=jnp)
        return out, parts['weighted_locomotion'], fall

    reward, locomotion, fall = jax.device_get(one_step(m))
    assert np.isfinite(reward).all() and np.isfinite(locomotion).all()
    assert fall[0] == 0.0 and fall[1] < 0.0

def test_training_config_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "config.json"
        original = mature_config(
            "graph", run_dir=Path(td) / "graph", steps=120_000_000,
            seed=811, device="cuda", envs=1024, resume=Path(td) / "ancestor.pkl",
        )
        original.save(path)
        assert TrainingConfig.load(path) == original

def test_final_publication_atomically_replaces_existing_checkpoint(tmp_path, monkeypatch):
    from emu.run_support import promote_checkpoint
    source = tmp_path / 'latest.pkl'
    target = tmp_path / 'final.pkl'
    source.write_bytes(b'new complete checkpoint')
    target.write_bytes(b'old complete checkpoint')
    assert promote_checkpoint(source, target) == target
    assert source.read_bytes() == target.read_bytes() == b'new complete checkpoint'
    assert not list(tmp_path.glob('*.tmp'))
    original_link = os.link
    def no_hard_links(*args, **kwargs):
        raise OSError('hard links unsupported')
    monkeypatch.setattr(os, 'link', no_hard_links)
    source.write_bytes(b'copy fallback')
    promote_checkpoint(source, target)
    assert target.read_bytes() == b'copy fallback'
    monkeypatch.setattr(os, 'link', original_link)
