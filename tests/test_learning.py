"""Five essential learning contracts: reward, JAX, config, resume, atomic save."""
import os
import pickle
import tempfile
import unittest
from pathlib import Path
from typing import NamedTuple, Any
import numpy as np
import pytest
from emu.rewards import weighted_training_terms, PAYMENT_NAMES
from emu.training_config import TrainingConfig, mature_config
from emu.curriculum import CapabilityField
from emu.checkpoint_io import save_checkpoint, validate_resume_payload
from emu.contracts import ACTION_DIM, OBS_DIM
from emu.results import augment_rollout_row
from emu.ppo import expected_param_shapes
from emu.checkpoint_io import resume_training_lineage
from emu.curriculum import TrainingHealth
from emu.curriculum import RegimeGovernor

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
            # The extracted resume owner must restore a fork's exact policy.
            resumed, step, done = resume_training_lineage(
                Path(td) / 'fork', resume=str(path), target_steps=223456, envs=1)
            assert step == 123456 and not done
            np.testing.assert_array_equal(
                resumed['state']['params']['pi'][0]['w'],
                state.params['pi'][0]['w'])
            # Source audit: auto-resume survives missing/corrupt latest.pkl
            # when an independent rotating snapshot passes the full contract.
            import shutil
            fallback_run = Path(td) / 'auto-run'
            checkpoints = fallback_run / 'checkpoints'
            checkpoints.mkdir(parents=True)
            snapshot = checkpoints / 'snapshot_step_000000123456.pkl'
            shutil.copyfile(path, snapshot)
            resumed_auto, step_auto, finished_auto = resume_training_lineage(
                fallback_run, resume='auto', target_steps=223456, envs=1)
            assert step_auto == 123456 and not finished_auto
            np.testing.assert_array_equal(
                resumed_auto['state']['params']['pi'][0]['w'], state.params['pi'][0]['w'])
            broken_latest = checkpoints / 'latest.pkl'
            broken_latest.write_bytes(b'invalid checkpoint')
            resumed_auto, step_auto, finished_auto = resume_training_lineage(
                fallback_run, resume='auto', target_steps=223456, envs=1)
            assert step_auto == 123456 and not finished_auto
            with pytest.raises(RuntimeError, match='resume failed'):
                resume_training_lineage(
                    fallback_run, resume=str(broken_latest), target_steps=223456, envs=1)
            snapshot.unlink()
            with pytest.raises(RuntimeError, match='no valid current-schema'):
                resume_training_lineage(
                    fallback_run, resume='auto', target_steps=223456, envs=1)

            # D4: a real checkpoint also rewinds the policy, optimiser,
            # teacher/health, field, normaliser, LR and physical episode RNG.
            import jax

            class LiveState(NamedTuple):
                params: Any
                opt_state: Any
                env_state: Any
                obs: Any

            class Teacher:
                def __init__(self):
                    self.restored = {'original': 'teacher'}
                def state_dict(self):
                    return dict(self.restored)
                def load_state_dict(self, value):
                    self.restored = value

            class Health:
                def __init__(self):
                    self.restored = {'original': 'health'}
                    self.recovered = False
                def state_dict(self):
                    return dict(self.restored)
                def load_state_dict(self, value):
                    self.restored = value
                def note_recovery_performed(self):
                    self.recovered = True

            class Worlds:
                def __init__(self):
                    self.key0 = jax.random.PRNGKey(733)
                    self.reset_key = None
                    self.reset_params = None
                def params_array(self, params):
                    return params
                def reset_jit(self, key, params):
                    self.reset_key = key
                    self.reset_params = params
                    return 'fresh-worlds'
                def obs(self, env_state):
                    return 'fresh-observations'

            from emu.checkpoint_io import try_restore_recovery
            teacher, health_tracker, worlds = Teacher(), Health(), Worlds()
            collapsed = LiveState(fake_policy_params(fill=-2.0), state.opt_state,
                                  'collapsed-worlds', 'collapsed-observations')
            seen_fields = []
            def selected_task(restored_field):
                seen_fields.append(restored_field.fingerprint())
                return {'teacher': 'restored'}
            restored_state = try_restore_recovery(
                path, state=collapsed,
                governor=teacher, health=health_tracker, env=worlds,
                step=130000, report={'stepping_collapse_streak': 101},
                obs_mean=np.zeros(OBS_DIM), obs_var=np.ones(OBS_DIM), obs_count=1.0,
                task_params_for_field=selected_task, log_prefix='[test]')
            assert restored_state is not None
            assert restored_state.state.env_state == 'fresh-worlds'
            assert restored_state.state.obs == 'fresh-observations'
            np.testing.assert_array_equal(restored_state.state.params['pi'][0]['w'],
                                          state.params['pi'][0]['w'])
            np.testing.assert_array_equal(restored_state.obs_mean, norm['mean'])
            np.testing.assert_array_equal(restored_state.obs_var, norm['var'])
            assert restored_state.obs_count == norm['count']
            assert restored_state.lr_scale == 0.5
            assert teacher.restored == curriculum and health_tracker.restored == health
            assert health_tracker.recovered
            assert seen_fields == [field.fingerprint()]
            np.testing.assert_array_equal(
                np.asarray(worlds.reset_key),
                np.asarray(jax.random.fold_in(worlds.key0, 130000)))
            assert worlds.reset_params == {'teacher': 'restored'}
            assert try_restore_recovery(
                Path(td) / 'missing.pkl', state=collapsed,
                governor=teacher, health=health_tracker, env=worlds,
                step=130001, report={'stepping_collapse_streak': 101},
                obs_mean=np.zeros(OBS_DIM), obs_var=np.ones(OBS_DIM), obs_count=1.0,
                task_params_for_field=selected_task, log_prefix='[test]') is None

            # Failure AFTER loading recovery state must not leave a partially
            # rewound teacher when PPO's policy remains unchanged.
            fail_teacher, fail_health = Teacher(), Health()
            original_teacher = fail_teacher.state_dict()
            original_health = fail_health.state_dict()
            def rejected_task(_):
                raise RuntimeError('simulated world reset failure')
            failed = try_restore_recovery(
                path, state=collapsed, governor=fail_teacher,
                health=fail_health, env=worlds, step=130002,
                report={'stepping_collapse_streak': 101},
                obs_mean=np.zeros(OBS_DIM), obs_var=np.ones(OBS_DIM),
                obs_count=1.0, task_params_for_field=rejected_task,
                log_prefix='[test]')
            assert failed is None
            assert fail_teacher.state_dict() == original_teacher
            assert fail_health.state_dict() == original_health
            assert not fail_health.recovered

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
        # E3: runtime event windows are ephemeral. Keeping a world count
        # preserves an in-flight window; changing count resets ALL node,
        # recovery and edge trackers without touching learned field state.
        original_fingerprint = field.fingerprint()
        field._ensure_runtime(3)
        field._node_active[0] = True
        field._rec_active[1] = True
        field._edge_phase[2] = 2
        field._ensure_runtime(3)
        assert field._node_active[0] and field._rec_active[1]
        assert field._edge_phase[2] == 2
        field._ensure_runtime(5)
        assert field._runtime_n == 5
        assert field._node_active.shape == field._rec_active.shape == (5,)
        assert field._edge_source.shape == (5, 3)
        assert not field._node_active.any() and not field._rec_active.any()
        assert not field._edge_phase.any()
        assert field.fingerprint() == original_fingerprint
        clean_resume = CapabilityField.from_state_dict(field.state_dict())
        assert clean_resume._runtime_n == 0
        assert clean_resume._node_active.size == 0
        assert clean_resume.fingerprint() == field.fingerprint()
        # E4: teaching feedback must remain separately measurable. An
        # uninterrupted *unbiased* gait window counts as node evidence.
        # A field-selected frame breaks a window even if the robot tracks
        # perfectly, preventing self-selected practice reinforcing itself.
        frames = 80
        nshape = (frames, 1)
        zeros = np.zeros(nshape, np.float32)
        ones = np.ones(nshape, np.float32)
        cmd = np.full(nshape, 0.04, np.float32)
        ages = np.full(nshape, 2.0, np.float32)
        experience = {
            'cmd_vx': cmd, 'cmd_vy': zeros, 'cmd_yaw': zeros,
            'vx': cmd, 'vy': zeros, 'yaw_rate': zeros,
            'gravity_z': ones, 'support': ones, 'bad_contact': zeros,
            'episode_fell': zeros, 'episode_done': zeros, 'age_s': ages,
            'transition_frac': zeros, 'cmd_resampled': zeros,
            'regime_resilience': zeros, 'birth_resilience_d': zeros,
            'target_cmd_vx': cmd, 'target_cmd_vy': zeros, 'target_cmd_yaw': zeros,
            'transition_ramp_speed': zeros, 'field_taught': zeros,
        }
        clean_field = CapabilityField(half_life_steps=1000.0)
        clean_field.update_from_rollout(experience, step=100,
                                        exclude_field_taught=True)
        assert clean_field.total_node_events > 0
        clean_samples = clean_field.node_evidence().copy()
        assert float(np.sum(clean_samples)) > 0
        # At exactly one half-life, both counts and effective evidence halve.
        clean_field.decay_to(1100)
        np.testing.assert_allclose(clean_field.node_evidence(),
                                   clean_samples * 0.5, rtol=1e-6, atol=1e-7)
        # Field-taught experience can still be seen in a non-isolated arm.
        selected_experience = {**experience, 'field_taught': ones}
        nonisolated = CapabilityField()
        nonisolated.update_from_rollout(selected_experience, step=100,
                                        exclude_field_taught=False)
        assert nonisolated.total_node_events > 0
        isolated = CapabilityField()
        isolated.update_from_rollout(selected_experience, step=100,
                                     exclude_field_taught=True)
        assert isolated.total_node_events == 0
        assert isolated.total_recovery_events == 0
        assert isolated.total_edge_events == 0
        assert float(np.sum(isolated.node_evidence())) == 0.0
        # A tagged frame must BREAK an active window, not just be skipped.
        split_tags = zeros.copy()
        split_tags[frames // 2] = 1.0
        interrupted = CapabilityField()
        interrupted.update_from_rollout({**experience, 'field_taught': split_tags},
                                        step=100, exclude_field_taught=True)
        assert interrupted.total_node_events == 0


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
    assert tuple(parts) == PAYMENT_NAMES
    np.testing.assert_array_equal(actual, original)
    np.testing.assert_array_equal(actual, sum(parts.values()))

def test_training_reward_and_terminal_payment_jit_compile_in_jax():
    import jax
    import jax.numpy as jnp
    from emu.rewards import training_reward, apply_early_fall_penalty

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

    # Mechanical power is currently an EXAM DIAGNOSTIC, not a secret
    # training penalty. Adding large simulated power/torque telemetry
    # must not change PPO's reward or early-fall payment.
    with_load = {
        **m,
        'servo_shaft_abs_power_w': jnp.array([0., 10000.]),
        'servo_over_rated_frac': jnp.array([0., 1.]),
        'spring_potential_energy_j': jnp.array([0., 500.]),
    }
    loaded_reward, loaded_locomotion, loaded_fall = jax.device_get(one_step(with_load))
    np.testing.assert_array_equal(loaded_reward, reward)
    np.testing.assert_array_equal(loaded_locomotion, locomotion)
    np.testing.assert_array_equal(loaded_fall, fall)

def test_training_config_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "config.json"
        original = mature_config(
            "graph", run_dir=Path(td) / "graph", steps=120_000_000,
            seed=811, device="cuda", envs=1024, resume=Path(td) / "ancestor.pkl",
        )
        original.save(path)
        assert TrainingConfig.load(path) == original
    # Real CUDA smoke exposed a post-examination NameError: the per-joint
    # clip diagnostics refer to ACTION_DIM, previously not imported into
    # rollout_metrics.py. Exercise the SAME final-rollout augmentation used
    # after a forced fixed evaluation, without needing an eight-world GPU.
    evidence = {}
    clips = {f'action_clip_j{j}': np.asarray(0.1 * (j + 1), np.float32)
             for j in range(ACTION_DIM)}
    augment_rollout_row(
        evidence, clips, rollout_params_for_metrics={},
        rollout_lr_scale_for_metrics=1.0, norm_diag={},
        latest_eval={'eval_broad_score': -0.554},
        obs_norm_mode='running')
    assert evidence['eval_broad_score'] == -0.554
    assert evidence['rollout_lr_scale'] == 1.0
    for j in range(ACTION_DIM):
        assert np.isclose(evidence[f'ppo_action_clip_j{j}'], 0.1 * (j + 1))
    assert 'ppo_action_clip_j6' not in evidence

    # D5: the KL controller is owned by training health; no change to the
    # finite upper/lower dead band, clamping, or nonfinite hold behaviour.
    adapt = TrainingHealth.adjust_learning_rate
    knobs = dict(target_kl=0.02, factor=1.5, min_scale=0.1, max_scale=2.0)
    assert adapt(0.25, approx_kl=0.02, **knobs) == 0.25
    assert adapt(0.25, approx_kl=0.001, **knobs) == 0.375
    assert adapt(0.25, approx_kl=0.1, **knobs) == 0.25 / 1.5
    assert adapt(2.0, approx_kl=0.001, **knobs) == 2.0
    assert adapt(0.1, approx_kl=0.1, **knobs) == 0.1
    assert adapt(0.25, approx_kl=float('nan'), **knobs) == 0.25

    # H1: a two-stage CUDA smoke must reject different or missing source SHA
    # before evaluating or exporting saved checkpoint objects.
    from scripts.smoke_training_lifecycle import require_pinned_source
    with tempfile.TemporaryDirectory() as scratch:
        folder = Path(scratch)
        for name in ('first', 'continued'):
            run = folder / name
            run.mkdir()
            (run / 'run_meta.json').write_text(
                '{"source_provenance":{"git_commit":"' + 'a'*40 + '"}}',
                encoding='utf-8')
            require_pinned_source(run, 'a'*40)
        (folder / 'continued/run_meta.json').write_text(
            '{"source_provenance":{"git_commit":"' + 'b'*40 + '"}}',
            encoding='utf-8')
        with pytest.raises(RuntimeError, match='source commit differs'):
            require_pinned_source(folder / 'continued', 'a'*40)
        (folder / 'first/run_meta.json').unlink()
        with pytest.raises(RuntimeError, match='source metadata'):
            require_pinned_source(folder / 'first', 'a'*40)

    # E1: settled real rollout evidence still expands physical limits, while
    # the non-adaptive curriculum leaves those limits unchanged.
    training_row = {
        'episodes': 100.0, 'fall_rate': 0.10, 'ep_len_mean': 350.0,
        'core_fall_rate': 0.10, 'core_ep_len_mean': 350.0,
        'fall_rate_steady': 0.10, 'episodes_steady': 100.0,
        'falls_steady': 10.0, 'ep_len_mean_steady': 350.0,
        'foot_mid_cone_steady': 1.0,
        'vx_mean_steady': 0.25, 'cmd_vx_mean_steady': 0.25,
        'precision_vx': 0.9, 'precision_vy': 0.9,
        'precision_yaw': 0.9, 'frontier_precision_vx_forward': 0.9,
    }
    active = RegimeGovernor(open_frontier=True, frontier_vx_hi_safety=1.2)
    tracked = active.update(training_row)
    assert tracked['frontier_event_vx_hi'] == 'expand_edge_proven'
    assert active.params.frontier_vx_hi > 0.60
    inactive = RegimeGovernor(open_frontier=False)
    untracked = inactive.update(training_row)
    assert untracked['frontier_event_vx_hi'] == 'off'
    assert inactive.params.frontier_vx_hi == 0.60

def test_final_publication_atomically_replaces_existing_checkpoint(tmp_path, monkeypatch):
    from emu.checkpoint_io import promote_checkpoint
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


def test_sensor_checkpoint_requires_explicit_schema_and_matching_physical_contract(tmp_path):
    """A 232-dimensional actor is not a 220 actor plus optional float fields."""
    from emu.checkpoint_io import load_checkpoint, SENSOR_CHECKPOINT_SCHEMA
    from emu.contracts import (OBS_SCHEMA_SENSOR, SENSOR_OBS_DIM,
                               sensor_observation_contract)
    from copy import deepcopy
    contract = sensor_observation_contract()
    assert contract == sensor_observation_contract(dict(contract['settings']))
    with pytest.raises(ValueError, match='keys mismatch'):
        sensor_observation_contract({'tof_period_frames': 2})
    params = fake_policy_params()
    for family in ('pi', 'vf'):
        old = params[family][0]['w']
        params[family][0]['w'] = np.zeros((SENSOR_OBS_DIM, old.shape[1]), np.float32)
    state = FakeState(params=params, opt_state={'count': np.asarray(1, np.int32)})
    norm = {'mean': np.zeros(SENSOR_OBS_DIM, np.float32),
            'var': np.ones(SENSOR_OBS_DIM, np.float32), 'count': 1.0}
    kwargs = dict(curriculum_state={'params': {}}, health_state={},
                  field_state={})
    meta = dict(step=2048, lr_scale=.25,
                checkpoint_schema=SENSOR_CHECKPOINT_SCHEMA,
                observation_schema=OBS_SCHEMA_SENSOR,
                obs_dim=SENSOR_OBS_DIM, sensor_contract=contract)
    path = tmp_path / 'new_actor.pkl'
    save_checkpoint(path, state, meta, norm, **kwargs)
    with pytest.raises(ValueError, match='observation schema mismatch'):
        load_checkpoint(path)
    new = load_checkpoint(
        path, expected_observation_schema=OBS_SCHEMA_SENSOR,
        expected_sensor_contract=contract)
    assert new['meta']['obs_dim'] == 232
    changed = dict(contract['settings'])
    changed['flow_period_frames'] = 2
    with pytest.raises(ValueError, match='differs from requested run'):
        load_checkpoint(
            path, expected_observation_schema=OBS_SCHEMA_SENSOR,
            expected_sensor_contract=sensor_observation_contract(changed))
    payload = deepcopy(new)
    payload['meta']['sensor_contract']['settings']['tof_delay_frames'] = 0
    with pytest.raises(ValueError, match='fingerprint'):
        validate_resume_payload(
            payload, expected_observation_schema=OBS_SCHEMA_SENSOR,
            expected_sensor_contract=contract)
    payload = deepcopy(new)
    payload['state']['params']['pi'][0]['w'] = np.zeros((220, 256), np.float32)
    with pytest.raises(ValueError, match='parameter-shape'):
        validate_resume_payload(
            payload, expected_observation_schema=OBS_SCHEMA_SENSOR,
            expected_sensor_contract=contract)
    bad = dict(meta, checkpoint_schema=7)
    with pytest.raises(ValueError, match='observation schema mismatch'):
        save_checkpoint(tmp_path / 'bad_legacy.pkl', state, bad, norm, **kwargs)


def test_old_schema_7_cannot_lie_about_sensor_semantics():
    from emu.checkpoint_io import _validate_observation_meta
    from emu.contracts import OBS_SCHEMA_SENSOR
    assert _validate_observation_meta({'checkpoint_schema': 7}) == 220
    with pytest.raises(ValueError, match='observation schema mismatch'):
        _validate_observation_meta(
            {'checkpoint_schema': 7, 'observation_schema': OBS_SCHEMA_SENSOR})
    with pytest.raises(ValueError, match='sensor metadata inside legacy'):
        _validate_observation_meta({'checkpoint_schema': 7, 'sensor_contract': {}})


def test_v5016_training_sensor_identity_and_schema8_resume(tmp_path):
    """Real on-disk policy rewind accepts only matching 232-observation runs."""
    from emu.contracts import OBS_SCHEMA_SENSOR, OBS_SCHEMA_LEGACY, SENSOR_OBS_DIM
    from emu.training_config import training_sensor_contract
    from emu.checkpoint_io import (
        load_checkpoint, find_valid_resume, resolve_target_checkpoint)
    sensor_cfg = TrainingConfig(observation_schema=OBS_SCHEMA_SENSOR)
    contract = training_sensor_contract(sensor_cfg)
    assert contract['settings']['sensor_noise_scale'] == 1.5
    assert training_sensor_contract(TrainingConfig()) is None
    run_dir = tmp_path / 'sim2real_run'
    checkpoints = run_dir / 'checkpoints'
    checkpoints.mkdir(parents=True)
    params = fake_policy_params()
    for head in ('pi', 'vf'):
        original = params[head][0]['w']
        params[head][0]['w'] = np.zeros((SENSOR_OBS_DIM, original.shape[1]), np.float32)
    state = FakeState(params=params, opt_state={'count': np.asarray(1, np.int32)})
    norm = {'mean': np.zeros(SENSOR_OBS_DIM, np.float32),
            'var': np.ones(SENSOR_OBS_DIM, np.float32), 'count': 12.}
    path = checkpoints / 'latest.pkl'
    save_checkpoint(path, state, {
        'step': 128, 'lr_scale': .25, 'checkpoint_schema': 8,
        'observation_schema': OBS_SCHEMA_SENSOR,
        'obs_dim': SENSOR_OBS_DIM, 'sensor_contract': contract,
    }, norm, curriculum_state={'params': {}}, health_state={},
                    field_state=CapabilityField().state_dict())
    assert find_valid_resume(run_dir) is None
    assert find_valid_resume(
        run_dir, expected_observation_schema=OBS_SCHEMA_SENSOR,
        expected_sensor_contract=contract) == path
    assert load_checkpoint(
        path, expected_observation_schema=OBS_SCHEMA_SENSOR,
        expected_sensor_contract=contract)['meta']['step'] == 128
    finished, partial, boundary = resolve_target_checkpoint(
        run_dir, target_steps=320, envs=2, label='sensor-test',
        expected_observation_schema=OBS_SCHEMA_SENSOR,
        expected_sensor_contract=contract)
    assert finished is None and partial == path and boundary == 320
    resumed, step, completed = resume_training_lineage(
        run_dir, resume='auto', target_steps=320, envs=2,
        expected_observation_schema=OBS_SCHEMA_SENSOR,
        expected_sensor_contract=contract)
    assert step == 128 and not completed and resumed['meta']['obs_dim'] == 232
    with pytest.raises(RuntimeError, match='no valid current-schema'):
        resume_training_lineage(run_dir, resume='auto', target_steps=320, envs=2)
    with pytest.raises(RuntimeError, match='none are valid current-schema'):
        resolve_target_checkpoint(run_dir, target_steps=320, envs=2, label='legacy')
    del OBS_SCHEMA_LEGACY


def test_v5016_sensor_actor_cli_plan_is_explicit(capsys, tmp_path):
    from emu.cli import main
    import json
    run_dir = tmp_path / 'sensor-actor'
    assert main(['train', '--run-dir', str(run_dir), '--profile', 'sim2real',
                 '--sensor-actor', '--steps', '2097152', '--envs', '512',
                 '--device', 'cuda', '--plan']) == 0
    content = capsys.readouterr().out
    config = json.loads(content[content.index('{'):content.rfind('}') + 1])
    assert config['observation_schema'] == 'sim2real_232'
    assert config['steps'] == 2097152
    assert config['envs'] == 512
    assert not run_dir.exists()
    with pytest.raises(ValueError, match='requires --profile sim2real'):
        main(['train', '--run-dir', str(run_dir), '--profile', 'foundation',
              '--sensor-actor', '--steps', '1024', '--envs', '2', '--plan'])
