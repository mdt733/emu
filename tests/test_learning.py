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

def _held_out_core_fixture():
    """Complete summary-shaped subset from independently evaluated panels."""
    from emu.results import FIXED_EXAM_GUARDS
    row={}
    for key,(direction,_) in FIXED_EXAM_GUARDS.items():
        if key.endswith("_survival") or key.endswith("_support") or key.endswith("_evidence"):
            row[key]=.92
        elif key.endswith("_m"):
            row[key]=.003
        else:
            row[key]=.05
    row.update(eval_policy_step=20_000_000,perf_eval_cohorts_used=2,
               perf_eval_env_steps=700_000)
    return row


def test_offline_fixed_exam_compare_uses_only_frozen_exam_csv(tmp_path):
    import csv
    import json
    from emu.results import main
    baseline=_held_out_core_fixture()
    improved=dict(baseline,eval_policy_step=50_000_000,
                  eval_dynamic80_survival=.97)
    a,b=(tmp_path/n for n in ("baseline.csv","candidate.csv"))
    for file,row in ((a,baseline),(b,improved)):
        with file.open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
    output=tmp_path/"evidence.json"
    assert main(["--compare-fixed",str(a),str(b),
                 "--reference-protocol","frozen-physics-1",
                 "--candidate-protocol","frozen-physics-1",
                 "--output",str(output)])==0
    report=json.loads(output.read_text())
    assert report["source"]=="held_out_exam"
    assert report["guardrail_status"]=="no_guardrail_regression"
    assert report["reference_step"]==20_000_000
    assert report["candidate_step"]==50_000_000
    with pytest.raises(ValueError,match="protocol fingerprints"):
        main(["--compare-fixed",str(a),str(b),
              "--reference-protocol","frozen-physics-1",
              "--candidate-protocol","other-world"])
    with pytest.raises(ValueError,match="no fixed-eval evidence"):
        main(["--compare-fixed",str(a),str(tmp_path/"missing.csv"),
              "--reference-protocol","frozen-physics-1",
              "--candidate-protocol","frozen-physics-1"])


def test_independent_exam_detects_core_regression_despite_other_gains():
    from emu.results import compare_fixed_exam_evidence
    baseline=_held_out_core_fixture()
    candidate={**baseline, "eval_policy_step":60_000_000,
               "eval_core_survival":baseline["eval_core_survival"]-.08,
               "eval_core_vx_mae":baseline["eval_core_vx_mae"]+.04,
               "eval_dynamic80_survival":1.0,
               "eval_robust40_survival":1.0}
    out=compare_fixed_exam_evidence(
        baseline,candidate,reference_protocol="frozen-core-v1-physics-abcdef",
        candidate_protocol="frozen-core-v1-physics-abcdef")
    assert out["source"]=="held_out_exam"
    assert out["guardrail_status"]=="regression"
    assert out["regressed_fields"]==["eval_core_survival","eval_core_vx_mae"]
    assert out["metrics"]["eval_dynamic80_survival"]["delta"]>0
    assert out["matched_cohorts"]==2
    assert out["reference_step"]==20_000_000
    assert "ranking" not in out
    assert "reward" not in out


def test_matched_exam_tolerances_and_incomplete_evidence_are_honest():
    from emu.results import compare_fixed_exam_evidence
    baseline=_held_out_core_fixture()
    candidate=dict(baseline,eval_policy_step=30_000_000)
    candidate["eval_core_survival"]-=.05
    out=compare_fixed_exam_evidence(
        baseline,candidate,reference_protocol="exam-A",
        candidate_protocol="exam-A")
    assert out["guardrail_status"]=="no_guardrail_regression"
    assert baseline["eval_core_survival"]==pytest.approx(.92)
    candidate["eval_core_survival"]-=.001
    out=compare_fixed_exam_evidence(
        baseline,candidate,reference_protocol="exam-A",
        candidate_protocol="exam-A")
    assert "eval_core_survival" in out["regressed_fields"]
    with pytest.raises(ValueError,match="fingerprints"):
        compare_fixed_exam_evidence(
            baseline,candidate,reference_protocol="exam-A",
            candidate_protocol="exam-B")
    with pytest.raises(ValueError,match="invalid eval_core_pitch_mae"):
        compare_fixed_exam_evidence(
            baseline,{**candidate,"eval_core_pitch_mae":None},
            reference_protocol="exam-A",candidate_protocol="exam-A")
    no_core=dict(candidate)
    no_core.pop("eval_core_support")
    with pytest.raises(ValueError,match="missing held-out"):
        compare_fixed_exam_evidence(
            baseline,no_core,reference_protocol="exam-A",
            candidate_protocol="exam-A")
    with pytest.raises(ValueError,match="cohort/size mismatch"):
        compare_fixed_exam_evidence(
            baseline,{**candidate,"perf_eval_cohorts_used":1},
            reference_protocol="exam-A",candidate_protocol="exam-A")
    with pytest.raises(ValueError,match="insufficient fixed-exam provenance"):
        compare_fixed_exam_evidence(
            baseline,{**candidate,"perf_eval_env_steps":0},
            reference_protocol="exam-A",candidate_protocol="exam-A")


def test_capability_evidence_keeps_success_quality_and_confidence_separate():
    from emu.results import capability_evidence_report
    field=CapabilityField()
    known=(0,0,0)
    field.node_success[known]=24.
    field.node_failure[known]=6.
    field.node_quality_sum[known]=18.
    before=field.fingerprint()
    report=capability_evidence_report(
        family="commands",source="training_rollout",
        success=field.node_success,failure=field.node_failure,
        lower=field.node_lower(),quality_sum=field.node_quality_sum)
    assert report["known_bins"]==1
    assert report["unknown_bins"]==field.node_success.size-1
    assert report["independent_trials_in_known_bins"]==30.
    assert report["success_rate_known"]==pytest.approx(.8)
    assert report["observed_quality_known"]==pytest.approx(.6)
    assert 0.<report["conservative_success_lower_mean"]<.8
    assert field.fingerprint()==before  # report does not teach or update PPO


def test_capability_field_evidence_projects_existing_three_families_without_mutation():
    from emu.results import capability_field_evidence_snapshot
    field=CapabilityField()
    idx=(0,)*len(field.shape)
    field.node_success[idx]=21.
    field.node_failure[idx]=3.
    field.node_quality_sum[idx]=12.
    field.rec_success[idx+(0,)]=18.
    field.rec_failure[idx+(0,)]=3.
    field.edge_success[idx+(0,)]=22.
    field.edge_failure[idx+(0,)]=2.
    before=field.fingerprint()
    report=capability_field_evidence_snapshot(field,min_events=20)
    assert report["training_only"] is True
    assert report["command_nodes"]["known_bins"]==1
    assert report["command_nodes"]["observed_quality_known"]==pytest.approx(.5)
    assert report["recovery"]["known_bins"]==1
    assert report["recovery"]["observed_quality_known"] is None
    assert report["transitions"]["known_bins"]==1
    assert field.fingerprint()==before


def test_capability_evidence_never_confuses_unknown_with_failure():
    from emu.results import capability_evidence_report
    success=np.array([0.,0.,0.,35.])
    failure=np.array([0.,1.,2.,5.])
    lower=np.array([0.,0.,0.,.72])
    report=capability_evidence_report(
        family="step_up",source="held_out_exam",
        success=success,failure=failure,lower=lower,min_events=20)
    assert report["known_bins"]==1
    assert report["unknown_bins"]==3
    assert report["success_rate_known"]==pytest.approx(.875)
    assert report["observed_quality_known"] is None
    assert report["conservative_success_lower_mean"]==pytest.approx(.72)
    assert "score" not in report
    with pytest.raises(ValueError,match="source"):
        capability_evidence_report(family="step_up",source="reward",
                                   success=success,failure=failure,lower=lower)
    with pytest.raises(ValueError,match="quality"):
        capability_evidence_report(family="step_up",source="held_out_exam",
                                   success=success,failure=failure,lower=lower,
                                   quality_sum=np.ones(4)*50.)


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

def test_terrain_governor_refuses_down_only_success():
    """Count distinct ascent/descent episodes across updates, then reset.

    Small 5%-dose PPO rollouts are sparse; checkpointing the evidence
    window must preserve the same decision after a restored run.
    """
    g = RegimeGovernor()
    g.core_competence = .95
    g.fall_ema['posture'] = .05
    kwargs = dict(step=20_000_001, foundation_steps=20_000_000, max_height_m=.04)
    assert g.advance_terrain({}, **kwargs) == 'terrain_seed_1cm_5pct'
    downhill_only = dict(terrain_attempts_up=6., terrain_successes_up=0.,
                         terrain_falls_up=0., terrain_attempts_down=6.,
                         terrain_successes_down=6., terrain_falls_down=0.)
    for _ in range(3):
        assert g.advance_terrain(downhill_only, **kwargs) == 'terrain_collect_directional_evidence'
    saved = g.state_dict()
    restored = RegimeGovernor()
    restored.load_state_dict(saved)
    assert restored.state_dict() == saved
    assert restored.advance_terrain(downhill_only, **kwargs) == 'terrain_retreat_failed_ascent'
    assert restored.params.terrain_height_m == .01
    assert restored.params.terrain_exposure == .05
    assert not any(restored.terrain_evidence.values())

    # Historical pooled-only rows cannot advance a new frontier.
    assert restored.advance_terrain(dict(terrain_attempts=100.,
                                         terrain_successes=100.), **kwargs) == 'await_directional_terrain_evidence'

    # Weak but genuine ascents can earn extra PRACTICE, NOT a taller step.
    weak_ascent = dict(downhill_only, terrain_successes_up=1.)
    for _ in range(3):
        assert restored.advance_terrain(weak_ascent, **kwargs) == 'terrain_collect_directional_evidence'
    assert restored.advance_terrain(weak_ascent, **kwargs) == 'terrain_exposure_practice_ascent'
    assert restored.params.terrain_exposure == .055
    assert restored.params.terrain_height_m == .01

    # Height is still gated: both directions must genuinely complete >=50%.
    strong_ascent = dict(downhill_only, terrain_successes_up=4.)
    for _ in range(3):
        assert restored.advance_terrain(strong_ascent, **kwargs) == 'terrain_collect_directional_evidence'
    assert restored.advance_terrain(strong_ascent, **kwargs) == 'terrain_exposure_expand_real_cross'
    assert restored.params.terrain_exposure > .055
    assert restored.params.terrain_height_m == .01

    # No episode proof means no improvement, even after a complete window.
    missing = dict.fromkeys(downhill_only, 0.)
    for _ in range(restored.TERRAIN_MAX_WINDOW_UPDATES-1):
        assert restored.advance_terrain(missing, **kwargs) == 'terrain_collect_directional_evidence'
    assert restored.advance_terrain(missing, **kwargs) == 'terrain_window_insufficient_attempts'
    assert not any(restored.terrain_evidence.values())



def test_easier_failures_survive_unrelated_near_frontier_hold():
    """An easy failure window must not starve behind frequent hard probes."""
    g=RegimeGovernor(open_frontier=True,unified_capability_focus=True)
    for _ in range(3):
        row={'command_probe_attempts_vx_hi':8.,
             'command_probe_successes_vx_hi':0.,
             'command_easy_attempts_vx_hi':2.,
             'command_easy_successes_vx_hi':0.}
        g._advance_specialist_commands(row,global_retreat=False,kl_ok=True)
    assert g.command_evidence['vx_hi']['easy_attempts']==6.
    old=g.params.frontier_vx_hi
    g._advance_specialist_commands({
        'command_probe_attempts_vx_hi':8.,
        'command_easy_attempts_vx_hi':2.},
        global_retreat=False,kl_ok=True)
    assert g.params.frontier_vx_hi==old  # seed floor
    assert g.command_evidence['vx_hi']['easy_attempts']==0.
    g.params.frontier_vx_hi=.10
    for _ in range(4):
        g._advance_specialist_commands({
            'command_probe_attempts_vx_hi':8.,
            'command_easy_attempts_vx_hi':2.},
            global_retreat=False,kl_ok=True)
    assert g.params.frontier_vx_hi<.10


def test_v5024_steering_teacher_reallocates_roles_and_is_zero_during_foundation():
    from emu.capability_focus import attention_weights, diamond_commands
    import numpy as np
    b=attention_weights(schedule='balanced',step=40_000_000,foundation_steps=10_000_000)
    s=attention_weights(schedule='steering',step=40_000_000,foundation_steps=10_000_000)
    assert abs(sum(s)-1.0)<1e-8
    assert s[0]>b[0] and s[2]>=b[2] and s[3]>0
    assert attention_weights(schedule='steering',step=1_000_000,
                             foundation_steps=10_000_000)==b
    # Pure axis steering probes are opt-in, never contaminate the
    # historical balanced branch's command distribution.
    import jax
    import jax.numpy as jnp
    keys=jax.random.split(jax.random.PRNGKey(4401),8192)
    args=(jax,jnp,keys,.20,-.10,.08,.30)
    base=np.asarray(diamond_commands(*args,steering_attention=0.))
    steer=np.asarray(diamond_commands(*args,steering_attention=1.))
    pure=lambda a:(np.abs(a)>1e-6).sum(axis=1)==1
    side=lambda a:pure(a)&(np.abs(a[:,1])>1e-6)
    turning=lambda a:pure(a)&(np.abs(a[:,2])>1e-6)
    assert side(steer).sum()>side(base).sum()
    assert turning(steer).sum()>turning(base).sum()


def test_v5024_matched_physical_terrain_matrix_has_complete_directional_coverage():
    from emu.evaluation import unified_terrain_matrix_cases
    cases=unified_terrain_matrix_cases()
    slopes=[x for x in cases if x['kind']=='slope']
    assert len(slopes)==16
    assert {(x['direction'],x['slope_deg']) for x in slopes}=={
        (direction,deg)
        for direction in ('uphill','downhill','cross_left','cross_right')
        for deg in (1.,2.,3.,5.)
    }
    assert {x['step_m'] for x in cases if x['kind']=='step'}=={
        .003,.005,.01,.02}


def test_v5024_slope_stability_advancement_does_not_borrow_tracking_evidence():
    from emu.curriculum import RegimeGovernor
    g=RegimeGovernor(open_frontier=True,unified_capability_focus=True)
    assert g.advance_slope({})=='seed_real_incline_0p7deg'
    start=g.params.slope_max_deg
    # Excellent command tracking without measured supported movement is
    # insufficient: only the distinct stability counters can promote.
    old={f'slope_near_{kind}_{d}':8. for d in g.SLOPE_DIRECTIONS
         for kind in ('attempts','successes')}
    assert g.advance_slope(old)=='hold_probe_learning'
    assert g.params.slope_max_deg==start
    real={f'slope_near_attempts_{d}':8. for d in g.SLOPE_DIRECTIONS}
    real.update({f'slope_near_stability_successes_{d}':8.
                 for d in g.SLOPE_DIRECTIONS})
    assert g.advance_slope(real)=='expand_real_slope_angle'
    assert g.params.slope_max_deg>start


def test_slope_local_dial_requires_real_severity_and_retreats_on_easy_failure():
    """Low-severity success must never certify the harder incline target."""
    g=RegimeGovernor(open_frontier=True,unified_capability_focus=True)
    assert g.advance_slope({})=='seed_real_incline_0p7deg'
    old=g.params.slope_max_deg
    # Flat/easy successes alone cannot make the grade grow.
    easy={f'slope_easy_{kind}_{direction}':8.
          for direction in g.SLOPE_DIRECTIONS
          for kind in ('attempts','stability_successes')}
    assert g.advance_slope(easy)=='collect_real_slope'
    assert g.params.slope_max_deg==old
    # Near-target success across all four bearings can nudge it upward.
    near={f'slope_near_{kind}_{direction}':8.
          for direction in g.SLOPE_DIRECTIONS
          for kind in ('attempts','stability_successes')}
    assert g.advance_slope(near)=='expand_real_slope_angle'
    assert g.params.slope_max_deg>old
    # A hard failure on its own is NOT evidence to shrink.
    hard={f'slope_near_attempts_{direction}':8.
          for direction in g.SLOPE_DIRECTIONS}
    assert g.advance_slope(hard)=='hold_probe_learning'
    raised=g.params.slope_max_deg
    assert raised>old
    # But repeated trouble with an easier real incline is.
    weak={'slope_easy_attempts_cross_left':8.,
          'slope_easy_stability_successes_cross_left':0.}
    assert g.advance_slope(weak)=='retreat_easy_incline_failure'
    assert old<=g.params.slope_max_deg<raised
    stored=g.state_dict()
    restored=RegimeGovernor(unified_capability_focus=True)
    restored.load_state_dict(stored)
    assert restored.state_dict()==stored
    # Legacy source can be restored but old pooled success has no claimed
    # near-grade proof; its evidence is reset rather than misattributed.
    stored['slope_evidence']={k:{'attempts':9.,'successes':9.}
                              for k in g.SLOPE_DIRECTIONS}
    restored.load_state_dict(stored)
    assert not any(v['near_attempts'] for v in restored.slope_evidence.values())


def test_unified_step_dial_needs_supported_crossing_at_true_height():
    """Easy steps teach, near-maximum ascents and descents prove a height."""
    g=RegimeGovernor(open_frontier=True,unified_capability_focus=True)
    opts=dict(step=20_000_001,max_height_m=.04,
              foundation_steps=20_000_000,unified=True)
    assert g.advance_terrain({},**opts)=='terrain_unified_seed_4mm'
    start=g.params.terrain_height_m
    easy={f'terrain_easy_{outcome}_{direction}':8.
          for direction in ('up','down') for outcome in ('attempts','successes')}
    assert g.advance_terrain(easy,**opts)=='terrain_collect_real_step'
    assert g.params.terrain_height_m==start
    near={f'terrain_near_{outcome}_{direction}':8.
          for direction in ('up','down') for outcome in ('attempts','successes')}
    assert g.advance_terrain(near,**opts)=='terrain_height_expand_real_cross'
    assert g.params.terrain_height_m>start
    high=g.params.terrain_height_m
    # Eight failed harder probes are learning opportunities, not immediate
    # evidence to shrink the only step-up training target.
    assert g.advance_terrain(
        {'terrain_near_attempts_up':8.,'terrain_near_attempts_down':8.},
        **opts)=='terrain_hold_probe_learning'
    assert g.params.terrain_height_m==high
    assert g.advance_terrain(
        {'terrain_easy_attempts_up':8.,
         'terrain_easy_successes_up':0.},
        **opts)=='terrain_retreat_easy_failure'
    assert start<=g.params.terrain_height_m<high
    saved=g.state_dict()
    again=RegimeGovernor(unified_capability_focus=True)
    again.load_state_dict(saved)
    assert again.state_dict()==saved
    # Older pooled-only evidence must not be interpreted as real near-height
    # proof, while the rest of the checkpoint remains intact.
    pooled=dict(saved)
    pooled['terrain_evidence']={k:v for k,v in saved['terrain_evidence'].items()
                                if k.endswith('_falls') or
                                   k in ('up_attempts','up_successes',
                                         'down_attempts','down_successes','updates')}
    again.load_state_dict(pooled)
    assert again.terrain_evidence['up_near_attempts']==0.


def test_unified_same_source_target_extension_requires_completed_final(monkeypatch,tmp_path):
    """A later larger target can continue without a checkpoint-era fork."""
    from dataclasses import replace
    from emu import run as runmod
    from emu.training_config import TrainingConfig
    from emu.checkpoint_io import update_boundary_at_or_after

    root=tmp_path/'growth'
    (root/'checkpoints').mkdir(parents=True)
    prior=TrainingConfig(
        run_dir=str(root),steps=40_000_000,envs=2048,seed=733,
        device='cuda',unified_capability_focus=True,
        observation_schema='terrain_360',terrain_step_max_m=.04)
    prior.save(root/'training_config.json')
    sha='a'*40
    (root/'run_meta.json').write_text(
        '{"source_provenance":{"git_commit":"' + sha + '"}}')
    final=root/'checkpoints/final.pkl'
    final.write_bytes(b'source-pinned valid checkpoint simulated by mock')
    new=replace(prior,steps=120_000_000)
    monkeypatch.setenv('EMU_UNIFIED_CONTINUOUS','cold-growth-approved')
    monkeypatch.setattr(runmod,'source_repo_root',lambda:None)
    monkeypatch.setattr(runmod,'resolve_target_checkpoint',
        lambda *a,**k:(None,final,update_boundary_at_or_after(new.steps,new.envs)))
    from emu import checkpoint_io
    monkeypatch.setattr(checkpoint_io,'load_checkpoint',
        lambda *a,**k:{'meta':{
            'step':update_boundary_at_or_after(prior.steps,prior.envs),
            'source_provenance':{'git_commit':sha}}})
    prov={'available':True,'commit':sha,'dirty':False}
    path,complete=runmod.inspect_run(new,provenance=prov,resume='auto')
    assert path==final and not complete
    with pytest.raises(RuntimeError,match='configuration differs'):
        runmod.inspect_run(replace(new,seed=5),provenance=prov,resume='auto')
    with pytest.raises(RuntimeError,match='configuration differs'):
        runmod.inspect_run(new,provenance=prov,resume='')
    with pytest.raises(RuntimeError,match='configuration differs'):
        runmod.inspect_run(replace(prior,steps=39_000_000),
                           provenance=prov,resume='auto')
    final.unlink()
    with pytest.raises(RuntimeError,match='configuration differs'):
        runmod.inspect_run(new,provenance=prov,resume='auto')


def test_v5024_campaign_is_standalone_source_pinned_and_plan_only():
    """No retired overnight launcher is needed to plan the current study."""
    import runpy
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    script=root/'scripts/v5024_campaign.py'
    source=script.read_text(encoding='utf-8')
    assert 'from overnight_teaching' not in source
    scope=runpy.run_path(str(script),run_name='__campaign_import_test__')
    assert scope['BRANCH']=='feature/v5024-release-polish'
    arms=scope['candidates'](733,60_000_000)
    assert len(arms)==3
    assert sum(a['steps'] for a in arms)==180_000_000
    assert [(a['recipe'],a['schedule']) for a in arms]==[
        ('main','balanced'),('command-heavy','balanced'),
        ('command-heavy','steering')]
    cmd=scope['cmd_for'](arms[2],Path('/tmp/campaign'),733,10_000_000,False)
    assert '--resume' not in cmd
    assert cmd[cmd.index('--foundation-steps')+1]=='10000000'
    assert cmd[cmd.index('--teaching-schedule')+1]=='steering'
    assert cmd[cmd.index('--steps')+1]=='60000000'
    assert cmd[cmd.index('--envs')+1]=='2048'
    assert scope['cmd_for'](
        arms[0],Path('/tmp/campaign'),733,10_000_000,True)[-2:]==[
        '--resume','auto']
    assert 'PLAN ONLY. --run is required' in source
    assert 'v5024_polish_study' in source


def test_dynamic_attention_schedule_stays_normalised_and_changes_smoothly():
    from emu.capability_focus import attention_weights,FOCUS_NAMES
    assert len(FOCUS_NAMES)==6
    for name in ('balanced','staged','terrain-early'):
        for age in (0,9_999_999,10_000_000,12_000_000,
                    20_000_000,30_000_000,50_000_000,80_000_000):
            w=attention_weights(schedule=name,step=age,
                                foundation_steps=10_000_000)
            assert len(w)==6 and min(w)>=0.
            assert sum(w)==pytest.approx(1.)
    first=attention_weights(schedule='staged',step=10_000_001,
                            foundation_steps=10_000_000)
    middle=attention_weights(schedule='staged',step=30_000_000,
                             foundation_steps=10_000_000)
    late=attention_weights(schedule='staged',step=50_000_000,
                           foundation_steps=10_000_000)
    assert first[0]>middle[0]>late[0]
    assert first[3]<1e-6 and middle[3]>0. and late[3]>middle[3]
    assert attention_weights(schedule='balanced',step=80_000_000,
                             foundation_steps=10_000_000)[0]==.20
    for bad in ('', 'invalid', 'main'):
        with pytest.raises(ValueError,match='teaching allocation'):
            attention_weights(schedule=bad,step=0,foundation_steps=10_000_000)


def test_unified_continuous_launch_one_source_and_same_run_only(monkeypatch, tmp_path):
    """One cold foundation continues indefinitely without special 40M state."""
    from dataclasses import replace
    from types import SimpleNamespace
    from emu.run import one_run_config
    from emu.training import unified_launch_mode

    root=tmp_path/'cold-unified'
    args=SimpleNamespace(run_dir=str(root),profile='sim2real',seed=733,
                         device='cuda',envs=2048,steps=120_000_000,
                         unified_capabilities=True,terrain_actor=True,
                         terrain_step_max_m=.04)
    cfg=one_run_config(args)
    monkeypatch.delenv('EMU_UNIFIED_CONTINUOUS',raising=False)
    with pytest.raises(RuntimeError,match='PRE-LAUNCH'):
        unified_launch_mode(cfg)
    monkeypatch.setenv('EMU_UNIFIED_CONTINUOUS','cold-growth-approved')
    assert unified_launch_mode(cfg)=='continuous'
    assert unified_launch_mode(replace(cfg,steps=20_000_000))=='continuous'
    assert unified_launch_mode(replace(cfg,resume='auto'))=='continuous'
    assert unified_launch_mode(replace(
        cfg,resume=str(root/'checkpoints/latest.pkl')))=='continuous'
    # Short cold starts and alternate seeds are now intentional experiments.
    assert unified_launch_mode(replace(cfg,steps=10_000_000,seed=42))=='continuous'
    for invalid in (dict(steps=999_999),dict(envs=8),dict(seed=-1),
                    dict(device='cpu'),dict(terrain_step_max_m=.05),
                    dict(resume=str(tmp_path/'other/checkpoints/final.pkl')),
                    dict(observation_schema='legacy_220'),
                    dict(motor_guidance_weight=.02),
                    dict(domain_randomization=True)):
        with pytest.raises(RuntimeError,match='PRE-LAUNCH'):
            unified_launch_mode(replace(cfg,**invalid))


def test_unified_overnight_guard_allows_only_explicit_bounded_acceptance(monkeypatch):
    """The lifecycle probe cannot silently turn into an overnight campaign."""
    from dataclasses import replace
    from emu.training import train
    from emu.training_config import TrainingConfig, ROLLOUT_STEPS
    from emu.contracts import OBS_SCHEMA_TERRAIN
    cfg=replace(TrainingConfig(), unified_capability_focus=True,
                observation_schema=OBS_SCHEMA_TERRAIN, terrain_step_max_m=.04,
                device='cuda', envs=8, open_frontier=True, steps=8*ROLLOUT_STEPS)
    monkeypatch.delenv('EMU_UNIFIED_ACCEPTANCE', raising=False)
    with pytest.raises(RuntimeError, match='PRE-LAUNCH'):
        train(cfg)
    monkeypatch.setenv('EMU_UNIFIED_ACCEPTANCE','bounded-ppo-lifecycle')
    with pytest.raises(RuntimeError, match='PRE-LAUNCH'):
        train(replace(cfg,steps=3*8*ROLLOUT_STEPS))
    with pytest.raises(RuntimeError, match='PRE-LAUNCH'):
        train(replace(cfg,envs=2048,steps=3*2048*ROLLOUT_STEPS))
    with pytest.raises(RuntimeError, match='PRE-LAUNCH'):
        train(replace(cfg,observation_schema='legacy_220'))
    # The bounded positive path runs through real CUDA in the existing
    # smoke_training_lifecycle.py --unified stages, not this CPU unit test.


def test_external_fork_only_inherits_same_source_recovery(monkeypatch, tmp_path):
    """Never bring the old curriculum back via an incompatible rollback."""
    import emu.checkpoint_io as io
    old = tmp_path / 'ancestor/checkpoints'
    old.mkdir(parents=True)
    (old / 'final.pkl').write_bytes(b'fake checkpoint for source-gating unit test')
    (old / 'recovery.pkl').write_bytes(b'fake rollback for source-gating unit test')
    parent_payload = {'meta': {
        'step': 128, 'source_provenance': {'git_commit': 'old-source'}}}
    recovery_payload = {'meta': {
        'step': 64, 'source_provenance': {'git_commit': 'old-source'}}}
    def fake_load(path, **kwargs):
        return recovery_payload if Path(path).name == 'recovery.pkl' else parent_payload
    monkeypatch.setattr(io, 'load_checkpoint', fake_load)

    monkeypatch.setenv('EMU_GIT_COMMIT', 'new-source')
    cross = tmp_path / 'cross-source'
    loaded, step, done = io.resume_training_lineage(
        cross, resume=str(old / 'final.pkl'), target_steps=256, envs=2)
    assert step == 128 and not done and loaded is parent_payload
    assert not (cross / 'checkpoints/recovery.pkl').exists()

    monkeypatch.setenv('EMU_GIT_COMMIT', 'old-source')
    same = tmp_path / 'same-source'
    io.resume_training_lineage(
        same, resume=str(old / 'final.pkl'), target_steps=256, envs=2)
    assert (same / 'checkpoints/recovery.pkl').read_bytes() == (
        old / 'recovery.pkl').read_bytes()

    # Even with a same-source parent, a rollback with different embedded
    # provenance is rejected before it can poison the new run.
    recovery_payload['meta']['source_provenance']['git_commit'] = 'other-source'
    mixed = tmp_path / 'mixed-recovery'
    io.resume_training_lineage(
        mixed, resume=str(old / 'final.pkl'), target_steps=256, envs=2)
    assert not (mixed / 'checkpoints/recovery.pkl').exists()


def test_two_teaching_profiles_keep_original_gait_training_alive():
    """The original adaptive allocation remains one selectable recipe."""
    from emu.capability_focus import teaching_profile
    from emu.training_config import FOUNDATION_STEPS
    g=RegimeGovernor(open_frontier=True,unified_capability_focus=True)
    assert g.BOOTSTRAP_UPDATES==70
    profile=teaching_profile(step=0, foundation_steps=FOUNDATION_STEPS,
                             steady_difficulty=g.params.steady_difficulty)
    assert profile.name=='foundation' and not profile.adaptive_specialists
    assert profile.specialist_fraction==0.0
    for n in range(70):
        g.advance_curriculum({},step=n*65536,max_height_m=.04,
                             foundation_steps=FOUNDATION_STEPS)
    assert g.params.regime_frac_steady==1.0
    assert g.params.regime_frac_transition==0.0
    assert g.params.regime_frac_resilience==0.0
    g.advance_curriculum({},step=70*65536,max_height_m=.04,
                         foundation_steps=FOUNDATION_STEPS)
    assert g.params.regime_frac_transition>0.0
    assert g.params.regime_frac_resilience>0.0
    assert not teaching_profile(
        step=FOUNDATION_STEPS, foundation_steps=FOUNDATION_STEPS,
        steady_difficulty=.8).adaptive_specialists
    growing=teaching_profile(
        step=FOUNDATION_STEPS+5_000_000,
        foundation_steps=FOUNDATION_STEPS,steady_difficulty=.8)
    assert growing.name=='growth'
    assert growing.adaptive_specialists
    assert growing.specialist_fraction==pytest.approx(.5)


def test_foundation_trials_choose_lessons_without_changing_learner():
    """Experiments can improve basic gait without a second PPO path."""
    from emu.capability_focus import teaching_profile,FOUNDATION_RECIPES
    from emu.run import one_run_config
    from emu.training import unified_launch_mode
    from types import SimpleNamespace
    from dataclasses import replace

    assert set(FOUNDATION_RECIPES)=={'main','command-heavy','curious'}
    for recipe in FOUNDATION_RECIPES:
        for boundary in (10_000_000,20_000_000):
            profile=teaching_profile(step=boundary-1,
                 foundation_steps=boundary,foundation_mix=recipe,
                 steady_difficulty=.4)
            assert profile.name=='foundation' and not profile.adaptive_specialists
            assert profile.specialist_fraction==0.
            assert (profile.regime_mix is None) == (recipe=='main')
            assert profile.foundation_control_mix==(.5 if recipe=='curious' else 0.)
            after=teaching_profile(step=boundary+5_000_000,
                 foundation_steps=boundary,foundation_mix=recipe,
                 steady_difficulty=.4)
            assert after.name=='growth'
            assert after.specialist_fraction==pytest.approx(.5)
            assert after.foundation_control_mix==0.
            assert after.regime_mix is None
    args=SimpleNamespace(
         run_dir='/tmp/emu_experiment_recipe',profile='sim2real',
         seed=734,device='cuda',envs=2048,steps=10_000_000,
         unified_capabilities=True,terrain_actor=True,
         terrain_step_max_m=.04, foundation_steps=10_000_000,
         foundation_mix='curious')
    cfg=one_run_config(args)
    assert cfg.foundation_steps==10_000_000 and cfg.foundation_mix=='curious'
    assert cfg.observation_schema=='terrain_360'
    assert cfg.ppo_lr==pytest.approx(2e-4)
    with pytest.MonkeyPatch.context() as m:
        m.setenv('EMU_UNIFIED_CONTINUOUS','cold-growth-approved')
        assert unified_launch_mode(cfg)=='continuous'
    assert one_run_config(SimpleNamespace(
        **{**vars(args),'steps':120_000_000})).foundation_mix=='curious'
    with pytest.raises(ValueError,match='foundation'):
        one_run_config(SimpleNamespace(**{**vars(args),'foundation_steps':7_000_000}))
    with pytest.raises(ValueError,match='foundation'):
        one_run_config(SimpleNamespace(**{**vars(args),'foundation_mix':'mystery'}))


def test_foundation_frozen_main_learning_recipe_and_rewards():
    """Static term-for-term reference from main SHA 2ec9e9db.

    Numerical parity of the *recipe* does NOT imply 360-observation actor
    parity with the 232-input reference. That still requires a GPU exam.
    """
    from emu.training_config import (
        ROLLOUT_STEPS,PPO_EPOCHS,PPO_MINIBATCHES,PPO_LR,
        FOUNDATION_STEPS,foundation_config,
    )
    from emu.rewards import weighted_training_terms,PAYMENT_NAMES
    from dataclasses import replace
    from pathlib import Path
    from emu.run import one_run_config
    from types import SimpleNamespace
    assert (ROLLOUT_STEPS,PPO_EPOCHS,PPO_MINIBATCHES,PPO_LR)==(32,4,8,2e-4)
    assert FOUNDATION_STEPS==20_000_000
    ref=foundation_config(run_dir=Path('/tmp/emu_foundation_reference'),
                          seed=733,device='cuda',envs=2048)
    candidate=one_run_config(SimpleNamespace(
        run_dir='/tmp/emu_unified_reference',profile='sim2real',
        seed=733,device='cuda',envs=2048,steps=120_000_000,
        unified_capabilities=True,terrain_actor=True,terrain_step_max_m=.04))
    # These are the physical core inputs to the foundation task sampler,
    # not a claim of equivalent actor observations or action trajectories.
    for key in ('vx_hi_max','vx_lo_min','vy_max_mps','yaw_max_rad_s',
                'phase_speed_gain','regime_command_frac','difficulty_base_step',
                'capability_field_mode','cloud_probes_per_eval',
                'cloud_final_probes','motor_guidance_weight'):
        assert getattr(candidate,key)==getattr(ref,key),key
    m=dict(alive=.83,locomotion=.44,posture=.51,contact=.62,
           anti_slip=.7,clearance=.32,smoothness=.95,
           joint_safety=.81,stepping_gate=.9)
    total,terms=weighted_training_terms(**m)
    baseline={
        'alive':.58*m['alive'],
        'locomotion':1.55*m['locomotion'],
        'posture':.68*m['posture']*m['stepping_gate'],
        'contact':.50*m['contact']*m['stepping_gate'],
        'anti_slip':.18*m['anti_slip'],
        'clearance':.16*m['clearance'],
        'smoothness':.12*m['smoothness'],
        'joint_safety':.17*m['joint_safety']}
    assert tuple(PAYMENT_NAMES)==tuple(baseline)
    for term,value in baseline.items():
        assert terms[term]==pytest.approx(value),term
    assert total==pytest.approx(sum(baseline.values()))


def test_unified_specialist_blend_then_independent_bidirectional_targets():
    """Before 20M: simple gait. Afterwards: slow exposure, local evidence."""
    from emu.capability_focus import growth_mix, focus_indices, FOCUS_NAMES
    g=RegimeGovernor(open_frontier=True,unified_capability_focus=True)
    start=20_000_000
    for step,share in ((0,0.),(start,0.),(start+2_500_000,.25),
                       (start+5_000_000,.50),
                       (start+10_000_000,1.0),(start+40_000_000,1.0)):
        assert growth_mix(step=step,foundation_steps=start,
                          steady_difficulty=0.)==pytest.approx(share)
    assert np.bincount(focus_indices(2048),minlength=len(FOCUS_NAMES)).min()>=200
    before=g.params.frontier_vx_hi
    frozen=g.advance_curriculum(
        {'command_probe_attempts_vx_hi':8.,
         'command_probe_successes_vx_hi':8.},
        step=start-1,foundation_steps=start,max_height_m=.04)
    assert frozen['terrain_event']=='foundation_flat'
    assert frozen['slope_event']=='foundation_flat'
    assert g.params.frontier_vx_hi==before
    assert g.params.terrain_height_m==0.
    assert g.params.slope_max_deg==0.
    seeded=g.advance_curriculum({},step=start,
             foundation_steps=start,max_height_m=.04)
    assert seeded['terrain_event']=='terrain_unified_seed_4mm'
    assert seeded['slope_event']=='seed_real_incline_0p7deg'
    grown=g.advance_curriculum(
        {'command_probe_attempts_vx_hi':8.,
         'command_probe_successes_vx_hi':8.},
        step=start+65_536,foundation_steps=start,max_height_m=.04)
    assert g.params.frontier_vx_hi>before
    higher=g.params.frontier_vx_hi
    g.advance_curriculum(
        {'command_easy_attempts_vx_hi':8.,
         'command_easy_successes_vx_hi':0.},
        step=start+131_072,foundation_steps=start,max_height_m=.04)
    assert before<=g.params.frontier_vx_hi<higher
    assert g.command_evidence['vx_hi']['easy_attempts']==0.
    # An old checkpoint's near-only window remains resumable; unknown
    # state shape is still rejected by the strict contract.
    old=g.state_dict()
    for v in old['command_evidence'].values():
        del v['easy_attempts']
        del v['easy_successes']
    restored=RegimeGovernor(open_frontier=True,unified_capability_focus=True)
    restored.load_state_dict(old)
    assert restored.command_evidence['vx_hi']['easy_attempts']==0.
    assert restored.state_dict()['params']==g.state_dict()['params']



def test_unified_curriculum_report_schema_stable_at_foundation_boundary(tmp_path):
    """No new metric keys suddenly appearing when specialists first activate."""
    from emu.results import append_metrics_csv
    g=RegimeGovernor(open_frontier=True,unified_capability_focus=True)
    early=g.advance_curriculum({},step=0,
            max_height_m=.04,foundation_steps=20_000_000)
    mature=g.advance_curriculum({},step=20_054_016,
            max_height_m=.04,foundation_steps=20_000_000)
    assert set(early)==set(mature)
    assert early['terrain_event']=='foundation_flat'
    assert mature['terrain_event']=='terrain_unified_seed_4mm'
    assert early['terrain_exposure']==0.
    assert mature['terrain_exposure']>0.
    keep=('terrain_exposure','terrain_height_m',
          'slope_exposure','slope_max_deg','frontier_vx_hi')
    csv_file=tmp_path/'metrics.csv'
    append_metrics_csv(csv_file,dict(step=65_536,**{k:early[k] for k in keep}))
    append_metrics_csv(csv_file,dict(step=20_054_016,**{k:mature[k] for k in keep}))
    assert len(csv_file.read_text().splitlines())==3


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


def test_unified_fixed_eval_uses_same_array_sensor_and_slab_world(monkeypatch):
    """Held-out scores may not silently use old-tripod 360 geometry."""
    from emu import evaluation
    from emu.contracts import (
        OBS_SCHEMA_TERRAIN,SENSOR_SETTINGS_DEFAULTS,sensor_observation_contract,
        TERRAIN_ARRAY_ONLY_REVISION,TERRAIN_SENSOR_REVISION)
    from emu.training_config import TrainingConfig,training_sensor_contract

    built=[]
    class StubWorld:
        def __init__(self, _xml, **kw):
            built.append(kw)
            self.sensor_contract=sensor_observation_contract(
                dict(SENSOR_SETTINGS_DEFAULTS,
                     sensor_noise_scale=kw['sensor_noise_scale']),
                schema=kw['observation_schema'],
                array_only=kw.get('unified_capability_focus',False))
    monkeypatch.setattr(evaluation,'EmuMJXEnv',StubWorld)
    monkeypatch.setattr(evaluation,'make_trainer',
                        lambda env,**kw: {'stub':True})

    new=TrainingConfig(observation_schema=OBS_SCHEMA_TERRAIN,
                       unified_capability_focus=True,
                       terrain_step_max_m=.04,envs=8)
    worlds=evaluation.build_worlds(new)
    assert len(worlds)==5
    assert len(built)==5
    assert all(item['unified_capability_focus'] is True and
               item['terrain_step_max_m']==.04 for item in built)
    assert all(env.sensor_contract==training_sensor_contract(new)
               for env,_ in worlds.values())
    assert all(env.sensor_contract['sensor_model_revision']==TERRAIN_ARRAY_ONLY_REVISION
               for env,_ in worlds.values())

    # Historic 360 actors preserve their separate tripod observation meaning.
    built.clear()
    old=TrainingConfig(observation_schema=OBS_SCHEMA_TERRAIN,
                       unified_capability_focus=False,
                       terrain_step_max_m=.04,envs=8)
    previous=evaluation.build_worlds(old)
    assert len(built)==5
    assert all(item['unified_capability_focus'] is False and
               item['terrain_step_max_m']==0.0 for item in built)
    assert all(env.sensor_contract['sensor_model_revision']==TERRAIN_SENSOR_REVISION
               for env,_ in previous.values())


def test_array_only_360_checkpoint_save_resume_keeps_sensor_geometry(tmp_path):
    """New 360 array-only and archived tripod 360 must never be swapped."""
    from copy import deepcopy
    from emu.contracts import (
        OBS_SCHEMA_TERRAIN, TERRAIN_OBS_DIM, TERRAIN_ARRAY_ONLY_REVISION,
        sensor_observation_contract)
    from emu.training_config import TrainingConfig, training_sensor_contract
    from emu.checkpoint_io import (
        save_checkpoint, load_checkpoint, validate_resume_payload,
        TERRAIN_CHECKPOINT_SCHEMA)

    cfg = TrainingConfig(observation_schema=OBS_SCHEMA_TERRAIN,
                         unified_capability_focus=True)
    array = training_sensor_contract(cfg)
    tripod = sensor_observation_contract(array['settings'], schema=OBS_SCHEMA_TERRAIN)
    assert array['sensor_model_revision'] == TERRAIN_ARRAY_ONLY_REVISION
    assert array != tripod
    assert array['sha256'] != tripod['sha256']
    assert 'tripod_mounts_body_m' not in array
    assert 'tripod_mounts_body_m' in tripod

    params=fake_policy_params()
    for family in ('pi','vf'):
        old=params[family][0]['w']
        params[family][0]['w']=np.zeros((TERRAIN_OBS_DIM,old.shape[1]),np.float32)
    state=FakeState(params=params,opt_state={'count': np.asarray(1,np.int32)})
    norm={'mean': np.zeros(TERRAIN_OBS_DIM,np.float32),
          'var': np.ones(TERRAIN_OBS_DIM,np.float32),'count': 8.0}
    kwargs=dict(curriculum_state={'params': {}},health_state={},field_state={})
    meta=dict(step=256,lr_scale=.25,
              checkpoint_schema=TERRAIN_CHECKPOINT_SCHEMA,
              observation_schema=OBS_SCHEMA_TERRAIN,
              obs_dim=TERRAIN_OBS_DIM,sensor_contract=array)

    checkpoint=tmp_path/'array_only_360.pkl'
    save_checkpoint(checkpoint,state,meta,norm,**kwargs)
    saved=load_checkpoint(checkpoint,
                          expected_observation_schema=OBS_SCHEMA_TERRAIN,
                          expected_sensor_contract=array)
    assert saved['meta']['sensor_contract']==array
    # Exercise the SAME script that runs on the owner's WSL system.
    # Formerly it implicitly loaded schema-7 and rejected a correctly
    # saved terrain_360 checkpoint AFTER wasting the actual CUDA update.
    from scripts.smoke_training_lifecycle import checkpoint as acceptance_checkpoint
    assert acceptance_checkpoint(checkpoint,256,config=cfg)['meta']['step']==256
    with pytest.raises(ValueError,match='differs from requested run'):
        acceptance_checkpoint(
            checkpoint,256,config=TrainingConfig(
                observation_schema=OBS_SCHEMA_TERRAIN,
                unified_capability_focus=False))
    with pytest.raises(ValueError,match='differs from requested run'):
        load_checkpoint(checkpoint,
                        expected_observation_schema=OBS_SCHEMA_TERRAIN,
                        expected_sensor_contract=tripod)
    with pytest.raises(ValueError,match='observation schema mismatch'):
        load_checkpoint(checkpoint)

    # The historical 360 tripod contract remains separately readable.
    old_path=tmp_path/'tripod_360.pkl'
    save_checkpoint(old_path,state,dict(meta,sensor_contract=tripod),norm,**kwargs)
    load_checkpoint(old_path,expected_observation_schema=OBS_SCHEMA_TERRAIN,
                    expected_sensor_contract=tripod)
    with pytest.raises(ValueError,match='differs from requested run'):
        load_checkpoint(old_path,expected_observation_schema=OBS_SCHEMA_TERRAIN,
                        expected_sensor_contract=array)

    # Even if the caller expects the tampered manifest, its independent
    # canonical geometry/hash must still fail validation.
    altered=deepcopy(saved)
    altered['meta']['sensor_contract']['tof4x4_yaw_deg'][0]=-10.0
    with pytest.raises(ValueError,match='fingerprint/geometry mismatch'):
        validate_resume_payload(
            altered,expected_observation_schema=OBS_SCHEMA_TERRAIN,
            expected_sensor_contract=altered['meta']['sensor_contract'])


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
    assert main(['train', '--run-dir', str(run_dir), '--profile', 'foundation',
                 '--sensor-actor', '--pitch-deg', '15', '--steps', '1024',
                 '--envs', '2', '--plan']) == 0
    output = capsys.readouterr().out
    foundation = json.loads(output[output.index('{'):output.rfind('}') + 1])
    assert foundation['observation_schema'] == 'sim2real_232'
    assert foundation['open_frontier'] is False
    assert foundation['pitch_command_max_deg'] == 15.
    with pytest.raises(ValueError, match='requires --profile foundation or sim2real'):
        main(['train', '--run-dir', str(run_dir), '--profile', 'mature',
              '--sensor-actor', '--steps', '1024', '--envs', '2', '--plan'])


def test_v5016_schema8_viewer_bundle_roundtrip(tmp_path):
    """Schema-8 export must be valid for BOTH browser and frozen MJX examiner."""
    import hashlib
    import json
    from emu.contracts import (OBS_SCHEMA_SENSOR, SENSOR_OBS_DIM, ROBOT_XML,
                               sensor_observation_contract)
    from emu.viewer_server import build_bundle, load_exported_actor_for_exam
    contract = sensor_observation_contract({
        "tof_period_frames":2,"flow_period_frames":1,
        "tof_delay_frames":1,"flow_delay_frames":1,
        "tof_dropout_prob":.01,"flow_dropout_prob":.02,
        "gyro_bias_std_rad_s":.015,"sensor_noise_scale":1.5,
    })
    params = fake_policy_params()
    for head in ('pi','vf'):
        original=params[head][0]['w']
        params[head][0]['w']=np.zeros((SENSOR_OBS_DIM,original.shape[1]),np.float32)
    saved=tmp_path/'checkpoints'/'final.pkl'
    norm={'mean':np.zeros(SENSOR_OBS_DIM,np.float32),
          'var':np.ones(SENSOR_OBS_DIM,np.float32),'count':123.}
    save_checkpoint(saved,FakeState(params,{'count':np.asarray(1,np.int32)}),
        {'step':20_054_016,'lr_scale':.25,'checkpoint_schema':8,
         'obs_dim':SENSOR_OBS_DIM,'observation_schema':OBS_SCHEMA_SENSOR,
         'robot_model_sha256':hashlib.sha256(ROBOT_XML.read_bytes()).hexdigest(),
         'sensor_contract':contract},norm,curriculum_state={'params':{}},
        health_state={},field_state=CapabilityField().state_dict())
    bundle=build_bundle(saved)
    assert bundle['schema']==2
    assert bundle['source']['checkpoint_schema']==8
    assert bundle['policy']['obs_dim']==232
    assert bundle['policy']['base_obs_dim']==58
    assert bundle['observation']['sensor_contract']==contract
    assert len(bundle['observation']['base_fields'])==58
    assert bundle['policy']['layers'][0]['w']['shape'][0]==232
    # The current physical examiner now supports schema 2, but must never
    # reinterpret its 232 observations as legacy 220-input channels.
    from emu.viewer_server import write_bundle
    artifact=tmp_path/'sensor.emupolicy'
    write_bundle(bundle,artifact)
    pi,normalizer,provenance=load_exported_actor_for_exam(artifact)
    assert provenance['observation_schema']==OBS_SCHEMA_SENSOR
    assert provenance['obs_dim']==SENSOR_OBS_DIM
    assert provenance['source']['checkpoint_schema']==8
    assert pi['pi'][0]['w'].shape[0]==SENSOR_OBS_DIM
    assert normalizer['mean'].shape==(SENSOR_OBS_DIM,)
    bad=json.loads(artifact.read_text())
    bad['observation']['sensor_contract']['settings']['tof_delay_frames']=2
    artifact.write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='sensor contract'):
        load_exported_actor_for_exam(artifact)
