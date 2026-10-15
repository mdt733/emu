from dataclasses import asdict
import os
from pathlib import Path
import tempfile
import time
from unittest.mock import patch

import pytest

from emu.checkpoints import find_evaluation, find_resume
from emu.training_config import TrainingConfig, foundation_config, mature_config


def test_training_config_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "config.json"
        original = mature_config(
            "graph", run_dir=Path(td) / "graph", steps=120_000_000,
            seed=811, device="cuda", envs=1024, resume=Path(td) / "ancestor.pkl",
        )
        original.save(path)
        assert TrainingConfig.load(path) == original


def test_shadow_and_graph_differ_only_by_field_mode():
    run = Path("/tmp/emu-contract")
    common = dict(run_dir=run, steps=120_000_000, seed=733, device="cuda", envs=2048, resume=Path("ancestor.pkl"))
    shadow = asdict(mature_config("shadow", **common))
    graph = asdict(mature_config("graph", **common))
    diffs = {k: (shadow[k], graph[k]) for k in shadow if shadow[k] != graph[k]}
    assert diffs == {"capability_field_mode": ("shadow", "graph")}


def test_foundation_cannot_network_rollback():
    cfg = foundation_config(run_dir=Path("/tmp/foundation20"), seed=733, device="cuda", envs=2048)
    assert cfg.steps == 20_000_000
    assert cfg.allow_rollback is False
    assert cfg.capability_field_mode == "shadow"
    assert cfg.cloud_probes_per_eval == 0
    assert cfg.cloud_final_probes == 0


def _touch(path: Path, age_s: float = 0.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"checkpoint")
    t = time.time() - age_s
    os.utime(path, (t, t))


def test_resume_checkpoint_order_is_canonical():
    with tempfile.TemporaryDirectory() as td:
        run = Path(td)
        snap = run / "checkpoints" / "snapshot_step_100.pkl"
        latest = run / "checkpoints" / "latest.pkl"
        final = run / "checkpoints" / "final.pkl"
        _touch(snap)
        assert find_resume(run) == snap
        _touch(latest, age_s=100)
        assert find_resume(run) == latest
        _touch(final, age_s=1000)
        # If latest was atomically committed after an older final, never
        # silently rewind to that earlier final checkpoint.
        assert find_resume(run) == latest


def test_evaluation_prefers_semantic_checkpoint_then_newest_arm():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _touch(root / "graph" / "checkpoints" / "best.pkl")
        _touch(root / "shadow" / "checkpoints" / "final.pkl", age_s=100)
        _touch(root / "foundation20" / "checkpoints" / "final.pkl", age_s=10)
        assert find_evaluation(root) == root / "foundation20" / "checkpoints" / "final.pkl"
        # A validated final takes priority over a newer in-progress latest.
        alternate = root / "other"
        _touch(alternate / "graph" / "checkpoints" / "final.pkl", age_s=3600)
        _touch(alternate / "shadow" / "checkpoints" / "latest.pkl")
        assert find_evaluation(alternate) == alternate / "graph" / "checkpoints" / "final.pkl"


def test_shared_packet_contract_is_fixed_and_self_consistent():
    from emu.contracts import (ACTION_DIM, BASE_OBS_DIM, HISTORY_LEN, OBS_DIM,
                           FIELD_NODE_ANCHORS, FIELD_EDGE_ANCHORS,
                           FIELD_NODE_STRIDE, FIELD_EDGE_STRIDE,
                           FIELD_NODE_FRAC, FIELD_NODE_START,
                           FIELD_EDGE_FRAC, FIELD_EDGE_START,
                           TASK_PARAM_DIM, TASK_PARAM_FIELDS, P)
    assert ACTION_DIM == 6
    assert BASE_OBS_DIM == 55 and HISTORY_LEN == 4 and OBS_DIM == 220
    assert len(TASK_PARAM_FIELDS) == 27
    assert P.steady_difficulty == 0 and P.probe_push_steps == 26
    assert FIELD_NODE_FRAC == 27 and FIELD_NODE_START == 28
    assert FIELD_EDGE_FRAC == FIELD_NODE_START + FIELD_NODE_ANCHORS * FIELD_NODE_STRIDE
    assert FIELD_EDGE_START == FIELD_EDGE_FRAC + 1
    assert TASK_PARAM_DIM == FIELD_EDGE_START + FIELD_EDGE_ANCHORS * FIELD_EDGE_STRIDE == 317


def test_open_frontier_expands_without_legacy_cadence_gate():
    from emu.curriculum import RegimeGovernor

    g = RegimeGovernor(open_frontier=True)
    row = {
        'ep_len_mean': 350.0, 'fall_rate': 0.0, 'episodes': 300.0,
        'precision_vx': 1.0, 'core_fall_rate': 0.0, 'core_ep_len_mean': 350.0,
        'ppo_kl_stopped_frac': 0.0,
    }
    for regime in g.REGIMES:
        row.update({
            f'fall_rate_{regime}': 0.0, f'episodes_{regime}': 100.0,
            f'falls_{regime}': 0.0, f'ep_len_mean_{regime}': 350.0,
            f'foot_mid_cone_{regime}': 1.0, f'vx_mean_{regime}': 0.2,
            f'cmd_vx_mean_{regime}': 0.2,
        })
    for key in ('frontier_precision_vx_forward', 'frontier_precision_vx_reverse',
                'frontier_precision_vy', 'frontier_precision_yaw'):
        row[key] = 1.0

    before = g.params.frontier_vx_hi
    report = g.update(row)
    assert g.params.frontier_vx_hi > before
    assert report['frontier_event_vx_hi'] == 'expand_edge_proven'



def test_actor_observation_schema_names_all_55_channels():
    from emu.contracts import BASE_OBS_FIELDS, BASE_OBS_DIM, HISTORY_LEN, OBS_DIM, OBS_NORMALIZED_FIELDS, PITCH_CMD_MAX_RAD
    from emu.ppo import build_obs_norm_mask
    expected = (
        'gravity_x','gravity_y','gravity_z','gyro_x','gyro_y','gyro_z',
        'q_r_hip_roll','q_r_hip_pitch','q_r_knee','q_l_hip_roll','q_l_hip_pitch','q_l_knee',
        'qv_r_hip_roll','qv_r_hip_pitch','qv_r_knee','qv_l_hip_roll','qv_l_hip_pitch','qv_l_knee',
        'prev_action_r_hip_roll','prev_action_r_hip_pitch','prev_action_r_knee','prev_action_l_hip_roll','prev_action_l_hip_pitch','prev_action_l_knee',
        'prev_ctrl_r_hip_roll','prev_ctrl_r_hip_pitch','prev_ctrl_r_knee','prev_ctrl_l_hip_roll','prev_ctrl_l_hip_pitch','prev_ctrl_l_knee',
        'foot_r_x','foot_r_y','foot_r_z','foot_l_x','foot_l_y','foot_l_z',
        'vel_x','vel_y','tof_height','vel_confidence',
        'cmd_vx','cmd_vy','cmd_yaw','cmd_height','cmd_pitch','cmd_roll','cmd_step_height',
        'phase_sin','phase_cos','shin_flex_r','shin_flex_l','shin_rate_r','shin_rate_l','shin_load_r','shin_load_l',
    )
    assert BASE_OBS_FIELDS == expected
    assert BASE_OBS_DIM == len(expected) == 55 and OBS_DIM == 55 * HISTORY_LEN == 220
    assert PITCH_CMD_MAX_RAD == .14
    mask = build_obs_norm_mask()[:BASE_OBS_DIM]
    masked_names = {name for name, enabled in zip(BASE_OBS_FIELDS, mask) if enabled > 0.5}
    assert masked_names == set(OBS_NORMALIZED_FIELDS)


def test_environment_task_schema_is_named_and_contiguous():
    from emu.environment import TASK_FIELDS, TASK_DIM, T
    expected = (
        'regime','vx','vy','yaw','height','pitch','roll','step_height',
        'resilience','phase_rate','birth_steady_d','birth_transition_d',
        'birth_resilience_d','birth_posture_d','athletic_rehearsal','sensor_scale','field_taught',
    )
    assert TASK_FIELDS == expected
    assert TASK_DIM == len(expected) == 17
    assert [getattr(T, name) for name in expected] == list(range(TASK_DIM))


def test_reward_telemetry_exports_only_consumed_components():
    """The five shaping diagnostics are defined by scoring, not by physics."""
    import ast
    import inspect
    from emu import scoring
    source = inspect.getsource(scoring.training_reward)
    tree = ast.parse(source)
    term_keys = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == 'terms' for t in node.targets):
            if isinstance(node.value, ast.Dict):
                term_keys.update(k.value for k in node.value.keys
                                 if isinstance(k, ast.Constant))
    assert term_keys == {'valid_alive', 'command_tracking', 'upright_height',
                         'feet_only_contact', 'swing_clearance'}
    # No legacy second reward owner hiding back in the simulation.
    import emu.environment as environment
    assert not hasattr(environment.EmuMJXEnv, '_reward')


def test_task_packet_contains_no_retired_assist_or_rehearsal_fields():
    from emu.contracts import TASK_PARAM_FIELDS
    names = {name for name, _ in TASK_PARAM_FIELDS}
    assert names.isdisjoint({'root_z_offset','assist_scale','assist_hip_amp','assist_knee_amp','assist_sign','core_rehearsal_frac'})


def test_environment_emits_regime_frontier_metrics_used_by_dynamic_reducer():
    import ast
    import emu.environment as environment
    source = Path(environment.__file__).read_text(encoding='utf-8')
    tree = ast.parse(source)
    keys=set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == '_metrics':
            for child in ast.walk(node):
                if isinstance(child, ast.Return) and isinstance(child.value, ast.Dict):
                    keys.update(k.value for k in child.value.keys
                                if isinstance(k, ast.Constant) and isinstance(k.value, str))
    assert {'regime_steady_frontier','regime_transition_frontier','regime_resilience_frontier'} <= keys

def test_modules_have_no_accidental_unresolved_globals():
    import builtins
    import symtable
    import emu

    package = Path(emu.__file__).resolve().parent
    allowed = {'__file__'}
    builtin_names = set(dir(builtins))
    problems = {}
    for path in package.glob('*.py'):
        table = symtable.symtable(path.read_text(encoding='utf-8'), str(path), 'exec')
        module_bound = {
            s.get_name() for s in table.get_symbols()
            if s.is_assigned() or s.is_imported() or s.is_namespace()
        }
        unresolved = set()

        def walk(scope):
            for symbol in scope.get_symbols():
                name = symbol.get_name()
                if (symbol.is_global() and symbol.is_referenced()
                        and name not in module_bound and name not in builtin_names and name not in allowed):
                    unresolved.add(name)
            for child in scope.get_children():
                walk(child)

        walk(table)
        if unresolved:
            problems[path.name] = sorted(unresolved)
    assert problems == {}


def test_capability_field_state_is_exact_not_best_effort():
    from emu.capability_field import CapabilityField
    state = CapabilityField().state_dict()
    state.pop('half_life_steps')
    try:
        CapabilityField.from_state_dict(state)
    except ValueError:
        return
    raise AssertionError('incomplete capability-field state was silently accepted')


def test_curriculum_state_is_exact_not_best_effort():
    from emu.curriculum import RegimeGovernor
    state = RegimeGovernor().state_dict()
    del state['core_bad_streak']
    try:
        RegimeGovernor().load_state_dict(state)
    except ValueError:
        return
    raise AssertionError('partial curriculum state was accepted')




def test_default_mature_teacher_matches_v5012_strength_until_tournament_overrides_it():
    from emu.training_config import doctrine_dict
    cfg=mature_config('graph',run_dir=Path('/tmp/g'),steps=120_000_000,seed=733,device='cuda',envs=2048,resume=Path('/tmp/a.pkl'))
    assert cfg.field_max_influence == .65
    assert cfg.field_teach_nodes and cfg.field_teach_edges and cfg.field_teach_recovery
    assert cfg.field_evidence_from_taught is True
    assert (cfg.field_influence_start_steps,cfg.field_influence_full_steps) == (20_000_000,30_000_000)
    assert cfg.field_influence_fade_start_steps == 0 and cfg.field_influence_zero_steps == 0
    assert 'FIELD_MAX_INFLUENCE' not in doctrine_dict()


def test_field_teacher_provenance_is_not_an_actor_channel():
    from emu.contracts import BASE_OBS_FIELDS
    from emu.environment import TASK_FIELDS
    assert 'field_taught' in TASK_FIELDS
    assert 'field_taught' not in BASE_OBS_FIELDS

def test_foundation_handoff_uses_complete_update_boundary():
    from emu.run_support import foundation_handoff_step, update_boundary_at_or_after
    batch = 2048 * 32
    assert foundation_handoff_step(2048) == 20_054_016
    assert foundation_handoff_step(2048) % batch == 0
    assert foundation_handoff_step(2048) >= 20_000_000
    assert update_boundary_at_or_after(120_000_000, 2048) == 120_061_952


def test_partial_foundation_checkpoint_is_resume_only_not_fork_ancestor():
    from emu.run_support import resolve_foundation_handoff
    foundation = Path('/tmp/foundation-contract')
    partial = foundation / 'checkpoints' / 'latest.pkl'
    with patch('emu.run_support.find_resume', return_value=partial), \
         patch('emu.run_support.find_valid_resume', return_value=partial), \
         patch('emu.run_support.checkpoint_step', return_value=12_000_000):
        complete, resume, expected = resolve_foundation_handoff(foundation, envs=2048)
    assert complete is None
    assert resume == partial
    assert expected == 20_054_016


def test_exact_handoff_checkpoint_is_promoted_to_final():
    from emu.run_support import resolve_foundation_handoff
    foundation = Path('/tmp/foundation-contract')
    latest = foundation / 'checkpoints' / 'latest.pkl'
    final = foundation / 'checkpoints' / 'final.pkl'
    with patch('emu.run_support.find_resume', return_value=latest), \
         patch('emu.run_support.find_valid_resume', return_value=latest), \
         patch('emu.run_support.checkpoint_step', return_value=20_054_016), \
         patch('emu.run_support.promote_checkpoint', return_value=final) as promote:
        complete, resume, expected = resolve_foundation_handoff(foundation, envs=2048)
    assert complete == final and resume is None and expected == 20_054_016
    promote.assert_called_once_with(latest, final)


def test_robot_xml_owns_effective_actuated_joint_physics():
    import xml.etree.ElementTree as ET
    from emu.paths import ROBOT_XML
    import emu.environment as environment

    root = ET.parse(ROBOT_XML).getroot()
    names = {'r_hip_roll','r_hip_pitch','r_knee','l_hip_roll','l_hip_pitch','l_knee'}
    joints = {j.attrib.get('name'): j for j in root.findall('.//joint') if j.attrib.get('name') in names}
    assert set(joints) == names
    for joint in joints.values():
        assert float(joint.attrib['armature']) == 0.012
        assert float(joint.attrib['frictionloss']) == 0.08
    assert root.findall('./contact/exclude') == []
    source = Path(environment.__file__).read_text(encoding='utf-8')
    assert 'dof_armature' not in source and 'dof_frictionloss' not in source


def test_governor_profile_values_are_instance_state_not_shadowed_class_constants():
    from emu.curriculum import RegimeGovernor
    g = RegimeGovernor(fall_target=0.34, difficulty_base_step=0.0024)
    assert g.fall_target == 0.34
    assert g.difficulty_base_step == 0.0024
    assert not hasattr(RegimeGovernor, 'FALL_TARGET')
    assert not hasattr(RegimeGovernor, 'DIFFICULTY_BASE_STEP')
    report = g.update({
        'ep_len_mean': 0.0, 'fall_rate': 0.0, 'episodes': 0.0,
        'core_fall_rate': 0.0, 'core_ep_len_mean': 0.0,
        'ppo_kl_stopped_frac': 0.0,
        **{f'fall_rate_{r}': 0.0 for r in g.REGIMES},
        **{f'episodes_{r}': 0.0 for r in g.REGIMES},
    })
    assert report['fall_target'] == 0.34


def test_optax_partition_routes_prefix_labels_when_dependency_is_available():
    optax = pytest.importorskip('optax')
    import jax.numpy as jnp
    import numpy as np
    from emu.ppo import parameter_group_labels

    params = {'pi': {'w': jnp.array([0.0])}, 'vf': {'w': jnp.array([0.0])}, 'log_std': jnp.array([0.0])}
    grads = {'pi': {'w': jnp.array([1.0])}, 'vf': {'w': jnp.array([1.0])}, 'log_std': jnp.array([1.0])}
    tx = optax.partition({'policy': optax.sgd(0.1), 'value': optax.sgd(0.2)}, parameter_group_labels)
    state = tx.init(params)
    updates, _ = tx.update(grads, state, params)
    np.testing.assert_allclose(np.asarray(updates['pi']['w']), [-0.1])
    np.testing.assert_allclose(np.asarray(updates['log_std']), [-0.1])
    np.testing.assert_allclose(np.asarray(updates['vf']['w']), [-0.2])


def test_perfbook_registered_timers_keep_metrics_schema_stable():
    from emu.runtime import PerfBook

    perf = PerfBook(enabled=True, known_names=('train_update', 'log_io', 'checkpoint_latest_total'))
    first = perf.row()
    assert first['perf_log_io_total_s'] == 0.0
    assert first['perf_log_io_count'] == 0.0

    perf.add('train_update', 1.25)
    second = perf.row()
    assert list(second) == list(first)

    perf.add('log_io', 0.05)
    third = perf.row()
    assert list(third) == list(first)
    assert third['perf_log_io_total_s'] == 0.05
    assert third['perf_log_io_count'] == 1.0


def test_training_registers_every_perf_timer_used_by_training_source():
    import ast
    import inspect
    import emu.training as training

    tree = ast.parse(inspect.getsource(training))
    used = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != 'add' or not isinstance(node.func.value, ast.Name) or node.func.value.id != 'perf':
            continue
        if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
            used.add(node.args[0].value)
    assert used <= set(training.TRAIN_PERF_TIMER_NAMES)


def test_initial_eval_metrics_declares_late_exam_columns_from_row_one():
    from emu.evaluation import PANELS, initial_metrics

    row = initial_metrics()
    expected = {
        'perf_fixed_eval_total_s',
        'perf_eval_cohorts_used',
        'perf_eval_env_steps',
        'perf_eval_env_steps_per_s',
        'perf_best_checkpoint_total_s',
        'perf_best_checkpoint_device_get_s',
        'perf_best_checkpoint_pickle_s',
        'field_calibration_mae',
        'field_calibration_brier',
        'field_calibration_points',
    }
    expected.update(f'perf_eval_{name}_s' for name in PANELS)
    assert expected <= set(row)
    assert row['field_calibration_mae'] == -1.0
    assert row['perf_fixed_eval_total_s'] == 0.0


def test_field_influence_schema_helper_is_shared_by_periodic_and_final_rows():
    from emu.training import attach_field_influence

    report = {}
    attach_field_influence(report, {'field_node_frac': 0.25, 'field_edge_frac': 0.10})
    assert report == {'field_node_influence': 0.25, 'field_edge_influence': 0.10}
