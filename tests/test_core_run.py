"""Four maintained run contracts: held-out physics, source-locked forks and checkpoint safety.

The 140M / 20M / eight-arm tournament launch scripts were disposable.
Test the *one training CLI and checkpoint contract* they exercised rather
than retaining obsolete campaign code just to keep its tests green.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, NamedTuple
from unittest.mock import patch

import numpy as np
import pytest

from scripts.smoke_training_lifecycle import configs
from emu.training_config import training_sensor_contract
from emu.contracts import OBS_SCHEMA_TERRAIN
from emu.run import one_run_config

EXAMPLE_STEPS=50_000_000
ENVS=2048

def training_flags(run_dir: Path, seed: int = 735) -> list[str]:
    """Representative modern training CLI, NOT a second experiment launcher."""
    return [
        "train", "--run-dir", str(run_dir),
        "--steps", str(EXAMPLE_STEPS), "--envs", str(ENVS),
        "--seed", str(seed), "--device", "cuda", "--profile", "sim2real",
        "--pitch-deg", "8", "--terrain-actor",
        "--terrain-step-max-m", ".08", "--unified-capabilities",
        "--terrain-carpet", "--carpet-prisms",
        "--carpet-transition-cap-deg", "3",
        "--foundation-steps", "10000000", "--foundation-mix", "main",
        "--growth-recipe", "main-mature", "--teaching-schedule", "terrain-early",
        "--terrain-lesson", "coherent-careful", "--force-program", "gentle",
        "--athletic-rehearsal-share", ".10", "--retention-share", ".15",
        "--academy-stream", "--academy-intro-steps", "600000000",
        "--academy-start-steps", "800000000",
        "--academy-intro-share", ".10", "--academy-ramp-steps", "10000000",
        "--academy-max-share", ".40", "--academy-height-cap-m", ".15",
        "--academy-course-episode-frames", "1200",
        "--fixed-eval-every-steps", "50000000",
        "--solver-iterations", "2", "--solver-ls-iterations", "4",
    ]


def test_current_cli_forks_preserve_sensor_and_reward_economics(tmp_path):
    from emu.cli import main as emu_cli_main
    cli_args=training_flags(tmp_path/"run")
    seen=[]
    def capture(ns):
        seen.append(one_run_config(ns))
        return 0
    with patch("emu.run.train_one",side_effect=capture):
        assert emu_cli_main(cli_args)==0
    cfg=seen[0]
    assert cfg.terrain_lesson=="coherent-careful"
    assert cfg.teaching_schedule=="terrain-early"
    assert cfg.fixed_eval_every_steps==50_000_000
    assert cfg.difficulty_base_step==pytest.approx(.0024)
    assert cfg.fall_target==pytest.approx(.34)
    assert (cfg.vx_hi_max,cfg.vx_lo_min,cfg.vy_max_mps,cfg.yaw_max_rad_s)==pytest.approx(
        (.60,-.20,.22,1.50))
    assert (cfg.combo_probe_frac,cfg.phase_speed_gain,cfg.regime_command_frac,
            cfg.combo_mix_frac)==pytest.approx((.12,1.25,.85,.88))
    assert cfg.frontier_push_safety_n==pytest.approx(.70)
    assert cfg.push_max_force_n==pytest.approx(4.)
    assert cfg.capability_field_mode=="shadow"
    assert cfg.freeze_obs_norm_after_steps==0
    assert cfg.stillness_birth_frac==0.
    assert cfg.science_step_finish_on_transfer is False
    with patch("emu.run.train_one",side_effect=capture):
        assert emu_cli_main(cli_args+["--science-step-finish-on-transfer"])==0
    assert seen[-1].science_step_finish_on_transfer is True
    with patch("emu.run.train_one",side_effect=capture):
        assert emu_cli_main(cli_args+[
            "--stair-survival-curriculum","--science-step-goal-weight",".12"])==0
    assert seen[-1].stair_survival_curriculum is True
    with pytest.raises(ValueError,match="external warm-start"):
        emu_cli_main(cli_args+[
            "--stair-survival-curriculum","--resume","/tmp/old.pkl"])
    with pytest.raises(ValueError,match="positive .*science-step-goal-weight"):
        emu_cli_main(cli_args+["--stair-survival-curriculum"])
    with pytest.raises(ValueError,match="full first-life"):
        emu_cli_main(cli_args+[
            "--stair-survival-curriculum","--science-step-goal-weight",".12",
            "--science-step-finish-on-transfer"])
    assert cfg.stair_survival_curriculum is False
    # Real research forks may change teacher and solver, but not the
    # 360-input actor geometry, physical robot or sensor history.
    from emu.run import verify_experiment_fork
    fork=replace(cfg,steps=100_000_000,academy_intro_steps=50_000_000,
                 academy_start_steps=50_000_000,stillness_birth_frac=.10)
    verify_experiment_fork(cfg,fork)
    with pytest.raises(RuntimeError,match="observation_schema"):
        verify_experiment_fork(cfg,replace(fork,observation_schema="sim2real_232"))
    with patch("emu.run.train_one",side_effect=capture):
        assert emu_cli_main(cli_args+["--stillness-birth-frac",".12"])==0
    assert seen[-1].stillness_birth_frac==pytest.approx(.12)
    with patch("emu.run.train_one",side_effect=lambda ns:one_run_config(ns)):
        with pytest.raises(ValueError,match="stillness-birth-frac"):
            emu_cli_main(cli_args+["--stillness-birth-frac",".4"])
    from emu.rewards import _gait_maturity
    m=dict(age_s=np.asarray([2.],np.float32),
           stepping_activity_so_far=np.asarray([0.],np.float32),
           cmd_vx=np.asarray([0.],np.float32),
           cmd_vy=np.asarray([0.],np.float32),
           cmd_yaw=np.asarray([0.],np.float32))
    assert float(_gait_maturity(m,np)[1][0])==pytest.approx(.30)
    assert float(_gait_maturity(m,np,allow_zero_command_balance=True)[1][0])==pytest.approx(1.)
    m["cmd_vx"]=np.asarray([.12],np.float32)
    assert float(_gait_maturity(m,np,allow_zero_command_balance=True)[1][0])==pytest.approx(.30)
    assert cfg.physical_profile=="h40"
    _check_spring_only_changes_nothing_except_knee_tendons()
    _check_school_teacher_gates()
    _check_signed_prism_exam_height_grid()
    assert not {"cloud_probes_per_eval","cloud_final_probes",
                "science_reset_step_evidence_on_fork"} & asdict(cfg).keys()
    assert training_sensor_contract(cfg) is not None
    # Keep 24 core tests: this UI is another assertion of the *existing*
    # public CLI contract, not a new training experiment/test family.
    _check_readonly_dashboard(tmp_path)


def _check_spring_only_changes_nothing_except_knee_tendons():
    """Research variant must not silently carry Spring V1 ROM/actuation changes."""
    import hashlib
    import math
    from xml.etree import ElementTree as ET

    import mujoco
    from emu.contracts import PACKAGE_ROOT
    from emu.physical_variants import (
        SPRING_ONLY_PROFILE, generate_xml, profile_numbers)

    source = (PACKAGE_ROOT / "robot.xml").read_text(encoding="utf-8")
    baseline = generate_xml(source, "h40")
    candidate = generate_xml(source, SPRING_ONLY_PROFILE)
    # Frozen original champion model bytes remain intact.
    assert hashlib.sha256(baseline.encode()).hexdigest() == (
        "b513bb3326a73d9ca661f761f1352b22c36d0f1b6d51be72c2eca15d8712ef5b")
    assert profile_numbers(SPRING_ONLY_PROFILE)["knee_strut_k_n_m"] == 150.
    assert baseline != candidate

    original = ET.fromstring(baseline)
    modified = ET.fromstring(candidate)
    original.set("model", modified.get("model"))
    for side in "rl":
        for tag, name, attr in (
            ("site", f"{side}_knee_strut_thigh", "pos"),
            ("site", f"{side}_knee_strut_shin", "pos"),
            ("spatial", f"{side}_knee_support", "stiffness"),
            ("spatial", f"{side}_knee_support", "springlength"),
        ):
            left = original.find(f".//{tag}[@name='{name}']")
            right = modified.find(f".//{tag}[@name='{name}']")
            assert left is not None and right is not None
            left.set(attr, right.get(attr))
    assert ET.tostring(original) == ET.tostring(modified), (
        "Spring-only candidate changed non-spring H40 XML or actuator limits")

    a = mujoco.MjModel.from_xml_string(baseline)
    c = mujoco.MjModel.from_xml_string(candidate)
    np.testing.assert_array_equal(a.jnt_range, c.jnt_range)
    np.testing.assert_array_equal(a.actuator_ctrlrange, c.actuator_ctrlrange)
    np.testing.assert_array_equal(a.actuator_forcerange, c.actuator_forcerange)
    np.testing.assert_allclose(c.tendon_stiffness, [150., 150.])
    for side in "rl":
        torques = []
        for model in (a, c):
            jid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_knee")
            assert jid >= 0
            data = mujoco.MjData(model)
            data.qpos[model.jnt_qposadr[jid]] = math.radians(-55.)
            mujoco.mj_forward(model, data)
            torques.append(float(data.qfrc_passive[model.jnt_dofadr[jid]]))
        assert torques[0] > torques[1] > 0.0, torques
        assert torques[1] < .980665, torques



def _check_signed_prism_exam_height_grid():
    """No old 5cm fixture masquerading as a tested 8cm target."""
    from scripts.v5053_matrix_exam import assignments
    trials=[t for t in assignments(2048,mode="stairs",
                                  levels=20,trials_per_level=50)
            if t.kind != "padding"]
    assert len(trials)==2000
    for direction,sign in (("step_up",1),("step_down",-1)):
        items=[t for t in trials if t.kind==direction]
        assert len(items)==1000
        assert sorted({round(t.exact_step_m*sign*1000,4) for t in items})==[
            float(4*i) for i in range(1,21)]


def _check_school_teacher_gates():
    """The same teacher grows difficulty from first-life physical proof."""
    from emu.curriculum import RegimeGovernor
    from emu.training_config import SCHOOL_ORDINARY_STEP_CAP_M
    assert SCHOOL_ORDINARY_STEP_CAP_M==.04
    def school():
        return RegimeGovernor(
            unified_capability_focus=True,safe_ascent_enabled=True)
    def evidence(attempts=24,survived=24,crossed=18):
        return dict(academy_active_fraction=.2,
                    academy_near_attempts_up=12,
                    academy_near_successes_up=12,
                    safe_ascent_near_attempts_up=attempts,
                    safe_ascent_near_survived_up=survived,
                    safe_ascent_near_crossed_up=crossed,
                    safe_ascent_near_safe_crossed_up=min(
                        attempts,survived,crossed))
    tutor=school()
    tutor.params.academy_height_up_m=.04
    tutor.params.academy_height_down_m=.04
    assert "up:expand_survival_and_cross" in tutor.advance_academy(
        evidence(),max_height_m=.055,enabled=True)
    assert tutor.params.academy_height_up_m==pytest.approx(.0405)
    assert tutor.params.academy_height_down_m==.04
    assert tutor.safe_ascent_evidence==tutor._empty_safe_ascent_evidence()
    # A success-only lesson is NOT sufficient for promotion.
    no_survival=school()
    no_survival.params.academy_height_up_m=.04
    events=no_survival.advance_academy(
        evidence(survived=21),max_height_m=.055,enabled=True)
    assert "up:hold_survival_or_cross" in events
    assert no_survival.params.academy_height_up_m==.04
    no_crossing=school()
    no_crossing.params.academy_height_up_m=.04
    events=no_crossing.advance_academy(
        evidence(crossed=0),max_height_m=.055,enabled=True)
    assert "up:hold_survival_or_cross" in events
    assert no_crossing.params.academy_height_up_m==.04
    # A fall-prone cohort actively retreats, however impressive its
    # supported-crossing or foothold-only payment may look.
    danger=school()
    danger.params.academy_height_up_m=.045
    assert "up:survival_retreat_near" in danger.advance_academy(
        evidence(survived=13,crossed=24),max_height_m=.055,enabled=True)
    assert danger.params.academy_height_up_m==pytest.approx(.0445)
    assert danger.params.academy_height_down_m==.003
    # First-life evidence is stored in the same checkpoint as PPO and
    # restored without losing a difficult, partially sampled cohort.
    partially=school()
    partial=evidence(attempts=12,survived=11,crossed=8)
    events=partially.advance_academy(
        partial,max_height_m=.055,enabled=True)
    assert "up:collect_first_life" in events
    full=partially.state_dict()
    restart=school()
    restart.load_state_dict(full)
    assert restart.state_dict()==full
    with pytest.raises(ValueError,match="curriculum state keys mismatch"):
        restart.load_state_dict(
            RegimeGovernor(unified_capability_focus=True).state_dict())
    with pytest.raises(ValueError,match="school episode outcomes"):
        broken=restart.state_dict()
        broken["safe_ascent_evidence"]["near_survived"]=40.
        restart.load_state_dict(broken)
    with pytest.raises(ValueError,match="school episode outcomes"):
        broken=restart.state_dict()
        broken["safe_ascent_evidence"]["near_safe_crossed"]=12.
        broken["safe_ascent_evidence"]["near_crossed"]=8.
        restart.load_state_dict(broken)


def _check_readonly_dashboard(tmp_path):
    import csv
    import json
    from emu.console_dashboard import (
        read_history, render, sampled_window, delta, target_from_config)
    from emu.cli import main as cli_main
    directory = tmp_path / "dashboard-run"
    directory.mkdir()
    (directory/"training_config.json").write_text(
        json.dumps({"steps":120000000}),encoding="utf-8")
    assert target_from_config(directory)==120000000
    rows=[
        dict(step=95000000,reward_mean=2.0,fall_rate=.2,
             academy_near_attempts_up=10,academy_near_successes_up=1),
        dict(step=100000000,reward_mean=2.5,fall_rate=.1,
             academy_near_attempts_up=8,academy_near_successes_up=1),
    ]
    source=directory/"metrics.csv"
    with source.open("w",newline="",encoding="utf-8") as fp:
        writer=csv.DictWriter(fp,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    before=source.read_bytes()
    history=read_history(source)
    assert delta(history,"reward_mean",precision=1)=="+0.5"
    assert sampled_window(history,"academy_near_attempts_up",
                          "academy_near_successes_up")=="1/8 (12.5%)"
    screen=render(history,run_name=directory.name,
                  target=120000000,width=100,height=27)
    assert "100.0M / 120.0M" in screen
    assert "STAIRS" in screen and "NOT all-episode counts" in screen
    assert "not yet evaluated" in screen
    assert cli_main(["dashboard","--run-dir",str(directory),
                     "--once","--plain"]) == 0
    assert source.read_bytes()==before

    # Switching Git branches must never leave a stale compiled Meadow visible.
    # This is a pure stdlib fingerprint check, not a Node build or PPO update.
    from emu.viewer_server import (
        VIEWER_SOURCE_STAMP, viewer_source_fingerprint,
        viewer_build_is_current)
    viewer_src=tmp_path/"viewer-source"
    viewer_src.mkdir()
    (viewer_src/"main.js").write_text("old Meadow", encoding="utf-8")
    viewer_dist=tmp_path/"viewer-dist"
    viewer_dist.mkdir()
    (viewer_dist/"index.html").write_text("<html/>", encoding="utf-8")
    assert not viewer_build_is_current(viewer_dist, viewer_src)
    (viewer_dist/VIEWER_SOURCE_STAMP).write_text(
        viewer_source_fingerprint(viewer_src)+"\n", encoding="ascii")
    assert viewer_build_is_current(viewer_dist, viewer_src)
    # Generated build content and installed npm dependencies are excluded.
    (viewer_src/"dist").mkdir()
    (viewer_src/"dist"/"app.js").write_text("built JS")
    (viewer_src/"node_modules").mkdir()
    (viewer_src/"node_modules"/"test.js").write_text("dependency")
    assert viewer_build_is_current(viewer_dist, viewer_src)
    # The actual source changes across branch checkouts, so the stamp expires.
    (viewer_src/"main.js").write_text("new Meadow", encoding="utf-8")
    assert not viewer_build_is_current(viewer_dist, viewer_src)



def test_actual_evaluator_uses_prisms_not_boxes(tmp_path, monkeypatch):
    import emu.evaluation as evaluate
    cli_args = training_flags(tmp_path / "run")
    config = []

    with patch("emu.run.train_one", side_effect=lambda ns: (config.append(one_run_config(ns)), 0)[1]):
        from emu.cli import main as emu_cli_main
        emu_cli_main(cli_args)
    cfg = config[0]
    call = {}

    class FakeEnv:
        def __init__(self, *args, **kwargs):
            call.update(kwargs)
            self.sensor_contract = training_sensor_contract(cfg)

    monkeypatch.setattr(evaluate, "EmuMJXEnv", FakeEnv)
    monkeypatch.setattr(evaluate, "make_trainer", lambda env, **kwargs: None)
    env, trainer = evaluate._world(cfg, seed_offset=0,
                                   limits=(.60, -.20, .22, 1.50))
    assert env is not None and trainer is None
    assert call["terrain_carpet"] is True
    assert call["terrain_carpet_prisms"] is True
    assert call["observation_schema"] == OBS_SCHEMA_TERRAIN
    assert call["solver_iterations"] == 2
    assert call["solver_ls_iterations"] == 4
    # Frozen panels must use the actor's actual trained physics,
    # including a changed 4/4 treatment rather than XML defaults.
    from dataclasses import replace
    evaluate._world(replace(cfg, solver_iterations=4, solver_ls_iterations=4),
                    seed_offset=0, limits=(.60,-.20,.22,1.50))
    assert call["solver_iterations"] == 4
    assert call["solver_ls_iterations"] == 4
    # A frozen exam may never fall back to old nine-box or slab contact
    # because it happens to request a flat lesson.
    from dataclasses import replace
    with pytest.raises(ValueError, match="real unified prism world"):
        evaluate._world(replace(cfg, terrain_carpet_prisms=False),
                        seed_offset=0, limits=(.60, -.20, .22, 1.50))


def test_prism_two_update_checkpoint_lifecycle_recipe(tmp_path):
    first, second = configs(tmp_path, unified=True, carpet=True,
                            prisms=True, envs=8, seed=735)
    assert first.terrain_carpet_prisms and second.terrain_carpet_prisms
    assert first.terrain_carpet and first.academy_stream
    assert first.observation_schema == OBS_SCHEMA_TERRAIN
    assert first.steps == 256 and second.steps == 512
    # A source-locked training config cannot silently rebuild missing 5045
    # physical fields using a pre-prism checkpoint's defaults.
    import json
    from emu.training_config import TrainingConfig
    cfg_path = tmp_path / "training_config.json"
    first.save(cfg_path)
    assert TrainingConfig.load(cfg_path) == first
    partial = json.loads(cfg_path.read_text(encoding="utf-8"))
    # The new physical lineage has no pre-v5054 migration exceptions.
    # A missing teaching field is a different recipe, not a compatible run.
    partial.pop("science_step_finish_on_transfer")
    cfg_path.write_text(json.dumps(partial), encoding="utf-8")
    with pytest.raises(ValueError,match="training config keys mismatch"):
        TrainingConfig.load(cfg_path)
    partial = asdict(first)
    partial.pop("terrain_carpet_prisms")
    cfg_path.write_text(json.dumps(partial), encoding="utf-8")
    with pytest.raises(ValueError,match="training config keys mismatch"):
        TrainingConfig.load(cfg_path)
    with pytest.raises(ValueError,match="prisms"):
        configs(tmp_path, unified=True, carpet=True, envs=8, seed=735)
    # Old synthetic terrain randomization is not an alternative collider.
    from emu.domain_randomization import DomainRandomizationConfig
    manufacturing = DomainRandomizationConfig(
        enabled=True, robot_radius=.6, hardware_radius=.6).normalized()
    assert manufacturing.robot_radius == .6 and manufacturing.hardware_radius == .6
    assert not hasattr(manufacturing, 'terrain_radius')
    assert not hasattr(first, 'domain_terrain_radius')
    with pytest.raises(TypeError):
        DomainRandomizationConfig(terrain_radius=.3)
    with pytest.raises(ValueError):
        configs(tmp_path, prisms=True, envs=8, seed=735)

    # Exact-current-source curriculum state is part of the same checkpoint
    # contract. Older pooled evidence cannot quietly become v5045 progress.
    from copy import deepcopy
    from emu.curriculum import RegimeGovernor
    gov = RegimeGovernor(unified_capability_focus=True)
    gov.terrain_evidence['up_near_attempts'] = 5.
    gov.terrain_evidence['up_near_successes'] = 3.
    gov.academy_evidence['up_near_attempts'] = 2.
    original = gov.state_dict()
    restored = RegimeGovernor(unified_capability_focus=True)
    restored.load_state_dict(original)
    assert restored.state_dict() == original
    # A successful instant stair finish is NOT a 40-frame fall. The
    # 350-frame equivalent is used solely for governor competence, while
    # truthful episode lengths remain available in the report.
    succeeded = RegimeGovernor(unified_capability_focus=True)
    corrected=succeeded.update(dict(
        fall_rate=0.,episodes=1.,ep_len_mean=40.,
        core_fall_rate=0.,core_ep_len_mean=40.,
        core_ep_len_competence_mean=350.))
    assert corrected["core_competence"]==pytest.approx(1.)
    uncorrected=RegimeGovernor(unified_capability_focus=True)
    old=uncorrected.update(dict(
        fall_rate=0.,episodes=1.,ep_len_mean=40.,
        core_fall_rate=0.,core_ep_len_mean=40.))
    assert old["core_competence"]<.12
    for remove in ('academy_evidence', 'terrain_evidence', 'command_evidence'):
        corrupt = deepcopy(original)
        del corrupt[remove]
        with pytest.raises(ValueError, match='curriculum state keys mismatch'):
            restored.load_state_dict(corrupt)
    corrupt = deepcopy(original)
    del corrupt['params']['academy_height_up_m']
    with pytest.raises(ValueError, match='curriculum parameter keys mismatch'):
        restored.load_state_dict(corrupt)
    corrupt = deepcopy(original)
    del corrupt['command_evidence']['vx_hi']['easy_attempts']
    with pytest.raises(ValueError, match='bad command frontier evidence'):
        restored.load_state_dict(corrupt)
    corrupt = deepcopy(original)
    corrupt['academy_evidence']['updates'] = 2.
    with pytest.raises(ValueError, match='Academy evidence checkpoint mismatch'):
        restored.load_state_dict(corrupt)
    # A STEP teacher advances only when both signs have independently
    # demonstrated supported near-frontier crossing.
    step_gov = RegimeGovernor(unified_capability_focus=True)
    assert step_gov.advance_terrain({}, max_height_m=.08) == 'terrain_unified_seed_4mm'
    before = step_gov.params.terrain_height_m
    near = {f'terrain_near_{kind}_{direction}': val
            for direction in ('up','down')
            for kind,val in (('attempts',4.),('successes',3.))}
    assert step_gov.advance_terrain(near,max_height_m=.08) == 'terrain_height_expand_real_cross'
    assert step_gov.params.terrain_height_m > before
    # An easy failed encounter has priority over near successes; this is
    # an actual retreat, not an academic metric-only diagnostic.
    easy = {f'terrain_easy_{kind}_up': val
            for kind,val in (('attempts',4.),('successes',0.))}
    assert step_gov.advance_terrain(easy,max_height_m=.08) == 'terrain_retreat_easy_failure'
    assert step_gov.params.terrain_height_m < before + .0011
    # The unified controller's generic frontier helper must NOT independently
    # promote signed command axes: six specialist proof cohorts own those.
    g = RegimeGovernor(unified_capability_focus=True, open_frontier=True)
    g.fall_ema['resilience'] = .20
    g.competence['resilience'] = .60
    old_vx = g.params.frontier_vx_hi
    old_push = g.params.frontier_push_n
    ev = g._advance_physical_frontiers(
        {},global_fall=.20,global_retreat=False,specialist_growth_enabled=True)
    assert ev['push_n'] == 'expand' and g.params.frontier_push_n > old_push
    assert ev['combo'] == 'no_edge_data'
    assert g.params.frontier_vx_hi == old_vx
    assert g._advance_physical_frontiers(
        {},global_fall=.20,global_retreat=False,specialist_growth_enabled=False) == {}


class _SavedLearner(NamedTuple):
    params: Any
    opt_state: Any


def test_current_360_checkpoint_roundtrip_rejects_sensor_drift(tmp_path):
    """A real 360-input contract must survive a saved optimizer+normalizer."""
    from emu.contracts import TERRAIN_OBS_DIM
    from emu.training_config import TrainingConfig
    from emu.checkpoint_io import (
        save_checkpoint, load_checkpoint, TERRAIN_CHECKPOINT_SCHEMA)
    from emu.ppo import expected_param_shapes

    cfg=TrainingConfig(observation_schema=OBS_SCHEMA_TERRAIN,
                       unified_capability_focus=True)
    contract=training_sensor_contract(cfg)
    def build(spec):
        if isinstance(spec, dict):
            return {k:build(v) for k,v in spec.items()}
        if isinstance(spec, list):
            return [build(v) for v in spec]
        return np.zeros(spec,dtype=np.float32)
    params=build(expected_param_shapes())
    for family in ('pi','vf'):
        old=params[family][0]['w']
        params[family][0]['w']=np.zeros((TERRAIN_OBS_DIM,old.shape[1]),np.float32)
    state=_SavedLearner(params,{'count':np.asarray(1,np.int32)})
    norm={'mean':np.zeros(TERRAIN_OBS_DIM,np.float32),
          'var':np.ones(TERRAIN_OBS_DIM,np.float32),'count':8.0}
    meta=dict(step=256,lr_scale=.25,
              checkpoint_schema=TERRAIN_CHECKPOINT_SCHEMA,
              observation_schema=OBS_SCHEMA_TERRAIN,
              obs_dim=TERRAIN_OBS_DIM,sensor_contract=contract)
    file=tmp_path/'current_360.pkl'
    save_checkpoint(file,state,meta,norm,
                    curriculum_state={'params':{}},health_state={},field_state={})
    restored=load_checkpoint(file,validate=True,
                expected_observation_schema=OBS_SCHEMA_TERRAIN,
                expected_sensor_contract=contract)
    assert restored['meta']['step']==256
    np.testing.assert_array_equal(restored['state']['params']['pi'][0]['w'],
                                  params['pi'][0]['w'])
    corrupted={**contract,'sensor_model_revision':'tripod_tof_flow_async_v1'}
    with pytest.raises(ValueError):
        load_checkpoint(file,expected_observation_schema=OBS_SCHEMA_TERRAIN,
                        expected_sensor_contract=corrupted)

