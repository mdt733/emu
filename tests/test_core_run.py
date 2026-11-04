"""Maintained EMU run contracts: real CLI, source-locked continuation, physics.

The 140M / 20M / eight-arm tournament launch scripts were disposable.
Test the *one training CLI and checkpoint contract* they exercised rather
than retaining obsolete campaign code just to keep its tests green.
"""
from __future__ import annotations

from dataclasses import replace
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


def test_training_cli_is_one_360_input_eight_prism_specialist_policy(tmp_path):
    from emu.cli import main
    calls=[]
    with patch("emu.run.train_one",side_effect=lambda ns:(calls.append(one_run_config(ns)),0)[1]):
        assert main(training_flags(tmp_path/"study")) == 0
    assert len(calls)==1
    cfg=calls[0]
    assert (cfg.steps,cfg.envs,cfg.seed)==(EXAMPLE_STEPS,ENVS,735)
    assert cfg.terrain_carpet and cfg.terrain_carpet_prisms
    assert cfg.unified_capability_focus and cfg.academy_stream
    assert cfg.observation_schema==OBS_SCHEMA_TERRAIN
    assert cfg.foundation_steps==10_000_000
    assert cfg.academy_intro_steps==600_000_000
    assert cfg.athletic_rehearsal_frac==pytest.approx(.10)
    assert cfg.retention_share==pytest.approx(.15)
    assert (cfg.solver_iterations,cfg.solver_ls_iterations)==(2,4)


def test_one_cli_preserves_config_through_resume_verify_and_plan(tmp_path):
    from emu.cli import main
    base=training_flags(tmp_path/"study")
    seen=[]
    with patch("emu.run.train_one",side_effect=lambda ns:(seen.append(one_run_config(ns)),0)[1]):
        assert main(base)==0
        assert main(base+["--resume","auto"])==0
        assert main(base+["--resume","auto","--check-resume-only"])==0
    assert len(seen)==3
    assert seen[0].resume in ("",None)
    assert seen[1].resume=="auto" and seen[2].resume=="auto"
    assert replace(seen[0],resume="auto")==seen[1]==seen[2]
    with patch("emu.run.train_one",side_effect=lambda ns:one_run_config(ns)):
        with pytest.raises(ValueError,match="solver-iterations"):
            main(base[:-4]+["--solver-iterations","0",
                           "--solver-ls-iterations","4"])
    # Source-owned plan can be checked without touching the GPU.
    assert main(base+["--plan"])==0


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
    assert cfg.cloud_probes_per_eval==0 and cfg.cloud_final_probes==0
    assert training_sensor_contract(cfg) is not None



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
    assert call["solver_iterations"] is None
    assert call["solver_ls_iterations"] is None
    # Frozen panels must use the trainer's experimental physics cap.
    from dataclasses import replace
    evaluate._world(replace(cfg, solver_iterations=2, solver_ls_iterations=4),
                    seed_offset=0, limits=(.60,-.20,.22,1.50))
    assert call["solver_iterations"] == 2
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
