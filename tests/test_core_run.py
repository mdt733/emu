"""Six high-signal run contracts: source-locked 140M recipe, actual evaluator, checkpoint.

No old-policy fixtures, no duplicate tests of tests; the bounded live
first/continue/verify test remains a separate explicit GPU acceptance gate.
"""
from __future__ import annotations
from typing import Any, NamedTuple
from unittest.mock import patch

import numpy as np
import pytest

from scripts.run_prism_140m import train_arguments, STEPS, ENVS, EXPECTED_BRANCH
from scripts.smoke_training_lifecycle import configs
from emu.training_config import training_sensor_contract
from emu.contracts import OBS_SCHEMA_TERRAIN
from emu.run import one_run_config


def test_big_run_is_140m_true_prisms_six_specialists(tmp_path):
    args = train_arguments(tmp_path / "run", 735, "start")
    assert args[3:5] == ["emu.cli", "train"]
    for option in ("--carpet-prisms", "--terrain-carpet", "--terrain-actor",
                   "--unified-capabilities", "--academy-stream"):
        assert args.count(option) == 1
    assert args[args.index("--envs") + 1] == str(ENVS)
    assert args[args.index("--steps") + 1] == str(STEPS)
    assert args[args.index("--foundation-steps") + 1] == "20000000"
    assert args[args.index("--academy-max-share") + 1] == ".50"
    assert "--resume" not in args
    assert "--plan" not in args
    assert EXPECTED_BRANCH == "feature/v5045-core-skeleton-audit"


def test_start_resume_check_and_plan_share_exact_physics(tmp_path):
    root = tmp_path / "run"
    base = train_arguments(root, 735, "start")
    assert train_arguments(root, 735, "plan") == base + ["--plan"]
    assert train_arguments(root, 735, "resume") == base + ["--resume", "auto"]
    assert train_arguments(root, 735, "verify") == (
        base + ["--resume", "auto", "--check-resume-only"]
    )
    with pytest.raises(ValueError):
        train_arguments(root, 735, "unknown")
    with pytest.raises(ValueError):
        train_arguments(root, -1, "start")


def test_actual_cli_parses_to_one_immutable_prism_recipe(tmp_path):
    from emu.cli import main as emu_cli_main
    cli_args = train_arguments(tmp_path / "run", 735, "start")[4:]
    configs_seen = []

    def capture(args):
        configs_seen.append(one_run_config(args))
        return 0

    with patch("emu.run.train_one", side_effect=capture):
        assert emu_cli_main(cli_args) == 0
    assert len(configs_seen) == 1
    cfg = configs_seen[0]
    assert cfg.steps == STEPS and cfg.envs == ENVS and cfg.seed == 735
    assert cfg.terrain_carpet and cfg.terrain_carpet_prisms and cfg.academy_stream
    assert cfg.unified_capability_focus
    assert cfg.observation_schema == OBS_SCHEMA_TERRAIN
    assert cfg.terrain_lesson == "coherent-careful"
    assert cfg.teaching_schedule == "terrain-early"
    assert cfg.athletic_rehearsal_frac == pytest.approx(.10)
    assert cfg.retention_share == pytest.approx(.15)
    assert cfg.fixed_eval_every_steps == 10_000_000
    # Keep the 20M cold-start/mature limits identical while deleting
    # unreachable legacy mode branches from the config builder.
    assert cfg.foundation_steps == 20_000_000
    assert cfg.difficulty_base_step == pytest.approx(.0024)
    assert cfg.fall_target == pytest.approx(.34)
    assert (cfg.vx_hi_max, cfg.vx_lo_min, cfg.vy_max_mps, cfg.yaw_max_rad_s) == pytest.approx(
        (.60, -.20, .22, 1.50))
    assert (cfg.combo_probe_frac, cfg.phase_speed_gain, cfg.regime_command_frac,
            cfg.combo_mix_frac) == pytest.approx((.12, 1.25, .85, .88))
    assert cfg.frontier_push_safety_n == pytest.approx(.70)
    assert cfg.push_max_force_n == pytest.approx(4.)
    assert cfg.capability_field_mode == "shadow"
    assert cfg.freeze_obs_norm_after_steps == 0
    # The retired sparse-cloud evaluator must not be used by this lineage.
    assert cfg.cloud_probes_per_eval == 0 and cfg.cloud_final_probes == 0
    assert training_sensor_contract(cfg) is not None
    # The 20M gait experiment must use this SAME proven trainer, explicit
    # physics caps and a saved 10M foundation, not a standalone pseudo-PPO.
    from argparse import Namespace
    from scripts.run_v5046_20m import train_cmd
    trial = Namespace(seed=735, solver_iterations=2, solver_ls_iterations=4)
    trial_args = train_cmd(trial, "start", tmp_path / "trial")[4:]
    cfg20=[]
    with patch("emu.run.train_one", side_effect=lambda ns: (cfg20.append(one_run_config(ns)), 0)[1]):
        assert emu_cli_main(trial_args) == 0
    assert len(cfg20)==1
    assert cfg20[0].steps==20_000_000
    assert cfg20[0].foundation_steps==10_000_000
    assert cfg20[0].solver_iterations==2
    assert cfg20[0].solver_ls_iterations==4
    assert cfg20[0].fixed_eval_every_steps==10_000_000
    assert train_cmd(trial, "resume", tmp_path / "trial") == (
        train_cmd(trial, "start", tmp_path / "trial") + ["--resume", "auto"])
    assert train_cmd(trial, "verify", tmp_path / "trial") == (
        train_cmd(trial, "resume", tmp_path / "trial") + ["--check-resume-only"])
    assert "--plan" in train_cmd(trial, "plan", tmp_path / "trial")
    with patch("emu.run.train_one", side_effect=lambda ns: one_run_config(ns)):
        with pytest.raises(ValueError,match="solver-iterations"):
            emu_cli_main(trial_args[:-2]+["--solver-iterations","0"])

    # Direct callers cannot revive the legacy launch mode or old actor
    # contract by skipping the prism-only CLI parser.
    from dataclasses import replace
    from emu.training import unified_launch_mode
    with pytest.raises(ValueError, match='v5045 trainer requires'):
        unified_launch_mode(replace(cfg, unified_capability_focus=False))
    with pytest.raises(ValueError, match='v5045 trainer requires'):
        unified_launch_mode(replace(cfg, observation_schema='sim2real_232'))
    with pytest.raises(ValueError, match='v5045 trainer requires'):
        unified_launch_mode(replace(cfg, terrain_carpet_prisms=False))
    # Public CLI must refuse a superficially valid legacy launch before GPU
    # model creation, rather than printing a plan for an impossible world.
    with patch("emu.run.train_one", side_effect=lambda ns: one_run_config(ns)):
        with pytest.raises(ValueError, match="v5045 prism-only"):
            emu_cli_main(["train", "--run-dir", str(tmp_path / "legacy"),
                          "--steps", "256"])
    # One current frozen-policy CLI; old two-slab exam routes are retired.
    with patch("emu.prism_exam.evaluate_prism_policy",
               return_value=(tmp_path / "prism_exam.json", tmp_path / "prism_exam.md")) as exam:
        assert emu_cli_main(["prism-exam", "--policy", str(tmp_path / "actor.emupolicy"),
                             "--output", str(tmp_path / "exam"), "--mode", "step"]) == 0
        assert exam.call_args.kwargs["mode"] == "step"


def test_actual_evaluator_uses_prisms_not_boxes(tmp_path, monkeypatch):
    import emu.evaluation as evaluate
    cli_args = train_arguments(tmp_path / "run", 735, "start")[4:]
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
