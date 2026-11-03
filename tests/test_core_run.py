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
    assert training_sensor_contract(cfg) is not None


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


def test_prism_two_update_checkpoint_lifecycle_recipe(tmp_path):
    first, second = configs(tmp_path, unified=True, carpet=True,
                            prisms=True, envs=8, seed=735)
    assert first.terrain_carpet_prisms and second.terrain_carpet_prisms
    assert first.terrain_carpet and first.academy_stream
    assert first.observation_schema == OBS_SCHEMA_TERRAIN
    assert first.steps == 256 and second.steps == 512
    with pytest.raises(ValueError,match="prisms"):
        configs(tmp_path, unified=True, carpet=True, envs=8, seed=735)
    with pytest.raises(ValueError):
        configs(tmp_path, prisms=True, envs=8, seed=735)


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
