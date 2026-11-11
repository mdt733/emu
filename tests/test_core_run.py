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


def test_current_cli_forks_preserve_sensor_and_reward_economics(tmp_path, monkeypatch):
    # Parent console is a read-only part of the launch/CLI contract, not
    # a new independent physics test or PPO teaching capability.
    _assert_readonly_console_dashboard_five_million_step_trends_and_tty(
        tmp_path, monkeypatch)
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
    # The real overnight school is ONE cold 180M trainer, not a
    # 120M checkpoint fork with walking and stair coaching overlapped.
    school_flags=[
        "train","--run-dir",str(tmp_path/"progressive"),
        "--steps","180000000","--envs","2048","--device","cuda",
        "--seed","735","--profile","sim2real",
        "--terrain-actor","--terrain-step-max-m",".08",
        "--unified-capabilities","--terrain-carpet","--carpet-prisms",
        "--academy-stream","--terrain-lesson","coherent-careful",
        "--growth-recipe","main-mature","--teaching-schedule","staged",
        "--progressive-school",
    ]
    with patch("emu.run.train_one",side_effect=capture):
        assert emu_cli_main(school_flags)==0
    school_cfg=seen[-1]
    assert school_cfg.progressive_school and school_cfg.outcome_graduation
    assert school_cfg.foundation_steps==20_000_000
    assert school_cfg.outcome_walk_start_steps==60_000_000
    assert school_cfg.academy_intro_steps==80_000_000
    assert school_cfg.academy_start_steps==80_000_000
    assert school_cfg.academy_ramp_steps==20_000_000
    assert school_cfg.academy_height_cap_m==pytest.approx(.08)
    assert school_cfg.science_step_finish_on_transfer
    from emu.capability_focus import school_lesson,school_without_steps
    assert school_lesson(step=0,core_competence=0.,
                         walking_graduated=False,stair_mastered_step=-1)['stage']=='foundation'
    assert school_lesson(step=40_000_000,core_competence=.9,
                         walking_graduated=False,stair_mastered_step=-1)['stage']=='supported_growth'
    fading=school_lesson(step=70_000_000,core_competence=.9,
                         walking_graduated=False,stair_mastered_step=-1)
    assert fading['stage']=='walking_graduation'
    assert fading['walk_alpha']==pytest.approx(.5)
    not_ready=school_lesson(step=100_000_000,core_competence=.38,
                         walking_graduated=False,stair_mastered_step=-1)
    assert not not_ready['stairs_enabled']
    assert not_ready['stair_alpha']==0.
    approved=school_lesson(step=80_000_000,core_competence=.55,
                         walking_graduated=True,stair_mastered_step=-1)
    assert approved['stairs_enabled'] and approved['stair_alpha']==0.
    assert approved['down_share']==0.
    mastered=school_lesson(step=110_000_000,core_competence=.20,
                         walking_graduated=True,stair_mastered_step=100_000_000)
    assert mastered['stair_alpha']==pytest.approx(.5)
    assert mastered['down_share']==pytest.approx(.1)
    weights=school_without_steps((.2,.2,.2,.2,.1,.1))
    assert sum(weights)==pytest.approx(1.) and weights[3]==0.
    with pytest.raises(ValueError,match='progressive school requires'):
        with patch("emu.run.train_one",side_effect=lambda ns:one_run_config(ns)):
            emu_cli_main(school_flags+["--foundation-steps","10000000"])
    assert not {"cloud_probes_per_eval","cloud_final_probes",
                "science_reset_step_evidence_on_fork"} & asdict(cfg).keys()
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
    # Actual H40 120M ancestor schema: five new graduation fields absent,
    # three retired settings present. Adapt only for an external research fork,
    # not for a same-source retry or checkpoint continuation.
    ancestor=asdict(first)
    for key in ('outcome_graduation', 'outcome_walk_start_steps',
                'outcome_stair_approved', 'outcome_approval_start_steps',
                'outcome_down_share'):
        ancestor.pop(key)
    ancestor.update(cloud_final_probes=0, cloud_probes_per_eval=0,
                    science_reset_step_evidence_on_fork=False)
    cfg_path.write_text(json.dumps(ancestor), encoding="utf-8")
    with pytest.raises(ValueError, match="training config keys mismatch"):
        TrainingConfig.load(cfg_path)
    migrated=TrainingConfig.load(cfg_path, research_parent=True)
    assert migrated == first
    from emu.run import verify_experiment_fork
    verify_experiment_fork(migrated, replace(first, outcome_graduation=True))
    # Parent recipe is parseable, but its old success semantics must not
    # masquerade as measured 50mm stair-completion evidence.
    from emu.training import compatible_academy_evidence_recipe
    assert not compatible_academy_evidence_recipe(
        migrated, replace(first, outcome_graduation=True,
                          science_step_finish_on_transfer=True))
    for key, value in (('cloud_final_probes', 10),
                       ('cloud_probes_per_eval', 1),
                       ('science_reset_step_evidence_on_fork', True)):
        invalid={**ancestor, key:value}
        cfg_path.write_text(json.dumps(invalid), encoding="utf-8")
        with pytest.raises(ValueError, match="non-default"):
            TrainingConfig.load(cfg_path, research_parent=True)
    cfg_path.write_text(json.dumps({**ancestor, 'unknown_legacy_switch':0}),
                        encoding="utf-8")
    with pytest.raises(ValueError, match="training config keys mismatch"):
        TrainingConfig.load(cfg_path, research_parent=True)
    first.save(cfg_path)
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



def _assert_readonly_console_dashboard_five_million_step_trends_and_tty(tmp_path, monkeypatch):
    """Display-only: compare two *sampled* 5M windows across resumed history."""
    import csv
    import io
    from emu.console_dashboard import (
        ConsoleRelay, console_mode, read_history, render_screen,
        metric_delta, stair_window_trend, latest_heldout,
    )
    path = tmp_path / "metrics.csv"
    data = [
        dict(step=str(i * 1_000_000),
             reward_mean=str(1. + i / 10),
             fall_rate=str(.40 - i / 100),
             ep_len_mean=str(200 + 10 * i),
             core_competence=str(.3 + i / 20),
             academy_episodes="10",
             academy_short_climb_successes="2" if 1 <= i <= 6 else "5",
             eval_policy_step=str((i // 5) * 5_000_000),
             eval_broad_score=str(.80 + (i // 5) * .02))
        for i in range(12)
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)
    history = read_history(path)
    assert len(history) == 12
    assert metric_delta(history, "reward_mean", precision=2) == "+0.50"
    assert metric_delta(history, "fall_rate", scale=100, precision=1,
                        suffix="pp") == "-5.0pp"
    assert metric_delta(history[:4], "reward_mean") == "--"
    assert "5M sampled: 25/50 (50.0%)" in stair_window_trend(history)
    assert "Δ vs prior 5M +30.0pp" in stair_window_trend(history)
    assert "Δ since 5M +0.020" in latest_heldout(history)
    screen = render_screen(history[-1], phase="PPO",
                           target_steps=300_000_000,
                           run_name="test-run", history=history,
                           width=80, height=24)
    assert "WALKING" in screen and "STAIRS" in screen
    assert "Δ vs prior 5M" in screen
    assert "-5.0pp" in screen
    assert all(len(line) < 80 for line in screen.splitlines())
    assert len(screen.splitlines()) <= 22

    # Empty or sparse histories must NOT fabricate a directional trend.
    sparse = [dict(step="10000000", academy_episodes="0")]
    assert metric_delta(sparse, "reward_mean") == "--"
    assert "insufficient" in stair_window_trend(sparse)

    stream = io.StringIO()
    assert console_mode("auto", stream) == "compact"
    compact = ConsoleRelay(tmp_path, target_steps=300_000_000,
                           mode="compact", stream=stream)
    compact.forward("[emu_v5015] step=11,000,000 rew=+2.100\n")
    assert "Δ5M=" in stream.getvalue()
    assert "\x1b[" not in stream.getvalue()

    class FakeTty(io.StringIO):
        def isatty(self):
            return True

    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.delenv("NO_COLOR", raising=False)
    tty = FakeTty()
    relay = ConsoleRelay(tmp_path, target_steps=300_000_000,
                         mode="dashboard", stream=tty)
    relay.forward("[emu_v5015] step=11,000,000 rew=+2.100\n")
    relay.forward("[emu_v5015] SCHOOL assisted_stairs at 11M\n")
    relay.close()
    assert "\x1b[2J" in tty.getvalue()
    assert "Δ vs prior 5M" in tty.getvalue()
    assert tty.getvalue().endswith("\x1b[?25h\n")
