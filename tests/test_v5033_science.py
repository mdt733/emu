"""v5033: one arm = one explicit saved intervention, with verified physical knobs.

CPU-only unit coverage. The standalone 24-world CPU / 2048 CUDA PPO
smokes additionally compile the actual actor, terrain and reward paths.
"""
from __future__ import annotations

from argparse import Namespace
from dataclasses import asdict
from pathlib import Path

import pytest

from emu.science import ARM_COUNT, ARM_NAMES, arm_name, configure_arm, expected_fields
from emu.run import one_run_config
from emu.curriculum import RegimeGovernor


def base_args():
    """Match the completed *effective* v5032 learner, not its broken manifest."""
    return Namespace(
        run_dir="/tmp/emu_v5033_test_arm",steps=268_000_000,
        envs=2048,seed=761,device="cuda",profile="sim2real",
        unified_capabilities=True,foundation_steps=20_000_000,
        foundation_mix="main",teaching_schedule="balanced",
        velocity_i_ramp_s=.4,velocity_i_multiplier=1.,
        growth_recipe="main-mature",retention_share=0.,
        athletic_rehearsal_share=.25,terrain_lesson="coherent-careful",
        temporal_vx_reward_share=.5,verified_ascent_reallocation=0.,
        force_program="gentle",fixed_eval_every_steps=20_000_000,
        terrain_actor=True,terrain_step_max_m=.08,
        sensor_actor=False,contact_terrain_radius=0.,
        contact_slope_deg=0.,motor_guidance_weight=0.,
        science_arm=0,pitch_deg=None,
    )


def test_registry_is_one_to_one_and_names_and_comparators_are_stable():
    assert ARM_COUNT==24
    assert len(ARM_NAMES)==24
    assert len(set(ARM_NAMES))==24
    assert arm_name(0)=="00_control"
    assert arm_name(5)=="05_ascent_contact_3"
    assert arm_name(23)=="23_walking_action_smoothing"
    for invalid in (-2,-1,24,100):
        with pytest.raises(ValueError):
            arm_name(invalid)


def test_actual_cli_training_configs_resolve_every_intervention():
    from emu.science import _CONFIG_OVERRIDES
    baseline=asdict(one_run_config(base_args()))
    assert baseline["science_arm"]==0
    assert baseline["verified_ascent_reallocation"]==0.
    assert baseline["temporal_vx_reward_share"]==.5
    assert baseline["athletic_rehearsal_frac"]==.25
    assert baseline["observation_schema"]=="terrain_360"
    for arm in range(ARM_COUNT):
        args=base_args()
        args.science_arm=arm
        actual=asdict(one_run_config(args))
        exp=expected_fields(arm)
        for field,value in exp.items():
            assert actual[field]==value,(arm,field,actual[field],value)
        # The ONLY config changes relative to the control are the arm ID
        # and that arm's explicitly registered ordinary training knobs.
        diffs={k for k in actual if actual[k]!=baseline[k]}
        assert diffs=={k for k in exp if exp[k]!=baseline[k]},(arm,diffs)
        for key in ("observation_schema","envs","seed","foundation_steps",
                    "terrain_step_max_m","unified_capability_focus"):
            assert actual[key]==baseline[key]


def test_old_cli_ascent_flag_really_reaches_saved_config():
    args=base_args()
    args.science_arm=-1
    args.verified_ascent_reallocation=3.
    c=one_run_config(args)
    assert c.science_arm==-1
    assert c.verified_ascent_reallocation==3.
    args.verified_ascent_reallocation=7.
    with pytest.raises(ValueError):
        one_run_config(args)
    args.science_arm=0
    args.verified_ascent_reallocation=3.
    with pytest.raises(ValueError,match="base"):
        one_run_config(args)


def _episode_row(*, near_attempts=0, near_wins=0, easy_attempts=0,
                 easy_wins=0):
    row={}
    for d in ("up","down"):
        row.update({f"terrain_near_attempts_{d}":near_attempts,
                    f"terrain_near_successes_{d}":near_wins,
                    f"terrain_easy_attempts_{d}":easy_attempts,
                    f"terrain_easy_successes_{d}":easy_wins})
    return row


def _governor(arm: int):
    g=RegimeGovernor(unified_capability_focus=True,open_frontier=True,
                     science_arm=arm)
    g.params.terrain_height_m=.030
    g.params.terrain_exposure=.15
    return g


def _advance(g,row):
    return g.advance_terrain(row,step=266_000_000,max_height_m=.08,
                             foundation_steps=20_000_000,unified=True)


def test_step_promotion_is_one_millimetre_only_in_arm_01():
    ctrl=_governor(0)
    fine=_governor(1)
    row=_episode_row(near_attempts=8,near_wins=8)
    assert _advance(ctrl,row)=="terrain_height_expand_real_cross"
    assert _advance(fine,row)=="terrain_height_expand_real_cross"
    assert ctrl.params.terrain_height_m==pytest.approx(.0354)
    assert fine.params.terrain_height_m==pytest.approx(.0310)


def test_step_retreat_uses_easy_failures_not_failed_hard_probes():
    ctrl=_governor(0)
    fine=_governor(2)
    row=_episode_row(easy_attempts=4,easy_wins=0)
    assert _advance(ctrl,row)=="terrain_retreat_easy_failure"
    assert _advance(fine,row)=="terrain_retreat_easy_failure"
    assert ctrl.params.terrain_height_m==pytest.approx(.0246)
    assert fine.params.terrain_height_m==pytest.approx(.029)
    untouched=_governor(2)
    assert _advance(untouched,_episode_row(near_attempts=4,near_wins=0)) == "terrain_hold_probe_learning"
    assert untouched.params.terrain_height_m==pytest.approx(.030)


def test_twelve_real_attempts_prevent_four_lucky_promotions():
    ctrl=_governor(0)
    conservative=_governor(3)
    row=_episode_row(near_attempts=4,near_wins=4)
    assert _advance(ctrl,row)=="terrain_height_expand_real_cross"
    assert _advance(conservative,row)=="terrain_collect_real_step"
    assert _advance(conservative,_episode_row(near_attempts=8,near_wins=8)) == "terrain_height_expand_real_cross"


def test_historical_training_config_accepts_missing_science_arm(tmp_path):
    from emu.training_config import TrainingConfig
    import json
    c=one_run_config(base_args())
    data=asdict(c)
    data.pop("science_arm")
    file=tmp_path/"old_config.json"
    file.write_text(json.dumps(data))
    restored=TrainingConfig.load(file)
    assert restored.science_arm==-1


def test_research_mode_does_not_change_360_actor_contract():
    for arm in range(ARM_COUNT):
        args=base_args();args.science_arm=arm
        cfg=one_run_config(args)
        assert cfg.observation_schema=="terrain_360"
        assert cfg.unified_capability_focus
        assert cfg.terrain_step_max_m==.08
