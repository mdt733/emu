"""No-CUDA acceptance of six-arm 25M continuous-step envelope driver."""
from __future__ import annotations

import argparse
import json

import pytest

from scripts.v5018_height_envelope_campaign import (
    ARMS, SCREEN_HEIGHTS, FINAL_HEIGHTS, assert_args, branch_args, config,
    choose_finalists, main,
)


def arguments(**overrides):
    args = dict(seed=733, envs=1024, foundation_steps=25_000_000,
                screen_increment=10_000_000, final_increment=15_000_000,
                exam_worlds=4, exam_horizon=350)
    args.update(overrides)
    return argparse.Namespace(**args)


def test_plan_only_without_gpu_or_files(capsys):
    assert main([]) == 0
    plan = json.loads(capsys.readouterr().out.split('\nUse --run explicitly')[0])
    assert plan["mode"] == "PLAN_ONLY"
    assert plan["steps"]["nominal_total_training_steps"] > 110_000_000
    assert len(plan["arms"]) == 6


def test_separate_hypotheses_include_real_terrain():
    assert set(ARMS) == {
        "flat_control", "height_cautious", "height_balanced",
        "height_edge", "height_guided", "height_guided_edge",
    }
    assert SCREEN_HEIGHTS == (0, 20, 40, 80, 120)
    assert FINAL_HEIGHTS == (0, 20, 40, 80, 120, 150)
    for name in ARMS:
        flags = branch_args(name)
        if name == "flat_control":
            assert flags == ["--terrain-box-height-mm", "0"]
        else:
            assert "--terrain-height-ceiling-mm" in flags
            assert "--terrain-height-exposure" in flags
            assert "--terrain-height-growth-mm" in flags
            assert "--terrain-progress-weight" in flags
    assert config(arguments())["foundation_steps"] == 25_000_000


def test_reject_unbounded_gpu_requests():
    assert_args(arguments())
    for change in (
        dict(envs=4096), dict(envs=37), dict(exam_worlds=1),
        dict(foundation_steps=8192), dict(screen_increment=8192),
        dict(final_increment=1000), dict(exam_horizon=80), dict(seed=-1)
    ):
        with pytest.raises(ValueError):
            assert_args(arguments(**change))


def record(*, flat=0, h20=0, h40=0, h80=0, h120=0, contact=0, progress=.2):
    return dict(physical=dict(
        by_height_mm={
            "0":dict(passed=flat), "20":dict(passed=h20),
            "40":dict(passed=h40), "80":dict(passed=h80),
            "120":dict(passed=h120),
        }, step_contact_world_frames=contact,
        mean_signed_progress_m=progress))


def test_real_crossings_beat_progress_only():
    scores = {arm: record(flat=3, contact=i, progress=.1 + i*.01)
              for i, arm in enumerate(ARMS)}
    scores["height_cautious"] = record(flat=3, h40=2, contact=1, progress=.4)
    scores["height_guided"] = record(flat=3, h80=1, contact=0, progress=.8)
    chosen = choose_finalists(scores)
    assert set(chosen) == {"height_cautious", "height_guided"}
    assert "flat_control" not in chosen


def test_no_success_is_exploratory_and_retains_gait():
    scores = {arm: record(flat=3, contact=i, progress=.1)
              for i, arm in enumerate(ARMS)}
    scores["height_guided_edge"] = record(flat=0, contact=999, progress=3.)
    chosen = choose_finalists(scores)
    assert len(chosen) == 2
    assert "flat_control" not in chosen
    assert "height_guided_edge" not in chosen
