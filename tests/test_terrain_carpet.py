"""v5041 terrain contract must be exact before replacing v5040 physics."""
from __future__ import annotations

import json
import math
import subprocess
import sys

import pytest

from emu.terrain_carpet import (
    Limits, ROLES, TILE_M, check_neighbourhood, edge, hash01,
    neighbours, role_limits, tile,
)


def test_tile_contract_every_role():
    for role in ("standing",*ROLES):
        bounds=role_limits(role,seed=735,slope_deg=12.,
                           transition_deg=4.,step_up_m=.04,
                           step_down_m=.08)
        for x,y in ((.25,.25),(-14.27,9.61),(124.13,-58.55)):
            world=neighbours(x,y,bounds)
            report=check_neighbourhood(world,bounds)
            assert report["tiles"]==9
            assert report["triangles"]==18
            assert report["seam_error_m"] < 1e-9
            assert report["largest_step_m"]<=min(
                bounds.max_step_up_m,bounds.max_step_down_m)+1e-8
            assert len({tuple(t["index"]) for t in world["tiles"]})==9


def test_unequal_up_down_budget_is_conservative_not_conflated():
    for seed in range(64):
        bounds=role_limits("step",seed=seed,slope_deg=7.,
             transition_deg=3.,step_up_m=.02,step_down_m=.09)
        report=check_neighbourhood(neighbours(4.2,-6.5,bounds),bounds)
        assert report["largest_step_m"]<=.02+1e-9
        assert bounds.max_step_up_m==.02
        assert bounds.max_step_down_m==.09


def test_deterministic_tiles_are_global_not_episode_cache():
    limits=Limits(seed=733,max_step_up_m=.07,max_step_down_m=.07)
    before=tile(-20,15,limits)
    for i in range(-9,10):
        neighbours(i*.49,-i*.37,limits)
    assert before==tile(-20,15,limits)


def test_shared_corners_and_real_vertical_lip():
    bounds=Limits(seed=915,max_slope_deg=7,max_transition_deg=3,
                  max_step_up_m=.045,max_step_down_m=.045,step_probability=1)
    touched=0
    for i in range(-12,12):
        for j in range(-8,8):
            a=tile(i,j,bounds)
            for b in (tile(i+1,j,bounds),tile(i,j+1,bounds)):
                e=edge(a,b)
                assert e["end_gaps_m"][0]==pytest.approx(e["end_gaps_m"][1],abs=1e-9)
                touched+=abs(e["rise_m"])>1e-6
    assert touched>100


def test_zero_specialist_limits_are_physically_flat():
    bounds=role_limits("standing",seed=735,slope_deg=16,
                       transition_deg=6,step_up_m=.15,step_down_m=.12)
    for i in range(-2,2):
        for j in range(-2,2):
            assert all(v[2]==0 for v in tile(i,j,bounds)["vertices"])


def test_invalid_difficulty_rejected():
    with pytest.raises(ValueError):
        Limits(max_slope_deg=math.nan).validated()
    with pytest.raises(ValueError):
        role_limits("unknown",seed=1,slope_deg=1,
                    transition_deg=1,step_up_m=.01,step_down_m=.01)
    with pytest.raises(ValueError):
        edge(tile(0,0,Limits()),tile(2,0,Limits()))


def test_actual_viewer_live_endpoint_exposes_real_training_frontiers(tmp_path):
    from emu.viewer_server import read_terrain_training_snapshot
    root=tmp_path/"run"
    root.mkdir()
    (root/"training_config.json").write_text(json.dumps({
        "seed":735, "observation_schema":"terrain_360",
        "academy_stream":True}))
    (root/"run_meta.json").write_text(json.dumps({
        "source_provenance":{"git_commit":"0123456789abcdef"}}))
    (root/"metrics.csv").write_text(
        "step,slope_max_deg,academy_height_up_m,academy_height_down_m,"
        "academy_up_crossings,academy_down_crossings\n"
        "65000000,2,.009,.012,7,9\n"
        "90000000,8,.022,.035,22,45\n")
    data=read_terrain_training_snapshot(root)
    assert data["schema"]=="emu_terrain_live_v1"
    assert data["training_step"]==90000000
    assert data["run_name"]=="run"
    assert data["source_sha"]=="0123456789abcdef"
    assert float(data["latest_metrics"]["slope_max_deg"])==8
    assert float(data["latest_metrics"]["academy_height_up_m"])==.022
    assert float(data["latest_metrics"]["academy_height_down_m"])==.035
    assert "source training metrics" in data["physics_mode"]
