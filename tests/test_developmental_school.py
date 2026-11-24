"""Next-school shared terrain: dose ramp, per-role physical clips and CLI.

This is a regression contract, NOT evidence that mixed tasks will
prevent forgetting or that 32-80mm uphill is physically achievable.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from emu.capability_focus import FOCUS_NAMES
from emu.contracts import P, TERRAIN_TASK_PARAM_DIM, UNIFIED_TERRAIN_FIELDS
from emu.developmental_school import (
    CROSS_STAIR_ROLE_CAP_M,CROSS_STAIR_ROLE_PROBABILITIES,
    cross_stair_exposure_blend,cross_stair_plan)
from emu.terrain_carpet_mjx import cross_stair_assignment, role_envelope


def test_birth_ramp_does_not_set_difficulty_or_change_existing_roles():
    assert len(CROSS_STAIR_ROLE_PROBABILITIES)==len(FOCUS_NAMES)==6
    assert CROSS_STAIR_ROLE_PROBABILITIES[FOCUS_NAMES.index("step")]==0.
    assert cross_stair_exposure_blend(step=0,enabled=True)==0.
    assert cross_stair_exposure_blend(step=90_000_000,enabled=True)==0.
    assert cross_stair_exposure_blend(step=100_000_000,enabled=True)==.5
    assert cross_stair_exposure_blend(step=110_000_000,enabled=True)==1.
    assert cross_stair_exposure_blend(step=150_000_000,enabled=True)==1.
    assert cross_stair_exposure_blend(step=140_000_000,enabled=False)==0.
    assert cross_stair_plan(.5)["slope"]==pytest.approx(.075)
    assert cross_stair_plan(1.)["control"]==pytest.approx(.06)
    with pytest.raises(ValueError):
        cross_stair_exposure_blend(step=-1,enabled=True)
    with pytest.raises(ValueError):
        cross_stair_exposure_blend(step=110_000_000,enabled=True,ramp_steps=0)
    assert UNIFIED_TERRAIN_FIELDS[-1]==("cross_stair_blend",0.)
    assert P.cross_stair_blend==TERRAIN_TASK_PARAM_DIM+18


def test_new_stair_births_are_measured_small_and_not_untaught_heights():
    n=8192
    seeds=np.arange(1,n+1,dtype=np.int32)
    roles=np.arange(n,dtype=np.int32)%6
    specialist=np.ones(n,dtype=bool)
    athletic=np.zeros(n,dtype=np.float32)
    academy=np.zeros(n,dtype=bool)
    zeros=np.zeros(n,dtype=np.float32)
    up=np.full(n,.003,dtype=np.float32)   # uphill Academy never earned >3mm
    down=np.full(n,.040,dtype=np.float32) # downhill already strong: do NOT copy 40mm
    old=role_envelope(np,roles,specialist,athletic,
                      np.full(n,3.,np.float32),up,down,academy,
                      0.,0.,seeds)
    assert all(np.array_equal(a,zeros) for a in old)
    assigned=cross_stair_assignment(np,roles,specialist,athletic,seeds,1.)
    newer=role_envelope(np,roles,specialist,athletic,
                        np.full(n,3.,np.float32),up,down,academy,
                        0.,0.,seeds,cross_stair_blend=1.)
    assert np.array_equal(newer[0],zeros) # mixed stair never adds unearned slope
    assert assigned.any()
    assert not assigned[roles==FOCUS_NAMES.index("step")].any()
    assert np.array_equal(newer[1]>0.,assigned)
    assert np.array_equal(newer[2]>0.,assigned)
    caps=np.asarray(CROSS_STAIR_ROLE_CAP_M,np.float32)[roles]
    assert np.all(newer[1]<=up+1e-7)
    assert np.all(newer[2]<=caps+1e-7)
    assert np.allclose(newer[1][assigned],.003,atol=1e-7)
    for focus,name in enumerate(FOCUS_NAMES):
        mask=roles==focus
        actual=assigned[mask].mean()
        budget=CROSS_STAIR_ROLE_PROBABILITIES[focus]
        if focus==FOCUS_NAMES.index("step"):
            assert actual==0.
        else:
            assert abs(actual-budget)<.03,(name,actual,budget)
    assert not cross_stair_assignment(
        np,roles,specialist,athletic,seeds,0.).any()
    assert not cross_stair_assignment(
        np,roles,np.zeros(n,dtype=bool),athletic,seeds,1.).any()
    assert not cross_stair_assignment(
        np,roles,specialist,np.ones(n,dtype=np.float32),seeds,1.).any()


def test_cli_requires_real_safe_ascent_school_for_mixed_stairs(tmp_path):
    from emu.cli import main
    from emu.run import one_run_config
    argv=["train","--run-dir",str(tmp_path/"nursery"),
          "--steps","120000000","--envs","16","--seed","9501",
          "--device","cpu","--profile","sim2real",
          "--terrain-actor","--terrain-step-max-m",".055",
          "--unified-capabilities","--terrain-carpet","--carpet-prisms",
          "--academy-stream","--terrain-lesson","coherent-careful",
          "--growth-recipe","main-mature"]
    with patch("emu.run.train_one",return_value=0) as training:
        assert main(argv)==0
        assert training.call_args.args[0].cross_specialty_stairs is False
        assert main(argv+["--stair-survival-curriculum",
                          "--cross-specialty-stairs"])==0
        cfg=training.call_args.args[0]
        assert cfg.cross_specialty_stairs is True
        assert cfg.cross_specialty_stairs_start_steps==90_000_000
        assert cfg.cross_specialty_stairs_ramp_steps==20_000_000
    with pytest.raises(ValueError,match="stair-survival-curriculum"):
        one_run_config(__import__("argparse").Namespace(
            **{**vars(__import__("argparse").Namespace(
                **{"run_dir":str(tmp_path/"bad"),"steps":120000000,
                   "envs":16,"seed":9501,"device":"cpu","profile":"sim2real",
                   "terrain_actor":True,"terrain_step_max_m":.055,
                   "unified_capabilities":True,"terrain_carpet":True,
                   "carpet_prisms":True,"academy_stream":True,
                   "terrain_lesson":"coherent-careful","growth_recipe":"main-mature",
                   "cross_specialty_stairs":True})),
            })))
