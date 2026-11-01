"""Step Academy geometric/curriculum invariants; no GPU needed."""
from __future__ import annotations

import numpy as np
import pytest

from emu.step_academy_world import (ACADEMY_LIPS,ACADEMY_INTERVAL_M,
                                    course_heights,expected_lip_directions)
from emu.capability_focus import (attention_weights,academy_birth_allocation,
                                  academy_focus_mix)
from emu.environment import add_unified_terrain
from emu.contracts import ROBOT_XML,UNIFIED_TERRAIN_FIELDS,P,TERRAIN_TASK_PARAM_DIM

def test_four_lips_have_real_signed_heights_in_all_sequences():
    index=np.arange(4,dtype=np.int32)
    samples=np.full((4,4),.70)
    heights=course_heights(np,index,samples,.02)
    directions=np.sign(expected_lip_directions(heights)).astype(int)
    assert heights.shape==(4,5)
    assert np.all(heights>=0)
    assert np.max(heights)<=.04+1e-6
    assert np.min(np.abs(np.diff(heights,axis=1)))>=.003
    assert directions.tolist()==[
        [1,1,-1,-1],[1,-1,1,-1],
        [-1,1,-1,1],[-1,-1,1,1]]

def test_four_lips_extend_existing_two_slab_model_without_robot_change():
    import mujoco
    original=add_unified_terrain(str(ROBOT_XML))
    upgraded=add_unified_terrain(str(ROBOT_XML),academy_lips=4)
    base=mujoco.MjModel.from_xml_string(original)
    academy=mujoco.MjModel.from_xml_string(upgraded)
    assert academy.nu==base.nu==6
    assert academy.nq==base.nq
    assert academy.nv==base.nv
    assert academy.ngeom==base.ngeom+3
    assert academy.nmocap==base.nmocap+3
    for i in range(3):
        gid=mujoco.mj_name2id(academy,mujoco.mjtObj.mjOBJ_GEOM,
                              f"academy_surface_{i}")
        assert gid>=0
        assert academy.geom_contype[gid]==4
        assert academy.geom_conaffinity[gid]==0
        assert np.isclose(academy.geom_size[gid,0],ACADEMY_INTERVAL_M/2)
    foot=mujoco.mj_name2id(academy,mujoco.mjtObj.mjOBJ_GEOM,
                            "r_foot_ball")
    assert academy.geom_conaffinity[foot]&4
    assert not base.geom_conaffinity[foot]&4

def test_scheduler_cold_start_growth_and_retention():
    def share(step):
        return academy_birth_allocation(step=step,intro_steps=60_000_000,
                   start_steps=80_000_000,ramp_steps=20_000_000,
                   intro_share=.10,max_share=.50)
    assert share(0)==share(20_000_000)==share(59_000_000)==0.
    assert share(70_000_000)==pytest.approx(.05)
    assert share(80_000_000)==pytest.approx(.10)
    assert share(90_000_000)==pytest.approx(.30)
    assert share(100_000_000)==pytest.approx(.50)
    assert share(260_000_000)==pytest.approx(.50)
    base=attention_weights(schedule="balanced",step=100_000_000,
                           foundation_steps=20_000_000)
    mixed,prob=academy_focus_mix(base,.50)
    assert sum(mixed)==pytest.approx(1.)
    assert mixed[3]==pytest.approx(.60)
    assert prob==pytest.approx(5/6)
    assert mixed[0]==pytest.approx(.10)
    assert mixed[1]==pytest.approx(.10)
    assert mixed[2]==pytest.approx(.10)
    # 50% multi-lip births, 10% ordinary single-lip STEP, 40% others.
    assert mixed[3]*prob==pytest.approx(.50)
    assert mixed[3]*(1-prob)==pytest.approx(.10)

def test_teacher_only_packet_extends_indexes_without_moving_old_fields():
    assert P.steering_attention==TERRAIN_TASK_PARAM_DIM+14
    assert P.academy_episode_probability==TERRAIN_TASK_PARAM_DIM+15
    assert UNIFIED_TERRAIN_FIELDS[-1]==("academy_episode_probability",0.)
