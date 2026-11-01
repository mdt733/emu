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

    from emu.capability_focus import academy_conditional_birth_share
    conditional,actual=academy_conditional_birth_share(.50,.25)
    assert actual==pytest.approx(.50)
    mix,selected=academy_focus_mix(base,conditional)
    # 25% preserve fast command/speed; 50% of ALL births get stairs;
    # remaining 25% remain ordinary slopes/control/yaw/recovery/steps.
    assert .75*mix[3]*selected==pytest.approx(.50)
    assert .25+.75*(1.-conditional)==pytest.approx(.50)
    conditional,actual=academy_conditional_birth_share(.50,.50)
    assert actual==pytest.approx(.35)
    assert .50*(1.-conditional)==pytest.approx(.15)

def test_teacher_only_packet_extends_indexes_without_moving_old_fields():
    assert P.steering_attention==TERRAIN_TASK_PARAM_DIM+14
    assert P.academy_episode_probability==TERRAIN_TASK_PARAM_DIM+15
    assert P.academy_height_m==TERRAIN_TASK_PARAM_DIM+16
    assert UNIFIED_TERRAIN_FIELDS[-2]==("academy_episode_probability",0.)
    assert UNIFIED_TERRAIN_FIELDS[-1]==("academy_height_m",0.003)

def test_independent_episode_budgets_are_native_to_the_one_mjx_environment():
    import inspect
    from emu.environment import EmuMJXEnv
    sig=inspect.signature(EmuMJXEnv)
    assert sig.parameters['max_episode_steps'].default==700
    assert sig.parameters['academy_course_episode_frames'].default==1200


def test_academy_promotes_only_at_signed_approached_near_height():
    from emu.curriculum import RegimeGovernor
    g=RegimeGovernor(unified_capability_focus=True)
    initial=g.params.terrain_height_m
    good={'academy_active_fraction':0.5,
          'academy_near_attempts_up':16.,'academy_near_successes_up':12.}
    # A brilliant uphill staircase alone is no reason to increase downhill.
    assert g.advance_academy(good,max_height_m=.04,enabled=True) == (
        'academy_collect_real_lips')
    assert g.params.academy_height_m==pytest.approx(.003)
    down={'academy_active_fraction':0.5,
          'academy_near_attempts_down':16.,'academy_near_successes_down':10.}
    assert g.advance_academy(down,max_height_m=.04,enabled=True) == (
        'academy_height_expand_real_four_lips')
    assert g.params.academy_height_m==pytest.approx(.0035)
    assert g.params.terrain_height_m==initial, 'Old single-lip frontier must remain independent'
    # No stale evidence from the lower 3mm stair can promote the new 3.5mm one.
    assert all(v==0 for v in g.academy_evidence.values())


def test_academy_retreat_on_failed_easy_but_not_hard_frontier():
    from emu.curriculum import RegimeGovernor
    g=RegimeGovernor(unified_capability_focus=True)
    g.params.academy_height_m=.012
    hard={'academy_active_fraction':.5,
          'academy_near_attempts_up':12.,'academy_near_successes_up':1.,
          'academy_near_attempts_down':12.,'academy_near_successes_down':8.}
    assert g.advance_academy(hard,max_height_m=.04,enabled=True)==(
        'academy_hold_near_support_learning')
    assert g.params.academy_height_m==pytest.approx(.012)
    easy={'academy_active_fraction':.5,
          'academy_easy_attempts_up':12.,'academy_easy_successes_up':1.}
    assert g.advance_academy(easy,max_height_m=.04,enabled=True)==(
        'academy_retreat_failed_easy_support')
    assert g.params.academy_height_m==pytest.approx(.0115)
    assert all(v==0 for v in g.academy_evidence.values())


def test_academy_governor_rollback_exact_and_off_means_no_effect():
    from emu.curriculum import RegimeGovernor
    g=RegimeGovernor(unified_capability_focus=True)
    partial={'academy_active_fraction':.3,
             'academy_near_attempts_up':4.,
             'academy_near_successes_up':2.}
    assert g.advance_academy(partial,max_height_m=.04,enabled=True)==(
        'academy_collect_real_lips')
    saved=g.state_dict()
    restored=RegimeGovernor(unified_capability_focus=True)
    restored.load_state_dict(saved)
    assert restored.state_dict()==saved
    assert restored.advance_academy(
        {'academy_active_fraction':.5,
         'academy_near_attempts_up':50.,
         'academy_near_successes_up':50.},
        max_height_m=.04,enabled=False)=='academy_disabled'
    assert restored.state_dict()==saved
    # Old saved unified checkpoints did not have the new school counters,
    # and still deserve a validated backward-compatible explicit fork.
    old=dict(saved)
    old.pop('academy_evidence')
    old['params']=dict(old['params'])
    old['params'].pop('academy_height_m')
    legacy=RegimeGovernor(unified_capability_focus=True)
    legacy.load_state_dict(old)
    assert legacy.params.academy_height_m==pytest.approx(.003)


def test_a_sphere_grazing_a_vertical_step_wall_is_not_a_supported_foothold():
    from emu.directional import supported_top_foot_contacts
    geom1=np.asarray([[11,11]])
    geom2=np.asarray([[22,23]])
    dist=np.asarray([[0.,0.]])
    frame=np.zeros((1,2,3,3),dtype=float)
    frame[0,0,0,:]=[0.,0.,1.]  # real top contact, upward normal
    frame[0,1,0,:]=[1.,0.,0.]  # vertical side grazing, NO upper support
    pairs=supported_top_foot_contacts(
        np,geom1,geom2,dist,frame,11,(22,23),
        np.asarray([.25]),np.asarray([[.25,.25]]))
    assert pairs.tolist()==[[True,False]]
    # A foot alongside the wall, even with an upward normal, cannot be
    # counted unless its sphere actually rests at the upper surface.
    frame[0,1,0,:]=[0.,0.,-1.]
    low=supported_top_foot_contacts(
        np,geom1,geom2,dist,frame,11,(22,23),
        np.asarray([.25]),np.asarray([[.25,.22]]))
    assert low.tolist()==[[True,False]]
    # MuJoCo swaps contact normal sign if geom pair ordering reverses.
    frame[0,0,0,:]=[0.,0.,-1.]
    reversed_pair=supported_top_foot_contacts(
        np,geom1,geom2,dist,frame,11,(22,23),
        np.asarray([.25]),np.asarray([[.25,.25]]))
    assert reversed_pair.tolist()==[[True,True]]
