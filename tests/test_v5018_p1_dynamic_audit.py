"""Pure behavioural checks of new P1 REAL-motion trace aggregation.

No MuJoCo required here. Validate our summary counts BEFORE spending GPU
time, including a frame with two simultaneous foot/box contacts. These are
reporting contracts, not proof of physically supported stepping.
"""
from __future__ import annotations

import numpy as np

from scripts.v5018_p1_dynamic_reward_audit import summarise_trace


def test_separate_box_world_frames_from_foot_contact_frames():
    n,w=5,2
    x=np.zeros((n,w),dtype=np.float32)
    positions=np.zeros((n,w,2),dtype=np.float32)
    positions[:,0,0]=np.arange(n,dtype=np.float32)*.04
    contacts=np.zeros((n,w,2),dtype=bool)
    contacts[1,0,0]=True
    contacts[2,0,:]=True        # TWO feet in ONE contact world-frame
    contacts[3,0,1]=True
    top=x.copy()
    top[1:3,0]=.5
    riser=x.copy()
    riser[2:4,0]=.5
    active=x.copy()
    active[1:4,0]=1.
    support=x.copy()
    support[1:4,0]=.5
    trace={
        'position_xy':positions,
        'foot_xyz':np.zeros((n,w,2,3),np.float32),
        'reward_scalar':x.copy(),
        'actor_action':np.zeros((n,w,6),np.float32),
        'challenge_contact':contacts,
        'floor_contact':contacts.copy(),
        'terrain_geometry_active':active,
        'terrain_top_contact_fraction':top,
        'terrain_riser_contact_fraction':riser,
        'terrain_floor_contact_fraction':x.copy(),
        'terrain_signed_toe_gap':x.copy(),
        'contact_support':support,
        'swing_clearance':x.copy(),
        'done':x.copy(),
        'fell':x.copy(),
    }
    result, arrays=summarise_trace(trace,horizon=n,worlds=w)
    assert result[0]['actual_box_contact_world_frames']==3
    assert result[0]['actual_box_foot_contact_frames']==4
    assert result[0]['box_contact_onsets']==2
    assert result[0]['top_normal_contact_frames']==2
    assert result[0]['riser_normal_contact_frames']==2
    assert result[0]['terrain_active_frames']==3
    assert abs(result[0]['signed_forward_progress_m']-.16)<1e-6
    assert result[1]['actual_box_contact_world_frames']==0
    assert arrays['actor_action'].shape==(n,w,6)


def test_first_terminal_truncates_reset_contaminated_physics():
    n,w=5,2
    x=np.zeros((n,w),dtype=np.float32)
    contacts=np.zeros((n,w,2),dtype=bool)
    contacts[3:,0,:]=True
    done=x.copy()
    done[2,0]=1.
    trace=dict(
        position_xy=np.zeros((n,w,2),np.float32),
        foot_xyz=np.zeros((n,w,2,3),np.float32),
        reward_scalar=x.copy(),actor_action=np.zeros((n,w,6),np.float32),
        challenge_contact=contacts,floor_contact=contacts.copy(),
        terrain_geometry_active=x.copy(),
        terrain_top_contact_fraction=x.copy(),
        terrain_riser_contact_fraction=x.copy(),
        terrain_floor_contact_fraction=x.copy(),
        terrain_signed_toe_gap=x.copy(),
        contact_support=x.copy(),swing_clearance=x.copy(),
        done=done,fell=x.copy(),
    )
    results,_=summarise_trace(trace,horizon=n,worlds=w)
    assert results[0]['terminal_before_horizon']
    assert results[0]['frames_to_first_terminal']==3
    assert results[0]['actual_box_contact_world_frames']==0
    assert not results[1]['terminal_before_horizon']
