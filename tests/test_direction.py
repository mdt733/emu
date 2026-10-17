"""Five directional-oracle contracts, including real downhill cheating controls."""
import math
import numpy as np
import pytest
from emu.directional import HeadingState, heading_hold_step, wrap_radians, examine_course
from emu.directional import estimate_ground_plane
from emu.directional import evaluate_trace_arrays

def ideal(commands, initial_yaw=0.0, dt=0.02):
    """Synthetic perfect command following; not a trained policy."""
    c = np.asarray(commands, dtype=float)
    p = np.zeros((len(c) + 1, 2))
    h = np.full(len(c) + 1, initial_yaw)
    for i, (vx, vy, yaw) in enumerate(c):
        p[i+1] = p[i] + dt * np.array([
            vx * np.cos(h[i]) - vy * np.sin(h[i]),
            vx * np.sin(h[i]) + vy * np.cos(h[i]),
        ])
        h[i+1] = h[i] + yaw * dt
    return p, h
def hits_for_world_plane(slope_degrees=0, chassis_pitch_degrees=0):
    """Intersect rigid body-fixed beams with z=tan(slope)*world_x."""
    a=math.radians(chassis_pitch_degrees)
    rot=np.array([[math.cos(a),0,math.sin(a)],
                  [0,1,0],
                  [-math.sin(a),0,math.cos(a)]])
    world_origins=ORIGINS @ rot.T + np.array([0,0,.20])
    world_beams=BEAMS @ rot.T
    t=math.tan(math.radians(slope_degrees))
    ranges=(t*world_origins[:,0]-world_origins[:,2])/(world_beams[:,2]-t*world_beams[:,0])
    gravity_body=rot.T@np.array([0,0,-1.0])
    return ranges,gravity_body
def fake_trace(n=42,worlds=2):
    pos=np.zeros((n,worlds,2))
    pos[:,:,0]=np.arange(n)[:,None]*.002
    pos[12:,1,0]=20.0+np.arange(n-12)*.02  # teleport AFTER termination
    cmd=np.zeros((n,worlds,3))
    cmd[:,:,0]=.1
    heads=np.zeros((n,worlds))
    feet=np.zeros((n,worlds,2,3))
    feet[:,:,:,2]=.018
    contact=np.ones((n,worlds,2),dtype=bool)
    done=np.zeros((n,worlds))
    fell=np.zeros((n,worlds))
    done[11,1]=fell[11,1]=1
    return dict(position_xy=pos,heading_rad=heads,foot_xyz=feet,
                floor_contact=contact,command=cmd,done=done,fell=fell,
                last_position_xy=np.array([[n*.002,0],[999,0]]),
                last_heading_rad=np.zeros(worlds),
                last_foot_xyz=feet[-1],last_floor_contact=contact[-1])
ORIGINS = np.array([[.035,-.05,0],[.035,.05,0],[-.04,0,0]], dtype=float)
BEAMS = np.tile([0,0,-1.0],(3,1))

def test_one_degree_directed_climb_is_credited_with_equivalent_elevation():
    cmd = np.tile([.1, 0, 0], (350, 1))
    pos, heading = ideal(cmd)
    result = examine_course(pos, heading, cmd, uphill_world_xy=(1, 0),
                            slope_deg=1.0, completed_strides=7)
    assert result['course_pass'] and result['walking_pass']
    assert result['signed_course_progress_m'] == pytest.approx(.7)
    assert result['uphill_progress_m'] == pytest.approx(.7)
    assert result['equivalent_elevation_m'] == pytest.approx(.7 * math.sin(math.radians(1)))
    assert result['heading_mae_deg'] == pytest.approx(0)

def test_turn_downhill_cheat_fails_even_if_body_forward_and_final_yaw_rate_are_good():
    cmd = np.tile([.1, 0, 0], (350, 1))
    pos = np.zeros((351, 2))
    heading = np.zeros(351)
    for i in range(350):
        heading[i+1] = min(math.pi, (i+1)*math.pi/75)
        pos[i+1] = pos[i] + .02*.1*np.array([math.cos(heading[i]), math.sin(heading[i])])
    assert abs(heading[-1]-math.pi) < 1e-12
    result = examine_course(pos, heading, cmd, uphill_world_xy=(1, 0),
                            slope_deg=1.0, completed_strides=10)
    assert not result['course_pass'] and not result['walking_pass']
    assert result['uphill_progress_m'] < 0
    assert result['heading_mae_deg'] > 60

def test_heading_wrap_saturation_and_invalid_sensor_reacquisition():
    assert abs(wrap_radians(math.radians(179)-math.radians(-179))) == pytest.approx(math.radians(2))
    state=HeadingState(target_rad=math.pi-.01, has_reference=True)
    state, rate, error = heading_hold_step(state, -math.pi+.01, 0, kp=100, max_rate=.5)
    assert abs(error) < .03 and abs(rate) <= .5
    lost, safe_rate, _ = heading_hold_step(state,float('nan'),.1,measurement_valid=False)
    assert not lost.has_reference and safe_rate == pytest.approx(.1)
    restored, rate, error = heading_hold_step(lost, 2.0, 0)
    assert restored.has_reference and rate == error == 0
    with pytest.raises(ValueError):
        heading_hold_step(restored, 0, 0, dt=0)

def test_tilted_robot_on_flat_floor_is_not_misread_as_inclined_ground():
    ranges,gravity=hits_for_world_plane(0,chassis_pitch_degrees=10)
    r=estimate_ground_plane(ORIGINS,BEAMS,ranges,gravity)
    assert r.valid
    assert r.slope_deg == pytest.approx(0,abs=1e-7)

def test_pre_reset_trace_never_scores_world_reset_teleport():
    result=evaluate_trace_arrays(
        fake_trace(),np.tile([0,0,-9.81],(2,1)),settle_frames=5)
    good,bad=result
    assert good['course_pass'] and not good['walking_pass']
    assert good['full_horizon_survival']
    assert bad['first_fall'] and not bad['course_pass']
    assert bad['first_done_step']==12
    assert bad['frames_recorded']==11
    assert 'uphill_progress_m' not in bad or abs(bad['signed_course_progress_m']) < 1
