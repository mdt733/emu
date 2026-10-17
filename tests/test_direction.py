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

def test_physical_trace_duty_work_and_terminal_truncation():
    """No post-reset torque may contaminate a terminal world's evidence."""
    from emu.directional import examine_actuator_trace
    torque = np.zeros((8, 2, 6))
    omega = np.zeros_like(torque)
    torque[1:5, 0, 0] = 1.0
    omega[1:5, 0, 0] = 2.0
    torque[5:, 0, :] = 999.0  # teleport/reset contaminant
    data = dict(
        servo_torque_nm=torque,
        servo_joint_rate_rad_s=omega,
        servo_peak_limit_nm=np.full((2, 6), 1.913),
        servo_shaft_abs_power_w=np.zeros((8, 2)),
        servo_shaft_drive_power_w=np.zeros((8, 2)),
        servo_shaft_brake_power_w=np.zeros((8, 2)),
        spring_potential_energy_j=np.zeros((8, 2)),
        shin_spring_energy_j=np.zeros((8, 2)),
    )
    result = examine_actuator_trace(data, 0, start=0, stop=4, dt=.02, terminal=True)
    assert result['actuator_sample_frames'] == 4
    assert result['servo_peak_abs_torque_nm'][0] == pytest.approx(1.0)
    assert result['servo_over_continuous_seconds'][0] == pytest.approx(.08)
    assert result['servo_positive_shaft_work_j'][0] == pytest.approx(.16)
    assert result['servo_thermal_exposure_proxy_s'][0] == pytest.approx(.08 / .638**2)
    assert result['servo_all_sample_over_continuous_fraction'] == pytest.approx(1 / 6)


def test_physical_trace_stall_has_heat_exposure_but_no_shaft_work():
    from emu.directional import examine_actuator_trace
    torque = np.ones((4, 1, 6))
    data = dict(servo_torque_nm=torque,
                servo_joint_rate_rad_s=np.zeros_like(torque),
                servo_peak_limit_nm=np.full((1,6),1.913),
                servo_shaft_abs_power_w=np.zeros((4,1)),
                servo_shaft_drive_power_w=np.zeros((4,1)),
                servo_shaft_brake_power_w=np.zeros((4,1)),
                spring_potential_energy_j=np.zeros((4,1)),
                shin_spring_energy_j=np.zeros((4,1)))
    result=examine_actuator_trace(data,0,start=0,stop=4,dt=.02,terminal=False)
    assert result['servo_measured_interval_abs_shaft_work_j'] == 0
    assert all(x > 0 for x in result['servo_thermal_exposure_proxy_s'])


def test_uninterrupted_physical_trace_includes_final_state():
    """An unbroken horizon has N scored intervals, not N-1."""
    from emu.directional import examine_actuator_trace
    torque=np.ones((5,1,6))*0.1
    speed=np.ones_like(torque)*2.
    trace=dict(
        servo_torque_nm=torque,
        servo_joint_rate_rad_s=speed,
        last_servo_torque_nm=np.ones((1,6))*.2,
        last_servo_joint_rate_rad_s=np.ones((1,6))*2.,
        servo_peak_limit_nm=np.full((1,6),1.913),
        servo_shaft_abs_power_w=np.zeros((5,1)),
        servo_shaft_drive_power_w=np.zeros((5,1)),
        servo_shaft_brake_power_w=np.zeros((5,1)),
        spring_potential_energy_j=np.zeros((5,1)),
        shin_spring_energy_j=np.zeros((5,1)),
    )
    result=examine_actuator_trace(trace,0,start=1,stop=5,dt=.02,terminal=False)
    assert result['actuator_sample_frames']==4
    assert result['servo_peak_abs_torque_nm'][0]==pytest.approx(.2)
    assert result['servo_positive_shaft_work_j'][0]==pytest.approx(
        (3*.1*2 + .2*2)*.02)


def test_report_energy_respects_settle_window_and_progress_direction():
    """No free efficiency credit for reverse drift or warm-up work."""
    n=10
    trace=dict(
        position_xy=np.zeros((n,1,2)),
        heading_rad=np.zeros((n,1)),
        foot_xyz=np.zeros((n,1,2,3)),
        floor_contact=np.ones((n,1,2),dtype=bool),
        command=np.zeros((n,1,3)),
        done=np.zeros((n,1)),
        fell=np.zeros((n,1)),
        last_position_xy=np.array([[.030,0.]]),
        last_heading_rad=np.zeros(1),
        last_foot_xyz=np.zeros((1,2,3)),
        last_floor_contact=np.ones((1,2),dtype=bool),
    )
    trace['position_xy'][:,:,0]=np.arange(n)[:,None]*.003
    trace['foot_xyz'][:,:,:,2]=.018
    trace['last_foot_xyz'][:,:,2]=.018
    trace['command'][:,:,0]=.1
    trace['servo_torque_nm']=np.ones((10,1,6))*.1
    trace['servo_joint_rate_rad_s']=np.ones((10,1,6))*.2
    trace['last_servo_torque_nm']=np.ones((1,6))*.1
    trace['last_servo_joint_rate_rad_s']=np.ones((1,6))*.2
    trace['servo_peak_limit_nm']=np.ones((1,6))*1.913
    for key in ('servo_shaft_abs_power_w','servo_shaft_drive_power_w',
                'servo_shaft_brake_power_w','spring_potential_energy_j',
                'shin_spring_energy_j'):
        trace[key]=np.zeros((10,1))
    trace['servo_shaft_abs_power_w'][:5]=999
    trace['servo_shaft_drive_power_w'][:5]=999
    trace['servo_shaft_abs_power_w'][5:]=2
    trace['servo_shaft_drive_power_w'][5:]=1
    report=evaluate_trace_arrays(trace,np.array([[0,0,-9.81]]),
                                 settle_frames=5)[0]
    p=report['physical_evidence']
    assert p['step_integrated_abs_shaft_work_j']==pytest.approx(5*.02*2)
    assert p['step_integrated_drive_shaft_work_j']==pytest.approx(5*.02*1)
    assert report['positive_shaft_work_per_progress_m_j_m']==pytest.approx(.1/.015)
    # Same measured power but movement against the command: no positive
    # metres denominator is available for a physical efficiency verdict.
    trace['position_xy'][:,:,0]=-np.arange(10)[:,None]*.003
    trace['last_position_xy']=np.array([[-.030,0.]])
    opposite=evaluate_trace_arrays(trace,np.array([[0,0,-9.81]]),
                                   settle_frames=5)[0]
    assert opposite['positive_shaft_work_per_progress_m_j_m'] is None
