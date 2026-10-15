"""First v5016 anti-cheating contracts, independent of simulation resets."""
import math
import numpy as np
import pytest

from emu.directional import (
    HeadingState, CourseLimits, heading_hold_step,
    examine_course, incline_from_tilted_gravity, wrap_radians,
)
from emu.observation_versions import declared_actor_contract


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


def test_idle_foot_tapping_cannot_qualify_as_walking():
    cmd = np.tile([.1, 0, 0], (350, 1))
    result = examine_course(np.zeros((351, 2)), np.zeros(351), cmd,
                            uphill_world_xy=(1, 0), slope_deg=1,
                            completed_strides=20)
    assert not result['course_pass'] and not result['walking_pass']
    assert result['progress_ratio'] == pytest.approx(0)


@pytest.mark.parametrize('vx,vy,start', [
    (-.12, 0, math.pi),  # reverse body motion can still climb uphill
    (0, .10, -math.pi/2),  # sideways body motion can also climb uphill
    (.10, 0, math.pi/2),  # a cross-slope walk is not penalised for obeying task
])
def test_reverse_lateral_and_cross_slope_follow_requested_course(vx, vy, start):
    cmd = np.tile([vx, vy, 0], (350, 1))
    pos, heading = ideal(cmd, initial_yaw=start)
    result = examine_course(pos, heading, cmd, uphill_world_xy=(1, 0),
                            slope_deg=1, completed_strides=5)
    assert result['course_pass'] and result['walking_pass']
    if start == math.pi/2:
        assert abs(result['uphill_progress_m']) < 1e-8
    else:
        assert result['uphill_progress_m'] > 0


def test_requested_turn_and_turn_stop_are_not_treated_as_unauthorised_yaw():
    cmd = np.array([[.14, 0, .25]]*180 + [[.14, 0, 0]]*170, dtype=float)
    pos, heading = ideal(cmd)
    r = examine_course(pos, heading, cmd, completed_strides=8)
    assert r['course_pass'] and r['walking_pass']
    assert r['yaw_rate_mae_rad_s'] < 1e-11


def test_zero_yaw_rate_with_oscillation_cannot_hide_in_signed_mean():
    cmd = np.tile([0, 0, 0], (350, 1))
    h = np.zeros(351)
    for i in range(350):
        h[i+1] = h[i] + (.5 if i%2==0 else -.5)*.02
    r = examine_course(np.zeros((351, 2)), h, cmd)
    assert abs(r['yaw_rate_signed_mean_error']) < 1e-9
    assert r['yaw_abs_error_1s'] > .49
    assert r['yaw_abs_error_3s'] > .49
    assert not r['course_pass']


def test_ballistic_spawn_motion_excluded_from_settled_course():
    cmd = np.tile([.1, 0, 0], (350, 1))
    pos = np.zeros((351, 2))
    pos[1:26, 0] = np.arange(1, 26)*.02
    pos[26:, 0] = .5
    r = examine_course(pos, np.zeros(351), cmd, settle_frames=25,
                       uphill_world_xy=(1, 0), slope_deg=1,
                       completed_strides=4)
    assert r['signed_course_progress_m'] == pytest.approx(0)
    assert not r['course_pass']


def test_fallen_or_reset_contaminated_episode_never_passes():
    cmd = np.tile([.1, 0, 0], (350, 1))
    pos, heading = ideal(cmd)
    assert not examine_course(pos, heading, cmd, fell=True, completed_strides=9)['walking_pass']
    flags = np.zeros(350, dtype=bool)
    flags[200] = True
    with pytest.raises(ValueError, match='reset or teleport'):
        examine_course(pos, heading, cmd, reset_flags=flags)


def test_stride_unknown_is_reported_instead_of_claimed():
    cmd = np.tile([.1, 0, 0], (350, 1))
    pos, heading = ideal(cmd)
    r = examine_course(pos, heading, cmd)
    assert r['course_pass'] and not r['gait_verified'] and not r['walking_pass']


def test_tilted_gravity_has_world_fixed_uphill_vector_and_flat_ground_is_safe():
    deg=1.0
    a=math.radians(deg)
    slope, uphill = incline_from_tilted_gravity([-9.81*math.sin(a),0,-9.81*math.cos(a)])
    assert slope == pytest.approx(deg)
    np.testing.assert_allclose(uphill, [1,0], atol=1e-12)
    flat, direction = incline_from_tilted_gravity([0,0,-9.81])
    assert flat == 0 and direction is None
    with pytest.raises(ValueError):
        incline_from_tilted_gravity([0,0,0])
    with pytest.raises(ValueError, match='uphill'):
        examine_course(np.zeros((2,2)), np.zeros(2), np.zeros((1,3)), slope_deg=1)


def test_heading_hold_rate_turns_stops_and_catches_drift():
    state=HeadingState()
    state, output, error = heading_hold_step(state, 0, 0)
    assert output == error == 0
    state, output, error = heading_hold_step(state, math.radians(-5), 0)
    assert output > 0 and error > 0
    # A deliberate turn moves the reference instead of locking initial heading.
    previous_target = state.target_rad
    state, output, _ = heading_hold_step(state, previous_target, .2, kp=0)
    assert state.target_rad == pytest.approx(previous_target+.004)
    assert output == pytest.approx(.2)
    state2, output, _ = heading_hold_step(state, state.target_rad, 0)
    assert state2.target_rad == pytest.approx(state.target_rad)
    assert output == pytest.approx(0)


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


def test_versioned_actor_contract_expands_without_reinterpreting_old_weights():
    old=declared_actor_contract('v5015')
    new=declared_actor_contract('v5016-candidate')
    assert old['observations']==220 and old['actions']==6
    assert new['observations']==244 and new['actions']==6
    assert not new['runtime_ready']
    assert new['base_fields'][:len(old['base_fields'])] == old['base_fields']
    assert len(set(new['base_fields'])) == 61
    with pytest.raises(ValueError):
        declared_actor_contract('anything')
