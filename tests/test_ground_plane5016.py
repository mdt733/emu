"""v5016 three-range-sensor / IMU estimator physical geometry contracts."""
import math
import numpy as np
import pytest

from emu.ground_plane import estimate_ground_plane

# Robot body coordinates: X forward, Y left, Z up; broad noncollinear triangle.
ORIGINS = np.array([[.035, -.05, 0], [.035, .05, 0], [-.04, 0, 0]], dtype=float)
BEAMS = np.tile([0, 0, -1.0], (3, 1))


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


@pytest.mark.parametrize('degree', [-2, -1, -.5, 0, .5, 1, 2])
def test_inclined_ground_from_three_ranges_and_gravity(degree):
    ranges,gravity=hits_for_world_plane(degree)
    r=estimate_ground_plane(ORIGINS,BEAMS,ranges,gravity)
    assert r.valid, r.reason
    assert r.slope_deg == pytest.approx(abs(degree),abs=1e-7)
    assert r.triangle_area_m2 > 0.001
    if degree:
        assert r.uphill_body is not None
        assert np.sign(r.uphill_body[0]) == np.sign(degree)
    else:
        assert r.uphill_body is None


def test_tilted_robot_on_flat_floor_is_not_misread_as_inclined_ground():
    ranges,gravity=hits_for_world_plane(0,chassis_pitch_degrees=10)
    r=estimate_ground_plane(ORIGINS,BEAMS,ranges,gravity)
    assert r.valid
    assert r.slope_deg == pytest.approx(0,abs=1e-7)


def test_tilted_robot_on_actual_slope_yields_ground_angle_not_chassis_angle():
    ranges,gravity=hits_for_world_plane(1,chassis_pitch_degrees=12)
    r=estimate_ground_plane(ORIGINS,BEAMS,ranges,gravity)
    assert r.valid
    assert r.slope_deg == pytest.approx(1,abs=1e-7)


@pytest.mark.parametrize('range_values', [
    [0.2,0.2,float('nan')], [0.2,-1,0.2],
    [0.2,float('inf'),0.2], [0.2,0.2,1.0],
])
def test_invalid_missing_or_out_of_range_return_never_fakes_flat_ground(range_values):
    r=estimate_ground_plane(ORIGINS,BEAMS,range_values,[0,0,-1])
    assert not r.valid and r.slope_deg is None


def test_collinear_ray_triangle_fails_closed():
    straight=np.array([[-.03,0,0],[0,0,0],[.03,0,0]])
    r=estimate_ground_plane(straight,BEAMS,[.2,.2,.2],[0,0,-1])
    assert not r.valid and r.triangle_area_m2 < 1e-4


def test_one_bad_return_cannot_be_reliably_identified_from_three_points():
    ranges,gravity=hits_for_world_plane(0)
    ranges[0] += .012
    r=estimate_ground_plane(ORIGINS,BEAMS,ranges,gravity)
    assert r.valid  # three hits always define a plane: no fake outlier guarantee
    assert r.slope_deg > 0
    assert 'no outlier redundancy' in r.reason


def test_bad_sensor_geometry_shape_raises_without_silent_truncation():
    with pytest.raises(ValueError):
        estimate_ground_plane(ORIGINS[:2],BEAMS[:2],[.2,.2],[0,0,-1])


def test_zero_gravity_disables_measurement_without_nan():
    r=estimate_ground_plane(ORIGINS,BEAMS,[.2,.2,.2],[0,0,0])
    assert not r.valid
    assert r.ground_normal_body is None
