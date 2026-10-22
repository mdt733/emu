"""Continuous ground changes and discontinuous height jumps stay distinct."""
import math
import numpy as np
import pytest
from emu.terrain_profiles import (
    course_surfaces, profile_transition_kind, surface_box_geometry,
)


def test_hillcrest_has_continuity_but_a_change_in_grade():
    s, seams = course_surfaces(-.6, 0., [.4,.4,.4,.4],
                              [0, 12, 0, -12], [0,0,0])
    assert all(not seam.discontinuous for seam in seams)
    assert [profile_transition_kind(x) for x in seams] == [
        'slope_steepens', 'slope_flattens', 'slope_flattens']
    for left, right in zip(s, s[1:]):
        assert left.x1_m == pytest.approx(right.x0_m)
        assert left.z1_m == pytest.approx(right.z0_m)
    assert s[-1].z1_m == pytest.approx(0., abs=1e-10)


def test_step_height_is_not_disguised_as_a_slope():
    surfaces, seams = course_surfaces(-.5,0., [.5,.5,.5],
                                      [0.,0.,0.], [.05,-.07])
    assert [x.discontinuous for x in seams] == [True, True]
    assert [profile_transition_kind(x) for x in seams] == ['step_up','step_down']
    assert surfaces[1].z0_m - surfaces[0].z1_m == pytest.approx(.05)
    assert surfaces[2].z0_m - surfaces[1].z1_m == pytest.approx(-.07)


def test_compound_terrain_crests_and_steps_can_be_independent():
    surfaces, seams = course_surfaces(-.5,0., [.4,.4,.4],
                                      [8,-8,0],[0,.04])
    assert [profile_transition_kind(x) for x in seams] == [
        'slope_flattens','step_up']
    assert seams[0].grade_change_rad == pytest.approx(math.radians(-16))
    assert seams[1].step_height_m == pytest.approx(.04)


def test_solid_box_upper_face_matches_slope_ends():
    surfaces, _ = course_surfaces(-.6,0.,[.4,.5,.4],[0,12,-8],[0,0])
    for s in surfaces:
        center, size, quat = surface_box_geometry(s)
        q = np.array(quat)
        assert np.linalg.norm(q) == pytest.approx(1.)
        half_length = size[0]
        a = s.grade_rad
        normal = np.array([-math.sin(a),0,math.cos(a)])
        tangent = np.array([math.cos(a),0,math.sin(a)])
        mid_top = np.asarray(center)+normal*size[2]
        for sign, x in [(-1,s.x0_m),(1,s.x1_m)]:
            corner = mid_top+sign*half_length*tangent
            assert corner[0] == pytest.approx(x, abs=1e-9)
            assert corner[2] == pytest.approx(s.height_at(x),abs=1e-9)


@pytest.mark.parametrize("lengths,angles,jumps", [
    ([0,1],[0,0],[0]), ([1,1],[90,0],[0]),
    ([1,1],[0,0],[0,0]), ([1,1],[0,0],[float('nan')]),
])
def test_invalid_profile_rejected(lengths,angles,jumps):
    with pytest.raises(ValueError):
        course_surfaces(0,0,lengths,angles,jumps)
