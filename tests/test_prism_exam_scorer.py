"""Prism scorer synthetic trajectory regressions (no MJX reward inputs)."""
import copy

import numpy as np
import pytest

from scripts.exam_prism_course import score_prism_crossings


def trace(*, rise=.02, axis=0, frames=12):
    cells=np.zeros((frames,1,2),dtype=np.int32)
    cells[2:,0,axis]=1
    z=np.zeros((frames,1,9,4),dtype=np.float64)
    z[:,:,axis==1 and 3 or 1,:]=rise
    pair=np.zeros((frames,1,18,2),dtype=bool)
    normal=np.zeros((frames,1,18,2),dtype=np.float64)
    foot=np.zeros((frames,1,2,3),dtype=np.float64)
    foot[:,:,0,:]=[.25,.25,.020]
    foot[:,:,1,:]=[.30,.30,.020]
    index=3 if axis==1 else 1
    for t in range(2,frames):
        foot[t,0,0,:]=[.25+(axis==0)*.50,.25+(axis==1)*.50,rise+.020]
        pair[t,0,2*index,0]=True
        normal[t,0,2*index,0]=.96
    return dict(carpet_root_cell=cells,carpet_prism_corner_z=z,
                carpet_raw_contact_pairs=pair,carpet_contact_vertical=normal,
                foot_xyz=foot,initial_foot_radii_m=np.array([.020,.020]),
                fell=np.zeros((frames,1),dtype=bool),
                done=np.zeros((frames,1),dtype=bool))


@pytest.mark.parametrize("rise,axis,up,down",[
    (.02,0,1,0),(-.035,0,0,1),(.003,1,1,0),(-.02,1,0,1)])
def test_real_top_supported_crossing_in_both_axes(rise,axis,up,down):
    t=trace(rise=rise,axis=axis)
    result=score_prism_crossings(t)
    assert (result["up"],result["down"],result["total"])==(up,down,1)
    assert result["worlds"][0]["independently_supported_lips"][0]["frame"]==6


def test_smooth_grade_is_not_a_step():
    t=trace(rise=0.)
    result=score_prism_crossings(t)
    assert result["total"]==0
    assert result["worlds"][0]["smooth_grade_transitions"]==1


def test_unphysical_side_contact_cannot_earn_lip():
    t=trace()
    t["carpet_contact_vertical"][:]=.04
    assert score_prism_crossings(t)["total"]==0


def test_no_contact_cannot_earn_lip():
    t=trace()
    t["carpet_raw_contact_pairs"][:]=False
    assert score_prism_crossings(t)["total"]==0


def test_false_foot_support_height_cannot_earn_lip():
    t=trace()
    t["foot_xyz"][:,:,0,2]+=.03
    assert score_prism_crossings(t)["total"]==0


def test_episode_end_censors_any_later_success():
    t=trace()
    t["done"][3,0]=True
    assert score_prism_crossings(t)["total"]==0


def test_fall_breaks_five_frame_support_streak():
    t=trace(frames=8)
    t["fell"][5,0]=True
    assert score_prism_crossings(t)["total"]==0


def test_coming_back_to_visited_cell_does_not_double_count():
    t=trace(frames=20)
    t["carpet_root_cell"][9:11,0,0]=0
    t["carpet_root_cell"][11:,0,0]=1
    result=score_prism_crossings(t)
    assert result["total"]==1


def test_signed_rise_not_slope_over_50cm():
    t=trace(rise=.02)
    # Shared edge is x=.50. Give both sides exactly the same grade
    # along y so the 1.5cm height transition is still a true lip.
    ramp=np.array([0.,0.,.01,.01])
    t["carpet_prism_corner_z"][:,:,0,:]=ramp
    t["carpet_prism_corner_z"][:,:,1,:]=ramp+.02
    result=score_prism_crossings(t)
    assert result["total"]==1
    assert result["worlds"][0]["independently_supported_lips"][0]["signed_height_cm"]==2.


def test_reject_missing_trace_or_invalid_shapes():
    t=trace()
    del t["carpet_prism_corner_z"]
    with pytest.raises(KeyError):
        score_prism_crossings(t)
    t=trace()
    t["carpet_raw_contact_pairs"]=np.zeros((12,1,9,2),bool)
    with pytest.raises(ValueError,match="expected"):
        score_prism_crossings(t)
