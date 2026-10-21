"""GPU-free measurement semantics shared by curriculum and examiner.

No reward weights, actor inputs or checkpoint formats are involved.
"""
import numpy as np
import pytest

from emu.scoring import terrain_edge_trials, graded_step_score, ExactStepResult


def _events(done):
    done=np.asarray(done,dtype=np.float32)
    return {
        "terrain_edge_mask":np.ones_like(done),
        "terrain_edge_done":done,
        "terrain_edge_crossing":np.zeros_like(done),
        "terrain_edge_contact":np.zeros_like(done),
        "terrain_edge_fell":np.zeros_like(done),
        "terrain_height_birth_m":np.full_like(done,.020),
    }


def test_unfinished_episodes_are_unknown_not_failures():
    row=terrain_edge_trials(_events(np.zeros((32,3))))
    assert row["terrain_edge_exposure"] == 1.
    assert row["terrain_edge_done_count"] == 0.
    assert row["terrain_edge_crossing"] == -1.
    assert row["terrain_edge_contact"] == -1.
    assert row["terrain_edge_fall_rate"] == -1.
    assert row["terrain_height_birth_m"] == pytest.approx(.020)
    assert terrain_edge_trials({}) == {}


def test_completed_edge_trials_have_correct_denominators_and_counts():
    m=_events([[0,0,0],[1,0,1],[0,1,0]])
    m["terrain_edge_crossing"][1,0]=1.
    m["terrain_edge_contact"][1,0]=1.
    m["terrain_edge_contact"][1,2]=1.
    m["terrain_edge_fell"][2,1]=1.
    row=terrain_edge_trials(m)
    assert row["terrain_edge_done_count"] == 3.
    assert row["terrain_edge_completed_fraction"] == pytest.approx(3/9)
    assert row["terrain_edge_crossing"] == pytest.approx(1/3)
    assert row["terrain_edge_contact"] == pytest.approx(2/3)
    assert row["terrain_edge_fall_rate"] == pytest.approx(1/3)


def test_invalid_training_event_shapes_cannot_be_reported_as_success():
    m=_events([[0,1],[1,0]])
    m["terrain_edge_crossing"]=np.ones((4,))
    with pytest.raises(ValueError,match="terrain_edge_crossing"):
        terrain_edge_trials(m)


def _exam(heights=(5,10,20)):
    d={"physical_step_exact_height":True}
    for mm in heights:
        d[f"course_{mm}mm_attempt_count"]=32.
        d[f"course_{mm}mm_contact_rate"]=.5
        d[f"course_{mm}mm_crossing_rate"]=.25 if mm==20 else 0.
    return d


def test_exact_step_grade_measures_completed_trails_not_training_rewards():
    heights=(5,10,20)
    weights=(.2,.2,.6)
    evidence=_exam(heights)
    evidence["reward_mean"]=1000000.
    score,trials=graded_step_score(evidence,heights,weights)
    assert score == pytest.approx(.15)
    assert all(isinstance(t,ExactStepResult) for t in trials)
    assert [t.attempts for t in trials] == [32,32,32]
    evidence["reward_mean"]=-1000000.
    assert graded_step_score(evidence,heights,weights)[0] == score


@pytest.mark.parametrize("change", [
    {"physical_step_exact_height":False},
    {"course_20mm_attempt_count":0.},
    {"course_20mm_attempt_count":7.},
    {"course_20mm_attempt_count":4.5},
    {"course_20mm_contact_rate":.01},
    {"course_20mm_crossing_rate":-1.},
    {"course_20mm_crossing_rate":float("nan")},
])
def test_exam_never_certifies_invalid_or_sparse_physical_evidence(change):
    e=_exam()
    e.update(change)
    with pytest.raises((ValueError,TypeError)):
        graded_step_score(e,(5,10,20),(.2,.2,.6))



def test_success_cannot_appear_before_trial_completes():
    m=_events([[0,0],[0,1]])
    m["terrain_edge_crossing"][0,0]=1.
    with pytest.raises(ValueError,match="without completed trial"):
        terrain_edge_trials(m)



def test_physical_peak_is_per_world_per_frame_not_a_misleading_mean():
    from emu.results import physical_peak_metric
    data={"spring_potential_energy_j":np.full((32,4096),.8)}
    data["spring_potential_energy_j"][11,1337]=2500000.
    peak=physical_peak_metric(data,"spring_potential_energy_j")
    assert peak==2500000.
    assert np.mean(data["spring_potential_energy_j"]) < 20.
    assert physical_peak_metric({},"spring_potential_energy_j")==-1.
    data["spring_potential_energy_j"][11,1337]=float("nan")
    assert np.isnan(physical_peak_metric(data,"spring_potential_energy_j"))
