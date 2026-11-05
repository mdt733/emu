"""Minimal physical-course assertions for the v5051 independent examiner."""
import numpy as np

from emu.prism_exam import score_prism_crossings, score_post_lip_missions


def _course():
    n = 72
    shape = (n, 1)
    root = np.zeros((n, 1, 2), dtype=np.int32)
    root[6:, 0, 0] = 1
    xy = np.zeros((n, 1, 2), dtype=float)
    xy[:6, 0, 0] = .70
    xy[6:, 0, 0] = .78 + .011 * np.arange(n - 6)
    xy[:, 0, 1] = .30
    feet = np.zeros((n, 1, 2, 3), dtype=float)
    feet[:, 0, 0, :] = (.82, .25, .038)
    feet[:, 0, 1, :] = (.70, .35, .018)
    feet[20:, 0, 1, :] = (.90, .35, .038)
    tops = np.zeros((n, 1, 4, 4), dtype=float)
    tops[:, 0, 1, :] = .020  # actual destination collision top
    pairs = np.zeros((n, 1, 8, 2), dtype=bool)
    vertical = np.zeros((n, 1, 8, 2), dtype=float)
    pairs[6:, 0, 2, 0] = True  # right foot on physical slot 1 triangle 0
    vertical[6:, 0, 2, 0] = 1.
    pairs[20:, 0, 2, 1] = True # trailing left foot recovers
    vertical[20:, 0, 2, 1] = 1.
    return {
        "carpet_root_cell": root,
        "position_xy": xy,
        "heading_rad": np.zeros(shape, dtype=float),
        "foot_xyz": feet,
        "carpet_prism_corner_z": tops,
        "carpet_raw_contact_pairs": pairs,
        "carpet_contact_vertical": vertical,
        "initial_foot_radii_m": np.array([.018, .018]),
        "done": np.zeros(shape, dtype=bool),
        "fell": np.zeros(shape, dtype=bool),
    }


def _score(frames):
    crossings = score_prism_crossings(frames)
    assert crossings["up"] == 1
    assert crossings["down"] == 0
    assert crossings["worlds"][0]["independently_supported_lips"][0]["leading_foot"] == 0
    return score_post_lip_missions(frames, crossings)


def test_full_course_needs_both_foot_contacts_and_50cm_progress():
    result = _score(_course())
    assert result["supported_events"] == 1
    assert result["completed"] == 1
    event = result["worlds"][0]["missions"][0]
    assert event["trailing_support_frame"] >= 22
    assert event["controlled_completion_frame"] > event["trailing_support_frame"]
    assert event["outcome"] == "complete_controlled_50cm"


def test_single_foot_tap_does_not_complete_climb():
    frames = _course()
    frames["carpet_raw_contact_pairs"][:, :, :, 1] = False
    result = _score(frames)
    assert result["completed"] == 0
    assert result["worlds"][0]["missions"][0]["outcome"] == (
        "upper_support_without_trailing_recovery")


def test_heading_drift_rejects_distance_only_success():
    frames = _course()
    frames["heading_rad"][24:, :] = np.deg2rad(55.)
    result = _score(frames)
    assert result["completed"] == 0
    assert result["worlds"][0]["missions"][0]["outcome"] == (
        "distance_reached_but_path_control_failed")


def test_a_fall_after_placing_leading_foot_is_not_a_completed_climb():
    frames = _course()
    frames["done"][34, 0] = True
    frames["fell"][34, 0] = True
    result = _score(frames)
    assert result["completed"] == 0
    assert result["worlds"][0]["missions"][0]["fell_after_upper_support"]


def test_no_crossing_is_not_a_failed_attempt():
    frames = _course()
    frames["carpet_raw_contact_pairs"][:] = False
    crossing = score_prism_crossings(frames)
    result = score_post_lip_missions(frames, crossing)
    assert result["supported_events"] == result["completed"] == 0
