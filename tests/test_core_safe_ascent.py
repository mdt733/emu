"""Research-only, CPU-free checks for the independent KS150 survival screen."""
from __future__ import annotations

from scripts.score_h40_ks150_heldout import (
    ACTORS, BANDS, EXPECTED_HEIGHTS, SEED, safe_ascent_blockers,
)


def _scores(*, safe_survivors: int, frontier_survivors: int) -> dict:
    return {
        "step_up_safe_4_40": {
            "survived": safe_survivors, "n": 500,
            "survival_frac": safe_survivors / 500,
        },
        "step_up_frontier_44_56": {
            "survived": frontier_survivors, "n": 200,
            "survival_frac": frontier_survivors / 200,
        },
    }


def test_actual_v1_hazard_fails_v2_hardware_safety_screen():
    # Real seed1701 matrix from frozen qualification evidence:
    # 120M 490/500 easy survival; 196/200 frontier;
    # 150M 490/500 easy survival; 107/200 frontier;
    # 160M 476/500 easy survival; 89/200 frontier.
    parent = _scores(safe_survivors=490, frontier_survivors=196)
    good_upto_4cm = _scores(safe_survivors=490, frontier_survivors=107)
    too_risky = _scores(safe_survivors=476, frontier_survivors=89)
    assert safe_ascent_blockers(parent, good_upto_4cm) == [
        "uphill 44–56mm survival below 90%"
    ]
    assert safe_ascent_blockers(parent, too_risky) == [
        "uphill 4–40mm survival regression",
        "uphill 44–56mm survival below 90%",
    ]


def test_safe_ascent_requires_separate_real_survival_not_just_crossings():
    reference = _scores(safe_survivors=490, frontier_survivors=196)
    safe = _scores(safe_survivors=480, frontier_survivors=180)
    assert safe_ascent_blockers(reference, safe) == []
    failed = _scores(safe_survivors=420, frontier_survivors=180)
    failed["step_up_safe_4_40"]["traversals"] = 500
    assert "uphill 4–40mm survival regression" in safe_ascent_blockers(
        reference, failed)


def test_exam_grid_and_actors_are_fixed_before_launch():
    assert SEED == 2701
    assert set(ACTORS) == {120, 150, 160}
    assert len(set(ACTORS.values())) == 3
    assert EXPECTED_HEIGHTS == tuple(float(i) for i in range(4, 81, 4))
    assert BANDS == (
        ("safe_4_40", 4, 40),
        ("frontier_44_56", 44, 56),
        ("stretch_44_80", 44, 80),
    )
