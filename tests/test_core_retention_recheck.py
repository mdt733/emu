"""Fast read-only retention study checks: no JAX, trainer or GPU required."""
from __future__ import annotations

from pathlib import Path
import ast
import pytest

from scripts.recheck_h40_retention_multiseed import (
    EXPECTED_FINGERPRINT, MODES, POLICIES, SEEDS,
    _cohort, _physical_metrics, fraction,
)


def _world(kind, height, success, *, lane=False, feet=False, survive=True):
    return {
        "kind": kind, "exact_step_m": str(height),
        "traversed_5frame": str(bool(lane)),
        "traversed_upright_5frame": str(bool(success)),
        "both_feet_top_five": str(bool(feet)),
        "survived": str(bool(survive)),
        "max_abs_lateral_offset_m": "0.12",
    }


def test_sampling_contract_and_physical_metrics():
    assert SEEDS == (1701, 1711, 1729)
    assert MODES == ("stairs", "flat")
    assert POLICIES == ("original-140", "original-160",
                        "150", "165", "175", "180")
    assert len(EXPECTED_FINGERPRINT) == 64
    # Exactly the real examiner's 20 heights x 50 x two directions.
    stairs = (
        [_world("step_up", .048, True, feet=True, lane=False)] * 500
        + [_world("step_up", .072, False, lane=False)] * 500
        + [_world("step_down", -.048, True, feet=True)] * 500
        + [_world("step_down", -.072, False)] * 500
    )
    flat = ([_world("flat_control", 0.0, True, lane=True)] * 750
            + [_world("flat_control", 0.0, False, lane=False)] * 1250)
    r = _physical_metrics(stairs, flat)
    assert r["flat_control"] == 37.5
    assert r["uphill_44_60"] == 100.0
    assert r["uphill_64_80"] == 0.0
    assert r["downhill_44_80"] == 50.0
    assert r["medium_both_feet"] == 100.0
    assert r["medium_strict_heading_lane"] == 0.0
    assert r["flat_survival"] == 100.0
    assert r["flat_lateral_peak_cm"] == pytest.approx(12.0)
    assert r["downhill_both_feet"] == 50.0
    assert len(_cohort(stairs, "down", .044, .080)) == 1000


def test_rejects_incomplete_physical_cohorts():
    stairs = [_world("step_up", .048, True)]
    flat = [_world("flat_control", 0.0, True)] * 2000
    with pytest.raises(ValueError, match="Unexpected cohort sizes"):
        _physical_metrics(stairs, flat)
    with pytest.raises(ValueError, match="Invalid literal"):
        fraction([{"traversed_5frame": "maybe"}], "traversed_5frame")


def test_study_source_parses_without_importing_learner():
    script = Path(__file__).resolve().parents[1] / \
             "scripts/recheck_h40_retention_multiseed.py"
    ast.parse(script.read_text(encoding="utf-8"))
