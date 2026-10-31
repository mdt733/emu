"""Contract tests for the post-Pinnacle 260M research screen; no CUDA or new PPO runs."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import runpy

import numpy as np
import pytest

from emu.curriculum import RegimeGovernor
from emu.rewards import _physical_qualities
from emu.training_config import TrainingConfig


def test_24_arm_design_has_balanced_nonconfounded_main_effects():
    module = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                                  "scripts/run_plateau_science.py"))
    arms = module["arms"]()
    assert len(arms) == 24
    assert arms[0].name == "control_seed761"
    assert arms[0].changes == {}
    assert arms[16].changes == {"seed": 762}
    names = module["FACTOR_NAMES"]
    for name in names:
        assert sum(name in arm.factors for arm in arms[:16]) == 8
        for other in names:
            if name != other:
                assert sum(name in arm.factors and other in arm.factors
                           for arm in arms[:16]) == 4
    assert len({arm.directory for arm in arms}) == 24


def test_old_training_configs_receive_off_by_default_science_knobs(tmp_path):
    old = asdict(TrainingConfig())
    keys = ("science_step_promotion_mm", "science_step_retreat_mm",
            "science_step_min_attempts", "science_step_crouch_tolerance_m",
            "science_reset_step_evidence_on_fork")
    for key in keys:
        old.pop(key)
    path = tmp_path / "training_config.json"
    import json
    path.write_text(json.dumps(old))
    loaded = TrainingConfig.load(path)
    assert loaded.science_step_promotion_mm == 0.
    assert loaded.science_step_retreat_mm == 0.
    assert loaded.science_step_min_attempts == 4
    assert loaded.science_step_crouch_tolerance_m == 0.
    assert not loaded.science_reset_step_evidence_on_fork


def evidence(near=0., easy=0., near_successes=None, easy_successes=None):
    row = {}
    for d in ("up", "down"):
        row[f"terrain_near_attempts_{d}"] = near
        row[f"terrain_near_successes_{d}"] = (near if near_successes is None
                                                else near_successes)
        row[f"terrain_easy_attempts_{d}"] = easy
        row[f"terrain_easy_successes_{d}"] = (easy if easy_successes is None
                                                else easy_successes)
    return row


def governor(**kw):
    gov = RegimeGovernor(unified_capability_focus=True, **kw)
    gov.params.terrain_exposure = .15
    gov.params.terrain_height_m = .030
    return gov


def advance(gov, row):
    return gov.advance_terrain(row, step=260_100_000, max_height_m=.08,
                               foundation_steps=20_000_000, unified=True)


def test_fixed_one_mm_step_growth_and_evidence_counts():
    old = governor()
    assert advance(old, evidence(near=4.)) == "terrain_height_expand_real_cross"
    assert old.params.terrain_height_m == pytest.approx(.0354)
    fixed = governor(science_step_promotion_mm=1., science_step_min_attempts=12)
    assert advance(fixed, evidence(near=4.)) == "terrain_collect_real_step"
    assert fixed.params.terrain_height_m == pytest.approx(.03)
    assert advance(fixed, evidence(near=8.)) == "terrain_height_expand_real_cross"
    assert fixed.params.terrain_height_m == pytest.approx(.031)


def test_fixed_one_mm_step_retreat_versus_original_multiplicative():
    old = governor()
    assert advance(old, evidence(easy=4., easy_successes=0.)) == "terrain_retreat_easy_failure"
    assert old.params.terrain_height_m == pytest.approx(.0246)
    fixed = governor(science_step_retreat_mm=1.)
    assert advance(fixed, evidence(easy=4., easy_successes=0.)) == "terrain_retreat_easy_failure"
    assert fixed.params.terrain_height_m == pytest.approx(.029)


def test_crouch_allowance_changes_height_reward_only_when_supplied():
    z = np.array([0.], dtype=np.float32)
    o = np.array([1.], dtype=np.float32)
    m = dict(gravity_z=o, height=z+.15, target_height=z+.20,
             cmd_pitch=z, height_error=z-.05, attitude_quality=o,
             foot_mid_cone=o, bad_contact=z, support=o, stance_slip=z,
             swing_clearance=z+.03, target_step_height=z+.03,
             organic_step_shape=o, action_delta_norm=z,
             joint_limit_pressure=z, scissor_pressure=z)
    baseline = _physical_qualities(m, "v5015", np)
    zero = _physical_qualities(m, "v5015", np, crouch_allowance=z)
    benefit = _physical_qualities(m, "v5015", np,
                                  crouch_allowance=np.array([.04], np.float32))
    np.testing.assert_array_equal(baseline[0], zero[0])
    assert float(benefit[0][0]) > float(baseline[0][0])
    for a, b in zip(benefit[1:], baseline[1:]):
        np.testing.assert_array_equal(a, b)
