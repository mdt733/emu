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



def test_explicit_260m_parent_validation_uses_saved_terrain_sensor_contract(tmp_path, monkeypatch):
    """Regression: load_checkpoint defaults to legacy Pitch15 unless told the 360 contract."""
    from dataclasses import replace
    import emu.checkpoint_io as checkpoint_io
    from emu.contracts import OBS_SCHEMA_TERRAIN
    from emu.training_config import training_sensor_contract

    source = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                                 "scripts/run_plateau_science.py"))
    check_parent = source["read_parent"]
    parent = tmp_path / "pinnacle_cold260" / "checkpoints" / "final.pkl"
    parent.parent.mkdir(parents=True)
    parent.write_bytes(b"synthetic test fixture, not a trained policy")
    cfg = replace(
        TrainingConfig(), observation_schema=OBS_SCHEMA_TERRAIN,
        unified_capability_focus=True, foundation_steps=20_000_000,
        growth_recipe="main-mature", terrain_lesson="coherent-careful",
        force_program="gentle", terrain_step_max_m=.08,
        athletic_rehearsal_frac=.25, temporal_vx_reward_share=.50,
        verified_ascent_reallocation=0., teaching_schedule="balanced",
        motor_guidance_weight=0., envs=2048, seed=761, open_frontier=True)
    cfg.save(parent.parent.parent / "training_config.json")
    expected_contract = training_sensor_contract(cfg)
    checked = []

    def fake_checkpoint(path, *, validate, expected_observation_schema,
                        expected_sensor_contract):
        assert path == parent
        assert validate is True
        assert expected_observation_schema == OBS_SCHEMA_TERRAIN
        assert expected_sensor_contract == expected_contract
        checked.append(True)
        return {"meta": {
            "source_provenance": {"git_commit": source["PARENT_SOURCE"]},
            "step": source["PARENT_STEP"]}}

    monkeypatch.setattr(checkpoint_io, "load_checkpoint", fake_checkpoint)
    # An on-disk synthetic test cannot hash to the immutable real checkpoint.
    # Mock *only* the file hash; keep config, contract and metadata checks live.
    monkeypatch.setitem(check_parent.__globals__, "sha256_file",
                        lambda path: source["EXPECTED_PARENT_SHA256"])
    accepted, digest = check_parent(parent)
    assert accepted == cfg
    assert digest == source["EXPECTED_PARENT_SHA256"]
    assert checked == [True]


def test_research_export_uses_real_actor_validator_and_marks_velocity_i_unplayable(tmp_path, capsys, monkeypatch):
    import json
    import sys
    from copy import deepcopy
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    source = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                                 "scripts/run_plateau_science.py"))
    validate = source["validate_research_export"]
    released = json.loads((Path(__file__).resolve().parents[1] /
                           "showcase/champion.emupolicy").read_text(encoding="utf-8"))
    baseline_path = tmp_path / "unchanged.emupolicy"
    baseline_path.write_text(json.dumps(released), encoding="utf-8")
    identity = validate(baseline_path)
    assert identity["schema"] == 3
    assert identity["obs_dim"] == 360
    assert identity["checkpoint_schema"] == 9
    assert identity["step"] == 260_046_848

    # Source is read-only: the 260M champion remains unchanged. The velocity-I
    # research export must validate weights/contract but not claim browser parity.
    research = deepcopy(released)
    research["control"]["velocity_i"]["multiplier"] = 1.5
    research_path = tmp_path / "velocity-i.emupolicy"
    research_path.write_text(json.dumps(research), encoding="utf-8")
    special = validate(research_path)
    assert special["obs_dim"] == 360
    assert special["training_source_commit"] == released["source"]["git_commit"]
    assert "RESEARCH-ONLY EXPORT" in capsys.readouterr().out
    assert json.loads(baseline_path.read_text()) == released


def test_campaign_manifest_json_roundtrip_does_not_block_resume(tmp_path):
    """Same lineage, factors and resolved config must survive tuple -> JSON list."""
    import json
    source = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                                 "scripts/run_plateau_science.py"))
    write = source["ensure_manifest"]
    root = tmp_path / "campaign"
    original = {
        "source_sha": "a" * 40,
        "parent_sha256": "b" * 64,
        "arms": [
            {"id": 0, "factors": (), "overrides": {}},
            {"id": 1, "factors": ("step_promotion_1mm",), "overrides": {"mm": 1.0}},
        ],
    }
    write(root, original)
    data = json.loads((root / "campaign_manifest.json").read_text())
    assert data["arms"][0]["factors"] == []
    assert data["arms"][1]["factors"] == ["step_promotion_1mm"]
    assert (root / "campaign_manifest.json").is_file()
    write(root, original)  # resume must pass, without rewriting files
    altered = {**original, "parent_sha256": "c" * 64}
    with pytest.raises(RuntimeError, match="lineage differs"):
        write(root, altered)
