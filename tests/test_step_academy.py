"""Step Academy Stage A contracts: no privileged sensors, single-lip-only honesty."""
from __future__ import annotations

import json
import runpy
from dataclasses import asdict
from pathlib import Path

import pytest

from emu.training_config import TrainingConfig

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_step_academy.py"


def stage():
    return runpy.run_path(str(SCRIPT))


def test_academy_arms_and_independent_comparisons(tmp_path):
    m = stage()
    arms = m["arms"]()
    assert len(arms) == 8
    assert [a.index for a in arms] == list(range(8))
    assert len(set(a.directory for a in arms)) == 8
    a0, a1, a2, a3, a4, a5, a6, a7 = arms
    assert a0.changes == {}
    assert a1.changes["academy_step_focus_share"] == .85
    assert a1.changes["academy_edge_yaw_max_deg"] == 3.
    assert a2.changes["academy_episode_frames"] == 300
    assert a3.changes["academy_step_height_cap_m"] == .020
    assert a4.changes["academy_step_focus_share"] == 1.
    assert a5.changes["verified_ascent_reallocation"] == 3.
    assert a6.changes["academy_step_speed_span"] == .11
    assert a7.changes["academy_edge_yaw_max_deg"] == 0.
    assert m["PARENT_STEP"] == 280_035_328
    assert "multi-lip" in m["__doc__"].lower()


def test_config_defaults_legacy_safe_and_roundtrip(tmp_path):
    cfg=TrainingConfig()
    assert cfg.academy_step_focus_share == 0.
    assert cfg.academy_step_exposure == 0.
    assert cfg.academy_edge_yaw_max_deg == -1.
    assert cfg.academy_episode_frames == 0
    assert cfg.academy_step_height_cap_m == 0.
    assert cfg.academy_step_speed_span == 0.
    p=tmp_path / "config.json"
    cfg.save(p)
    assert TrainingConfig.load(p)==cfg
    older=json.loads(p.read_text())
    for key in ("academy_step_focus_share","academy_step_exposure",
                "academy_edge_yaw_max_deg","academy_step_height_cap_m",
                "academy_episode_frames","academy_step_speed_span"):
        older.pop(key)
    p.write_text(json.dumps(older))
    assert TrainingConfig.load(p)==cfg


def test_manifest_resumes_identical_json_and_refuses_changed_parent(tmp_path):
    m=stage()
    parent=tmp_path / "old" / "checkpoints" / "final.pkl"
    root=tmp_path / "stage"
    source={"commit":"c"*40,"branch":m["BRANCH"]}
    cfg=TrainingConfig()
    expected=m["manifest"](source,parent,"a"*64,cfg,root,290_000_000)
    m["ensure_manifest"](root,expected)
    original=(root/"campaign_manifest.json").read_bytes()
    m["ensure_manifest"](root,expected)
    assert original==(root/"campaign_manifest.json").read_bytes()
    assert expected["arms"][1]["resolved_config"]["academy_edge_yaw_max_deg"]==3.
    altered={**expected,"parent_sha256":"b"*64}
    with pytest.raises(RuntimeError, match="identity/lineage"):
        m["ensure_manifest"](root,altered)
    assert original==(root/"campaign_manifest.json").read_bytes()


def test_no_new_privileged_actor_inputs_or_robot_hardware_modification():
    from emu.contracts import TERRAIN_OBS_DIM, SENSOR_OBS_DIM
    assert TERRAIN_OBS_DIM == 360
    assert SENSOR_OBS_DIM == 232
    script=SCRIPT.read_text()
    env=(ROOT / "emu/environment.py").read_text()
    train=(ROOT / "emu/training.py").read_text()
    assert "academy_edge_yaw_max_deg" in env
    assert "academy_step_focus_share" in train
    assert "academy_step_exposure" in env
    assert "self.academy_step_height_cap_m" in env
    assert "self.academy_step_speed_span" in env
    assert "resolve_target_checkpoint" in script
    assert "research_parent_sha256" in script
    assert "validate_browser_bundle" not in script


def test_stage_a_uses_immutable_same_parent_with_unique_run_dirs(tmp_path):
    m=stage()
    parent_config=TrainingConfig(
        run_dir=str(tmp_path / "parent"),envs=2048,steps=m["PARENT_STEP"],
        observation_schema="terrain_360",unified_capability_focus=True,
        terrain_lesson="coherent-careful"
    )
    children=[m["config_for"](parent_config,a,tmp_path/"academy",m["TARGET_STEP"])
              for a in m["arms"]()]
    assert len(set(x.run_dir for x in children))==8
    assert all(x.steps==290_000_000 and x.envs==2048 for x in children)
    assert all(x.observation_schema=="terrain_360" for x in children)
    assert all(x.unified_capability_focus for x in children)
    assert all(x.terrain_lesson=="coherent-careful" for x in children)
    assert all(x.science_reset_step_evidence_on_fork for x in children)
