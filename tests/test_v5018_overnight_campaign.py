"""No-GPU regression tests for the branch-only overnight experiment driver."""
from __future__ import annotations

import csv
import json
import tarfile

import pytest

from scripts.v5018_overnight_campaign import (
    archive, extension_eligible, required_target,
    summarize_crossings, summarize_training, validate_config,
)


def _config():
    return dict(envs=1024, seeds=[733, 734],
                foundation_steps=20_000_000, screen_more_steps=2_100_000,
                extend_more_steps=6_000_000, pitch_deg=8.,
                exam_worlds=4, exam_horizon=350, study_revision=1)


def test_steps_are_absolute_ppo_boundaries():
    assert required_target(20_000_000, 1024) == 20_021_248
    assert required_target(20_000_000, 512) % (512 * 32) == 0
    assert required_target(20_000_000 + 2_100_000, 1024) > 20_000_000


def test_config_guards_large_gpu_and_duplicate_seeds():
    validate_config(_config())
    for change in ({"envs": 4096}, {"seeds": [733, 733]},
                   {"foundation_steps": 1024}, {"exam_worlds": 1},
                   {"screen_more_steps": 16_000_000}):
        d = {**_config(), **change}
        with pytest.raises(ValueError):
            validate_config(d)


def test_training_telemetry_missing_is_unknown_not_zero(tmp_path):
    run = tmp_path / "missing"
    assert summarize_training(run)["last_core_survival"] is None
    assert summarize_training(run)["max_logged_box_foot_contact_fraction"] is None

    run.mkdir()
    with (run / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "step", "terrain_box_foot_contact_any",
            "servo_over_rated_frac", "fall_rate"])
        writer.writeheader()
        writer.writerow(dict(step=200, terrain_box_foot_contact_any="0",
                             servo_over_rated_frac=".45", fall_rate=".9"))
        writer.writerow(dict(step=400, terrain_box_foot_contact_any=".008",
                             servo_over_rated_frac=".47", fall_rate=".6"))
    with (run / "fixed_eval.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "eval_core_survival"])
        writer.writeheader()
        writer.writerow(dict(step=400, eval_core_survival=".75"))
    d = summarize_training(run)
    assert d["max_logged_box_foot_contact_fraction"] == pytest.approx(.008)
    assert d["logged_box_contact_positive_rows"] == 1
    assert d["last_core_survival"] == pytest.approx(.75)


def test_progression_requires_contact_and_retained_flat_skill():
    healthy = dict(max_logged_box_foot_contact_fraction=.001,
                   last_core_survival=.8, last_servo_over_rated_frac=.65)
    assert extension_eligible(healthy, 2, 1)[0]
    assert not extension_eligible(healthy, 2, 0)[0]
    for bad in (
        {**healthy, "max_logged_box_foot_contact_fraction": 0.},
        {**healthy, "max_logged_box_foot_contact_fraction": None},
        {**healthy, "last_core_survival": None},
        {**healthy, "last_core_survival": .34},
        {**healthy, "last_servo_over_rated_frac": .87},
    ):
        assert not extension_eligible(bad, 2, 2)[0]


def test_crossing_exams_are_scored_independent_of_process_exit(tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    def record(path, height, outcomes, contact):
        path.write_text(json.dumps(dict(
            worlds=len(outcomes), step_height_mm=height,
            crossing_success_worlds=sum(outcomes),
            step_contact_world_frames=contact,
            per_world=[
                dict(crossing_pass=x,
                     rejection_reasons=[] if x else ["fall_or_reset"],
                     final_signed_progress_m=.22)
                for x in outcomes])))

    record(a, 5, [False, False, False, True], 7)
    record(b, 0, [True, False, True, False], 0)
    r = summarize_crossings([a, b])
    assert r["evaluated_worlds"] == 8
    assert r["actual_crossings"] == 3
    assert r["by_height_mm"]["5"]["passed"] == 1
    assert r["step_contact_world_frames"] == 7
    assert r["NOT_a_PPO_reward_metric"]
    bad = json.loads(a.read_text())
    bad["crossing_success_worlds"] = 99
    a.write_text(json.dumps(bad))
    with pytest.raises(RuntimeError, match="Contradictory"):
        summarize_crossings([a])


def test_archive_does_not_expose_pkl_or_big_policy(tmp_path):
    root = tmp_path / "overnight_example"
    (root / "seed733" / "checkpoints").mkdir(parents=True)
    (root / "stage.log").write_text("PASS\n")
    (root / "seed733" / "checkpoints" / "final.pkl").write_bytes(b"pretend large checkpoint")
    (root / "candidate.emupolicy").write_text("not for archive")
    archive(root)
    with tarfile.open(root.with_suffix(".tar.gz")) as t:
        names = t.getnames()
    assert any(n.endswith("stage.log") for n in names)
    assert not any(n.endswith((".pkl", ".emupolicy")) for n in names)
