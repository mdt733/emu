"""Live skill counts must reflect real event denominators, not frame counts."""
import json
import numpy as np
import pytest

from emu.live_skill_observer import LiveSkillObserver
from emu.capability_field import CapabilityField, RECOVERY_BINS, EDGE_DIRS
from emu.training_config import ppo_minibatches_for_envs


def _observer(tmp_path, worlds=4096):
    field = CapabilityField()
    return LiveSkillObserver(tmp_path, start_step=0, worlds=worlds,
                             field_shape=field.shape, field_bounds=field.bounds,
                             source_commit="a"*40, recovery_bins=RECOVERY_BINS,
                             edge_directions=EDGE_DIRS, window_steps=1)


def _rollout(worlds=4096):
    shape = (32, worlds)
    names = ["episode_done", "episode_fell", "specialist_focus", "academy_active",
             "skill_first_birth_lip_m", "academy_episode_first_lip_armed",
             "academy_episode_two_foot_ready", "skill_first_goal_completed",
             "physical_step_height_m", "physical_slope_deg"]
    names += [f"terrain_episode_{k}_{d}" for d in ("up", "down")
              for k in ("attempt", "success")]
    names += [f"slope_episode_{k}_{d}" for d in
              ("uphill", "downhill", "cross_left", "cross_right")
              for k in ("attempt", "stable", "success")]
    m = {k: np.zeros(shape, np.float32) for k in names}
    m["specialist_focus"].fill(-1)
    return m


def test_live_skill_observer_4096_terminal_evidence(tmp_path):
    obs = _observer(tmp_path)
    m = _rollout()
    # Alive, armed, bilateral and goal signals on ongoing frames must not
    # manufacture additional first-life attempts.
    m["academy_active"][:, :5] = 1
    m["specialist_focus"][:, :5] = 3
    m["skill_first_birth_lip_m"][:, :4] = .026
    m["skill_first_birth_lip_m"][:, 4] = -.026
    m["academy_episode_first_lip_armed"][:, :3] = 1
    m["academy_episode_two_foot_ready"][:, :2] = 1
    m["skill_first_goal_completed"][:, 0] = 1
    m["episode_done"][31, :5] = 1
    m["episode_fell"][31, 1:4] = 1
    m["episode_done"][31, 5:7] = 1
    m["terrain_episode_attempt_up"][31, 5:7] = 1
    m["terrain_episode_success_up"][31, 5] = 1
    m["physical_step_height_m"][31, 5:7] = .018
    m["episode_done"][31, 7] = 1
    m["physical_slope_deg"][31, 7] = 3
    m["slope_episode_attempt_uphill"][31, 7] = 1
    m["slope_episode_stable_uphill"][31, 7] = 1
    obs.observe(m, step=131072)
    obs.field_events("recovery_relative", [11, 11, 12], [1, 0, 1])
    obs.flush()
    row = json.loads(obs.path.read_text())
    c = row["cells"]
    def count(key, attempts, passes):
        assert c[key] == dict(attempts=attempts, passes=passes)
    count("academy_first_birth/up/first_lip_armed/7", 4, 3)
    count("academy_first_birth/up/bilateral_ready/7", 4, 2)
    count("academy_first_birth/up/safe_bilateral/7", 4, 1)
    count("academy_first_birth/up/goal_completed/7", 4, 1)
    count("academy_first_birth/down/survived/7", 1, 1)
    count("ordinary_first_encounter/up/safe_cross/5", 2, 1)
    count("slope_terminal_grade/uphill/stable/3", 1, 1)
    count("slope_terminal_grade/uphill/success/3", 1, 0)
    count("recovery_relative/11", 2, 1)
    assert row["worlds"] == 4096 and row["updates"] == 1
    with pytest.raises(ValueError, match="unordered"):
        obs.observe(m, step=131072)
    obs.rollback()
    obs.observe(_rollout(), step=262144)
    obs.flush(reason="final", force=True)
    rows = [json.loads(s) for s in obs.path.read_text().splitlines()]
    assert rows[-1]["policy_epoch"] == 1
    assert rows[-1]["cells"] == {}
    assert rows[-1]["start_step"] == rows[0]["end_step"]


def test_field_event_sink_does_not_change_field(tmp_path):
    a, b = CapabilityField(), CapabilityField()
    obs = _observer(tmp_path)
    flat = [np.array([1, 1, 2], np.int64)]
    success = [np.array([1, 0, 1], np.float32)]
    quality = [np.array([.8, .2, .9], np.float32)]
    bins = [np.array([0, 0, 2], np.int64)]
    src = [np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]], np.float32)]
    tgt = [np.array([[.2, 0., 0.], [0., .2, 0.], [0., 0., 0.]], np.float32)]
    rate = [np.ones(3, np.float32)]
    a._add_node_events(flat, success, quality, 1)
    b._add_node_events(flat, success, quality, 1, obs.field_events)
    a._add_recovery_events(flat, bins, success)
    b._add_recovery_events(flat, bins, success, obs.field_events)
    a._add_edge_events(flat, src, tgt, rate, success)
    b._add_edge_events(flat, src, tgt, rate, success, obs.field_events)
    for key in ("node_success", "node_failure", "node_quality_sum",
                "rec_success", "rec_failure", "edge_success", "edge_failure",
                "edge_rate_sum"):
        np.testing.assert_array_equal(getattr(a, key), getattr(b, key))
    for family, n in (("command_window", 3), ("recovery_relative", 3),
                      ("transition_edge", 2)):
        assert sum(v[0] for k, v in obs.counts.items()
                   if k.startswith(family + "/")) == n
    assert ppo_minibatches_for_envs(2048) == 8
    assert ppo_minibatches_for_envs(4096) == 16
    assert 2048*32//8 == 4096*32//16
