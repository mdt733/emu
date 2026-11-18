"""Pure scoring contract tests: no GPU, PPO or trained actor required."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from emu.physical_capability_exam import (
    Scenario, score_first_life, summarize, save_evidence)


def fixture(seconds=1, worlds=2):
    n = 50 * seconds
    zeros = np.zeros((n, worlds))
    pos = np.zeros((n, worlds, 2))
    return dict(
        done=np.zeros((n, worlds), dtype=bool),
        fell=np.zeros((n, worlds), dtype=bool),
        position_xy=pos.copy(),
        physical_position_xy_after_step=pos.copy(),
        heading_rad=zeros.copy(),
        physical_heading_rad_after_step=zeros.copy(),
        command=np.zeros((n, worlds, 3)),
        terrain_gravity_z=np.ones((n, worlds)),
        terrain_bad_contact=zeros.copy(),
        floor_contact=np.ones((n, worlds, 2), dtype=bool),
        servo_shaft_drive_power_w=zeros.copy(),
    )


class PhysicalV1ScoreTests(unittest.TestCase):
    def test_zero_command_real_standing(self):
        case = Scenario("still", 1, ((1, 0., 0., 0.),), "balance")
        rows = score_first_life(fixture(), case)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(x["bounded_standing_pass"] for x in rows))
        self.assertTrue(all(x["survived_full_horizon"] for x in rows))
        self.assertAlmostEqual(rows[0]["final_drift_m"], 0.)

    def test_first_fall_excludes_subsequent_reset_lives(self):
        case = Scenario("still", 1, ((1, 0., 0., 0.),), "balance")
        x = fixture()
        x["done"][6, 1] = True
        x["fell"][6, 1] = True
        x["physical_position_xy_after_step"][7:, 1, 0] = 99.
        rows = score_first_life(x, case)
        self.assertEqual(rows[1]["first_life_frames"], 7)
        self.assertAlmostEqual(rows[1]["time_to_first_fall_s"], .14)
        self.assertEqual(rows[1]["terminal_reason"], "fall")
        self.assertFalse(rows[1]["survived_full_horizon"])
        self.assertFalse(rows[1]["bounded_standing_pass"])
        self.assertAlmostEqual(rows[1]["max_radius_m"], 0.)
        self.assertTrue(rows[0]["survived_full_horizon"])

    def test_walk_away_is_not_standing(self):
        case = Scenario("still", 1, ((1, 0., 0., 0.),), "balance")
        x = fixture()
        x["physical_position_xy_after_step"][:, 0, 0] = (
            .01 * np.arange(1, 51))
        rows = score_first_life(x, case)
        self.assertTrue(rows[0]["survived_full_horizon"])
        self.assertFalse(rows[0]["bounded_standing_pass"])
        self.assertAlmostEqual(rows[0]["max_radius_m"], .5)

    def test_requested_trajectory_and_actor_are_distinct(self):
        case = Scenario("forward", 1, ((1, .25, 0., 0.),),
                        "locomotion")
        x = fixture()
        x["command"][:, :, 0] = .25
        x["physical_position_xy_after_step"][:, :, 0] = (
            .005 * np.arange(1, 51)[:, None])
        rows = score_first_life(x, case)
        self.assertAlmostEqual(rows[0]["vx_mae_mps"], 0., places=5)
        self.assertAlmostEqual(rows[0]["final_trajectory_error_m"],
                               0., places=5)

    def test_transition_requires_real_stop(self):
        case = Scenario("forward_stop_reverse_30s", 30,
                        ((10, .25, 0., 0.), (10, 0., 0., 0.),
                         (10, -.20, 0., 0.)), "transitions")
        x = fixture(seconds=30)
        x["command"][:, :, :] = case.command_schedule()[:, None, :]
        # A policy moving forward through the middle rest period.
        x["physical_position_xy_after_step"][:, :, 0] = (
            .005 * np.arange(1, 1501)[:, None])
        rows = score_first_life(x, case)
        self.assertGreater(rows[0]["stop_interval_mean_speed_mps"], .20)

    def test_partial_scenario_cannot_become_complete_summary(self):
        case = Scenario("still", 1, ((1, 0., 0., 0.),), "balance")
        rows = score_first_life(fixture(), case)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            summarize(rows[:-1], expected_worlds=2)

    def test_zip_does_not_overwrite_prior_exam(self):
        case = Scenario("still", 1, ((1, 0., 0., 0.),), "balance")
        rows = score_first_life(fixture(), case)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "results.zip"
            save_evidence(path, {"schema": "test"}, rows,
                          summarize(rows, expected_worlds=2))
            self.assertTrue(path.is_file())
            with self.assertRaises(FileExistsError):
                save_evidence(path, {"schema": "test"}, rows,
                              summarize(rows, expected_worlds=2))


if __name__ == "__main__":
    unittest.main()
