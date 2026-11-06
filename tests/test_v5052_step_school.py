"""v5052 experiment contracts: real PPO forks and source-pinned step teaching.

CPU-only arithmetic and CLI tests; no GPU or unnecessary physics compilation.
The existing true-prism MJX smoke tests continue to test physical contracts.
"""
from __future__ import annotations
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from emu.rewards import REWARD_ONLY_METRICS, training_reward
from emu.terrain_carpet_mjx import TILE_M, academy_near_lip_spawn_x
from emu.cli import main as emu_cli
from scripts.v5052_step_school import (
    PARENT_STEPS, TARGET_STEPS, ARMS, EXAMS, command,
    contract, forensic, physical_protocol, select_arms,
)


def metrics(*, vx=.12, active=True, support=1., slip=0.):
    # Two identical physical worlds, differing ONLY in Academy assignment.
    values=dict(
        terminal=0.,vx=vx,cmd_vx=.12,vy=0.,cmd_vy=0.,
        yaw_rate=0.,cmd_yaw=0.,body_pitch=0.,cmd_pitch=0.,
        age_s=2.,ramp_frac=1.,attitude_quality=1.,
        height_error=0.,foot_mid_cone=1.,bad_contact=0.,
        support=support,stance_slip=slip,swing_clearance=.03,
        target_step_height=.02,organic_step_shape=1.,phase_sync=0.,
        step_evidence=.2,action_delta_norm=0.,joint_limit_pressure=0.,
        scissor_pressure=0.,stepping_activity_so_far=.2,gravity_z=1.,
        science_step_focus=1.,
    )
    return {**{k:np.array([v,v],dtype=np.float32) for k,v in values.items()},
            "science_step_goal_active":np.array([float(active),0.],dtype=np.float32)}


class StepSchoolTests(unittest.TestCase):
    def test_seven_real_forks_and_physical_exams(self):
        self.assertEqual(PARENT_STEPS,40_000_000)
        self.assertEqual(TARGET_STEPS,80_000_000)
        self.assertEqual(len(ARMS),7)
        self.assertEqual(len({a.name for a in ARMS}),len(ARMS))
        self.assertEqual(len(EXAMS),8)
        self.assertEqual({mode for _,mode,_,_ in EXAMS},{"flat","step"})
        for arm in ARMS:
            self.assertEqual(len(arm.overrides)%2,0)
        with self.assertRaises(ValueError):
            select_arms("00_control,00_control")

    def test_true_live_trainer_parses_every_arm(self):
        with TemporaryDirectory() as temp:
            root=Path(temp)/"study"
            for arm in (None,*ARMS):
                cmd=command(root,seed=735,envs=2048,arm=arm,
                            resume=(root/"parent/checkpoints/final.pkl"
                                    if arm is not None else None))
                self.assertIn("--carpet-prisms",cmd)
                self.assertIn("--academy-stream",cmd)
                self.assertEqual(cmd[cmd.index("--steps")+1],
                                 str(TARGET_STEPS if arm else PARENT_STEPS))
                self.assertEqual(0,emu_cli([*cmd[3:],"--plan"]))

    def test_reward_goal_and_velocity_relief_are_academy_only(self):
        m=metrics(vx=.08)
        regime=np.zeros((2,),dtype=np.float32)
        original,_=training_reward(m,regime,xp=np)
        goal,pay=training_reward(m,regime,xp=np,science_step_goal_weight=.30)
        self.assertGreater(goal[0],original[0])
        self.assertAlmostEqual(float(goal[1]),float(original[1]),places=6)
        self.assertIn("weighted_goal_progress",pay)
        relief,_=training_reward(m,regime,xp=np,
                                 science_step_velocity_relief=.8)
        self.assertGreater(relief[0],original[0])
        self.assertAlmostEqual(float(relief[1]),float(original[1]),places=6)
        no_motion,_=training_reward(metrics(vx=0.),regime,xp=np,
                                    science_step_goal_weight=.30)
        no_motion_baseline,_=training_reward(metrics(vx=0.),regime,xp=np)
        np.testing.assert_allclose(no_motion,no_motion_baseline,atol=1e-7)
        no_contact,_=training_reward(metrics(vx=.08,support=0.),regime,xp=np,
                                     science_step_goal_weight=.30)
        no_contact_baseline,_=training_reward(metrics(vx=.08,support=0.),regime,xp=np)
        # Support=0 is a partial penalty, not full zero, because the existing
        # support signal is averaged per-frame; goal still cannot be awarded
        # as a certified stair crossing. No zero-contact bonus is expected.
        np.testing.assert_allclose(no_contact,no_contact_baseline,atol=1e-7)
        self.assertIn("science_step_goal_active",REWARD_ONLY_METRICS)

    def test_academy_birth_is_exactly_upstream_of_real_prism_seam(self):
        starts=np.array([-.02,.02,.37,.80],dtype=np.float32)
        near=academy_near_lip_spawn_x(np,starts,.22)
        np.testing.assert_allclose(near,[-.22,.53,.53,1.28],atol=1e-6)
        # A robot that begins 22cm before a tile edge should always see
        # the boundary ahead without changing actual prism heights.
        np.testing.assert_allclose(
            (np.floor(near/TILE_M)+1)*TILE_M-near,
            np.full_like(near,.22),atol=1e-6)
        np.testing.assert_array_equal(
            academy_near_lip_spawn_x(np,starts,0.),starts)

    def test_protocol_is_immutable(self):
        with TemporaryDirectory() as temp:
            root=Path(temp)
            first=physical_protocol(root,8,650)
            self.assertEqual(first,physical_protocol(root,8,650))
            with self.assertRaisesRegex(RuntimeError,"protocol"):
                physical_protocol(root,16,650)

    def test_physical_failure_stages_do_not_invent_success(self):
        with TemporaryDirectory() as temp:
            root=Path(temp)
            dest=root/"exams"/"00_control"/"s1701-lip15_slow"
            dest.mkdir(parents=True)
            (dest/"prism_exam.json").write_text(json.dumps(dict(
                mode="step",protocol=dict(requested_vx_mps=.12),
                worlds=[{}],
                mission_autopsy=dict(events=[dict(
                    verdict="trailing_foot_not_verified",
                    signed_actual_height_cm=2.1,
                    both_feet_upper_support=False,
                    post_contact_progress_m=0.,complete=False
                )]))))
            results=forensic(root,select_arms("00_control"))
            self.assertEqual(results[0]["eligible_contacts"],1)
            self.assertEqual(results[0]["up_contact_ge_2cm"],1)
            self.assertEqual(results[0]["bilateral"],0)
            self.assertEqual(results[0]["complete"],0)


if __name__=="__main__":
    unittest.main()
