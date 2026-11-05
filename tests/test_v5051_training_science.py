"""v5051 source-pinned PPO experimental controls; no CUDA required."""
from __future__ import annotations
from contextlib import redirect_stdout
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from emu.cli import main as emu_cli
from emu.rewards import _gait_maturity, _physical_qualities
from scripts.v5051_train_campaign import (
    ARMS, PARENT_STEPS, FINAL_STEPS, cli_command, select_arms)


class ScienceTrainingTests(unittest.TestCase):
    def test_fifteen_independent_60m_forks(self):
        self.assertEqual(len(ARMS),15)
        self.assertEqual(len({a.name for a in ARMS}),len(ARMS))
        self.assertEqual(PARENT_STEPS,60_000_000)
        self.assertEqual(FINAL_STEPS,120_000_000)
        for arm in ARMS:
            self.assertEqual(len(arm.overrides)%2,0)

    def test_all_arms_parse_to_real_trainer_without_training(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)/"v5051"
            for arm in (None,*ARMS):
                cmd=cli_command(root,seed=735,envs=2048,arm=arm,
                                resume=(root/"foundation/checkpoints/final.pkl"
                                        if arm is not None else None))
                self.assertIn("--steps",cmd)
                self.assertEqual(cmd[cmd.index("--steps")+1],
                                 str(FINAL_STEPS if arm else PARENT_STEPS))
                self.assertIn("--carpet-prisms",cmd)
                self.assertIn("--academy-stream",cmd)
                with redirect_stdout(io.StringIO()):
                    rc=emu_cli([*cmd[3:],"--plan"])
                self.assertEqual(rc,0,arm.name if arm else "foundation")

    def test_four_world_actual_mjx_science_update(self):
        # Compile the *real physics step* including new transfer event
        # memory and all reward-side research branches, not a toy reward
        # calculation. Four worlds are deliberately CPU-bounded.
        from emu.contracts import ROBOT_XML, OBS_SCHEMA_TERRAIN
        from emu.environment import EmuMJXEnv
        import jax.numpy as jnp
        import jax
        env=EmuMJXEnv(
            str(ROBOT_XML),num_envs=4,seed=735,
            terrain_step_max_m=.08,observation_schema=OBS_SCHEMA_TERRAIN,
            unified_capability_focus=True,terrain_carpet=True,
            terrain_carpet_prisms=True,academy_stream=True,
            terrain_lesson="coherent-careful",
            solver_iterations=2,solver_ls_iterations=4,
            science_gate_mode="zero-only",
            science_phase_lowcmd_scale=.25,
            science_phase_sync_strength=0.,
            science_crouch_allowance_m=.03,
            science_stride_efficiency_weight=.08,
            science_supported_cross_bonus=6.,
            science_foothold_bonus=3.,
            science_bilateral_support_bonus=4.,
            science_followthrough_bonus=10.,
            science_sensor_teacher_bonus=3.,
        )
        packet=env.params_array(dict(specialist_mix=1.,
                                     focus_weight_step=1.,
                                     academy_episode_probability=1.,
                                     terrain_height_m=.02,
                                     academy_height_up_m=.02,
                                     academy_height_down_m=.02))
        state,obs=env.reset(packet)
        out=env.step(state,jnp.zeros((4,6),jnp.float32),packet)
        self.assertEqual(out.obs.shape,(4,360))
        self.assertTrue(np.isfinite(np.asarray(jax.device_get(out.reward))).all())
        self.assertEqual(out.state.academy_progress_origin_xy.shape,(4,2))
        self.assertIn("academy_followthrough_new",out.metrics)

    def test_no_duplicate_arm_selection(self):
        self.assertEqual(select_arms("00_control")[0].name,"00_control")
        with self.assertRaises(ValueError):
            select_arms("00_control,00_control")
        with self.assertRaises(ValueError):
            select_arms("nonexistent")

    def test_gate_zero_only_keeps_motion_requirement(self):
        m=dict(age_s=np.array([2.,2.]),
               stepping_activity_so_far=np.array([0.,0.]),
               cmd_vx=np.array([0.,.25]),cmd_vy=np.zeros(2),
               cmd_yaw=np.zeros(2))
        _,legacy=_gait_maturity(m,np)
        _,zero=_gait_maturity(m,np,gate_mode="zero-only")
        _,ungated=_gait_maturity(m,np,gate_mode="off")
        np.testing.assert_allclose(legacy,[.3,.3])
        np.testing.assert_allclose(zero,[1.,.3])
        np.testing.assert_allclose(ungated,[1.,1.])

    def test_sync_off_reduces_only_phase_payment(self):
        m=dict(cmd_pitch=np.ones(1)*0.,height_error=np.zeros(1),
               attitude_quality=np.ones(1),foot_mid_cone=np.ones(1),
               bad_contact=np.zeros(1),support=np.ones(1),
               stance_slip=np.zeros(1),swing_clearance=np.zeros(1),
               target_step_height=np.ones(1)*.03,
               organic_step_shape=np.array([.7]),
               phase_sync=np.array([1.]),step_evidence=np.array([1.]),
               action_delta_norm=np.zeros(1),
               joint_limit_pressure=np.zeros(1),scissor_pressure=np.zeros(1))
        baseline=_physical_qualities(m,np)
        stripped=_physical_qualities(m,np,phase_sync_strength=0.)
        self.assertAlmostEqual(float(baseline[0][0]),float(stripped[0][0]))
        self.assertAlmostEqual(float(baseline[3][0]-stripped[3][0]),.135)
        # Crouch tolerance uses a teacher-only STEP flag; non-STEP unchanged.
        low=dict(m,height_error=np.array([-.045]),
                 science_step_focus=np.array([1.]))
        step=_physical_qualities(low,np,crouch_allowance_m=.03)
        plain=_physical_qualities(low,np)
        self.assertGreater(float(step[0][0]),float(plain[0][0]))
        low["science_step_focus"]=np.array([0.])
        no_step=_physical_qualities(low,np,crouch_allowance_m=.03)
        np.testing.assert_allclose(no_step[0],plain[0])


if __name__=="__main__":
    unittest.main()
