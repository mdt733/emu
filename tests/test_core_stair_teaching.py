"""Core stair physics and gait-payment contracts (former v5051 investigation)."""

from __future__ import annotations
import unittest
import numpy as np
from emu.rewards import _physical_qualities


class CoreStairPhysicsTests(unittest.TestCase):

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
            outcome_graduation=True,
            science_step_finish_on_transfer=True,
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
            science_step_goal_weight=.30,
            science_step_velocity_relief=.80,
            science_step_approach_m=.22,
        )
        packet=env.params_array(dict(specialist_mix=1.,
                                     focus_weight_step=1.,
                                     academy_episode_probability=1.,
                                     terrain_height_m=.02,
                                     academy_height_up_m=.02,
                                     academy_height_down_m=.02))
        state,obs=env.reset(dict(specialist_mix=1.,
                                 focus_weight_step=1.,
                                 academy_episode_probability=1.,
                                 terrain_height_m=.02,
                                 academy_height_up_m=.02,
                                 academy_height_down_m=.02))
        out=env.step(state,jnp.zeros((4,6),jnp.float32),packet)
        self.assertEqual(out.obs.shape,(4,360))
        self.assertTrue(np.isfinite(np.asarray(jax.device_get(out.reward))).all())
        self.assertEqual(out.state.academy_progress_origin_xy.shape,(4,2))
        self.assertEqual(out.state.nursery_best.shape,(4,6))
        self.assertIn('nursery_reward',out.metrics)
        self.assertIn('nursery_progress_r_lift',out.metrics)
        self.assertIn('nursery_progress_l_over',out.metrics)
        # Real 50mm (+/-4%) prism seam exists at birth, not a 3mm seed.
        lips=np.asarray(jax.device_get(state.academy_birth_lip_m))
        self.assertTrue(np.all((lips>.047)&(lips<.053)),lips)
        born=np.asarray(jax.device_get(state.data.qpos))
        self.assertTrue(np.all((born[:,0]>=.70)&(born[:,0]<=.85)))
        self.assertIn("academy_followthrough_new",out.metrics)
        self.assertIn("academy_goal_complete_new",out.metrics)
        self.assertIn("academy_episode_first_lip_armed",out.metrics)
        self.assertIn("academy_episode_two_foot_ready",out.metrics)
        self.assertIn("hip_roll_motor_load_frac_left",out.metrics)
        self.assertIn("hip_roll_motor_load_frac_right",out.metrics)
        self.assertIn("hip_roll_spring_abs_nm_left",out.metrics)
        self.assertIn("hip_roll_spring_abs_nm_right",out.metrics)
        self.assertIn("actuator_joint_viscous_damping_w",out.metrics)
        self.assertIn("knee_strut_energy_j",out.metrics)
        self.assertIn("knee_strut_force_n_left",out.metrics)
        self.assertIn("knee_strut_force_n_right",out.metrics)
        self.assertTrue(np.all(np.asarray(jax.device_get(
            out.metrics["knee_strut_energy_j"])) > 0.))
        self.assertTrue(np.all(np.asarray(jax.device_get(
            out.metrics["knee_strut_force_n_left"])) > 0.))
        np.testing.assert_allclose(
            np.asarray(jax.device_get(out.metrics["knee_strut_force_n_left"])),
            np.asarray(jax.device_get(out.metrics["knee_strut_force_n_right"])),
            atol=10.,rtol=.5)  # dynamic worlds may lean, no strict equal-load requirement
        self.assertTrue(np.all(np.asarray(jax.device_get(
            out.metrics["actuator_joint_viscous_damping_w"]))>=0.))
        self.assertTrue(np.isfinite(np.asarray(
            jax.device_get(out.metrics["hip_roll_motor_load_frac_left"]))).all())
        self.assertIn("physical_position_xy_after_step",out.metrics)
        self.assertIn("academy_goal_remaining_m",out.metrics)
        self.assertEqual(out.state.academy_goal_xy.shape,(4,2))
        self.assertTrue(np.isfinite(
            np.asarray(jax.device_get(out.state.academy_goal_xy))).all())

    def test_real_millimetre_credit_cannot_farm_or_cross_wall_low(self):
        from emu.step_goal import nursery_progress
        foot=np.array([[[.96,.48,.14],[.90,.53,.12]]],np.float32)
        body=np.array([[.79,.5]],np.float32)
        bottom=np.array([[.13,.11]],np.float32)
        top=np.array([[.10,.10]],np.float32)
        lip=np.array([.05],np.float32)
        edge=np.array([1.],np.float32)
        seen=np.array([[False,False]])
        ok=np.array([True])
        zero=np.zeros((1,6),np.float32)
        best,first,gains=nursery_progress(np,zero,body,foot,bottom,edge,top,lip,seen,ok)
        self.assertGreater(float(first[0]),0.)
        self.assertEqual(float(best[0,3]),0.)  # below upper surface; no crossing credit
        same,twice,_=nursery_progress(np,best,body,foot,bottom,edge,top,lip,seen,ok)
        np.testing.assert_array_equal(same,best)
        self.assertEqual(float(twice[0]),0.)
        down,again,_=nursery_progress(np,best,body,foot,np.array([[.11,.10]],np.float32),edge,top,lip,seen,ok)
        self.assertEqual(float(again[0]),0.)
        high=foot.copy()
        high[:,:,2]=.20
        third,earned,_=nursery_progress(np,best,body,high,
                            np.array([[.19,.19]],np.float32),edge,top,lip,seen,ok)
        self.assertGreater(float(earned[0]),0.)
        exhausted,last,_=nursery_progress(np,third,body,high,
                            np.array([[.19,.19]],np.float32),edge,top,lip,seen,ok)
        self.assertEqual(float(last[0]),0.)
        self.assertLessEqual(float(third.max()),1.)
        disabled,pay,_=nursery_progress(np,zero,body,high,
                            np.array([[.19,.19]],np.float32),edge,top,lip,seen,np.array([False]))
        self.assertEqual(float(pay[0]),0.)
        np.testing.assert_array_equal(disabled,zero)

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

