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

        # Capacity-matched oracle experiment: the extra actor channels
        # must use real signed prism mid-edge heights, not requested
        # curriculum height or a plane approximating the collision mesh.
        from emu.environment import research_step_oracle_features
        root=np.asarray([[.78,.50,.40],[1.78,.50,.40]],np.float32)
        feet=np.asarray([
            [[.84,.45,.10],[.70,.55,.05]],
            [[1.84,.45,.10],[1.70,.55,.05]]],np.float32)
        corners=np.zeros((2,4,4),np.float32)
        # In world 0, east tile 1 is 5cm above tile 0.
        # In world 1, east tile 0 is 4cm below tile 1.
        mocap=np.asarray([[0.,.05,0.,0.],[-.04,0.,0.,0.]],np.float32)
        goal=np.asarray([[1.60,.50],[2.60,.50]],np.float32)
        oracle=research_step_oracle_features(
            np,root,feet,corners,mocap,goal)
        self.assertEqual(oracle.shape,(2,10))
        np.testing.assert_allclose(oracle[:,0],[.22,.22],atol=1e-6)
        # Explicit feature scale is decimetres for signed lip height.
        np.testing.assert_allclose(oracle[:,1],[.5,-.4],atol=1e-6)
        np.testing.assert_allclose(oracle[:,8],[.82,.82],atol=1e-6)
        self.assertTrue(np.isfinite(oracle).all())


if __name__=="__main__":

    unittest.main()

