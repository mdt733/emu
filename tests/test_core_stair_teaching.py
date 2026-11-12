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
            progressive_school=True,
            refined_school=True,
            bidirectional_school=True,
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
        # The true birth lip, not a moving difficulty bucket, owns
        # the separate 50mm retained-skill evidence.
        self.assertIn('academy_anchor_attempt_up',out.metrics)
        self.assertIn('academy_anchor_success_up',out.metrics)
        self.assertIn('academy_anchor_attempt_down',out.metrics)
        self.assertIn('academy_anchor_success_down',out.metrics)
        self.assertTrue(np.all(np.asarray(jax.device_get(
            out.metrics['academy_anchor_success_up']))<=np.asarray(
            jax.device_get(out.metrics['academy_anchor_attempt_up']))))
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
        # Exercise the REAL signed -20mm MJX seam too. Historically the
        # >30mm arm gate made this whole cohort physically unteachable.
        down_params=dict(specialist_mix=1.,focus_weight_step=1.,
                         academy_episode_probability=1.,
                         terrain_height_m=.02,academy_height_up_m=.05,
                         academy_height_down_m=.02,outcome_down_share=1.,
                         outcome_down_stair_alpha=0.)
        down_state,_=env.reset(down_params)
        signed=np.asarray(jax.device_get(down_state.academy_birth_lip_m))
        self.assertTrue(np.all((signed<-.0191)&(signed>-.0209)),signed)
        down_step=env.step(
            down_state,jnp.zeros((4,6),jnp.float32),
            env.params_array(down_params))
        self.assertEqual(down_step.obs.shape,(4,360))
        self.assertTrue(np.isfinite(np.asarray(
            jax.device_get(down_step.reward))).all())
        self.assertIn('academy_anchor_attempt_down',down_step.metrics)
        self.assertIn('outcome_down_stair_alpha',down_step.metrics)
        self.assertTrue(np.all(np.asarray(jax.device_get(
            down_step.metrics['outcome_down_stair_alpha']))==0.))

    def _assert_real_millimetre_credit_cannot_farm_or_cross_wall_low(self):
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

    def _assert_refined_school_proof_and_checkpoint_contract(self):
        """Second generation school, starting at the *actual* 180M parent.

        Coached 5cm completion must not change height/downhill on graduation;
        two fresh 100-encounter windows after full coaching fade must prove
        independent 5cm support. Later 7.5cm and downhill are separate gates.
        """
        from emu.curriculum import RegimeGovernor
        from emu.capability_focus import school_lesson
        base=RegimeGovernor(unified_capability_focus=True,
                            outcome_mode=True, progressive_school=True)
        base.school_walking_graduated=True
        base.params.academy_height_up_m=.05
        parent=base.state_dict()
        parent.pop('school_refined_state')  # source-pinned 180M format
        school=RegimeGovernor(unified_capability_focus=True,
                              outcome_mode=True,progressive_school=True,
                              refined_school=True)
        school.load_state_dict(parent)
        self.assertEqual(school.school_refined_state['coaching_off_step'],-1)
        self.assertEqual(school.school_stair_mastered_step,-1)
        school.school_stair_mastered_step=264_000_000
        school.params.academy_height_up_m=.05
        assisted=school_lesson(
            step=270_000_000,core_competence=.8,walking_graduated=True,
            stair_mastered_step=264_000_000,refined_school=True)
        self.assertEqual(assisted['down_share'],0.)
        self.assertLess(assisted['stair_alpha'],1.)
        self.assertEqual(school.observe_refined_school_evidence(
            {},step=270_000_000,stair_alpha=assisted['stair_alpha']),
            'fade_5cm_stair_coaching')
        self.assertEqual(school.observe_refined_school_evidence(
            {},step=284_000_000,stair_alpha=1.),
            'coaching_off_begin_fresh_proof')
        def near(success, step):
            return school.observe_refined_school_evidence(
                {'academy_near_attempts_up':100.,
                 'academy_near_successes_up':float(success)},
                step=step,stair_alpha=1.)
        self.assertEqual(near(66,284_100_000),
                         'uncoached_50mm_first_window_pass')
        self.assertEqual(near(20,284_200_000),'uncoached_50mm_retry')
        self.assertEqual(near(69,284_300_000),
                         'uncoached_50mm_first_window_pass')
        self.assertEqual(near(72,284_400_000),
                         'uncoached_50mm_certified')
        self.assertEqual(school.school_refined_state[
            'uncoached_certified_step'],284_400_000)
        pre=school_lesson(
            step=284_400_000,core_competence=.8,
            walking_graduated=True,stair_mastered_step=264_000_000,
            refined_school=True,
            uncoached_certified_step=284_400_000)
        self.assertEqual(pre['stage'],'expand_uphill_retaining_5cm')
        self.assertEqual(pre['down_share'],0.)
        school.observe_refined_school_evidence(
            {'academy_anchor_attempts_up':100.,
             'academy_anchor_successes_up':20.},
            step=290_000_000,stair_alpha=1.)
        self.assertEqual(school.school_refined_state[
            'retention_hold_until'],300_000_000)
        school.params.academy_height_up_m=.075
        def high(success,step):
            return school.observe_refined_school_evidence(
                {'academy_near_attempts_up':100.,
                 'academy_near_successes_up':float(success)},
                step=step,stair_alpha=1.)
        self.assertEqual(high(70,301_000_000),
                         'prove_75mm_first_window_pass')
        self.assertEqual(high(69,302_000_000),
                         'uphill_75mm_certified_descent_opened')
        opened=school_lesson(
            step=302_000_000,core_competence=.8,
            walking_graduated=True,stair_mastered_step=264_000_000,
            refined_school=True,uncoached_certified_step=284_400_000,
            descent_unlocked_step=302_000_000)
        self.assertEqual(opened['down_share'],.10)
        self.assertEqual(opened['stage'],'introduce_descent')
        school.params.academy_height_down_m=.05
        def down(success,step):
            return school.observe_refined_school_evidence(
                {'academy_near_attempts_down':100.,
                 'academy_near_successes_down':float(success)},
                step=step,stair_alpha=1.)
        self.assertEqual(down(60,303_000_000),'prove_downhill_50mm_retry')
        self.assertEqual(down(70,304_000_000),
                         'prove_downhill_50mm_first_window_pass')
        self.assertEqual(down(72,305_000_000),
                         'downhill_50mm_certified')
        done=school_lesson(
            step=305_000_000,core_competence=.8,
            walking_graduated=True,stair_mastered_step=264_000_000,
            refined_school=True,uncoached_certified_step=284_400_000,
            descent_unlocked_step=302_000_000,
            descent_certified_step=305_000_000)
        self.assertEqual(done['down_share'],.20)
        # A school that repeatedly fails no-help climbing must be able to
        # re-teach without restoring gait-phase rewards or growing the lip.
        remedial=RegimeGovernor(unified_capability_focus=True,
                                outcome_mode=True,progressive_school=True,
                                refined_school=True)
        remedial.load_state_dict(parent)
        remedial.school_stair_mastered_step=264_000_000
        remedial.observe_refined_school_evidence(
            {},step=284_000_000,stair_alpha=1.)
        bad={'academy_near_attempts_up':100.,
             'academy_near_successes_up':25.}
        self.assertEqual(remedial.observe_refined_school_evidence(
            bad,step=284_100_000,stair_alpha=1.),
            'uncoached_50mm_retry')
        self.assertEqual(remedial.observe_refined_school_evidence(
            bad,step=284_200_000,stair_alpha=1.),
            'uncoached_50mm_rescue_started')
        end=remedial.school_refined_state['rescue_end_step']
        recovered=school_lesson(
            step=284_300_000,core_competence=.8,walking_graduated=True,
            stair_mastered_step=264_000_000,refined_school=True,
            rescue_end_step=end)
        self.assertEqual(recovered['stage'],'recover_5cm_guidance')
        self.assertEqual(recovered['stair_alpha'],.75)
        self.assertEqual(school_lesson(
            step=end+2_500_000,core_competence=.8,
            walking_graduated=True,stair_mastered_step=264_000_000,
            refined_school=True,rescue_end_step=end)['stair_alpha'],.875)
        self.assertEqual(school_lesson(
            step=end+5_000_000,core_competence=.8,
            walking_graduated=True,stair_mastered_step=264_000_000,
            refined_school=True,rescue_end_step=end)['stair_alpha'],1.)
        state=school.state_dict()
        resumed=RegimeGovernor(unified_capability_focus=True,
                              outcome_mode=True,progressive_school=True,
                              refined_school=True)
        resumed.load_state_dict(state)
        self.assertEqual(state,resumed.state_dict())
        broken=dict(state,school_refined_state={
            **state['school_refined_state'],'down_successes':101.,
            'down_attempts':100.})
        with self.assertRaisesRegex(ValueError,'successes exceed'):
            resumed.load_state_dict(broken)

    def _assert_bidirectional_school_proof_and_checkpoint(self):
        """20mm downhill must be learned at the SAME time as 50mm uphill."""
        from emu.curriculum import RegimeGovernor
        from emu.capability_focus import bidirectional_down_lesson
        # The old 180M source contains neither graduate's new evidence.
        old=RegimeGovernor(unified_capability_focus=True,
                           outcome_mode=True,progressive_school=True)
        old.school_walking_graduated=True
        parent=old.state_dict()
        parent.pop('school_refined_state')
        g=RegimeGovernor(unified_capability_focus=True,outcome_mode=True,
                         progressive_school=True,refined_school=True,
                         bidirectional_school=True)
        g.load_state_dict(parent)
        self.assertEqual(g.school_stair_mastered_step,-1)
        self.assertEqual(g.school_bidirectional_state['down_mastered_step'],-1)
        g.params.academy_height_down_m=.020
        before=bidirectional_down_lesson(
            step=180_000_000,stairs_enabled=True,mastered_step=-1,
            uncoached_step=-1,certified_5cm_step=-1)
        self.assertEqual(before['down_share'],.50)
        self.assertEqual(before['down_alpha'],0.)
        self.assertEqual(g.observe_bidirectional_school_evidence(
            {'academy_near_attempts_down':100.,
             'academy_near_successes_down':75.},step=181_000_000),
            'downhill_20mm_assisted_first_window_pass')
        self.assertEqual(g.observe_bidirectional_school_evidence(
            {'academy_near_attempts_down':100.,
             'academy_near_successes_down':71.},step=182_000_000),
            'downhill_20mm_coached_certified')
        self.assertEqual(g.school_stair_mastered_step,-1,
                         'downhill certification must NOT graduate uphill')
        during=bidirectional_down_lesson(
            step=192_000_000,stairs_enabled=True,
            mastered_step=182_000_000,uncoached_step=-1,
            certified_5cm_step=-1)
        self.assertAlmostEqual(during['down_alpha'],.5)
        self.assertEqual(g.observe_bidirectional_school_evidence(
            {},step=202_000_000),
            'downhill_coaching_off_start_clean_proof')
        self.assertEqual(g.observe_bidirectional_school_evidence(
            {'academy_near_attempts_down':100.,
             'academy_near_successes_down':70.},step=202_100_000),
            'downhill_20mm_uncoached_first_window_pass')
        self.assertEqual(g.observe_bidirectional_school_evidence(
            {'academy_near_attempts_down':100.,
             'academy_near_successes_down':69.},step=202_200_000),
            'downhill_20mm_uncoached_certified')
        b=g.school_bidirectional_state
        self.assertEqual(b['down_uncoached_certified_step'],202_200_000)
        self.assertEqual(g.observe_bidirectional_school_evidence(
            {},step=203_000_000),'downhill_grow_to_50mm')
        g.params.academy_height_down_m=.05
        self.assertEqual(g.observe_bidirectional_school_evidence(
            {'academy_anchor_attempts_down':100.,
             'academy_anchor_successes_down':80.,
             'academy_near_attempts_down':100.,
             'academy_near_successes_down':71.},step=244_000_000),
            'downhill_50mm_first_window_pass')
        self.assertEqual(g.observe_bidirectional_school_evidence(
            {'academy_anchor_attempts_down':100.,
             'academy_anchor_successes_down':80.,
             'academy_near_attempts_down':100.,
             'academy_near_successes_down':69.},step=245_000_000),
            'downhill_50mm_uncoached_certified')
        self.assertEqual(g.school_refined_state['descent_unlocked_step'],-1,
                         'downhill 5cm certification does not fake uphill 7.5cm')
        g2=RegimeGovernor(unified_capability_focus=True,outcome_mode=True,
                          progressive_school=True,refined_school=True,
                          bidirectional_school=True)
        g2.load_state_dict(g.state_dict())
        self.assertEqual(g2.state_dict(),g.state_dict())
        bad=g.state_dict()
        bad['school_bidirectional_state']={
            **bad['school_bidirectional_state'],
            'down_high_successes':5.,'down_high_attempts':4.}
        with self.assertRaisesRegex(ValueError,'successes exceed'):
            g2.load_state_dict(bad)
        # The actual down-directed teacher is a landing lesson, NOT a
        # copied uphill foot-lifting incentive or XY-only fake crossing.
        from emu.step_goal import downhill_nursery_progress
        foot=np.array([[[.88,.50,.20],[.89,.50,.20]]])
        low=np.array([[.18,.18]])
        prev=np.zeros((1,6),dtype=np.float32)
        args=(np,np.array([[.87,.50]]),foot,low,np.array([1.]),
              np.array([[.20,.20]]),np.array([-.02]))
        a,pay,gains=downhill_nursery_progress(
            args[0],prev,*args[1:],np.array([[False,False]]),
            np.array([True]))
        self.assertGreater(float(pay[0]),0.)  # approach milestone
        self.assertEqual(float(gains[0,3]),0.)
        self.assertEqual(float(gains[0,4]),0.)

    def test_sync_off_reduces_only_phase_payment(self):
        self._assert_bidirectional_school_proof_and_checkpoint()
        self._assert_refined_school_proof_and_checkpoint_contract()
        self._assert_real_millimetre_credit_cannot_farm_or_cross_wall_low()
        # Checkpoint-persistent stair graduation: two independent successful
        # 100-attempt windows at REAL near-50mm uphill lips. A failed
        # window must reset proof streak; 0/partial footholds never pass.
        from emu.curriculum import RegimeGovernor
        g=RegimeGovernor(unified_capability_focus=True,outcome_mode=True,
                         progressive_school=True)
        g.params.academy_height_up_m=.05
        assert g.observe_school_evidence(
            {'academy_near_attempts_up':100.,
             'academy_near_successes_up':100.},
            step=80_000_000,stairs_enabled=False)=='no_new_50mm_proof'
        self.assertEqual(g.school_up_window_attempts,0.)
        g.school_walking_graduated=True
        def evidence(attempts,successes,step):
            return g.observe_school_evidence(
                {'academy_near_attempts_up':attempts,
                 'academy_near_successes_up':successes},
                step=step,stairs_enabled=True)
        self.assertEqual(evidence(100.,75.,90_000_000),'50mm_window_pass')
        self.assertEqual(g.school_up_good_windows,1)
        self.assertEqual(evidence(100.,40.,90_100_000),'50mm_window_retry')
        self.assertEqual(g.school_up_good_windows,0)
        self.assertEqual(evidence(100.,80.,90_200_000),'50mm_window_pass')
        self.assertEqual(evidence(100.,71.,90_300_000),'50mm_training_mastery')
        self.assertEqual(g.school_stair_mastered_step,90_300_000)
        trained=g.state_dict()
        restored=RegimeGovernor(unified_capability_focus=True,outcome_mode=True,
                                progressive_school=True)
        restored.load_state_dict(trained)
        self.assertEqual(restored.state_dict(),trained)
        damaged=dict(trained,school_up_window_successes=1.,
                     school_up_window_attempts=0.)
        with self.assertRaisesRegex(ValueError,'invalid progressive school'):
            restored.load_state_dict(damaged)
        # Mastery is a TRAINING gate. Independent frozen +50mm exams are
        # still mandatory before making an external capability claim.
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
        # The 60–80M walking graduation removes hidden foot pose shaping
        # while continuing to reward real body uprightness/height.
        low_cone=dict(m,foot_mid_cone=np.zeros(1))
        coached_low=_physical_qualities(low_cone,np)
        self.assertLess(float(coached_low[0][0]),float(baseline[0][0]))
        independent_low=_physical_qualities(
            low_cone,np,outcome_walk_alpha=1.)
        independent_high=_physical_qualities(
            m,np,outcome_walk_alpha=1.)
        self.assertAlmostEqual(float(independent_low[0][0]),
                               float(independent_high[0][0]))
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

