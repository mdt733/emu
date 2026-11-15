"""Core stair physics and gait-payment contracts (former v5051 investigation)."""

from __future__ import annotations
import unittest
import numpy as np
from emu.rewards import _physical_qualities


class CoreStairPhysicsTests(unittest.TestCase):

    def test_four_world_actual_mjx_science_update(self):
        # Pure, deterministic examiner gate tests inside existing core case:
        # do NOT add a new permanent test or another physics simulator.
        from emu.d_science_exam import _first5, _run5, _moving_window_good
        self.assertFalse(_run5([0,1,1,1,0,1,1,1,1]))
        self.assertTrue(_run5([0,1,1,1,1,1]))
        self.assertEqual(_first5([0,1,1,1,1,1]),5)
        self.assertEqual(_first5([1,1,1,0,1,1]),-1)
        self.assertEqual(_moving_window_good(
            np.full(25,.12),np.full(25,.12)),19)
        self.assertEqual(_moving_window_good(
            np.full(25,.0),np.full(25,.12)),-1)
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
            two_lip_continuation_school=True,
            # No obsolete D-factor shaping: match the new A/B physics.
            stair_tutor_mode='escrow',  # compile actual 8-prism escrow branch
            solver_iterations=2,solver_ls_iterations=4,
            science_gate_mode="zero-only",
            science_phase_lowcmd_scale=.25,
            science_phase_sync_strength=0.,
            science_crouch_allowance_m=.03,
            science_stride_efficiency_weight=.08,
            science_step_goal_weight=0.,  # D disallows first-lip goal jackpot
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
        self.assertIn('tutor_escrow_release_new',out.metrics)
        self.assertIn('tutor_pre_root_plant_new',out.metrics)
        self.assertEqual(out.state.stair_tutor_recovery_streak.shape,(4,))
        self.assertEqual(out.state.stair_tutor_escrow_paid.shape,(4,))
        self.assertTrue(np.isfinite(np.asarray(jax.device_get(
            out.metrics['tutor_escrow_release_new']))).all())
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
        self.assertIn("academy_two_lip_continued_new",out.metrics)
        self.assertIn("academy_first_lip_supported_at_end",out.metrics)
        self.assertIn("academy_two_lip_crossed_at_end",out.metrics)
        self.assertIn("academy_intro_lesson_new",out.metrics)
        self.assertIn("academy_second_bilateral_at_end",out.metrics)
        self.assertIn("academy_posttransfer_50cm_at_end",out.metrics)
        self.assertIn("academy_vx_window_at_end",out.metrics)
        self.assertEqual(out.state.academy_second_transfer_seen.shape,(4,))
        # Never award the C terminal on birth / first control step.
        self.assertTrue(np.all(np.asarray(jax.device_get(
            out.metrics['academy_two_lip_continued_new']))==0.))
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

    def _assert_stair_tutor_staged_credit_and_escrow(self):
        """No lunge reward, true top-first geometry, once-only recovery."""
        from emu.step_goal import (stair_tutor_foot_progress,
                                   stair_tutor_recovery,academy_event_payment)
        # [UP Academy, DOWN Academy, mixed second UP, ordinary STEP].
        # Preserve non-UP event credit in escrow mode, not just the
        # ordinary +10 STEP-crossing bonus calculated elsewhere.
        old=np.array([3.,13.,10.,3.,10.],np.float32)
        up=np.array([True,False,True,False,False])
        expected=np.array([0.,13.,0.,3.,10.],np.float32)
        np.testing.assert_array_equal(
            academy_event_payment(np,old,"legacy",up),old)
        np.testing.assert_array_equal(
            academy_event_payment(np,old,"escrow",up),expected)
        with self.assertRaises(ValueError):
            academy_event_payment(np,old,"unknown",up)
        zeros=np.zeros((1,6),np.float32)
        body=np.array([[.80,.50]],np.float32)
        edge=np.array([1.],np.float32)
        lip=np.array([.05],np.float32)
        top=np.array([[.10,.10]],np.float32)
        foot=np.array([[[.92,.48,.14],[.87,.53,.13]]],np.float32)
        radius=np.array([.012,.012],np.float32)
        bottom=np.array([[.13,.12]],np.float32)
        live=np.array([True])
        # Below the actual upstairs surface: cannot be paid for crossing.
        b0,p0,g0=stair_tutor_foot_progress(
            np,zeros,body,foot,bottom,edge,top,lip,radius,live)
        self.assertEqual(float(b0[0,3]),0.)
        self.assertEqual(float(b0[0,4]),0.)
        self.assertGreater(float(p0[0]),0.)  # small approach only
        # Lifting near the edge earns bounded teaching credit once.
        raised=foot.copy()
        raised[0,0,2]=.18
        bottom2=bottom.copy()
        bottom2[0,0]=.17
        b1,p1,_=stair_tutor_foot_progress(
            np,b0,body,raised,bottom2,edge,top,lip,radius,live)
        self.assertGreater(float(p1[0]),0.)
        landed=raised.copy()
        landed[0,0,0]=1.052
        b2,p2,_=stair_tutor_foot_progress(
            np,b1,body,landed,bottom2,edge,top,lip,radius,live)
        self.assertGreater(float(b2[0,3]),.99)
        self.assertGreater(float(p2[0]),0.)
        _,again,_=stair_tutor_foot_progress(
            np,b2,body,landed,bottom2,edge,top,lip,radius,live)
        self.assertEqual(float(again[0]),0.)
        _,disabled,_=stair_tutor_foot_progress(
            np,zeros,body,landed,bottom2,edge,top,lip,radius,
            np.array([False]))
        self.assertEqual(float(disabled[0]),0.)
        # Escrow is *not* the maximum reached by a pitching chassis:
        # no 2-foot physical support, no payment even at x=1.6.
        def recover(x=1.2, both=True, top_now=True, pitch=.05,
                    pitch_rate=.08, vx=.18, bad=0., streak=0, paid=False):
            return stair_tutor_recovery(
                np,body_xy=np.array([[x,.50]]),edge_x=edge,
                transfer_seen=np.array([both]),
                upper_top_now=np.array([[top_now,False]]),
                pitch_rad=np.array([pitch]),
                pitch_rate_rad_s=np.array([pitch_rate]),
                gravity_z=np.array([.99]),
                support=np.array([.9]),
                bad_contact=np.array([bad]),
                vx_samples=np.array([20]),
                mean_vx=np.array([vx]),
                mean_cmd_vx=np.array([.18]),
                previous_streak=np.array([streak]),
                previous_paid=np.array([paid]),
                eligible=np.array([True]))
        for kw in (dict(both=False),dict(top_now=False),
                   dict(pitch=.6),dict(pitch_rate=1.5),
                   dict(vx=0.),dict(bad=1.),
                   dict(x=.99),dict(x=1.82)):
            st,was_paid,payment,released=recover(**kw)
            self.assertEqual(int(st[0]),0.)
            self.assertFalse(bool(released[0]))
            self.assertEqual(float(payment[0]),0.)
        # Supported, moving, post-foot-contact recovery for 0.4 seconds.
        st=0
        for _ in range(19):
            new_st,paid,amt,release=recover(streak=st)
            st=int(new_st[0])
            self.assertFalse(bool(release[0]))
            self.assertEqual(float(amt[0]),0.)
        st,paid,amt,release=recover(streak=st)
        self.assertTrue(bool(release[0]))
        self.assertGreater(float(amt[0]),0.)
        self.assertTrue(bool(paid[0]))
        _,_,repeat,released_again=recover(streak=int(st[0]),paid=True)
        self.assertEqual(float(repeat[0]),0.)
        self.assertFalse(bool(released_again[0]))
        # Vectorisation contract holds for JAX too (same pure maths).
        import jax.numpy as jnp
        np.testing.assert_array_equal(
            np.asarray(academy_event_payment(
                jnp,jnp.asarray(old),"escrow",jnp.asarray(up))),expected)
        bj,pj,_=stair_tutor_foot_progress(
            jnp,jnp.asarray(b1),jnp.asarray(body),jnp.asarray(landed),
            jnp.asarray(bottom2),jnp.asarray(edge),jnp.asarray(top),
            jnp.asarray(lip),jnp.asarray(radius),jnp.asarray(live))
        np.testing.assert_allclose(np.asarray(bj),b2,rtol=1e-5)
        self.assertGreater(float(pj[0]),0.)

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
        self._assert_stair_tutor_staged_credit_and_escrow()
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

