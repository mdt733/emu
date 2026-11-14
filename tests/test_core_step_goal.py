"""Live Academy destination safety: signed progress, real contact and arrival."""

from __future__ import annotations
import unittest
import numpy as np
from emu.rewards import REWARD_ONLY_METRICS, training_reward
from emu.terrain_carpet_mjx import (
    TILE_M, PROFILE_HALF, academy_near_lip_spawn_x,
    two_lip_continuation_profiles)
from emu.step_goal import (destination_from_birth, signed_distance_progress,
                           arrival_quality, stable_arrival, GOAL_STABLE_FRAMES,
                           short_climb_quality, CLIMB_AFTER_LIP_M, CLIMB_STABLE_FRAMES,
                           two_lip_continuation_success)
from emu.step_autopsy import inspect_destinations


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
        science_step_goal_delta_m=(vx*.02),
    )
    return {**{k:np.array([v,v],dtype=np.float32) for k,v in values.items()},
            "science_step_goal_active":np.array([float(active),0.],dtype=np.float32)}


class CoreAcademyTests(unittest.TestCase):

    def test_reward_goal_and_velocity_relief_are_academy_only(self):
        # C continuation regression proof shares this existing core test:
        first=np.asarray([.050,-.020],np.float32)
        second=np.asarray([.050,+.050],np.float32)
        x,y=two_lip_continuation_profiles(np,first,second)
        self.assertEqual(x.shape,(2,2*PROFILE_HALF))
        np.testing.assert_allclose(x[:,PROFILE_HALF],0,atol=1e-8)
        np.testing.assert_allclose(x[:,PROFILE_HALF+1],first,atol=1e-8)
        np.testing.assert_allclose(x[:,PROFILE_HALF+2],first+second,atol=1e-8)
        np.testing.assert_allclose(x[:,PROFILE_HALF+3],first+second,atol=1e-8)
        self.assertTrue(np.all(y==0))
        def qualify(lips=2,first_supported=True,second_supported=True,
                    follow=True,required_lips=2,
                    samples=25,vx=.16,command=.16,
                    posture=1.,contact=1.,yaw=0.,fell=False):
            return bool(two_lip_continuation_success(np,
                academy_active=np.array([True]),
                supported_lips=np.array([lips]),
                first_lip_two_feet=np.array([first_supported]),
                second_lip_two_feet=np.array([second_supported]),
                required_lips=required_lips,
                half_metre_after_second=np.array([follow]),
                samples=np.array([samples]),mean_vx=np.array([vx]),
                mean_cmd_vx=np.array([command]),gravity_z=np.array([posture]),
                support=np.array([contact]),bad_contact=np.array([0.]),
                yaw_error=np.array([yaw]),terminal=np.array([fell]))[0])
        self.assertTrue(qualify())
        self.assertFalse(qualify(lips=1),'first lip cannot finish course')
        self.assertFalse(qualify(first_supported=False),
                         'first step needs BOTH feet genuinely transferred')
        self.assertFalse(qualify(second_supported=False),
                         'second step also needs BOTH genuine top contacts')
        self.assertTrue(qualify(lips=1,required_lips=1),
                        'coached intro may end after a genuine first lip plus 50cm')
        self.assertFalse(qualify(lips=0,required_lips=1),
                         'intro cannot pass without any real lip')
        self.assertFalse(qualify(follow=False),'second lip alone cannot finish')
        self.assertFalse(qualify(samples=5),'one quick dive is not walking')
        self.assertFalse(qualify(vx=0.),'stationary after transfer')
        self.assertFalse(qualify(vx=.40),'ballistic overspeed is not vx tracking')
        self.assertFalse(qualify(contact=0.),'unsupported landing')
        self.assertFalse(qualify(posture=.5),'nose dive')
        self.assertFalse(qualify(yaw=.6),'turning away')
        self.assertFalse(qualify(fell=True),'a fall can never be success')


        m=metrics(vx=.04)
        m['science_step_near_lip_frac']=np.asarray([1.,0.],np.float32)
        reg=np.zeros((2,),np.float32)
        original,_=training_reward(m,reg,xp=np)
        relaxed,_=training_reward(m,reg,xp=np,science_step_velocity_relief=.8)
        self.assertGreater(relaxed[0],original[0])
        self.assertAlmostEqual(float(relaxed[1]),float(original[1]),places=6)
        self.assertIn('science_step_near_lip_frac',REWARD_ONLY_METRICS)
        self.assertIn('science_step_phase_free_frac',REWARD_ONLY_METRICS)
        phase_metrics=metrics(vx=.12)
        phase_metrics['phase_sync']=np.ones(2,dtype=np.float32)
        phase_metrics['organic_step_shape']=np.full(2,.65,dtype=np.float32)
        phase_metrics['science_step_phase_free_frac']=np.array([1.,0.],np.float32)
        free,_=training_reward(phase_metrics,reg,xp=np)
        coached,_=training_reward(metrics(vx=.12,active=True),reg,xp=np)
        self.assertTrue(np.isfinite(free).all())

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
        backward=metrics(vx=-.08)
        backward_pay,terms=training_reward(
            backward,regime,xp=np,science_step_goal_weight=.30)
        backward_base,_=training_reward(backward,regime,xp=np)
        self.assertLess(backward_pay[0],backward_base[0])
        self.assertAlmostEqual(float(backward_pay[1]),float(backward_base[1]),places=6)
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

    def _assert_academy_birth_is_exactly_upstream_of_real_prism_seam(self):
        starts=np.array([-.02,.02,TILE_M*.5,TILE_M+.05],dtype=np.float32)
        near=academy_near_lip_spawn_x(np,starts,.22)
        np.testing.assert_allclose(near,[-.22,TILE_M-.22,TILE_M-.22,2*TILE_M-.22],atol=1e-6)
        # A robot that begins 22cm before a tile edge should always see
        # the boundary ahead without changing actual prism heights.
        np.testing.assert_allclose(
            (np.floor(near/TILE_M)+1)*TILE_M-near,
            np.full_like(near,.22),atol=1e-6)
        np.testing.assert_array_equal(
            academy_near_lip_spawn_x(np,starts,0.),starts)

    def test_fixed_destination_never_chases_next_tile(self):
        self._assert_academy_birth_is_exactly_upstream_of_real_prism_seam()
        start=np.array([[TILE_M-.22,.10],[-.22,-.06]],dtype=np.float32)
        goal=destination_from_birth(np,start)
        np.testing.assert_allclose(goal,[[TILE_M+.60,.10],[.60,-.06]],atol=1e-6)
        # Crossing a tile boundary does not redefine the goal.
        pre=np.array([[TILE_M-.05,.10],[-.10,-.06]],dtype=np.float32)
        post=np.array([[TILE_M+.03,.10],[.02,-.06]],dtype=np.float32)
        delta=signed_distance_progress(np,goal,pre,post)
        np.testing.assert_allclose(delta,[.02,.02],atol=1e-6)
        backwards=signed_distance_progress(np,goal,post,pre)
        np.testing.assert_allclose(backwards,[-.02,-.02],atol=1e-6)
        np.testing.assert_allclose(goal,destination_from_birth(np,start))

    def test_goal_requires_actual_bilateral_transfer_and_stable_arrival(self):
        goal=np.array([[TILE_M+.60,0.]])
        point=np.array([[TILE_M+.60,0.]])
        q=lambda transfer,slip=0.,yaw=0.:arrival_quality(
            np,goal,point,np.array([yaw]),np.ones(1),
            np.ones(1),np.zeros(1),np.full(1,slip),
            np.array([transfer]))
        self.assertFalse(q(False)[0])
        self.assertFalse(q(True,slip=.2)[0])
        self.assertFalse(q(True,yaw=.5)[0])
        streak=np.zeros(1,dtype=np.int32)
        completed=np.array([False])
        for i in range(GOAL_STABLE_FRAMES-1):
            streak,new=stable_arrival(np,q(True),streak,completed)
            self.assertFalse(new[0])
        streak,new=stable_arrival(np,q(True),streak,completed)
        self.assertTrue(new[0])
        completed=completed|new
        streak,replayed=stable_arrival(np,q(True),streak,completed)
        self.assertFalse(replayed[0])
        # One frame of unsafe/unsupported contact resets the streak.
        reset,_=stable_arrival(np,q(False),streak,completed)
        self.assertEqual(int(reset[0]),0)

        # The nursery succeeds early, but never on one foot, flat ground,
        # a fall, a side escape, a U-turn or an unconfirmed real prism.
        seam=goal[0,0]-.60
        xyz=np.array([[seam+CLIMB_AFTER_LIP_M+.02,0.]],dtype=np.float32)
        def climb(*,feet=True,rise=.02,grav=1.,yaw=0.,bad=0.,side=0.):
            pos=xyz.copy()
            pos[:,1]=side
            return short_climb_quality(
                np,goal,pos,np.asarray([yaw]),np.asarray([grav]),
                np.ones(1),np.asarray([bad]),np.asarray([feet]),
                np.asarray([rise]))
        self.assertTrue(climb()[0])
        self.assertTrue(climb(rise=-.02)[0])
        self.assertFalse(climb(feet=False)[0])
        self.assertFalse(climb(rise=0.)[0])
        self.assertFalse(climb(grav=.4)[0])
        self.assertFalse(climb(bad=1.)[0])
        self.assertFalse(climb(yaw=1.)[0])
        self.assertFalse(climb(side=.30)[0])
        self.assertEqual(CLIMB_STABLE_FRAMES,3)

        # Genuine completed nursery resets continue into the NEXT birth.
        # The old absorbing terminal would have assigned negative credit
        # (8-50=-42); the correct transition uses 8+.99*50-50=+7.5.
        # The done mask *still* cuts GAE, including across a goal reset.
        from emu.ppo import (stair_terminal_credit_stats,
                             nursery_bootstrap_transition)
        old_obs=np.arange(4.,dtype=np.float32)[:,None]
        reset_obs=old_obs+100.
        fall=np.array([1.,0.,0.,0.],dtype=np.float32)
        finish=np.array([0.,1.,0.,1.],dtype=np.float32)
        terminal, boot=nursery_bootstrap_transition(
            np,old_obs,reset_obs,fall,finish)
        np.testing.assert_array_equal(terminal,fall)
        np.testing.assert_array_equal(boot[:,0],[0.,101.,2.,103.])
        credit=stair_terminal_credit_stats(
            np,np.array([7.5,2.,3.,4.]),
            np.array([8.,0.,0.,0.]),
            np.array([50.,0.,0.,0.]),
            np.array([1.,0.,0.,0.]),
            np.array([50.,0.,0.,0.]),np.zeros(4))
        self.assertEqual(float(credit['stair_credit_finish_events']),1.)
        self.assertEqual(float(credit['stair_credit_finish_reward']),8.)
        self.assertEqual(float(credit['stair_credit_finish_bootstrap_value']),50.)
        self.assertAlmostEqual(float(credit['stair_credit_finish_td']),7.5)
        self.assertEqual(float(credit['stair_credit_finish_adv_raw']),7.5)
        self.assertGreater(float(credit['stair_credit_finish_adv_normalised']),0.)
        self.assertEqual(float(credit['stair_credit_finish_negative_fraction']),0.)
        # Not every successful reset must give positive advantage if the
        # critic's new-birth value is lower. Keep the negative-credit audit.
        negative=stair_terminal_credit_stats(
            np,np.array([-42.,2.,3.,4.]),
            np.array([8.,0.,0.,0.]),np.array([50.,0.,0.,0.]),
            np.array([1.,0.,0.,0.]),np.zeros(4),np.zeros(4))
        self.assertEqual(float(negative['stair_credit_finish_td']),-42.)
        self.assertEqual(float(negative['stair_credit_finish_negative_fraction']),1.)
        empty=stair_terminal_credit_stats(
            np,np.array([1.,2.]),np.array([0.,0.]),
            np.array([0.,0.]),np.zeros(2),np.zeros(2),np.zeros(2))
        self.assertEqual(float(empty['stair_credit_finish_events']),0.)
        self.assertEqual(float(empty['stair_credit_finish_adv_raw']),-1.)

        # A logged row represents only one of ten PPO updates. Evidence
        # must not disappear because the successful update was not logged.
        from emu.results import StairCreditAudit
        audit=StairCreditAudit()
        audit.observe({
            'academy_episodes':3.,'academy_episodes_first_lip_armed':2.,
            'academy_episodes_two_foot_ready':1.,
            'academy_short_climb_successes':1.,
            'nursery_reward_sum':9.,
            'nursery_progress_approach_sum':2.,
            'nursery_progress_r_lift_sum':1.,
            'nursery_progress_l_lift_sum':.5,
            'nursery_progress_r_over_sum':.25,
            'nursery_progress_l_over_sum':0.,
            'nursery_progress_chassis_sum':.1,
            'academy_active_fraction':.5,
            'academy_actual_cmd_vx_mps':.1,
            'academy_actual_vx_mps':.08,
        }, {
            'stair_credit_finish_events':1.,
            'stair_credit_finish_reward':8.,'stair_credit_finish_value':50.,
            'stair_credit_finish_bootstrap_value':0.,
            'stair_credit_finish_td':-42.,
            'stair_credit_finish_adv_raw':-42.,
            'stair_credit_finish_adv_normalised':-1.5,
            'stair_credit_finish_negative_fraction':1.,
        })
        audit.observe({
            'academy_episodes':5.,'academy_episodes_first_lip_armed':3.,
            'academy_episodes_two_foot_ready':2.,
            'academy_short_climb_successes':0.,
            'academy_active_fraction':.25,
            'academy_actual_cmd_vx_mps':.2,
            'academy_actual_vx_mps':.16,
        },{'stair_credit_finish_events':0.})
        audit.observe({
            'academy_episodes':4.,'academy_episodes_first_lip_armed':3.,
            'academy_episodes_two_foot_ready':2.,
            'academy_short_climb_successes':2.,
            'academy_active_fraction':.25,
            'academy_actual_cmd_vx_mps':.2,
            'academy_actual_vx_mps':.16,
        },{
            'stair_credit_finish_events':2.,
            'stair_credit_finish_reward':10.,'stair_credit_finish_value':5.,
            'stair_credit_finish_bootstrap_value':0.,
            'stair_credit_finish_td':5.,
            'stair_credit_finish_adv_raw':4.,
            'stair_credit_finish_adv_normalised':.5,
            'stair_credit_finish_negative_fraction':0.,
        })
        window=audit.summary()
        self.assertEqual(window['ppo_stair_audit_rollouts_since_process_start'],3.)
        self.assertEqual(window['ppo_stair_audit_finish_events'],3.)
        self.assertEqual(window['ppo_stair_audit_academy_episodes'],12.)
        self.assertEqual(window['ppo_stair_audit_academy_episodes_first_lip_armed'],8.)
        self.assertEqual(window['ppo_stair_audit_academy_episodes_two_foot_ready'],5.)
        self.assertEqual(window['ppo_stair_audit_nursery_reward_sum'],9.)
        self.assertEqual(window['ppo_stair_audit_nursery_progress_approach_sum'],2.)
        self.assertEqual(window['ppo_stair_audit_nursery_progress_r_lift_sum'],1.)
        self.assertAlmostEqual(window['ppo_stair_audit_nursery_progress_chassis_sum'],.1)
        self.assertAlmostEqual(window['ppo_stair_audit_finish_adv_normalised'],-1./6.)
        self.assertAlmostEqual(window['ppo_stair_audit_finish_negative_fraction'],1./3.)
        self.assertAlmostEqual(window['ppo_stair_audit_academy_cmd_vx_mps'],.15)
        self.assertAlmostEqual(window['ppo_stair_audit_academy_vx_mps'],.12)

        from emu.ppo import hip_roll_action_saturation
        # Two Academy frames saturate ONLY left roll (+, +);
        # two generalist frames saturate ONLY right roll (-, -).
        a=np.zeros((2,2,6),dtype=np.float32)
        a[0,:,3]=[1.3,1.6]
        a[1,:,0]=[-1.2,-1.4]
        cohorts=np.array([[1.,1.],[0.,0.]],dtype=np.float32)
        hip=hip_roll_action_saturation(np,a,cohorts)
        self.assertEqual(float(hip['hip_roll_clip_left_academy']),1.)
        self.assertEqual(float(hip['hip_roll_clip_left_other']),0.)
        self.assertEqual(float(hip['hip_roll_clip_right_academy']),0.)
        self.assertEqual(float(hip['hip_roll_clip_right_other']),1.)
        self.assertAlmostEqual(float(hip['hip_roll_raw_left_academy']),1.45,places=6)
        self.assertAlmostEqual(float(hip['hip_roll_raw_right_other']),-1.3,places=6)

        self.assertEqual(float(hip['hip_roll_frames_academy']),2.)
        self.assertEqual(float(hip['hip_roll_frames_other']),2.)
        hip_audit=StairCreditAudit()
        hip_audit.observe({'academy_active_fraction':0.},
            {'stair_credit_finish_events':0., **hip})
        aggregate=hip_audit.summary()
        self.assertEqual(aggregate['ppo_stair_audit_hip_roll_frames_academy'],2.)
        self.assertEqual(aggregate['ppo_stair_audit_hip_roll_frames_other'],2.)
        self.assertEqual(aggregate['ppo_stair_audit_hip_roll_clip_left_academy'],1.)
        self.assertEqual(aggregate['ppo_stair_audit_hip_roll_clip_left_other'],0.)

        from dataclasses import replace
        from emu.training_config import TrainingConfig
        from emu.training import compatible_academy_evidence_recipe
        parent=replace(TrainingConfig(),
            science_step_finish_on_transfer=True,
            academy_stream=True, science_step_goal_weight=.3,
            science_step_approach_m=.22, terrain_step_max_m=.08,
            terrain_carpet=True, terrain_carpet_prisms=True,
            terrain_lesson='coherent-careful',
            solver_iterations=2, solver_ls_iterations=4)
        same=replace(parent,run_dir='diagnostic-fork',
                     steps=190_000_000,resume='parent/final.pkl')
        self.assertTrue(compatible_academy_evidence_recipe(parent,same))
        self.assertFalse(compatible_academy_evidence_recipe(
            replace(parent,science_step_finish_on_transfer=False),same))
        self.assertFalse(compatible_academy_evidence_recipe(
            parent,replace(same,science_step_approach_m=.18)))
        self.assertFalse(compatible_academy_evidence_recipe(
            parent,replace(same,solver_iterations=1)))

    def test_independent_real_prism_trace_proves_first_lip_destination(self):
        # No teacher success, no reward fields, and no nominal height!
        # The scorer sees only raw physical top contacts + first-life motion.
        n=80
        xy=np.zeros((n,1,2),dtype=np.float64)
        xy[:,:,1]=TILE_M*.5
        xy[:10,:,0]=TILE_M-.22
        xy[10:35,:,0]=np.linspace(TILE_M+.05,TILE_M+.60,25)[:,None]
        xy[35:,:,0]=TILE_M+.60
        cells=np.zeros((n,1,2),dtype=np.int32)
        cells[10:,:,0]=1
        tops=np.zeros((n,1,4,4))
        tops[:,:,1,:]=.02  # measured 2cm positive-X lip
        feet=np.zeros((n,1,2,3))
        feet[:,:,:,0]=TILE_M+.10
        feet[:,:,:,1]=TILE_M*.5
        feet[:,:,:,2]=.038  # 18mm foot ball resting on real 20mm top
        pairs=np.zeros((n,1,8,2),dtype=bool)
        vertical=np.zeros((n,1,8,2))
        pairs[10:,:,2:4,0]=True
        vertical[10:,:,2:4,0]=1.
        pairs[20:,:,2:4,1]=True
        vertical[20:,:,2:4,1]=1.
        trace=dict(position_xy=xy,
                   physical_position_xy_after_step=xy,
                   heading_rad=np.zeros((n,1)),
                   physical_heading_rad_after_step=np.zeros((n,1)),
                   carpet_root_cell=cells,
                   carpet_prism_corner_z=tops,
                   carpet_raw_contact_pairs=pairs,
                   carpet_contact_vertical=vertical,
                   foot_xyz=feet,initial_foot_radii_m=np.array([.018,.018]),
                   done=np.zeros((n,1),dtype=bool),
                   fell=np.zeros((n,1),dtype=bool),
                   terrain_gravity_z=np.ones((n,1)),
                   terrain_support=np.ones((n,1)),
                   terrain_bad_contact=np.zeros((n,1)),
                   stance_slip_proxy=np.zeros((n,1)))
        successful=inspect_destinations(trace)
        self.assertEqual(successful["completed_missions"],1)
        self.assertEqual(successful["uphill_completed"],1)
        # First contact but no second foot is NOT a completed climb.
        missing=dict(trace,carpet_raw_contact_pairs=pairs.copy())
        missing["carpet_raw_contact_pairs"][:,:,2:4,1]=False
        self.assertEqual(inspect_destinations(missing)["completed_missions"],0)
        # Same contacts but no travel to the point is NOT a completion.
        stalled=dict(trace,physical_position_xy_after_step=xy.copy())
        stalled["physical_position_xy_after_step"][30:,:,0]=TILE_M+.10
        self.assertEqual(inspect_destinations(stalled)["completed_missions"],0)


if __name__=="__main__":

    unittest.main()

