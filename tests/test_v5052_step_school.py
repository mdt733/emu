"""Live Academy destination safety: signed progress, real contact and arrival."""

from __future__ import annotations
import unittest
import numpy as np
from emu.rewards import REWARD_ONLY_METRICS, training_reward
from emu.terrain_carpet_mjx import TILE_M, academy_near_lip_spawn_x
from emu.step_goal import (destination_from_birth, signed_distance_progress,
                           arrival_quality, stable_arrival, GOAL_STABLE_FRAMES)
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

    def test_fixed_destination_never_chases_next_tile(self):
        start=np.array([[.53,.10],[-.22,-.06]],dtype=np.float32)
        goal=destination_from_birth(np,start)
        np.testing.assert_allclose(goal,[[1.35,.10],[.60,-.06]],atol=1e-6)
        # Crossing a tile boundary does not redefine the goal.
        pre=np.array([[.70,.10],[-.10,-.06]],dtype=np.float32)
        post=np.array([[.78,.10],[.02,-.06]],dtype=np.float32)
        delta=signed_distance_progress(np,goal,pre,post)
        np.testing.assert_allclose(delta,[.02,.02],atol=1e-6)
        backwards=signed_distance_progress(np,goal,post,pre)
        np.testing.assert_allclose(backwards,[-.02,-.02],atol=1e-6)
        np.testing.assert_allclose(goal,destination_from_birth(np,start))

    def test_goal_requires_actual_bilateral_transfer_and_stable_arrival(self):
        goal=np.array([[1.35,0.]])
        point=np.array([[1.35,0.]])
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

    def test_independent_real_prism_trace_proves_first_lip_destination(self):
        # No teacher success, no reward fields, and no nominal height!
        # The scorer sees only raw physical top contacts + first-life motion.
        n=80
        xy=np.zeros((n,1,2),dtype=np.float64)
        xy[:,:,1]=.375
        xy[:10,:,0]=.53
        xy[10:35,:,0]=np.linspace(.80,1.35,25)[:,None]
        xy[35:,:,0]=1.35
        cells=np.zeros((n,1,2),dtype=np.int32)
        cells[10:,:,0]=1
        tops=np.zeros((n,1,4,4))
        tops[:,:,1,:]=.02  # measured 2cm positive-X lip
        feet=np.zeros((n,1,2,3))
        feet[:,:,:,0]=.85
        feet[:,:,:,1]=.375
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
        stalled["physical_position_xy_after_step"][30:,:,0]=.85
        self.assertEqual(inspect_destinations(stalled)["completed_missions"],0)


if __name__=="__main__":

    unittest.main()

