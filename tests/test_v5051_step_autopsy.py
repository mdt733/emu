"""High-value deterministic v5051 examiner checks, no GPU or PPO needed."""
from __future__ import annotations

import unittest
import numpy as np

from emu.step_autopsy import inspect_missions
from scripts.v5051_science_battery import wilson, aggregate


def trace(*, second_foot=True, continued=True, heading=0.):
    n=100
    cells=np.zeros((n,1,2),dtype=np.int32)
    cells[10:,:,0]=1
    top=np.zeros((n,1,4,4))
    top[:,:,1,:]=.02
    pairs=np.zeros((n,1,8,2),dtype=bool)
    vertical=np.zeros((n,1,8,2))
    pairs[10:,:,2:4,0]=True
    vertical[10:,:,2:4,0]=1.
    if second_foot:
        pairs[20:,:,2:4,1]=True
        vertical[20:,:,2:4,1]=1.
    feet=np.zeros((n,1,2,3))
    feet[:,:,:,0]=.85
    feet[:,:,:,1]=.375
    feet[:,:,:,2]=.038
    # The source-to-target crossing at frame 10 is supported for >=5
    # frames; body continues from x=.80 to x=1.35 (if requested).
    xy=np.zeros((n,1,2))
    xy[:,:,0]=.8
    if continued:
        xy[30:,:,0]=np.linspace(.8,1.35,n-30)[:,None]
    yaw=np.full((n,1),heading)
    return dict(
        carpet_root_cell=cells,carpet_prism_corner_z=top,
        carpet_raw_contact_pairs=pairs,
        carpet_contact_vertical=vertical,foot_xyz=feet,
        initial_foot_radii_m=np.array([.018,.018]),
        done=np.zeros((n,1),bool),fell=np.zeros((n,1),bool),
        position_xy=xy,heading_rad=yaw,
    )


class ExamTests(unittest.TestCase):
    def test_completed_followup(self):
        result=inspect_missions(trace(),max_followup_s=4.)
        self.assertEqual(result["eligible_supported_contacts"],1)
        self.assertEqual(result["completed_missions"],1)
        self.assertAlmostEqual(result["events"][0]["signed_actual_height_cm"],2.)
        self.assertEqual(result["events"][0]["detection"],"unmeasured")

    def test_no_trailing_foot(self):
        result=inspect_missions(trace(second_foot=False),max_followup_s=4.)
        self.assertEqual(result["eligible_supported_contacts"],1)
        self.assertEqual(result["completed_missions"],0)
        self.assertEqual(result["events"][0]["verdict"],"trailing_foot_not_verified")

    def test_no_progress(self):
        result=inspect_missions(trace(continued=False),max_followup_s=4.)
        self.assertEqual(result["completed_missions"],0)
        self.assertEqual(result["events"][0]["verdict"],"insufficient_controlled_progress")

    def test_wrong_heading(self):
        result=inspect_missions(trace(heading=np.pi/2),max_followup_s=4.)
        self.assertEqual(result["completed_missions"],0)

    def test_fixed_wall_has_exact_physical_height_and_front(self):
        from scripts.v5051_fixed_wall_screen import make_wall_model
        model, geom = make_wall_model(.03)
        self.assertAlmostEqual(float(model.geom_pos[geom,2] +
                                     model.geom_size[geom,2]), .03, places=7)
        self.assertAlmostEqual(float(model.geom_pos[geom,0] -
                                     model.geom_size[geom,0]), .24, places=7)

    def test_fixed_wall_mujoco_short_rollout(self):
        from scripts.v5051_fixed_wall_screen import (
            make_wall_model, test_swing as wall_trial)
        model, geom = make_wall_model(.02)
        result=wall_trial(model,geom,.02,"left",.2,.4,settle=2,frames=12)
        self.assertEqual(result["actual_height_cm"],2.)
        self.assertLessEqual(result["max_servo_torque_nm"],1.914)
        self.assertLessEqual(result["completed_frames"],12)

    def test_bounded_standing_short_cpu_integration(self):
        import mujoco
        from emu.contracts import ROBOT_XML
        from scripts.v5051_balance_screen import run_case
        model=mujoco.MjModel.from_xml_path(str(ROBOT_XML))
        result=run_case(model,gain=.5,damping=.1,
                        delay_frames=1,noise_rad=.005,seed=735,
                        horizon_s=.20)
        self.assertLessEqual(result["duration_s"],.20)
        self.assertLessEqual(result["max_torque_nm"],1.914)

    def test_wilson_limits_and_empty(self):
        self.assertIsNone(wilson(0,0))
        self.assertEqual(wilson(0,32)[0],0.)
        self.assertEqual(wilson(32,32)[1],1.)
        self.assertLess(wilson(16,32)[0],.5)
        self.assertGreater(wilson(16,32)[1],.5)

    def test_aggregate_never_calls_no_contacts_success(self):
        sample=dict(mode="step",speed=.1,lip_cm=3.,
                    worlds=[dict(survived=True,footfall_rate_hz=3.,
                                 same_foot_stride_mean_m=None,
                                 net_xy_drift_m=0.,
                                 achieved_vx_mps=.1,
                                 shaft_drive_j_per_progress_m=None)],
                    mission_autopsy=dict(eligible_supported_contacts=0,
                                         completed_missions=0))
        summary=aggregate([sample])[0]
        self.assertIsNone(summary["completion_fraction"])
        self.assertEqual(summary["supported_contact_events"],0)


if __name__ == "__main__":
    unittest.main()
