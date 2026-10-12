import unittest
import numpy as np
from emu.tracking_contract import (
    node_window_success_numpy, recovery_outcome_numpy, FRONTIER_VX_ABS_TOL, FRONTIER_VX_REL_TOL,
    FRONTIER_VY_ABS_TOL, FRONTIER_VY_REL_TOL, FRONTIER_YAW_ABS_TOL,
    FRONTIER_YAW_REL_TOL, FRONTIER_EDGE_FRAC, FRONTIER_SETTLE_S,
)
from emu.rollout_metrics import frontier_tracking_metrics

class TrackingContractTests(unittest.TestCase):
    def test_mean_absolute_error_does_not_cancel_oscillation(self):
        n=50; cmd=np.zeros(n,np.float32); zero=np.zeros(n,np.float32); one=np.ones(n,np.float32)
        vx=np.tile(np.asarray([.12,-.12],np.float32),n//2)
        ok=node_window_success_numpy(cmd,cmd,cmd,vx,zero,zero,one,one,zero)
        self.assertFalse(bool(ok))



    def test_recovery_outcome_requires_survival(self):
        ff=np.asarray([-1,-1,5,7],np.int32)
        fr=np.asarray([10,-1,3,6],np.int32)
        out=recovery_outcome_numpy(ff,fr,not_before_steps=2,direction_bins=2)
        self.assertEqual(out['survival'],.5)
        self.assertEqual(out['recovery_rate'],.25)
        self.assertAlmostEqual(out['recovery_time_p90_s'],.16)
        self.assertEqual(out['recovery_worst_direction'],0.0)

    def test_frontier_measurement_contract_executes(self):
        # Exercise the real physical-frontier reduction path so stale or
        # missing measurement-contract values cannot hide behind simulator-only code.
        self.assertEqual((FRONTIER_VX_ABS_TOL, FRONTIER_VX_REL_TOL), (.05, .12))
        self.assertEqual((FRONTIER_VY_ABS_TOL, FRONTIER_VY_REL_TOL), (.035, .18))
        self.assertEqual((FRONTIER_YAW_ABS_TOL, FRONTIER_YAW_REL_TOL), (.12, .12))
        self.assertEqual((FRONTIER_EDGE_FRAC, FRONTIER_SETTLE_S), (.72, .35))
        shape=(32,4); z=np.zeros(shape,np.float32); vx=np.full(shape,.60,np.float32)
        metrics={
            'vx':vx,'vy':z,'yaw_rate':z,'cmd_vx':vx.copy(),'cmd_vy':z.copy(),'cmd_yaw':z.copy(),
            'regime_steady_frontier':np.ones(shape,np.float32),
            'age_s':np.full(shape,1.0,np.float32),'transition_frac':z.copy(),
        }
        out=frontier_tracking_metrics(metrics,None)
        self.assertEqual(out['precision_vx'],1.0)
        self.assertEqual(out['frontier_precision_vx_forward'],1.0)

    def test_healthy_window_passes(self):
        n=50; cvx=np.full(n,.2,np.float32); zero=np.zeros(n,np.float32); one=np.ones(n,np.float32)
        vx=cvx+np.tile(np.asarray([.03,-.03],np.float32),n//2)
        ok=node_window_success_numpy(cvx,zero,zero,vx,zero,zero,one,one,zero)
        self.assertTrue(bool(ok))

if __name__=='__main__': unittest.main()
