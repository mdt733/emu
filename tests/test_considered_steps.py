"""Tiny reward-contract checks for the v5028 STEP-only treatment.

Full 12-world MJX CPU and 2048-world CUDA PPO acceptance is performed
by the launcher's own smoke preflight; no synthetic gait claims here.
"""
from __future__ import annotations
import numpy as np

from emu.rewards import _command_tracking


def packets(actual, requested=0.12):
    z=np.zeros((1,),dtype=np.float32)
    return dict(vx=np.array([actual],dtype=np.float32),
                cmd_vx=np.array([requested],dtype=np.float32),
                vy=z,cmd_vy=z,yaw_rate=z,cmd_yaw=z,
                body_pitch=z,cmd_pitch=z,
                ramp_frac=np.ones_like(z),age_s=np.ones_like(z))


def score(m,flag=None):
    return float(_command_tracking(
        m,np.zeros((1,),dtype=np.float32),
        np.zeros((1,),dtype=np.float32),np,careful_step=flag)[0])


def test_step_relief_only_underspeed():
    slow=packets(0.0)
    original=score(slow)
    assert 0 < original < 1
    assert score(slow,np.zeros((1,),dtype=np.float32)) == original
    assert score(slow,np.ones((1,),dtype=np.float32)) > original

    for actual in (0.12,0.20):
        m=packets(actual)
        assert score(m,np.ones((1,),dtype=np.float32)) == score(m)

    # The other velocities retain their penalties even inside relief.
    drifting=packets(0.0)
    drifting['vy']=np.array([0.2],dtype=np.float32)
    assert score(drifting,np.ones((1,),dtype=np.float32)) < score(
        slow,np.ones((1,),dtype=np.float32))
