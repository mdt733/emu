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

def test_cli_accepts_considered_step_lesson(capsys):
    """Prevent a valid reward/runner lesson being blocked by argparse again."""
    from emu.cli import main
    try:
        main(["train", "--help"])
    except SystemExit as exc:
        assert exc.code == 0
    else:
        raise AssertionError("train --help should exit successfully")
    assert "coherent-careful" in capsys.readouterr().out

def test_exact_fifteen_degree_exam_boundary_without_mjx():
    """An exact single-axis 15° packet must not fail float32 roundoff."""
    from emu.environment import EmuMJXEnv
    from emu.contracts import UNIFIED_TERRAIN_FIELDS
    # params_array needs only its data contract; no MuJoCo/JAX model.
    env=object.__new__(EmuMJXEnv)
    env.jnp=np
    env.sensor_noise_scale=0.
    env.terrain_step_mocap_id=0
    env.terrain_step_max_m=.04
    env.unified_capability_focus=True
    for pitch,roll in ((15.,0.),(-15.,0.),(0.,15.),(0.,-15.)):
        packet=dict(terrain_pitch_deg=pitch,terrain_roll_deg=roll)
        arr=env.params_array(packet)
        assert arr.size > 0
    # The cap remains a physical cap, not a blanket 15-degree permission.
    for pitch,roll in ((15.01,0.),(0.,-15.01),(12.,12.)):
        try:
            env.params_array(dict(terrain_pitch_deg=pitch,terrain_roll_deg=roll))
        except ValueError as exc:
            assert "15-degree exam cap" in str(exc)
        else:
            raise AssertionError(f"unsafe total grade accepted: {pitch}/{roll}")
