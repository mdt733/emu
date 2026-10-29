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


def test_v5030_causal_half_second_velocity_comparison_preserves_original():
    from emu.rewards import counterfactual_forward_tracking, trailing_mean_trace
    cmd = np.full(80, .40, dtype=np.float64)
    actual = np.full(80, .40, dtype=np.float64)
    actual[31:36] = .0  # brief 0.1 second careful movement
    original = counterfactual_forward_tracking(actual, cmd, xp=np,
                                                history_share=0)
    blended = counterfactual_forward_tracking(actual, cmd, xp=np,
                                               history_share=.5)
    assert np.array_equal(original, counterfactual_forward_tracking(
        actual, cmd, xp=np, frames=25, history_share=0))
    assert blended[33]>original[33]
    assert blended[34]<1.0   # not a free reward for stopping
    assert original[70]==blended[70]==1.0
    np.testing.assert_allclose(trailing_mean_trace(
        np.asarray([1., 3., 8., 4.]), 2, xp=np),
        [1., 2., 5.5, 6.])
    with __import__('pytest').raises(ValueError):
        counterfactual_forward_tracking(actual,cmd[:-1],xp=np)


def test_v5030_verified_support_height_requires_real_landing_and_no_farming():
    from emu.rewards import verified_support_elevation_credit
    def measured(height,com,foot,support,peak,direction=1):
        return verified_support_elevation_credit(
            np.asarray([height]),np.asarray([com]),
            np.asarray([foot]),np.asarray([support]),
            np.asarray([peak]),direction=direction,xp=np)
    credit,peak=measured(0.,.03,True,True,0.)
    assert float(credit[0])==0 and float(peak[0])==0 # squatting on flat
    credit,peak=measured(.03,.03,False,True,0.)
    assert float(credit[0])==0                   # airborne above a lip
    credit,peak=measured(.03,.03,True,False,0.)
    assert float(credit[0])==0                   # no safe support
    credit,peak=measured(.03,.03,True,True,0.)
    assert 0 < float(credit[0]) <= 4.
    again,nextpeak=measured(.03,.03,True,True,float(peak[0]))
    assert float(again[0])==0 and float(nextpeak[0])==float(peak[0])
    bounce,_=measured(.03,.01,True,True,float(peak[0]))
    assert float(bounce[0])==0                   # lower/higher oscillation
    down,new=measured(-.025,-.025,True,True,0.,direction=-1)
    assert float(down[0])>0
    wrong,_=measured(-.025,.025,True,True,0.,direction=-1)
    assert float(wrong[0])==0                   # no reward for falling wrong way
    import pytest
    with pytest.raises(ValueError):
        measured(.03,.03,True,True,0.,direction=0)
