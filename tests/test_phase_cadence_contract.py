"""Regression for v5014's broken three-argument jnp.minimum cadence."""
import pytest


def test_command_cadence_is_finite_and_bounded_with_real_jax():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    from emu.environment import command_phase_rate
    f = jax.jit(command_phase_rate)
    for vx,vy in ((0.0,0.0), (0.2,0.0), (0.6,0.4), (-0.3,0.2)):
        got = float(f(jnp.float32(vx),jnp.float32(vy),0.6,1.25,1.025))
        assert 1.02499 <= got <= 2.60001
    assert float(f(0.6,0.0,0.6,1.25,1.025)) > float(f(0.1,0.0,0.6,1.25,1.025))


def test_cadence_not_capped_at_pitch_command_limit():
    from emu.environment import command_phase_rate
    from emu.contracts import PITCH_CMD_MAX_RAD
    got = float(command_phase_rate(0.0,0.0,0.6,1.25,1.025))
    assert got > 0.8 > PITCH_CMD_MAX_RAD
