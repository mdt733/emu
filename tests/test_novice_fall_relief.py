"""Practice-only fall economics, with no success credit or reset changes."""
import numpy as np
from emu.rewards import novice_fall_scale,apply_early_fall_penalty

def check_contract(xp):
    ids=xp.array([-1,0,5,6,7,8,9,10,11,12,6,7])
    stages=xp.array([4,4,5,4,5,1,2,3,0,4,4,5])
    bonus=xp.array([4.,4.,4.,4.,3.,2.,1.,0.,0.,4.,4.,4.])
    diverged=xp.array([False]*10+[True,False])
    failed=xp.array([True]*11+[False])
    ages=xp.full(ids.shape,100.)
    reward=xp.full(ids.shape,2.)
    scale=novice_fall_scale(xp,ids,stages,bonus,diverged)
    actual,payment=apply_early_fall_penalty(reward,failed,ages,700,xp=xp,penalty_scale=scale)
    baseline,full=apply_early_fall_penalty(reward,failed,ages,700,xp=xp)
    np.testing.assert_allclose(scale,[1,1,1,.1,.325,.55,.775,1,1,1,1,.1],atol=1e-6)
    np.testing.assert_allclose(payment,[-125,-125,-125,-12.5,-40.625,-68.75,-96.875,-125,-125,-125,-125,0],atol=2e-5)
    np.testing.assert_allclose(actual,reward+payment,atol=2e-5)
    assert np.all(np.asarray(actual)[np.asarray(failed)]<0.)
    assert actual[-1]==baseline[-1]==2.
    assert np.all(np.asarray(payment-full)>=0)

def test_numpy_fall_contract():
    check_contract(np)

def test_jax_fall_contract_and_jit():
    import jax
    import jax.numpy as jnp
    check_contract(jnp)
    n=4096
    ids=jnp.arange(n)%13-1
    stage=jnp.arange(n)%6
    bonus=jnp.where(stage>0,4.,0.)
    fail=jnp.arange(n)%2==0
    divergence=jnp.arange(n)%7==0
    @jax.jit
    def run(ids,stage,bonus,fail,divergence):
        scale=novice_fall_scale(jnp,ids,stage,bonus,divergence)
        return apply_early_fall_penalty(jnp.ones(n),fail,jnp.ones(n)*100,700,
                                      xp=jnp,penalty_scale=scale)[1]
    payment=np.asarray(run(ids,stage,bonus,fail,divergence))
    expected=np.where(np.asarray(fail),
        np.where((np.asarray(ids)>=6)&(np.asarray(stage)>0)&~np.asarray(divergence),-12.5,-125.),0.)
    np.testing.assert_allclose(payment,expected,atol=2e-5)
