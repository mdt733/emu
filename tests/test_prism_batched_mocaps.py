"""Verify batched prism mocap updates equal the old 18-scatter semantics."""
import numpy as np
import pytest

from emu.carpet_prism_mjx import batch_update_prism_mocaps, PRISM_COUNT, PRISMS_PER_TILE


def test_batched_mocaps_match_old_birth_and_recycling():
    import jax
    import jax.numpy as jnp

    rng=np.random.default_rng(733)
    worlds,mocaps=4,57
    ids=tuple(int(v) for v in rng.choice(mocaps,size=PRISM_COUNT,replace=False))
    original=jnp.asarray(rng.normal(size=(worlds,mocaps,3)),dtype=jnp.float32)
    centers=jnp.asarray(rng.normal(size=(worlds,PRISM_COUNT//PRISMS_PER_TILE,3)),
                        dtype=jnp.float32)

    for active in (True,False):
        changed=jnp.asarray(
            np.ones((worlds,9),bool) if active else
            (rng.uniform(size=(worlds,9))>.5))
        old=original
        for i,m in enumerate(ids):
            tile=i//PRISMS_PER_TILE
            old=old.at[:,m,:].set(jnp.where(
                changed[:,tile,None],centers[:,tile,:],old[:,m,:]))
        fresh=batch_update_prism_mocaps(jnp,original,ids,centers,changed)
        np.testing.assert_array_equal(np.asarray(old),np.asarray(fresh))
        fast=jax.jit(lambda p,c,m:batch_update_prism_mocaps(
            jnp,p,ids,c,m))(original,centers,changed)
        np.testing.assert_array_equal(np.asarray(old),np.asarray(fast))


def test_mocap_batch_rejects_nonunique_ids():
    import jax.numpy as jnp
    positions=jnp.zeros((1,30,3))
    centers=jnp.zeros((1,9,3))
    changed=jnp.zeros((1,9),dtype=jnp.bool_)
    with pytest.raises(ValueError):
        batch_update_prism_mocaps(jnp,positions,(1,)*PRISM_COUNT,centers,changed)
