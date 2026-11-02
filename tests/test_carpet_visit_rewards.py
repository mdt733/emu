"""Trainer-side proof that repeated physical lip rocking pays only once.

A 64-cell bounded exact visited-tile ring is per-world / episode state,
never an actor input. Legitimate new terrain still gets supported credit.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from emu.terrain_carpet_mjx import (
    VISIT_HISTORY, unseen_tile, note_confirmed_tile)


def test_new_tile_only_once_even_after_shuffling_across_lip():
    v=jnp.full((2,VISIT_HISTORY,2),-2**30,dtype=jnp.int32)
    origin=jnp.asarray([[0,0],[10,-4]],dtype=jnp.int32)
    v=v.at[:,0,:].set(origin)
    write=jnp.ones((2,),dtype=jnp.int32)
    forward=jnp.asarray([[1,0],[11,-4]],dtype=jnp.int32)
    assert np.all(unseen_tile(jnp,forward,v))
    v,write=note_confirmed_tile(jnp,forward,v,write,jnp.asarray([True,True]))
    assert not np.any(unseen_tile(jnp,forward,v))
    assert not np.any(unseen_tile(jnp,origin,v))
    after=jnp.asarray([[2,0],[12,-4]],dtype=jnp.int32)
    assert np.all(unseen_tile(jnp,after,v))
    # The same policy can turn/strafe: independent x/y global tiles.
    sideways=jnp.asarray([[1,1],[11,-3]],dtype=jnp.int32)
    assert np.all(unseen_tile(jnp,sideways,v))


def test_one_world_recycles_without_touching_another():
    v=jnp.full((2,VISIT_HISTORY,2),-2**30,dtype=jnp.int32)
    initial=jnp.asarray([[0,0],[20,20]],dtype=jnp.int32)
    v=v.at[:,0,:].set(initial)
    write=jnp.ones((2,),dtype=jnp.int32)
    steps=jnp.asarray([[1,0],[21,20]],dtype=jnp.int32)
    new,w=note_confirmed_tile(jnp,steps,v,write,jnp.asarray([True,False]))
    assert not bool(unseen_tile(jnp,steps,new)[0])
    assert bool(unseen_tile(jnp,steps,new)[1])
    assert np.array_equal(np.asarray(new[1]),np.asarray(v[1]))
    assert np.array_equal(np.asarray(w),[2,1])


def test_repeated_backtracking_is_rejected_for_full_episode_distance():
    v=jnp.full((1,VISIT_HISTORY,2),-2**30,dtype=jnp.int32)
    v=v.at[:,0,:].set(jnp.asarray([[0,0]]))
    write=jnp.ones((1,),dtype=jnp.int32)
    for i in range(1,40):
        coord=jnp.asarray([[i,0]],dtype=jnp.int32)
        assert bool(unseen_tile(jnp,coord,v)[0])
        v,write=note_confirmed_tile(
            jnp,coord,v,write,jnp.asarray([True]))
    for i in range(39,-1,-1):
        coord=jnp.asarray([[i,0]],dtype=jnp.int32)
        assert not bool(unseen_tile(jnp,coord,v)[0]),i
