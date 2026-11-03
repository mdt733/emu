"""Prism-only recycler: true 50cm prism silhouette protects feet."""
import numpy as np
import jax
import jax.numpy as jnp
import mujoco

from emu.terrain_carpet_mjx import (
    feet_clear_of_abandoned_prisms, add_carpet_geoms)
from emu.carpet_prism_mjx import PRISM_COUNT,PRISM_GEOM_NAMES


def foot_guard(old, new, feet, radii=(.018,.018)):
    return np.asarray(feet_clear_of_abandoned_prisms(
        jnp,jnp.asarray(old,dtype=jnp.int32),
        jnp.asarray(new,dtype=jnp.int32),
        jnp.asarray(feet,dtype=jnp.float32),
        jnp.asarray(radii,dtype=jnp.float32)))


def test_prism_model_has_no_inactive_box_bodies_at_all():
    model=mujoco.MjModel.from_xml_string(
        add_carpet_geoms('<mujoco><worldbody/></mujoco>',prisms=True))
    names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i)
           for i in range(model.ngeom)]
    assert not any(n.startswith('carpet_solid_geom_') for n in names)
    assert len([n for n in names if n in PRISM_GEOM_NAMES])==PRISM_COUNT
    assert model.nhfield==9  # four source-height values per tile, inert


def test_true_prism_footprint_blocks_sidewall_not_far_foot():
    old=[[0,0],[0,0],[0,0]]
    new=[[1,0],[1,0],[1,1]]
    feet=[
        [[.018,.25],[.75,.25]],  # 18mm foot overlaps x=0 wall
        [[.040,.25],[.75,.25]],  # 40mm foot is outside all old squares
        [[-.01,-.01],[.75,.75]], # diagonal abandoned corner touch
    ]
    expected=[False,True,False]
    np.testing.assert_array_equal(
        foot_guard(old,new,feet),expected)
    compiled=jax.jit(lambda a,b,c:feet_clear_of_abandoned_prisms(
        jnp,a,b,c,jnp.asarray([.018,.018],dtype=jnp.float32)))
    np.testing.assert_array_equal(
        np.asarray(compiled(jnp.asarray(old),jnp.asarray(new),
                            jnp.asarray(feet))),expected)


def test_no_move_requires_no_foot_clearance():
    # Empty outgoing tile set: nothing can be teleported by the guard.
    pos=[[[.0,.0],[.1,.1]]]
    assert foot_guard([[0,0]],[[0,0]],pos)[0]
