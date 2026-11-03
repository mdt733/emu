"""Exact synthetic ray regressions for GPU Training Carpet observations.

The 4×4 ToF is a physical ray synthesizer. It may be teacher-side JAX
calculation, but it must see the SAME uphill wall the GPU feet collide with.
"""
from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from emu.terrain_carpet_mjx import (
    TILE_SIZE_Z_M, ray_surface, sample_surface,
    signed_height_over_support, supported_tile_foot_pairs)


def fixture():
    positions=np.zeros((1,9,3),np.float32)
    positions[:,:,:]=[0.,0.,-30.]
    positions[0,0]=[-.25,0.,-TILE_SIZE_Z_M*.5]
    positions[0,1]=[+.25,0.,.15-TILE_SIZE_Z_M*.5]
    samples=np.full((1,36),.5,dtype=np.float32)
    return SimpleNamespace(mocap_pos=jnp.asarray(positions)),jnp.asarray(samples)


def test_tof_sees_real_15cm_vertical_uphill_wall():
    data,samples=fixture()
    origin=jnp.asarray([[-.15,0.,.07]],dtype=jnp.float32)
    rays=jnp.asarray([[[1.,0.,-.05]]],dtype=jnp.float32)
    t=np.asarray(ray_surface(jnp,data,tuple(range(9)),samples,origin,rays))
    # Lower top is at zero, so the slant line reaches the uphill
    # vertical side at x=0 before it could touch the upper top.
    assert t.shape==(1,1)
    assert abs(float(t[0,0])-.15)<1e-4,t


def test_tof_does_not_invent_wall_above_real_top():
    data,samples=fixture()
    origin=jnp.asarray([[-.15,0.,.22]],dtype=jnp.float32)
    rays=jnp.asarray([[[1.,0.,-.05]]],dtype=jnp.float32)
    t=np.asarray(ray_surface(jnp,data,tuple(range(9)),samples,origin,rays))
    assert np.isinf(t[0,0]),t


def test_actual_heightfield_vertex_buffer_drives_support_estimate():
    data,samples=fixture()
    p=jnp.asarray([[[-.13,0.,.21],[+.13,0.,.21]]],dtype=jnp.float32)
    heights=np.asarray(sample_surface(jnp,data,tuple(range(9)),samples,p))
    assert np.allclose(heights,[ [0.,.15] ],rtol=0,atol=1e-6),heights


def test_no_phantom_support_beyond_nine_tile_footprint():
    data,samples=fixture()
    p=jnp.asarray([[[2.5,2.5,.5]]],dtype=jnp.float32)
    h=np.asarray(sample_surface(jnp,data,tuple(range(9)),samples,p))
    assert np.isneginf(h[0,0]),h


def test_no_infinite_clearance_for_a_robot_leaving_generated_world():
    p=jnp.asarray([[[0.,0.,.21],[2.,2.,.37]]],dtype=jnp.float32)
    z=jnp.asarray([[0.,-jnp.inf]],dtype=jnp.float32)
    d=np.asarray(signed_height_over_support(jnp,p,z))
    assert np.all(np.isfinite(d)),d
    assert abs(float(d[0,0])-.21)<1e-6
    assert d[0,1]==-1.0  # fail-eligible instead of positive infinity


def test_side_contact_cannot_be_counted_as_supported_top_foothold():
    # Two contacts to the SAME upward tile. The first sphere is pressed
    # into the vertical step riser; the second is standing on the top.
    frame=jnp.zeros((1,2,3,3),dtype=jnp.float32)
    frame=frame.at[0,0,0,:].set(jnp.asarray([1.,0.,0.]))
    frame=frame.at[0,1,0,:].set(jnp.asarray([0.,0.,1.]))
    support=supported_tile_foot_pairs(
        jnp,jnp.asarray([[7,7]]),jnp.asarray([[1,2]]),
        jnp.asarray([[-.002,-.002]]),frame,7,(1,2),
        jnp.asarray([[.15,.15]]),jnp.asarray([[.15,.15]]))
    assert np.array_equal(np.asarray(support),[[False,True]]),support


def test_top_contact_must_match_foot_sphere_bottom_elevation():
    frame=jnp.zeros((1,2,3,3),dtype=jnp.float32)
    frame=frame.at[0,:,0,2].set(1.)
    support=supported_tile_foot_pairs(
        jnp,jnp.asarray([[7,7]]),jnp.asarray([[1,2]]),
        jnp.asarray([[-.002,-.002]]),frame,7,(1,2),
        jnp.asarray([[.09,.14]]),jnp.asarray([[.15,.15]]))
    assert np.array_equal(np.asarray(support),[[False,True]]),support
