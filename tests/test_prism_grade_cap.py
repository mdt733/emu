"""v5043 seeded physical triangle grade and transition curriculum bounds.

A grid of independent JAX worlds; checks actual top-triangle plane normals,
not merely the average hfield/sample gradient printed by the teacher.
"""
import numpy as np
import pytest


@pytest.mark.parametrize("slope_deg,transition_deg",[
    (.7,3.), (3.,3.), (8.,3.), (15.,3.), (15.,1.), (10.,.5)])
def test_both_physical_triangle_normals_respect_curriculum(
        slope_deg,transition_deg):
    import jax.numpy as jnp
    from emu.terrain_carpet_mjx import (
        make_profiles, tile_geometry, TILE_M, TILE_SIZE_Z_M)

    nworld=16
    side=5
    coords=np.asarray([(x,y) for x in range(-2,3)
                       for y in range(-2,3)],dtype=np.int32)
    seeds=np.repeat(np.arange(735,735+nworld,dtype=np.int32),len(coords))
    xy=np.tile(coords,(nworld,1))
    size=len(seeds)
    zero=jnp.zeros((size,),dtype=jnp.float32)
    xp,yp=make_profiles(jnp,jnp.asarray(seeds),zero,zero)
    centers,vert=tile_geometry(
        jnp,jnp.asarray(xy[:,0]),jnp.asarray(xy[:,1]),
        jnp.asarray(seeds),xp,yp,zero,zero,
        jnp.full((size,),slope_deg,dtype=jnp.float32),
        transition_limit_deg=transition_deg)
    # World-coordinate elevations, not normalized 0.5-ish hfield bytes.
    h=(np.asarray(centers[:,2,None])+
       TILE_SIZE_Z_M*np.asarray(vert)).reshape((nworld,side,side,4))
    a,b,c,d=(h[...,i] for i in range(4))
    g0=np.stack((-(b-a)/TILE_M,-(c-a)/TILE_M,
                 np.ones_like(a)),axis=-1)
    g1=np.stack((-(d-c)/TILE_M,-(d-b)/TILE_M,
                 np.ones_like(a)),axis=-1)
    n0=g0/np.linalg.norm(g0,axis=-1,keepdims=True)
    n1=g1/np.linalg.norm(g1,axis=-1,keepdims=True)

    def angular(left,right):
        return np.rad2deg(np.arccos(np.clip(
            np.sum(left*right,axis=-1),-1.,1.)))

    grade=np.concatenate((
        np.rad2deg(np.arccos(np.clip(n0[...,2],-1.,1.))).ravel(),
        np.rad2deg(np.arccos(np.clip(n1[...,2],-1.,1.))).ravel()))
    # The actual two triangles are continuous, but their normals may
    # differ; this tests that crease as a *real* slope transition.
    changes=np.concatenate((
        angular(n0,n1).ravel(),
        angular(n1[:,:-1,:,:],n0[:,1:,:,:]).ravel(),
        angular(n1[:,:,:-1,:],n0[:,:,1:,:]).ravel()))
    assert np.isfinite(grade).all() and np.isfinite(changes).all()
    assert float(np.max(grade))<=slope_deg+.25, (
        'Unexpected actual physical grade',slope_deg,grade.max())
    assert float(np.max(changes))<=transition_deg+.25, (
        'Actual triangle normal exceeded transition cap',
        transition_deg,changes.max())
    # Exposure must mean *current training difficulty*, not an almost
    # flat approximation (historical 140M v5041 failure).
    in_frontier=(grade>=.78*slope_deg)&(grade<=1.05*slope_deg)
    assert float(np.mean(in_frontier))>.20, (
        'The actual triangles no longer expose the earned frontier',
        slope_deg,float(np.mean(in_frontier)))

    # Both independent X/Y grid directions reuse exactly the same edge
    # endpoints. Any vertical difference here would be an accidental step.
    dx=np.stack((b[:,:-1,:]-a[:,1:,:],
                 d[:,:-1,:]-c[:,1:,:]),axis=-1)
    dy=np.stack((c[:,:,:-1]-a[:,:,1:],
                 d[:,:,:-1]-b[:,:,1:]),axis=-1)
    assert np.max(np.abs(dx))<3e-5
    assert np.max(np.abs(dy))<3e-5


def test_signed_x_and_y_lips_are_seeded_but_physically_detectable():
    import jax.numpy as jnp
    from emu.terrain_carpet_mjx import make_profiles,PROFILE_HALF
    n=128
    seeds=jnp.arange(735,735+n,dtype=jnp.int32)
    up=jnp.full((n,),.003)
    down=jnp.full((n,),.003)
    x,y=make_profiles(jnp,seeds,up,down)
    for axis in (x,y):
        changes=np.diff(np.asarray(axis),axis=1)
        near=changes[:,PROFILE_HALF:PROFILE_HALF+3]
        assert np.min(np.abs(near))>.0015,near
        assert (near>0.).any() and (near<0.).any()
        assert np.max(np.abs(near))<.004
