"""Eleven physical invariants: actual prism triangles, sensors, contacts and evidence.

This verifies the *real 18-prism world*, not the old 9-box placeholder.
A 4-world MJX physical trajectory probe is run separately in CI.
"""
from __future__ import annotations
from types import SimpleNamespace

import numpy as np
import pytest
import jax
import jax.numpy as jnp
import mujoco

from emu.carpet_prism_mjx import (batch_update_prism_mocaps, PRISM_COUNT,
                                  PRISMS_PER_TILE, PRISM_GEOM_NAMES)
from emu.terrain_carpet_mjx import (
    update_tile_mocaps,
    feet_clear_of_abandoned_prisms, add_carpet_geoms,
    TILE_SIZE_Z_M, ray_surface, supported_tile_foot_pairs,
    VISIT_HISTORY, unseen_tile, note_confirmed_tile)
from scripts.exam_prism_course import score_prism_crossings


def foot_guard(old, new, feet, radii=(.018,.018)):
    return np.asarray(feet_clear_of_abandoned_prisms(
        jnp,jnp.asarray(old,dtype=jnp.int32),
        jnp.asarray(new,dtype=jnp.int32),
        jnp.asarray(feet,dtype=jnp.float32),
        jnp.asarray(radii,dtype=jnp.float32)))


def trace(*, rise=.02, axis=0, frames=12):
    cells=np.zeros((frames,1,2),dtype=np.int32)
    cells[2:,0,axis]=1
    z=np.zeros((frames,1,9,4),dtype=np.float64)
    z[:,:,axis==1 and 3 or 1,:]=rise
    pair=np.zeros((frames,1,18,2),dtype=bool)
    normal=np.zeros((frames,1,18,2),dtype=np.float64)
    foot=np.zeros((frames,1,2,3),dtype=np.float64)
    foot[:,:,0,:]=[.25,.25,.020]
    foot[:,:,1,:]=[.30,.30,.020]
    index=3 if axis==1 else 1
    for t in range(2,frames):
        foot[t,0,0,:]=[.25+(axis==0)*.50,.25+(axis==1)*.50,rise+.020]
        pair[t,0,2*index,0]=True
        normal[t,0,2*index,0]=.96
    return dict(carpet_root_cell=cells,carpet_prism_corner_z=z,
                carpet_raw_contact_pairs=pair,carpet_contact_vertical=normal,
                foot_xyz=foot,initial_foot_radii_m=np.array([.020,.020]),
                fell=np.zeros((frames,1),dtype=bool),
                done=np.zeros((frames,1),dtype=bool))


def fixture():
    positions=np.zeros((1,9,3),np.float32)
    positions[:,:,:]=[0.,0.,-30.]
    positions[0,0]=[-.25,0.,-TILE_SIZE_Z_M*.5]
    positions[0,1]=[+.25,0.,.15-TILE_SIZE_Z_M*.5]
    samples=np.full((1,36),.5,dtype=np.float32)
    return SimpleNamespace(mocap_pos=jnp.asarray(positions)),jnp.asarray(samples)


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
        # Source-height hfields use nine mocaps rather than 18. These
        # noncontact samples still drive EVERY ray and prism-top vertex.
        source_ids=ids[:9]
        old_tiles=original
        for i,mocap in enumerate(source_ids):
            old_tiles=old_tiles.at[:,mocap,:].set(jnp.where(
                changed[:,i,None],centers[:,i,:],old_tiles[:,mocap,:]))
        fresh_tiles=update_tile_mocaps(
            jnp,original,source_ids,centers,changed)
        np.testing.assert_array_equal(
            np.asarray(old_tiles),np.asarray(fresh_tiles))
        compiled_tiles=jax.jit(lambda p,c,m:update_tile_mocaps(
            jnp,p,source_ids,c,m))(original,centers,changed)
        np.testing.assert_array_equal(
            np.asarray(old_tiles),np.asarray(compiled_tiles))
        if active:
            direct=update_tile_mocaps(jnp,original,source_ids,centers)
            np.testing.assert_array_equal(
                np.asarray(old_tiles),np.asarray(direct))


    # This is also where the same nine source tile heights become every
    # foot-support sample. Prove the vectorised nine-tile calculation
    # reproduces the previous sequential support-height calculation.
    from emu.terrain_carpet_mjx import (
        sample_surface, TILE_HALF_M, TILE_M, TILE_SIZE_Z_M)
    source_ids=ids[:9]
    samples=jnp.asarray(rng.uniform(.1,.9,(worlds,36)),dtype=jnp.float32)
    query=original[:,source_ids,:]+jnp.asarray(
        [.06,-.04,.10],dtype=jnp.float32)[None,None,:]
    old_surface=jnp.full((worlds,9),-jnp.inf,dtype=jnp.float32)
    for i,mocap in enumerate(source_ids):
        p=original[:,mocap,:]
        u=(query[:,:,0]-p[:,None,0]+TILE_HALF_M)/TILE_M
        v=(query[:,:,1]-p[:,None,1]+TILE_HALF_M)/TILE_M
        a,b,c,d=[samples[:,i*4+k,None] for k in range(4)]
        near=a*(1.-u-v)+b*u+c*v
        far=d*(u+v-1.)+b*(1.-v)+c*(1.-u)
        z=p[:,None,2]+TILE_SIZE_Z_M*jnp.where(u+v<=1.,near,far)
        inside=(u>=0.)&(u<=1.)&(v>=0.)&(v<=1.)
        old_surface=jnp.maximum(old_surface,jnp.where(inside,z,-jnp.inf))
    new_surface=sample_surface(
        jnp,SimpleNamespace(mocap_pos=original),source_ids,samples,query)
    np.testing.assert_array_equal(
        np.asarray(old_surface),np.asarray(new_surface))
    compiled_surface=jax.jit(lambda p,h,q:sample_surface(
        jnp,SimpleNamespace(mocap_pos=p),source_ids,h,q))(
            original,samples,query)
    np.testing.assert_array_equal(
        np.asarray(old_surface),np.asarray(compiled_surface))


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


@pytest.mark.parametrize("rise,axis,up,down",[
    (.02,0,1,0),(-.035,0,0,1),(.003,1,1,0),(-.02,1,0,1)])
def test_real_top_supported_crossing_in_both_axes(rise,axis,up,down):
    t=trace(rise=rise,axis=axis)
    result=score_prism_crossings(t)
    assert (result["up"],result["down"],result["total"])==(up,down,1)
    assert result["worlds"][0]["independently_supported_lips"][0]["frame"]==6


def test_unphysical_side_contact_cannot_earn_lip():
    t=trace()
    t["carpet_contact_vertical"][:]=.04
    assert score_prism_crossings(t)["total"]==0


def test_tof_sees_real_15cm_vertical_uphill_wall():
    data,samples=fixture()
    origin=jnp.asarray([[-.15,0.,.07]],dtype=jnp.float32)
    rays=jnp.asarray([[[1.,0.,-.05]]],dtype=jnp.float32)
    t=np.asarray(ray_surface(jnp,data,tuple(range(9)),samples,origin,rays))
    # Lower top is at zero, so the slant line reaches the uphill
    # vertical side at x=0 before it could touch the upper top.
    assert t.shape==(1,1)
    assert abs(float(t[0,0])-.15)<1e-4,t


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


def test_carpet_birth_exposure_is_probability_not_obstacle_amplitude():
    """Regression for the 140M run: .15 exposure must not shrink a .7deg
    SLOPE world to .105deg or a 3mm Academy lip to .45mm."""
    import jax.numpy as jnp
    import numpy as np
    from emu.capability_focus import SLOPE, STEP
    from emu.terrain_carpet_mjx import role_envelope
    count=512
    seeds=jnp.arange(12000,12000+count,dtype=jnp.int32)
    active=jnp.ones((count,),dtype=jnp.bool_)
    idle=jnp.zeros((count,),dtype=jnp.float32)
    no_academy=jnp.zeros((count,),dtype=jnp.bool_)
    slope,up,down=role_envelope(
        jnp,jnp.full((count,),SLOPE),active,idle,
        .7,.003,.003,no_academy,.15,.15,seeds)
    slope=np.asarray(slope)
    assert 30 < np.count_nonzero(slope) < 120
    assert np.allclose(slope[slope>0],.7,rtol=0,atol=1e-5)
    # Non-academy STEP worlds follow the *same* exposure probability,
    # retaining the full 3mm height of the STEP specialist when active.
    _,step_up,step_down=role_envelope(
        jnp,jnp.full((count,),STEP),active,idle,
        .7,.003,.003,no_academy,.15,.15,seeds)
    step_up=np.asarray(step_up)
    assert 30 < np.count_nonzero(step_up) < 120
    assert np.allclose(step_up[step_up>0],.003,rtol=0,atol=1e-7)
    assert np.array_equal(step_up,np.asarray(step_down))
    # Assigned Academy episodes get physical lips regardless of the
    # ordinary STEP exposure probability; foundation stays flat.
    academy=jnp.arange(count)%2==0
    _,academy_up,academy_down=role_envelope(
        jnp,jnp.full((count,),STEP),active,idle,
        0.,.003,.003,academy,0.,0.,seeds)
    assert np.array_equal(np.asarray(academy_up)>0,np.asarray(academy))
    assert np.allclose(np.asarray(academy_up)[::2],.003,atol=1e-7)
    assert np.array_equal(np.asarray(academy_up),np.asarray(academy_down))
    slope_flat,up_flat,down_flat=role_envelope(
        jnp,jnp.full((count,),SLOPE),active,idle,
        .7,.003,.003,no_academy,0.,0.,seeds)
    for x in (slope_flat,up_flat,down_flat):
        assert not np.any(np.asarray(x))
