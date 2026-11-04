"""Eleven physical invariants: actual prism triangles, sensors, contacts and evidence.

This verifies the *real eight-prism world*, not the old 9-box placeholder.
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
    TILE_SIZE_Z_M, TILE_M, TILE_COUNT, TILE_WINDOW_SIDE,
    midpoint_anchor, slot_indices, ray_surface, supported_tile_foot_pairs,
    VISIT_HISTORY, unseen_tile, note_confirmed_tile)
from emu.prism_exam import score_prism_crossings


def foot_guard(old, new, feet, radii=(.018,.018)):
    return np.asarray(feet_clear_of_abandoned_prisms(
        jnp,jnp.asarray(old,dtype=jnp.int32),
        jnp.asarray(new,dtype=jnp.int32),
        jnp.asarray(feet,dtype=jnp.float32),
        jnp.asarray(radii,dtype=jnp.float32)))


def trace(*, rise=.02, axis=0, frames=12):
    cells=np.zeros((frames,1,2),dtype=np.int32)
    cells[2:,0,axis]=1
    z=np.zeros((frames,1,TILE_COUNT,4),dtype=np.float64)
    z[:,:,axis==1 and 2 or 1,:]=rise
    pair=np.zeros((frames,1,PRISM_COUNT,2),dtype=bool)
    normal=np.zeros((frames,1,PRISM_COUNT,2),dtype=np.float64)
    foot=np.zeros((frames,1,2,3),dtype=np.float64)
    foot[:,:,0,:]=[TILE_M/2,TILE_M/2,.020]
    foot[:,:,1,:]=[TILE_M*.60,TILE_M*.60,.020]
    index=2 if axis==1 else 1
    for t in range(2,frames):
        foot[t,0,0,:]=[TILE_M/2+(axis==0)*TILE_M,TILE_M/2+(axis==1)*TILE_M,rise+.020]
        pair[t,0,2*index,0]=True
        normal[t,0,2*index,0]=.96
    return dict(carpet_root_cell=cells,carpet_prism_corner_z=z,
                carpet_raw_contact_pairs=pair,carpet_contact_vertical=normal,
                foot_xyz=foot,initial_foot_radii_m=np.array([.020,.020]),
                fell=np.zeros((frames,1),dtype=bool),
                done=np.zeros((frames,1),dtype=bool))


def fixture():
    positions=np.zeros((1,TILE_COUNT,3),np.float32)
    positions[:,:,:]=[0.,0.,-30.]
    positions[0,0]=[-TILE_M*.5,0.,-TILE_SIZE_Z_M*.5]
    positions[0,1]=[+TILE_M*.5,0.,.15-TILE_SIZE_Z_M*.5]
    samples=np.full((1,TILE_COUNT*4),.5,dtype=np.float32)
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
            np.ones((worlds,TILE_COUNT),bool) if active else
            (rng.uniform(size=(worlds,TILE_COUNT))>.5))
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
        # Source-height hfields use four mocaps rather than eight. These
        # noncontact samples still drive EVERY ray and prism-top vertex.
        source_ids=ids[:TILE_COUNT]
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


    # This is also where the same four source tile heights become every
    # foot-support sample. Prove the vectorised four-tile calculation
    # reproduces the previous sequential support-height calculation.
    from emu.terrain_carpet_mjx import (
        sample_surface, TILE_HALF_M, TILE_M, TILE_SIZE_Z_M)
    source_ids=ids[:TILE_COUNT]
    samples=jnp.asarray(rng.uniform(.1,.9,(worlds,TILE_COUNT*4)),dtype=jnp.float32)
    query=original[:,source_ids,:]+jnp.asarray(
        [.06,-.04,.10],dtype=jnp.float32)[None,None,:]
    old_surface=jnp.full((worlds,TILE_COUNT),-jnp.inf,dtype=jnp.float32)
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
    # XLA may fuse triangle interpolation multiply-adds, introducing
    # <0.12 micrometre roundoff vs the sequential eager reference.
    # This is numerical equivalence, not a change in physical geometry.
    np.testing.assert_allclose(
        np.asarray(old_surface),np.asarray(compiled_surface),
        rtol=0,atol=2e-7)


def test_prism_model_has_no_inactive_box_bodies_at_all():
    """Compile the ACTUAL robot.xml, not an artificial empty XML fixture."""
    from pathlib import Path
    robot=Path(__file__).resolve().parents[1]/'emu'/'robot.xml'
    model=mujoco.MjModel.from_xml_string(
        add_carpet_geoms(robot.read_text(encoding='utf-8'),prisms=True))
    names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i)
           for i in range(model.ngeom)]
    assert not any((n or '').startswith('carpet_solid_geom_') for n in names)
    assert len([n for n in names if n in PRISM_GEOM_NAMES])==PRISM_COUNT
    assert model.nhfield==TILE_COUNT  # noncontact source vertex samples
    assert mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'unified_ground')<0
    assert mujoco.mj_name2id(
        model,mujoco.mjtObj.mjOBJ_GEOM,'episode_terrain_surface')<0
    floor=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'floor')
    assert floor>=0 and model.geom_contype[floor]==0
    assert model.nmocap>=TILE_COUNT+PRISM_COUNT
    with pytest.raises(ValueError,match='real prism contacts only'):
        add_carpet_geoms(robot.read_text(encoding='utf-8'),prisms=False)


def test_foot_only_contacts_preserve_mass_and_remove_torso_collision():
    """Remove solver contact PAIRS, not the 300g chassis or its inertia."""
    from pathlib import Path
    from mujoco import mjx
    source=(Path(__file__).resolve().parents[1]/'emu'/'robot.xml').read_text()
    reference=mujoco.MjModel.from_xml_string(
        add_carpet_geoms(source,feet_only_contacts=False))
    light=mujoco.MjModel.from_xml_string(
        add_carpet_geoms(source,feet_only_contacts=True))
    np.testing.assert_array_equal(light.body_mass,reference.body_mass)
    np.testing.assert_array_equal(light.body_inertia,reference.body_inertia)
    np.testing.assert_array_equal(light.geom_size,reference.geom_size)
    for name in ('torso_box','floor','r_thigh_rod','l_thigh_rod',
                 'r_shin_upper','l_shin_upper',
                 'r_shin_lower_rod','l_shin_lower_rod'):
        geom=mujoco.mj_name2id(light,mujoco.mjtObj.mjOBJ_GEOM,name)
        assert geom>=0
        assert light.geom_contype[geom]==0
        assert light.geom_conaffinity[geom]==0
    torso=mujoco.mj_name2id(reference,mujoco.mjtObj.mjOBJ_GEOM,
                           'torso_box')
    assert reference.geom_contype[torso]!=0
    for name in ('r_foot_ball','l_foot_ball'):
        geom=mujoco.mj_name2id(light,mujoco.mjtObj.mjOBJ_GEOM,name)
        assert light.geom_contype[geom]!=0
        assert light.geom_conaffinity[geom]&4
    # Actual MJX padded solver shape must shrink as well; merely
    # removing an XML contact flag is not sufficient evidence.
    old=mjx.make_data(mjx.put_model(reference))
    new=mjx.make_data(mjx.put_model(light))
    assert new._impl.contact.geom1.shape[0]<old._impl.contact.geom1.shape[0]
    assert new._impl.efc_J.shape[0]<old._impl.efc_J.shape[0]


def test_midpoint_2x2_window_covers_30cm_and_preserves_present_tile():
    """All torso positions are >=37.5cm from an unloaded XY boundary."""
    fractions=np.asarray([0.,.001,.24,.499,.5,.501,.77,.9999])
    points=np.asarray([[(c+f)*TILE_M,(d+g)*TILE_M]
        for c in (-2,-1,0,1) for d in (-2,-1,0,1)
        for f in fractions for g in fractions],dtype=np.float32)
    anchors=np.asarray(midpoint_anchor(jnp,jnp.asarray(points)))
    clearance=np.minimum(points-anchors*TILE_M,
                         (anchors+TILE_WINDOW_SIDE)*TILE_M-points)
    assert clearance.min()>=TILE_M*.5-2e-5
    assert clearance.min()>.30
    x,y=slot_indices(jnp,jnp.asarray(anchors[:,0]),jnp.asarray(anchors[:,1]))
    cells=np.stack((np.asarray(x),np.asarray(y)),axis=2)
    present=np.floor(points/TILE_M).astype(np.int32)
    assert np.all(np.any(np.all(cells==present[:,None,:],axis=2),axis=1))
    p=lambda x,y:np.asarray(midpoint_anchor(jnp,jnp.asarray([[x*TILE_M,y*TILE_M]])))
    old=p(.499,.499)
    sx,sy=slot_indices(jnp,jnp.asarray(old[:,0]),jnp.asarray(old[:,1]))
    for root,expected in ((p(.501,.499),2),(p(.501,.501),3)):
        tx,ty=slot_indices(jnp,jnp.asarray(root[:,0]),jnp.asarray(root[:,1]))
        changed=(np.asarray(tx)!=np.asarray(sx))|(np.asarray(ty)!=np.asarray(sy))
        assert int(changed.sum())==expected
    np.testing.assert_array_equal(p(.499,.499),old)


def test_true_prism_footprint_blocks_sidewall_not_far_foot():
    old=[[0,0],[0,0],[0,0]]
    new=[[1,0],[1,0],[1,1]]
    feet=[
        [[TILE_M+.018,TILE_M*.5],[TILE_M*1.5,TILE_M*.5]],  # 18mm foot overlaps x=0 wall
        [[TILE_M+.040,TILE_M*.5],[TILE_M*1.5,TILE_M*.5]],  # 40mm foot is outside all old squares
        [[TILE_M+.01,TILE_M+.01],[TILE_M*1.5,TILE_M*1.5]], # diagonal abandoned corner touch
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



def test_batched_rays_preserve_old_4x4_and_sidewall_geometry():
    """Keep a source-frozen sequential ray oracle for exactly this GPU optimization."""
    from emu.terrain_carpet_mjx import TILE_M,TILE_HALF_M,TILE_BASE_M,TILE_COUNT
    def old_rays(xp,data,mocap_ids,hfield_data,origin,rays,*,
                    prism_bottom_depth_m=None):
        """Slant range against both real triangles in every 75cm tile.
    
        Iterative ray-plane intersections use the same planar interpolation as
        sample_surface. Only finite ray distances inside a tile are admitted.
        The old two-slab ray never leaks its buried ground into 360 observations.
        """
        nearest=xp.full(rays.shape[:2],xp.inf,dtype=xp.float32)
        vertical=rays[:,:,2]
        safe=xp.where(xp.abs(vertical)>1e-6,vertical,-1.)
        for i,mocap in enumerate(mocap_ids):
            pos=data.mocap_pos[:,mocap,:]
            sample=hfield_data[:,i*4:i*4+4]
            middle=pos[:,2]+TILE_SIZE_Z_M*xp.mean(sample,axis=1)
            t=(middle[:,None]-origin[:,None,2])/safe
            for _ in range(3):
                u=(origin[:,None,0]+t*rays[:,:,0]-pos[:,None,0]+TILE_HALF_M)/TILE_M
                v=(origin[:,None,1]+t*rays[:,:,1]-pos[:,None,1]+TILE_HALF_M)/TILE_M
                a,b,c,d=[sample[:,k,None] for k in range(4)]
                near=a*(1.-u-v)+b*u+c*v
                far=d*(u+v-1.)+b*(1.-v)+c*(1.-u)
                height=pos[:,None,2]+TILE_SIZE_Z_M*xp.where(u+v<=1.,near,far)
                t=(height-origin[:,None,2])/safe
            inside=(u>=0.)&(u<=1.)&(v>=0.)&(v<=1.)
            good=inside&(t>0.)&(rays[:,:,2]<-.05)
            nearest=xp.minimum(nearest,xp.where(good,t,xp.inf))
    
            # A two-triangle MuJoCo hfield is a solid height volume with
            # vertical *outer side walls*. Omitting those sides would let the
            # 4x4 ToF miss precisely the uphill riser the feet collide with.
            # This matches the SAME samples and base thickness as MJX, not a
            # separate privileged ray obstacle. Two x-faces and two y-faces.
            for axis,other,edges in ((0,1,((0,2),(1,3))),
                                     (1,0,((0,1),(2,3)))):
                direction=rays[:,:,axis]
                safe_dir=xp.where(xp.abs(direction)>1e-7,direction,1.)
                other_dir=rays[:,:,other]
                for side,(lo,hi) in enumerate(edges):
                    edge_pos=pos[:,axis]+(-TILE_HALF_M if side==0 else TILE_HALF_M)
                    tt=(edge_pos[:,None]-origin[:,None,axis])/safe_dir
                    other_at=origin[:,None,other]+tt*other_dir
                    fraction=(other_at-pos[:,None,other]+TILE_HALF_M)/TILE_M
                    z_at=origin[:,None,2]+tt*rays[:,:,2]
                    z_top=pos[:,None,2]+TILE_SIZE_Z_M*(
                          sample[:,lo,None]*(1.-fraction)
                          +sample[:,hi,None]*fraction)
                    # v5043 prisms have a sheared lower triangle by the configured prism depth
                    # below each top vertex. The old heightfield base used a
                    # different constant Z and would expose phantom sidewalls
                    # below the actual prism collision body.
                    z_bottom=(z_top-prism_bottom_depth_m if
                              prism_bottom_depth_m is not None else
                              pos[:,None,2]-TILE_BASE_M)
                    hit=(xp.abs(direction)>1e-7)&(tt>1e-7)&(
                        fraction>=0.)&(fraction<=1.)&(
                        z_at<=z_top)&(z_at>=z_bottom)
                    nearest=xp.minimum(nearest,xp.where(hit,tt,xp.inf))
        return nearest
    
    rng=np.random.default_rng(1701)
    worlds=3
    positions=np.zeros((worlds,TILE_COUNT,3),dtype=np.float32)
    for tile in range(TILE_COUNT):
        positions[:,tile,0]=(tile%TILE_WINDOW_SIDE-.5)*TILE_M
        positions[:,tile,1]=(tile//TILE_WINDOW_SIDE-.5)*TILE_M
    positions[:,:,2]=rng.uniform(-.12,.13,(worlds,TILE_COUNT)).astype(np.float32)
    heights=jnp.asarray(rng.uniform(.15,.85,(worlds,TILE_COUNT*4)),dtype=jnp.float32)
    origin=jnp.asarray([[.02,-.08,.18],[-.12,.04,.15],[.17,.17,.25]],dtype=jnp.float32)
    pitch=np.deg2rad([40,60,80,100])
    yaw=np.deg2rad([-22.5,-7.5,7.5,22.5])
    beam=np.asarray([[np.cos(p)*np.cos(y),np.cos(p)*np.sin(y),-np.sin(p)]
                     for p in pitch for y in yaw],dtype=np.float32)
    rays=jnp.broadcast_to(jnp.asarray(beam),(worlds,16,3))
    points=jnp.asarray(positions)
    ids=tuple(range(TILE_COUNT))
    for bottom in (None,.65):
        expected=np.asarray(old_rays(jnp,SimpleNamespace(mocap_pos=points),
                                     ids,heights,origin,rays,
                                     prism_bottom_depth_m=bottom))
        new=jax.jit(lambda p,h,o,ray:ray_surface(
            jnp,SimpleNamespace(mocap_pos=p),ids,h,o,ray,
            prism_bottom_depth_m=bottom))
        actual=np.asarray(new(points,heights,origin,rays))
        assert actual.shape==(worlds,16)
        np.testing.assert_array_equal(np.isfinite(actual),np.isfinite(expected))
        finite=np.isfinite(expected)
        np.testing.assert_allclose(actual[finite],expected[finite],
                                   rtol=0,atol=5e-6)
        # Fusing the sixteen ToF beams with the downward flow-height
        # beam must preserve both sets of ray intersections.
        down=jnp.broadcast_to(jnp.asarray([[[0.,0.,-1.]]],dtype=jnp.float32),
                              (worlds,1,3))
        combined=jnp.concatenate([rays,down],axis=1)
        fused=np.asarray(jax.jit(lambda p,h,o,ray:ray_surface(
            jnp,SimpleNamespace(mocap_pos=p),ids,h,o,ray,
            prism_bottom_depth_m=bottom))(points,heights,origin,combined))
        original_depth=np.asarray(ray_surface(
            jnp,SimpleNamespace(mocap_pos=points),ids,heights,
            origin,down,prism_bottom_depth_m=bottom))
        assert fused.shape==(worlds,17)
        np.testing.assert_array_equal(np.isfinite(fused[:,:16]),np.isfinite(actual))
        np.testing.assert_array_equal(np.isfinite(fused[:,16:]),np.isfinite(original_depth))
        np.testing.assert_allclose(fused[:,:16][finite],actual[finite],
                                   rtol=0,atol=5e-6)
        valid_depth=np.isfinite(original_depth)
        np.testing.assert_allclose(fused[:,16:][valid_depth],
                                   original_depth[valid_depth],rtol=0,atol=5e-6)


def test_tof_sees_real_15cm_vertical_uphill_wall():
    data,samples=fixture()
    origin=jnp.asarray([[-.15,0.,.07]],dtype=jnp.float32)
    rays=jnp.asarray([[[1.,0.,-.05]]],dtype=jnp.float32)
    t=np.asarray(ray_surface(jnp,data,tuple(range(TILE_COUNT)),samples,origin,rays))
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
