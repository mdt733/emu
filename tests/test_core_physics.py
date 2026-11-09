"""Nine retained physical invariants: prism triangles, sensors, contacts and evidence.

This verifies the *real eight-prism world*, not the old 9-box placeholder.
A 4-world MJX physical trajectory probe is run separately in CI.
"""
from __future__ import annotations
from types import SimpleNamespace

import numpy as np
import jax
import jax.numpy as jnp
import mujoco

from emu.carpet_prism_mjx import (batch_update_prism_mocaps, PRISM_COUNT,
                                  PRISMS_PER_TILE)
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


def test_foot_only_contacts_preserve_mass_and_remove_torso_collision():
    """Remove solver contacts, not the 460g central chassis or its inertia."""
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
    # v5057: compile every taller *real* model on CPU, never merely
    # compare XML strings. Neutral sole contact, leg inertia, scaled spring
    # loading and left/right knee passive torque are physical acceptance.
    from emu.physical_variants import (
        PROFILES, generate_xml, profile_numbers,
        FOOT_RADIUS_M, SERVO_RATED_CONTINUOUS_TORQUE_NM)
    base_source=(Path(__file__).resolve().parents[1]/'emu'/'robot.xml').read_text()
    for variant in PROFILES:
        row=profile_numbers(variant)
        from emu.tof_geometry import (TOF_MOUNT_BODY_M, TOF_RAYS_BODY,
                                      TOF_HARDWARE_ZONE_IDS)
        assert TOF_MOUNT_BODY_M==(0.,0.,-.030)
        assert len(TOF_RAYS_BODY)==16
        # ST UM3109 scene top-right is raw zone 0 after lens flip;
        # raw SPAD array location is NOT the observed scene location.
        assert TOF_HARDWARE_ZONE_IDS==(3,2,1,0,7,6,5,4,
                                       11,10,9,8,15,14,13,12)
        np.testing.assert_allclose(
            np.linalg.norm(np.asarray(TOF_RAYS_BODY),axis=1),
            np.ones(16),atol=1e-6)
        # At a 40cm chassis height the 45° FoV stays forward of its
        # own mount, and can reach level ground inside a 1m half tile.
        assert all(v[2]<0 for v in TOF_RAYS_BODY)
        assert all(v[0]>0 for v in TOF_RAYS_BODY)
        max_forward=max((row['hip_height_m']-.030)*v[0]/-v[2]
                        for v in TOF_RAYS_BODY)
        assert max_forward<.5, max_forward
        body=mujoco.MjModel.from_xml_string(
            generate_xml(base_source, variant))
        state=mujoco.MjData(body)
        mujoco.mj_forward(body,state)
        assert body.nu==6 and body.ntendon==2
        assert body.nq==15 and body.nv==14
        np.testing.assert_allclose(sum(body.body_mass),
                                   row['mass_kg'], atol=1e-4, rtol=0)
        torso_id=mujoco.mj_name2id(
            body,mujoco.mjtObj.mjOBJ_BODY,'torso')
        # All six 55g servos are carried by the torso. No knee motors.
        # COM is based on light carbon mass, not an invisible compensating
        # translation of the chassis at different robot heights.
        np.testing.assert_allclose(
            state.subtree_com[torso_id,0],row['neutral_com_x_m'],
            atol=2e-4,rtol=0)
        torso_geom=mujoco.mj_name2id(
            body,mujoco.mjtObj.mjOBJ_GEOM,'torso_box')
        np.testing.assert_allclose(body.geom_pos[torso_geom,0],.005,atol=1e-7)
        assert body.geom_pos[torso_geom,0]-body.geom_size[torso_geom,0]<-.01
        assert .49 <= body.body_mass.sum() <= .52
        for side in 'rl':
            assert mujoco.mj_name2id(
                body,mujoco.mjtObj.mjOBJ_GEOM,f'{side}_thigh_rod_b')==-1
            thigh_id=mujoco.mj_name2id(
                body,mujoco.mjtObj.mjOBJ_GEOM,f'{side}_thigh_rod')
            assert body.geom_type[thigh_id]==mujoco.mjtGeom.mjGEOM_BOX
            np.testing.assert_allclose(body.geom_size[thigh_id,:2],
                                       [.001,.001],atol=1e-7)
            assert mujoco.mj_name2id(
                body,mujoco.mjtObj.mjOBJ_GEOM,f'{side}_knee_servo')==-1
        foot_heights=[]
        knee_loads=[]
        for side in 'rl':
            fi=mujoco.mj_name2id(body,mujoco.mjtObj.mjOBJ_BODY,
                                f'{side}_foot')
            ki=mujoco.mj_name2id(body,mujoco.mjtObj.mjOBJ_JOINT,
                                f'{side}_knee')
            ti=mujoco.mj_name2id(body,mujoco.mjtObj.mjOBJ_TENDON,
                                f'{side}_knee_support')
            foot_heights.append(float(state.xpos[fi,2]))
            knee_loads.append(float(state.qfrc_passive[
                body.jnt_dofadr[ki]]))
            flex=mujoco.mj_name2id(body,mujoco.mjtObj.mjOBJ_JOINT,
                                  f'{side}_shin_flex')
            np.testing.assert_allclose(
                body.jnt_stiffness[flex],
                row['shin_flex_stiffness_nm_rad'],atol=1e-5,rtol=0)
            np.testing.assert_allclose(
                state.ten_length[ti],.060*(2**-.5),
                atol=1e-5,rtol=0)
        np.testing.assert_allclose(
            foot_heights,[FOOT_RADIUS_M]*2,atol=2e-5,rtol=0)
        assert all(t>0 for t in knee_loads)
        np.testing.assert_allclose(knee_loads[0],knee_loads[1],
                                   atol=1e-5,rtol=0)
        k=float(row['knee_strut_k_n_m'])
        expected=k*.060*(1.-2**-.5)*.030*(2**-.5)
        np.testing.assert_allclose(knee_loads,[expected]*2,
                                   atol=.01,rtol=.02)
        rated=SERVO_RATED_CONTINUOUS_TORQUE_NM
        np.testing.assert_allclose(body.actuator_forcerange,
             np.tile([-rated,rated],(6,1)),
             atol=1e-6,rtol=0)
        assert np.all(body.actuator_forcelimited == 1)
        # Speed feedback is active actuator torque, inside the hard
        # rated force limit, not an unlimited passive joint brake.
        dofs=[body.jnt_dofadr[body.actuator_trnid[i,0]]
              for i in range(body.nu)]
        np.testing.assert_allclose(body.dof_damping[dofs],0.,atol=1e-8)
        np.testing.assert_allclose(body.actuator_biasprm[:,2],
                                   -.624,atol=1e-6)
        for rate in (-4.7165,4.7165):
            mujoco.mj_resetData(body,state)
            state.ctrl[:] = 0.
            state.qvel[dofs] = rate
            mujoco.mj_forward(body,state)
            assert np.max(np.abs(state.actuator_force)) <= rated+1e-6
            assert np.max(np.abs(state.actuator_force)) >= rated*.90
            assert np.max(np.abs(state.qfrc_actuator[dofs])) <= rated+1e-6
        assert np.all(body.actuator_forcelimited == 1)
        # REAL MuJoCo stress: command every servo to both extreme targets.
        # The delivered actuator and 1:1 generalized forces must NOT exceed
        # 10 kgf.cm, regardless of how unreasonable the requested position is.
        for side in (0,1):
            mujoco.mj_resetData(body,state)
            state.ctrl[:] = body.actuator_ctrlrange[:,side]
            mujoco.mj_forward(body,state)
            assert np.max(np.abs(state.actuator_force)) <= rated + 1e-6
            dofs=[body.jnt_dofadr[body.actuator_trnid[i,0]]
                  for i in range(body.nu)]
            assert np.max(np.abs(state.qfrc_actuator[dofs])) <= rated + 1e-6
            # At least some servos should actually saturate; otherwise this
            # could be a useless small-command test.
            assert np.max(np.abs(state.actuator_force)) > rated * .95
    # v5056: this physical topology is still six motors and eight joint DoFs.
    # Restoring *passive* knee support must not sneak in motors or new DoFs.
    assert light.nu == 6
    assert light.ntendon == 2
    np.testing.assert_allclose(
        light.actuator_forcerange,
        np.tile([-SERVO_RATED_CONTINUOUS_TORQUE_NM,
                 SERVO_RATED_CONTINUOUS_TORQUE_NM], (6, 1)),
        rtol=0, atol=1e-6)
    for side in ('r', 'l'):
        for joint in (f'{side}_hip_pitch', f'{side}_knee'):
            ji=mujoco.mj_name2id(light,mujoco.mjtObj.mjOBJ_JOINT,joint)
            assert light.jnt_stiffness[ji] == 0.0
        assert mujoco.mj_name2id(
            light,mujoco.mjtObj.mjOBJ_TENDON,
            f'{side}_knee_support') >= 0

    # CPU's ACTUAL MuJoCo passive force, not an unconnected curve assertion.
    # With servos unpowered and qvel=0, each tendon must PUSH the folded knee
    # towards straight (q increases), and reach zero at q=+90 degrees.
    # A mere joint spring with equilibrium at +45 fails this contract.
    data = mujoco.MjData(light)
    for knee_deg in (-55., -40., 0., 45., 90.):
        mujoco.mj_resetData(light, data)
        beta = np.deg2rad((90. + knee_deg) / 2.)
        compression = 0.060 * (1.0 - np.sin(beta))
        torque = 750.0 * compression * 0.030 * np.cos(beta)
        for side in ('r', 'l'):
            ji=mujoco.mj_name2id(light,mujoco.mjtObj.mjOBJ_JOINT,
                                 f'{side}_knee')
            data.qpos[light.jnt_qposadr[ji]] = np.deg2rad(knee_deg)
        mujoco.mj_forward(light,data)
        observed = []
        for side in ('r', 'l'):
            ji=mujoco.mj_name2id(light,mujoco.mjtObj.mjOBJ_JOINT,
                                 f'{side}_knee')
            ti=mujoco.mj_name2id(light,mujoco.mjtObj.mjOBJ_TENDON,
                                 f'{side}_knee_support')
            np.testing.assert_allclose(
                data.ten_length[ti],0.060-compression,atol=4e-6,rtol=0)
            observed.append(data.qfrc_passive[light.jnt_dofadr[ji]])
        np.testing.assert_allclose(observed, [torque,torque],
                                   atol=.015,rtol=.03)
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
    # Per-world randomized motor force is ALSO hard bounded, including
    # samples whose nominal hardware strength multiplier is >1.
    from emu.domain_randomization import (
        build_batched_model, DomainRandomizationConfig)
    dofs=[light.jnt_dofadr[light.actuator_trnid[i,0]]
          for i in range(light.nu)]
    varied,_,_=build_batched_model(
        mjx.put_model(light),
        ids={'actuators':np.arange(light.nu), 'act_qvel':np.asarray(dofs)},
        num_envs=8, seed=735,
        config=DomainRandomizationConfig(
            enabled=True,robot_radius=1.,hardware_radius=1.))
    varied_limits=np.asarray(varied.actuator_forcerange)
    assert varied_limits.shape == (8,6,2)
    assert np.max(np.abs(varied_limits)) <= SERVO_RATED_CONTINUOUS_TORQUE_NM+1e-6


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
    assert TILE_M == 1.0, 'Four 1m source tiles must match CPU sensor lab'
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


def _check_both_physical_triangle_normals_respect_curriculum(
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


def test_both_physical_triangle_normals_respect_curriculum():
    # Academy cannot retreat into a first-lip classifiability dead zone.
    # Test ACTUAL delivered first seam for both possible spawn cells,
    # including independent per-tile base jitter; no fake nominal heights.
    from emu.curriculum import RegimeGovernor
    from emu.terrain_carpet_mjx import (
        make_profiles, tile_level, PROFILE_HALF)
    assert RegimeGovernor.ACADEMY_MIN_LIP_M >= .003
    seeds=np.arange(4096,dtype=np.int32)+37001
    frontier=np.full(len(seeds),.003,dtype=np.float32)
    px,py=make_profiles(np,seeds,frontier,frontier)
    y=np.zeros(len(seeds),dtype=np.int32)
    for left,right in ((-1,0),(0,1)):
        a=tile_level(np,np.full(len(seeds),left,dtype=np.int32),
                     y,seeds,px,py,frontier,frontier)
        b=tile_level(np,np.full(len(seeds),right,dtype=np.int32),
                     y,seeds,px,py,frontier,frontier)
        delivered=np.abs(b-a)
        assert np.all(delivered>.0015), float(delivered.min())
        near=(delivered>=.78*.003)&(delivered<=1.05*.003)
        assert near.mean()>.65,(left,right,float(near.mean()))
    # Keep all six slope/crease envelopes in one physical-invariant test.
    # These variants are not six independent tests.
    for slope_deg, transition_deg in (
            (.7,3.), (3.,3.), (8.,3.), (15.,3.), (15.,1.), (10.,.5)):
        _check_both_physical_triangle_normals_respect_curriculum(
            slope_deg, transition_deg)


def _check_real_top_supported_crossing_in_both_axes(rise,axis,up,down):
    t=trace(rise=rise,axis=axis)
    result=score_prism_crossings(t)
    assert (result["up"],result["down"],result["total"])==(up,down,1)
    assert result["worlds"][0]["independently_supported_lips"][0]["frame"]==6


def test_real_top_supported_crossing_in_both_axes():
    # Exercise all four signed/directional physical crossings together.
    for rise, axis, up, down in (
            (.02,0,1,0), (-.035,0,0,1),
            (.003,1,1,0), (-.02,1,0,1)):
        _check_real_top_supported_crossing_in_both_axes(
            rise, axis, up, down)


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

