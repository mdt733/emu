"""Real MJX-JAX contact + EMU six-servo triangular-prism acceptance.

Not a performance claim or evidence of learned locomotion.
"""
from __future__ import annotations
import argparse
import json
import xml.etree.ElementTree as ET
import numpy as np


def run_probe(worlds=4, frames=2):
    import jax
    import jax.numpy as jnp
    import mujoco
    from mujoco import mjx
    from emu.carpet_prism_mjx import (
        prism_mesh_xml, PRISM_COUNT, PRISM_BODY_NAMES,
        PRISM_GEOM_NAMES, PrismMeshBuilder)
    from emu.contracts import ROBOT_XML, OBS_SCHEMA_TERRAIN
    from emu.environment import EmuMJXEnv
    from emu.terrain_carpet_mjx import (validate_prism_only_geoms, TILE_COUNT, TILE_M, TILE_HALF_M, TILE_SIZE_Z_M)

    def sphere_contact(rise):
        root=ET.fromstring('''<mujoco model="prism_sphere_exam">
          <option gravity="0 0 0" timestep=".005"/>
          <worldbody>
            <body name="ball" pos="-.016 0 .018"><freejoint/>
              <geom name="ball_geom" type="sphere" size=".020" mass=".10"
                    contype="1" conaffinity="4"/>
            </body>
          </worldbody></mujoco>''')
        prism_mesh_xml(root,active=True)
        model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding="unicode"))
        geom_ids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,name)
                  for name in PRISM_GEOM_NAMES]
        body_ids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,name)
                  for name in PRISM_BODY_NAMES]
        template=mjx.put_model(model)
        builder=PrismMeshBuilder(model,template,geom_ids)
        h=jnp.full((1,TILE_COUNT*4),.5,dtype=jnp.float32)
        meshes=builder.deform(jnp,h,template._impl.mesh_convex)
        # Probe a single MJX scene, strip the optional leading world axis.
        meshes=tuple(m.replace(
            vert=m.vert[0],face=m.face[0],face_normal=m.face_normal[0],
            edge_face_normal=m.edge_face_normal[0])
            if i in {v['mesh_id'] for v in builder.meta} else m
            for i,m in enumerate(meshes))
        dynamic=template.replace(_impl=template._impl.replace(mesh_convex=meshes))
        mids=[int(model.body_mocapid[b]) for b in body_ids]
        poses=np.zeros((model.nmocap,3),dtype=np.float32)
        poses[:,2]=-40.
        for k in (0,1):
            poses[mids[k]]=[-TILE_HALF_M,0.,0.]
        for k in (2,3):
            poses[mids[k]]=[+TILE_HALF_M,0.,rise]
        data=mjx.make_data(template).replace(
            qpos=jnp.asarray([-.016,0.,.018,1.,0.,0.,0.]),
            mocap_pos=jnp.asarray(poses))
        out=jax.jit(lambda mm,dd:mjx.forward(mm,dd))(dynamic,data)
        c=out._impl.contact
        ids1=np.asarray(jax.device_get(c.geom1))
        ids2=np.asarray(jax.device_get(c.geom2))
        dist=np.asarray(jax.device_get(c.dist))
        frame=np.asarray(jax.device_get(c.frame))
        normals=frame[:,0,:] if frame.ndim==3 else frame[:,:3]
        actual=[]
        for g1,g2,dd,n in zip(ids1,ids2,dist,normals):
            if min(int(g1),int(g2))<0 or not np.isfinite(dd) or dd>.004:continue
            actual.append(dict(pair=[int(g1),int(g2)],dist=float(dd),
                               normal=[float(x) for x in n]))
        if rise>0:
            upper=set(geom_ids[2:4])
            walls=[c for c in actual if any(g in upper for g in c['pair'])
                   and abs(c['normal'][2])<.55]
            if not walls:
                raise AssertionError(f"No actual horizontal riser contact: {actual}")
        seam_wall=[c for c in actual if abs(c['normal'][2])<.55] if rise==0 else []
        return dict(rise_m=rise,contacts=actual,flat_seam_walls=seam_wall)

    seam=sphere_contact(0.)
    step=sphere_contact(.04)
    # Stress physical skirt depth: a high tile must still have a real side
    # collider BELOW its top, not open air at lower terrain elevation.
    # 40cm here probes geometry integrity, NOT an earned 40cm STEP frontier.
    deep_step=sphere_contact(.40)
    env=EmuMJXEnv(str(ROBOT_XML),num_envs=worlds,seed=735,
                  max_episode_steps=700,terrain_step_max_m=.08,
                  observation_schema=OBS_SCHEMA_TERRAIN,
                  unified_capability_focus=True,
                  terrain_lesson="coherent-careful",
                  academy_stream=True,academy_height_cap_m=.15,
                  terrain_carpet=True,terrain_carpet_prisms=True,
                  attitude_profile='v5015',pitch_command_max_deg=8.0)
    settings=dict(specialist_mix=1.,terrain_exposure=1.,slope_exposure=1.,
                  slope_max_deg=8.,terrain_height_m=.02,
                  academy_height_up_m=.02,academy_height_down_m=.03,
                  focus_weight_step=1.,academy_episode_probability=1.,
                  athletic_rehearsal_frac=0.)
    state,obs=env.reset(settings)
    assert obs.shape==(worlds,360),obs.shape
    assert len(env.carpet_prism_geom_ids)==PRISM_COUNT
    validate_prism_only_geoms(mujoco,env.mj_model)
    hf=np.asarray(jax.device_get(state.carpet_hfields))
    assert np.isfinite(hf).all() and hf.shape==(worlds,TILE_COUNT*4)
    assert np.isfinite(np.asarray(jax.device_get(obs))).all()
    surface=env._carpet_real_top_height(
        state.data,state.data.qpos[:,:3][:,None,:],
        state.carpet_hfields)
    assert np.isfinite(np.asarray(jax.device_get(surface))).all()
    # Validate that EACH of the eight deformed MJX convex collision meshes
    # has top corners exactly equal to the source 2x2 height samples.
    # A 3mm training lip cannot tolerate an 8mm fitted-box discrepancy.
    live=env._carpet_dynamic_model(state.carpet_hfields)
    meshes=live._impl.mesh_convex
    xpos=np.asarray(jax.device_get(state.data.geom_xpos))
    xmat=np.asarray(jax.device_get(state.data.geom_xmat))
    mocap=np.asarray(jax.device_get(state.data.mocap_pos))
    max_vertex_error=0.
    for w in range(worlds):
        for tile in range(TILE_COUNT):
            original=mocap[w,env.carpet_mocap_ids[tile]]
            centre=original[:2]
            target_xy=np.array([
                centre+[-TILE_HALF_M,-TILE_HALF_M],centre+[TILE_HALF_M,-TILE_HALF_M],
                centre+[-TILE_HALF_M,TILE_HALF_M],centre+[TILE_HALF_M,TILE_HALF_M]])
            target_z=original[2]+hf[w,4*tile:4*tile+4]*TILE_SIZE_Z_M
            cloud=[]
            for tri in (2*tile,2*tile+1):
                geom=env.carpet_prism_geom_ids[tri]
                mid=int(env.mj_model.geom_dataid[geom])
                verts=np.asarray(jax.device_get(meshes[mid].vert))[w]
                rot=xmat[w,geom].reshape(3,3)
                cloud.extend(verts@rot.T+xpos[w,geom])
            cloud=np.asarray(cloud)
            for corner,height in zip(target_xy,target_z):
                pair=np.linalg.norm(cloud[:,:2]-corner[None,:],axis=1)
                candidates=cloud[pair<1e-4]
                if not len(candidates):
                    raise AssertionError(
                        f"Prism {tile} in world {w} missing exact corner {corner}")
                error=abs(np.max(candidates[:,2])-height)
                max_vertex_error=max(max_vertex_error,float(error))
                if error>3e-5:
                    raise AssertionError(
                        f"Physical prism vertex differs from sensor by {error} m")
    if worlds>1 and np.max(np.abs(hf[0]-hf[1]))<1e-5:
        raise AssertionError("Distinct worlds unexpectedly share identical terrain")

    # Shared smooth edge endpoints must agree after subtracting their
    # intentional tile-level offset. Different offsets mean a vertical
    # riser, never an accidental wedge/twist between neighbouring tiles.
    cells=np.asarray(jax.device_get(state.carpet_center))
    from emu.terrain_carpet_mjx import slot_indices
    xs,ys=slot_indices(jnp,state.carpet_center[:,0],state.carpet_center[:,1])
    gridx=np.asarray(jax.device_get(xs))
    gridy=np.asarray(jax.device_get(ys))
    max_edge_wedge=0.
    for w in range(worlds):
        bycell={(int(gridx[w,t]),int(gridy[w,t])):t for t in range(TILE_COUNT)}
        for (i,j),a in bycell.items():
            az=mocap[w,env.carpet_mocap_ids[a],2]+hf[w,4*a:4*a+4]*TILE_SIZE_Z_M
            for di,dj,ia,ib in ((1,0,(1,3),(0,2)),
                                (0,1,(2,3),(0,1))):
                b=bycell.get((i+di,j+dj))
                if b is None:
                    continue
                bz=mocap[w,env.carpet_mocap_ids[b],2]+hf[w,4*b:4*b+4]*TILE_SIZE_Z_M
                edge_delta=np.asarray([az[ia[0]]-bz[ib[0]],
                                       az[ia[1]]-bz[ib[1]]])
                wedge=abs(edge_delta[0]-edge_delta[1])
                max_edge_wedge=max(max_edge_wedge,float(wedge))
                if wedge>3e-5:
                    raise AssertionError(
                        f"Unexpected tilted seam across {i,j}->{i+di,j+dj}")
    # REAL modulo-two recycling: four prisms from the outgoing column
    # must move as two inseparable top-triangle pairs. Moving retired
    # non-contacting boxes alone is not a valid physics proof.
    old_centers=state.carpet_center
    desired=old_centers+jnp.asarray([1,0],dtype=jnp.int32)[None,:]
    shifted_root=(desired.astype(jnp.float32)+.75)*TILE_M
    fake_qpos=state.data.qpos.at[:,:2].set(shifted_root)
    feet_new=jnp.broadcast_to(shifted_root[:,None,:],(worlds,2,2))
    fake_data=state.data.replace(
        qpos=fake_qpos,
        xpos=state.data.xpos.at[:,env.foot_bodies,:2].set(feet_new))
    moved,after_h,after_cent=env._carpet_recycle(state,fake_data)
    assert np.array_equal(np.asarray(jax.device_get(after_cent)),
                          np.asarray(jax.device_get(desired))),(
                          "Safe root tile advance failed")
    prism_ids=list(env.carpet_prism_mocap_ids)
    source_ids=list(env.carpet_mocap_ids)
    before_pos=np.asarray(jax.device_get(
        state.data.mocap_pos[:,prism_ids,:]))
    moved_pos=np.asarray(jax.device_get(moved.mocap_pos[:,prism_ids,:]))
    changed=np.any(np.abs(moved_pos-before_pos)>1e-5,axis=2)
    if not np.array_equal(changed[:,0::2],changed[:,1::2]):
        raise AssertionError("Two physical prism halves recycled independently")
    if not np.all(changed.sum(axis=1)==4):
        raise AssertionError("Expected exactly 2 retired tiles x 2 prisms")
    moved_samples=np.asarray(jax.device_get(moved.mocap_pos[:,source_ids,:]))
    wanted=moved_samples+np.asarray([0.,0.,.5*TILE_SIZE_Z_M],dtype=np.float32)
    np.testing.assert_allclose(moved_pos[:,0::2,:],wanted,rtol=0,atol=2e-5)
    np.testing.assert_allclose(moved_pos[:,1::2,:],wanted,rtol=0,atol=2e-5)
    # One foot still resting on the retired row must veto the move.
    trailing=((old_centers-jnp.asarray([1,0],dtype=jnp.int32)[None,:]
              ).astype(jnp.float32)+.5)*TILE_M
    feet_old=feet_new.at[:,0,:].set(trailing)
    blocked_data=fake_data.replace(
        xpos=fake_data.xpos.at[:,env.foot_bodies,:2].set(feet_old))
    blocked,blocked_h,blocked_cent=env._carpet_recycle(state,blocked_data)
    assert np.array_equal(np.asarray(jax.device_get(blocked_cent)),
                          np.asarray(jax.device_get(old_centers)))
    assert np.array_equal(np.asarray(jax.device_get(blocked_h)),
                          np.asarray(jax.device_get(state.carpet_hfields)))
    assert np.array_equal(np.asarray(jax.device_get(
        blocked.mocap_pos[:,prism_ids,:])),before_pos),(
        "Prism collider teleported away under an outgoing foot")

    # Compact reset must preserve all MJX forward-derived values for
    # finished worlds and fall back to the full batch on overflow.
    full=env.forward_batched(live,state.data)
    original_capacity=env.reset_forward_capacity
    env.reset_forward_capacity=2
    compact_jit=jax.jit(
        lambda model,data,mask:env._forward_reset_data(model,data,mask))
    masks=(
        ("no_death",[False]*worlds),
        ("packed",[True,False,True,False]+[False]*(worlds-4)),
        ("overflow",[True,True,True,False]+[False]*(worlds-4)),
    )
    for mode,chosen in masks:
        selected=np.asarray(chosen,dtype=bool)
        actual=compact_jit(live,state.data,jnp.asarray(selected))
        reference=state.data if mode=="no_death" else full
        actual_leaves,_=jax.tree_util.tree_flatten_with_path(
            jax.device_get(actual))
        expected_leaves,_=jax.tree_util.tree_flatten_with_path(
            jax.device_get(reference))
        assert len(actual_leaves)==len(expected_leaves)
        for i,((path,got),(expected_path,want)) in enumerate(
                zip(actual_leaves,expected_leaves)):
            # A raw 'leaf 13' error hid a CUDA-only 0.28mm-scale
            # discrepancy in compact vs full reset-forward. Report the
            # physical field name and numeric scale without changing the
            # acceptance tolerance or the production physics.
            field=jax.tree_util.keystr(path) or "<root>"
            assert path==expected_path,(
                f"Reset-forward {mode} field order mismatch "
                f"actual={field}, expected={jax.tree_util.keystr(expected_path)}")
            got=np.asarray(got)
            want=np.asarray(want)
            assert got.shape==want.shape,(
                f"Reset-forward {mode} {field} leaf {i} "
                f"shapes: {got.shape} vs {want.shape}")
            checked_got=got[selected] if mode!="no_death" else got
            checked_want=want[selected] if mode!="no_death" else want
            try:
                np.testing.assert_allclose(
                    checked_got,checked_want,
                    rtol=2e-4,atol=2e-5,equal_nan=True,
                    err_msg=f"Reset-forward {mode} field={field} leaf={i} disagrees")
            except AssertionError as ex:
                difference=np.abs(checked_got-checked_want)
                max_abs=float(np.nanmax(difference))
                scaled=difference/np.maximum(np.abs(checked_want),1e-6)
                max_scaled=float(np.nanmax(scaled))
                raise AssertionError(
                    f"CUDA RESET PARITY FAILURE: mode={mode} field={field} "
                    f"leaf={i} max_abs={max_abs:.9g} "
                    f"max_relative_safeguarded={max_scaled:.9g}; "
                    "strict tolerances rtol=2e-4 atol=2e-5. "
                    "Do not dismiss until physical units are identified.\n"
                    +str(ex)) from ex
    env.reset_forward_capacity=original_capacity
    print("reset-forward PASS: no-death bypass, compact model/data "
          "parity and overflow fallback",flush=True)

    assert env.feet_only_contacts
    # Sentinel checks physical terrain against knee pose rather than
    # introducing even one more (expensive) non-foot contact constraint.
    params=env.params_array(settings)
    idle_action=jnp.zeros((worlds,6),dtype=jnp.float32)
    m=env._metrics(state.data,state.task,idle_action,state.prev_action,
                   state.phase,state.prev_foot_pos,
                   carpet_heights=state.carpet_hfields)
    assert np.isfinite(np.asarray(jax.device_get(
        m['virtual_nonfoot_clearance_m']))).all()
    knee=int(np.asarray(jax.device_get(env.bad_bodies[3])))
    query=state.data.xpos[:,knee:knee+1,:]
    top=env._carpet_real_top_height(state.data,query,state.carpet_hfields)
    assert np.isfinite(np.asarray(jax.device_get(top))).all()
    fake_pose=state.data.xpos.at[:,knee,2].set(top[:,0]-.06)
    struck=env._metrics(state.data.replace(xpos=fake_pose),state.task,
                        idle_action,state.prev_action,state.phase,
                        state.prev_foot_pos,carpet_heights=state.carpet_hfields)
    assert np.all(np.asarray(jax.device_get(
        struck['fall_virtual_contact']))>.5),(
        "Deep knee penetration must terminate despite no MJX knee collider")
    print("virtual-strike PASS: real terrain, knee penetration -> fall",
          flush=True)

    # This same physical smoke must catch telemetry/report regressions:
    # PPO's governor reads episode and terrain evidence every frame.
    # No additional test world or PPO update is required.
    essential_evidence=(
        'episode_done', 'episode_fell', 'episode_timeout',
        'episode_return', 'episode_support', 'physical_slope_deg',
        'terrain_cross_new', 'academy_cross_new',
        'slope_episode_attempt_uphill', 'terrain_episode_success_up',
        'exam_stance_slip_proxy', 'reward_command_tracking',
        'fall_virtual_contact','virtual_nonfoot_strike',
    )
    for _ in range(frames):
        out=env.step(state,jnp.zeros((worlds,6),dtype=jnp.float32),params)
        state=out.state
        assert np.isfinite(np.asarray(jax.device_get(out.obs))).all()
        assert np.isfinite(np.asarray(jax.device_get(out.reward))).all()
        missing=set(essential_evidence)-set(out.metrics)
        assert not missing, f"Missing live curriculum/evaluation evidence: {sorted(missing)}"
        evidence=jax.device_get({name:out.metrics[name]
                                for name in essential_evidence})
        for name,values in evidence.items():
            observed=np.asarray(values)
            assert observed.shape==(worlds,), (name,observed.shape)
            assert np.isfinite(observed).all(), f"Non-finite evidence: {name}"
    print(json.dumps(dict(status="PASS",backend=jax.default_backend(),
        worlds=worlds,frames=frames,convex_prisms=PRISM_COUNT,
        flat_seam=seam,physical_4cm_riser=step,
        deep_sidewall_40cm_geometry_only=deep_step,
        safe_recycle_prisms_moved_per_world=4,
        blocked_recycle_under_foot=True,
        full_emu_360_actor=True,
        physical_vertex_max_error_mm=round(max_vertex_error*1000,6),
        max_unwanted_edge_wedge_mm=round(max_edge_wedge*1000,6)),indent=2),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--worlds",type=int,default=4)
    p.add_argument("--frames",type=int,default=2)
    a=p.parse_args()
    run_probe(a.worlds,a.frames)
