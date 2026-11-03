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
        h=jnp.full((1,9*4),.5,dtype=jnp.float32)
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
            poses[mids[k]]=[-.25,0.,0.]
        for k in (2,3):
            poses[mids[k]]=[+.25,0.,rise]
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
    env=EmuMJXEnv(str(ROBOT_XML),num_envs=worlds,seed=735,
                  max_episode_steps=700,terrain_step_max_m=.08,
                  observation_schema=OBS_SCHEMA_TERRAIN,
                  unified_capability_focus=True,
                  terrain_lesson="coherent-careful",
                  academy_stream=True,academy_height_cap_m=.15,
                  terrain_carpet=True,terrain_carpet_prisms=True)
    settings=dict(specialist_mix=1.,terrain_exposure=1.,slope_exposure=1.,
                  slope_max_deg=8.,terrain_height_m=.02,
                  academy_height_up_m=.02,academy_height_down_m=.03,
                  focus_weight_step=1.,academy_episode_probability=1.,
                  athletic_rehearsal_frac=0.)
    state,obs=env.reset(settings)
    assert obs.shape==(worlds,360),obs.shape
    assert len(env.carpet_prism_geom_ids)==PRISM_COUNT
    assert not np.any(np.asarray(env.mj_model.geom_contype[
        list(env.carpet_solid_geom_ids)]))
    hf=np.asarray(jax.device_get(state.carpet_hfields))
    assert np.isfinite(hf).all() and hf.shape==(worlds,36)
    assert np.isfinite(np.asarray(jax.device_get(obs))).all()
    surface=env._carpet_real_top_height(
        state.data,state.data.qpos[:,:3][:,None,:],
        state.carpet_hfields)
    assert np.isfinite(np.asarray(jax.device_get(surface))).all()
    params=env.params_array(settings)
    for _ in range(frames):
        out=env.step(state,jnp.zeros((worlds,6),dtype=jnp.float32),params)
        state=out.state
        assert np.isfinite(np.asarray(jax.device_get(out.obs))).all()
    print(json.dumps(dict(status="PASS",backend=jax.default_backend(),
        worlds=worlds,frames=frames,convex_prisms=PRISM_COUNT,
        flat_seam=seam,physical_4cm_riser=step,
        full_emu_360_actor=True),indent=2),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--worlds",type=int,default=4)
    p.add_argument("--frames",type=int,default=2)
    a=p.parse_args()
    run_probe(a.worlds,a.frames)
