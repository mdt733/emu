"""A/B: does an explicit box at a step face correct MJX's wrong HField
vertical-contact normal? Uses one 20mm sphere and two 2-triangle surfaces.
"""
from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np

from emu.terrain_carpet_mjx import add_carpet_geoms,TILE_SIZE_Z_M


def probe(rise_m=.15,include_wall=False):
    xml='''<mujoco model="uphill_wall_probe">
      <option timestep=".005" gravity="0 0 0" integrator="implicitfast"/>
      <worldbody>
        <body name="probe" pos="-.018 0 .075">
          <freejoint/>
          <geom name="foot" type="sphere" size=".02" mass=".12"
                contype="3" conaffinity="7"/>
        </body>
        <body name="wall" mocap="true" pos="0 0 -30">
          <geom name="wall_face" type="box" size=".005 .245 .20"
                contype="4" conaffinity="0" friction="1.5 .005 .0001"/>
        </body>
      </worldbody>
    </mujoco>'''
    model=mujoco.MjModel.from_xml_string(add_carpet_geoms(xml))
    model.hfield_data[:]=.5
    cpu=mujoco.MjData(model)
    ids={n:mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,n)
         for n in ["wall",*[f"carpet_tile_body_{i}" for i in range(9)]]}
    for body in ids.values():
        cpu.mocap_pos[int(model.body_mocapid[body])]=[0.,0.,-30.]
    cpu.mocap_pos[int(model.body_mocapid[ids["carpet_tile_body_0"]])]=[
        -.25,0.,-TILE_SIZE_Z_M*.5]
    cpu.mocap_pos[int(model.body_mocapid[ids["carpet_tile_body_1"]])]=[
        +.25,0.,rise_m-TILE_SIZE_Z_M*.5]
    if include_wall:
        cpu.mocap_pos[int(model.body_mocapid[ids["wall"]])]=[
            +.005,0.,rise_m-.20]
    cpu.qpos[:3]=[-.018,0.,rise_m*.5]
    mujoco.mj_forward(model,cpu)
    jmodel=mjx.put_model(model)
    data=mjx.make_data(jmodel).replace(
        qpos=jnp.asarray(cpu.qpos),
        mocap_pos=jnp.asarray(cpu.mocap_pos))
    forwarded=jax.jit(lambda d:mjx.forward(jmodel,d))(data)
    contacts=forwarded._impl.contact
    names={i:mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i)
           for i in range(model.ngeom)}
    row=[]
    for a,b,d,frame in zip(
        np.asarray(contacts.geom1),np.asarray(contacts.geom2),
        np.asarray(contacts.dist),np.asarray(contacts.frame)):
        if int(a)<0 or int(b)<0 or not np.isfinite(d) or d>.005:continue
        row.append({"pair":[names.get(int(a)),names.get(int(b))],
                    "depth_m":round(float(d),6),
                    "normal":[round(float(v),3) for v in frame[0]]})
    def step_fn(d):
        def f(q,_):
            return mjx.step(jmodel,q),None
        return jax.lax.scan(f,d,None,length=8)[0]
    stepped=jax.jit(step_fn)(data)
    out=np.asarray(stepped.qpos[:3])
    return {"explicit_wall":include_wall,"rise_cm":rise_m*100,
            "contacts":row,
            "delta_root_xyz_m":[round(float(v),6)
                for v in (out-np.asarray(cpu.qpos[:3]))]}


def main():
    for rise in (.04,.15):
        for use_wall in (False,True):
            print(json.dumps(probe(rise,use_wall),sort_keys=True),flush=True)


if __name__=="__main__":
    main()
