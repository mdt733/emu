"""v5041 design gate: compare the actual MJX physical step responses of
(a) hfield alone, (b) hfield + riser wall, and (c) box-only physical tiles.

In (c), two 50cm solid platforms have flat, exactly collidable top
surfaces and a genuine, MJX-safe vertical face. This prototype does NOT
switch the production trainer until source-correct contact tests pass.
"""
from __future__ import annotations

import json
import jax
import jax.numpy as jnp
import numpy as np
import mujoco
from mujoco import mjx

def model_case(rise,mode):
    if mode not in ("hfield","hfield_wall","box"):raise ValueError(mode)
    xml=f'''<mujoco model="carpet_box_comparison">
     <option timestep=".005" gravity="0 0 0"/>
     <asset>
       <hfield name="a" nrow="2" ncol="2" size=".25 .25 .50 .14"
               elevation=".5 .5 .5 .5"/>
       <hfield name="b" nrow="2" ncol="2" size=".25 .25 .50 .14"
               elevation=".5 .5 .5 .5"/>
     </asset>
     <worldbody>
       <body name="foot" pos="-.018 0 .075">
         <freejoint/>
         <geom name="foot_geom" type="sphere" size=".02" mass=".12"
               contype="1" conaffinity="2"/>
       </body>
       <body name="lower" mocap="true" pos="-.25 0 -30">
         <geom name="lower_geom" type="{'box' if mode=='box' else 'hfield'}"
          {('size=".25 .25 .25"' if mode=='box' else 'hfield="a"')}
          contype="2" conaffinity="0" />
       </body>
       <body name="upper" mocap="true" pos=".25 0 -30">
         <geom name="upper_geom" type="{'box' if mode=='box' else 'hfield'}"
          {('size=".25 .25 .25"' if mode=='box' else 'hfield="b"')}
          contype="2" conaffinity="0" />
       </body>
       <body name="face" mocap="true" pos="0 0 -30">
         <geom name="wall" type="box" size=".005 .245 .20"
           contype="{'2' if mode=='hfield_wall' else '0'}" conaffinity="0" />
       </body>
     </worldbody></mujoco>'''
    m=mujoco.MjModel.from_xml_string(xml)
    data=mujoco.MjData(m)
    for body,pos in (("lower",[-.25,0,-.25]),
                     ("upper",[+.25,0,rise-.25]),
                     ("face",[+.005,0,rise-.20])):
        bid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,body)
        data.mocap_pos[m.body_mocapid[bid]]=pos
    data.qpos[:3]=[-.018,0,rise*.5]
    mujoco.mj_forward(m,data)
    return m,data

def exam(rise,mode):
    model,cpu=model_case(rise,mode)
    mj=mjx.put_model(model)
    d=mjx.make_data(mj).replace(qpos=jnp.asarray(cpu.qpos),
                                mocap_pos=jnp.asarray(cpu.mocap_pos))
    d=jax.jit(lambda v:mjx.forward(mj,v))(d)
    c=d._impl.contact
    names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i)
           for i in range(model.ngeom)]
    contacts=[]
    for a,b,dist,frm in zip(np.asarray(c.geom1),np.asarray(c.geom2),
                            np.asarray(c.dist),np.asarray(c.frame)):
        if min(int(a),int(b))<0 or not np.isfinite(dist) or dist>.003:continue
        contacts.append(dict(pair=(names[int(a)],names[int(b)]),
                        dist=round(float(dist),5),
                        normal=[round(float(t),3) for t in frm[0]]))
    def run(v):
        def tick(d,_):return mjx.step(mj,d),None
        return jax.lax.scan(tick,v,None,length=8)[0]
    out=jax.jit(run)(d)
    delta=np.asarray(out.qpos[:3])-np.asarray(cpu.qpos[:3])
    return dict(rise_cm=rise*100,mode=mode,contacts=contacts,
                dz_8_steps_m=round(float(delta[2]),7),
                dx_8_steps_m=round(float(delta[0]),7))

def main():
    for rise in (.04,.15):
        results=[exam(rise,mode) for mode in ("hfield","hfield_wall","box")]
        for r in results:print(json.dumps(r),flush=True)
        box=results[2]
        good=[r for r in box["contacts"] if "upper_geom" in r["pair"]
              and abs(r["normal"][2])<.5]
        assert good,f"BOX did not deliver horizontal MJX contact: {box}"
        # A perfectly horizontal box-wall normal can still exhibit a
        # tiny z drift from MJX's contact solver (<0.1mm in 8 steps).
        # Compare against an absolute meaningful tolerance, not a
        # baseline case that contains *no collision at all*.
        assert abs(box["dz_8_steps_m"])<.0001,results
        assert box["dx_8_steps_m"]<-.0001,results
    print("MJX box-only true vertical riser PASS",flush=True)

if __name__=="__main__":
    main()
