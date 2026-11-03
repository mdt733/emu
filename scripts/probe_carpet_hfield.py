"""Preflight: can MJX-JAX use runtime-updated 2x2 heightfield vertices?

No policy, no training and no VRAM claims. A 2x2 hfield = two actual
collidable triangles, and nine such 0.5m tiles would be one local carpet.
"""
from __future__ import annotations

import argparse
import json
import time


def probe():
    import numpy as np
    import jax
    import jax.numpy as jnp
    import mujoco
    from mujoco import mjx

    xml="""<mujoco model="tile_probe">
      <option timestep="0.005" gravity="0 0 -9.81" />
      <asset><hfield name="patch" nrow="2" ncol="2" size=".25 .25 .25 .04"/></asset>
      <worldbody>
        <geom name="tile" type="hfield" hfield="patch" pos="0 0 0"
              friction="1.5 .005 .0001" contype="1" conaffinity="1"/>
        <body name="ball" pos="0 0 .28">
          <freejoint/><geom name="foot" type="sphere" size=".04" mass=".1"
                          contype="1" conaffinity="1"/>
        </body>
      </worldbody>
    </mujoco>"""
    m=mujoco.MjModel.from_xml_string(xml)
    m.hfield_data[:]=np.asarray([0,.1,.2,.3],dtype=np.float32)
    # CPU MuJoCo must first accept real collision triangles and the
    # per-vertex hfield payload before using the MJX compiler.
    d=mujoco.MjData(m)
    mujoco.mj_forward(m,d)
    mj=mjx.put_model(m)
    result={"mujoco_version":mujoco.__version__,
            "jax_version":jax.__version__,
            "hfield_data_present":hasattr(mj,"hfield_data"),
            "n_hfield":int(m.nhfield),"elevation_shape":list(m.hfield_data.shape)}
    if not hasattr(mj,"hfield_data"):
        raise RuntimeError("MJX Model lacks hfield_data: no dynamic terrain")
    if len(mj.hfield_data)!=4:
        raise RuntimeError("Expected four data samples for two physical triangles")
    # The model data changes per world without compiling a new XML.
    # This is a structural feasibility test, NOT a 2048-env VRAM test.
    zero=jnp.zeros_like(mj.hfield_data)
    one=jnp.asarray([0.,.05,.2,.3],dtype=jnp.float32)
    dat=mjx.make_data(mj)
    def one_forward(samples):
        model=mj.replace(hfield_data=samples)
        return mjx.forward(model,dat).geom_xpos
    batched=jax.jit(jax.vmap(one_forward))(jnp.stack((zero,one)))
    if batched.shape[0]!=2 or not bool(jnp.all(jnp.isfinite(batched))):
        raise RuntimeError("Per-world runtime hfield update failed")
    result["dynamic_batched_model"]=True
    # Actual gravity/contact simulation with different terrain arrays.
    def simulate(samples):
        model=mj.replace(hfield_data=samples)
        data=mjx.make_data(model)
        def tick(d,_):
            d=mjx.step(model,d)
            return d,None
        data,_=jax.lax.scan(tick,data,None,length=12)
        return data.qpos[2]
    t0=time.perf_counter()
    height=jax.jit(jax.vmap(simulate))(jnp.stack((zero,one)))
    height.block_until_ready()
    result["12_steps_final_root_z"]=[float(v) for v in height]
    result["compile_and_run_s"]=round(time.perf_counter()-t0,3)
    if not bool(jnp.all(jnp.isfinite(height))):
        raise RuntimeError("MJX heightfield dynamics produced non-finite root positions")
    result["status"]="PASS"
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    result=probe()
    print(json.dumps(result,indent=2,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
