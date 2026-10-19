"""P0A: Prove world-specific obstacle positions with one compiled MJX topology.

Unlike two different ball positions in the same course, this varies the actual
static box geom position per world using batched `geom_pos` leaves. It measures
a JIT reuse after swapping positions, and compares to CPU MuJoCo controls.
A pass proves only one fixed-size box position leaf, not arbitrary meshes,
heightfields, multiple simultaneous courses, PPO or 2048-world throughput.
"""
from __future__ import annotations

import json
import time

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

STEP_Z=.015
STEPS=200


def xml(x):
    return f"""<mujoco><option timestep=".005"/>
    <worldbody><geom name="floor" type="plane" size="2 2 .1"/>
    <geom name="box" type="box" pos="{x} 0 {STEP_Z}" size=".15 .30 .015"/>
    <body name="foot" pos="0 0 .08">
      <joint name="vertical" type="slide" axis="0 0 1"/>
      <geom name="ball" type="sphere" size=".018" mass=".01"/>
    </body></worldbody></mujoco>"""


def cpu_z(x):
    model=mujoco.MjModel.from_xml_string(xml(x))
    data=mujoco.MjData(model)
    for _ in range(STEPS):
        mujoco.mj_step(model,data)
    return .08 + float(data.qpos[0])


def main():
    cpu = [cpu_z(0.), cpu_z(.5)]
    model = mujoco.MjModel.from_xml_string(xml(0.))
    box = model.geom("box").id
    mm = mjx.put_model(model)
    raw = mjx.make_data(mm)
    stacked_data = jax.tree_util.tree_map(lambda a,b: jnp.stack((a,b)),raw,raw)
    model_axes = jax.tree_util.tree_map(lambda _:None, mm)
    model_axes = model_axes.tree_replace({"geom_pos":0})

    @jax.jit
    def rollout(world_model, worlds):
        def tick(state,_):
            return jax.vmap(mjx.step,in_axes=(model_axes,0))(world_model,state),None
        return jax.lax.scan(tick,worlds,None,length=STEPS)[0]

    def run(xs):
        positions = jnp.stack([
            mm.geom_pos.at[box,0].set(xs[0]),
            mm.geom_pos.at[box,0].set(xs[1])
        ])
        world_model = mm.tree_replace({"geom_pos":positions})
        t0=time.perf_counter()
        final=rollout(world_model,stacked_data)
        jax.block_until_ready(final.qpos)
        result = (np.asarray(final.qpos[:,0])+.08).tolist()
        return result, time.perf_counter()-t0

    primary, first_s = run((0.,.5))
    swapped, second_s = run((.5,0.))
    discrepancy = max(abs(primary[i]-cpu[i]) for i in range(2))
    swapped_discrepancy = max(abs(swapped[i]-cpu[1-i]) for i in range(2))
    passed = bool(discrepancy < .002 and swapped_discrepancy < .002
                  and abs(primary[0]-primary[1]) > .02
                  and abs(swapped[0]-swapped[1]) > .02)
    result={
        "pass":passed,"jax":jax.__version__,"mujoco":mujoco.__version__,
        "devices":[str(x) for x in jax.devices()],
        "same_static_geometry_topology":True,
        "per_world_varied_leaf":"geom_pos",
        "cpu_reference_centre_z_m":cpu,
        "first_world_centre_z_m":primary,
        "swapped_world_centre_z_m":swapped,
        "first_compilation_and_execution_s":round(first_s,3),
        "second_execution_s":round(second_s,3),
        "max_cpu_jax_discrepancy_m":max(discrepancy,swapped_discrepancy),
        "limitations":[
            "Different box positions only, not differing geom sizes/types or topology.",
            "Resting constrained sphere, not robot or actor rollout.",
            "Timing includes JIT overhead and is not a benchmark of 2048 worlds.",
            "JIT cache reuse inferred from same function and shapes, not independently profiled.",
        ]
    }
    print(json.dumps(result,indent=2))
    with open("v5018_per_world_terrain.json","w") as f:
        json.dump(result,f,indent=2)
    if not passed:
        raise SystemExit("Per-world geom_pos variance unproven; P0 remains open")


if __name__=="__main__":
    main()
