"""P0A minimal collidable MJX-JAX terrain spike. Run: python scripts/v5018_terrain_spike.py
Requires mujoco, jax and a working MJX-JAX installation. Does NOT benchmark EMU PPO.
"""
import json
import platform
import time
import numpy as np
import mujoco
import jax
import jax.numpy as jnp
from mujoco import mjx

CASES = {
    "flat": '<geom name="ground" type="plane" size="2 2 .1"/>',
    "step_up": '<geom name="ground" type="plane" size="2 2 .1"/><geom name="obstacle" type="box" pos="0 0 .015" size=".15 .3 .015"/>',
    "step_down": '<geom name="lower" type="plane" pos="0 0 -.03" size="2 2 .1"/><geom name="upper" type="box" pos="-.3 0 -.015" size=".3 .4 .015"/>',
    "diagonal": '<geom name="ground" type="plane" size="2 2 .1"/><geom name="ramp" type="box" pos="0 0 -.015" euler="0 8 12" size=".25 .3 .02"/>',
    "log": '<geom name="ground" type="plane" size="2 2 .1"/><geom name="log" type="capsule" fromto="0 -.25 .018 0 .25 .018" size=".012"/>',
}
def model_xml(terrain):
    return f"""<mujoco><option timestep=".005" gravity="0 0 -9.81"/>
    <worldbody>{terrain}<body name="foot" pos="0 0 .08"><freejoint/>
    <geom name="ball" type="sphere" size=".018" mass=".01"/></body></worldbody></mujoco>"""

def run_case(name, terrain):
    model = mujoco.MjModel.from_xml_string(model_xml(terrain))
    cpu = mujoco.MjData(model)
    for _ in range(200):
        mujoco.mj_step(model, cpu)
    cpu_contacts = [(model.geom(i.geom1).name, model.geom(i.geom2).name) for i in cpu.contact]
    mm = mjx.put_model(model)
    initial = mjx.make_data(mm)
    @jax.jit
    def rollout(d):
        def step(c, _):
            nxt = mjx.step(mm, c)
            return nxt, None
        return jax.lax.scan(step, d, None, length=200)[0]
    t0 = time.perf_counter()
    out = rollout(initial)
    out.qpos.block_until_ready()
    return dict(case=name, cpu_z=float(cpu.qpos[2]), mjx_z=float(out.qpos[2]),
                z_delta_m=float(out.qpos[2]-cpu.qpos[2]),
                cpu_contacts=cpu_contacts, mjx_contact_count=int(np.asarray(out.contact.dist < 0).sum()),
                compile_and_run_s=round(time.perf_counter()-t0,3),
                cpu_contact_count=int(cpu.ncon))

def main():
    print(json.dumps({"python":platform.python_version(),"mujoco":mujoco.__version__,
                      "jax":jax.__version__,"devices":[str(x) for x in jax.devices()]},indent=2))
    results=[]
    for name,terrain in CASES.items():
        try:
            result=run_case(name,terrain)
            results.append(result)
            print(json.dumps(result))
        except Exception as exc:
            results.append({"case":name,"error":repr(exc)})
            print(json.dumps(results[-1]))
    with open("v5018_p0a_spike_results.json","w") as f:
        json.dump(results,f,indent=2)
    if any("error" in r for r in results):
        raise SystemExit("P0A spike incomplete: see JSON errors")
if __name__=="__main__":
    main()
