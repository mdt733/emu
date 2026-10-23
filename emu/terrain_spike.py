"""G3 physical terrain spike: same-topology flat and raised-step MJX worlds.

Runs independently of PPO, curriculum and production observation schemas.
It deliberately FAILS when a buried box behaves like a raised one. Nothing
from this diagnostic may be fed to the actor as a future-terrain oracle.

Run: python -m emu.terrain_spike --worlds 2
Optional scale check: --worlds 256 (then 2048 on a suitable GPU).
"""
from __future__ import annotations

import argparse
import math
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .contracts import SIM_DT

ROBOT_XML = Path(__file__).with_name("robot.xml")
STEP_X_M = 0.20
STEP_HALF_LENGTH_M = 0.13
STEP_HALF_WIDTH_M = 0.24
STEP_HEIGHT_M = 0.04
BURIED_Z_M = -0.50


def spike_model():
    """Compile the actual robot with ONE finite box; no production XML edits.

    Flat worlds bury this same geom completely below the ground plane.
    Its shape, contacts and all MJX structural arrays stay identical.
    """
    import mujoco

    root = ET.parse(ROBOT_XML).getroot()
    world = root.find("worldbody")
    if world is None:
        raise RuntimeError("robot.xml has no worldbody")
    ET.SubElement(world, "geom", {
        "name": "g3_step", "type": "box",
        "pos": f"{STEP_X_M} 0 {STEP_HEIGHT_M / 2}",
        "size": f"{STEP_HALF_LENGTH_M} {STEP_HALF_WIDTH_M} {STEP_HEIGHT_M / 2}",
        "rgba": "0.75 0.45 0.15 1", "friction": "1.5 0.005 0.0001",
        "group": "4",
    })
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    model.opt.timestep = SIM_DT
    gid = lambda name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    floor, step = gid("floor"), gid("g3_step")
    feet = (gid("r_foot_ball"), gid("l_foot_ball"))
    if min(floor, step, *feet) < 0:
        raise RuntimeError("missing floor, step or foot collision geom")
    # A world geom with a compiled identity frame may ignore an updated
    # geom_pos. Clear the compiler shortcut BEFORE mjx.put_model.
    model.geom_sameframe[step] = 0
    model.geom_group[floor] = 4  # diagnostic terrain-only CPU rays
    return model, floor, step, feet


def sensor_rays_cpu(model, step_id: int, height_m: float):
    """Actual MuJoCo collision rays, not fitted floor or synthetic grades.

    A provisional 4x4 fan from the torso origin: down-pitch varies by row,
    azimuth by column. These measure floor/step surfaces only, deliberately
    NOT leg occlusion or a production hardware mount.
    """
    import mujoco

    model.geom_pos[step_id, 2] = height_m / 2 if height_m else BURIED_Z_M
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    origin = np.array([0.0, 0.0, 0.2584], dtype=np.float64)
    geomgroup = np.array([0, 0, 0, 0, 1, 0], dtype=np.uint8)
    rays = []
    for pitch_deg in (55, 65, 70, 75):
        for yaw_deg in (-15, -5, 5, 15):
            pitch, yaw = math.radians(pitch_deg), math.radians(yaw_deg)
            direction = np.array([
                math.cos(pitch) * math.cos(yaw),
                math.cos(pitch) * math.sin(yaw),
                -math.sin(pitch),
            ], dtype=np.float64)
            geom_id = np.array([-1], dtype=np.int32)
            distance = mujoco.mj_ray(
                model, data, origin, direction, geomgroup, 1, -1, geom_id)
            rays.append(float(distance) if distance >= 0 else float("nan"))
    return np.asarray(rays, dtype=np.float64).reshape(4, 4)


def run(worlds: int = 2, steps: int = 12, height_m: float = STEP_HEIGHT_M):
    import jax
    import jax.numpy as jnp
    import mujoco
    from mujoco import mjx

    if worlds < 2 or worlds % 2:
        raise ValueError("worlds must be even and at least 2")
    if steps < 1 or not 0 < height_m <= STEP_HEIGHT_M:
        raise ValueError("steps must be positive; height must be in (0, 0.04] m")

    model, floor, step, feet = spike_model()
    base = mjx.put_model(model)
    active = np.arange(worlds) % 2 == 1
    positions = np.tile(np.asarray(model.geom_pos), (worlds, 1, 1))
    positions[~active, step, 2] = BURIED_Z_M
    positions[active, step, 2] = height_m / 2

    # Model leaves are batched exactly like environment.build_batched_model.
    batched_model = base.tree_replace({"geom_pos": jnp.asarray(positions)})
    model_axes = jax.tree_util.tree_map(lambda _: None, base)
    model_axes = model_axes.tree_replace({"geom_pos": 0})
    initial = jax.vmap(lambda _: mjx.make_data(base))(jnp.arange(worlds))
    # A true robot foot-settling probe, not a free sphere or altered robot.
    # Both feet are placed above the centre of their respective surfaces.
    qpos = initial.qpos.at[:, 0].set(STEP_X_M)
    qpos = qpos.at[:, 2].add(jnp.asarray(active, dtype=jnp.float32) * height_m)
    initial = initial.replace(qpos=qpos)

    def contact_flags(data):
        a, b = data.contact.geom1, data.contact.geom2
        near = data.contact.dist < 0.002
        foot = ((a == feet[0]) | (a == feet[1]) |
                (b == feet[0]) | (b == feet[1]))
        on_floor = jnp.any(near & foot & ((a == floor) | (b == floor)), axis=1)
        on_step = jnp.any(near & foot & ((a == step) | (b == step)), axis=1)
        return jnp.stack([on_floor, on_step], axis=1)

    @jax.jit
    def settle(data):
        data = jax.vmap(mjx.forward, in_axes=(model_axes, 0))(
            batched_model, data)
        def one_tick(d, _):
            d = jax.vmap(mjx.step, in_axes=(model_axes, 0))(
                batched_model, d)
            return d, contact_flags(d)
        data, hit_history = jax.lax.scan(one_tick, data, None, length=steps)
        return data, hit_history

    t0 = time.perf_counter()
    final, contacts = settle(initial)
    contacts.block_until_ready()
    compile_and_run_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    final, contacts = settle(initial)
    contacts.block_until_ready()
    steady_s = time.perf_counter() - t1

    touched = np.asarray(jax.device_get(contacts)).any(axis=0)
    if not np.all(touched[~active, 0]) or np.any(touched[~active, 1]):
        raise AssertionError("flat worlds lack floor contact or touch the buried box")
    if not np.all(touched[active, 1]):
        raise AssertionError("raised worlds lack actual MJX foot/step contact")

    # Check collision and ray geometry against the SAME CPU-compiled model.
    # Sensor proof is diagnostic only; the current actor still has tripod ToF.
    if worlds == 2:
        flat = sensor_rays_cpu(model, step, 0.0)
        raised = sensor_rays_cpu(model, step, height_m)
        if not (np.isfinite(flat).all() and np.isfinite(raised).all()):
            raise AssertionError("terrain sensor rays missed collision surfaces")
        detected = raised < flat - 0.01
        if not np.any(detected):
            raise AssertionError("4x4 rays failed to see finite raised geometry")
        print("ToF distance metres: flat / raised (4x4, CPU MuJoCo geometry)")
        print(np.array2string(flat, precision=3))
        print(np.array2string(raised, precision=3))
        print(f"Rays seeing raised terrain: {int(detected.sum())}/16")

    print(f"MJX foot contacts: flat={int(touched[~active, 0].sum())}/{worlds // 2} "
          f"step={int(touched[active, 1].sum())}/{worlds // 2}")
    print(f"Backend={jax.default_backend()} worlds={worlds} substeps={steps} "
          f"height={height_m:.3f}m first_s={compile_and_run_s:.2f} "
          f"repeat_s={steady_s:.3f} "
          f"sim_steps_per_s={worlds * steps / max(steady_s, 1e-6):.0f}")
    print("PASS: physical contact + terrain-only ray agreement. "
          "NOT YET: foot occlusion, 4x4 actor input, PPO, GPU memory.")
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worlds", type=int, default=2)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--height-m", type=float, default=STEP_HEIGHT_M)
    args = parser.parse_args(argv)
    run(worlds=args.worlds, steps=args.steps, height_m=args.height_m)


if __name__ == "__main__":
    main()
