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


def spike_model(height_m: float):
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
        "pos": f"{STEP_X_M} 0 {height_m / 2}",
        "size": f"{STEP_HALF_LENGTH_M} {STEP_HALF_WIDTH_M} {height_m / 2}",
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

    These are the 16 centre rays of the viewer's 45-degree-square FOV
    at 0 mm forward, 70 degrees down and 40 mm below torso centre.
    They measure floor/step only, deliberately NOT leg self-occlusion,
    return histogram, sub-ray aggregation or acquisition delays.
    """
    import mujoco

    model.geom_pos[step_id, 2] = height_m / 2 if height_m else BURIED_Z_M
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    origin = np.array([0.0, 0.0, 0.2584 - 0.040], dtype=np.float64)
    geomgroup = np.array([0, 0, 0, 0, 1, 0], dtype=np.uint8)
    rays = []
    tilt = math.radians(70)
    half_fov = math.tan(math.radians(45) / 2)
    for row in range(4):
        for col in range(4):
            # Identical image-plane basis to viewer_src/src/sensor.js
            u = (col - 1.5) * half_fov / 2
            v = (1.5 - row) * half_fov / 2
            direction = np.array([
                math.cos(tilt) + math.sin(tilt) * v,
                u,
                -math.sin(tilt) + math.cos(tilt) * v,
            ], dtype=np.float64)
            direction /= np.linalg.norm(direction)
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

    model, floor, step, feet = spike_model(height_m)
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
        # MJX 3.17 exposes the implementation's contact arrays directly;
        # older MJX versions keep the same arrays on Data.
        contact = getattr(data, "_impl", data).contact
        a, b = contact.geom1, contact.geom2
        near = contact.dist < 0.002
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
    _, contacts = settle(initial)
    contacts.block_until_ready()
    compile_and_run_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    _, contacts = settle(initial)
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


def run_environment(worlds: int = 4):
    """Exercise the EXISTING EmuMJXEnv episodic physics, not a second trainer.

    Prove that changing data.mocap_pos affects real contact inside a single
    compiled model. Teleported stance is a contact diagnostic, not walking.
    """
    import jax
    import jax.numpy as jnp
    import mujoco
    from .environment import EmuMJXEnv, NEUTRAL_ROOT_Z

    if worlds < 2:
        raise ValueError('environment probe needs at least 2 worlds')
    env = EmuMJXEnv(str(ROBOT_XML), num_envs=worlds, seed=5023,
                    terrain_step_max_m=.04, sensor_noise_scale=0,
                    attitude_profile='v5015')
    state, _ = env.reset()
    idx = env.terrain_step_mocap_id
    geom = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_GEOM,
                             'episode_terrain_surface')
    floor = env.ids['floor_geom']
    feet = env.ids['foot_geoms']
    positions = np.asarray(jax.device_get(state.data.mocap_pos))[:, idx, :]
    top = positions[:, 2] + .15
    if not (np.all(top > 0.011) and np.all(top <= .04001)):
        raise AssertionError('episode terrain sampled invalid platform heights')
    # x<0 => raised starting support/step-down; x>0 => approach a step-up.
    down = positions[:, 0] < 0
    if not (down.any() and (~down).any()):
        raise AssertionError('no mixture of step-up and step-down worlds')
    root = np.asarray(jax.device_get(state.data.qpos))
    if not np.allclose(root[:, 2] - top*down, NEUTRAL_ROOT_Z, atol=.04):
        raise AssertionError('step-down births must start at raised support height')

    p = env.params_array({})
    differently_seeded = env.reset_jit(jax.random.PRNGKey(5024), p)
    next_pos = np.asarray(jax.device_get(differently_seeded.data.mocap_pos))[:, idx, :]
    if np.array_equal(positions, next_pos):
        raise AssertionError('episode reset did not move physical terrain')

    centers = jnp.asarray(positions[:, :2], dtype=jnp.float32)
    heights = jnp.asarray(top, dtype=jnp.float32)
    beyond = jnp.stack((jnp.full((worlds,), 3.0), jnp.zeros((worlds,))), axis=1)
    zero = jnp.zeros((worlds,), dtype=jnp.float32)
    @jax.jit
    def contacts_at(xy, elevation):
        # Deliberately position neutral ball feet 2mm into the requested
        # support, then measure contacts before and DURING settling.
        # Contact only at the final frame is not required: these robots
        # can legitimately rebound or lift a foot within 12 substeps.
        q = state.data.qpos.at[:, 0:2].set(xy)
        q = q.at[:, 2].set(NEUTRAL_ROOT_Z + elevation - .002)
        q = q.at[:, 3:7].set(jnp.broadcast_to(
            jnp.asarray([1., 0., 0., 0.], dtype=jnp.float32), (worlds, 4)))
        q = q.at[:, env.act_qpos].set(0.)
        d = state.data.replace(qpos=q, qvel=jnp.zeros_like(state.data.qvel))
        d = env.forward_batched(env.mjx_model, d)

        def flags(physical):
            contact = getattr(physical, '_impl', physical).contact
            a, b = contact.geom1, contact.geom2
            near = contact.dist < .003
            foot = ((a == feet[0]) | (a == feet[1]) |
                    (b == feet[0]) | (b == feet[1]))
            platform = jnp.any(near & foot & ((a == geom) | (b == geom)), axis=1)
            ground = jnp.any(near & foot & ((a == floor) | (b == floor)), axis=1)
            gap = jnp.min(jnp.where(
                ((a == geom) | (b == geom)) & foot,
                contact.dist, jnp.inf), axis=1)
            return platform, ground, gap

        seen_platform, seen_ground, gap = flags(d)
        def one(_, carry):
            physical, p, g, nearest = carry
            physical = env.step_batched(env.mjx_model, physical)
            pn, gn, distance = flags(physical)
            return (physical, p | pn, g | gn, jnp.minimum(nearest, distance))
        d, seen_platform, seen_ground, gap = jax.lax.fori_loop(
            0, 12, one, (d, seen_platform, seen_ground, gap))
        return (seen_platform, seen_ground, d.qpos[:, :3],
                d.geom_xpos[:, geom, :], gap)

    platform_contact, _, on_top, geom_center, gap = jax.device_get(
        contacts_at(centers, heights))
    _, floor_contact, on_floor, _, _ = jax.device_get(
        contacts_at(beyond, zero))
    if not np.all(platform_contact):
        print('G3 physical diagnostics: platform centres', positions)
        print('MJX geom centres after moving mocap body:', geom_center)
        print('Robot pose after physical settling:', on_top)
        print('Closest foot/platform MJX contact distances:', gap)
        print('Per-world platform contact flags:', platform_contact)
        raise AssertionError('MJX could not physically support feet on movable platforms')
    if not np.all(floor_contact):
        raise AssertionError('MJX missed true floor contact beyond the platform')
    if not (np.isfinite(on_top).all() and np.isfinite(on_floor).all()):
        raise AssertionError('unstable episode terrain stance')

    # Inspect the real collision at four stations around the *actual*
    # angled riser. There is no virtual ramp: both sides must support a foot
    # on their own surfaces, in either travel direction. Each station is a
    # deliberately posed physical contact check, NOT a continuous traversal.
    # The later evaluator must measure uninterrupted progress with one actor.
    yaw_quat = np.asarray(jax.device_get(state.data.mocap_quat))[:, idx, :]
    yaw = 2.0*np.arctan2(yaw_quat[:, 3], yaw_quat[:, 0])
    axis = np.stack((np.cos(yaw), np.sin(yaw)), axis=1)
    # Mocap box centre is offset exactly one half-length into the elevated
    # side from the crossing edge. This edge point is shared by up/down.
    from .environment import TERRAIN_PLATFORM_HALF_LENGTH_M
    side = np.where(down, -1.0, 1.0)
    edges = positions[:, :2] - side[:, None]*TERRAIN_PLATFORM_HALF_LENGTH_M*axis
    edge_contacts = []
    for offset in (-.14, -.055, .055, .14):
        station_xy = jnp.asarray(edges + offset*axis, dtype=jnp.float32)
        on_platform = np.where(down, offset < 0, offset > 0)
        station_z = jnp.asarray(np.where(on_platform, top, 0.0), dtype=jnp.float32)
        touched_platform, touched_floor, _, _, _ = jax.device_get(
            contacts_at(station_xy, station_z))
        supported = np.where(on_platform, touched_platform, touched_floor)
        if not np.all(supported):
            print(f'G3 edge station offset {offset:+.3f} m: '
                  f'expected raised={on_platform}, '
                  f'platform={touched_platform}, floor={touched_floor}, '
                  f'edge={edges}')
            raise AssertionError('real MJX support was absent near an angled step edge')
        edge_contacts.append(int(np.sum(supported)))
    print('Angled edge station contacts (approach / before / after / exit): '
          + ' / '.join(f'{n}/{worlds}' for n in edge_contacts))
    # Full stepping path must accept the opt-in data schema and keep it finite.
    stepped = env.step(state, jnp.zeros((worlds, 6), dtype=jnp.float32), p)
    stepped.obs.block_until_ready()
    if not bool(jax.device_get(jnp.all(jnp.isfinite(stepped.obs)))):
        raise AssertionError('episode terrain step produced nonfinite actor observations')
    print(f'PASS: existing environment {worlds} worlds; '
          f'{int(down.sum())} step-down / {int((~down).sum())} step-up; '
          f'platform contact {int(np.sum(platform_contact))}/{worlds}, '
          f'floor beyond {int(np.sum(floor_contact))}/{worlds}, '
          'same topology and different geometry after reset')
    print('NOT YET: traversed step, accurate 4x4 actor sensing, governor, GPU memory')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worlds", type=int, default=2)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--height-m", type=float, default=STEP_HEIGHT_M)
    parser.add_argument("--environment", action="store_true",
                        help="probe reset-varying real terrain inside production EmuMJXEnv")
    args = parser.parse_args(argv)
    if args.environment:
        run_environment(worlds=max(args.worlds, 4))
    else:
        run(worlds=args.worlds, steps=args.steps, height_m=args.height_m)


if __name__ == "__main__":
    main()
