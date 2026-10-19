"""P0A MuJoCo CPU/MJX contact + preliminary primitive-ray feasibility.

Run: python scripts/v5018_terrain_spike.py
Optional isolated two-world same-topology test: python scripts/v5018_terrain_spike.py --batch

This is a small vertical-foot geometry test, NOT EMU locomotion, scanner design
validation, a heterogeneous-geometry training benchmark, or a PPO throughput test.
The terrain height/ray and contact fixtures deliberately test different things.
"""
from __future__ import annotations

import argparse
import json
import platform
import time

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx


# A real lowered plane; separate positive-height obstacle; tilted collision slab.
CASES = {
    "flat": {
        "terrain": '<geom name="ground" type="plane" size="2 2 .1"/>',
        "foot_x": 0.0,
        "support": "ground",
        "rays": [("floor", (0.0, 0, .25), (0, 0, -1), "ground")],
    },
    "step_up": {
        "terrain": '<geom name="ground" type="plane" size="2 2 .1"/>'
                   '<geom name="obstacle" type="box" pos="0 0 .015" size=".15 .3 .015"/>',
        "foot_x": 0.0,
        "support": "obstacle",
        "rays": [
            ("step_top", (0, 0, .25), (0, 0, -1), "obstacle"),
            ("far_floor", (.25, 0, .25), (0, 0, -1), "ground"),
            ("vertical_riser", (.27, 0, .015), (-1, 0, 0), "obstacle"),
        ],
    },
    "step_down": {
        "terrain": '<geom name="lower" type="plane" pos="0 0 -.03" size="2 2 .1"/>'
                   '<geom name="upper" type="box" pos="-.22 0 -.015" size=".22 .4 .015"/>',
        "foot_x": .12,  # safely beyond the upper edge, not straddling it
        "support": "lower",
        "rays": [
            ("upper_shelf", (-.12, 0, .25), (0, 0, -1), "upper"),
            ("lower_shelf", (.12, 0, .25), (0, 0, -1), "lower"),
            ("descending_riser", (.20, 0, -.015), (-1, 0, 0), "upper"),
        ],
    },
    "diagonal": {
        "terrain": '<geom name="lower" type="plane" pos="0 0 -.12" size="2 2 .1"/>'
                   '<geom name="ramp" type="box" pos="0 0 -.005" euler="8 8 12" size=".25 .30 .02"/>',
        "foot_x": 0.0,
        "support": "ramp",
        "rays": [
            ("sloped_top", (0, 0, .25), (0, 0, -1), "ramp"),
            ("off_ramp", (.4, 0, .25), (0, 0, -1), "lower"),
        ],
    },
    "log": {
        "terrain": '<geom name="ground" type="plane" size="2 2 .1"/>'
                   '<geom name="log" type="capsule" fromto="0 -.25 .018 0 .25 .018" size=".012"/>',
        "foot_x": 0.0,
        "support": "log",
        "rays": [
            ("log_top", (0, 0, .25), (0, 0, -1), "log"),
            ("log_side", (.10, 0, .018), (-1, 0, 0), "log"),
            ("off_log", (.10, 0, .25), (0, 0, -1), "ground"),
        ],
    },
}


def model_xml(terrain, foot_x=0.0):
    # Vertical slider fixes horizontal position. A free ball would roll off
    # the diagonal incline, confounding contact support with gait stability.
    return f"""<mujoco><option timestep=".005" gravity="0 0 -9.81"/>
      <worldbody>{terrain}
        <body name="foot" pos="{foot_x} 0 .08">
          <joint name="z_slide" type="slide" axis="0 0 1"/>
          <geom name="ball" type="sphere" size=".018" mass=".01"/>
        </body>
      </worldbody></mujoco>"""


def geometry_pairs(model, contact, penetrating_only=False):
    pairs = []
    for g1, g2, distance in zip(np.asarray(contact.geom1),
                                np.asarray(contact.geom2),
                                np.asarray(contact.dist)):
        if penetrating_only and float(distance) >= 0:
            continue
        if int(g1) < 0 or int(g2) < 0:
            continue
        pairs.append((model.geom(int(g1)).name, model.geom(int(g2)).name))
    return pairs


def has_pair(pairs, name):
    return any(name in pair and "ball" in pair for pair in pairs)


def primitive_ray(model, data, origin, direction):
    """JAX static-geometry nearest hit: plane, oriented box, capsule.

    Matches the collision geometry's compiled MuJoCo world transforms.
    Capsule is cylinder + spherical endpoints, not a fake height map.
    Plane is mathematically infinite, exactly as a MuJoCo plane geom.
    """
    specs = []
    supported = {int(mujoco.mjtGeom.mjGEOM_BOX),
                 int(mujoco.mjtGeom.mjGEOM_PLANE),
                 int(mujoco.mjtGeom.mjGEOM_CAPSULE)}
    for i in range(model.ngeom):
        typ = int(model.geom_type[i])
        if typ not in supported or int(model.geom_bodyid[i]) != 0:
            continue
        specs.append((typ, np.asarray(data.geom_xpos[i]).copy(),
                      np.asarray(data.geom_xmat[i]).reshape(3, 3).copy(),
                      np.asarray(model.geom_size[i]).copy()))
    if not specs:
        return None

    @jax.jit
    def intersect(p, v):
        hits = []
        inf = jnp.asarray(jnp.inf, dtype=p.dtype)
        for typ, pos, rot, size in specs:
            x = jnp.asarray(pos, dtype=p.dtype)
            R = jnp.asarray(rot, dtype=p.dtype)
            half = jnp.asarray(size, dtype=p.dtype)
            if typ == int(mujoco.mjtGeom.mjGEOM_PLANE):
                normal = R[:, 2]
                denom = jnp.dot(v, normal)
                t = jnp.dot(x - p, normal) / jnp.where(jnp.abs(denom) > 1e-8, denom, 1.)
                hit = jnp.where((jnp.abs(denom) > 1e-8) & (t >= 0), t, inf)
            elif typ == int(mujoco.mjtGeom.mjGEOM_BOX):
                lo = (p - x) @ R
                ray = v @ R
                parallel = jnp.abs(ray) < 1e-8
                safe = jnp.where(parallel, 1., ray)
                t0 = (-half - lo) / safe
                t1 = (half - lo) / safe
                near = jnp.where(parallel, -inf, jnp.minimum(t0, t1))
                far = jnp.where(parallel, inf, jnp.maximum(t0, t1))
                entry, leave = jnp.max(near), jnp.min(far)
                t = jnp.where(entry >= 0, entry, leave)
                valid = (jnp.all(jnp.where(parallel, jnp.abs(lo) <= half, True))
                         & (leave >= jnp.maximum(entry, 0.)) & (t >= 0))
                hit = jnp.where(valid, t, inf)
            else:
                # MuJoCo capsule's local Z is its segment axis, and size[1]
                # is the cylinder half-length (the two sphere centres).
                axis = R[:, 2]
                length = 2. * half[1]
                A = x - half[1] * axis
                m = p - A
                md, nd = jnp.dot(m, axis), jnp.dot(v, axis)
                perp_m = m - md * axis
                perp_v = v - nd * axis
                aa = jnp.dot(perp_v, perp_v)
                bb = 2. * jnp.dot(perp_m, perp_v)
                cc = jnp.dot(perp_m, perp_m) - half[0] ** 2
                disc = bb ** 2 - 4. * aa * cc
                denom = jnp.where(aa > 1e-10, 2. * aa, 1.)
                roots = jnp.stack(((-bb - jnp.sqrt(jnp.maximum(0., disc))) / denom,
                                   (-bb + jnp.sqrt(jnp.maximum(0., disc))) / denom))
                axial = md + roots * nd
                valid = ((aa > 1e-10) & (disc >= 0) & (roots >= 0)
                         & (axial >= 0) & (axial <= length))
                cyl_hit = jnp.min(jnp.where(valid, roots, inf))
                # Both hemispherical ends. Using full spheres gives the same
                # outer surface as the cylinder union (inside overlaps).
                sph_hits = []
                for C in (A, A + length * axis):
                    oc = p - C
                    sa = jnp.dot(v, v)
                    sb = 2. * jnp.dot(oc, v)
                    sc = jnp.dot(oc, oc) - half[0] ** 2
                    sd = sb ** 2 - 4. * sa * sc
                    sd_sqrt = jnp.sqrt(jnp.maximum(0., sd))
                    safe_a = jnp.where(sa > 1e-10, sa, 1.)
                    rt = jnp.stack(((-sb - sd_sqrt) / (2. * safe_a),
                                    (-sb + sd_sqrt) / (2. * safe_a)))
                    sph_hits.append(jnp.min(jnp.where(
                        (sd >= 0) & (sa > 1e-10) & (rt >= 0), rt, inf)))
                hit = jnp.minimum(cyl_hit, jnp.minimum(*sph_hits))
            hits.append(hit)
        return jnp.min(jnp.stack(hits))
    hit = float(intersect(jnp.asarray(origin, dtype=jnp.float32),
                          jnp.asarray(direction, dtype=jnp.float32)))
    return hit if np.isfinite(hit) else None


def ray_tests(model, data, rays):
    results = []
    foot_id = model.body("foot").id
    for name, org, vec, expected in rays:
        geomid = np.array([-1], dtype=np.int32)
        dist = float(mujoco.mj_ray(
            model, data, np.asarray(org, dtype=np.float64),
            np.asarray(vec, dtype=np.float64), None, 1, foot_id, geomid))
        hit = model.geom(int(geomid[0])).name if dist >= 0 and geomid[0] >= 0 else None
        analytic = primitive_ray(model, data, org, vec)
        applicable = True
        delta_mm = (1000. * abs(analytic - dist)
                    if applicable and analytic is not None and dist >= 0 else None)
        result = {"name": name, "expected": expected, "cpu_hit": hit,
                  "cpu_range_m": dist, "jax_supported_geometry": applicable,
                  "jax_primitive_range_m": analytic if applicable else None,
                  "delta_mm": delta_mm,
                  "pass": hit == expected and (
                      not applicable or (delta_mm is not None and delta_mm <= .1))}
        results.append(result)
    return results


def run_case(name, case):
    model = mujoco.MjModel.from_xml_string(model_xml(case["terrain"], case["foot_x"]))
    cpu = mujoco.MjData(model)
    for _ in range(200):
        mujoco.mj_step(model, cpu)
    body_id = model.body("foot").id
    cpu_contacts = geometry_pairs(model, cpu.contact)
    total_force = 0.0
    for i, c in enumerate(cpu.contact):
        if case["support"] in (model.geom(c.geom1).name, model.geom(c.geom2).name):
            f = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(model, cpu, i, f)
            total_force += float(np.linalg.norm(f[:3]))
    mm = mjx.put_model(model)
    initial = mjx.make_data(mm)
    @jax.jit
    def rollout(d):
        def step(c, _):
            nxt = mjx.step(mm, c)
            return nxt, None
        return jax.lax.scan(step, d, None, length=200)[0]
    start = time.perf_counter()
    out = rollout(initial)
    out.qpos.block_until_ready()
    runtime = time.perf_counter() - start
    # On MuJoCo 3.16 ._impl avoids deprecated direct Data.contact access.
    contact = getattr(out, "_impl", out).contact
    mjx_contacts = geometry_pairs(model, contact, penetrating_only=True)
    cpu_z = float(cpu.xpos[body_id, 2])
    mjx_z = float(np.asarray(out.xpos[body_id, 2]))
    rays = ray_tests(model, cpu, case["rays"])
    passed = (has_pair(cpu_contacts, case["support"]) and
              has_pair(mjx_contacts, case["support"]) and
              abs(mjx_z - cpu_z) < .002 and
              total_force > 0 and all(r["pass"] for r in rays))
    return {"case": name, "pass": bool(passed),
            "expected_support": case["support"],
            "cpu_z": cpu_z, "mjx_z": mjx_z,
            "z_delta_m": mjx_z - cpu_z,
            "cpu_contacts": cpu_contacts, "mjx_contacts": mjx_contacts,
            "cpu_support_force_magnitude_n": total_force,
            "rays": rays, "compile_and_run_s": round(runtime, 3),
            "cpu_contact_count": int(cpu.ncon),
            "mjx_contact_count": len(mjx_contacts)}


def two_world_batch():
    """Two simultaneous copies of a single MJX topology; no timing benchmark."""
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><option timestep=".005"/>'
        '<worldbody><geom name="ground" type="plane" size="2 2 .1"/>'
        '<geom name="step" type="box" pos="0 0 .015" size=".15 .3 .015"/>'
        '<body pos="0 0 .08"><freejoint/>'
        '<geom name="ball" type="sphere" size=".018" mass=".01"/></body>'
        '</worldbody></mujoco>')
    mm = mjx.put_model(model)
    initial = mjx.make_data(mm)
    a = initial.replace(qpos=initial.qpos.at[0].set(0.0))
    b = initial.replace(qpos=initial.qpos.at[0].set(.30))
    data = jax.tree_util.tree_map(lambda x, y: jnp.stack([x, y]), a, b)
    @jax.jit
    def rollout(d):
        def one_step(c, _):
            return jax.vmap(lambda state: mjx.step(mm, state))(c), None
        return jax.lax.scan(one_step, d, None, length=200)[0]
    t0 = time.perf_counter()
    result = rollout(data)
    result.qpos.block_until_ready()
    z = np.asarray(result.qpos[:, 2], dtype=float).tolist()
    return {"worlds": 2, "same_geometry_topology": True,
            "final_foot_centre_z_m": z,
            "expected": ["step", "ground"],
            "pass": bool(abs(z[0] - .048) < .005 and abs(z[1] - .018) < .005),
            "compile_and_run_s": round(time.perf_counter() - t0, 3)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", action="store_true",
                        help="also execute experimental 2-world same-topology JAX vmap")
    args = parser.parse_args()
    info = {"python": platform.python_version(), "mujoco": mujoco.__version__,
            "jax": jax.__version__, "devices": [str(x) for x in jax.devices()]}
    print(json.dumps(info))
    results = []
    for name, case in CASES.items():
        try:
            outcome = run_case(name, case)
        except Exception as exc:
            outcome = {"case": name, "pass": False, "error": repr(exc)}
        results.append(outcome)
        print(json.dumps(outcome))
    batch = None
    if args.batch:
        try:
            batch = two_world_batch()
        except Exception as exc:
            batch = {"pass": False, "error": repr(exc)}
        print("batch:", json.dumps(batch))
    payload = {"environment": info, "cases": results, "batch": batch,
               "limitations": ["z-only constrained sphere fixture, not robot gait",
                               "capsule/log ray comparison requires independent validation",
                               "no JAX heightfield rays; only plane/box/capsule",
                               "no train-loop throughput or VRAM measurement",
                               "same-topology pair is not randomized obstacle geometry"]}
    with open("v5018_p0a_spike_v3_results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    if not all(r.get("pass") for r in results) or (args.batch and not batch.get("pass")):
        raise SystemExit("P0A v3 INCOMPLETE: inspect JSON failure(s); no gate pass")


if __name__ == "__main__":
    main()
