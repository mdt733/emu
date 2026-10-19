"""P0A: CPU MuJoCo vs MJX-JAX actual *moving* spherical-foot impacts.

The simple 10 g sphere is constrained to a horizontal or vertical slider, not
a six-servo learned gait. Tests wall strikes, stepping onto a top, and a
high-clearance trajectory (negative control). JSON includes first-impact
timings and CPU force impulses; MJX impulse agreement is not yet asserted.

Run: python scripts/v5018_dynamic_contact.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

DT = .005
FRAMES = 320

PROBES = (
    ("toe_strike_slow", "1 0 0", "-.16 0 .018", +.16, True),
    ("toe_strike_fast", "1 0 0", "-.16 0 .018", +.35, True),
    ("clearance_above", "1 0 0", "-.16 0 .071", +.35, False),
    ("descend_onto_top", "0 0 1", ".12 0 .12", -.30, True),
)


def fixture(axis, pos):
    return f"""<mujoco>
      <option timestep="{DT}" gravity="0 0 0" integrator="Euler"/>
      <worldbody>
        <geom name="step" type="box" pos=".12 0 .02" size=".12 .14 .02"/>
        <body name="ball_body" pos="{pos}">
          <joint name="slider" type="slide" axis="{axis}" damping="0"/>
          <geom name="ball" type="sphere" size=".018" mass=".01"/>
        </body>
      </worldbody>
    </mujoco>"""


def run(name, axis, pos, speed, expected):
    model = mujoco.MjModel.from_xml_string(fixture(axis, pos))
    ball_id = model.geom("ball").id
    step_id = model.geom("step").id
    cpu = mujoco.MjData(model)
    cpu.qvel[0] = speed
    cpu_times = []
    peak_n = 0.
    impulse_n_s = 0.
    cpu_history = []
    for frame in range(FRAMES):
        mujoco.mj_step(model, cpu)
        cpu_history.append((float(cpu.qpos[0]), float(cpu.qvel[0])))
        for ci, contact in enumerate(cpu.contact):
            if {int(contact.geom1), int(contact.geom2)} == {ball_id, step_id}:
                if contact.dist <= 0:
                    cpu_times.append(frame)
                    forces = np.zeros(6)
                    mujoco.mj_contactForce(model, cpu, ci, forces)
                    force = float(np.linalg.norm(forces[:3]))
                    peak_n = max(peak_n, force)
                    impulse_n_s += force * DT

    mm = mjx.put_model(model)
    initial = mjx.make_data(mm)
    initial = initial.replace(qvel=initial.qvel.at[0].set(speed))
    @jax.jit
    def advance(data):
        def substep(state, _):
            nxt = mjx.step(mm, state)
            contacts = getattr(nxt, "_impl", nxt).contact
            touched = jnp.any(
                (((contacts.geom1 == ball_id) & (contacts.geom2 == step_id)) |
                 ((contacts.geom2 == ball_id) & (contacts.geom1 == step_id)))
                & (contacts.dist <= 0))
            return nxt, (nxt.qpos[0], nxt.qvel[0], touched)
        return jax.lax.scan(substep, data, None, length=FRAMES)

    t0 = time.perf_counter()
    final, traces = advance(initial)
    jax.block_until_ready(final.qpos)
    duration = time.perf_counter() - t0
    q, v, flags = (np.asarray(field) for field in traces)
    mjx_times = np.flatnonzero(flags)
    first_cpu = int(cpu_times[0]) if cpu_times else None
    first_mjx = int(mjx_times[0]) if len(mjx_times) else None
    error_ms = (abs(first_cpu - first_mjx) * DT * 1000.
                if first_cpu is not None and first_mjx is not None else None)
    contact_agreement = bool(cpu_times) == bool(len(mjx_times)) == expected
    passed = bool(contact_agreement and
                  (error_ms is None or error_ms <= 20.) and
                  np.isfinite(q).all() and np.isfinite(v).all())
    cpu_hist = np.asarray(cpu_history)
    return {
        "case": name, "pass": passed,
        "expected_impact": expected, "cpu_impact": bool(cpu_times),
        "mjx_impact": bool(len(mjx_times)),
        "cpu_first_impact_frame": first_cpu,
        "mjx_first_impact_frame": first_mjx,
        "first_impact_timing_disagreement_ms": error_ms,
        "cpu_peak_contact_force_n": peak_n,
        "cpu_integrated_contact_force_n_s": impulse_n_s,
        "max_position_trajectory_error_m": float(
            np.max(np.abs(cpu_hist[:, 0] - q))),
        "cpu_final_joint_position_m": float(cpu.qpos[0]),
        "mjx_final_joint_position_m": float(q[-1]),
        "compiled_and_executed_s": round(duration, 3),
        "limitations": "Fixed-axis sphere impulse fixture, not independent biped foot clearance."
    }


def main():
    results = []
    for probe in PROBES:
        try:
            row = run(*probe)
        except Exception as exc:
            row = {"case": probe[0], "pass": False, "error": repr(exc)}
        results.append(row)
        print(json.dumps(row), flush=True)
    out = Path("v5018_dynamic_contact.json")
    out.write_text(json.dumps({
        "environment": {"jax": jax.__version__, "mujoco": mujoco.__version__,
                        "devices": [str(d) for d in jax.devices()]},
        "results": results}, indent=2) + "\n")
    if not all(r["pass"] for r in results):
        raise SystemExit("Dynamic contact mismatches; P0 not complete")


if __name__ == "__main__":
    main()
