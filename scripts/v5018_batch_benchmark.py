"""P0A full-EMU MJX rollout performance vs flat, without PPO gradients.

Runs each configuration in a FRESH subprocess so GPU allocations and JIT
compilations cannot contaminate the next size. Run after the contact fixture.

    python scripts/v5018_batch_benchmark.py --envs 256 512
    python scripts/v5018_batch_benchmark.py --envs 1024 2048

The output is emu_v5018_batch_benchmark.json (one file per invocation).
This is actual 232-input EMU environment/reset/reward step throughput,
NOT PPO training FPS; the terrain variant still uses pre-P1 plane-relative
foot metrics, so its rewards are NOT interpretable as terrain learning.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET


def obstacle_xml(source: Path, target: Path):
    root = ET.parse(source)
    world = root.getroot().find("worldbody")
    if world is None:
        raise RuntimeError("robot model missing worldbody")
    ET.SubElement(world, "geom", {
        "name": "v5018_benchmark_step", "type": "box",
        "pos": ".17 0 .015", "size": ".12 .30 .015",
        "friction": "1.5 .005 .0001",
    })
    root.write(target, encoding="unicode", xml_declaration=True)
    return target


def worker(case, worlds, frames, repeats):
    # Runtime imports stay in child process so each GPU benchmark is isolated.
    import jax
    import jax.numpy as jnp
    import mujoco
    import numpy as np

    from emu.contracts import ROBOT_XML, ACTION_DIM, OBS_SCHEMA_SENSOR
    from emu.environment import EmuMJXEnv

    if jax.default_backend() != "gpu":
        raise RuntimeError("Expected CUDA GPU backend; do not label a CPU run as GPU throughput")
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="emu5018_bench_") as scratch:
        xml_path = ROBOT_XML
        if case == "step":
            xml_path = obstacle_xml(ROBOT_XML, Path(scratch) / "terrain_robot.xml")
        env = EmuMJXEnv(
            str(xml_path), num_envs=worlds, seed=1701, observation_schema=OBS_SCHEMA_SENSOR,
            sensor_noise_scale=0., push_max_force_n=0., combo_mix_frac=0.,
            combo_probe_frac=0., field_teach_recovery=False)
        build_s = time.perf_counter() - start
        params = env.params_array({
            "probe_cmd_vx": .04, "probe_cmd_vy": 0., "probe_cmd_yaw": 0.,
            "steady_difficulty": 0., "posture_difficulty": 0.})
        state, obs = env.reset({
            "probe_cmd_vx": .04, "probe_cmd_vy": 0., "probe_cmd_yaw": 0.,
            "steady_difficulty": 0., "posture_difficulty": 0.})
        jax.block_until_ready(obs)
        action = jnp.zeros((worlds, ACTION_DIM), dtype=jnp.float32)

        @jax.jit
        def rollout(initial):
            def tick(s, _):
                outcome = env.step(s, action, params)
                return outcome.state, (jnp.mean(outcome.reward),
                                       jnp.mean(outcome.done.astype(jnp.float32)))
            return jax.lax.scan(tick, initial, xs=None, length=frames)

        t0 = time.perf_counter()
        state, metric = rollout(state)
        jax.block_until_ready(state.data.qpos)
        first_s = time.perf_counter() - t0
        durations = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            state, metric = rollout(state)
            jax.block_until_ready(state.data.qpos)
            durations.append(time.perf_counter() - t0)
        elapsed = sum(durations)
        gpu_stats = jax.devices()[0].memory_stats() or {}
        memory = {
            k: int(v) for k, v in gpu_stats.items()
            if k in ("peak_bytes_in_use", "bytes_in_use", "bytes_limit",
                     "pool_bytes", "peak_pool_bytes") and isinstance(v, (int, float))
        }
        # The benchmark course adds a real geom. Check whether the final
        # physical contacts actually involve it; do not presume they do.
        contacts = getattr(state.data, "_impl", state.data).contact
        ids = env.ids
        step_id = (mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_GEOM,
                                     "v5018_benchmark_step") if case == "step" else -1)
        pair_hits = 0
        if step_id >= 0:
            g1 = np.asarray(contacts.geom1)
            g2 = np.asarray(contacts.geom2)
            dist = np.asarray(contacts.dist)
            pair_hits = int(np.sum(((g1 == step_id) | (g2 == step_id)) & (dist < 0)))
        return {
            "case": case, "envs": worlds, "control_frames_per_trial": frames,
            "repeat_count": repeats, "policy_observation": "sim2real_232",
            "model_load_s": build_s,
            "first_rollout_compile_and_run_s": first_s,
            "repeat_wall_seconds": durations,
            "total_steady_step_s": elapsed,
            "steady_env_steps_per_second": worlds * frames * repeats / elapsed,
            "measured_final_step_contact_points": pair_hits,
            "peak_allocator_memory": memory,
            "jax": jax.__version__, "mujoco": mujoco.__version__,
            "backend": jax.default_backend(),
            "devices": [str(d) for d in jax.devices()],
            "final_mean_reward_uninterpretable_on_step": float(np.asarray(metric[0]).mean()),
            "limitations": [
                "Not PPO gradients / update FPS.",
                "Obstacle reward/foot-clearance semantics are still plane-based until P1.",
                "Obstacle foot contact may be zero for this short test.",
                "GPU allocator peak is not the same as total physical GPU VRAM.",
                "Same course geometry in all worlds; no randomized per-world obstacles.",
            ],
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, nargs="+", default=[256, 512],
                        help="batch sizes; test 1024/2048 after inspecting smaller runs")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--case", choices=["flat", "step"], help=argparse.SUPPRESS)
    parser.add_argument("--worlds", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, default=Path("emu_v5018_batch_benchmark.json"))
    args = parser.parse_args()
    if args.frames < 1 or args.repeats < 1 or any(n < 1 for n in args.envs):
        parser.error("positive frames, repeats and environment counts required")
    if args.worker:
        result = worker(args.case, args.worlds, args.frames, args.repeats)
        print("EMU_BENCH_JSON=" + json.dumps(result, allow_nan=False), flush=True)
        return
    rows = []
    for worlds in args.envs:
        for case in ("flat", "step"):
            print(f"=== {case}, {worlds} worlds ===", flush=True)
            cmd = [sys.executable, __file__, "--worker", "--case", case,
                   "--worlds", str(worlds), "--frames", str(args.frames),
                   "--repeats", str(args.repeats)]
            try:
                completed = subprocess.run(cmd, text=True, capture_output=True,
                                           timeout=900, check=False,
                                           env={**os.environ,
                                                "XLA_PYTHON_CLIENT_PREALLOCATE": "false"})
                prefix = "EMU_BENCH_JSON="
                lines = [l for l in completed.stdout.splitlines() if l.startswith(prefix)]
                if completed.returncode != 0 or not lines:
                    record = {"case": case, "envs": worlds, "pass": False,
                              "returncode": completed.returncode,
                              "stderr_tail": completed.stderr[-6000:],
                              "stdout_tail": completed.stdout[-2000:]}
                else:
                    record = json.loads(lines[-1][len(prefix):])
                    record["pass"] = True
            except subprocess.TimeoutExpired as exc:
                record = {"case": case, "envs": worlds, "pass": False,
                          "error": f"timeout after {exc.timeout} s"}
            rows.append(record)
            print(json.dumps(record, allow_nan=False), flush=True)
            args.output.write_text(json.dumps({"runs": rows}, indent=2) + "\n")
            # Don't escalate batch count after failing at a smaller size:
            # the purpose is to establish a safe working ceiling on 8 GB.
        if not all(r.get("pass") for r in rows[-2:]):
            print("Stopping at first failing batch size; inspect JSON.", flush=True)
            break
    print(f"Results: {args.output.resolve()}")
    if not all(r.get("pass") for r in rows):
        raise SystemExit("Batch feasibility INCOMPLETE. Inspect errors; do not claim pass.")


if __name__ == "__main__":
    main()
