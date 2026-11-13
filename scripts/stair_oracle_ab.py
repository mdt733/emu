"""Isolated matched-sensor vs privileged-geometry PPO stair laboratory.

Research only. Never exports a .emupolicy or touches main's checkpoint layout.
This reuses EMU's ONE MJX environment and ONE PPO implementation, with exactly
the same rewards/physics and fixed stair teaching schedule in both arms.

Example (separate clean directories):
    python -m scripts.stair_oracle_ab --mode sensor --device cuda \
        --worlds 2048 --steps 100000000 --run-dir ~/emu-data/runs/oracle-ab-sensor
    python -m scripts.stair_oracle_ab --mode oracle --device cuda \
        --worlds 2048 --steps 100000000 --run-dir ~/emu-data/runs/oracle-ab-oracle
    python -m scripts.stair_oracle_ab --mode sham --device cuda \\
        --worlds 2048 --steps 100000000 --run-dir ~/emu-data/runs/oracle-ab-sham

The sham is the primary matched control: identical 390-input architecture,
identical initial weights/seed, and the 10 added channels zeroed. The 360-input
sensor mode is a secondary deployed-architecture reference.
The oracle actor sees 10 research-only true-geometry inputs per history
frame (390 inputs total); the sensor actor remains exactly 360. These are
INCOMPATIBLE policies. This is an upper-bound information intervention, not
a deployable policy, a privileged critic, or a validated step-learning result.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
from pathlib import Path

import numpy as np


def source_identity() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    def git(*args):
        return subprocess.check_output(
            ["git", *args], cwd=root, text=True).strip()
    if git("status", "--porcelain"):
        raise RuntimeError("Refusing research run from dirty Git source")
    commit = git("rev-parse", "HEAD")
    branch = git("branch", "--show-current")
    if not branch.startswith("research/"):
        raise RuntimeError("Oracle laboratory requires an isolated research/ branch")
    return {"commit": commit, "branch": branch}


def checked_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", required=True, choices=("sensor", "sham", "oracle"))
    p.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    p.add_argument("--worlds", type=int, default=128)
    p.add_argument("--steps", type=int, default=20_000_000,
                   help="absolute world-step target; a short smoke proves no skill")
    p.add_argument("--seed", type=int, default=735)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--resume", action="store_true",
                   help="resume ONLY this exact run/config/source")
    p.add_argument("--foundation-steps", type=int, default=20_000_000)
    p.add_argument("--ramp-end-steps", type=int, default=80_000_000)
    p.add_argument("--start-height-mm", type=float, default=5.)
    p.add_argument("--target-height-mm", type=float, default=50.)
    p.add_argument("--checkpoint-every-updates", type=int, default=100)
    p.add_argument("--log-every-updates", type=int, default=10)
    a = p.parse_args()
    if (a.worlds < 4 or a.steps < a.worlds * 32 or a.seed < 0
            or a.foundation_steps < 0
            or a.ramp_end_steps <= a.foundation_steps
            or not 3. <= a.start_height_mm <= a.target_height_mm <= 80.
            or a.checkpoint_every_updates < 1 or a.log_every_updates < 1):
        p.error("Invalid world count, budget, curriculum heights or log cadence")
    return a


def lesson(global_step: int, args) -> dict[str, float]:
    # Identical fixed scheduling for the information-only A/B arms.
    training = global_step >= args.foundation_steps
    progress = np.clip(
        (global_step - args.foundation_steps) /
        (args.ramp_end_steps - args.foundation_steps), 0., 1.)
    height = .001 * (args.start_height_mm +
                     progress * (args.target_height_mm - args.start_height_mm))
    return dict(
        specialist_mix=1. if training else 0.,
        focus_weight_step=1.,
        academy_episode_probability=.75 if training else 0.,
        terrain_exposure=1. if training else 0.,
        terrain_height_m=height if training else 0.,
        academy_height_up_m=max(.003, height),
        academy_height_down_m=.003,
        slope_exposure=0., slope_max_deg=0.,
        steady_difficulty=.35,
        frontier_vx_hi=.25, frontier_vx_lo=-.15,
        frontier_vy=.18, frontier_yaw=1.2,
        athletic_rehearsal_frac=0.,
    )


def atomic_pickle(path: Path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def main():
    args = checked_args()
    # Must be chosen BEFORE importing EMU physical variants.
    os.environ["EMU_PHYSICAL_PROFILE"] = "h40"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    identity = source_identity()
    from emu.ppo import (
        make_trainer, configure_jax, update_running_stats,
    )
    configure_jax(args.device)
    import jax
    import jax.numpy as jnp
    from emu.contracts import ROBOT_XML, OBS_SCHEMA_TERRAIN
    from emu.environment import EmuMJXEnv
    from emu.training_config import (ROLLOUT_STEPS, PPO_EPOCHS, PPO_MINIBATCHES,
                                     PPO_LR, PPO_ENTROPY_COST, PPO_VALUE_COST)

    if args.device == "cuda" and jax.default_backend() != "gpu":
        raise RuntimeError("CUDA requested but JAX has no GPU backend")
    args.run_dir = args.run_dir.expanduser().resolve()
    settings = {
        "kind": "nondeployable-oracle-step-research-v1",
        "source": identity,
        "mode": args.mode,
        "device": args.device, "worlds": args.worlds,
        "steps": args.steps, "seed": args.seed,
        "foundation_steps": args.foundation_steps,
        "ramp_end_steps": args.ramp_end_steps,
        "start_height_mm": args.start_height_mm,
        "target_height_mm": args.target_height_mm,
        "rollout": ROLLOUT_STEPS, "epochs": PPO_EPOCHS,
        "minibatches": PPO_MINIBATCHES,
    }
    cfg = args.run_dir / "research_config.json"
    checkpoint = args.run_dir / "research_only.pkl"
    if args.resume:
        if not cfg.is_file() or not checkpoint.is_file():
            raise RuntimeError("No matching run to resume")
        if json.loads(cfg.read_text()) != settings:
            raise RuntimeError("Research config/source mismatch; refusing resume")
    else:
        if args.run_dir.exists() and any(args.run_dir.iterdir()):
            raise RuntimeError("Refusing to overwrite an existing run directory")
        args.run_dir.mkdir(parents=True, exist_ok=True)
        cfg.write_text(json.dumps(settings, indent=2) + "\n")

    env = EmuMJXEnv(
        str(ROBOT_XML), num_envs=args.worlds, seed=args.seed,
        terrain_step_max_m=.08, observation_schema=OBS_SCHEMA_TERRAIN,
        unified_capability_focus=True, terrain_lesson="coherent-careful",
        academy_stream=True, academy_height_cap_m=.08,
        terrain_carpet=True, terrain_carpet_prisms=True,
        research_step_oracle=(args.mode != "sensor"),
        research_step_oracle_masked=(args.mode == "sham"),
        science_step_approach_m=.22,
        science_step_finish_on_transfer=True,
        science_step_goal_weight=.30,
        science_step_velocity_relief=.8,
        science_foothold_bonus=3.,
        science_bilateral_support_bonus=4.,
        science_supported_cross_bonus=6.,
        science_followthrough_bonus=10.,
    )
    want = 360 if args.mode == "sensor" else 390
    if env.obs_dim != want:
        raise RuntimeError(f"Policy input contract error: {env.obs_dim} != {want}")
    trainer = make_trainer(
        env, seed=args.seed, lr=PPO_LR, rollout=ROLLOUT_STEPS,
        epochs=PPO_EPOCHS, minibatches=PPO_MINIBATCHES,
        entropy_cost=PPO_ENTROPY_COST, value_cost=PPO_VALUE_COST)
    if args.resume:
        with checkpoint.open("rb") as f:
            saved = pickle.load(f)
        state, mean, var, count, completed = saved
        if completed % (args.worlds * ROLLOUT_STEPS):
            raise RuntimeError("Invalid checkpoint step alignment")
    else:
        state = trainer["make_initial_state"](env.params_array(lesson(0, args)))
        mean = jnp.zeros((env.obs_dim,), jnp.float32)
        var = jnp.ones((env.obs_dim,), jnp.float32)
        count = jnp.asarray(1e-4, jnp.float32)
        completed = 0
    batch_steps = args.worlds * ROLLOUT_STEPS
    total_updates = (args.steps + batch_steps - 1) // batch_steps
    start_update = completed // batch_steps
    print(f"[ORACLE-LAB] {args.mode} source={identity['commit']} "
          f"actor_inputs={env.obs_dim} worlds={args.worlds} "
          f"starting={completed:,} target={args.steps:,}", flush=True)
    sums = {}
    metrics_to_keep = (
        "academy_lip_attempt_new", "academy_foothold_new",
        "academy_transfer_new", "academy_followthrough_new",
        "academy_cross_up", "academy_cross_down",
        "academy_episode_attempt", "academy_episode_complete",
        "academy_episode_two_foot_ready", "academy_finish_new",
        "episode_fell",
    )
    log_path = args.run_dir / "rollout.jsonl"
    try:
        for idx in range(start_update, total_updates):
            task = env.params_array(lesson(completed, args))
            state, metrics, aux, raw, _ = trainer["train_update"](
                state, task, mean, jnp.sqrt(jnp.maximum(var, 1e-4)), 1.)
            mean, var, count = update_running_stats(mean, var, count, raw)
            completed += batch_steps
            sample = {k: jnp.sum(metrics[k]) for k in metrics_to_keep}
            sample["mean_reward"] = aux["rollout_reward_mean"]
            sample["approx_kl"] = aux["approx_kl_final"]
            sample["explained_variance"] = aux["explained_variance"]
            row = {k: float(v) for k, v in jax.device_get(sample).items()}
            for k in sums:
                sums[k] += row.get(k, 0.)
            if not sums:
                sums = row.copy()
            if ((idx + 1) % args.log_every_updates == 0 or
                    idx == total_updates - 1):
                height = lesson(completed, args)["academy_height_up_m"]
                summary = dict(
                    step=completed, mode=args.mode,
                    target_up_mm=round(1000 * height, 2), **row,
                    # Rates refer ONLY to ended real Academy episodes.
                    academy_completion_per_attempt=(
                        sums["academy_episode_complete"] /
                        max(1., sums["academy_episode_attempt"])),
                    **{"window_" + k: v for k, v in sums.items()
                       if k in ("academy_episode_attempt", "academy_episode_complete",
                                "academy_foothold_new", "academy_transfer_new",
                                "academy_cross_up", "academy_cross_down")})
                with log_path.open("a") as f:
                    f.write(json.dumps(summary, sort_keys=True) + "\n")
                print(f"[ORACLE-LAB] {completed:,} reward={row['mean_reward']:+.3f} "
                      f"UP={sums['academy_cross_up']:.0f} "
                      f"footholds={sums['academy_foothold_new']:.0f} "
                      f"complete/attempt={sums['academy_episode_complete']:.0f}/"
                      f"{sums['academy_episode_attempt']:.0f} "
                      f"height={height*1000:.1f}mm", flush=True)
                sums = {}
            if ((idx + 1) % args.checkpoint_every_updates == 0 or
                    idx == total_updates - 1):
                atomic_pickle(checkpoint, jax.device_get(
                    (state, mean, var, count, completed)))
    except KeyboardInterrupt:
        atomic_pickle(checkpoint, jax.device_get(
            (state, mean, var, count, completed)))
        print(f"[ORACLE-LAB] saved interruption at {completed:,}", flush=True)
        return
    print(f"[ORACLE-LAB] complete: {args.run_dir}. "
          "Learning result NOT validated by an independent held-out exam.",
          flush=True)


if __name__ == "__main__":
    main()
