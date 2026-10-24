#!/usr/bin/env python3
"""Source-local, no-checkpoint CUDA/CPU G6 arm smoke before an overnight run.

Checks exact terrain_360 observation width, real exposed steps, finite metrics
and one PPO optimizer update for EACH new training arm. No disk outputs.
"""
from __future__ import annotations

import argparse
import os


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--envs", type=int, default=4)
    args = p.parse_args()
    if not 2 <= args.envs <= 16 or args.envs % 2:
        raise SystemExit("Choose an even 2–16 worlds for tiny G6 smoke")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    from emu.ppo import configure_jax, require_backend, make_trainer, build_obs_norm_mask
    configure_jax(args.device)
    import jax
    import jax.numpy as jnp
    from emu.contracts import OBS_SCHEMA_TERRAIN, TERRAIN_OBS_DIM, ROBOT_XML
    from emu.environment import EmuMJXEnv
    from emu.curriculum import RegimeGovernor
    require_backend(args.device == "cpu")

    for arm in ("speed", "foothold"):
        print(f"G6 {arm}: real 360 observation + PPO gradient...", flush=True)
        env = EmuMJXEnv(str(ROBOT_XML), num_envs=args.envs, seed=733,
                        max_episode_steps=40, observation_schema=OBS_SCHEMA_TERRAIN,
                        attitude_profile="v5015", terrain_step_max_m=.04,
                        terrain_training_arm=arm, phase_speed_gain=1.25,
                        open_frontier=True, combo_mix_frac=.88)
        params = RegimeGovernor().params_dict()
        params.update(terrain_exposure=.15, terrain_height_m=.01,
                      probe_cmd_vx=float("nan"))
        state, obs = env.reset(params)
        obs.block_until_ready()
        assert obs.shape == (args.envs, TERRAIN_OBS_DIM), obs.shape
        assert build_obs_norm_mask(OBS_SCHEMA_TERRAIN).shape == (TERRAIN_OBS_DIM,)
        assert state.hist.shape == (args.envs, 4, 90)
        stepped = env.step(state, jnp.zeros((args.envs,6),dtype=jnp.float32),params)
        stepped.obs.block_until_ready()
        assert stepped.obs.shape == (args.envs,TERRAIN_OBS_DIM)
        for key in ("terrain_foothold_payment", "terrain_episode_attempt_up_slow",
                    "terrain_episode_attempt_up_medium", "terrain_episode_attempt_up_fast"):
            assert key in stepped.metrics and stepped.metrics[key].shape == (args.envs,),key
        learner = make_trainer(env, seed=733, lr=2e-4, rollout=2,
                               epochs=1, minibatches=1)
        initial = learner["make_initial_state"](params)
        output = learner["train_update"](
            initial, env.params_array(params),
            jnp.zeros((TERRAIN_OBS_DIM,),dtype=jnp.float32),
            jnp.ones((TERRAIN_OBS_DIM,),dtype=jnp.float32),
            jnp.asarray(.25,dtype=jnp.float32))
        updated, _, diagnostics, _, _ = output
        if not all(bool(jax.device_get(jnp.all(jnp.isfinite(leaf))))
                   for leaf in jax.tree_util.tree_leaves(updated.params)):
            raise RuntimeError(f"{arm}: nonfinite new PPO actor")
        if not bool(jax.device_get(jnp.isfinite(diagnostics["rollout_reward_mean"]))):
            raise RuntimeError(f"{arm}: nonfinite reward")
        print(f"G6 {arm}: PASS ({args.envs} worlds, 360 inputs, 2 frames, PPO update)",flush=True)
    print("G6 CUDA/CPU arm smoke: PASS, no training checkpoints written.",flush=True)


if __name__ == "__main__":
    main()
