#!/usr/bin/env python3
"""Read-only frozen 312-input policy: flat -> 0mm platform -> tiny step probes.

NO PPO updates, no checkpoints saved, no GPU training. The parent pickle is
validated under its original observation contract; physical worlds run on CPU.
This is an acceptance test before committing 2048 worlds to a continuation.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def check(parent: Path, *, worlds: int = 4, horizon: int = 160) -> dict:
    import jax
    import jax.numpy as jnp
    import numpy as np
    from emu.checkpoint_io import load_checkpoint
    from emu.contracts import ROBOT_XML, OBS_SCHEMA_PREVIEW
    from emu.curriculum import RegimeGovernor
    from emu.environment import EmuMJXEnv
    from emu.ppo import make_trainer
    from emu.terrain import adaptive_height_course
    from emu.training import env_kwargs_from_args
    from emu.training_config import TrainingConfig, training_sensor_contract

    if jax.default_backend() != "cpu":
        raise RuntimeError("Frozen preflight must run on CPU, not consume the training GPU")
    parent = parent.expanduser().resolve()
    cfg = TrainingConfig.load(parent.parent.parent / "training_config.json")
    if cfg.observation_schema != OBS_SCHEMA_PREVIEW or cfg.terrain_height_safety_m != 0.:
        raise ValueError("Parent must be a flat-ground 312-input foundation")
    if cfg.envs != 2048:
        raise ValueError("Parent was not trained with the required 2048 environments")
    payload = load_checkpoint(
        parent, validate=True, expected_observation_schema=OBS_SCHEMA_PREVIEW,
        expected_sensor_contract=training_sensor_contract(cfg))
    governor = RegimeGovernor(terrain_height_safety_m=0.)
    governor.load_state_dict(payload["curriculum_state"])
    # Foundation may contain a dormant 3mm field; zero it just as the
    # trainer now does at the external parent handoff.
    governor.params.frontier_terrain_height_m = 0.
    commands = governor.params_dict()
    commands.update(probe_cmd_vx=.10, probe_cmd_vy=0., probe_cmd_yaw=0.,
                    # The frozen study probes a representative forward
                    # request. Both worlds see precisely the SAME packet.
                    )
    policy = jax.device_put(payload["state"]["params"])
    mean = jnp.asarray(payload["obs_norm"]["mean"])
    std = jnp.sqrt(jnp.asarray(payload["obs_norm"]["var"]))
    shared = env_kwargs_from_args(cfg)
    n = int(worlds)
    if n < 2 or n > 16 or horizon < 80 or horizon > 400:
        raise ValueError("Frozen sanity study: 2..16 worlds, 80..400 frames")

    def score(env):
        trainer = make_trainer(env, seed=733, lr=cfg.ppo_lr,
                               rollout=32, epochs=1, minibatches=1,
                               entropy_cost=cfg.ppo_entropy_cost,
                               value_cost=cfg.ppo_value_cost)
        packet = env.params_array(commands)
        output = trainer["evaluate_summary"](
            policy, packet, mean, std, jax.random.PRNGKey(1784), horizon)
        result = np.asarray(jax.device_get(output),dtype=np.float64)
        if result.shape != (14,) or not np.all(np.isfinite(result)):
            raise RuntimeError(f"invalid frozen examiner output: {result!r}")
        return result

    # Physical 0mm plateau retains collision topology and 312-input sensors,
    # but its optional positive-height challenge sampler can be disabled
    # solely for the parity examination. No training run uses this override.
    with tempfile.TemporaryDirectory(prefix="emu5020_frozen_bridge_") as tmp:
        course = adaptive_height_course(
            ROBOT_XML, Path(tmp) / "course.xml",
            immutable=True, elevation_platform=True)
        old = EmuMJXEnv(str(ROBOT_XML),num_envs=n,seed=733,
                         max_episode_steps=700, **shared)
        opts = {**shared, "terrain_contact_mode":"box_surface",
                "terrain_adaptive":True,
                "terrain_random_placement":True,
                "terrain_elevation_mode":"up",
                "terrain_height_exposure":.40,
                "terrain_height_edge_fraction":.30,
                "terrain_exploration_safety_m":.006}
        plateau = EmuMJXEnv(str(course),num_envs=n,seed=733,
                            max_episode_steps=700, **opts)
        # One SAME policy/norm/task/simulator seed across three conditions.
        reference = score(old)
        plateau.terrain_exploration_probe_m = 0.  # genuine 0mm baseline
        zero = score(plateau)
        plateau.terrain_exploration_probe_m = .002  # actual <=0.25mm probe
        micro = score(plateau)

    names = ("survival", "first_fall_step", "vx_mae", "vy_mae",
             "yaw_mae", "gravity_z", "cone", "gait", "support",
             "pitch_mae", "height_shortfall_m", "walking_pass",
             "forward_progress_m", "commanded_forward_m")
    result = {
        "source_checkpoint":str(parent),
        "source_step":int(payload["meta"]["step"]),
        "worlds":n,"horizon":horizon,
        "flat":dict(zip(names,map(float,reference))),
        "plateau_0mm":dict(zip(names,map(float,zero))),
        "microprobe_0_to_0_25mm":dict(zip(names,map(float,micro))),
    }
    print(json.dumps(result,indent=2,sort_keys=True),flush=True)
    # Guard the original checkpoint. If it already fails elementary
    # locomotion here, do not blame the new platform for that failure.
    if reference[0] < .50 or reference[11] < .25:
        raise RuntimeError(
            "FAIL: source foundation did not demonstrate enough frozen "
            "forward walking on the original flat world; no training permitted")
    for label, candidate in (("zero plateau",zero),("microprobe",micro)):
        if candidate[0] < max(.50,reference[0]-.25):
            raise RuntimeError(
                f"FAIL: {label} survival {candidate[0]:.2f} "
                f"vs original {reference[0]:.2f}; check physical/contact/sensor parity")
        if candidate[11] < max(.20,reference[11]-.25):
            raise RuntimeError(
                f"FAIL: {label} commanded walking {candidate[11]:.2f} "
                f"vs original {reference[11]:.2f}")
        if candidate[1] < max(40., reference[1]*.50):
            raise RuntimeError(
                f"FAIL: {label} first-fall time {candidate[1]:.1f} "
                f"vs original {reference[1]:.1f}")
        if candidate[2] > reference[2] + .16:
            raise RuntimeError(
                f"FAIL: {label} vx tracking error worsened: "
                f"{candidate[2]:.3f} vs {reference[2]:.3f}")
    print("PASS: frozen policy retained basic walking on 0mm and microprobe course.",flush=True)
    return result


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent",type=Path,required=True)
    parser.add_argument("--worlds",type=int,default=4)
    parser.add_argument("--horizon",type=int,default=160)
    args=parser.parse_args()
    check(args.parent,worlds=args.worlds,horizon=args.horizon)


if __name__=="__main__":
    main()
