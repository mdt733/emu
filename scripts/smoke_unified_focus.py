#!/usr/bin/env python3
"""Small real MJX + PPO smoke for the ONE shared-policy six-specialist director.

This is not acceptance of real slopes and does not write any policy files.
The unified overnight trainer remains intentionally blocked until true
incline+finite-step contact physics has been demonstrated separately.
"""
from __future__ import annotations
import argparse
import os

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    ap.add_argument("--envs",type=int,default=12)
    args=ap.parse_args()
    if args.envs<12 or args.envs>32:
        raise SystemExit("use 12–32 worlds so all six specialists are represented")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false")
    from emu.ppo import configure_jax, require_backend, make_trainer, build_obs_norm_mask
    configure_jax(args.device)
    import jax
    import jax.numpy as jnp
    from emu.contracts import OBS_SCHEMA_TERRAIN,TERRAIN_OBS_DIM,ROBOT_XML
    from emu.environment import EmuMJXEnv
    from emu.curriculum import RegimeGovernor
    from emu.capability_focus import FOCUS_NAMES,focus_indices
    ids=focus_indices(args.envs)
    assert len(set(int(x) for x in ids))==6
    env=EmuMJXEnv(str(ROBOT_XML),num_envs=args.envs,seed=733,
        max_episode_steps=40,observation_schema=OBS_SCHEMA_TERRAIN,
        attitude_profile="v5015",terrain_step_max_m=.04,
        unified_capability_focus=True,phase_speed_gain=1.25,
        open_frontier=True,combo_mix_frac=.88)
    governor=RegimeGovernor(open_frontier=True,unified_capability_focus=True)
    params=governor.params_dict()
    params.update(terrain_exposure=.15,terrain_height_m=.004)
    state,obs=env.reset(params)
    obs.block_until_ready()
    assert obs.shape==(args.envs,TERRAIN_OBS_DIM),obs.shape
    assert state.hist.shape==(args.envs,4,90),state.hist.shape
    assert build_obs_norm_mask(OBS_SCHEMA_TERRAIN).shape==(TERRAIN_OBS_DIM,)
    assert governor.params.frontier_vx_hi<.6
    stepped=env.step(state,jnp.zeros((args.envs,6),jnp.float32),params)
    stepped.obs.block_until_ready()
    for key in ("specialist_focus","episode_command_quality",
                "command_probe_radius","command_probe_weight_x",
                "terrain_active_up","servo_shaft_drive_power_w"):
        arr=stepped.metrics[key]
        assert arr.shape==(args.envs,),(key,arr.shape)
        assert bool(jax.device_get(jnp.all(jnp.isfinite(arr)))),key
    trainer=make_trainer(env,seed=733,lr=2e-4,rollout=2,
                         epochs=1,minibatches=1)
    first=trainer["make_initial_state"](params)
    updated,rolled,diag,_,_=trainer["train_update"](
        first,env.params_array(params),
        jnp.zeros((TERRAIN_OBS_DIM,),jnp.float32),
        jnp.ones((TERRAIN_OBS_DIM,),jnp.float32),
        jnp.asarray(.25,jnp.float32))
    assert all(bool(jax.device_get(jnp.all(jnp.isfinite(p))))
               for p in jax.tree_util.tree_leaves(updated.params))
    assert bool(jax.device_get(jnp.isfinite(diag["rollout_reward_mean"])))
    from emu.results import summarize_rollout
    row=summarize_rollout(rolled,frontier_params=params)
    for k in ("focus_worlds_command","focus_worlds_control",
              "command_probe_attempts_vx_hi","control_shaft_j_per_m"):
        assert k in row,k
    # Evidence serialises with the same governor as the neural policy.
    events=governor.update(row)
    archived=governor.state_dict()
    recovered=RegimeGovernor(open_frontier=True,unified_capability_focus=True)
    recovered.load_state_dict(archived)
    assert recovered.state_dict()==archived
    print(f"PASS: {args.envs} real MJX worlds; six specialist roles; "
          f"360 actor channels; one PPO gradient; persistent evidence",flush=True)
    print("World allocation: "+", ".join(
        f"{FOCUS_NAMES[i]}={int((ids==i).sum())}" for i in range(6)),flush=True)
    print("NOTE: real combined slope+step physics NOT accepted; GPU run blocked.")

if __name__=="__main__":
    main()
