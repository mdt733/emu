#!/usr/bin/env python3
"""Physical mixed-terrain + one-policy six-specialist MJX PPO smoke.

Checks both rotated contact boxes, step top displacement, actual four-by-four
actor ranges, continuous ground angle and one real PPO gradient. No file writes.
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
    # Verify the true cold start retains main's steady-only curriculum:
    # role IDs exist but specialty lessons and motor burden are inactive.
    cold, cold_obs = env.reset(params)
    cold_obs.block_until_ready()
    assert not bool(jax.device_get(jnp.any(cold.specialist_active)))
    # Exercise ALL six live branches without simulating millions of frames.
    # This synthetic difficulty is test input, not a training curriculum jump.
    params['steady_difficulty']=.35
    # Verify genuinely random pitch/roll and steps in one JAX model, not
    # just simulated tilted gravity or inactive world labels.
    params.update(slope_exposure=1.0,slope_max_deg=5.0,
                  terrain_exposure=1.0,terrain_height_m=.035)
    state,obs=env.reset(params)
    obs.block_until_ready()
    assert bool(jax.device_get(jnp.all(state.specialist_active)))
    import numpy as np
    floor_id=int(env.ids['floor_geom'])
    upper_id=int(env.terrain_step_geom_id)
    n=np.asarray(jax.device_get(state.data.geom_xmat[:,floor_id,:])).reshape(
        (args.envs,3,3))[:,:,2]
    tilt=np.rad2deg(np.arccos(np.clip(n[:,2],-1.,1.)))
    ups=np.asarray(jax.device_get(state.data.mocap_pos[:,env.terrain_step_mocap_id,:]))
    base=np.asarray(jax.device_get(state.data.mocap_pos[:,env.unified_ground_mocap_id,:]))
    active=ups[:,2]>-1.
    assert np.any(tilt>1.0),"NO real inclined collision world"
    assert np.any(active),"NO real physical steps"
    from emu.environment import UNIFIED_SLAB_HALF_LENGTH_M, UNIFIED_SLAB_HALF_DEPTH_M
    R=np.asarray(jax.device_get(state.data.geom_xmat[:,upper_id,:])).reshape(
        (args.envs,3,3))
    axis=R[:,:,0]
    side=np.where(np.arange(args.envs)%2==1,-1.,1.)
    lower_lip=base+side[:,None]*UNIFIED_SLAB_HALF_LENGTH_M*axis+UNIFIED_SLAB_HALF_DEPTH_M*n
    upper_lip=ups-side[:,None]*UNIFIED_SLAB_HALF_LENGTH_M*axis+UNIFIED_SLAB_HALF_DEPTH_M*n
    rise=np.sum((upper_lip-lower_lip)*n,axis=1)
    assert np.all((rise[active]>.005)&(rise[active]<=.036)),rise[active]
    assert np.allclose(np.sum((upper_lip-lower_lip)*axis,axis=1)[active],
                       0,atol=1e-4),"sharp lip does not align in common surface basis"
    assert np.allclose(
        np.asarray(jax.device_get(state.data.geom_xmat[:,floor_id,:])),
        np.asarray(jax.device_get(state.data.geom_xmat[:,upper_id,:])),
        atol=1e-4),"upper/lower tangent planes disagree"
    rays=np.asarray(jax.device_get(
        state.sensor_state.samples.terrain_ranges_m[:,-1,:]))
    assert np.all(np.isfinite(rays)) and np.any(rays>0),"terrain actor blind"
    print(f"Real collision slabs: {int((tilt>1).sum())} slopes, "
          f"{int(active.sum())} real steps, "
          f"{int(((tilt>1)&active).sum())} mixed, "
          f"normal rise {float(rise[active].min()):.4f}.."
          f"{float(rise[active].max()):.4f}m",flush=True)
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
    print("Geometric MJX slope+step smoke passed; full curriculum and GPU "
          "acceptance are separate launch gates.")

if __name__=="__main__":
    main()
