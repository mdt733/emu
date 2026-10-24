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
    randomized,_=env.reset(params)
    import numpy as np
    random_n=np.asarray(jax.device_get(
        randomized.data.geom_xmat[:,int(env.ids['floor_geom']),:])).reshape(
            (args.envs,3,3))[:,:,2]
    random_angles=np.rad2deg(np.arccos(np.clip(random_n[:,2],-1,1)))
    assert np.any(random_angles>1.),"randomized physical incline absent"
    # Exact replayable 3deg pitch, -2deg roll, 2cm sharp rise, +17deg
    # edge yaw. Every world has both a slope AND a step for contact probing.
    params.update(terrain_pitch_deg=3.0,terrain_roll_deg=-2.0,
                  terrain_fixed_height_m=.020,terrain_edge_angle_deg=17.0)
    state,obs=env.reset(params)
    obs.block_until_ready()
    assert bool(jax.device_get(jnp.all(state.specialist_active)))
    floor_id=int(env.ids['floor_geom'])
    upper_id=int(env.terrain_step_geom_id)
    n=np.asarray(jax.device_get(state.data.geom_xmat[:,floor_id,:])).reshape(
        (args.envs,3,3))[:,:,2]
    tilt=np.rad2deg(np.arccos(np.clip(n[:,2],-1.,1.)))
    ups=np.asarray(jax.device_get(state.data.mocap_pos[:,env.terrain_step_mocap_id,:]))
    base=np.asarray(jax.device_get(state.data.mocap_pos[:,env.unified_ground_mocap_id,:]))
    active=ups[:,2]>-1.
    assert np.all(tilt>3.),"fixed pitch/roll did not rotate real collision"
    assert np.all(active),"fixed physical step override did not activate"
    expected=np.asarray([-np.tan(np.deg2rad(3.)),
                         -np.tan(np.deg2rad(-2.)),1.])
    expected=expected/np.linalg.norm(expected)
    assert np.allclose(n,expected[None,:],atol=2e-4),"incorrect 2D pitch/roll normal"
    from emu.environment import UNIFIED_SLAB_HALF_LENGTH_M, UNIFIED_SLAB_HALF_DEPTH_M
    R=np.asarray(jax.device_get(state.data.geom_xmat[:,upper_id,:])).reshape(
        (args.envs,3,3))
    axis=R[:,:,0]
    side=np.where(np.arange(args.envs)%2==1,-1.,1.)
    lower_lip=base+side[:,None]*UNIFIED_SLAB_HALF_LENGTH_M*axis+UNIFIED_SLAB_HALF_DEPTH_M*n
    upper_lip=ups-side[:,None]*UNIFIED_SLAB_HALF_LENGTH_M*axis+UNIFIED_SLAB_HALF_DEPTH_M*n
    rise=np.sum((upper_lip-lower_lip)*n,axis=1)
    assert np.allclose(rise[active],.020,atol=2e-4),rise[active]
    assert np.allclose(np.sum((upper_lip-lower_lip)*axis,axis=1)[active],
                       0,atol=1e-4),"sharp lip does not align in common surface basis"
    assert np.allclose(
        np.asarray(jax.device_get(state.data.geom_xmat[:,floor_id,:])),
        np.asarray(jax.device_get(state.data.geom_xmat[:,upper_id,:])),
        atol=1e-4),"upper/lower tangent planes disagree"
    # Physically pin one ball 4mm INTO the top slab with the robot on
    # the target side; actual MJX contact pairs must report the upper geom.
    # This proves collision, rather than only trusting our ray mathematics.
    up_world=(np.arange(args.envs)%2==0)&active
    if not np.any(up_world):
        raise AssertionError("no upward physical platform worlds")
    translated=state.data.replace(
        qpos=state.data.qpos.at[:,0].add(.50))
    translated=env.forward_batched(env.mjx_model,translated)
    from emu.directional import contact_pair_mask
    feet=np.asarray(env.ids['foot_geoms'])
    xyz=np.asarray(jax.device_get(translated.geom_xpos[:,feet,:]))
    top=ups[:,None,:]+UNIFIED_SLAB_HALF_DEPTH_M*n[:,None,:]
    above=np.sum((xyz-top)*n[:,None,:],axis=2)
    # Raise/lower the free torso so the lowest ball intersects the target
    # surface. This is teacher-only contact acceptance, not a policy action.
    dz=(np.min(above,axis=1)-(.018-.004))/np.maximum(n[:,2],.1)
    pressed=translated.replace(
        qpos=translated.qpos.at[:,2].add(-jnp.asarray(dz)))
    pressed=env.forward_batched(env.mjx_model,pressed)
    contacts=getattr(pressed,'_impl',pressed).contact
    pair=contact_pair_mask(jnp,contacts.geom1,contacts.geom2,
                           contacts.dist,upper_id,tuple(int(v) for v in feet))
    got=np.asarray(jax.device_get(jnp.any(pair,axis=1)))
    assert np.any(got & up_world),"real foot/upper-box collision never appeared"
    rays=np.asarray(jax.device_get(
        state.sensor_state.samples.terrain_ranges_m[:,-1,:]))
    assert np.all(np.isfinite(rays)) and np.any(rays>0),"terrain actor blind"
    print(f"Real collision slabs: {int((tilt>1).sum())} slopes, "
          f"{int(active.sum())} real steps, "
          f"{int(((tilt>1)&active).sum())} mixed, "
          f"normal rise {float(rise[active].min()):.4f}.."
          f"{float(rise[active].max()):.4f}m",flush=True)
    # No simulated 3-ToF hardware exists in the new actor. The archived
    # 232-input and earlier 360-input sensor identities remain untouched.
    from emu.contracts import TERRAIN_ARRAY_ONLY_REVISION
    assert env.sensor_contract['sensor_model_revision']==TERRAIN_ARRAY_ONLY_REVISION
    assert 'tripod_mounts_body_m' not in env.sensor_contract
    assert bool(jax.device_get(jnp.all(state.sensor_state.samples.tof_ranges_m==0)))
    from emu.directional import fit_forward_array_ground
    delayed=state.sensor_state.samples
    estimate=fit_forward_array_ground(jnp,delayed.terrain_ranges_m[:,-1,:],
          delayed.terrain_valid[:,-1,:],
          env._body_packets(state.data)[0])
    assert bool(jax.device_get(jnp.all(jnp.isfinite(
        estimate['ground_visibility_confidence']))))
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
