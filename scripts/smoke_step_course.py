"""Physics acceptance for genuine 4-lip Academy and unchanged 360 actor.

Run from a clean Academy feature checkout:
  python scripts/smoke_step_course.py --worlds 4 --frames 3 --device cpu
This is a bounded MJX compile/step test, NEVER PPO training.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worlds",type=int,default=4)
    parser.add_argument("--frames",type=int,default=3)
    parser.add_argument("--device",choices=("cpu","gpu","cuda"),default="cpu")
    parser.add_argument("--ppo",action="store_true",
                        help="also run one real 360-input PPO gradient, never save weights")
    args=parser.parse_args()
    if args.worlds<4 or args.worlds%4 or args.frames<1 or args.frames>20:
        parser.error("worlds must be divisible by 4 and >=4; frames 1..20")
    from emu.ppo import configure_jax
    configure_jax("cuda" if args.device=="gpu" else args.device)
    import jax
    import jax.numpy as jnp
    import numpy as np
    import mujoco
    from emu.contracts import ROBOT_XML,OBS_SCHEMA_TERRAIN
    from emu.environment import EmuMJXEnv
    from emu.step_academy_world import (ACADEMY_INTERVAL_M,ACADEMY_EDGE_X_M,
                                        expected_lip_directions)
    print("JAX",jax.__version__,"backend",jax.default_backend(),"devices",jax.devices(),flush=True)
    if args.device in ("gpu","cuda") and jax.default_backend() not in ("gpu","cuda"):
        raise RuntimeError("GPU requested but JAX has no CUDA backend")
    env=EmuMJXEnv(str(ROBOT_XML),num_envs=args.worlds,seed=5040,
                   max_episode_steps=240,
                   terrain_step_max_m=.08,
                   observation_schema=OBS_SCHEMA_TERRAIN,
                   unified_capability_focus=True,
                   terrain_lesson="coherent-careful",
                   academy_course_lips=4,
                   academy_height_cap_m=.020)
    params={
        "specialist_mix":1.0,
        "terrain_exposure":1.0,
        "terrain_height_m":.020,
        "focus_weight_step":1.0,
        "academy_episode_probability":1.0,
        "terrain_edge_angle_deg":3.0,
        "athletic_rehearsal_frac":0.0,
    }
    packet=env.params_array(params)
    state,obs=env.reset(params)
    active=np.asarray(jax.device_get(state.academy_active),dtype=bool)
    print("Academy worlds active",active.sum(),"/",args.worlds,flush=True)
    if not active.any():
        raise RuntimeError("Step Academy allocation was requested but no course was sampled")
    if obs.shape!=(args.worlds,360) or not np.isfinite(np.asarray(obs)).all():
        raise RuntimeError("360-input sensor state is invalid")
    m=env.mj_model
    ids=env.academy_geom_ids
    assert len(ids)==5 and len(env.academy_mid_mocap_ids)==3
    robot_foot=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,"r_foot_ball")
    for geom_id in env.academy_mid_geom_ids:
        assert int(m.geom_contype[geom_id])==4
        assert int(m.geom_conaffinity[geom_id])==0
        assert int(m.geom_conaffinity[robot_foot])&4
    geom=np.asarray(jax.device_get(state.data.geom_xpos))[:,list(ids),:]
    R=np.asarray(jax.device_get(state.data.geom_xmat))[:,int(ids[0]),:].reshape(args.worlds,3,3)
    for w in np.flatnonzero(active):
        axis=R[w,:,0]
        xalong=geom[w]@axis
        lengths=m.geom_size[np.asarray(ids),0]
        # First floor terminates exactly where the first middle box begins,
        # and the final large upper slab begins at the fourth lip.
        for j in range(4):
            left=xalong[j]+lengths[j]
            right=xalong[j+1]-lengths[j+1]
            if abs(left-right)>1e-3:
                raise RuntimeError(f"physical gap/overlap world {w} lip {j}: {left-right}")
        assert np.allclose(xalong[:],(
            ACADEMY_EDGE_X_M-32., ACADEMY_EDGE_X_M+.5*ACADEMY_INTERVAL_M,
            ACADEMY_EDGE_X_M+1.5*ACADEMY_INTERVAL_M,
            ACADEMY_EDGE_X_M+2.5*ACADEMY_INTERVAL_M,
            ACADEMY_EDGE_X_M+3*ACADEMY_INTERVAL_M+32.),atol=.001)
        top=geom[w,:,2]+m.geom_size[np.asarray(ids),2]
        rise=np.diff(top)
        if any(abs(x)<.001 for x in rise):
            raise RuntimeError(f"missing actual height transition world={w} {rise}")
        expected=((1,1,-1,-1),(1,-1,1,-1),(-1,1,-1,1),(-1,-1,1,1))[w%4]
        if tuple(np.sign(rise).astype(int)) != expected:
            raise RuntimeError(f"wrong signed lips world={w}: {rise} vs {expected}")
    # Verify a real vertical slant ray intersects a middle contact box,
    # rather than a separate synthetic height map.
    origin=jnp.asarray(geom[:,1,:])+jnp.asarray((0.,0.,.55))[None,:]
    ray=jnp.tile(jnp.asarray((0.,0.,-1.))[None,None,:],(args.worlds,1,1))
    mid_ranges=np.asarray(jax.device_get(env._ray_box_distance(
        state.data,int(ids[1]),origin,ray)))[:,0]
    if not np.all(np.isfinite(mid_ranges[active])):
        raise RuntimeError("Academy first platform is not visible to physical ray-OBB test")
    # The test also steps real MJX and verifies the actor packet stays finite.
    for i in range(args.frames):
        result=env.step(state,jnp.zeros((args.worlds,6),dtype=jnp.float32),packet)
        state=result.state
        if not np.isfinite(np.asarray(jax.device_get(result.obs))).all():
            raise RuntimeError("nonfinite observation during Academy real physics")
        print("MJX frame",i+1,"reward",float(jnp.mean(result.reward)),flush=True)
    if args.ppo:
        from emu.ppo import make_trainer
        trainer=make_trainer(env,seed=5040,lr=2e-4,rollout=2,
                             epochs=1,minibatches=1)
        before=trainer["make_initial_state"](params)
        after,metrics,diag,_,_=trainer["train_update"](
            before,packet,jnp.zeros((360,),jnp.float32),
            jnp.ones((360,),jnp.float32),jnp.asarray(.25,jnp.float32))
        if not all(bool(jax.device_get(jnp.all(jnp.isfinite(p))))
                   for p in jax.tree_util.tree_leaves(after.params)):
            raise RuntimeError("Academy PPO produced nonfinite weights")
        assert bool(jax.device_get(jnp.isfinite(diag["rollout_reward_mean"])))
        from emu.results import summarize_rollout
        summary=summarize_rollout(metrics,frontier_params=params)
        print("Academy PPO PASS: rollout events",
              {k:summary[k] for k in (
                  "academy_active_fraction","academy_crossings",
                  "academy_up_crossings","academy_down_crossings")},
              flush=True)
    print(json.dumps(dict(status="PASS",academy_worlds=int(active.sum()),
                          lips=4,obs_dim=360,physics_frames=args.frames,
                          ppo_gradient=bool(args.ppo),
                          source="real MJX collision and actual ray-OBB surfaces")),flush=True)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
