"""One optional real-MJX sanity run or 2048-world GPU PPO memory check.

No test matrix, no saved weights and no separate training framework.
"""
from __future__ import annotations

import argparse
import json
import os


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device",choices=("cpu","cuda","gpu"),default="cpu")
    p.add_argument("--worlds",type=int,default=4)
    p.add_argument("--frames",type=int,default=3)
    p.add_argument("--ppo",action="store_true")
    p.add_argument("--full-update",action="store_true",
                   help="one 2048x32 GPU PPO update; reports memory; saves nothing")
    args=p.parse_args()
    if args.worlds<4 or not 1<=args.frames<=20:
        p.error("--worlds must be >=4; --frames must be 1..20")
    if args.full_update and (args.worlds!=2048 or not args.ppo or
                             args.device not in ("cuda","gpu")):
        p.error("--full-update requires --worlds 2048 --device cuda --ppo")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false")
    from emu.ppo import configure_jax
    configure_jax("cuda" if args.device=="gpu" else args.device)
    import jax
    import jax.numpy as jnp
    import numpy as np
    from emu.contracts import ROBOT_XML,OBS_SCHEMA_TERRAIN
    from emu.environment import EmuMJXEnv

    if args.device in ("cuda","gpu") and jax.default_backend()!="gpu":
        raise RuntimeError("CUDA requested but JAX has no GPU backend")
    env=EmuMJXEnv(str(ROBOT_XML),num_envs=args.worlds,seed=5040,
                  max_episode_steps=240,terrain_step_max_m=.08,
                  observation_schema=OBS_SCHEMA_TERRAIN,
                  unified_capability_focus=True,
                  terrain_lesson="coherent-careful",
                  academy_course_lips=4,academy_height_cap_m=.020)
    params=dict(specialist_mix=1.,terrain_exposure=1.,
                terrain_height_m=.020,academy_height_m=.020,
                focus_weight_step=1.,academy_episode_probability=1.,
                athletic_rehearsal_frac=0.)
    packet=env.params_array(params)
    state,obs=env.reset(params)
    active=np.asarray(jax.device_get(state.academy_active),dtype=bool)
    if not active.any() or obs.shape!=(args.worlds,360):
        raise RuntimeError("Academy birth or 360-input sensor shape is wrong")
    if not np.isfinite(np.asarray(jax.device_get(obs))).all():
        raise RuntimeError("Non-finite initial 360-input observation")
    ids=env.academy_geom_ids
    if len(ids)!=5:
        raise RuntimeError("Four-lip collision geometry is incomplete")
    geom=np.asarray(jax.device_get(state.data.geom_xpos[:,ids,2]))
    tops=geom+np.asarray(env.mj_model.geom_size[list(ids),2])[None,:]
    rise=np.diff(tops,axis=1)[active]
    if not (np.all((rise>.001).sum(axis=1)==2) and
            np.all((rise<-.001).sum(axis=1)==2)):
        raise RuntimeError("Expected four physical lips, two up and two down")
    for _ in range(args.frames):
        result=env.step(state,jnp.zeros((args.worlds,6),jnp.float32),packet)
        state=result.state
        if not np.isfinite(np.asarray(jax.device_get(result.obs))).all():
            raise RuntimeError("Non-finite MJX step observation")
    print(f"Real MJX: {int(active.sum())}/{args.worlds} academy births, "
          f"four lips, 360 sensors, {args.frames} frames",flush=True)

    if args.ppo:
        from emu.ppo import make_trainer
        trainer=make_trainer(env,seed=5040,lr=2e-4,
                             rollout=32 if args.full_update else 2,
                             epochs=4 if args.full_update else 1,
                             minibatches=8 if args.full_update else 1)
        before=trainer["make_initial_state"](params)
        after,metrics,diag,_,_=trainer["train_update"](
            before,packet,jnp.zeros((360,),jnp.float32),
            jnp.ones((360,),jnp.float32),jnp.asarray(.25,jnp.float32))
        leaves=jax.tree_util.tree_leaves(after.params)
        if not all(bool(jax.device_get(jnp.all(jnp.isfinite(x)))) for x in leaves):
            raise RuntimeError("Non-finite PPO weights")
        if not bool(jax.device_get(jnp.isfinite(diag["rollout_reward_mean"]))):
            raise RuntimeError("Non-finite PPO reward")
        print("One PPO update completed; no training checkpoint written",flush=True)
        if args.full_update:
            memory=jax.devices("gpu")[0].memory_stats() or {}
            print("GPU memory:",{k:v for k,v in memory.items()
                  if "peak" in k or "limit" in k},flush=True)
    print(json.dumps(dict(status="PASS",worlds=args.worlds,
                          ppo=bool(args.ppo),production_shape=bool(args.full_update))))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
