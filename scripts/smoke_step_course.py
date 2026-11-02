"""Small real-MJX streaming terrain acceptance and optional GPU PPO memory check.

One ring of three actual collision plates. No checkpoint or new trainer.
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
                   help="one 2048x32 GPU PPO update, no saved weights")
    args=p.parse_args()
    if args.worlds<4 or not 1<=args.frames<=20:
        p.error("--worlds >= 4 and --frames between 1 and 20")
    if args.full_update and (args.worlds!=2048 or not args.ppo
                             or args.device not in ("cuda","gpu")):
        p.error("--full-update requires 2048 worlds, --ppo and CUDA")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false")
    from emu.ppo import configure_jax
    configure_jax("cuda" if args.device=="gpu" else args.device)
    import jax
    import jax.numpy as jnp
    import numpy as np
    from emu.contracts import ROBOT_XML,OBS_SCHEMA_TERRAIN
    from emu.environment import EmuMJXEnv
    from emu.step_academy_world import ACADEMY_PLATFORMS,ACADEMY_INTERVAL_M

    if args.device in ("cuda","gpu") and jax.default_backend() not in ("gpu","cuda"):
        raise RuntimeError("Requested CUDA backend is not active")
    env=EmuMJXEnv(str(ROBOT_XML),num_envs=args.worlds,seed=5042,
                  max_episode_steps=700,terrain_step_max_m=.08,
                  observation_schema=OBS_SCHEMA_TERRAIN,
                  unified_capability_focus=True,
                  terrain_lesson="coherent-careful",
                  academy_stream=True,academy_height_cap_m=.15)
    params=dict(specialist_mix=1.,terrain_exposure=1.,
                terrain_height_m=.02,academy_height_up_m=.02,
                academy_height_down_m=.03,
                focus_weight_step=1.,academy_episode_probability=1.,
                athletic_rehearsal_frac=0.)
    packed=env.params_array(params)
    state,obs=env.reset(params)
    active=np.asarray(jax.device_get(state.academy_active),dtype=bool)
    if not active.all() or obs.shape!=(args.worlds,360):
        raise RuntimeError("Streaming STEP births or observation shape invalid")
    if not np.isfinite(np.asarray(jax.device_get(obs))).all():
        raise RuntimeError("Nonfinite starting 360 sensor observation")
    ids=env.academy_geom_ids
    if len(ids)!=ACADEMY_PLATFORMS:
        raise RuntimeError("Streaming ring needs precisely three solid geoms")
    points=np.asarray(jax.device_get(state.data.geom_xpos[:,ids,:]))
    tops=points[:,:,2]+np.asarray(env.mj_model.geom_size[list(ids),2])[None,:]
    rise=np.diff(tops,axis=1)
    if not np.all(np.abs(rise)>.001):
        raise RuntimeError("Initial two step lips are not real discontinuities")

    # Prove the same THREE fixed slots can be physically recycled: the
    # trailing plate moves beyond the two look-ahead plates, not under feet.
    fake_cross=dict(cross_new=jnp.ones((args.worlds,),jnp.bool_))
    moved=env._recycle_academy_platform(state,state.data,fake_cross,
                                        state.rng,packed)
    before=np.asarray(jax.device_get(state.data.mocap_pos[
        :,env.academy_platform_mocap_ids[0],:]))
    after=np.asarray(jax.device_get(moved.mocap_pos[
        :,env.academy_platform_mocap_ids[0],:]))
    if not np.all(np.linalg.norm(after[:,:2]-before[:,:2],axis=1) >
                  2*ACADEMY_INTERVAL_M):
        raise RuntimeError("Old platform was not recycled beyond the third")
    # The real state has NOT been altered: only actual foot-supported
    # crossings can trigger this path in env.step.
    for _ in range(args.frames):
        out=env.step(state,jnp.zeros((args.worlds,6),jnp.float32),packed)
        state=out.state
        if not np.isfinite(np.asarray(jax.device_get(out.obs))).all():
            raise RuntimeError("Nonfinite MJX physics observation")
    print(f"Real three-platform MJX ring: {args.worlds} worlds, "
          f"two initial lips, unlimited recycled lips, 360 inputs",flush=True)

    if args.ppo:
        from emu.ppo import make_trainer
        trainer=make_trainer(env,seed=5042,lr=2e-4,
            rollout=32 if args.full_update else 2,
            epochs=4 if args.full_update else 1,
            minibatches=8 if args.full_update else 1)
        start=trainer["make_initial_state"](params)
        updated,_,diag,_,_=trainer["train_update"](
            start,packed,jnp.zeros((360,),jnp.float32),
            jnp.ones((360,),jnp.float32),jnp.asarray(.25,jnp.float32))
        if not all(bool(jax.device_get(jnp.all(jnp.isfinite(x))))
                   for x in jax.tree_util.tree_leaves(updated.params)):
            raise RuntimeError("Nonfinite PPO weights")
        if not bool(jax.device_get(jnp.isfinite(diag["rollout_reward_mean"]))):
            raise RuntimeError("Nonfinite PPO reward")
        print("One PPO update: finite, no checkpoint written",flush=True)
        if args.full_update:
            memory=jax.devices("gpu")[0].memory_stats() or {}
            print("GPU memory:",{k:v for k,v in memory.items()
                if "peak" in k or "limit" in k},flush=True)
    print(json.dumps(dict(status="PASS",worlds=args.worlds,
          streaming=True,ppo=bool(args.ppo),production_shape=bool(args.full_update))))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
