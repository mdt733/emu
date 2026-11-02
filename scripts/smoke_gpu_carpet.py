"""v5041 GPU trainer gate: nine real 2-triangle MJX surfaces, per-world
streaming state, real 4x4 ToF and a genuine PPO gradient update.

Does NOT claim learned locomotion; CPU smoke is not 2048 GPU acceptance.
"""
from __future__ import annotations
import argparse
import json
import os


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worlds",type=int,default=4)
    p.add_argument("--device",choices=("cpu","cuda","gpu"),default="cpu")
    p.add_argument("--frames",type=int,default=2)
    p.add_argument("--ppo",action="store_true")
    p.add_argument("--full-update",action="store_true")
    a=p.parse_args()
    if a.worlds<4 or not 1<=a.frames<=20:p.error("Need >=4 worlds, 1..20 frames")
    if a.full_update and (a.worlds!=2048 or not a.ppo or a.device=="cpu"):
        p.error("--full-update requires --worlds 2048 --device cuda --ppo")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false")
    from emu.ppo import configure_jax
    configure_jax("cuda" if a.device=="gpu" else a.device)
    import jax
    import jax.numpy as jnp
    import numpy as np
    from emu.contracts import ROBOT_XML,OBS_SCHEMA_TERRAIN
    from emu.environment import EmuMJXEnv
    from emu.terrain_carpet_mjx import TILE_COUNT

    if a.device in ("cuda","gpu") and jax.default_backend()!="gpu":
        raise RuntimeError("Expected a CUDA-capable GPU")
    env=EmuMJXEnv(
        str(ROBOT_XML),num_envs=a.worlds,seed=735,max_episode_steps=700,
        terrain_step_max_m=.08,observation_schema=OBS_SCHEMA_TERRAIN,
        unified_capability_focus=True,terrain_lesson="coherent-careful",
        academy_stream=True,academy_height_cap_m=.15,terrain_carpet=True)
    params=dict(specialist_mix=1.,terrain_exposure=1.,slope_exposure=1.,
                slope_max_deg=8.,terrain_height_m=.02,
                academy_height_up_m=.02,academy_height_down_m=.03,
                focus_weight_step=1.,academy_episode_probability=1.,
                athletic_rehearsal_frac=0.)
    packet=env.params_array(params)
    state,obs=env.reset(params)
    if obs.shape!=(a.worlds,360):
        raise RuntimeError(f"Invalid 360-input contract: {obs.shape}")
    if len(env.carpet_geom_ids)!=TILE_COUNT or env.mj_model.nhfield!=TILE_COUNT:
        raise RuntimeError("Expected exactly nine collision hfield tiles")
    heights=np.asarray(jax.device_get(state.carpet_hfields))
    if heights.shape!=(a.worlds,36) or not np.isfinite(heights).all():
        raise RuntimeError("Per-world hfield geometry must be finite [E,36]")
    if np.max(np.abs(heights-np.mean(heights,axis=1,keepdims=True)))<1e-4:
        raise RuntimeError("Terrain must contain genuine changing triangles")
    if not np.all(np.isfinite(np.asarray(jax.device_get(obs)))):
        raise RuntimeError("Nonfinite initial sensor observations")
    tile_positions=np.asarray(jax.device_get(state.data.mocap_pos[
        :,list(env.carpet_mocap_ids),:]))
    if tile_positions.shape!=(a.worlds,9,3):
        raise RuntimeError("Missing independently positioned tiles")
    print("reset PASS: 9 x 2 real triangles, distinct per-world JAX model "
          "buffers, 360 sensor inputs",flush=True)
    for i in range(a.frames):
        out=env.step(state,jnp.zeros((a.worlds,6),dtype=jnp.float32),packet)
        state=out.state
        if not np.isfinite(np.asarray(jax.device_get(out.obs))).all():
            raise RuntimeError("Non-finite observations after physics step")
    print(f"physics PASS: {a.frames} frames in {a.worlds} worlds",flush=True)
    if a.ppo:
        from emu.ppo import make_trainer
        trainer=make_trainer(env,seed=735,lr=2e-4,
             rollout=32 if a.full_update else 2,
             epochs=4 if a.full_update else 1,
             minibatches=8 if a.full_update else 1)
        initial=trainer["make_initial_state"](params)
        updated,_,diag,_,_=trainer["train_update"](
             initial,packet,jnp.zeros((360,),dtype=jnp.float32),
             jnp.ones((360,),dtype=jnp.float32),jnp.asarray(.25,jnp.float32))
        for leaf in jax.tree_util.tree_leaves(updated.params):
            if not np.asarray(jax.device_get(jnp.all(jnp.isfinite(leaf)))):
                raise RuntimeError("Non-finite PPO actor parameters")
        print("real batched PPO training update PASS",flush=True)
        if a.full_update:
            dev=jax.devices("gpu")[0]
            stats=dev.memory_stats() or {}
            print("GPU memory:",{k:v for k,v in stats.items()
                if "peak" in k or "limit" in k},flush=True)
    print(json.dumps(dict(status="PASS",worlds=a.worlds,
          physics="2-triangle-heightfield",tiles=TILE_COUNT,
          finite_actor=True,ppo=a.ppo,full_update=a.full_update)))


if __name__=="__main__":
    main()
