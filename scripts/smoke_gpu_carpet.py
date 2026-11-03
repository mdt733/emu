"""v5045 GPU trainer gate: one 18-prism MJX model, per-world
streaming state, real 4x4 ToF and a genuine PPO gradient update.

Does NOT claim learned locomotion; CPU smoke is not 2048 GPU acceptance.
"""
from __future__ import annotations
import argparse
import json
import os
import time


def physics_report_fields() -> dict:
    """Report the only active collision topology."""
    return dict(
        physics="eighteen-dynamic-convex-triangular-prisms",
        active_collision_geoms=18,
        terrain_carpet_prisms=True,
    )


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worlds",type=int,default=4)
    p.add_argument("--device",choices=("cpu","cuda","gpu"),default="cpu")
    p.add_argument("--frames",type=int,default=2)
    p.add_argument("--ppo",action="store_true")
    p.add_argument("--prisms",action="store_true",
                   help="required compatibility flag: current EMU always uses real prisms")
    p.add_argument("--full-update",action="store_true")
    p.add_argument("--benchmark-updates",type=int,default=1,
                   help="1..8 full PPO updates; report synchronized steady FPS "
                        "after the first compilation; no checkpoint")
    a=p.parse_args()
    if not a.prisms:
        p.error("The active EMU simulator is prism-only; pass --prisms")
    if a.worlds<4 or not 1<=a.frames<=20:p.error("Need >=4 worlds, 1..20 frames")
    if a.full_update and (a.worlds not in (1024,2048) or not a.ppo or a.device=="cpu"):
        p.error("--full-update requires --worlds 1024 or 2048, --device cuda --ppo")
    if (not 1<=a.benchmark_updates<=8 or
            (a.benchmark_updates>1 and not a.full_update)):
        p.error("--benchmark-updates must be 1..8 and >1 requires --full-update")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false")
    from emu.ppo import configure_jax
    configure_jax("cuda" if a.device=="gpu" else a.device)
    import jax
    import jax.numpy as jnp
    import numpy as np
    from emu.contracts import ROBOT_XML,OBS_SCHEMA_TERRAIN
    from emu.environment import EmuMJXEnv
    from emu.terrain_carpet_mjx import TILE_COUNT, validate_prism_only_geoms

    if a.device in ("cuda","gpu") and jax.default_backend()!="gpu":
        raise RuntimeError("Expected a CUDA-capable GPU")
    env=EmuMJXEnv(
        str(ROBOT_XML),num_envs=a.worlds,seed=735,max_episode_steps=700,
        terrain_step_max_m=.08,observation_schema=OBS_SCHEMA_TERRAIN,
        unified_capability_focus=True,terrain_lesson="coherent-careful",
        academy_stream=True,academy_height_cap_m=.15,terrain_carpet=True,
        terrain_carpet_prisms=True)
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
        raise RuntimeError("Expected exactly nine inactive source hfields")
    validate_prism_only_geoms(__import__('mujoco'),env.mj_model)
    if np.any(env.mj_model.geom_contype[np.asarray(
            env.carpet_geom_ids,dtype=np.int32)]!=0):
        raise RuntimeError("Source heightfield skirts must not collide")
    if len(env.carpet_prism_geom_ids)!=18:
        raise RuntimeError("Expected exactly 18 physical prism geoms")
    if np.any(env.mj_model.geom_contype[list(env.carpet_prism_geom_ids)]!=4):
        raise RuntimeError("Prism bit-4 contacts are not enabled")
    if env.mj_model.geom_contype[env.ids["floor_geom"]] != 0:
        raise RuntimeError("Static XML floor must not enter prism contact pairs")
    if np.any((env.mj_model.geom_conaffinity[
            np.asarray(env.ids["foot_geoms"],dtype=np.int32)] & 4)==0):
        raise RuntimeError("Robot feet must collide with carpet bit-4 surfaces")
    if state.carpet_seen_cells.shape!=(a.worlds,64,2):
        raise RuntimeError("Per-world visited-tile anti-rocking memory missing")
    if state.carpet_seen_write.shape!=(a.worlds,):
        raise RuntimeError("Per-world visit write pointer missing")
    heights=np.asarray(jax.device_get(state.carpet_hfields))
    if heights.shape!=(a.worlds,36) or not np.isfinite(heights).all():
        raise RuntimeError("Per-world hfield geometry must be finite [E,36]")
    if np.max(np.abs(heights-np.mean(heights,axis=1,keepdims=True)))<1e-4:
        raise RuntimeError("Terrain must contain genuine changing triangles")
    if not np.all(np.isfinite(np.asarray(jax.device_get(obs)))):
        raise RuntimeError("Nonfinite initial sensor observations")
    from emu.terrain_carpet_mjx import sample_surface
    roots=state.data.qpos[:,:3][:,None,:]
    source=sample_surface(jnp,state.data,env.carpet_mocap_ids,
                          state.carpet_hfields,roots)[:,0]
    actual=env._carpet_real_top_height(
        state.data,roots,state.carpet_hfields)[:,0]
    gap=np.asarray(jax.device_get(jnp.abs(actual-source)))
    if not np.all(np.isfinite(gap)) or np.max(gap)>.00005:
        raise RuntimeError(f"Collision surface diverges from analytic teacher by {gap}")
    print("surface PASS: two triangles agree with convex prism tops "
          f"within {float(np.max(gap))*1000:.3f}mm",flush=True)
    tile_positions=np.asarray(jax.device_get(state.data.mocap_pos[
        :,list(env.carpet_mocap_ids),:]))
    if tile_positions.shape!=(a.worlds,9,3):
        raise RuntimeError("Missing independently positioned tiles")
    physical_ids=env.carpet_prism_mocap_ids
    solid_pos=np.asarray(jax.device_get(state.data.mocap_pos[
        :,list(physical_ids),:]))
    if solid_pos.shape!=(a.worlds,18,3) or not np.isfinite(solid_pos).all():
        raise RuntimeError("Invalid per-world physical collision tile poses")
    print("reset PASS: 18 convex prisms, distinct per-world 360-input JAX physics",
          flush=True)
    # One robot crossing into a neighbouring 50cm tile must recycle only
    # abandoned slots. If a rear foot still occupies an abandoned tile,
    # the whole window must remain unchanged.
    old=state.carpet_center
    desired=old+jnp.asarray([1,0],dtype=jnp.int32)[None,:]
    root_xy=(desired.astype(jnp.float32)+.5)*.5
    fake_q=state.data.qpos.at[:,0:2].set(root_xy)
    foot_xpos=state.data.xpos.at[:,env.foot_bodies,0:2].set(
        jnp.broadcast_to(root_xy[:,None,:],(a.worlds,2,2)))
    safe_data=state.data.replace(qpos=fake_q,xpos=foot_xpos)
    relocated,new_h,new_c=env._carpet_recycle(state,safe_data)
    if not np.array_equal(np.asarray(jax.device_get(new_c)),
                          np.asarray(jax.device_get(desired))):
        raise RuntimeError("Safe 50cm passage did not advance the tile ring")
    before=np.asarray(jax.device_get(
        safe_data.mocap_pos[:,list(env.carpet_mocap_ids),:]))
    after=np.asarray(jax.device_get(
        relocated.mocap_pos[:,list(env.carpet_mocap_ids),:]))
    if not np.any(np.abs(after-before)>.3):
        raise RuntimeError("Abandoned physical tile colliders did not recycle")
    solid_after=np.asarray(jax.device_get(
        relocated.mocap_pos[:,list(physical_ids),:]))
    if not np.any(np.abs(solid_after-solid_pos)>.3):
        raise RuntimeError("Abandoned real prism collision tiles did not recycle")
    trailing=((old-jnp.asarray([1,0],dtype=jnp.int32)[None,:]
               ).astype(jnp.float32)+.5)*.5
    blocked_data=safe_data.replace(xpos=state.data.xpos.at[
        :,env.foot_bodies,0:2].set(
        jnp.broadcast_to(trailing[:,None,:],(a.worlds,2,2))))
    blocked,blocked_h,blocked_c=env._carpet_recycle(state,blocked_data)
    if not np.array_equal(np.asarray(jax.device_get(blocked_c)),
                          np.asarray(jax.device_get(old))):
        raise RuntimeError("Recycler moved a tile while a trailing foot stood on it")
    if not np.array_equal(np.asarray(jax.device_get(blocked_h)),heights):
        raise RuntimeError("Blocked recycling changed live collision triangles")
    old_solids=np.asarray(jax.device_get(
        blocked.mocap_pos[:,list(physical_ids),:]))
    if not np.array_equal(old_solids,solid_pos):
        raise RuntimeError("Blocked recycling moved a physical collision tile under a trailing foot")
    # A spherical foot may still touch an outgoing solid even when its
    # CENTRE has entered the new tile cell. The old integer-cell test
    # incorrectly recycled such a tile, potentially teleporting support.
    seam_x=(desired[:,0].astype(jnp.float32)-1.)*.5+.020
    edge_feet=jnp.broadcast_to(root_xy[:,None,:],(a.worlds,2,2))
    edge_feet=edge_feet.at[:,0,0].set(seam_x)
    edge_data=safe_data.replace(xpos=safe_data.xpos.at[
        :,env.foot_bodies,0:2].set(edge_feet))
    edge_shifted,edge_h,edge_c=env._carpet_recycle(state,edge_data)
    if not np.array_equal(np.asarray(jax.device_get(edge_c)),
                          np.asarray(jax.device_get(old))):
        raise RuntimeError("Recycler moved a physical prism sidewall under a foot sphere")
    if not np.array_equal(np.asarray(jax.device_get(edge_h)),heights):
        raise RuntimeError("Recycler altered physical triangle heights before a foot cleared a sidewall")
    print("recycler PASS: safe tile advance, trailing-foot hold, "
          "real sphere/solid-footprint clearance",flush=True)
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
        initial_leaves=jax.tree_util.tree_leaves(jax.device_get(initial.params))
        initial_actor=jax.tree_util.tree_leaves(jax.device_get(initial.params["pi"]))
        start_update=time.perf_counter()
        updated,_,diag,_,_=trainer["train_update"](
             initial,packet,jnp.zeros((360,),dtype=jnp.float32),
             jnp.ones((360,),dtype=jnp.float32),jnp.asarray(.25,jnp.float32))
        # Device transfer synchronizes the asynchronous XLA update. Include
        # compilation in this first-update time; it is NOT steady-state FPS.
        updated_leaves=jax.tree_util.tree_leaves(jax.device_get(updated.params))
        updated_actor=jax.tree_util.tree_leaves(jax.device_get(updated.params["pi"]))
        first_update_seconds=time.perf_counter()-start_update
        if len(initial_leaves)!=len(updated_leaves):
            raise RuntimeError("PPO parameter tree structure changed unexpectedly")
        if not all(np.isfinite(leaf).all() for leaf in updated_leaves):
            raise RuntimeError("Non-finite PPO actor parameters")
        max_param_delta=max(float(np.max(np.abs(after-before))) for before,after
                            in zip(initial_leaves,updated_leaves))
        if not np.isfinite(max_param_delta) or max_param_delta<=1e-10:
            raise RuntimeError("PPO update did not change any network parameters")
        max_actor_delta=max(float(np.max(np.abs(after-before))) for before,after
                            in zip(initial_actor,updated_actor))
        if not np.isfinite(max_actor_delta) or max_actor_delta<=1e-10:
            raise RuntimeError("PPO update changed the critic but not the actor")
        rollout_frames=32 if a.full_update else 2
        first_update_world_steps=a.worlds*rollout_frames
        print("real batched PPO training update PASS; "
              f"max_parameter_delta={max_param_delta:.8g}; "
              f"max_actor_delta={max_actor_delta:.8g}; "
              f"first_update_including_compile={first_update_seconds:.2f}s; "
              f"world_steps={first_update_world_steps:,}",flush=True)
        # The first update compiles XLA and CANNOT measure sustained
        # performance. Subsequent calls use the SAME 2048x32 rollout and
        # real four-epoch PPO, synchronized before reading each elapsed time.
        # This mode never writes a checkpoint or launches an overnight run.
        steady_update_times=[]
        if a.benchmark_updates>1:
            last=updated
            for i in range(1,a.benchmark_updates):
                start_steady=time.perf_counter()
                nxt,_,_,_,_=trainer["train_update"](
                    last,packet,jnp.zeros((360,),dtype=jnp.float32),
                    jnp.ones((360,),dtype=jnp.float32),
                    jnp.asarray(.25,jnp.float32))
                jax.block_until_ready((nxt.obs,nxt.params["pi"][-1]["w"]))
                steady_update_times.append(time.perf_counter()-start_steady)
                last=nxt
            # Keep output meaningful if a later PPO update blows up.
            if not bool(np.asarray(jax.device_get(
                    jnp.all(jnp.isfinite(last.params["pi"][-1]["w"]))))):
                raise RuntimeError("Nonfinite actor parameters during sustained PPO benchmark")
            steady_seconds=sum(steady_update_times)
            steady_fps=(a.benchmark_updates-1)*first_update_world_steps/steady_seconds
            print(f"steady state PPO PASS: {a.benchmark_updates-1} "
                  f"post-compile full updates; "
                  f"mean={steady_seconds/(a.benchmark_updates-1):.3f}s/update; "
                  f"throughput={steady_fps:,.0f} world-steps/s "
                  "(synchronized; no checkpoint)",flush=True)
        if a.full_update:
            dev=jax.devices("gpu")[0]
            stats=dev.memory_stats() or {}
            print("GPU memory:",{k:v for k,v in stats.items()
                if "peak" in k or "limit" in k},flush=True)
    print(json.dumps(dict(status="PASS",worlds=a.worlds,
          **physics_report_fields(),tiles=TILE_COUNT,
          finite_actor=True,ppo=a.ppo,full_update=a.full_update,
          **(dict(ppo_parameter_delta=max_param_delta,
                  ppo_actor_delta=max_actor_delta,
                  first_update_seconds=round(first_update_seconds,3),
                  first_update_world_steps=first_update_world_steps,
                  benchmark_updates=a.benchmark_updates,
                  **(dict(steady_update_seconds_mean=round(
                             sum(steady_update_times)/len(steady_update_times),3),
                          steady_world_steps_per_second=round(
                             (a.benchmark_updates-1)*first_update_world_steps/
                             sum(steady_update_times),1))
                     if steady_update_times else {}))
             if a.ppo else {}))))


if __name__=="__main__":
    main()
