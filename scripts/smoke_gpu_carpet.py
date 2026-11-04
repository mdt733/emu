"""v5045 GPU trainer gate: one eight-prism MJX model, per-world
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
        physics="eight-dynamic-convex-triangular-prisms",
        active_collision_geoms=8,
        terrain_carpet_prisms=True,
    )


def _gpu_tree_bytes(name, tree):
    """Logical JAX array inventory; never copies data from the accelerator.

    This is NOT an allocator/peak report: XLA may reuse buffers or
    compile constants into executables. It distinguishes retained
    JAX GPU arrays from host metadata so they aren't double-counted.
    """
    import jax
    import numpy as np
    items=[]
    for path,leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        if not isinstance(leaf,jax.Array):
            continue
        size=int(np.prod(leaf.shape))*int(np.dtype(leaf.dtype).itemsize)
        items.append(dict(path=jax.tree_util.keystr(path),
                          shape=list(leaf.shape),dtype=str(leaf.dtype),
                          logical_bytes=size))
    items.sort(key=lambda item:item['logical_bytes'],reverse=True)
    return dict(group=name,arrays=len(items),
                logical_bytes=sum(item['logical_bytes'] for item in items),
                top_arrays=items[:16])


def _memory_counters(device):
    """Allocator snapshot, not bytes allocated per physics frame."""
    return {str(k):int(v) for k,v in (device.memory_stats() or {}).items()
            if isinstance(v,(int,float))}



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
    p.add_argument("--profile-rollout",action="store_true",
                   help="independently time the exact 32-frame training rollout "
                        "without PPO gradients; requires >=3 full updates")
    p.add_argument("--reset-forward-capacity",type=int,default=128,metavar="N",
                   help="fixed-size exact MJX forward batch for episode resets; "
                        "0 uses the original all-world reference for matched A/B")
    p.add_argument("--gpu-audit-dir",default=None,metavar="DIR",
                   help="opt-in: capture XProf GPU kernel trace, JAX live "
                        "allocations, XLA buffer analysis and full rollout "
                        "array shapes in DIR; requires --profile-rollout")
    a=p.parse_args()
    if not a.prisms:
        p.error("The active EMU simulator is prism-only; pass --prisms")
    if a.worlds<4 or not 1<=a.frames<=20:p.error("Need >=4 worlds, 1..20 frames")
    if a.reset_forward_capacity<0:p.error("--reset-forward-capacity must be >=0")
    if a.full_update and (a.worlds not in (1024,2048) or not a.ppo or a.device=="cpu"):
        p.error("--full-update requires --worlds 1024 or 2048, --device cuda --ppo")
    if (not 1<=a.benchmark_updates<=8 or
            (a.benchmark_updates>1 and not a.full_update)):
        p.error("--benchmark-updates must be 1..8 and >1 requires --full-update")
    if a.profile_rollout and (not a.full_update or a.benchmark_updates<3):
        p.error("--profile-rollout requires --ppo --full-update and --benchmark-updates >=3")
    if a.gpu_audit_dir and not a.profile_rollout:
        p.error("--gpu-audit-dir requires --profile-rollout and full 2048/1024-world PPO")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false")
    from emu.ppo import configure_jax
    configure_jax("cuda" if a.device=="gpu" else a.device)
    import jax
    import jax.numpy as jnp
    import numpy as np
    from emu.contracts import ROBOT_XML,OBS_SCHEMA_TERRAIN
    from emu.environment import EmuMJXEnv
    from emu.terrain_carpet_mjx import TILE_COUNT, TILE_M, validate_prism_only_geoms

    if a.device in ("cuda","gpu") and jax.default_backend()!="gpu":
        raise RuntimeError("Expected a CUDA-capable GPU")
    env=EmuMJXEnv(
        str(ROBOT_XML),num_envs=a.worlds,seed=735,max_episode_steps=700,
        terrain_step_max_m=.08,observation_schema=OBS_SCHEMA_TERRAIN,
        unified_capability_focus=True,terrain_lesson="coherent-careful",
        academy_stream=True,academy_height_cap_m=.15,terrain_carpet=True,
        terrain_carpet_prisms=True,
        attitude_profile='v5015',pitch_command_max_deg=8.0,
        reset_forward_capacity=a.reset_forward_capacity)
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
        raise RuntimeError("Expected exactly four inactive source hfields")
    validate_prism_only_geoms(__import__('mujoco'),env.mj_model)
    if np.any(env.mj_model.geom_contype[np.asarray(
            env.carpet_geom_ids,dtype=np.int32)]!=0):
        raise RuntimeError("Source heightfield skirts must not collide")
    if len(env.carpet_prism_geom_ids)!=8:
        raise RuntimeError("Expected exactly eight physical prism geoms")
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
    if heights.shape!=(a.worlds,TILE_COUNT*4) or not np.isfinite(heights).all():
        raise RuntimeError("Per-world source hfield geometry must be finite [E,4*TILE_COUNT]")
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
    if tile_positions.shape!=(a.worlds,TILE_COUNT,3):
        raise RuntimeError("Missing independently positioned tiles")
    physical_ids=env.carpet_prism_mocap_ids
    solid_pos=np.asarray(jax.device_get(state.data.mocap_pos[
        :,list(physical_ids),:]))
    if solid_pos.shape!=(a.worlds,8,3) or not np.isfinite(solid_pos).all():
        raise RuntimeError("Invalid per-world physical collision tile poses")
    print("reset PASS: 8 convex prisms, distinct per-world 360-input JAX physics",
          flush=True)
    # One robot crossing into a 75cm tile midpoint must recycle only
    # abandoned slots. If a rear foot still occupies an abandoned tile,
    # the whole window must remain unchanged.
    old=state.carpet_center
    desired=old+jnp.asarray([1,0],dtype=jnp.int32)[None,:]
    root_xy=(desired.astype(jnp.float32)+.75)*TILE_M
    fake_q=state.data.qpos.at[:,0:2].set(root_xy)
    foot_xpos=state.data.xpos.at[:,env.foot_bodies,0:2].set(
        jnp.broadcast_to(root_xy[:,None,:],(a.worlds,2,2)))
    safe_data=state.data.replace(qpos=fake_q,xpos=foot_xpos)
    relocated,new_h,new_c=env._carpet_recycle(state,safe_data)
    if not np.array_equal(np.asarray(jax.device_get(new_c)),
                          np.asarray(jax.device_get(desired))):
        raise RuntimeError("Safe midpoint passage did not advance the tile ring")
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
               ).astype(jnp.float32)+.5)*TILE_M
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
    seam_x=desired[:,0].astype(jnp.float32)*TILE_M+.020
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
            final_steady_metrics=None
            for i in range(1,a.benchmark_updates):
                start_steady=time.perf_counter()
                nxt,steady_metrics,_,_,_=trainer["train_update"](
                    last,packet,jnp.zeros((360,),dtype=jnp.float32),
                    jnp.ones((360,),dtype=jnp.float32),
                    jnp.asarray(.25,jnp.float32))
                jax.block_until_ready((nxt.obs,nxt.params["pi"][-1]["w"]))
                steady_update_times.append(time.perf_counter()-start_steady)
                last=nxt
                final_steady_metrics=steady_metrics
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
            # This mask is already retained for curriculum scoring;
            # analyse it AFTER timing so host transfer cannot skew FPS.
            deaths=np.asarray(jax.device_get(
                jnp.sum(final_steady_metrics["episode_done"]>.5,axis=1)))
            cap=env.reset_forward_capacity
            print("RESET BATCH STATS: "
                  f"mean={np.mean(deaths):.2f}/frame "
                  f"p95={np.percentile(deaths,95):.0f} "
                  f"max={np.max(deaths):.0f} "
                  f"capacity={cap} "
                  f"fallback_frames={np.sum(deaths>cap) if cap else 32}/32",
                  flush=True)
        if a.gpu_audit_dir:
            from pathlib import Path
            out_dir=Path(a.gpu_audit_dir).expanduser().resolve()
            out_dir.mkdir(parents=True,exist_ok=True)
            dev=jax.devices("gpu")[0]
            audit={
                "explanation":"Array totals describe logical trees, NOT "
                              "simultaneously live VRAM or per-step transfers.",
                "jax_version":jax.__version__,
                "backend":jax.default_backend(),
                "worlds":a.worlds,"rollout_frames":32,
                "reset_forward_capacity":env.reset_forward_capacity,
                "tiles":TILE_COUNT,"active_prisms":len(env.carpet_prism_geom_ids),
                "environment":_gpu_tree_bytes("mjx_model",env.mjx_model),
                "mjx_data":_gpu_tree_bytes("mjx_data",last.env_state.data),
                # Static contact slots (NOT active contacts) are central to
                # the 18-vs-8-prism hypothesis. Ask the actual installed
                # MJX model, never derive capacity from geom counts.
                "mjx_contact":_gpu_tree_bytes(
                    "mjx_contact",last.env_state.data._impl.contact),
                "contact_capacity_per_world":int(
                    last.env_state.data._impl.contact.geom1.shape[1]),
                "physical_mujoco_model_counts":{
                    key:int(getattr(env.mj_model,key))
                    for key in ("nq","nv","nbody","ngeom","nu","nmocap")
                    if hasattr(env.mj_model,key)},
                "env_state_excluding_mjx_data":_gpu_tree_bytes(
                    "env_state_excluding_mjx_data",
                    last.env_state._replace(data=None)),
                "network_weights":_gpu_tree_bytes("network_weights",last.params),
                "adam_and_optax_state":_gpu_tree_bytes("adam_and_optax_state",last.opt_state),
                "live_allocator_after_full_ppo":_memory_counters(dev),
            }
            # Pprof shows resident arrays/executables when the fully
            # synchronized update has completed. It is not a peak view.
            jax.profiler.save_device_memory_profile(
                str(out_dir/"device_after_full_ppo.pprof"))
            print("GPU AUDIT: tracing ONE extra steady full PPO update "
                  "(profiler overhead excluded from throughput figures)...",
                  flush=True)
            with jax.profiler.trace(str(out_dir/"xprof")):
                traced,_,_,_,_=trainer["train_update"](
                    last,packet,jnp.zeros((360,),dtype=jnp.float32),
                    jnp.ones((360,),dtype=jnp.float32),
                    jnp.asarray(.25,jnp.float32))
                jax.block_until_ready(traced)
            last=traced
            audit["live_allocator_after_trace"]=_memory_counters(dev)
        if a.profile_rollout:
            # Separate JIT: same 32-frame rollout with actor inference,
            # real prism MJX physics, sensors, reward and next-state value
            # bootstrap, but NO GAE/optimizer epochs. Materializing all
            # output leaves is deliberate: timing only nxt.obs could
            # otherwise dead-code-eliminate reward/sensor diagnostics.
            # The JIT boundary prevents an exact additive split:
            # report this as independent evidence, not "physics = X%".
            rollout_state=last
            rollout_times=[]
            for i in range(3):
                t_rollout=time.perf_counter()
                rollout_state,traj,next_values=trainer["benchmark_rollout"](
                    rollout_state,packet,jnp.zeros((360,),dtype=jnp.float32),
                    jnp.ones((360,),dtype=jnp.float32))
                jax.block_until_ready((rollout_state,traj,next_values))
                seconds=time.perf_counter()-t_rollout
                if i==0:
                    print("separate rollout-only first use "
                          f"(includes JIT)={seconds:.2f}s",flush=True)
                else:
                    rollout_times.append(seconds)
            rollout_mean=sum(rollout_times)/len(rollout_times)
            full_mean=sum(steady_update_times)/len(steady_update_times)
            print("ROLLOUT-ONLY PROFILE: "
                  f"policy+MJX+sensor+reward+bootstrap={rollout_mean:.3f}s "
                  f"for {first_update_world_steps:,} steps; "
                  f"full_PPO={full_mean:.3f}s/update; "
                  f"rollout/full={rollout_mean/full_mean:.1%} "
                  "(separate JIT, not an additive decomposition)",flush=True)
            print(json.dumps(dict(stage="rollout_only_profile",
                full_update_seconds_mean=round(full_mean,4),
                standalone_rollout_seconds_mean=round(rollout_mean,4),
                standalone_rollout_world_steps_per_second=round(
                    first_update_world_steps/rollout_mean,1),
                standalone_rollout_over_full_ratio=round(
                    rollout_mean/full_mean,4))),flush=True)
            if a.gpu_audit_dir:
                audit["rollout_outputs"]=_gpu_tree_bytes("all_rollout_outputs",
                                                        (traj,next_values))
                audit["rollout_metrics"]=_gpu_tree_bytes(
                    "rollout_metrics",traj["metrics"])
                audit["rollout_observations"]=_gpu_tree_bytes(
                    "rollout_observations",traj["obs"])
                audit["rollout_bootstrap_observations"]=_gpu_tree_bytes(
                    "rollout_bootstrap_observations",traj["bootstrap_obs"])
                audit["allocator_with_materialized_rollout"]=_memory_counters(dev)
                jax.profiler.save_device_memory_profile(
                    str(out_dir/"device_with_rollout_outputs.pprof"))
                # Ask XLA for its compiled buffer-assignment budget, which
                # includes temp allocations invisible in Python JAX trees.
                # This lower/compile may incur an ADDITIONAL JIT compile.
                executable=trainer["train_update"].lower(
                    last,packet,jnp.zeros((360,),dtype=jnp.float32),
                    jnp.ones((360,),dtype=jnp.float32),
                    jnp.asarray(.25,jnp.float32)).compile()
                mem=executable.memory_analysis()
                audit["xla_compiled_full_ppo_memory"]=(
                    {key:int(getattr(mem,key)) for key in (
                        "argument_size_in_bytes","output_size_in_bytes",
                        "temp_size_in_bytes","alias_size_in_bytes",
                        "generated_code_size_in_bytes")
                     if getattr(mem,key,None) is not None}
                    if mem is not None else None)
                audit["allocator_after_compiler_report"]=_memory_counters(dev)
                (out_dir/"gpu_allocation_audit.json").write_text(
                    json.dumps(audit,indent=2)+"\n",encoding="utf-8")
                print("GPU AUDIT artifacts:",str(out_dir),flush=True)
                print("GPU AUDIT summary:",json.dumps({
                    "worlds":a.worlds,
                    "mjx_data_logical_bytes":audit["mjx_data"]["logical_bytes"],
                    "rollout_metrics_count":audit["rollout_metrics"]["arrays"],
                    "rollout_metrics_bytes":audit["rollout_metrics"]["logical_bytes"],
                    "rollout_outputs_logical_bytes":audit["rollout_outputs"]["logical_bytes"],
                    "xla_full_ppo_memory":audit["xla_compiled_full_ppo_memory"],
                    "allocator_after_full_ppo":audit["live_allocator_after_full_ppo"]
                }),flush=True)
        if a.full_update:
            dev=jax.devices("gpu")[0]
            stats=dev.memory_stats() or {}
            print("GPU memory:",{k:v for k,v in stats.items()
                if "peak" in k or "limit" in k},flush=True)
    print(json.dumps(dict(status="PASS",worlds=a.worlds,
          reset_forward_capacity=env.reset_forward_capacity,
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
