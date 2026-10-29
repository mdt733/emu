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
    ap.add_argument("--terrain-max-m",type=float,default=.04)
    ap.add_argument("--growth-recipe",choices=("legacy","main-mature"),default="legacy")
    ap.add_argument("--retention-share",type=float,default=0.)
    ap.add_argument("--terrain-lesson",choices=("mixed","coherent","coherent-direct","coherent-careful"),default="mixed")
    ap.add_argument("--force-program",choices=("gentle","aggressive"),default="aggressive")
    ap.add_argument("--velocity-i-ramp-s",type=float,default=.4)
    ap.add_argument("--velocity-i-multiplier",type=float,default=1.)
    ap.add_argument("--fast-rehearsal-share",type=float,default=0.,
                    help="v5029 physical fast-gait cohort (0..0.5), test real births")
    ap.add_argument("--full-update",action="store_true",
                    help="explicit 2048-world CUDA PPO preflight (no checkpoints/training campaign)")
    args=ap.parse_args()
    if not 0.<=args.fast_rehearsal_share<=.5:
        ap.error("--fast-rehearsal-share must be between 0 and 0.5")
    if args.full_update:
        if args.envs!=2048 or args.device!="cuda":
            ap.error("--full-update requires --device cuda --envs 2048")
    elif not 12<=args.envs<=32:
        ap.error("ordinary smoke uses 12..32 worlds; use --full-update for 2048 CUDA worlds")
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
        max_episode_steps=(700 if args.full_update else 40),
        observation_schema=OBS_SCHEMA_TERRAIN,
        attitude_profile="v5015",terrain_step_max_m=args.terrain_max_m,
        push_max_force_n=4.0,
        retention_share=args.retention_share,
        terrain_lesson=args.terrain_lesson,
        unified_capability_focus=True,
        velocity_i_ramp_s=args.velocity_i_ramp_s,
        velocity_i_multiplier=args.velocity_i_multiplier,
        phase_speed_gain=(1.25 if args.growth_recipe=="main-mature" else .75),
        open_frontier=True,combo_mix_frac=(.88 if args.growth_recipe=="main-mature" else 0.))
    governor=RegimeGovernor(open_frontier=True,unified_capability_focus=True,
                           frontier_push_safety_n=(.7 if args.force_program=="gentle" else 4.))
    params=governor.params_dict()
    params.update(terrain_exposure=.15,terrain_height_m=.004)
    # Verify the true cold start retains main's steady-only curriculum:
    # role IDs exist but specialty lessons and motor burden are inactive.
    cold, cold_obs = env.reset(params)
    cold_obs.block_until_ready()
    assert not bool(jax.device_get(jnp.any(cold.specialist_active)))
    # Even if a governor has seeded a prospective slope/step envelope,
    # the first-20M teacher blend keeps real collision terrain dormant.
    import numpy as np
    cold_n=np.asarray(jax.device_get(
        cold.data.geom_xmat[:,int(env.ids['floor_geom']),:])).reshape(
        (args.envs,3,3))[:,:,2]
    assert np.allclose(cold_n,np.tile([0.,0.,1.],(args.envs,1)),atol=1e-5)
    cold_upper=np.asarray(jax.device_get(
        cold.data.mocap_pos[:,env.terrain_step_mocap_id,2]))
    assert np.all(cold_upper<-1.),'foundation terrain slab must stay buried'
    # Verify the optional early grace-only cohort cannot accidentally
    # activate slope/step/advanced-command roles during foundation.
    from emu.capability_focus import CONTROL
    params['foundation_control_mix']=1.0
    nursery,_=env.reset(params)
    active=np.asarray(jax.device_get(nursery.specialist_active))
    assert active.any()
    assert np.all(ids[active]==CONTROL)
    params['foundation_control_mix']=0.0
    # Changing the dynamic categorical weights actually reassigns worlds
    # on episode birth; all active worlds can be taught STEP regardless of
    # their immutable environment index.
    params.update({f'focus_weight_{name}':(1. if name=='step' else 0.)
                   for name in FOCUS_NAMES})
    params['specialist_mix']=1.0
    reassigned,_=env.reset(params)
    assert np.all(np.asarray(jax.device_get(reassigned.focus_id))==3)
    assert bool(jax.device_get(jnp.all(reassigned.specialist_active)))
    if args.fast_rehearsal_share>0.:
        # The complete v5029 birth contract, on REAL batched MJX worlds:
        # several roles plus 25% (configured) plain fast flat rehearsal,
        # with no commander rewrite and no phantom step under the runner.
        from emu.environment import T, UNIFIED_SLAB_HALF_LENGTH_M, UNIFIED_SLAB_HALF_WIDTH_M
        from emu.contracts import CONTROL_DT
        assert UNIFIED_SLAB_HALF_LENGTH_M>2.*700*CONTROL_DT
        assert UNIFIED_SLAB_HALF_WIDTH_M>700*CONTROL_DT
        params.update(athletic_rehearsal_frac=args.fast_rehearsal_share,
                      terrain_exposure=1., terrain_height_m=.035,
                      slope_exposure=1., slope_max_deg=5.)
        fast,_=env.reset(params)
        ath=np.asarray(jax.device_get(fast.task[:,T.athletic_rehearsal]))>.5
        selected=int(ath.sum())
        assert selected>0, "No fast rehearsal selected despite configured share"
        if args.envs==2048:
            assert abs(ath.mean()-args.fast_rehearsal_share)<.045
        ath_commands=np.asarray(jax.device_get(fast.task[:,T.vx]))[ath]
        ath_lateral=np.asarray(jax.device_get(fast.task[:,T.vy]))[ath]
        ath_turn=np.asarray(jax.device_get(fast.task[:,T.yaw]))[ath]
        assert np.all(ath_commands>=.0599)
        assert np.max(np.abs(ath_lateral))<=.0501
        assert np.max(np.abs(ath_turn))<=.2501
        if args.envs==2048:
            assert np.any(ath_commands>.30), "No real >0.3m/s commands in fast cohort"
        selected_active=np.asarray(jax.device_get(fast.specialist_active))
        assert not np.any(selected_active[ath]),"specialist rewrote fast runner"
        upper=np.asarray(jax.device_get(
            fast.data.mocap_pos[:,env.terrain_step_mocap_id,2]))
        assert np.all(upper[ath]<-1.),"fast runner has an unexpected step"
        normals=np.asarray(jax.device_get(
            fast.data.geom_xmat[:,int(env.ids['floor_geom']),:])).reshape(
                (args.envs,3,3))[:,:,2]
        assert np.allclose(normals[ath],(0.,0.,1.),atol=1e-5), (
            "fast rehearsal must use real flat support")
        print(f"FAST REHEARSAL PASS: {selected}/{args.envs} full-speed flat births; "
              f"max requested={ath_commands.max():.3f}m/s; "
              f"collision half-length={UNIFIED_SLAB_HALF_LENGTH_M:g}m",flush=True)
        params.update(athletic_rehearsal_frac=0., terrain_exposure=.15,
                      terrain_height_m=.004,slope_exposure=0.,slope_max_deg=0.)
    if args.terrain_lesson=="coherent-careful":
        # The treatment preserves A's slower actual STEP command band:
        # 0.07-0.13m/s, not the unsuccessful B direct-command change.
        from emu.environment import T
        speeds=np.asarray(jax.device_get(reassigned.task[:,T.vx]))
        assert np.all((speeds>=.0699)&(speeds<=.1301)), speeds
        print(f"Considered STEP request: {speeds.min():.3f}.."
              f"{speeds.max():.3f}m/s",flush=True)
        if not args.full_update:
            # Force a real UP lip and position the torso 4cm before it,
            # to prove the new per-world relief flag is reachable ONLY
            # on the up-facing STEP worlds, not all focused environments.
            params['terrain_exposure']=1.0
            lip_state,_=env.reset(params)
            geoms=np.asarray(jax.device_get(lip_state.data.geom_xmat[:,int(env.terrain_step_geom_id),:])).reshape(
                (args.envs,3,3))
            center=np.asarray(jax.device_get(lip_state.data.mocap_pos[:,env.terrain_step_mocap_id,:]))
            from emu.environment import UNIFIED_SLAB_HALF_LENGTH_M
            lip=center-UNIFIED_SLAB_HALF_LENGTH_M*geoms[:,:,0]
            target=lip-.04*geoms[:,:,0]
            altered=lip_state.data.replace(
                qpos=lip_state.data.qpos.at[:,:2].set(jnp.asarray(target[:,:2])))
            altered=env.forward_batched(env.mjx_model,altered)
            flags=np.asarray(jax.device_get(env._considered_step_window(lip_state,altered)))
            assert flags[::2].all() and not flags[1::2].any(), flags
            assert np.all(np.asarray(jax.device_get(lip_state.terrain_care_frames))==0)
            params['terrain_exposure']=.15
            print("Considered STEP gate: actual UP near-lip only; no DOWN relief",flush=True)
    if args.terrain_lesson=="coherent-direct":
        # Test the REAL post-specialist teacher command, not merely
        # the intended .10-.22 pre-attenuation sample.
        from emu.environment import T
        episode_vx=np.asarray(jax.device_get(reassigned.task[:,T.vx]))
        episode_vy=np.asarray(jax.device_get(reassigned.task[:,T.vy]))
        episode_yaw=np.asarray(jax.device_get(reassigned.task[:,T.yaw]))
        assert np.all((episode_vx>=.0999)&(episode_vx<=.2201)),episode_vx
        assert np.all(np.abs(episode_vy)<1e-6),episode_vy
        assert np.all(np.abs(episode_yaw)<1e-6),episode_yaw
        print(f"coherent-direct physical lesson request "
              f"{episode_vx.min():.3f}..{episode_vx.max():.3f}m/s, "
              "vx above ascent reward eligibility",flush=True)
    # Restore static reference roles for the remaining geometry smoke.
    params.update({f'focus_weight_{name}':0. for name in FOCUS_NAMES})
    params['specialist_mix']=0.0
    # Explicitly override specialist_mix for full physical acceptance.
    # The ordinary cold profile can hand over at either 10M or 20M.
    from emu.capability_focus import growth_mix
    assert growth_mix(step=0,foundation_steps=20_000_000,
                      steady_difficulty=.0) == 0.
    assert growth_mix(step=20_000_000,foundation_steps=20_000_000,
                      steady_difficulty=.40) == 0.
    assert growth_mix(step=25_000_000,foundation_steps=20_000_000,
                      steady_difficulty=.0) == growth_mix(
                          step=25_000_000,foundation_steps=20_000_000,
                          steady_difficulty=.95)
    assert 0. < growth_mix(step=25_000_000,foundation_steps=20_000_000,
                           steady_difficulty=.0) < 1.0
    assert growth_mix(step=30_000_000,foundation_steps=20_000_000,
                      steady_difficulty=.0) == 1.0
    params['steady_difficulty']=.35
    params['specialist_mix']=1.0
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
    active=np.linalg.norm(ups-np.asarray((0.,0.,-2.)),axis=1)>1.
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
    if args.velocity_i_multiplier>1.:
        assert stepped.state.velocity_i_integral.shape==(args.envs,2)
        assert bool(jax.device_get(jnp.all(jnp.isfinite(
            stepped.state.velocity_i_integral))))
        for field in ('policy_cmd_vx','policy_cmd_vy','velocity_i_x','velocity_i_y'):
            assert field in stepped.metrics
    for key in ("specialist_focus","episode_command_quality",
                "command_probe_radius","command_probe_weight_x",
                "terrain_active_up","servo_shaft_drive_power_w",
                "physical_step_height_m","terrain_episode_near_attempt_up",
                "terrain_episode_easy_success_down",
                "physical_slope_deg","slope_episode_attempt_uphill",
                "slope_episode_success_cross_right",
                "slope_episode_near_attempt_uphill",
                "slope_episode_easy_success_downhill"):
        arr=stepped.metrics[key]
        assert arr.shape==(args.envs,),(key,arr.shape)
        assert bool(jax.device_get(jnp.all(jnp.isfinite(arr)))),key
    # Large mode exercises the ACTUAL production update shape, not a tiny
    # two-frame proxy. Use physically plausible post-foundation challenges
    # rather than the geometric acceptance overrides above.
    if args.full_update:
        params.update(terrain_pitch_deg=float("nan"),terrain_roll_deg=float("nan"),
                      terrain_fixed_height_m=float("nan"),terrain_edge_angle_deg=float("nan"),
                      terrain_exposure=.15,terrain_height_m=.004,
                      slope_exposure=.15,slope_max_deg=.7,specialist_mix=.35)
    rollout=32 if args.full_update else 2
    trainer=make_trainer(env,seed=733,lr=2e-4,rollout=rollout,
                         epochs=(4 if args.full_update else 1),
                         minibatches=(8 if args.full_update else 1))
    first=trainer["make_initial_state"](params)
    import time
    started=time.perf_counter()
    updated,rolled,diag,_,_=trainer["train_update"](
        first,env.params_array(params),
        jnp.zeros((TERRAIN_OBS_DIM,),jnp.float32),
        jnp.ones((TERRAIN_OBS_DIM,),jnp.float32),
        jnp.asarray(.25,jnp.float32))
    assert all(bool(jax.device_get(jnp.all(jnp.isfinite(p))))
               for p in jax.tree_util.tree_leaves(updated.params))
    assert bool(jax.device_get(jnp.isfinite(diag["rollout_reward_mean"])))
    if args.full_update:
        elapsed=time.perf_counter()-started
        print(f"CUDA 2048 production-shape PPO: {args.envs*rollout:,} frames, "
              f"{elapsed:.1f}s including compile, "
              f"{args.envs*rollout/max(elapsed,1e-6):,.0f} compile-inclusive fps",flush=True)
        try:
            stats=jax.devices("gpu")[0].memory_stats()
            if stats:
                print("GPU memory: "+str({k:v for k,v in stats.items()
                      if "peak" in k or "limit" in k}),flush=True)
        except (AttributeError,RuntimeError,ValueError):
            pass
    from emu.results import summarize_rollout
    row=summarize_rollout(rolled,frontier_params=params)
    for k in ("focus_worlds_command","focus_worlds_control",
              "command_probe_attempts_vx_hi","control_shaft_j_per_m"):
        assert k in row,k
    # Evidence serialises with the same governor as the neural policy.
    events=governor.update(row)
    assert 'physical_slope_deg_max' in row
    for direction in governor.SLOPE_DIRECTIONS:
        assert f'slope_attempts_{direction}' in row
        assert f'slope_successes_{direction}' in row
    # Synthetic EPISODE evidence isolates the governance decision from
    # learning effectiveness: no lucky frame, no surrogate PPO reward.
    # The slope specialist starts collecting tiny real inclines without
    # waiting for a bootstrap update count or another regime's competence.
    governor.core_competence=.0
    governor.fall_ema['posture']=1.
    assert governor.advance_slope(row)=='seed_real_incline_0p7deg'
    assert governor.advance_slope({})=='collect_real_slope'
    qualified={f'slope_near_{kind}_{direction}':12.
               for direction in governor.SLOPE_DIRECTIONS
               for kind in ('attempts','stability_successes')}
    # Command accuracy alone cannot certify physical supported practice.
    tracking_only={f'slope_near_successes_{d}':12.
                   for d in governor.SLOPE_DIRECTIONS}
    assert governor.advance_slope(tracking_only)=='collect_real_slope'
    assert governor.advance_slope(qualified)=='expand_real_slope_angle'
    assert governor.params.slope_max_deg>.70
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
