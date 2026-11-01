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
    parser.add_argument("--events",action="store_true",
                        help="prove each of four real supported contacts is scored once")
    parser.add_argument("--sensor-foresight",action="store_true",
                        help="verify forward 4x4 ToF rays see successive real platform faces")
    parser.add_argument("--full-update",action="store_true",
                        help="production 2048x32 PPO gradient and memory preflight; no training saved")
    args=parser.parse_args()
    if args.worlds<4 or args.worlds%4 or args.frames<1 or args.frames>20:
        parser.error("worlds must be divisible by 4 and >=4; frames 1..20")
    if args.full_update and (args.worlds!=2048 or not args.ppo or
                             args.device not in ("gpu","cuda")):
        parser.error("--full-update requires GPU, --worlds 2048 and --ppo")
    if args.events and args.worlds>32:
        parser.error("--events is a small-world physical state-machine check, not a full batch")
    import os
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false")
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
        "academy_height_m":.020,
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
    # Prove a robot FOOT can physically contact the middle platform, not
    # merely that the platform is ray-visible. Test several feasible body
    # heights, using actual MJX contact pairs and the real six-servo robot.
    from emu.directional import contact_pair_mask
    from emu.environment import NEUTRAL_ROOT_Z
    mid=int(ids[1])
    centre=jnp.asarray(geom[:,1,:])
    top=centre[:,2]+float(m.geom_size[mid,2])
    all_contacts=[]
    for offset in (-.005,-.018,-.030):
        qpos=(state.data.qpos.at[:,0:2].set(centre[:,:2])
              .at[:,2].set(top+NEUTRAL_ROOT_Z+offset))
        placed=env.forward_batched(env.mjx_model,
                                   state.data.replace(qpos=qpos))
        contact=getattr(placed,'_impl',placed).contact
        foot_pair=contact_pair_mask(jnp,contact.geom1,contact.geom2,
                                    contact.dist,mid,
                                    tuple(int(v) for v in env.ids['foot_geoms']))
        all_contacts.append(np.asarray(jax.device_get(jnp.any(
            foot_pair,axis=1))))
    contacted=np.any(np.stack(all_contacts,axis=0),axis=0)
    if not np.any(contacted & active):
        raise RuntimeError("Ray-visible Academy middle platform cannot support a real MJX foot")
    print("Academy intermediate-platform FOOT CONTACT:",
          int(np.sum(contacted&active)),"/",int(active.sum()),flush=True)
    if args.sensor_foresight:
        # A vertical diagnostic ray alone proves a box exists, NOT that the
        # robot's real forward-facing 4x4 has useful anticipatory data. Move
        # the actual torso to an approach pose before each distinct lip and
        # prove at least one of its sixteen slant rays returns the NEXT box.
        from emu.contracts import (TERRAIN_TOF_PITCH_DEG,TERRAIN_TOF_YAW_DEG,
                                   TERRAIN_TOF_MIN_M,TERRAIN_TOF_MAX_M)
        import math
        beam=jnp.asarray([
            [math.cos(math.radians(p))*math.cos(math.radians(y)),
             math.cos(math.radians(p))*math.sin(math.radians(y)),
             -math.sin(math.radians(p))]
            for p in TERRAIN_TOF_PITCH_DEG for y in TERRAIN_TOF_YAW_DEG],
            dtype=jnp.float32)
        beam_hits=[]
        for lip_i in range(ACADEMY_LIPS):
            R_floor=jnp.asarray(
                state.data.geom_xmat[:,int(ids[0]),:]).reshape(args.worlds,3,3)
            forward=R_floor[:,:,0]
            lip_x=ACADEMY_EDGE_X_M+lip_i*ACADEMY_INTERVAL_M
            xyz=state.data.qpos[:,:3]
            previous_top=state.data.geom_xpos[:,int(ids[lip_i]),2]+float(
                m.geom_size[int(ids[lip_i]),2])
            # Probe multiple legitimate approach distances. It is enough
            # that the next physical lip is visible BEFORE root crossing.
            found=np.zeros((args.worlds,),dtype=bool)
            for standoff in (.08,.13,.19):
                xyz_new=(xyz.at[:,:2].set(
                    forward[:,:2]*(lip_x-standoff))
                    .at[:,2].set(previous_top+NEUTRAL_ROOT_Z))
                physical=env.forward_batched(env.mjx_model,
                    state.data.replace(qpos=state.data.qpos.at[:,:3].set(
                        xyz_new)))
                torso=physical.xpos[:,env.torso_id,:]
                ray=jnp.einsum('eij,kj->eki',
                    env._rotation_body_to_world(physical),beam)
                actual,valid=env._terrain_4x4_ranges(physical)
                closest=jnp.min(jnp.stack([
                    env._ray_box_distance(physical,int(g),torso,ray)
                    for g in ids],axis=1),axis=1)
                target=env._ray_box_distance(
                    physical,int(ids[lip_i+1]),torso,ray)
                front=(jnp.abs(target-closest)<1e-5) & (
                    target>=TERRAIN_TOF_MIN_M) & (
                    target<=TERRAIN_TOF_MAX_M) & valid
                # Actual no-noise sensor packet must match target face range
                # for at least one unoccluded forward ray per world.
                front=front & (jnp.abs(actual-target)<1e-4)
                found|=np.asarray(jax.device_get(jnp.any(front,axis=1)))
            print(f"ToF lip {lip_i+1} forward visibility",
                  int(found.sum()),"/",args.worlds,flush=True)
            if not found.all():
                raise RuntimeError(f"Forward 4x4 blind to physical lip {lip_i+1} "
                                   f"in worlds {np.flatnonzero(~found).tolist()}")
            beam_hits.append(int(found.sum()))
        print("Academy 4x4 SENSOR FORESIGHT PASS: all four consecutive lips "
              "visible BEFORE robot reaches them",flush=True)
    # The test also steps real MJX and verifies the actor packet stays finite.
    for i in range(args.frames):
        result=env.step(state,jnp.zeros((args.worlds,6),dtype=jnp.float32),packet)
        state=result.state
        if not np.isfinite(np.asarray(jax.device_get(result.obs))).all():
            raise RuntimeError("nonfinite observation during Academy real physics")
        print("MJX frame",i+1,"reward",float(jnp.mean(result.reward)),flush=True)
    if args.events:
        # Controller-independent evidence test: position the ACTUAL robot
        # with REAL foot contact on four sequential platforms. This is not
        # a learning claim. It validates that the physical ledger advances
        # from lip 1 through 4, signs each as up/down and cannot farm an
        # already completed course. Nothing is written to checkpoints.
        evidence_state=state._replace(
            data=state.data,
            academy_lip_index=jnp.zeros_like(state.academy_lip_index),
            academy_streak=jnp.zeros_like(state.academy_streak),
            academy_crossings=jnp.zeros_like(state.academy_crossings),
            academy_up_crossings=jnp.zeros_like(state.academy_up_crossings),
            academy_down_crossings=jnp.zeros_like(state.academy_down_crossings),
            academy_clearance_seen=jnp.zeros_like(state.academy_clearance_seen),
            academy_foothold_seen=jnp.zeros_like(state.academy_foothold_seen),
            academy_transfer_seen=jnp.zeros_like(state.academy_transfer_seen))
        foot_geoms=tuple(int(x) for x in env.ids['foot_geoms'])
        for lip_i in range(4):
            target_geom=int(ids[lip_i+1])
            R=jnp.asarray(evidence_state.data.geom_xmat[:,int(ids[0]),:]).reshape(
                args.worlds,3,3)
            along=R[:,:,0]
            edge=ACADEMY_EDGE_X_M+lip_i*ACADEMY_INTERVAL_M
            xy=along[:,:2]*(edge+0.5*ACADEMY_INTERVAL_M)
            target_top=evidence_state.data.geom_xpos[:,target_geom,2]+float(
                m.geom_size[target_geom,2])
            selected=None
            for offset in (-.005,-.018,-.030,-.044):
                qpos=(evidence_state.data.qpos.at[:,0:2].set(xy)
                    .at[:,2].set(target_top+NEUTRAL_ROOT_Z+offset))
                candidate=env.forward_batched(env.mjx_model,
                    evidence_state.data.replace(qpos=qpos))
                collision=getattr(candidate,'_impl',candidate).contact
                pairs=contact_pair_mask(jnp,collision.geom1,collision.geom2,
                        collision.dist,target_geom,foot_geoms)
                contact_ok=jnp.any(pairs,axis=1)
                if selected is None:
                    selected=(qpos,contact_ok)
                else:
                    chosen,has=selected
                    selected=(jnp.where((~has&contact_ok)[:,None],qpos,chosen),
                              has|contact_ok)
            chosen,has=selected
            if not bool(jax.device_get(jnp.all(has))):
                raise RuntimeError(f"lip {lip_i} target cannot physically support every real foot")
            placed=env.forward_batched(env.mjx_model,
                evidence_state.data.replace(qpos=chosen))
            evidence_state=evidence_state._replace(data=placed)
            zero=jnp.zeros((args.worlds,),dtype=jnp.float32)
            physical=dict(bad_contact=zero,terminal=zero,
                          support=jnp.ones_like(zero))
            for frame in range(5):
                ev=env._academy_events(evidence_state,placed,physical)
                evidence_state=evidence_state._replace(
                    academy_lip_index=ev['lip_index'],
                    academy_streak=ev['streak'],
                    academy_crossings=ev['crossings'],
                    academy_up_crossings=ev['up_count'],
                    academy_down_crossings=ev['down_count'],
                    academy_clearance_seen=ev['clearance_seen'],
                    academy_foothold_seen=ev['foothold_seen'],
                    academy_transfer_seen=ev['transfer_seen'])
                now=jax.device_get(ev['cross_new'])
                if frame<4 and bool(np.any(now)):
                    raise RuntimeError(f"lip {lip_i} scored without five physical support frames")
            if not bool(jax.device_get(jnp.all(ev['cross_new']))):
                raise RuntimeError(f"lip {lip_i} not credited after five supported frames")
            if not bool(jax.device_get(jnp.all(
                    evidence_state.academy_lip_index==lip_i+1))):
                raise RuntimeError(f"lip {lip_i} index failed to advance")
        if not bool(jax.device_get(jnp.all(evidence_state.academy_crossings==4))):
            raise RuntimeError("four real lip contacts were not counted independently")
        if not bool(jax.device_get(jnp.all(evidence_state.academy_up_crossings==2))):
            raise RuntimeError("uphill lip direction evidence was incorrect")
        if not bool(jax.device_get(jnp.all(evidence_state.academy_down_crossings==2))):
            raise RuntimeError("downhill lip direction evidence was incorrect")
        replay=env._academy_events(evidence_state,evidence_state.data,physical)
        if bool(jax.device_get(jnp.any(replay['cross_new']))) or bool(
                jax.device_get(jnp.any(replay['reward']>0))):
            raise RuntimeError("finished four-lip course can farm repeated completion rewards")
        print("Academy EVENT LEDGER PASS: 4 genuine support stages, "
              "2 up/2 down, five-frame crossing guard, no repeat reward",
              flush=True)
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
        if args.full_update:
            try:
                mem=jax.devices("gpu")[0].memory_stats()
                print("ACADEMY GPU memory:",
                      {k:v for k,v in (mem or {}).items()
                       if "peak" in k or "limit" in k},flush=True)
            except (AttributeError,RuntimeError,ValueError):
                print("GPU memory stats not exposed by this JAX backend",flush=True)
    print(json.dumps(dict(status="PASS",academy_worlds=int(active.sum()),
                          lips=4,obs_dim=360,physics_frames=args.frames,
                          ppo_gradient=bool(args.ppo),
                          production_batch=bool(args.full_update),
                          per_lip_events=bool(args.events),
                          source="real MJX collision and actual ray-OBB surfaces")),flush=True)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
