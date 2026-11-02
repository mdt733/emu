"""Read-only frozen-policy STEP exam: legacy three-slab or v5041 nine-box carpet.

Pass --carpet only for a source-tagged nine-box policy. All reported
crossings come from live MJX contact evidence and are censored after
each world's first terminal event; a policy reset cannot pad its score.
No learning, checkpoint changes or curriculum promotion occurs here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def first_episode_mask(done):
    """Include the first death frame, never any reset-policy frames thereafter."""
    import numpy as np
    terminated=np.asarray(done,dtype=bool)
    if terminated.ndim!=2:
        raise ValueError("Expected [frames, worlds] terminal array")
    earlier=np.concatenate((
        np.zeros((1,terminated.shape[1]),dtype=np.int32),
        np.cumsum(terminated[:-1],axis=0)),axis=0)
    return earlier==0


def independently_score_carpet_crossings(frames):
    """Rescore actual MJX trajectories without reading the STEP reward.

    Reconstruct each live box-top plane using recorded physical geom
    transforms. A new orthogonally adjacent global cell counts only after
    FIVE successive first-life frames with real upward-normal foot contact,
    ball-foot bottom near that target top, and a valid physical footprint.
    Return visits never earn another crossing. Signed lip height is measured
    at the two box planes' SHARED world-space edge midpoint. A smooth
    grade-only crossing is recorded separately, never called a step.

    This scorer consumes NO academy_cross/foothold/teacher success flags.
    """
    import numpy as np

    cells=np.asarray(frames['carpet_root_cell'],dtype=np.int32)
    centres=np.asarray(frames['carpet_solid_geom_xpos'],dtype=float)
    rotations=np.asarray(frames['carpet_solid_geom_xmat'],dtype=float)
    normals=np.asarray(frames['carpet_contact_vertical'],dtype=float)
    contacts=np.asarray(frames['carpet_raw_contact_pairs'],dtype=bool)
    feet=np.asarray(frames['foot_xyz'],dtype=float)
    radii=np.asarray(frames['initial_foot_radii_m'],dtype=float)
    half=np.asarray(frames['initial_carpet_solid_half_sizes'],dtype=float)
    valid=first_episode_mask(frames['done'])
    tcount,worlds=cells.shape[:2]
    if (cells.shape!=(tcount,worlds,2) or
        centres.shape!=(tcount,worlds,9,3) or
        rotations.shape!=(tcount,worlds,9,3,3) or
        normals.shape!=(tcount,worlds,9,2) or
        contacts.shape!=(tcount,worlds,9,2) or
        feet.shape!=(tcount,worlds,2,3) or
        half.shape!=(9,3) or radii.shape!=(2,)):
        raise ValueError('Invalid raw carpet trace shapes')
    if (not np.isfinite(centres).all() or
        not np.isfinite(rotations).all() or
        not np.isfinite(feet).all()):
        raise ValueError('Nonfinite physical carpet trajectory')

    def slot(cell):
        return int(cell[0]%3+3*(cell[1]%3))

    def box_top_at(t,w,idx,xy):
        centre=centres[t,w,idx]
        R=rotations[t,w,idx]
        normal=R[:,2]
        if normal[2]<=.70:
            raise ValueError('Invalid upward real box top normal')
        top=centre+normal*half[idx,2]
        z=float(top[2]-np.dot(normal[:2],xy-top[:2])/normal[2])
        projected=np.asarray((xy[0],xy[1],z))-centre
        local=R.T@projected
        return z, bool(np.all(np.abs(local[:2])<=half[idx,:2]+.003))

    def edge_midpoint(before,after):
        dx=int(after[0]-before[0]);dy=int(after[1]-before[1])
        if (abs(dx)+abs(dy))!=1:
            raise ValueError('Only one shared physical tile edge is valid')
        if dx:
            x=float(max(before[0],after[0])*.50)
            y=float((before[1]+.5)*.50)
        else:
            x=float((before[0]+.5)*.50)
            y=float(max(before[1],after[1])*.50)
        return np.asarray((x,y),dtype=float)

    results=[]
    for w in range(worlds):
        confirmed=tuple(map(int,cells[0,w]))
        visited={confirmed}
        candidate=None
        support_streak=0
        crossings=[]
        smooth_transitions=0
        for t in range(tcount):
            if not valid[t,w]:
                break
            new_cell=tuple(map(int,cells[t,w]))
            dx=new_cell[0]-confirmed[0]
            dy=new_cell[1]-confirmed[1]
            if abs(dx)+abs(dy)!=1 or new_cell in visited:
                support_streak=0
                candidate=None
                continue
            if new_cell!=candidate:
                candidate=new_cell
                support_streak=0
            target=slot(new_cell)
            supported=False
            for foot in range(2):
                if not contacts[t,w,target,foot] or normals[t,w,target,foot]<.70:
                    continue
                foot_xyz=feet[t,w,foot]
                z,inside=box_top_at(t,w,target,foot_xyz[:2])
                if (inside and
                    abs(foot_xyz[2]-radii[foot]-z)<=.009):
                    supported=True
                    break
            # Out-of-contact, unhealthy or collapsed poses cannot accrue
            # the required sustained physical target support streak.
            if not supported or bool(np.asarray(frames['fell'])[t,w]):
                support_streak=0
                continue
            support_streak+=1
            if support_streak<5:
                continue
            xy=edge_midpoint(confirmed,new_cell)
            old_z,_=box_top_at(t,w,slot(confirmed),xy)
            new_z,_=box_top_at(t,w,slot(new_cell),xy)
            rise=float(new_z-old_z)
            if abs(rise)>=.0015:
                crossings.append(dict(from_cell=list(confirmed),
                                      to_cell=list(new_cell),
                                      signed_height_cm=round(rise*100,3),
                                      frame=int(t)))
            else:
                smooth_transitions+=1
            confirmed=new_cell
            visited.add(confirmed)
            candidate=None
            support_streak=0
        results.append(dict(
            world=w,
            independent_up=sum(x['signed_height_cm']>0 for x in crossings),
            independent_down=sum(x['signed_height_cm']<0 for x in crossings),
            smooth_grade_transitions=smooth_transitions,
            independently_supported_lips=crossings))
    return dict(
        up=sum(x['independent_up'] for x in results),
        down=sum(x['independent_down'] for x in results),
        total=sum(x['independent_up']+x['independent_down'] for x in results),
        worlds=results)


def examine(policy,output,*,worlds=8,horizon=1050,speed=.16,seed=1701,
            device="cpu",heights_cm=(1.,2.,3.),carpet=False,slope_deg=0.):
    if (worlds<4 or worlds>32 or horizon<200 or horizon>1200 or
        speed<.04 or speed>.30 or not heights_cm or
        any(h<.2 or h>15 for h in heights_cm) or
        not 0.<=slope_deg<=15.):
        raise ValueError("Need 4..32 worlds, 200..1200 frames, 0.04..0.30m/s, 0.2..15cm")
    policy=Path(policy).expanduser().resolve()
    output=Path(output).expanduser().resolve()
    if not policy.is_file() or policy.suffix!=".emupolicy":
        raise ValueError("Exam requires a frozen exported .emupolicy")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("Exam output must be new/empty")
    if output==policy.parent or policy in output.parents:
        raise ValueError("Cannot overwrite policy with examination results")

    from emu.viewer_server import load_exported_actor_for_exam
    from emu.ppo import configure_jax,make_trainer
    from emu.physical_exam import make_schedule,source_commit
    from emu.contracts import ROBOT_XML,OBS_SCHEMA_TERRAIN
    from emu.environment import EmuMJXEnv
    configure_jax(device)
    import jax
    import jax.numpy as jnp
    import numpy as np

    params,norm,actor=load_exported_actor_for_exam(policy)
    if actor.get("observation_schema")!=OBS_SCHEMA_TERRAIN or not actor.get("array_only"):
        raise ValueError("Only authentic terrain_360 array-only actor supported")
    trained_carpet=(actor.get("training") or {}).get("terrain_carpet") is True
    if bool(carpet)!=trained_carpet:
        raise ValueError("Selected exam collision topology does not match exported policy; "
                         "a nine-box actor requires --carpet and a legacy actor must omit it")
    sensor_settings=dict(actor["sensor_settings"])
    noise=sensor_settings.pop("sensor_noise_scale",0.)
    env=EmuMJXEnv(str(ROBOT_XML),num_envs=worlds,seed=seed,
        max_episode_steps=1300,observation_schema=OBS_SCHEMA_TERRAIN,
        unified_capability_focus=True,terrain_step_max_m=.08,
        terrain_carpet=bool(carpet),
        carpet_transition_cap_deg=float((actor.get("training") or {}).get(
            "carpet_transition_cap_deg",3.)),
        academy_stream=True,academy_height_cap_m=.15,
        academy_course_episode_frames=1300,
        terrain_lesson="coherent-careful",
        attitude_profile=actor["training"]["attitude_profile"],
        pitch_command_max_deg=float(actor["training"]["pitch_command_max_deg"]),
        phase_speed_gain=float(actor["control"]["phase_speed_gain"]),
        velocity_i_ramp_s=float((actor["control"].get("velocity_i") or {}).get("ramp_s",.4)),
        velocity_i_multiplier=float((actor["control"].get("velocity_i") or {}).get("multiplier",1.)),
        sensor_noise_scale=noise,open_frontier=True,combo_mix_frac=0.,
        combo_probe_frac=0.,field_teach_recovery=False,**sensor_settings)
    trainer=make_trainer(env,seed=seed,lr=2e-4,rollout=32,epochs=1,minibatches=1)
    schedule=make_schedule("steady",horizon,speed=speed,lateral=0.,
                           yaw_rate=0.,turn_rate=.25)
    outcomes=[]
    for h in heights_cm:
        packet=dict(specialist_mix=1.,academy_episode_probability=1.,
          terrain_exposure=1.,terrain_height_m=h/100,
          slope_exposure=1. if slope_deg>0. else 0.,
          slope_max_deg=float(slope_deg),
          academy_height_up_m=h/100,academy_height_down_m=h/100,
          focus_weight_step=1.,athletic_rehearsal_frac=0.,
          regime_frac_steady=1.,regime_frac_transition=0.,
          regime_frac_resilience=0.,steady_difficulty=0.,
          transition_difficulty=0.,resilience_difficulty=0.,
          probe_cmd_vx=speed,probe_cmd_vy=0.,probe_cmd_yaw=0.,
          probe_tilt_rad=0.,probe_push_n=0.,
          terrain_edge_angle_deg=float("nan"),
          terrain_fixed_height_m=float("nan") if carpet else h/100)
        print(f"Independent {'nine-box carpet' if carpet else 'streaming ring'} "
              f"exam nominal frontier {h:g}cm, grade≤{slope_deg:g}°, "
              f"{worlds} worlds",flush=True)
        trace=trainer["evaluate_trace"](
            params,env.params_array(packet),
            jnp.asarray(norm["mean"]),jnp.asarray(norm["std"]),
            jax.random.PRNGKey(seed),horizon,0.,None,jnp.asarray(schedule))
        out=jax.device_get(trace)
        if carpet:
            # The old three Academy slabs are deliberately DISABLED. Verify
            # the physical poses of all nine live SOLID tiles instead.
            solids=np.asarray(out["initial_carpet_solid_geom_xpos"])
            orientations=np.asarray(out["initial_carpet_solid_geom_xmat"])
            cells=np.asarray(out["initial_carpet_cells"])
            if (solids.shape!=(worlds,9,3) or
                orientations.shape!=(worlds,9,3,3) or
                cells.shape!=(worlds,9,2) or
                not np.all(np.isfinite(solids)) or
                not np.all(np.isfinite(orientations)) or
                not np.all(orientations[:,:,:,2][:,:,2]>.75)):
                raise RuntimeError("Frozen carpet examiner did not load nine valid solid colliders")
            signed=None
        else:
            initial=np.asarray(out["academy_initial_platform_positions"])
            signed=np.diff(initial[:,:,2]+.25,axis=1)
            if (initial.shape!=(worlds,3,3) or
                not np.allclose(np.abs(signed),h/100,atol=.00025)):
                raise RuntimeError("Three physical initial support slabs were not constructed")
        if not (np.asarray(out["academy_active"])[0]>.5).all():
            raise RuntimeError("Frozen exam must start in streaming Academy")
        # Reset-on-fall is useful for PPO, but a frozen exam must not
        # let repeated post-death restarts inflate the same world's score.
        first_life=first_episode_mask(out["done"])
        signals=("academy_cross_new","academy_cross_up","academy_cross_down",
                 "academy_course_new","academy_lip_attempt_new",
                 "academy_clearance_new","academy_foothold_new",
                 "academy_transfer_new")
        events={k:(np.asarray(out[k])>.5)&first_life for k in signals}
        lip_height=np.asarray(out["academy_lip_height_m"])
        attempted=events["academy_lip_attempt_new"]
        # A failed approach may be reported only on the eventual fall, at
        # which moment the robot may have backed away from the lip. Recover
        # the LAST observed signed physical lip, rather than classifying a
        # real failed UP attempt as height=0 and dropping it silently.
        attempt_height=lip_height.copy()
        for w in range(worlds):
            last=0.
            for t in range(horizon):
                if first_life[t,w] and abs(lip_height[t,w])>.0015:
                    last=float(lip_height[t,w])
                if attempted[t,w] and abs(attempt_height[t,w])<=.0015:
                    attempt_height[t,w]=last
        up_attempts=attempted & (attempt_height>.0015)
        down_attempts=attempted & (attempt_height<-.0015)
        unknown_attempts=attempted & ~(up_attempts|down_attempts)
        falls=(np.asarray(out["fell"])>.5)&first_life
        def total(name):
            return int(events[name].sum())
        supported=events["academy_cross_new"]
        four_lips=(sum(int(supported[:,w].sum())//4 for w in range(worlds))
                   if carpet else total("academy_course_new"))
        observed_heights=np.abs(lip_height[supported])*100
        per_world=[]
        for w in range(worlds):
            valid=np.flatnonzero(first_life[:,w])
            last_index=int(valid[-1]) if len(valid) else 0
            if carpet:
                # These are OBSERVED signed lip samples, not an invented
                # promise that the generator placed exactly h centimetres.
                heights=lip_height[first_life[:,w],w]
                starts=np.flatnonzero((np.abs(heights)>.0015)&
                         np.r_[True,np.abs(heights[:-1])<=.0015])
                first_two=np.round(heights[starts[:2]]*100,3).tolist()
            else:
                first_two=np.round(signed[w]*100,3).tolist()
            per_world.append(dict(
                world=w,
                first_two_lips_cm=first_two,
                supported_up=int(events["academy_cross_up"][:,w].sum()),
                supported_down=int(events["academy_cross_down"][:,w].sum()),
                approached_up=int(up_attempts[:,w].sum()),
                approached_down=int(down_attempts[:,w].sum()),
                unclassified_attempts=int(unknown_attempts[:,w].sum()),
                four_lip_milestones=(
                    int(supported[:,w].sum())//4 if carpet
                    else int(events["academy_course_new"][:,w].sum())),
                footholds=int(events["academy_foothold_new"][:,w].sum()),
                weight_transfers=int(events["academy_transfer_new"][:,w].sum()),
                forward_displacement_m=round(float(
                    np.asarray(out["position_xy"])[last_index,w,0]-
                    np.asarray(out["position_xy"])[0,w,0]),4),
                mean_forward_velocity_mps=round(float(np.mean(
                    np.asarray(out["terrain_vx"])[first_life[:,w],w])),4),
                fell=bool(falls[:,w].any())))
        outcome=dict(height_cm=h,worlds=per_world,
            up=total("academy_cross_up"),down=total("academy_cross_down"),
            up_attempts=int(up_attempts.sum()),down_attempts=int(down_attempts.sum()),
            unclassified_attempts=int(unknown_attempts.sum()),
            lip_crossings=total("academy_cross_new"),
            four_lip_milestones=four_lips,
            observed_crossing_height_cm=(dict(
                count=int(len(observed_heights)),
                min=round(float(np.min(observed_heights)),3),
                median=round(float(np.median(observed_heights)),3),
                max=round(float(np.max(observed_heights)),3))
                if len(observed_heights) else None),
            footholds=total("academy_foothold_new"),
            transfers=total("academy_transfer_new"),
            clearances=total("academy_clearance_new"),
            worlds_fell=int(falls.any(axis=0).sum()))
        if carpet:
            independent=independently_score_carpet_crossings(out)
            outcome["independent_trajectory"]=independent
            outcome["teacher_minus_independent"]={
                "up":outcome["up"]-independent["up"],
                "down":outcome["down"]-independent["down"]}
            for i,world in enumerate(per_world):
                world["independent_supported_up"]=independent["worlds"][i]["independent_up"]
                world["independent_supported_down"]=independent["worlds"][i]["independent_down"]
        outcomes.append(outcome)
        if carpet:
            print(f"Independent physical scorer: UP {independent['up']}, "
                  f"DOWN {independent['down']}; "
                  f"teacher credits UP {outcome['up']}, DOWN {outcome['down']}; "
                  "differences are reported, not suppressed",flush=True)
        print(f"UP {outcome['up']}/{outcome['up_attempts']} approached; "
              f"DOWN {outcome['down']}/{outcome['down_attempts']} approached; "
              f"four-lip milestones {outcome['four_lip_milestones']}; "
              f"falls {outcome['worlds_fell']}/{worlds}",flush=True)

    document=dict(
        schema=("emu_carpet_frozen_physical_v1" if carpet else
                "emu_step_streaming_frozen_physical_v2"),
        actor_sha256=actor["artifact_sha256"],actor_step=actor["source"]["step"],
        actor_source_commit=actor["source"].get("git_commit"),
        examiner_source_commit=source_commit(),
        protocol=dict(seed=seed,worlds=worlds,horizon=horizon,speed=speed,
                      device=device,heights_cm=list(heights_cm),
                      slope_deg=slope_deg,terrain_carpet=bool(carpet),
                      recycled_platforms=9 if carpet else 3,
                      collision_geometry=("nine_tilted_solid_boxes" if carpet else
                                          "three_streaming_slab_boxes"),
                      scoring="first_termination_only",
                      actual_collision=True,no_training=True),
        warnings=[
          "Only first-life observations and crossings count; post-fall resets are excluded.",
          "Approaches that never end in supported crossing or terminal failure "
          "may be censored, so 0/0 is NOT a proof of competence.",
          "Unclassified attempts are reported, not silently discarded.",
          "Height is the nominal Academy frontier; actual source-generated "
          "lip heights vary by global location and seed." if carpet else
          "Old three-slab world tests the assigned constant height.",
          "Teacher crossing credits and independent trajectory crossings "
          "are BOTH reported for carpet; disagreement requires investigation. "
          "Independent scoring reconstructs new box-top support for five frames "
          "from raw MJX contacts/poses and signed physical edge heights.",
          "One frozen policy and one seed are not repeatability proof.",
          "Simulated 15cm capability does not certify physical hardware."],
        results=outcomes)
    output.mkdir(parents=True,exist_ok=True)
    (output/"academy_exam.json").write_text(json.dumps(document,indent=2)+"\n")
    lines=[("# EMU nine-box carpet STEP exam" if carpet else
            "# EMU streaming STEP physical exam"),"",
      f"Actor SHA256: {document['actor_sha256']}",
      f"Actor source: {document['actor_source_commit']}","",
      "First lifetime only: post-fall policy resets never count.","",
      "| Lip size | Supported UP / approaches | Supported DOWN / approaches | 4-lip milestones | Fell |",
      "|---|---:|---:|---:|---:|"]
    for r in outcomes:
        lines.append(f"| {r['height_cm']:g}cm | {r['up']}/{r['up_attempts']} "
                     f"| {r['down']}/{r['down_attempts']} | {r['four_lip_milestones']} "
                     f"| {r['worlds_fell']}/{worlds} |")
    if carpet:
        lines.extend(["",
          "| Nominal | Independent UP | Independent DOWN | Teacher UP | Teacher DOWN |",
          "|---|---:|---:|---:|---:|"])
        for result in outcomes:
            physical=result["independent_trajectory"]
            lines.append(f"| {result['height_cm']:g}cm | {physical['up']} "
                         f"| {physical['down']} | {result['up']} | {result['down']} |")
        lines.extend(["",
            "Independent crossings reconstruct raw box geometry and five "
            "consecutive top-support contacts, separately from training rewards.",
            "Any disagreement with teacher credit requires physical trace review."])
    lines.extend([
        "",
        ("Each crossing requires five consecutive physically supported "
         "frames on a newly visited live tile top; nine solid tiles recycle."
         if carpet else
         "Each crossing requires far-side top support and a fully "
         "cleared previous platform; three physical slabs recycle."),
        ("Nominal frontier challenges vary in height; see JSON for actual "
         "crossed lip height ranges." if carpet else
         "Each assigned lip has the specified fixed height."),
        "A 0/0 denominator does not mean success; few approached lips "
        "may be inconclusive.",
        "See academy_exam.json for per-world first-lifetime evidence.",""])
    (output/"academy_exam.md").write_text("\n".join(lines)+"\n")
    return document


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--worlds",type=int,default=8)
    p.add_argument("--horizon",type=int,default=1050)
    p.add_argument("--speed",type=float,default=.16)
    p.add_argument("--seed",type=int,default=1701)
    p.add_argument("--device",choices=("cpu","cuda","gpu","auto"),default="cpu")
    p.add_argument("--heights-cm",type=float,nargs="+",default=[1.,2.,3.])
    p.add_argument("--carpet",action="store_true",
                   help="use actual nine-box v5041 terrain, refuse older actors")
    p.add_argument("--slope-deg",type=float,default=0.,
                   help="optional mixed grade on carpet; 0 gives step-only exam")
    a=p.parse_args()
    examine(a.policy,a.output,worlds=a.worlds,horizon=a.horizon,
            speed=a.speed,seed=a.seed,device=a.device,
            heights_cm=tuple(a.heights_cm),
            carpet=a.carpet,slope_deg=a.slope_deg)


if __name__=="__main__":
    main()
