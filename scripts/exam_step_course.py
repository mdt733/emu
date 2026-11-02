"""Read-only frozen-policy exam on the real three-platform streaming ring.

Reports every approached UP/DOWN lip separately, with true top-support
crossings, and four-lip milestones. The policy learns nothing here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def examine(policy,output,*,worlds=8,horizon=1050,speed=.16,seed=1701,
            device="cpu",heights_cm=(1.,2.,3.)):
    if (worlds<4 or worlds>32 or horizon<200 or horizon>1200 or
        speed<.04 or speed>.30 or not heights_cm or
        any(h<.2 or h>15 for h in heights_cm)):
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
    from emu.physical_exam import make_schedule
    from emu.contracts import ROBOT_XML,OBS_SCHEMA_TERRAIN
    from emu.environment import EmuMJXEnv
    configure_jax(device)
    import jax
    import jax.numpy as jnp
    import numpy as np

    params,norm,actor=load_exported_actor_for_exam(policy)
    if actor.get("observation_schema")!=OBS_SCHEMA_TERRAIN or not actor.get("array_only"):
        raise ValueError("Only authentic terrain_360 array-only actor supported")
    sensor_settings=dict(actor["sensor_settings"])
    noise=sensor_settings.pop("sensor_noise_scale",0.)
    env=EmuMJXEnv(str(ROBOT_XML),num_envs=worlds,seed=seed,
        max_episode_steps=1300,observation_schema=OBS_SCHEMA_TERRAIN,
        unified_capability_focus=True,terrain_step_max_m=.08,
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
          academy_height_up_m=h/100,academy_height_down_m=h/100,
          focus_weight_step=1.,athletic_rehearsal_frac=0.,
          regime_frac_steady=1.,regime_frac_transition=0.,
          regime_frac_resilience=0.,steady_difficulty=0.,
          transition_difficulty=0.,resilience_difficulty=0.,
          probe_cmd_vx=speed,probe_cmd_vy=0.,probe_cmd_yaw=0.,
          probe_tilt_rad=0.,probe_push_n=0.,
          terrain_edge_angle_deg=float("nan"),
          terrain_fixed_height_m=h/100)
        print(f"Independent streaming ring exam {h:g}cm, {worlds} worlds",flush=True)
        trace=trainer["evaluate_trace"](
            params,env.params_array(packet),
            jnp.asarray(norm["mean"]),jnp.asarray(norm["std"]),
            jax.random.PRNGKey(seed),horizon,0.,None,jnp.asarray(schedule))
        out=jax.device_get(trace)
        initial=np.asarray(out["academy_initial_platform_positions"])
        signed=np.diff(initial[:,:,2]+.25,axis=1)
        if (initial.shape!=(worlds,3,3) or
            not np.allclose(np.abs(signed),h/100,atol=.00025)):
            raise RuntimeError("Three physical initial support slabs were not constructed")
        if not (np.asarray(out["academy_active"])[0]>.5).all():
            raise RuntimeError("Frozen exam must start in streaming Academy")
        signals=("academy_cross_new","academy_cross_up","academy_cross_down",
                 "academy_course_new","academy_lip_attempt_new",
                 "academy_clearance_new","academy_foothold_new",
                 "academy_transfer_new")
        events={k:np.asarray(out[k])>.5 for k in signals}
        lip_height=np.asarray(out["academy_lip_height_m"])
        attempted=events["academy_lip_attempt_new"]
        up_attempts=attempted & (lip_height>.001)
        down_attempts=attempted & (lip_height<-.001)
        falls=np.asarray(out["fell"])>.5
        def total(name):
            return int(events[name].sum())
        per_world=[]
        for w in range(worlds):
            per_world.append(dict(
                world=w,
                first_two_lips_cm=np.round(signed[w]*100,3).tolist(),
                supported_up=int(events["academy_cross_up"][:,w].sum()),
                supported_down=int(events["academy_cross_down"][:,w].sum()),
                approached_up=int(up_attempts[:,w].sum()),
                approached_down=int(down_attempts[:,w].sum()),
                four_lip_milestones=int(events["academy_course_new"][:,w].sum()),
                footholds=int(events["academy_foothold_new"][:,w].sum()),
                weight_transfers=int(events["academy_transfer_new"][:,w].sum()),
                fell=bool(falls[:,w].any())))
        outcome=dict(height_cm=h,worlds=per_world,
            up=total("academy_cross_up"),down=total("academy_cross_down"),
            up_attempts=int(up_attempts.sum()),down_attempts=int(down_attempts.sum()),
            lip_crossings=total("academy_cross_new"),
            four_lip_milestones=total("academy_course_new"),
            footholds=total("academy_foothold_new"),
            transfers=total("academy_transfer_new"),
            clearances=total("academy_clearance_new"),
            worlds_fell=int(falls.any(axis=0).sum()))
        outcomes.append(outcome)
        print(f"UP {outcome['up']}/{outcome['up_attempts']} approached; "
              f"DOWN {outcome['down']}/{outcome['down_attempts']} approached; "
              f"four-lip milestones {outcome['four_lip_milestones']}; "
              f"falls {outcome['worlds_fell']}/{worlds}",flush=True)

    document=dict(schema="emu_step_streaming_frozen_physical_v2",
        actor_sha256=actor["artifact_sha256"],actor_step=actor["source"]["step"],
        protocol=dict(seed=seed,worlds=worlds,horizon=horizon,speed=speed,
                      device=device,heights_cm=list(heights_cm),
                      recycled_platforms=3,actual_collision=True,no_training=True),
        warnings=["Denominators count physically approached lip encounters, not assigned terrain.",
                  "A single trained policy and one seed are not repeatability proof.",
                  "Simulation does not certify that 15cm is possible on real hardware."],
        results=outcomes)
    output.mkdir(parents=True,exist_ok=True)
    (output/"academy_exam.json").write_text(json.dumps(document,indent=2)+"\n")
    lines=["# EMU streaming STEP physical exam","",
      f"Actor SHA256: {document['actor_sha256']}","",
      "| Lip size | Supported UP / approaches | Supported DOWN / approaches | 4-lip milestones | Fell |",
      "|---|---:|---:|---:|---:|"]
    for r in outcomes:
        lines.append(f"| {r['height_cm']:g}cm | {r['up']}/{r['up_attempts']} "
                     f"| {r['down']}/{r['down_attempts']} | {r['four_lip_milestones']} "
                     f"| {r['worlds_fell']}/{worlds} |")
    lines.extend(["","Each crossing requires far-side upper foot support and a fully cleared previous platform.",
                  "Three physical boxes recycle, but an episode has no fixed number of lips.",
                  "See academy_exam.json for first-lip signed heights and each world's evidence.",""])
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
    a=p.parse_args()
    examine(a.policy,a.output,worlds=a.worlds,horizon=a.horizon,
            speed=a.speed,seed=a.seed,device=a.device,heights_cm=tuple(a.heights_cm))


if __name__=="__main__":
    main()
