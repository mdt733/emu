"""Read-only frozen-policy exam on actual FOUR-LIP Academy course geometry.

No training updates. Explicit up/down signs, foot-supported crossings and
complete-course events from the SAME MJX step function used during PPO.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def examine(policy,output,*,worlds=8,horizon=650,speed=.16,seed=1701,
            device="cpu",heights_cm=(1.,2.,3.)):
    if (worlds<4 or worlds>32 or horizon<200 or horizon>700 or
        speed<.04 or speed>.30 or not heights_cm or
        any(h<.2 or h>4 for h in heights_cm)):
        raise ValueError("Need 4..32 worlds, 200..700 frames, 0.04..0.30m/s, 0.2..4cm")
    policy=Path(policy).expanduser().resolve()
    output=Path(output).expanduser().resolve()
    if not policy.is_file() or policy.suffix!=".emupolicy":
        raise ValueError("Exam requires a frozen exported .emupolicy")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("Exam output must be new/empty")
    if output==policy.parent or policy in output.parents:
        raise ValueError("Cannot overwrite a policy with examination results")
    from emu.viewer_server import load_exported_actor_for_exam
    from emu.ppo import configure_jax,make_trainer
    from emu.physical_exam import make_schedule
    from emu.contracts import ROBOT_XML,OBS_SCHEMA_TERRAIN
    from emu.environment import EmuMJXEnv
    from emu.step_academy_world import ACADEMY_PATTERNS
    params,norm,actor=load_exported_actor_for_exam(policy)
    if actor.get("observation_schema")!=OBS_SCHEMA_TERRAIN or not actor.get("array_only"):
        raise ValueError("Only authentic terrain_360 array-only actor supported")
    configure_jax(device)
    import jax
    import jax.numpy as jnp
    import numpy as np
    sensor_settings=dict(actor["sensor_settings"])
    noise=sensor_settings.pop("sensor_noise_scale",0.)
    env=EmuMJXEnv(str(ROBOT_XML),num_envs=worlds,seed=seed,
        max_episode_steps=max(750,horizon+50),observation_schema=OBS_SCHEMA_TERRAIN,
        unified_capability_focus=True,terrain_step_max_m=.08,
        academy_course_lips=4,academy_height_cap_m=.04,
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
          focus_weight_step=1.,athletic_rehearsal_frac=0.,
          regime_frac_steady=1.,regime_frac_transition=0.,
          regime_frac_resilience=0.,steady_difficulty=0.,
          transition_difficulty=0.,resilience_difficulty=0.,
          probe_cmd_vx=speed,probe_cmd_vy=0.,probe_cmd_yaw=0.,
          probe_tilt_rad=0.,probe_push_n=0.,
          terrain_edge_angle_deg=float("nan"),
          terrain_fixed_height_m=h/100)
        print(f"Independent 4-lip exam cap {h:g}cm, {worlds} worlds",flush=True)
        trace=trainer["evaluate_trace"](
            params,env.params_array(packet),
            jnp.asarray(norm["mean"]),jnp.asarray(norm["std"]),
            jax.random.PRNGKey(seed),horizon,0.,None,jnp.asarray(schedule))
        out=jax.device_get(trace)
        initial=np.asarray(out["academy_initial_platform_positions"])
        signed=np.diff(initial[:,:,2]+.25,axis=1)
        if any(tuple(np.sign(signed[w]).astype(int)) !=
               ACADEMY_PATTERNS[w % len(ACADEMY_PATTERNS)]
               for w in range(worlds)):
            raise RuntimeError("Four real physical signed steps were not created")
        active=np.asarray(out["academy_active"])[0]>.5
        if not active.all():
            raise RuntimeError("Frozen exam failed to assign all four-lip worlds")
        names=("academy_cross_new","academy_cross_up","academy_cross_down",
               "academy_course_new","academy_clearance_new",
               "academy_foothold_new","academy_transfer_new")
        ev={k:np.asarray(out[k])>.5 for k in names}
        falls=np.asarray(out["fell"])>.5
        per_world=[]
        for w in range(worlds):
            per_world.append(dict(world=w,pattern=w%len(ACADEMY_PATTERNS),
              signed_lip_heights_cm=np.round(signed[w]*100,3).tolist(),
              supported_up=int(ev["academy_cross_up"][:,w].sum()),
              supported_down=int(ev["academy_cross_down"][:,w].sum()),
              footholds=int(ev["academy_foothold_new"][:,w].sum()),
              weight_transfers=int(ev["academy_transfer_new"][:,w].sum()),
              clearances=int(ev["academy_clearance_new"][:,w].sum()),
              course_completed=bool(ev["academy_course_new"][:,w].any()),
              fell=bool(falls[:,w].any())))
        def total(key):
            return int(ev[key].sum())
        rec=dict(cap_height_cm=h,worlds=per_world,
                 up=total("academy_cross_up"),
                 down=total("academy_cross_down"),
                 lip_crossings=total("academy_cross_new"),
                 courses=int(ev["academy_course_new"].any(axis=0).sum()),
                 footholds=total("academy_foothold_new"),
                 transfers=total("academy_transfer_new"),
                 clearances=total("academy_clearance_new"),
                 worlds_fell=int(falls.any(axis=0).sum()),
                 up_opportunities=worlds*2,down_opportunities=worlds*2)
        outcomes.append(rec)
        print(f"real uphill {rec['up']}/{worlds*2}; downhill {rec['down']}/{worlds*2}; "
              f"complete {rec['courses']}/{worlds}; falls {rec['worlds_fell']}/{worlds}",
              flush=True)
    doc=dict(schema="emu_step_academy_frozen_physical_v1",
             actor_sha256=actor["artifact_sha256"],
             actor_step=actor["source"]["step"],
             protocol=dict(seed=seed,worlds=worlds,horizon=horizon,
                           speed=speed,device=device,
                           heights_cm=list(heights_cm),lips=4,
                           actual_collision=True,no_training=True),
             warnings=["Opportunity counts include assigned lips, including some not reached.",
                       "One trained policy and one seed are not repeatability proof.",
                       "Simulated motor forces do not demonstrate real hardware safety."],
             results=outcomes)
    output.mkdir(parents=True,exist_ok=True)
    (output/"academy_exam.json").write_text(json.dumps(doc,indent=2)+"\n")
    lines=["# EMU four-lip independent frozen MJX physical exam","",
        f"Actor SHA256: {doc['actor_sha256']}","",
        "| Height cap | Supported uphill | Supported downhill | Complete courses | Fell |",
        "|---|---:|---:|---:|---:|"]
    for x in outcomes:
        lines.append(f"| {x['cap_height_cm']:g}cm | {x['up']}/{x['up_opportunities']} "
          f"| {x['down']}/{x['down_opportunities']} | {x['courses']}/{worlds} "
          f"| {x['worlds_fell']}/{worlds} |")
    lines.extend(["","Each counted crossing requires actual far-side foot contact.",
                  "Denominators are assigned opportunities, not necessarily approached lips.",
                  "See academy_exam.json for real signed heights, contact and weight-transfer evidence.",""])
    (output/"academy_exam.md").write_text("\n".join(lines)+"\n")
    return doc


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--worlds",type=int,default=8)
    p.add_argument("--horizon",type=int,default=650)
    p.add_argument("--speed",type=float,default=.16)
    p.add_argument("--seed",type=int,default=1701)
    p.add_argument("--device",choices=("cpu","cuda","gpu","auto"),default="cpu")
    p.add_argument("--heights-cm",type=float,nargs="+",default=[1.,2.,3.])
    a=p.parse_args()
    examine(a.policy,a.output,worlds=a.worlds,horizon=a.horizon,
            speed=a.speed,seed=a.seed,device=a.device,heights_cm=tuple(a.heights_cm))


if __name__=="__main__":
    main()
