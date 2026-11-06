"""v5053: one GPU matrix of independent physical capability trials.

Read-only frozen .emupolicy evaluation. EXACT signed 75cm-seam steps and
continuous signed grades use the same 8 real convex prisms as live training.
The per-world result accumulator is JIT-compiled and returns no traces.
Nothing here updates a neural weight, optimizer or curriculum.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np

KINDS=("stand","slow","fast","lateral","yaw","combined",
       "slope_up","slope_down","step_up","step_down")


@dataclass(frozen=True)
class Trial:
    world: int
    kind: str
    level: int
    replicate: int
    vx: float
    vy: float
    yaw: float
    exact_step_m: float
    exact_grade_deg: float
    initial_kick_mps: float


def trial_spec(kind: str, level: int, levels: int, replicate: int, world: int) -> Trial:
    if kind not in KINDS or not 1<=level<=levels:
        raise ValueError("unknown trial kind/level")
    f=level/levels
    sign=1. if replicate%2==0 else -1.
    vx=vy=yaw=step=grade=kick=0.
    if kind=="stand":
        kick=sign*.30*f
    elif kind=="slow":
        vx=.02+.18*f
    elif kind=="fast":
        vx=.20+.75*f
    elif kind=="lateral":
        vy=sign*.20*f
    elif kind=="yaw":
        yaw=sign*1.3*f
    elif kind=="combined":
        vx=.18+.32*f
        vy=sign*.09*f
        yaw=sign*.65*f
    elif kind=="slope_up":
        vx=.25
        grade=12.*f
    elif kind=="slope_down":
        vx=.25
        grade=-12.*f
    elif kind=="step_up":
        vx=.25
        step=.05*f
    elif kind=="step_down":
        vx=.25
        step=-.05*f
    return Trial(world,kind,level,replicate,vx,vy,yaw,step,grade,kick)


def assignments(worlds: int, *, mode: str="balanced", levels: int=20,
                trials_per_level: int|None=None) -> list[Trial]:
    if mode not in ("balanced","stairs") or not 1<=levels<=40 or worlds<2:
        raise ValueError("invalid matrix mode/geometry")
    trials=10 if trials_per_level is None and mode=="balanced" else (
        100 if trials_per_level is None else trials_per_level)
    if not 1<=trials<=1000:
        raise ValueError("trials_per_level must be 1..1000")
    kinds=KINDS if mode=="balanced" else ("step_up",)
    count=len(kinds)*levels*trials
    if count>worlds:
        raise ValueError(
            f"Need at least {count} worlds for {mode}: "
            f"{len(kinds)} kinds x {levels} heights x {trials} attempts")
    result=[]
    for kind in kinds:
        for level in range(1,levels+1):
            for rep in range(trials):
                result.append(trial_spec(kind,level,levels,rep,len(result)))
    # Unused worlds stay in batch as explicitly labelled filler; never
    # enter the registered challenge's success denominator.
    while len(result)<worlds:
        result.append(Trial(len(result),"padding",0,0,0.,0.,0.,0.,0.,0.))
    return result


def wilson_95(successes:int,n:int)->tuple[float,float]:
    if n<=0:
        return 0.,0.
    p=successes/n
    z=1.95996398454
    d=1.+z*z/n
    m=(p+z*z/(2*n))/d
    h=z*np.sqrt(p*(1.-p)/n+z*z/(4*n*n))/d
    return max(0.,m-h),min(1.,m+h)


def reduce_results(trials:list[Trial],result:dict[str,np.ndarray]):
    """Host arithmetic over per-world GPU output: never average labels away."""
    grouped={}
    detail=[]
    for i,tr in enumerate(trials):
        if tr.kind=="padding":
            continue
        record=asdict(tr)
        for key,values in result.items():
            v=np.asarray(values)[i]
            record[key]=(v.item() if hasattr(v,"item") else v)
        record["actual_lip_error_mm"]=round(
            1000.*(record["actual_lip_m"]-tr.exact_step_m),4)
        record["actual_grade_error_deg"]=round(
            record["actual_grade_deg"]-tr.exact_grade_deg,4)
        detail.append(record)
        grouped.setdefault((tr.kind,tr.level),[]).append(record)
    rows=[]
    for (kind,level),records in sorted(grouped.items()):
        n=len(records)
        step=kind.startswith("step_")
        finished=sum(bool(x["completed_destination"]) for x in records) if step else 0
        arrived=sum(bool(x["both_feet_top_five"]) for x in records) if step else 0
        leading=sum(bool(x["lead_foot_top"]) for x in records) if step else 0
        survived=sum(bool(x["survived"]) for x in records)
        low,high=wilson_95(finished,n)
        rows.append(dict(
            kind=kind,level=level,worlds=n,
            requested_step_mm=round(records[0]["exact_step_m"]*1000.,4),
            measured_step_mm_mean=round(float(np.mean(
                [x["actual_lip_m"]*1000. for x in records])),4),
            requested_grade_deg=records[0]["exact_grade_deg"],
            measured_grade_deg_mean=round(float(np.mean(
                [x["actual_grade_deg"] for x in records])),4),
            leading_top=leading,each_foot_top_five=arrived,
            completed_destinations=finished,
            completion_rate=round(finished/n,5) if step else None,
            completion_wilson95_lo=round(low,5) if step else None,
            completion_wilson95_hi=round(high,5) if step else None,
            survival_rate=round(survived/n,5),
            speed_mae_mps=round(float(np.mean(
                [x["speed_mae_mps"] for x in records])),5),
            yaw_mae_rad_s=round(float(np.mean(
                [x["yaw_mae_rad_s"] for x in records])),5),
            best_forward_progress_m=round(float(np.mean(
                [x["best_forward_progress_m"] for x in records])),5)))
    return detail,rows


def save_csv(path:Path,rows:list[dict]):
    if not rows: raise ValueError("No rows to save")
    with path.open("w",newline="",encoding="utf-8") as fd:
        writer=csv.DictWriter(fd,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_matrix(*,policy:Path,output:Path,mode:str="balanced",
               worlds:int=2048,horizon:int=350,seed:int=1701,
               levels:int=20,trials_per_level:int|None=None,
               device:str="cuda"):
    from emu.contracts import ROBOT_XML,OBS_SCHEMA_TERRAIN
    from emu.policy_export import load_exported_actor_for_exam
    from emu.ppo import make_trainer,configure_jax,require_backend
    from emu.training_config import (
        PPO_LR,ROLLOUT_STEPS,PPO_ENTROPY_COST,PPO_VALUE_COST)
    from emu.environment import EmuMJXEnv
    from emu.contracts import sensor_observation_contract
    if not 2<=worlds<=4096 or not 80<=horizon<=700 or seed<0:
        raise ValueError("Invalid matrix worlds/horizon/seed")
    trials=assignments(worlds,mode=mode,levels=levels,
                       trials_per_level=trials_per_level)
    policy=policy.expanduser().resolve()
    output=output.expanduser().resolve()
    if not policy.is_file() or policy.suffix!=".emupolicy":
        raise FileNotFoundError("Supply verified exported .emupolicy")
    if output==policy.parent or output in policy.parents or policy in output.parents:
        raise ValueError("Results directory must be independent of actor")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("Output directory must be new or empty")
    params,norm,actor=load_exported_actor_for_exam(policy)
    if (actor["observation_schema"]!=OBS_SCHEMA_TERRAIN or
        actor["training"].get("terrain_carpet_prisms") is not True or
        not actor["array_only"]):
        raise RuntimeError("Matrix exam requires real-prism 360-input frozen actor")
    configure_jax(device)
    import jax
    import jax.numpy as jnp
    require_backend(device=="cpu")
    train=actor["training"]
    sensor=actor["sensor_settings"]
    control=actor["control"]
    vi=control.get("velocity_i") or {}
    def col(name):
        return np.asarray([getattr(x,name) for x in trials],dtype=np.float32)
    env=EmuMJXEnv(
        str(ROBOT_XML),num_envs=worlds,seed=seed,
        max_episode_steps=700,push_max_force_n=.7,
        observation_schema=OBS_SCHEMA_TERRAIN,
        terrain_step_max_m=.08,unified_capability_focus=True,
        terrain_carpet=True,terrain_carpet_prisms=True,
        solver_iterations=train.get("solver_iterations"),
        solver_ls_iterations=train.get("solver_ls_iterations"),
        academy_stream=True,academy_height_cap_m=.15,
        terrain_lesson="coherent-careful",
        attitude_profile=train["attitude_profile"],
        pitch_command_max_deg=float(train["pitch_command_max_deg"]),
        phase_speed_gain=float(control["phase_speed_gain"]),
        science_phase_lowcmd_scale=float(train.get("science_phase_lowcmd_scale",1.)),
        velocity_i_ramp_s=float(vi.get("ramp_s",.4)),
        velocity_i_multiplier=float(vi.get("multiplier",1.)),
        sensor_noise_scale=float(sensor.get("sensor_noise_scale",0.)),
        exam_fixture_steps_m=col("exact_step_m"),
        exam_fixture_grades_deg=col("exact_grade_deg"),
        exam_fixture_kicks_mps=col("initial_kick_mps"),
        **{k:v for k,v in sensor.items() if k!="sensor_noise_scale"})
    expected=sensor_observation_contract(
        sensor,schema=OBS_SCHEMA_TERRAIN,array_only=True)
    if env.sensor_contract!=expected:
        raise RuntimeError("Frozen actor/exam sensor topology differs")
    trainer=make_trainer(env,seed=seed,lr=PPO_LR,
        rollout=ROLLOUT_STEPS,epochs=1,minibatches=1,
        entropy_cost=PPO_ENTROPY_COST,value_cost=PPO_VALUE_COST)
    teacher=dict(
        specialist_mix=0.,athletic_rehearsal_frac=0.,
        regime_frac_steady=1.,regime_frac_transition=0.,
        regime_frac_resilience=0.,
        probe_cmd_vx=0.,probe_cmd_vy=0.,probe_cmd_yaw=0.,
        probe_push_n=0.,probe_tilt_rad=0.,
        probe_linvel_mps=0.,probe_angvel_rad_s=0.,
        probe_joint_rad=0.,terrain_exposure=0.,slope_exposure=0.,
        academy_episode_probability=0.)
    cmd=np.stack((col("vx"),col("vy"),col("yaw")),axis=1)
    print(f"v5053 MATRIX {worlds} real-prism MJX worlds, "
          f"{sum(x.kind!='padding' for x in trials)} assigned, "
          f"{horizon} frames, {mode} — compiling and measuring",flush=True)
    started=time.perf_counter()
    measured=jax.device_get(trainer["evaluate_matrix"](
        params,env.params_array(teacher),
        jnp.asarray(norm["mean"]),jnp.asarray(norm["std"]),
        jnp.asarray(cmd),jax.random.PRNGKey(seed),horizon))
    wall=time.perf_counter()-started
    # HARD acceptance: no matrix may silently label a random ±lip as an
    # exact 2.5mm/5mm/.. step. Measure the physical mesh-backed surfaces.
    step=col("exact_step_m")
    grade=col("exact_grade_deg")
    measured_step=np.asarray(measured["actual_lip_m"])
    measured_grade=np.asarray(measured["actual_grade_deg"])
    if np.max(np.abs(measured_step-step))>.0005:
        raise RuntimeError("Physical exact-step discrepancy exceeds 0.5mm")
    if np.max(np.abs(measured_grade-grade))>.30:
        raise RuntimeError("Physical exact-grade discrepancy exceeds 0.30 degrees")
    detail,curves=reduce_results(trials,measured)
    output.mkdir(parents=True,exist_ok=True)
    save_csv(output/"matrix_worlds.csv",detail)
    save_csv(output/"matrix_curve.csv",curves)
    manifest=dict(
        schema="v5053_true_prism_heterogeneous_matrix_v1",
        actor_sha256=hashlib.sha256(policy.read_bytes()).hexdigest(),
        actor_training_source=actor["source"],
        seed=seed,worlds=worlds,assigned_worlds=len(detail),
        horizon=horizon,mode=mode,levels=levels,
        trials_per_level=trials_per_level if trials_per_level is not None
                           else (100 if mode=="stairs" else 10),
        device=jax.default_backend(),wall_seconds=round(wall,3),
        world_steps_per_second=round(worlds*horizon/max(wall,.001),1),
        collision="8 real convex-triangle prisms, 4 source tiles, exact signed exam-only X seam",
        zero_optimizer_updates=True,
        first_life_only=True,
        completion="actual top contact each foot >=5 frames, then 5 stable destination frames",
        confidence="Wilson 95% conditional on trial sampling, not independent PPO seeds",
        measured_source="actual physical hfield/prism tile surface; actor never sees privileged goal",
        worlds_csv="matrix_worlds.csv",curve_csv="matrix_curve.csv")
    (output/"matrix_summary.json").write_text(
        json.dumps(dict(manifest=manifest,curves=curves),indent=2)+"\n")
    up=[x for x in curves if x["kind"]=="step_up"]
    lines=["# v5053 heterogeneous GPU physical capability matrix","",
           f"Actor SHA256: `{manifest['actor_sha256']}`",
           f"Hardware backend: {manifest['device']}; "
           f"worlds={worlds}, physical frames={horizon}, "
           f"one GPU rollout; wall={wall:.1f}s","",
           "Exact-height **real** prism topology, independent per-world commands.",
           "A leading contact is not a completed ascent. All completion",
           "numerators require verified **both** feet + stable arrival.",
           "Wilson intervals quantify trial sampling only; multiple PPO seeds",
           "would be required to establish training-method improvement.","",
           "## Uphill step completion curve","",
           "| Height | Attempts | Lead top | Each foot >=5 | Stable completed | Rate (95% Wilson) |",
           "|---:|---:|---:|---:|---:|---:|"]
    for x in up:
        lines.append(
            f"| {x['requested_step_mm']:.1f}mm | {x['worlds']} | "
            f"{x['leading_top']} | {x['each_foot_top_five']} | "
            f"{x['completed_destinations']} | "
            f"{x['completion_rate']:.1%} "
            f"({x['completion_wilson95_lo']:.1%}–"
            f"{x['completion_wilson95_hi']:.1%}) |")
    lines+=["","## All groups","",
            "| Type | Level | Tested | Survival | Speed MAE | Goal completions |",
            "|---|---:|---:|---:|---:|---:|"]
    for x in curves:
        lines.append(
            f"| {x['kind']} | {x['level']} | {x['worlds']} | "
            f"{x['survival_rate']:.0%} | "
            f"{x['speed_mae_mps']:.3f}m/s | "
            f"{x['completed_destinations']} |")
    lines+=["","Exact step/grade error is logged per world.",
            "A frozen exam does NOT expand/retreat the learning curriculum.",
            "No claim of a winning policy until physical comparison against",
            "unchanged control on multiple independent seeds."]
    (output/"matrix_summary.md").write_text("\n".join(lines)+"\n")
    print(f"v5053 complete: {len(detail)} independent trials, "
          f"{worlds*horizon/max(wall,.001):,.0f} world-steps/s including "
          f"compile; report {output/'matrix_summary.md'}",flush=True)
    return manifest,curves


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--mode",choices=("balanced","stairs"),default="balanced")
    p.add_argument("--worlds",type=int,default=2048)
    p.add_argument("--horizon",type=int,default=350)
    p.add_argument("--seed",type=int,default=1701)
    p.add_argument("--levels",type=int,default=20)
    p.add_argument("--trials-per-level",type=int,default=None)
    p.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    a=p.parse_args(argv)
    run_matrix(**vars(a))


if __name__=="__main__":
    main()
