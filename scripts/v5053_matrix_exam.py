"""v5053: one GPU matrix of independent physical capability trials.

Read-only frozen .emupolicy evaluation. EXACT signed 1m-seam steps and
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
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

# Fingerprint the actual frozen examiner implementation, not only its actor.
# Two runs with matching policy hashes can still differ in evaluation physics.
# Fixed scientific subject; the released original-H40 actor must be refused.
SCHOOL_DIAGNOSTIC_SOURCE="4048e8450c36882501cbb624f4ca656e11da9419"
SCHOOL_DIAGNOSTIC_MODEL="21b5856b7a248e5d5b988e2735255d57ec79bae1a7cab0bcb72f481bfc36f470"
SCHOOL_DIAGNOSTIC_CASES=(("slow_350",.10,350),("slow_1200",.10,1200),
                         ("fast_350",.25,350),("fast_1200",.25,1200))
# Preserve the SAME >1200 first life in all four cells; no 700-frame timeout.
SCHOOL_DIAGNOSTIC_EPISODE_LIMIT=1201

EXAM_SOURCE_PATHS=(
    "scripts/v5053_matrix_exam.py","emu/environment.py","emu/ppo.py",
    "emu/terrain_carpet_mjx.py","emu/carpet_prism_mjx.py",
    "emu/contracts.py","emu/tof_geometry.py","emu/training_config.py",
    "emu/rewards.py","emu/physical_variants.py","emu/policy_export.py",
    "emu/robot.xml",
)


def exam_source_fingerprint():
    root=Path(__file__).resolve().parent.parent
    h=hashlib.sha256()
    for rel in EXAM_SOURCE_PATHS:
        h.update(rel.encode("utf-8")+b"\0")
        h.update((root/rel).read_bytes())
    return h.hexdigest()


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


def trial_spec(kind: str, level: int, levels: int, replicate: int, world: int,
               *, diagnostic_vx_mps: float|None=None) -> Trial:
    if kind not in (*KINDS,"flat_control") or not 1<=level<=levels:
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
        step=.08*f
    elif kind=="flat_control":
        vx=.25
    elif kind=="step_down":
        vx=.25
        step=-.08*f
    if diagnostic_vx_mps is not None:
        if kind!="step_up" or diagnostic_vx_mps not in (.10,.25):
            raise ValueError("Diagnostic speed requires uphill 0.10 or 0.25 m/s")
        vx=diagnostic_vx_mps
    return Trial(world,kind,level,replicate,vx,vy,yaw,step,grade,kick)


def assignments(worlds: int, *, mode: str="balanced", levels: int=20,
                trials_per_level: int|None=None,
                diagnostic_vx_mps: float|None=None) -> list[Trial]:
    if mode not in ("balanced","stairs","stairs_up","stairs_down","flat") or not 1<=levels<=40 or worlds<2:
        raise ValueError("invalid matrix mode/geometry")
    if diagnostic_vx_mps is not None and mode!="stairs_up":
        raise ValueError("Diagnostic speed applies only to stairs_up")
    if mode=="flat":
        levels=1  # A zero-height seam, exactly matched to the stair commands.
    trials=(10 if mode=="balanced" else 50 if mode=="stairs" else 100)
    if trials_per_level is not None:
        trials=trials_per_level
    # Matched flat-vs-stair evaluation needs all 2,000 *same indexed*
    # reset keys in a 2,048-world JAX batch. Other modes keep their cap.
    cap=2000 if mode=="flat" else 1000
    if not 1<=trials<=cap:
        raise ValueError(f"trials_per_level must be 1..{cap} for {mode}")
    kinds=(KINDS if mode=="balanced" else
           ("flat_control",) if mode=="flat" else
           ("step_up","step_down") if mode=="stairs" else
           ("step_up",) if mode=="stairs_up" else ("step_down",))
    count=len(kinds)*levels*trials
    if count>worlds:
        raise ValueError(
            f"Need at least {count} worlds for {mode}: "
            f"{len(kinds)} kinds x {levels} heights x {trials} attempts")
    result=[]
    for kind in kinds:
        for level in range(1,levels+1):
            for rep in range(trials):
                result.append(trial_spec(
                    kind,level,levels,rep,len(result),
                    diagnostic_vx_mps=diagnostic_vx_mps))
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
        target=kind.startswith("step_") or kind=="flat_control"
        gates=("goal_line_crossed","goal_line_after_both",
               "goal_x_window_seen","goal_x_window_after_both",
               "goal_window_after_both","goal_window_aligned",
               "goal_window_safe_except_slip","goal_window_all_gates",
               "traversed_upright_5frame","traversed_in_lane_5frame",
               "traversed_5frame")
        counts={k:sum(bool(x[k]) for x in records) if target else 0
                for k in gates}
        finished=sum(bool(x["completed_destination"]) for x in records) if target else 0
        arrived=sum(bool(x["both_feet_top_five"]) for x in records) if target else 0
        leading=sum(bool(x["lead_foot_top"]) for x in records) if target else 0
        survived=sum(bool(x["survived"]) for x in records)
        low,high=wilson_95(finished,n)
        traverse_lo,traverse_hi=wilson_95(counts["traversed_5frame"],n)
        first_contact_x=[x["first_both_foot_body_x_m"] for x in records
                         if x["first_both_foot_frame"]>=0]
        first_contact_frames=[x["first_both_foot_frame"] for x in records
                              if x["first_both_foot_frame"]>=0]
        rows.append(dict(
            kind=kind,level=level,worlds=n,
            requested_step_mm=round(records[0]["exact_step_m"]*1000.,4),
            measured_step_mm_mean=round(float(np.mean(
                [x["actual_lip_m"]*1000. for x in records])),4),
            requested_grade_deg=records[0]["exact_grade_deg"],
            measured_grade_deg_mean=round(float(np.mean(
                [x["actual_grade_deg"] for x in records])),4),
            leading_top=leading,each_foot_top_five=arrived,
            **counts,
            first_both_foot_x_mean_m=round(float(np.mean(first_contact_x)),4)
                if first_contact_x else None,
            first_both_foot_frame_mean=round(float(np.mean(first_contact_frames)),1)
                if first_contact_frames else None,
            completed_destinations=finished,
            completion_rate=round(finished/n,5) if target else None,
            completion_wilson95_lo=round(low,5) if target else None,
            completion_wilson95_hi=round(high,5) if target else None,
            traversal_rate=round(counts["traversed_5frame"]/n,5) if target else None,
            traversal_wilson95_lo=round(traverse_lo,5) if target else None,
            traversal_wilson95_hi=round(traverse_hi,5) if target else None,
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
               device:str="cuda",school_diagnostic:bool=False,
               stair_vx_mps:float|None=None):
    from emu.contracts import ROBOT_XML,OBS_SCHEMA_TERRAIN
    from emu.policy_export import load_exported_actor_for_exam
    from emu.ppo import make_trainer,configure_jax,require_backend
    from emu.training_config import (
        PPO_LR,ROLLOUT_STEPS,PPO_ENTROPY_COST,PPO_VALUE_COST)
    from emu.environment import EmuMJXEnv
    from emu.contracts import sensor_observation_contract
    if not 2<=worlds<=4096 or seed<0:
        raise ValueError("Invalid matrix worlds/seed")
    if school_diagnostic:
        if (mode!="stairs_up" or horizon not in (350,1200) or
                stair_vx_mps not in (.10,.25) or worlds>1024):
            raise ValueError("Diagnostic requires uphill 0.10/0.25 m/s, 350/1200 frames and <=1024 worlds")
    elif not 80<=horizon<=700 or stair_vx_mps is not None:
        raise ValueError("Official exam requires 80..700 frames and fixed .25m/s commands")
    if mode=="flat":
        levels=1
    trials=assignments(worlds,mode=mode,levels=levels,
                       trials_per_level=trials_per_level,
                       diagnostic_vx_mps=stair_vx_mps)
    policy=policy.expanduser().resolve()
    output=output.expanduser().resolve()
    if not policy.is_file() or policy.suffix!=".emupolicy":
        raise FileNotFoundError("Supply verified exported .emupolicy")
    if output==policy.parent or output in policy.parents or policy in output.parents:
        raise ValueError("Results directory must be independent of actor")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("Output directory must be new or empty")
    params,norm,actor=load_exported_actor_for_exam(policy)
    if school_diagnostic and (
            actor["source"].get("git_commit")!=SCHOOL_DIAGNOSTIC_SOURCE or
            actor["source"].get("seed")!=736 or
            actor["source"].get("step")!=120061952 or
            actor["robot_model_sha256"]!=SCHOOL_DIAGNOSTIC_MODEL):
        raise ValueError("Require exact frozen seed-736 120M KS150 export")
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
        max_episode_steps=(SCHOOL_DIAGNOSTIC_EPISODE_LIMIT
                           if school_diagnostic else 700),
        push_max_force_n=.7,
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
    score_mask=np.asarray(
        [t.kind in ("step_up","step_down","flat_control") for t in trials],
        dtype=np.bool_)
    print(f"v5053 MATRIX {worlds} real-prism MJX worlds, "
          f"{sum(x.kind!='padding' for x in trials)} assigned, "
          f"{horizon} frames, {mode} — compiling and measuring",flush=True)
    started=time.perf_counter()
    measured=jax.device_get(trainer["evaluate_matrix"](
        params,env.params_array(teacher),
        jnp.asarray(norm["mean"]),jnp.asarray(norm["std"]),
        jnp.asarray(cmd),jnp.asarray(score_mask),
        jax.random.PRNGKey(seed),horizon))
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
        schema="h40_true_prism_heterogeneous_matrix_v5_signed80",
        exam_source_sha256=exam_source_fingerprint(),
        policy_sha256=actor['artifact_sha256'],
        actor_sha256=actor['source']['actor_sha256'],
        robot_model_sha256=actor['robot_model_sha256'],
        actor_training_source=actor["source"],
        seed=seed,worlds=worlds,assigned_worlds=len(detail),
        horizon=horizon,mode=mode,levels=levels,
        trials_per_level=trials_per_level if trials_per_level is not None
                           else (10 if mode=="balanced" else 50 if mode=="stairs" else 100),
        device=jax.default_backend(),wall_seconds=round(wall,3),
        world_steps_per_second=round(worlds*horizon/max(wall,.001),1),
        collision="8 real convex-triangle prisms, 4 source tiles, exact signed exam-only X seam",
        zero_optimizer_updates=True,
        first_life_only=True,
        completion="strict: independent 5-frame top contact per foot, then 5 stable goal frames",
        traversal="independent 5-frame top contact per foot, then 5 upright/aligned frames >=60cm beyond seam",
        confidence="Wilson 95% conditional on trial sampling, not independent PPO seeds",
        measured_source="actual physical hfield/prism tile surface; actor never sees privileged goal",
        worlds_csv="matrix_worlds.csv",curve_csv="matrix_curve.csv")
    if school_diagnostic:
        manifest.update(
            diagnostic_school=True,qualification_eligible=False,
            diagnostic_vx_mps=stair_vx_mps,
            episode_limit_frames=SCHOOL_DIAGNOSTIC_EPISODE_LIMIT,
            diagnostic_scope="one first lip and fixed speed, no Academy governor",
            historical_heldout=False)
    (output/"matrix_summary.json").write_text(
        json.dumps(dict(manifest=manifest,curves=curves),indent=2)+"\n")
    stair_rows=[x for x in curves if x["kind"] in ("step_up","step_down","flat_control")]
    stage_counts={}
    for kind in ("flat_control","step_up","step_down"):
        selected=[r for r in curves if r["kind"]==kind]
        if not selected:
            continue
        gates=("leading_top","each_foot_top_five",
               "traversed_upright_5frame","traversed_in_lane_5frame",
               "traversed_5frame","completed_destinations")
        stage_counts[kind]=dict(
            trials=sum(r["worlds"] for r in selected),
            **{gate:sum(r[gate] for r in selected) for gate in gates})
    def stage_cell(kind,gate):
        data=stage_counts[kind]
        count=data[gate]
        return f"{count}/{data['trials']} ({count/data['trials']:.1%})"
    lines=["# H40 heterogeneous GPU physical capability matrix","",
           f"Actor weights SHA256: `{manifest['actor_sha256']}`",
           f"Exported policy SHA256: `{manifest['policy_sha256']}`",
           f"Verified robot XML SHA256: `{manifest['robot_model_sha256']}`",
           f"Hardware backend: {manifest['device']}; "
           f"worlds={worlds}, physical frames={horizon}, "
           f"one GPU rollout; wall={wall:.1f}s","",
           "Both-feet contact, continuous traversal and strict stable arrival",
           "are three DIFFERENT physical outcomes. Original strict score is",
           "preserved; traversal does not require a stop or low stance slip.",
           "Signed step heights: positive = uphill, negative = downhill.",
           ("DESIGN-TIME KS150 DIAGNOSTIC, not release qualification."
            if school_diagnostic else
            "The flat control is a real zero-height seam at 0.25m/s."),
           f"This run: mode={mode}, {manifest['trials_per_level']} trials per height",
           f"and direction where applicable; {len(detail)} active worlds.",
           "For a truly paired random-reset comparison, use the same 2,048",
           "world batch and seed in BOTH modes (2,000 active flat trials).",
           "A 128-world flat batch does not share the same per-world JAX",
           "random reset keys as a 2,048-world stair batch.","",
           "## Physical outcome funnel (first life, full batch)","",
           "Each row is a separate increasingly strict event gate;",
           "five-frame proof of *each* foot does not imply simultaneous stance.",
           "Crossing requires five frames beyond the lip; lane and heading",
           "requirements are applied to increasingly strict crossing gates.",
           "Strict stable arrival is a different task from continuing gait.","",
           "| Event | " + " | ".join(stage_counts) + " |",
           "|---|" + "|---:|"*len(stage_counts)]
    for gate,label in (
        ("leading_top","First destination foot"),
        ("each_foot_top_five","Each foot supported for 5 frames"),
        ("traversed_upright_5frame","Upright crossing for 5 frames"),
        ("traversed_in_lane_5frame","Crossing plus lane for 5 frames"),
        ("traversed_5frame","Crossing plus lane and heading for 5 frames"),
        ("completed_destinations","Strict stable destination arrival")):
        lines.append("| "+label+" | "+" | ".join(
            stage_cell(kind,gate) for kind in stage_counts)+" |")
    lines += ["","## Upright stair traversal versus strict arrival","",
              "| Height | Trials | Lead top | Both feet >=5 | Cross after feet | Traversed 5f | Strict stable |",
              "|---:|---:|---:|---:|---:|---:|---:|"]
    for x in stair_rows:
        lines.append(
            f"| {x['requested_step_mm']:.1f}mm | {x['worlds']} | "
            f"{x['leading_top']} | {x['each_foot_top_five']} | "
            f"{x['goal_line_after_both']} | {x['traversed_5frame']} "
            f"({x['traversal_rate']:.1%}) | "
            f"{x['completed_destinations']} ({x['completion_rate']:.1%}) |")
    lines+=["","## Why didn't the robot traverse or arrive?","",
            "| Height | Cross (after feet) | 5f upright crossing | 5f also within Y lane | 5f also aligned | X window after feet | X+Y after feet | Mean X at both-feet proof |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for x in stair_rows:
        first=(f"{x['first_both_foot_x_mean_m']:.3f}m"
               if x["first_both_foot_x_mean_m"] is not None else "—")
        lines.append(
            f"| {x['requested_step_mm']:.1f}mm | {x['goal_line_after_both']} | "
            f"{x['traversed_upright_5frame']} | "
            f"{x['traversed_in_lane_5frame']} | "
            f"{x['traversed_5frame']} | "
            f"{x['goal_x_window_after_both']} | "
            f"{x['goal_window_after_both']} | {first} |")
    lines+=["","The X-only goal window separates **trailing-foot timing**",
            "from lateral drift. The 5-frame crossing gates separate",
            "upright physical crossing from lane and heading control.",
            "Each gate is counted per world in its first life.","",
            "## Ordered strict-arrival gates (ever reached within first life)","",
            "| Height | Goal line | After both feet | Inside goal window | Heading | Upright/support/contact | Also low slip | Five stable frames |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for x in stair_rows:
        lines.append(
            f"| {x['requested_step_mm']:.1f}mm | {x['goal_line_crossed']} | "
            f"{x['goal_line_after_both']} | {x['goal_window_after_both']} | "
            f"{x['goal_window_aligned']} | "
            f"{x['goal_window_safe_except_slip']} | "
            f"{x['goal_window_all_gates']} | "
            f"{x['completed_destinations']} |")
    lines+=["","The gates are cumulative AND same-frame, not independent",
            "flags mixed from different moments. Five-frame strict arrival",
            "can still fail when all gates hold momentarily.",
            "The downstream traversal needs five consecutive upright/",
            "aligned, clean-contact frames after crossing, but does NOT",
            "require stopping inside the strict X window or low slip.","",
            "## All groups","",
            "| Type | Level | Tested | Survival | Speed MAE | Traversals | Stable arrivals |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for x in curves:
        lines.append(
            f"| {x['kind']} | {x['level']} | {x['worlds']} | "
            f"{x['survival_rate']:.0%} | "
            f"{x['speed_mae_mps']:.3f}m/s | "
            f"{x['traversed_5frame']} | {x['completed_destinations']} |")
    lines+=["","Full per-trial gates and exact geometry errors: `matrix_worlds.csv`.",
            "No neural weight or curriculum updates; confidence intervals",
            "reflect repeated trials, not PPO training-seed uncertainty."]
    (output/"matrix_summary.md").write_text("\n".join(lines)+"\n")
    print(f"v5053 complete: {len(detail)} independent trials, "
          f"{worlds*horizon/max(wall,.001):,.0f} world-steps/s including "
          f"compile; report {output/'matrix_summary.md'}",flush=True)
    return manifest,curves




# --- Frozen same-actor speed x duration design-time diagnostic ------------
SCHOOL_DIAGNOSTIC_GATES=(
    "lead_foot_top","both_feet_top_five","traversed_5frame","survived")


def _diagnostic_gate(value) -> int:
    token=str(value).strip().lower()
    if token not in ("true","false","1","0","1.0","0.0"):
        raise ValueError(f"Unrecognised physical gate {value!r}")
    return int(token in ("true","1","1.0"))


def compare_school_diagnostic_cases(cases:dict) -> dict:
    """Compare four cases with identical actor, fixtures and reset keys.

    The four independently JIT-compiled runs share actor, world-indexed
    random seeds and fixtures, NOT an observed byte-identical trajectory
    prefix. Floating-point physics can diverge between executable variants.
    A shorter-window gate disappearing in a separately compiled long run
    is an important discrepancy to report, not a reason to discard all
    four expensive measurements. Actor/model/fixture mismatches still fail.
    """
    if set(cases)!={label for label,_,_ in SCHOOL_DIAGNOSTIC_CASES}:
        raise ValueError("Require exactly four school diagnostic conditions")
    reference=cases["slow_350"][0]["manifest"]
    same=("schema","policy_sha256","actor_sha256","robot_model_sha256",
          "actor_training_source","exam_source_sha256","seed","worlds",
          "levels","trials_per_level","mode","device","collision",
          "first_life_only","zero_optimizer_updates",
          "episode_limit_frames","assigned_worlds")
    by_case={}
    for label,vx,horizon in SCHOOL_DIAGNOSTIC_CASES:
        summary,rows=cases[label]
        m=summary["manifest"]
        if (not m.get("diagnostic_school") or
                m.get("qualification_eligible") is not False or
                m.get("diagnostic_vx_mps")!=vx or m.get("horizon")!=horizon or
                m.get("episode_limit_frames")!=SCHOOL_DIAGNOSTIC_EPISODE_LIMIT):
            raise ValueError(f"{label}: invalid school diagnostic case")
        if (m.get("actor_training_source",{}).get("git_commit")!=
                SCHOOL_DIAGNOSTIC_SOURCE or
                m.get("robot_model_sha256")!=SCHOOL_DIAGNOSTIC_MODEL):
            raise ValueError(f"{label}: wrong frozen KS150 source/model")
        if any(m.get(k)!=reference.get(k) for k in same):
            raise ValueError(f"{label}: unpaired source/actor/fixture")
        indexed={int(r["world"]):r for r in rows}
        if (len(indexed)!=len(rows) or len(rows)!=m["assigned_worlds"] or
                set(indexed)!=set(range(len(rows)))):
            raise ValueError(f"{label}: missing, duplicated or misindexed worlds")
        if any(r["kind"]!="step_up" or abs(float(r["vx"])-vx)>1e-7
               for r in rows):
            raise ValueError(f"{label}: wrong command/stair direction")
        by_case[label]=indexed
    prefix_disagreements=[]
    for world in by_case["slow_350"]:
        ref=by_case["slow_350"][world]
        for label,_,_ in SCHOOL_DIAGNOSTIC_CASES:
            row=by_case[label][world]
            for k in ("level","replicate","exact_step_m","exact_grade_deg",
                      "initial_kick_mps","vy","yaw"):
                if not np.isclose(float(row[k]),float(ref[k]),rtol=0.,atol=1e-7):
                    raise ValueError(f"World {world}: unpaired {k}")
            if (abs(float(row["actual_lip_m"])-float(row["exact_step_m"]))>.0005
                    or abs(float(row["actual_lip_m"])-
                           float(ref["actual_lip_m"]))>.0005):
                raise ValueError(f"World {world}: physical lip differs")
        for speed in ("slow","fast"):
            short=by_case[f"{speed}_350"][world]
            long=by_case[f"{speed}_1200"][world]
            for gate in SCHOOL_DIAGNOSTIC_GATES[:-1]:
                if _diagnostic_gate(short[gate]) and not _diagnostic_gate(long[gate]):
                    prefix_disagreements.append(dict(
                        world=world, speed=speed, gate=gate,
                        short_350=1, long_1200=0))
            # Within ONE identical first life, survival at 1200 implies
            # survival at 350. The experiments are separately compiled,
            # however, so record an apparent reversal rather than falsely
            # diagnosing second-life contamination in either individual run.
            if _diagnostic_gate(long["survived"]) and not _diagnostic_gate(short["survived"]):
                prefix_disagreements.append(dict(
                    world=world,speed=speed,gate="survived",
                    short_350=0,long_1200=1))
    levels=sorted({int(r["level"]) for r in by_case["slow_350"].values()})
    curves=[]
    for level in levels:
        worlds=sorted(w for w,r in by_case["slow_350"].items()
                      if int(r["level"])==level)
        stage=dict(level=level,trials=len(worlds),
                   step_mm=round(1000.*float(
                       by_case["slow_350"][worlds[0]]["exact_step_m"]),4))
        for label,_,_ in SCHOOL_DIAGNOSTIC_CASES:
            for gate in SCHOOL_DIAGNOSTIC_GATES:
                stage[f"{label}_{gate}"]=sum(
                    _diagnostic_gate(by_case[label][w][gate]) for w in worlds)
        curves.append(stage)
    totals=dict(worlds_per_condition=len(by_case["slow_350"]))
    for label,_,_ in SCHOOL_DIAGNOSTIC_CASES:
        for gate in SCHOOL_DIAGNOSTIC_GATES:
            totals[f"{label}_{gate}"]=sum(
                _diagnostic_gate(r[gate]) for r in by_case[label].values())
    disagreement_counts={
        speed:{
            gate:sum(item["speed"]==speed and item["gate"]==gate
                     for item in prefix_disagreements)
            for gate in SCHOOL_DIAGNOSTIC_GATES
        } for speed in ("slow","fast")
    }
    return dict(schema="emu_school_speed_horizon_diagnostic_v1",
                qualification_eligible=False,same_initial_worlds=True,
                prefix_event_gates_consistent=(not prefix_disagreements),
                same_trajectory_prefix_verified=False,
                independently_compiled_executables=True,
                horizon_prefix_disagreement_count=len(prefix_disagreements),
                horizon_prefix_disagreements=prefix_disagreements,
                horizon_prefix_disagreement_counts=disagreement_counts,
                actor_sha256=reference["actor_sha256"],
                model_sha256=reference["robot_model_sha256"],
                source_sha=reference["actor_training_source"]["git_commit"],
                seed=reference["seed"],
                episode_limit_frames=SCHOOL_DIAGNOSTIC_EPISODE_LIMIT,
                warning=("Design-time first lip only; no Academy, PPO update "
                         "or independent qualification. Different horizons "
                         "were separately JIT-compiled; shared seeds do NOT "
                         "prove identical trajectories. If horizon prefix "
                         "disagreements are present, duration differences "
                         "cannot be attributed to elapsed time alone."),
                totals=totals,curves=curves)


def run_school_diagnostic_suite(*,policy:Path,output:Path,
                                worlds:int=256,seed:int=5701,
                                levels:int=20,trials_per_level:int=10,
                                device:str="cuda") -> dict:
    """Launch four separate GPU processes, never retraining or overwriting."""
    policy,output=policy.expanduser().resolve(),output.expanduser().resolve()
    if (not policy.is_file() or policy.suffix!=".emupolicy" or
            output.exists() or output==policy.parent or
            output in policy.parents or policy in output.parents):
        raise ValueError("Frozen policy must exist; output must be a NEW separate directory")
    if (levels!=20 or not 1<=trials_per_level<=50 or
            not levels*trials_per_level<=worlds<=1024 or seed<0):
        raise ValueError("Diagnostic needs 20 stair heights, <=1024 worlds")
    actor=json.loads(policy.read_text(encoding="utf-8"))
    if (actor.get("source",{}).get("git_commit")!=SCHOOL_DIAGNOSTIC_SOURCE or
            actor.get("source",{}).get("seed")!=736 or
            actor.get("source",{}).get("step")!=120061952 or
            actor.get("source",{}).get("robot_model_sha256")!=SCHOOL_DIAGNOSTIC_MODEL):
        raise ValueError("Refuse anything except frozen 120M seed-736 KS150 actor")
    output.mkdir(parents=True)
    env=os.environ.copy()
    env.update(EMU_PHYSICAL_PROFILE="h40-spring-only",
               XLA_PYTHON_CLIENT_PREALLOCATE="false")
    for label,vx,horizon in SCHOOL_DIAGNOSTIC_CASES:
        cmd=[sys.executable,str(Path(__file__).resolve()),
             "--school-diagnostic-case",
             "--policy",str(policy),"--output",str(output/label),
             "--mode","stairs_up","--worlds",str(worlds),
             "--levels",str(levels),"--trials-per-level",str(trials_per_level),
             "--horizon",str(horizon),"--stair-vx-mps",str(vx),
             "--seed",str(seed),"--device",device]
        print(f"Frozen school first-lip diagnostic: {label}",flush=True)
        subprocess.run(cmd,env=env,check=True)
    return summarize_existing_school_diagnostic(output)


def summarize_existing_school_diagnostic(output:Path) -> dict:
    """Aggregate completed GPU case files only: no JAX, policy or GPU run.

    Retain raw per-world CSVs and all four examiner manifests unchanged.
    Write new derived summaries only; fail rather than overwrite a prior
    analysis. A different analysis must use a new directory or be reviewed.
    """
    output=Path(output).expanduser().resolve()
    if not output.is_dir():
        raise FileNotFoundError("School diagnostic output directory is absent")
    products=tuple(output/name for name in (
        "school_diagnostic.json","school_diagnostic_heights.csv",
        "school_diagnostic_prefix_disagreements.csv","school_diagnostic.md"))
    existing=[p for p in products if p.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite prior school diagnostic summary: "+
            ", ".join(str(p) for p in existing))
    cases={}
    for label,_,_ in SCHOOL_DIAGNOSTIC_CASES:
        directory=output/label
        summary=json.loads((directory/"matrix_summary.json").read_text())
        with (directory/"matrix_worlds.csv").open(newline="",encoding="utf-8") as fd:
            cases[label]=(summary,list(csv.DictReader(fd)))
    report=compare_school_diagnostic_cases(cases)
    # All fail-closed identity and physical fixture checks happen before
    # touching any combined output. Original case files are never rewritten.
    (output/"school_diagnostic.json").write_text(
        json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    save_csv(output/"school_diagnostic_heights.csv",report["curves"])
    with (output/"school_diagnostic_prefix_disagreements.csv").open(
            "w",newline="",encoding="utf-8") as fd:
        fields=("world","speed","gate","short_350","long_1200")
        writer=csv.DictWriter(fd,fieldnames=fields)
        writer.writeheader()
        writer.writerows(report["horizon_prefix_disagreements"])
    totals=report["totals"]
    lines=["# KS150: frozen first-lip speed x horizon diagnostic",
           "","**DESIGN-TIME ONLY — NOT RELEASE QUALIFICATION**","",
           f"Actor SHA256: {report['actor_sha256']}",
           f"Source commit: {report['source_sha']}",
           f"KS150 model SHA256: {report['model_sha256']}",
           f"Same indexed initial-world seeds, fixtures and actor.",
           "**Different horizon cases are independently JIT-compiled rollouts.**",
           "",
           "| Speed / frames | Trials | First top | Both feet (5f) | Traversed (5f) | First-life survived |",
           "|---|---:|---:|---:|---:|---:"+"|"]
    for label,_,_ in SCHOOL_DIAGNOSTIC_CASES:
        lines.append(f"| {label} | {totals['worlds_per_condition']} | "
                     f"{totals[label+'_lead_foot_top']} | "
                     f"{totals[label+'_both_feet_top_five']} | "
                     f"{totals[label+'_traversed_5frame']} | "
                     f"{totals[label+'_survived']} |")
    counts=report["horizon_prefix_disagreement_counts"]
    lines+=["",
            "## Horizon-prefix disagreement audit",
            "",
            f"Total discordant same-world gates: {report['horizon_prefix_disagreement_count']}.",
            "",
            "| Speed | First-top lost | Two-foot lost | Traversal lost | Survival reversed |",
            "|---|---:|---:|---:|---:|"]
    for speed in ("slow","fast"):
        c=counts[speed]
        lines.append(f"| {speed} | {c['lead_foot_top']} | "
                     f"{c['both_feet_top_five']} | {c['traversed_5frame']} "
                     f"| {c['survived']} |")
    lines+=["",
            "This is a discrepancy in separate rollouts, NOT proof that a "
            "previously achieved event was physically forgotten. Numerically "
            "divergent prefixes can confound causal interpretation of horizon.",
            "Inspect school_diagnostic_prefix_disagreements.csv before "
            "changing the Academy; a strict same-trajectory horizon study "
            "requires 350-frame snapshots INSIDE one 1200-frame rollout.",
            "",
            "Every condition uses a 1201-frame episode limit; the old "
            "700-frame timeout cannot cause a false long-horizon failure.",
            "This is a fixed-velocity first-lip test, NOT the complete Academy.",
            "Per-height CSV and four original per-world CSV files are preserved."]
    (output/"school_diagnostic.md").write_text(
        "\n".join(lines)+"\n",encoding="utf-8")
    print(f"Offline school diagnostic summary: {output/'school_diagnostic.md'}")
    if report["horizon_prefix_disagreement_count"]:
        print(f"CAUTION: {report['horizon_prefix_disagreement_count']} "
              "cross-horizon event discrepancies; see the CSV and report")
    return report




PAIRED_GATES=("lead_foot_top","both_feet_top_five",
              "traversed_upright_5frame","traversed_in_lane_5frame",
              "traversed_5frame","completed_destination")


def compare_paired_exams(stair_summary,stair_rows,flat_summary,flat_rows):
    """Compare read-only, same-index first-life worlds. Fail closed on mismatch."""
    a,b=stair_summary["manifest"],flat_summary["manifest"]
    if a.get("mode")!="stairs" or b.get("mode")!="flat":
        raise ValueError("Need stairs and flat modes for matched comparison")
    for key in ("schema","policy_sha256","actor_sha256","robot_model_sha256",
                "actor_training_source","exam_source_sha256",
                "seed","worlds","horizon","device",
                "collision","first_life_only","zero_optimizer_updates"):
        if a.get(key)!=b.get(key):
            raise ValueError(f"Unmatched frozen exam: {key}")
    if (not a.get("exam_source_sha256") or
            a["schema"]!="h40_true_prism_heterogeneous_matrix_v5_signed80" or
            not a["first_life_only"] or not a["zero_optimizer_updates"]):
        raise ValueError("Unsupported or trainable exam")
    flat={int(r["world"]):r for r in flat_rows}
    stair_worlds={int(r["world"]) for r in stair_rows}
    if (len(flat)!=len(flat_rows) or len(stair_worlds)!=len(stair_rows)
            or stair_worlds!=set(flat) or stair_worlds!=set(range(len(flat)))
            or len(flat)!=a["assigned_worlds"] or len(flat)!=b["assigned_worlds"]):
        raise ValueError("Paired worlds are missing, duplicated or misindexed")

    def gate(value):
        v=str(value).strip().lower()
        if v not in ("true","false","1","0","1.0","0.0"):
            raise ValueError(f"Unrecognised physical contact gate {value}")
        return int(v in ("true","1","1.0"))

    pairs=[]
    grouped={}
    for s in stair_rows:
        world=int(s["world"])
        f=flat[world]
        if s["kind"] not in ("step_up","step_down") or f["kind"]!="flat_control":
            raise ValueError(f"Wrong challenge type at world {world}")
        for key in ("vx","vy","yaw","initial_kick_mps","exact_grade_deg"):
            if not np.isclose(float(s[key]),float(f[key]),atol=1e-7):
                raise ValueError(f"World {world}: unmatched {key}")
        if (abs(float(f["exact_step_m"]))>1e-7 or
                abs(float(f["actual_lip_m"]))>.0005 or
                abs(float(s["actual_lip_m"])-float(s["exact_step_m"]))>.0005):
            raise ValueError(f"World {world}: mismatched physical seam")
        row=dict(world=world,kind=s["kind"],level=int(s["level"]),
                 signed_step_mm=round(1000.*float(s["exact_step_m"]),4))
        for key in PAIRED_GATES:
            stair_ok,flat_ok=gate(s[key]),gate(f[key])
            row.update({f"{key}_{tag}":value for tag,value in (
                ("stair",stair_ok),("flat",flat_ok),
                ("lost",int(flat_ok and not stair_ok)),
                ("gained",int(stair_ok and not flat_ok)))})
        row["speed_mae_excess_mps"]=round(
            float(s["speed_mae_mps"])-float(f["speed_mae_mps"]),5)
        pairs.append(row)
        grouped.setdefault((row["kind"],row["level"]),[]).append(row)
    curves=[]
    for (kind,level),rows in sorted(grouped.items()):
        n=len(rows)
        result=dict(kind=kind,level=level,pairs=n,
                    signed_step_mm=rows[0]["signed_step_mm"])
        for key in PAIRED_GATES:
            for tag in ("stair","flat","lost","gained"):
                result[f"{key}_{tag}"]=sum(r[f"{key}_{tag}"] for r in rows)
            result[f"{key}_difference_pct"]=round(100.*(
                result[f"{key}_stair"]-result[f"{key}_flat"])/n,1)
        result["speed_mae_excess_mps"]=round(float(np.mean(
            [r["speed_mae_excess_mps"] for r in rows])),5)
        curves.append(result)
    return pairs,curves


def write_paired_comparison(stairs:Path,flat:Path,output:Path):
    stairs,flat,output=(p.expanduser().resolve() for p in (stairs,flat,output))
    if (stairs==flat or output in (stairs,flat) or
            output in stairs.parents or output in flat.parents or
            stairs in output.parents or flat in output.parents):
        raise ValueError("Input/output directories must be separate")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("Comparison output must be new or empty")
    def read(directory):
        summary=json.loads((directory/"matrix_summary.json").read_text())
        with (directory/"matrix_worlds.csv").open(newline="",encoding="utf-8") as fd:
            return summary,list(csv.DictReader(fd))
    pairs,curves=compare_paired_exams(*read(stairs),*read(flat))
    output.mkdir(parents=True,exist_ok=True)
    save_csv(output/"matrix_pairs.csv",pairs)
    save_csv(output/"matrix_paired_curves.csv",curves)
    (output/"matrix_compare.json").write_text(
        json.dumps(dict(stairs=str(stairs),flat=str(flat),curves=curves),
                   indent=2)+"\n")
    lines=["# H40 same-world signed stair vs flat physical comparison","",
           "Same actor, seed, world index, command, robot and exam geometry.",
           "Lost = passed flat, failed stair. Gained = failed flat, passed stair.",
           "Differences are stair minus flat, not changes from PPO training.","",
           "| Lip | Paired | Flat traverse | Stair traverse | Lost | Gained | Difference |",
           "|---:|---:|---:|---:|---:|---:|---:|"]
    for r in curves:
        lines.append(
            f"| {r['signed_step_mm']:+.1f}mm | {r['pairs']} | "
            f"{r['traversed_5frame_flat']} | {r['traversed_5frame_stair']} | "
            f"{r['traversed_5frame_lost']} | {r['traversed_5frame_gained']} | "
            f"{r['traversed_5frame_difference_pct']:+.1f}pp |")
    lines+=["","All six first-life physical gates and per-world speed MAE",
            "differences are in matrix_pairs.csv. Independently verify both",
            "checkout source revisions before claiming exact solver parity."]
    (output/"matrix_compare.md").write_text("\n".join(lines)+"\n")
    return pairs,curves


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy",type=Path)
    p.add_argument("--compare-stairs",type=Path)
    p.add_argument("--compare-flat",type=Path)
    p.add_argument("--school-diagnostic-suite",action="store_true",
                   help="four frozen KS150 first-lip speed/horizon exams, NOT qualification")
    p.add_argument("--school-diagnostic-summarize",action="store_true",
                   help="aggregate four existing cases on CPU without rerunning GPU physics")
    p.add_argument("--school-diagnostic-case",action="store_true",
                   help=argparse.SUPPRESS)
    p.add_argument("--stair-vx-mps",type=float,default=None,
                   help=argparse.SUPPRESS)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--mode",choices=("balanced","stairs","stairs_up","stairs_down","flat"),default="balanced")
    p.add_argument("--worlds",type=int,default=None)
    p.add_argument("--horizon",type=int,default=350)
    p.add_argument("--seed",type=int,default=1701)
    p.add_argument("--levels",type=int,default=20)
    p.add_argument("--trials-per-level",type=int,default=None)
    p.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    a=p.parse_args(argv)
    if a.school_diagnostic_summarize:
        if (a.policy is not None or a.school_diagnostic_suite or
                a.school_diagnostic_case or a.stair_vx_mps is not None or
                a.compare_stairs is not None or a.compare_flat is not None):
            p.error("Offline summarize requires only --output CASES_DIRECTORY")
        summarize_existing_school_diagnostic(a.output)
        return
    if a.school_diagnostic_suite:
        if (a.policy is None or a.compare_stairs is not None or
                a.compare_flat is not None or a.school_diagnostic_case or
                a.stair_vx_mps is not None):
            p.error("School diagnostic requires one frozen policy and a new output directory")
        return run_school_diagnostic_suite(
            policy=a.policy,output=a.output,
            worlds=(256 if a.worlds is None else a.worlds),
            levels=a.levels,trials_per_level=(
                10 if a.trials_per_level is None else a.trials_per_level),
            seed=a.seed,device=a.device)
    if a.compare_stairs is not None or a.compare_flat is not None:
        if a.compare_stairs is None or a.compare_flat is None or a.policy is not None:
            p.error("provide both --compare-stairs and --compare-flat, without --policy")
        pairs,curves=write_paired_comparison(
            a.compare_stairs,a.compare_flat,a.output)
        print(f"Paired {len(pairs)} frozen worlds across {len(curves)} exact lips")
    else:
        if a.policy is None:
            p.error("frozen actor evaluation requires --policy")
        kw=vars(a)
        kw.pop("compare_stairs")
        kw.pop("compare_flat")
        kw.pop("school_diagnostic_suite")
        kw.pop("school_diagnostic_summarize")
        kw["school_diagnostic"]=kw.pop("school_diagnostic_case")
        kw["worlds"]=2048 if kw["worlds"] is None else kw["worlds"]
        run_matrix(**kw)


if __name__=="__main__":
    main()
