"""v5022 *experimental* matched front-ToF PPO campaign (not production trainer).

Reuses the ONE EmuMJXEnv / PPO / RegimeGovernor implementations. Both arms
have physically identical worlds, curriculum, seeds, and 364-input networks.
The blinded arm receives zeros/invalidity in the front 33 channels.

Run: python -m emu.v5022_experiment --run --root ~/robot_lab/emu_runs/v5022
Smoke: python -m emu.v5022_experiment --smoke --device cpu --envs 4
Nothing runs until --run or --smoke is explicitly given.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import subprocess
import tempfile
import time

import numpy as np

from .contracts import (ROBOT_XML, CONTROL_DT, OBS_SCHEMA_TERRAIN, TERRAIN_OBS_DIM,
                        terrain_observation_contract)
from .environment import EmuMJXEnv
from .curriculum import RegimeGovernor, TrainingHealth, CapabilityField
from .ppo import configure_jax, require_backend, make_trainer, update_running_stats
from .results import summarize_rollout
from .training_config import (ROLLOUT_STEPS, PPO_EPOCHS, PPO_MINIBATCHES,
                              PPO_LR, PPO_ENTROPY_COST, PPO_VALUE_COST,
                              PPO_LR_SCALE_INIT, PPO_LR_SCALE_MIN, PPO_LR_SCALE_MAX,
                              PPO_LR_ADJUST, PPO_TARGET_KL, foundation_config)

FORMAT = "v5022_terrain_ppo_v2"
EVAL_NAMES = ("survival","first_fall_step","vx_mae","vy_mae","yaw_mae",
              "gravity_z","foot_mid_cone","step_evidence","support",
              "pitch_mae","height_shortfall_m")


def source_info() -> dict:
    def git(*argv):
        p=subprocess.run(("git",*argv),capture_output=True,text=True)
        return p.stdout.strip() if p.returncode==0 else ""
    return {"commit":git("rev-parse","HEAD"),
            "branch":git("branch","--show-current"),
            "dirty":bool(git("status","--porcelain")),
            "robot_xml_sha256":hashlib.sha256(ROBOT_XML.read_bytes()).hexdigest()}


def atomic_pickle(target: Path, value):
    target.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=".v5022_",suffix=".tmp",dir=target.parent)
    try:
        with os.fdopen(fd,"wb") as stream:
            pickle.dump(value,stream,protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp,target)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def run_arm(*, arm: str, root: Path | None, envs: int, steps: int,
            seed: int, smoke: bool, device: str) -> dict:
    import jax
    import jax.numpy as jnp
    visible = arm == "visible"
    contract = terrain_observation_contract(front_visible=visible)
    source = source_info()
    # MAIN owns the cold-start recipe. A/B arms differ in front visibility,
    # NOT an invented set of vx/vy/yaw/phase/push teaching values.
    foundation=foundation_config(run_dir=(root or Path.cwd()), seed=seed,
                                 device=device, envs=envs)
    foundation_fields=('difficulty_base_step','regime_command_frac',
        'fall_target','open_frontier','vx_hi_max','vx_lo_min','vy_max_mps',
        'yaw_max_rad_s','phase_speed_gain','push_max_force_n',
        'ppo_lr','ppo_entropy_cost','ppo_value_cost')
    main_recipe={k:getattr(foundation,k) for k in foundation_fields}
    if not smoke:
        if source["branch"]!="feature/v5022-forward-terrain" or source["dirty"]:
            raise RuntimeError("Refusing unreviewed training: use the clean v5022 feature branch")
        assert root is not None
        target=root / arm
        target.mkdir(parents=True,exist_ok=True)
        ckpt_path=target/"checkpoint.pkl"
        output_path=target/"metrics.csv"
        manifest_path=target/"experiment.json"
        identity=dict(format=FORMAT,contract=contract,seed=seed,
                      envs=envs,source=source,rollout=ROLLOUT_STEPS,
                      main_foundation_recipe=main_recipe)
        if manifest_path.exists():
            old=json.loads(manifest_path.read_text())
            if old!=identity:
                raise ValueError("Run identity/source changed; use a new directory")
        else:
            # Do not overwrite an existing checkpoint lacking a matching
            # manifest. An explicit new directory makes lineage unambiguous.
            if ckpt_path.exists() or output_path.exists():
                raise FileExistsError(f"Existing unverified v5022 files in {target}")
            manifest_path.write_text(json.dumps(identity,indent=2,sort_keys=True))
    else:
        target=ckpt_path=output_path=None
        identity={}

    cfg=dict(observation_schema=OBS_SCHEMA_TERRAIN,front_visible=visible,
             # Use the established production v5015 posture model.
             attitude_profile="v5015",
             pitch_command_max_deg=foundation.pitch_command_max_deg,
             domain_config=None,
             vx_hi_max=foundation.vx_hi_max,vx_lo_min=foundation.vx_lo_min,
             vy_max_mps=foundation.vy_max_mps,yaw_max_rad_s=foundation.yaw_max_rad_s,
             phase_speed_gain=foundation.phase_speed_gain,
             push_max_force_n=foundation.push_max_force_n,
             open_frontier=foundation.open_frontier,
             sensor_noise_scale=1.)
    env=EmuMJXEnv(str(ROBOT_XML),num_envs=envs,seed=seed,
                  max_episode_steps=700,**cfg)
    if env.obs_dim!=TERRAIN_OBS_DIM or env.sensor_contract!=contract:
        raise RuntimeError("New 364-input sensor/physics contract drift")
    # A CUDA smoke compiles and updates the FULL training-shaped PPO batch;
    # a CPU smoke is intentionally tiny and checks only correctness.
    full_size=not smoke or device=="cuda"
    rollout=ROLLOUT_STEPS if full_size else 2
    trainer=make_trainer(env,seed=seed,lr=PPO_LR,
                         rollout=rollout,
                         epochs=PPO_EPOCHS if full_size else 1,
                         minibatches=PPO_MINIBATCHES if full_size else 1,
                         entropy_cost=PPO_ENTROPY_COST,value_cost=PPO_VALUE_COST)
    governor=RegimeGovernor(
        fall_target=foundation.fall_target,
        difficulty_base_step=foundation.difficulty_base_step,
        regime_command_frac=foundation.regime_command_frac,
        athletic_rehearsal_frac=foundation.athletic_rehearsal_frac,
        open_frontier=foundation.open_frontier)
    for suffix,limit in (
        ('vx_hi',foundation.vx_hi_max),('vx_lo',foundation.vx_lo_min),
        ('vy',foundation.vy_max_mps),('yaw',foundation.yaw_max_rad_s)):
        setattr(governor.params,'frontier_'+suffix,float(limit))
    health=TrainingHealth()
    # Main foundation runs capability feedback in SHADOW: evidence is
    # recorded continuously, but it never steers a cold-start lesson.
    field=CapabilityField()
    lr_scale=float(PPO_LR_SCALE_INIT)
    obs_mean=jnp.zeros((TERRAIN_OBS_DIM,),dtype=jnp.float32)
    obs_var=jnp.ones((TERRAIN_OBS_DIM,),dtype=jnp.float32)
    obs_count=1e-4
    state=trainer["make_initial_state"](governor.params_dict())
    step=0
    if ckpt_path is not None and ckpt_path.is_file():
        with ckpt_path.open("rb") as stream:
            loaded=pickle.load(stream)
        if loaded.get("identity") != identity or loaded.get("format")!=FORMAT:
            raise ValueError("Checkpoint identity doesn't match this physical actor")
        if loaded["step"]%int(envs*ROLLOUT_STEPS)!=0:
            raise ValueError("Checkpoint off PPO rollout boundary")
        step=int(loaded["step"])
        state=state._replace(params=jax.device_put(loaded["params"]),
                             opt_state=jax.device_put(loaded["opt_state"]),
                             key=jax.device_put(loaded["key"]))
        obs_mean=jax.device_put(loaded["obs_mean"])
        obs_var=jax.device_put(loaded["obs_var"])
        obs_count=float(loaded["obs_count"])
        governor.load_state_dict(loaded["governor"])
        health.load_state_dict(loaded["health"])
        field=CapabilityField.from_state_dict(loaded["field"])
        lr_scale=float(loaded["lr_scale"])
        print(f"[v5022 {arm}] resumed {step:,} steps; fresh episodes, intact PPO/optimizer/normalizer/curriculum",flush=True)

    nstep=envs*rollout
    target_step=step+nstep if smoke else steps
    last_save=step
    records=[]
    started=time.perf_counter()
    print(f"[v5022 {arm}] physics=real-step-and-crest actor={TERRAIN_OBS_DIM} "
          f"front={visible} envs={envs} target={target_step:,} "
          f"jax={jax.default_backend()}",flush=True)

    def save():
        if ckpt_path is None:return
        payload=dict(format=FORMAT,identity=identity,step=step,
                     params=jax.device_get(state.params),
                     opt_state=jax.device_get(state.opt_state),
                     key=jax.device_get(state.key),
                     obs_mean=np.asarray(obs_mean),obs_var=np.asarray(obs_var),
                     obs_count=obs_count,governor=governor.state_dict(),
                     health=health.state_dict(),lr_scale=lr_scale,
                     field=field.state_dict())
        atomic_pickle(ckpt_path,payload)

    try:
        while step < target_step:
            current=governor.params_dict()
            params=env.params_array(current)
            norm_std=jnp.sqrt(jnp.maximum(obs_var,1e-6))
            state,metrics,aux,raw,_=trainer["train_update"](
                state,params,obs_mean,norm_std,jnp.asarray(lr_scale,jnp.float32))
            obs_mean,obs_var,obs_count=update_running_stats(
                obs_mean,obs_var,obs_count,raw)
            step+=nstep
            # Host transfer is *after* the compiled PPO update.
            m=jax.device_get(metrics)
            aux=jax.device_get(aux)
            row=summarize_rollout(m,current)
            field_diag=field.update_from_rollout(
                m,step=step,control_dt=CONTROL_DT,
                exclude_field_taught=not foundation.field_evidence_from_taught)
            row.update(field_diag)
            row['ppo_kl_stopped_frac']=float(aux.get('kl_stopped',0.))
            ppo_health=health.observe_ppo(row)
            report=governor.update(
                row,kl_ok_to_expand=bool(ppo_health['kl_ok_to_expand']))
            collapse=health.observe_collapse(
                float(report['fall_ema_posture']))
            # Preserve MAIN's conservative adaptive-KL LR controller.
            lr_scale=health.adjust_learning_rate(
                lr_scale,approx_kl=float(aux.get('approx_kl_final',0.)),
                target_kl=PPO_TARGET_KL,factor=PPO_LR_ADJUST,
                min_scale=PPO_LR_SCALE_MIN,max_scale=PPO_LR_SCALE_MAX)
            if collapse['needs_recovery'] and not smoke:
                print(f"[v5022 {arm}] WARNING sustained collapse "
                      f"({health.stepping_collapse_streak} updates); "
                      "main foundation does not roll back automatically",
                      flush=True)
            record={
                "arm":arm,"step":step,
                "reward_mean":float(aux["rollout_reward_mean"]),
                "fall_rate":float(row.get("fall_rate",float("nan"))),
                "step_evidence":float(row.get("step_evidence",0.)),
                "terrain_step_approach":float(np.mean(m["terrain_step_approach"])),
                "terrain_step_reached":float(np.mean(m["terrain_step_reached"])),
                "front_valid_fraction":float(np.mean(m["front_valid_fraction"])),
                "ppo_approx_kl":float(aux["approx_kl"]),
                "governor_steady":float(governor.params.steady_difficulty),
                "ppo_lr_scale":lr_scale,
                "collapse_streak":float(health.stepping_collapse_streak),
                "field_known_nodes":float(field.summary()["field_known_nodes"]),
                "field_known_edges":float(field.summary()["field_known_edges"]),
            }
            if not all(math.isfinite(v) for k,v in record.items()
                       if isinstance(v,float)):
                raise FloatingPointError(f"nonfinite v5022 metric at step={step}")
            records.append(record)
            if output_path is not None:
                with output_path.open("a",newline="") as f:
                    w=csv.DictWriter(f,fieldnames=list(record))
                    if f.tell()==0:w.writeheader()
                    w.writerow(record)
            if smoke or (step//nstep)%10==0:
                rate=(step-(target_step-nstep if smoke else 0))/max(
                    1e-6,time.perf_counter()-started)
                print(f"[v5022 {arm}] step={step:,} "
                      f"reward={record['reward_mean']:+.3f} "
                      f"fall={record['fall_rate']:.1%} "
                      f"reached={record['terrain_step_reached']:.3f} "
                      f"front_valid={record['front_valid_fraction']:.2f} "
                      f"fps={rate:,.0f}",flush=True)
            if not smoke and (step-last_save>=1_000_000 or step>=target_step):
                save()
                last_save=step
        # Evaluate with compilation caches released on GPU; this avoids
        # retaining a huge PPO executable alongside the long trace exam.
        if device=="cuda":
            import gc
            jax.clear_caches()
            gc.collect()
        # Independent fixed-seed policy evaluation, not the training reward.
        p=env.params_array(governor.params_dict())
        std=jnp.sqrt(jnp.maximum(obs_var,1e-6))
        # Both arms face EXACTLY the same held-out 0.22m/s forward request.
        # Passing x=.96 alive and supported is a physical result, not reward.
        crossing=trainer["evaluate_terrain_crossing"](
            state.params,p,obs_mean,std,
            jax.random.PRNGKey(seed+9017),
            300 if (not smoke or device=="cuda") else 8,.22)
        crossing_names=("step_crossed_live_fraction","step_approached_live_fraction",
                        "episode_ended_fraction","max_x_mean_m")
        cross_report=dict(zip(crossing_names,map(float,np.asarray(crossing))))
        if not all(math.isfinite(v) for v in cross_report.values()):
            raise FloatingPointError("nonfinite held-out terrain exam")
        if not smoke or device=="cuda":
            # CUDA smoke also checks full-sized FIXED EVALUATION memory.
            ev=trainer["evaluate_summary"](
                state.params,p,obs_mean,std,
                jax.random.PRNGKey(seed+90919),150)
            measured={k:float(v) for k,v in zip(EVAL_NAMES,np.asarray(ev))}
            if not all(math.isfinite(v) for v in measured.values()):
                raise FloatingPointError("nonfinite held-out fixed evaluation")
            if not smoke:
                (target/"fixed_eval.json").write_text(json.dumps(
                    {"step":step,"arm":arm,"values":measured,
                     "terrain_crossing":cross_report},indent=2))
            print(f"[v5022 {arm}] fixed eval {measured} crossing={cross_report}",flush=True)
        else:
            print(f"[v5022 {arm}] CPU smoke held-out crossing {cross_report}",flush=True)
        return {"arm":arm,"step":step,"latest":records[-1] if records else None,
                "held_out_crossing":cross_report,
                "fixed_eval":measured if not smoke else None}
    finally:
        # Checkpoint current entire learned state on graceful interruption
        # where possible; no partial serialized checkpoint is ever promoted.
        if not smoke and step>last_save:
            save()


def main(argv=None):
    p=argparse.ArgumentParser(description="Cold-start v5022 matched 364-input terrain/ToF experiment")
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan",action="store_true")
    mode.add_argument("--smoke",action="store_true")
    mode.add_argument("--run",action="store_true")
    p.add_argument("--root",type=Path,default=None,
                   help="parent directory (distinct visible/blinded lineages)")
    p.add_argument("--arm",choices=("both","visible","blinded"),default="both")
    p.add_argument("--envs",type=int,default=2048)
    p.add_argument("--steps",type=int,default=10_000_000,
                   help="absolute PPO steps EACH arm, not combined total")
    p.add_argument("--seed",type=int,default=733)
    p.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    args=p.parse_args(argv)
    if args.plan:
        print(json.dumps({
            "question":"Does honest delayed 4x4 ToF improve cold-start locomotion on real solid steps/crest?",
            "terrain":"identical physical step-up/down ledge and crest for both",
            "arms":"364 inputs each; visible raw depths vs zeros and invalid masks",
            "mount":"(0,0,-.018)m, 70 degrees downward, 4 rays/zone",
            "steps_per_arm":args.steps,"envs":args.envs,"seed":args.seed,
            "outputs":"separate checkpoint, metrics.csv, fixed_eval.json",
            "hard_gates":"smoke, no nonfinite PPO, train/validation support and fresh GPU memory checks"},indent=2))
        return 0
    if args.run and args.root is None:
        p.error("--run requires --root for isolated durable checkpoints")
    if args.envs<2 or args.envs>4096 or args.steps<1:
        p.error("envs must be 2–4096, steps positive")
    if args.run and args.device!="cuda":
        p.error("full campaign requires CUDA; use --smoke for CPU checks")
    configure_jax(args.device if args.run else ("cpu" if args.device=="cpu" else "cuda"))
    require_backend(args.device=="cpu" and args.smoke)
    arms=("visible","blinded") if args.arm=="both" else (args.arm,)
    results=[]
    for arm in arms:
        results.append(run_arm(arm=arm,root=args.root,envs=args.envs,
                     steps=args.steps,seed=args.seed,smoke=args.smoke,
                     device=args.device))
        # Experiment arms have identical models but separate static captured
        # JAX values. Release old compiled executables on an 8GB card.
        import gc
        import jax
        gc.collect()
        jax.clear_caches()
    if args.root is not None and args.run:
        summary={"arms":results}
        if len(results)==2:
            a,b=results
            summary["visible_minus_blinded"]={
                k:a["held_out_crossing"][k]-b["held_out_crossing"][k]
                for k in a["held_out_crossing"]}
            # Report the measured difference; never silently crown a new
            # champion on raw training reward or a single noisy exam.
            summary["decision"]="review fixed-seed crossing, survival and locomotion; no automatic promotion"
        (args.root/"summary.json").write_text(json.dumps(summary,indent=2))
    print("[v5022] "+json.dumps(results),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
