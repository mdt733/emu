"""v5022 front-ToF A/B orchestration; the actual trainer is main's.

Long runs use training.train with its established learning, governor,
health/recovery, checked checkpoints and evaluations. This file supplies
only paired run identities and explicit preflight optimizer smoke. Archived
220/232 policies and main remain untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np

from .contracts import (ROBOT_XML, OBS_SCHEMA_TERRAIN, TERRAIN_OBS_DIM,
                        terrain_observation_contract)
from .environment import EmuMJXEnv
from .curriculum import RegimeGovernor
from .ppo import configure_jax, require_backend, make_trainer
from .training_config import (ROLLOUT_STEPS, PPO_EPOCHS, PPO_MINIBATCHES,
                              PPO_LR, PPO_ENTROPY_COST, PPO_VALUE_COST)

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


def run_arm(*, arm: str, root: Path | None, envs: int, steps: int,
            seed: int, smoke: bool, device: str) -> dict:
    """Only the preflight PPO smoke. Never a second long-run trainer."""
    if not smoke:
        raise ValueError("Long-running v5022 policies must use main training.train")
    import jax
    import jax.numpy as jnp
    visible=(arm=="visible")
    full_size=(device=="cuda")
    rollout=ROLLOUT_STEPS if full_size else 2
    env=EmuMJXEnv(str(ROBOT_XML),num_envs=envs,seed=seed,
                  observation_schema=OBS_SCHEMA_TERRAIN,
                  front_visible=visible,attitude_profile="v5015",
                  sensor_noise_scale=1.5,max_episode_steps=700)
    if env.obs_dim!=TERRAIN_OBS_DIM:
        raise RuntimeError("v5022 sensor actor schema changed")
    trainer=make_trainer(env,seed=seed,lr=PPO_LR,rollout=rollout,
                         epochs=PPO_EPOCHS if full_size else 1,
                         minibatches=PPO_MINIBATCHES if full_size else 1,
                         entropy_cost=PPO_ENTROPY_COST,
                         value_cost=PPO_VALUE_COST)
    governor=RegimeGovernor(difficulty_base_step=.0014,
                            regime_command_frac=.6,open_frontier=False)
    governor.params.frontier_vx_hi=.32
    governor.params.frontier_vx_lo=-.10
    governor.params.frontier_vy=.12
    governor.params.frontier_yaw=.8
    params=env.params_array(governor.params_dict())
    state=trainer["make_initial_state"](params)
    mean=jnp.zeros((TERRAIN_OBS_DIM,),dtype=jnp.float32)
    std=jnp.ones_like(mean)
    print(f"[v5022 {arm}] smoke starting {envs} real MJX worlds, "
          f"{rollout} frames, {TERRAIN_OBS_DIM} actor inputs",flush=True)
    state,metrics,aux,raw,_=trainer["train_update"](
        state,params,mean,std,jnp.asarray(.25,jnp.float32))
    aux=jax.device_get(aux)
    reward=float(aux["rollout_reward_mean"])
    if not math.isfinite(reward):
        raise FloatingPointError("nonfinite PPO rollout reward")
    # Full-shaped CUDA preflight includes the actual independent held-out
    # step-exam compilation, with training executables evicted first.
    if full_size:
        import gc
        jax.clear_caches()
        gc.collect()
    crossing=trainer["evaluate_terrain_crossing"](
        state.params,params,mean,std,
        jax.random.PRNGKey(seed+9017),300 if full_size else 8,.22)
    names=("step_crossed_live_fraction","step_approached_live_fraction",
           "episode_ended_fraction","max_x_mean_m")
    result=dict(zip(names,map(float,np.asarray(crossing))))
    fixed=None
    if full_size:
        measured=trainer["evaluate_summary"](
            state.params,params,mean,std,
            jax.random.PRNGKey(seed+90919),150)
        fixed=dict(zip(EVAL_NAMES,map(float,np.asarray(measured))))
    if not all(math.isfinite(x) for x in
               (*result.values(),*(fixed or {}).values())):
        raise FloatingPointError("nonfinite v5022 smoke evaluator")
    print(f"[v5022 {arm}] smoke PASS reward={reward:.3f} "
          f"crossing={result}",flush=True)
    return {"arm":arm,"step":envs*rollout,
            "latest":{"reward_mean":reward},
            "held_out_crossing":result,"fixed_eval":fixed}


def run_main_arm(*, arm: str, root: Path, envs: int, steps: int,
                 seed: int, device: str) -> dict:
    """The ONE production training lifecycle: main's trainer, no forked loop.

    Reuse main training's KL controller, retreat/recovery, snapshots,
    optimizer state, curriculum field, fixed evaluation and resume.
    Only schema-9 sensor/physical geometry and A/B visibility differ.
    """
    from dataclasses import replace
    import gc
    import jax
    import jax.numpy as jnp
    from .training_config import foundation_config, training_sensor_contract
    from .training import train, env_kwargs_from_args
    from .checkpoint_io import load_checkpoint
    from .ppo import make_trainer
    source=source_info()
    if source["branch"]!="feature/v5022-forward-terrain" or source["dirty"]:
        raise RuntimeError("v5022 requires a clean, named feature branch; main must remain protected")
    run_dir=root/arm
    run_dir.mkdir(parents=True,exist_ok=True)
    from .contracts import SENSOR_SETTINGS_DEFAULTS
    sensor_settings=dict(SENSOR_SETTINGS_DEFAULTS)
    sensor_settings["sensor_noise_scale"]=1.5  # main's SENSOR_NOISE_SCALE
    identity=dict(format=FORMAT,arm=arm,source=source,seed=seed,envs=envs,
                  contract=terrain_observation_contract(
                      sensor_settings,front_visible=arm=="visible"))
    manifest=run_dir/"v5022_identity.json"
    if manifest.is_file():
        if json.loads(manifest.read_text())!=identity:
            raise ValueError("v5022 physical sensor/source identity changed; choose a new root")
    else:
        if (run_dir/"checkpoints").exists():
            raise RuntimeError("Unidentified checkpoint directory: refusing to overwrite it")
        manifest.write_text(json.dumps(identity,indent=2,sort_keys=True))

    ckpt=run_dir/"checkpoints"/"latest.pkl"
    final=run_dir/"checkpoints"/"final.pkl"
    resume="auto" if ckpt.exists() or final.exists() else ""
    base=foundation_config(run_dir=run_dir,seed=seed,device=device,envs=envs)
    config=replace(
        base,steps=steps,observation_schema=OBS_SCHEMA_TERRAIN,
        front_visible=arm=="visible",attitude_profile="v5015",
        allow_rollback=True,cloud_probes_per_eval=0,
        cloud_final_probes=0,capability_field_mode="shadow",resume=resume)
    expected=training_sensor_contract(config)
    if expected != identity["contract"]:
        raise RuntimeError("v5022 front+old sensor contract mismatch with main trainer")
    print(f"[v5022 {arm}] running canonical EMU training.train "
          "with main's health, recovery, snapshots and KL governor",flush=True)
    train(config)
    if not final.is_file():
        raise RuntimeError(f"main trainer did not publish final checkpoint at {final}")
    payload=load_checkpoint(
        final,expected_observation_schema=OBS_SCHEMA_TERRAIN,
        expected_sensor_contract=expected)
    final_step=int(payload["meta"]["step"])
    # Use a small held-out cohort on a new seeded course; this is not PPO
    # training and does not let trial labels reach the actor.
    jax.clear_caches()
    gc.collect()
    check_env=EmuMJXEnv(str(ROBOT_XML),num_envs=min(128,envs),
                       seed=seed+9017,max_episode_steps=700,
                       **env_kwargs_from_args(config))
    learner=make_trainer(check_env,seed=seed+9017,lr=PPO_LR,rollout=2,
                         epochs=1,minibatches=1,entropy_cost=PPO_ENTROPY_COST,
                         value_cost=PPO_VALUE_COST)
    norm=payload["obs_norm"]
    mean=jnp.asarray(norm["mean"])
    std=jnp.sqrt(jnp.asarray(norm["var"]))
    params=jax.device_put(payload["state"]["params"])
    task=check_env.params_array({})
    crossing=learner["evaluate_terrain_crossing"](
        params,task,mean,std,jax.random.PRNGKey(seed+9017),300,.22)
    names=("step_crossed_live_fraction","step_approached_live_fraction",
           "episode_ended_fraction","max_x_mean_m")
    crossing=dict(zip(names,map(float,np.asarray(crossing))))
    ev=learner["evaluate_summary"](
        params,task,mean,std,jax.random.PRNGKey(seed+90919),150)
    metrics=dict(zip(EVAL_NAMES,map(float,np.asarray(ev))))
    if not all(math.isfinite(v) for v in (*crossing.values(),*metrics.values())):
        raise FloatingPointError("nonfinite v5022 held-out evaluation")
    report={"step":final_step,"arm":arm,
            "source_checkpoint":str(final),
            "held_out_crossing":crossing,"fixed_eval":metrics}
    (run_dir/"v5022_crossing.json").write_text(json.dumps(report,indent=2))
    print(f"[v5022 {arm}] held-out {crossing}",flush=True)
    return report


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
        if args.run:
            results.append(run_main_arm(arm=arm,root=args.root,envs=args.envs,
                       steps=args.steps,seed=args.seed,device=args.device))
        else:
            results.append(run_arm(arm=arm,root=None,envs=args.envs,
                       steps=args.steps,seed=args.seed,smoke=True,
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
