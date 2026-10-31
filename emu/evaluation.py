"""Read-only fixed evaluation panels for the training loop.

The evaluator owns deterministic worlds/panels and metric reduction. It receives a
policy plus observation-normalizer snapshot and returns evidence; it has no path to
mutate PPO, curriculum, health or the capability field.
"""
from __future__ import annotations

from typing import Any, Callable
from pathlib import Path
from .checkpoint_io import save_checkpoint, load_checkpoint
from .results import append_metrics_csv, write_json, write_rows_csv
from dataclasses import dataclass
import time
import math
import argparse
import json

import numpy as np

from .environment import EmuMJXEnv
from .contracts import task_param_defaults
from .contracts import ROBOT_XML, VERSION, CONTROL_DT, recovery_outcome_numpy
from .ppo import make_trainer, configure_jax, require_backend
from .training_config import (
    TrainingConfig, MAX_EPISODE_STEPS,
    PPO_LR, ROLLOUT_STEPS, PPO_ENTROPY_COST,
    PPO_VALUE_COST, EVAL_SEED, EVAL_HORIZON, SENSOR_NOISE_SCALE,
)

@dataclass(frozen=True)
class PolicySnapshot:
    """Exact post-update policy and normalizer selected for held-out evaluation."""
    step: int
    state: Any
    obs_mean: Any
    obs_var: Any
    obs_count: float

    @property
    def params(self):
        return self.state.params

    def norm(self) -> dict[str, Any]:
        return {'mean': self.obs_mean, 'var': self.obs_var, 'count': self.obs_count}


def outranks_best(current_score: float, best_score: float) -> bool:
    return math.isfinite(current_score) and current_score > best_score


def checkpoint_eval_score(meta: dict[str, Any]) -> float:
    """Read the score of THIS policy, not a historical rolling average."""
    score = float(meta.get('instant_eval_score', meta.get('best_eval_score', -1e9)))
    return score if math.isfinite(score) else -1e9


EVAL_FIELDS = ('survival', 'first_fall_step_mean', 'vx_mae', 'vy_mae', 'yaw_mae',
               'gravity_z', 'foot_mid_cone', 'step_evidence', 'support',
               'pitch_mae', 'height_shortfall_m')


def _episode(steady: float, transition: float, resilience: float, posture: float,
             mix: tuple[float, float, float], command_frac: float) -> dict[str, float]:
    ep = task_param_defaults()
    ep.update(steady_difficulty=steady, transition_difficulty=transition,
              resilience_difficulty=resilience, posture_difficulty=posture,
              regime_frac_steady=mix[0], regime_frac_transition=mix[1],
              regime_frac_resilience=mix[2], regime_command_frac=command_frac)
    return ep


# name -> (world, deterministic key salt, episode parameters)
PANELS = {
    'core': ('fixed', 100, _episode(0.0, 0.0, 0.0, 0.0, (1.0, 0.0, 0.0), 0.50)),
    'stress30': ('fixed', 101, _episode(.30, .30, .30, .20, (1/3, 1/3, 1/3), .50)),
    'dynamic50': ('fixed', 102, _episode(.45, .60, .25, .25, (.20, .65, .15), .75)),
    'resilience50': ('fixed', 103, _episode(.35, .35, .60, .40, (.15, .20, .65), .65)),
    'dynamic80': ('fixed', 104, _episode(.75, .85, .35, .35, (.15, .70, .15), .90)),
    'combo70': ('combo', 105, _episode(.70, .72, .30, .35, (.35, .50, .15), .90)),
    'resilience75': ('fixed', 106, _episode(.55, .55, .80, .55, (.10, .20, .70), .80)),
    'robust40': ('robust', 404, _episode(.40, .45, .50, .35, (.25, .30, .45), .65)),
    'growth110': ('growth', 811, _episode(1.0, .80, .30, .35, (.55, .35, .10), .90)),
    'growthcombo110': ('growth_combo', 919, _episode(1.0, .80, .30, .35, (.55, .35, .10), .90)),
}

EVAL_PERF_TIMER_NAMES = tuple(f'eval_{name}' for name in PANELS)


def _world(config: TrainingConfig, *, seed_offset: int, limits: tuple[float, float, float, float],
           combo_mix: float = 0.0, sensor_scale: float | None = None):
    seed = EVAL_SEED + seed_offset
    vx_hi, vx_lo, vy, yaw = limits
    env = EmuMJXEnv(
        str(ROBOT_XML), num_envs=config.envs, seed=seed,
        max_episode_steps=MAX_EPISODE_STEPS,
        # A held-out physical exam must use the learner's actual push
        # physics: the new 360 learner can earn >0.70N challenges.
        push_max_force_n=(config.push_max_force_n if config.unified_capability_focus else 0.70),
        vx_hi_max=vx_hi, vx_lo_min=vx_lo,
        vy_max_mps=vy, yaw_max_rad_s=yaw, phase_speed_gain=config.phase_speed_gain,
        attitude_profile=config.attitude_profile, pitch_command_max_deg=config.pitch_command_max_deg,
        sensor_noise_scale=SENSOR_NOISE_SCALE if sensor_scale is None else sensor_scale,
        observation_schema=config.observation_schema, open_frontier=False,
        # Fixed evaluation of the new array-only 360 actor must use its real
        # two-slab world and sensor meanings even with ZERO terrain exposure.
        # Archived 220/232/older tripod-360 panels remain unchanged.
        unified_capability_focus=bool(config.unified_capability_focus),
        velocity_i_ramp_s=config.velocity_i_ramp_s,
        velocity_i_multiplier=config.velocity_i_multiplier,
        terrain_step_max_m=(config.terrain_step_max_m if config.unified_capability_focus else 0.0),
        combo_mix_frac=combo_mix, combo_probe_frac=0.0,
    )
    if config.unified_capability_focus:
        from .training_config import training_sensor_contract
        if env.sensor_contract != training_sensor_contract(config):
            raise RuntimeError('frozen fixed evaluator sensor geometry differs from unified policy')
    trainer = make_trainer(env, seed=seed, lr=PPO_LR, rollout=ROLLOUT_STEPS,
                           epochs=1, minibatches=1, entropy_cost=PPO_ENTROPY_COST,
                           value_cost=PPO_VALUE_COST)
    return env, trainer


def build_worlds(config: TrainingConfig) -> dict[str, tuple[Any, dict[str, Any]]]:
    benchmark = (.60, -.20, .22, 1.50)
    growth = (.75, -.28, .30, 1.90)
    live = (config.vx_hi_max, config.vx_lo_min, config.vy_max_mps, config.yaw_max_rad_s)
    return {
        'fixed': _world(config, seed_offset=0, limits=benchmark),
        'combo': _world(config, seed_offset=606, limits=benchmark, combo_mix=1.0),
        'growth': _world(config, seed_offset=811, limits=growth),
        'growth_combo': _world(config, seed_offset=919, limits=growth, combo_mix=1.0),
        'robust': _world(config, seed_offset=404, limits=live,
                         sensor_scale=SENSOR_NOISE_SCALE),
    }


def initial_metrics() -> dict[str, float]:
    """Complete held-out metric schema, including fields populated only after an exam.

    ``metrics.csv`` is intentionally strict about column drift. Therefore every
    held-out metric that can appear later in a run must exist from row one, with
    a neutral/sentinel value until the corresponding exam has executed.
    """
    out = {f'eval_{name}_{field}': -1.0 for name in PANELS for field in EVAL_FIELDS}
    out.update(eval_policy_step=-1.0, eval_broad_score=-1e9, eval_broad_tail3=-1e9)
    out.update({
        'perf_fixed_eval_total_s': 0.0,
        'perf_eval_cohorts_used': 0.0,
        'perf_eval_env_steps': 0.0,
        'perf_eval_env_steps_per_s': 0.0,
        'perf_best_checkpoint_total_s': 0.0,
        'perf_best_checkpoint_device_get_s': 0.0,
        'perf_best_checkpoint_pickle_s': 0.0,
        'field_calibration_mae': -1.0,
        'field_calibration_brier': -1.0,
        'field_calibration_points': 0.0,
    })
    out.update({f'perf_eval_{name}_s': 0.0 for name in PANELS})
    return out


def broad_score(out: dict[str, float], *, profile: str = 'v5014') -> float:
    survival_weights = {
        'core': .05, 'stress30': .08, 'dynamic50': .10, 'dynamic80': .16,
        'combo70': .12, 'resilience50': .12, 'resilience75': .14,
        'robust40': .11, 'growth110': .06, 'growthcombo110': .06,
    }
    survival = sum(w * out.get(f'eval_{name}_survival', 0.0)
                   for name, w in survival_weights.items())
    penalties = (
        .10 * min(1.0, out.get('eval_dynamic80_vx_mae', 1.0) / .12) +
        .10 * min(1.0, out.get('eval_dynamic80_yaw_mae', 2.0) / .40) +
        .08 * min(1.0, out.get('eval_combo70_vy_mae', 1.0) / .12) +
        .10 * min(1.0, out.get('eval_combo70_yaw_mae', 2.0) / .40) +
        .06 * min(1.0, out.get('eval_growthcombo110_yaw_mae', 2.0) / .55)
    )
    if profile == 'v5015':
        # v5015 checkpoint selection must not reward a descending gait.
        panels = ('dynamic50', 'dynamic80', 'combo70', 'resilience75')
        shortfall = sum(max(0.0, out.get(f'eval_{p}_height_shortfall_m', 0.0)) for p in panels) / len(panels)
        pitch_err = sum(max(0.0, out.get(f'eval_{p}_pitch_mae', 0.0)) for p in panels) / len(panels)
        penalties += 0.15 * min(1.0, shortfall / 0.035) + 0.10 * min(1.0, pitch_err / 0.35)
    return float(survival - penalties)


def evaluate_fixed(worlds: dict[str, tuple[Any, dict[str, Any]]], policy_params: Any,
                   obs_norm: dict[str, Any], *, step: int, cohorts: int, perf=None) -> dict[str, float]:
    """Evaluate all fixed panels without mutating any training subsystem."""
    import jax

    base_key = jax.random.PRNGKey(EVAL_SEED)
    out: dict[str, float] = {'eval_policy_step': float(step)}
    timings: dict[str, float] = {}
    env_steps = 0
    started = time.perf_counter()
    obs_std = obs_norm['var'] ** 0.5

    for name, (world_name, salt, episode) in PANELS.items():
        env, trainer = worlds[world_name]
        t0 = time.perf_counter()
        summaries = []
        for cohort in range(max(1, int(cohorts))):
            key = jax.random.fold_in(base_key, salt + 10007 * cohort)
            compact = trainer['evaluate_summary'](
                policy_params, env.params_array(episode), obs_norm['mean'], obs_std,
                key, EVAL_HORIZON)
            compact = np.asarray(jax.device_get(compact), dtype=np.float64)
            summaries.append({k: float(v) for k, v in zip(EVAL_FIELDS, compact)})
            env_steps += int(env.num_envs) * EVAL_HORIZON
        summary = {k: float(sum(x[k] for x in summaries) / len(summaries)) for k in EVAL_FIELDS}
        out.update({f'eval_{name}_{k}': v for k, v in summary.items()})
        timings[name] = time.perf_counter() - t0
        if perf is not None:
            perf.add(f'eval_{name}', timings[name])

    elapsed = time.perf_counter() - started
    scoring_profile = next(iter(worlds.values()))[0].attitude_profile
    out['eval_broad_score'] = broad_score(out, profile=scoring_profile)
    out['perf_fixed_eval_total_s'] = float(elapsed)
    out['perf_eval_cohorts_used'] = float(max(1, int(cohorts)))
    out['perf_eval_env_steps'] = float(env_steps)
    out['perf_eval_env_steps_per_s'] = float(env_steps / max(elapsed, 1e-9))
    out.update({f'perf_eval_{name}_s': float(seconds) for name, seconds in timings.items()})
    return out


def record_fixed_exam(
    snapshot: PolicySnapshot, *, worlds: dict, cohorts: int, run_dir: Path,
    broad_history: list[float], best_score: float, best_path: Path,
    curriculum_state: Callable[[], dict], health_state: Callable[[], dict],
    field_state: Callable[[], dict], lr_scale: float, version: str,
    perf: Any, log_prefix: str, checkpoint_meta: dict | None = None,
) -> tuple[dict[str, float], float]:
    """Evaluate one frozen policy and retain the strongest exact checkpoint.

    State getters are called only if the policy earns a new best checkpoint.
    Evidence never flows into PPO or the training curriculum.
    """
    step = snapshot.step
    out = evaluate_fixed(
        worlds, snapshot.params, snapshot.norm(),
        step=snapshot.step, cohorts=cohorts, perf=perf)
    broad_history.append(float(out['eval_broad_score']))
    out['eval_broad_tail3'] = float(sum(broad_history[-3:]) / len(broad_history[-3:]))

    best_ckpt_timing = {'device_get_s': 0.0, 'pickle_write_s': 0.0, 'total_s': 0.0, 'bytes': 0.0, 'saved': 0.0}
    if outranks_best(float(out['eval_broad_score']), best_score):
        best_score = float(out['eval_broad_score'])
        best_ckpt_timing = save_checkpoint(
            best_path, snapshot.state,
            {'step': snapshot.step, 'policy_step': snapshot.step, 'version': version,
             'best_eval_score': best_score,
             'instant_eval_score': out['eval_broad_score'], 'evaluated_policy': True,
             'lr_scale': lr_scale, **(checkpoint_meta or {})},
            snapshot.norm(),
            curriculum_state=curriculum_state(),
            health_state=health_state(),
            field_state=field_state(), require_finite_state=True)
        if not best_ckpt_timing.get('saved', 0.0):
            raise RuntimeError('Best-policy publication refused: non-finite actor or optimizer state')
        perf.add('checkpoint_best_device_get', best_ckpt_timing['device_get_s'])
        perf.add('checkpoint_best_pickle_write', best_ckpt_timing['pickle_write_s'])
        perf.add('checkpoint_best_total', best_ckpt_timing['total_s'])

    perf.add('fixed_eval_total', out['perf_fixed_eval_total_s'])
    out['perf_best_checkpoint_total_s'] = float(best_ckpt_timing['total_s'])
    out['perf_best_checkpoint_device_get_s'] = float(best_ckpt_timing['device_get_s'])
    out['perf_best_checkpoint_pickle_s'] = float(best_ckpt_timing['pickle_write_s'])
    append_metrics_csv(run_dir / 'fixed_eval.csv', {'step': step, **out})
    if perf.enabled:
        write_json(run_dir / 'performance.json', {'step': step, 'cumulative': perf.row(prefix='')})

    print(f"{log_prefix} fixed-eval step={step:,} score={out['eval_broad_score']:.3f} tail3={out['eval_broad_tail3']:.3f} "
          f"core={out['eval_core_survival']:.3f} dyn80={out['eval_dynamic80_survival']:.3f} "
          f"combo70={out['eval_combo70_survival']:.3f}/{out['eval_combo70_yaw_mae']:.3f}yaw "
          f"res75={out['eval_resilience75_survival']:.3f} robust40={out['eval_robust40_survival']:.3f} "
          f"growth={out['eval_growth110_survival']:.3f}/{out['eval_growthcombo110_survival']:.3f} "
          f"eval_wall={out.get('perf_fixed_eval_total_s',0.0):.1f}s", flush=True)
    return out, best_score


# Saved-checkpoint command and recovery capability atlas.
def checkpoint_training_profile(checkpoint: Path) -> tuple[str, float]:
    """Identify a policy's training contract for independent atlas tests."""
    ck = Path(checkpoint).expanduser().resolve()
    for path in (ck.parent.parent / 'training_config.json', ck.parent / 'training_config.json'):
        if path.is_file():
            cfg = json.loads(path.read_text(encoding='utf-8'))
            return (str(cfg.get('attitude_profile') or 'v5014'),
                    float(cfg.get('pitch_command_max_deg', 8.0214)))
    root = ck.parent / 'lineage.json'
    if root.is_file():
        lineage = json.loads(root.read_text(encoding='utf-8'))
        if lineage.get('version') == 'v5015' and lineage.get('phases'):
            return 'v5015', float(lineage['phases'][-1]['pitch_max_deg'])
    return 'v5014', 8.0214


def fmt(x, d=3):
    try: return f"{float(x):.{d}f}"
    except Exception: return '-'


def command_tol(axis: str, value: float) -> float:
    a=abs(float(value))
    if axis.startswith('vx'): return max(0.06, 0.20*a)
    if axis.startswith('vy'): return max(0.05, 0.25*a)
    return max(0.18, 0.20*a)


def physical_frontiers(rows: list[dict], value_key='level_abs') -> dict:
    out={}
    for target in (0.95,0.80,0.50):
        ok=[r for r in rows if float(r.get('survival',0)) >= target and float(r.get('tracking_ok',0)) >= 0.5]
        out[f'frontier_{int(target*100)}'] = max((float(r[value_key]) for r in ok), default=None)
    return out


def recovery_frontiers(rows: list[dict], value_key='level') -> dict:
    out={}
    for target in (0.95,0.80,0.50):
        ok=[r for r in rows if float(r.get('recovery_rate',0)) >= target]
        out[f'frontier_{int(target*100)}'] = max((float(r[value_key]) for r in ok), default=None)
    return out


def atlas_main(argv=None):
    ap=argparse.ArgumentParser(description=f'EMU {VERSION} physical capability/recovery atlas')
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--device', choices=['auto','cpu','cuda','gpu'], default='cuda')
    ap.add_argument('--allow-cpu-mjx', action='store_true')
    ap.add_argument('--envs', type=int, default=1024)
    ap.add_argument('--horizon', type=int, default=300, help='control steps; 300 = 6s at 20ms')
    ap.add_argument('--seed', type=int, default=733)
    args=ap.parse_args(argv)

    configure_jax(args.device)
    import jax
    require_backend(args.allow_cpu_mjx)

    ck=Path(args.checkpoint).expanduser().resolve(); out=Path(args.out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    try:
        payload = load_checkpoint(ck)
    except Exception as exc:
        raise SystemExit(f'invalid checkpoint {ck}: {exc}') from exc
    state=payload['state']; params=jax.device_put(state['params'])
    norm=payload.get('obs_norm') or {}

    attitude_profile, pitch_deg = checkpoint_training_profile(ck)
    env=EmuMJXEnv(str(ROBOT_XML), num_envs=args.envs, seed=args.seed,
                  attitude_profile=attitude_profile, pitch_command_max_deg=pitch_deg,
                  max_episode_steps=max(args.horizon+50, 700), push_max_force_n=0.70, vx_hi_max=1.8, vx_lo_min=-0.9,
                  vy_max_mps=0.70, yaw_max_rad_s=4.5,
                  phase_speed_gain=1.25, sensor_noise_scale=1.5,
                  open_frontier=False, combo_mix_frac=0.0)
    tr=make_trainer(env, seed=args.seed, lr=2e-4, rollout=32, epochs=1, minibatches=1)
    obs_mean=jax.numpy.asarray(norm.get('mean', np.zeros((tr['obs_dim'],),np.float32)), dtype=jax.numpy.float32)
    obs_std=jax.numpy.asarray(np.sqrt(norm.get('var', np.ones((tr['obs_dim'],),np.float32))), dtype=jax.numpy.float32)

    eval_fields=('survival','first_fall_step_mean','vx_mae','vy_mae','yaw_mae','gravity_z','foot_mid_cone','step_evidence','support','pitch_mae','height_shortfall_m')
    rows=[]; summary={'checkpoint':str(ck),'attitude_profile':attitude_profile,'pitch_command_max_deg':pitch_deg,'envs':args.envs,'horizon_steps':args.horizon,'control_dt_s':CONTROL_DT,'commands':{},'recovery':{}}
    t0=time.time(); basekey=jax.random.PRNGKey(args.seed+9000); salt=0

    def base_ep():
        ep = task_param_defaults()
        ep.update(regime_command_frac=1.0, probe_cmd_vx=0.08, probe_cmd_vy=0.0, probe_cmd_yaw=0.0)
        return ep

    def eval_command(axis, value):
        nonlocal salt
        ep=base_ep()
        if axis=='vx_forward' or axis=='vx_reverse': ep['probe_cmd_vx']=value
        elif axis.startswith('vy'): ep['probe_cmd_vy']=value
        elif axis.startswith('yaw'): ep['probe_cmd_yaw']=value
        salt+=1; k=jax.random.fold_in(basekey,salt)
        arr=np.asarray(jax.device_get(tr['evaluate_summary'](params,env.params_array(ep),obs_mean,obs_std,k,args.horizon)),dtype=float)
        d={k2:float(v) for k2,v in zip(eval_fields,arr)}
        mae=d['vx_mae'] if axis.startswith('vx') else d['vy_mae'] if axis.startswith('vy') else d['yaw_mae']
        tol=command_tol(axis,value)
        return {**d,'axis':axis,'level':float(value),'level_abs':abs(float(value)),'tracking_mae':mae,'tracking_tol':tol,'tracking_ok':float(mae<=tol)}

    command_sweeps={
        'vx_forward':[0.20,0.40,0.60,0.80,1.00,1.20,1.40,1.60],
        'vx_reverse':[-0.10,-0.20,-0.30,-0.40,-0.50,-0.60,-0.70,-0.80],
        'vy_left':[0.10,0.18,0.26,0.34,0.42,0.50,0.60],
        'vy_right':[-0.10,-0.18,-0.26,-0.34,-0.42,-0.50,-0.60],
        'yaw_left':[0.50,1.0,1.5,2.0,2.5,3.0,3.5,4.0],
        'yaw_right':[-0.50,-1.0,-1.5,-2.0,-2.5,-3.0,-3.5,-4.0],
    }
    for axis,levels in command_sweeps.items():
        rr=[]
        for x in levels:
            r=eval_command(axis,x); r['kind']='command'; rr.append(r); rows.append(r)
            print(f"[atlas] {axis:12s} {x:+.3f}: survival={r['survival']:.3f} mae={r['tracking_mae']:.3f} tol={r['tracking_tol']:.3f}",flush=True)
        summary['commands'][axis]={**physical_frontiers(rr),'curve':rr}

    def eval_recovery(name, level, overrides):
        nonlocal salt
        ep=base_ep(); ep.update(overrides)
        salt+=1; k=jax.random.fold_in(basekey,salt)
        ff,fr=tr['evaluate_recovery'](params,env.params_array(ep),obs_mean,obs_std,k,args.horizon,10)
        ff,fr=jax.device_get((ff,fr))
        return {'kind':'recovery','axis':name,'level':float(level),
                **recovery_outcome_numpy(ff,fr)}

    recovery_specs={
        'tilt_deg':[(x,{'probe_tilt_rad':math.radians(x)}) for x in [5,10,15,20,25,30,35,40,50,60]],
        'linear_velocity_mps':[(x,{'probe_linvel_mps':x}) for x in [0.10,0.20,0.30,0.40,0.55,0.70,0.90,1.10,1.30]],
        'angular_velocity_rad_s':[(x,{'probe_angvel_rad_s':x}) for x in [0.5,1.0,1.5,2.0,2.5,3.0,4.0,5.0,6.0]],
        'joint_offset_deg':[(x,{'probe_joint_rad':math.radians(x)}) for x in [2,5,8,12,16,20,25,30,35]],
    }
    chaos_levels=[0.25,0.50,0.75,1.00,1.25,1.50]
    recovery_specs['combined_chaos_scale']=[(x,{
        'probe_tilt_rad':math.radians(40.0)*x,
        'probe_linvel_mps':0.80*x,
        'probe_angvel_rad_s':4.0*x,
        'probe_joint_rad':math.radians(20.0)*x,
    }) for x in chaos_levels]

    for name,specs in recovery_specs.items():
        rr=[]
        for level,ov in specs:
            r=eval_recovery(name,level,ov); rr.append(r); rows.append(r)
            print(f"[atlas] {name:24s} {level:.3f}: survive={r['survival']:.3f} recover={r['recovery_rate']:.3f} p90={r['recovery_time_p90_s']:.2f}s",flush=True)
        summary['recovery'][name]={**recovery_frontiers(rr),'curve':rr}

    summary['wall_seconds']=round(time.time()-t0,2)
    write_rows_csv(out/'capability_atlas.csv',rows)
    (out/'capability_atlas.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')

    md=[f'# EMU {VERSION} capability / recovery atlas','',
        f'- checkpoint: `{ck}`', f'- envs per point: `{args.envs}`', f'- horizon: `{args.horizon}` control steps ({args.horizon*CONTROL_DT:.1f}s)',
        '', 'This report intentionally avoids a single aggregate score. The 95/80/50 frontiers are read-offs from the complete physical-unit curves in `capability_atlas.csv`.',
        '', '## Command competence frontiers','', '| axis | 95% | 80% | 50% |','|---|---:|---:|---:|']
    for name,v in summary['commands'].items(): md.append(f"| {name} | {fmt(v['frontier_95'])} | {fmt(v['frontier_80'])} | {fmt(v['frontier_50'])} |")
    md += ['', 'A command point counts as competent only when survival clears the stated percentage **and** mean tracking MAE stays inside an explicit axis tolerance (recorded per row in the CSV).',
           '', '## Recovery frontiers','', '| disturbance | 95% recovered | 80% recovered | 50% recovered |','|---|---:|---:|---:|']
    for name,v in summary['recovery'].items(): md.append(f"| {name} | {fmt(v['frontier_95'])} | {fmt(v['frontier_80'])} | {fmt(v['frontier_50'])} |")
    md += ['', 'Recovery means the robot remains unfallen and holds an appropriately commanded chassis attitude, command-tracking, supported state for 0.20s continuously. Recovery time distributions are retained for every point; survival by itself is not treated as recovery.',
           '', f"Wall time: {summary['wall_seconds']:.1f}s", '']
    (out/'atlas.md').write_text('\n'.join(md),encoding='utf-8')
    print(f"[atlas] complete -> {out/'atlas.md'}",flush=True)
    return 0


# Stable compatibility entrypoints for CLI, scripts and existing tests.
# All independent physical-exam implementations are in physical_exam.py.
from .physical_exam import (
    exam_gravity, sha256_file, source_commit, validate_request, exam_request,
    evaluate_directional_run, score_terrain_course_trace, evaluate_terrain_course,
    _summarize_worlds, evaluate_directional_pair, directional_main,
    SCENARIOS, make_schedule, score_unified_capability_trace, _unified_exam_json,
    unified_terrain_matrix_cases, evaluate_unified_capabilities,
)
