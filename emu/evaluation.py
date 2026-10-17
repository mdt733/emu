"""Read-only fixed evaluation panels for the training loop.

The evaluator owns deterministic worlds/panels and metric reduction. It receives a
policy plus observation-normalizer snapshot and returns evidence; it has no path to
mutate PPO, curriculum, health or the capability field.
"""
from __future__ import annotations

from typing import Any, Callable
from pathlib import Path
from .checkpoint_io import save_checkpoint
from .results import append_metrics_csv, write_json
from dataclasses import dataclass
import time
import math

import numpy as np

from .environment import EmuMJXEnv
from .contracts import task_param_defaults
from .contracts import ROBOT_XML
from .ppo import make_trainer
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
        push_max_force_n=0.70, vx_hi_max=vx_hi, vx_lo_min=vx_lo,
        vy_max_mps=vy, yaw_max_rad_s=yaw, phase_speed_gain=config.phase_speed_gain,
        attitude_profile=config.attitude_profile, pitch_command_max_deg=config.pitch_command_max_deg,
        sensor_noise_scale=SENSOR_NOISE_SCALE if sensor_scale is None else sensor_scale,
        open_frontier=False, combo_mix_frac=combo_mix, combo_probe_frac=0.0,
    )
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
    perf: Any, log_prefix: str,
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
             'lr_scale': lr_scale},
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


# Matched held-out branch screens; never fed to PPO's teacher.
import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .checkpoint_io import load_checkpoint
from .contracts import task_param_defaults
from .environment import EmuMJXEnv
from .contracts import ROBOT_XML
from .ppo import make_trainer
from .ppo import configure_jax, require_backend
from .training_config import (
    TrainingConfig, MAX_EPISODE_STEPS, PPO_LR, ROLLOUT_STEPS,
    PPO_ENTROPY_COST, PPO_VALUE_COST, EVAL_SEED,
)

SCREEN_FIELDS = ('survival', 'first_fall_step_mean', 'vx_mae', 'vy_mae', 'yaw_mae',
                 'gravity_z', 'foot_mid_cone', 'step_evidence', 'support',
                 'pitch_mae', 'height_shortfall_m')


def _screen_episode(steady: float, transition: float, resilience: float, posture: float,
             mix: tuple[float, float, float], command_frac: float) -> dict[str, float]:
    ep = task_param_defaults()
    ep.update(steady_difficulty=steady, transition_difficulty=transition,
              resilience_difficulty=resilience, posture_difficulty=posture,
              regime_frac_steady=mix[0], regime_frac_transition=mix[1],
              regime_frac_resilience=mix[2], regime_command_frac=command_frac)
    return ep


SCREEN_PANELS = {
    # Same conceptual coordinates that exposed the v5012 graph-vs-shadow trade.
    # Current v5013 hardware doctrine keeps its one-frame target delay; these are
    # therefore regression-style stress probes, not claims of physical limits.
    'resilience100': ('robust', 1103, _screen_episode(.60, .65, 1.00, .65, (.08, .17, .75), .90)),
    'robust70': ('robust', 1207, _screen_episode(.70, .78, .75, .60, (.20, .35, .45), .90)),
    'growth140': ('growth', 1301, _screen_episode(1.0, .92, .35, .45, (.50, .38, .12), 1.0)),
    'growthcombo140': ('growth_combo', 1409, _screen_episode(1.0, .92, .35, .45, (.50, .38, .12), 1.0)),
}


def _screen_world(config: TrainingConfig, *, envs: int, seed_offset: int,
           limits: tuple[float, float, float, float], push_n: float,
           sensor_scale: float, combo_mix: float, terrain_radius: float = 0.0):
    vx_hi, vx_lo, vy, yaw = limits
    seed = EVAL_SEED + seed_offset
    from .environment import DomainRandomizationConfig
    domain = DomainRandomizationConfig(enabled=terrain_radius > 0, terrain_radius=terrain_radius)
    env = EmuMJXEnv(
        str(ROBOT_XML), num_envs=int(envs), seed=seed,
        max_episode_steps=MAX_EPISODE_STEPS,
        push_max_force_n=push_n,
        vx_hi_max=vx_hi, vx_lo_min=vx_lo, vy_max_mps=vy, yaw_max_rad_s=yaw,
        phase_speed_gain=config.phase_speed_gain,
        attitude_profile=config.attitude_profile, pitch_command_max_deg=config.pitch_command_max_deg,
        sensor_noise_scale=sensor_scale, open_frontier=False,
        combo_mix_frac=combo_mix, combo_probe_frac=0.0,
        field_teach_recovery=False,
        domain_config=domain,
    )
    trainer = make_trainer(env, seed=seed, lr=PPO_LR, rollout=ROLLOUT_STEPS,
                           epochs=1, minibatches=1,
                           entropy_cost=PPO_ENTROPY_COST, value_cost=PPO_VALUE_COST)
    return env, trainer


def evaluate_runs(run_dirs: list[Path], *, envs: int = 512, horizon: int = 350,
                  device: str | None = None, eval_pitch_deg: float | None = None) -> dict[str, dict[str, Any]]:
    """Evaluate multiple final checkpoints through one set of compiled worlds."""
    if not run_dirs:
        return {}
    run_dirs=[Path(x).expanduser().resolve() for x in run_dirs]
    configs=[TrainingConfig.load(r/'training_config.json') for r in run_dirs]
    config=configs[0]
    # The stress worlds depend on physical/evaluation doctrine, not teacher
    # settings. Guard the one config value they currently consume so batching
    # can never silently compare arms through different evaluation dynamics.
    if any(c.phase_speed_gain != config.phase_speed_gain for c in configs[1:]):
        raise ValueError('batched screening requires identical phase_speed_gain')
    if device is not None:
        config=replace(config,device=device)
    if eval_pitch_deg is not None:
        # Freeze identical v5015 exam posture conditions for every contender,
        # independent of the pitch and terrain distribution used for training.
        if not (0 < eval_pitch_deg <= 45):
            raise ValueError('eval_pitch_deg must be in (0, 45]')
        config=replace(config,attitude_profile='v5015',pitch_command_max_deg=float(eval_pitch_deg))
    configure_jax(config.device)
    import jax
    require_backend(config.device == 'cpu')

    worlds={
        'robust': _screen_world(config,envs=envs,seed_offset=1447,
                         limits=(.80,-.32,.32,2.00),push_n=1.50,
                         sensor_scale=2.0,combo_mix=.35),
        'growth': _screen_world(config,envs=envs,seed_offset=1211,
                         limits=(1.00,-.42,.40,2.40),push_n=.90,
                         sensor_scale=1.5,combo_mix=0.0),
        'growth_combo': _screen_world(config,envs=envs,seed_offset=1319,
                               limits=(1.00,-.42,.40,2.40),push_n=.90,
                               sensor_scale=1.5,combo_mix=1.0),
    }
    panels = dict(SCREEN_PANELS)
    if eval_pitch_deg is not None:
        # Fixed slope-equivalent stress for v5015: one additional compiled
        # evaluation world for every arm, independent of training exposure.
        worlds['gentle_slope'] = _screen_world(config,envs=envs,seed_offset=1555,
            limits=(.65,-.20,.24,1.50),push_n=.80,sensor_scale=1.5,
            combo_mix=.20,terrain_radius=.20)
        panels['gentle_slope'] = ('gentle_slope',1555,
            _screen_episode(.70,.65,.35,.70,(.45,.40,.15),.80))
    key0=jax.random.PRNGKey(EVAL_SEED)
    results: dict[str,dict[str,Any]]={}
    for run_dir in run_dirs:
        ck=run_dir/'checkpoints'/'final.pkl'
        payload=load_checkpoint(ck,validate=True)
        params=payload['state']['params']; norm=payload['obs_norm']
        obs_std=np.asarray(norm['var'])**.5
        out: dict[str,Any]={'policy_step':int(payload['meta']['step']),
            'envs_per_panel':int(envs),'horizon_steps':int(horizon),'checkpoint':str(ck),
            'eval_pitch_deg': float(eval_pitch_deg) if eval_pitch_deg is not None else None}
        for name,(world_name,salt,episode) in panels.items():
            env,trainer=worlds[world_name]
            compact=trainer['evaluate_summary'](
                params,env.params_array(episode),norm['mean'],obs_std,
                jax.random.fold_in(key0,salt),int(horizon))
            vals=np.asarray(jax.device_get(compact),dtype=np.float64)
            out.update({f'screen_{name}_{k}':float(v) for k,v in zip(SCREEN_FIELDS,vals)})
        out['screen_hard_recovery']=float(np.mean([
            out['screen_resilience100_survival'],out['screen_robust70_survival']]))
        out['screen_growth_headroom']=float(np.mean([
            out['screen_growth140_survival'],out['screen_growthcombo140_survival']]))
        (run_dir/'screen_eval.json').write_text(json.dumps(out,indent=2,sort_keys=True),encoding='utf-8')
        results[run_dir.name]=out
    return results


def evaluate_run(run_dir: Path, *, envs: int = 512, horizon: int = 350,
                 device: str | None = None) -> dict[str, Any]:
    run_dir=Path(run_dir).expanduser().resolve()
    return evaluate_runs([run_dir],envs=envs,horizon=horizon,device=device)[run_dir.name]


def screen_main(argv=None) -> int:
    p=argparse.ArgumentParser(description='EMU tournament held-out stress screen')
    p.add_argument('--run-dir',required=True,nargs='+')
    p.add_argument('--envs',type=int,default=512)
    p.add_argument('--horizon',type=int,default=350)
    p.add_argument('--eval-pitch-deg',type=float,default=None)
    p.add_argument('--device',choices=['auto','cpu','cuda','gpu'],default=None)
    a=p.parse_args(argv)
    out=evaluate_runs([Path(x) for x in a.run_dir],envs=a.envs,horizon=a.horizon,device=a.device,eval_pitch_deg=a.eval_pitch_deg)
    print(json.dumps(out,indent=2,sort_keys=True))
    return 0





# Physical command and recovery frontier examination.
import argparse, json, math, time
from pathlib import Path

from .contracts import VERSION, ROBOT_XML
import numpy as np
from .contracts import CONTROL_DT
from .contracts import recovery_outcome_numpy

from .environment import EmuMJXEnv
from .ppo import configure_jax, require_backend
from .ppo import make_trainer
from .checkpoint_io import load_checkpoint
from .results import write_rows_csv
from .contracts import task_param_defaults


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



# Physical signed-direction/contact-course exams and repeatable batch profiles.
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np

from .contracts import OBS_DIM
from .directional import evaluate_trace_arrays


def exam_gravity(angle_deg, direction):
    """Exact tilted-gravity infinite-plane experiment; uphill is world-fixed."""
    if not math.isfinite(angle_deg) or not 0 <= angle_deg <= 5:
        raise ValueError('exam angle must lie in [0,5] degrees')
    uphill = dict(uphill=(1., 0.), downhill=(-1., 0.),
                  cross_left=(0., 1.), cross_right=(0., -1.)).get(direction)
    if uphill is None:
        raise ValueError('invalid slope direction')
    a = math.radians(angle_deg)
    return np.asarray((-9.81*math.sin(a)*uphill[0],
                       -9.81*math.sin(a)*uphill[1],
                       -9.81*math.cos(a)),dtype=np.float32)


def sha256_file(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()


def source_commit():
    try:
        r=subprocess.run(['git','rev-parse','HEAD'],
              cwd=Path(__file__).resolve().parents[1],
              check=True,capture_output=True,text=True,timeout=5)
        return r.stdout.strip()
    except (OSError,subprocess.SubprocessError):
        return 'unknown'


def validate_request(*, run_dir, output, worlds, horizon, settle_frames,
                     angle_deg, direction, speed, lateral, yaw_rate):
    run=Path(run_dir).expanduser().resolve()
    dest=Path(output).expanduser().resolve()
    if not run.is_dir() and not (run.is_file() and run.suffix == '.emupolicy'):
        raise FileNotFoundError(run)
    if dest == run or run in dest.parents or dest in run.parents:
        raise ValueError('output directory must be separate from training directory')
    if dest.exists() and (not dest.is_dir() or any(dest.iterdir())):
        raise FileExistsError('use a new/empty exam output directory')
    if not 2 <= worlds <= 64 or not 30 <= horizon <= 700:
        raise ValueError('bounded exams require 2–64 worlds and 30–700 frames')
    if not 0 <= settle_frames < horizon - 2:
        raise ValueError('invalid settle window')
    if not all(math.isfinite(x) for x in (speed,lateral,yaw_rate)):
        raise ValueError('nonfinite request')
    if abs(speed) > .6 or abs(lateral) > .25 or abs(yaw_rate) > 1.5:
        raise ValueError('command outside bounded legacy-policy exam limits')
    g=exam_gravity(angle_deg,direction)
    return run,dest,g


def evaluate_directional_run(run_dir=None, *, policy_file=None,
                 angle_deg, direction, output, worlds=8,
                 horizon=350, speed=.10, lateral=0., yaw_rate=0.,
                 seed=1701, device='cuda', settle_frames=25, heading_kp=0.0,
                 scenario='steady', turn_rate=.25,
                 gyro_bias=0., gyro_noise=0., gyro_delay_frames=0,
                 contact_slope=False):
    if not math.isfinite(heading_kp) or not 0 <= heading_kp <= 2:
        raise ValueError('heading-kp must lie in [0,2] per second')
    if not all(math.isfinite(v) for v in (gyro_bias,gyro_noise)) or (
        abs(gyro_bias) > .12 or not 0 <= gyro_noise <= .20 or
        not isinstance(gyro_delay_frames, int) or not 0 <= gyro_delay_frames <= 5):
        raise ValueError('gyro realism parameters outside bounded test limits')
    schedule=make_schedule(scenario,int(horizon),speed=speed,lateral=lateral,
                           yaw_rate=yaw_rate,turn_rate=turn_rate)
    if (run_dir is None) == (policy_file is None):
        raise ValueError('specify exactly one of full run directory or exported policy')
    run_dir,output,g=validate_request(
        run_dir=policy_file if policy_file is not None else run_dir,
        output=output,worlds=int(worlds),horizon=int(horizon),
        settle_frames=int(settle_frames),angle_deg=angle_deg,direction=direction,
        speed=speed,lateral=lateral,yaw_rate=yaw_rate)
    from .checkpoint_io import load_checkpoint
    from .environment import EmuMJXEnv
    from .contracts import ROBOT_XML
    from .ppo import make_trainer
    from .ppo import configure_jax, require_backend
    from .training_config import (
        TrainingConfig,PPO_LR,PPO_ENTROPY_COST,PPO_VALUE_COST,ROLLOUT_STEPS)

    if run_dir.is_file():
        # The protected 120M Pitch15 .emupolicy is not a rewindable checkpoint.
        # Validate its physical/actor contracts and use its exported mean/std
        # through exactly the same JAX actor inference and MJX exam.
        from types import SimpleNamespace
        from .viewer_server import load_exported_actor_for_exam
        params, norm, export = load_exported_actor_for_exam(run_dir)
        cfg = SimpleNamespace(
            phase_speed_gain=float(export['control']['phase_speed_gain']),
            attitude_profile=export['training']['attitude_profile'],
            pitch_command_max_deg=float(export['training']['pitch_command_max_deg']))
        actor_std = norm['std']
        artifact_sha = export['artifact_sha256']
        training_step = int(export['source']['step'])
        checkpoint_sha = None
        source_type = 'exported_actor'
        source_training_commit = export['source'].get('git_commit')
        source_actor_sha = export['source'].get('actor_sha256')
    else:
        cfg=TrainingConfig.load(run_dir/'training_config.json')
        checkpoint=run_dir/'checkpoints/final.pkl'
        saved=load_checkpoint(checkpoint,validate=True)
        params=saved['state']['params']
        norm=saved['obs_norm']
        actor_std = np.sqrt(np.asarray(norm['var']))
        artifact_sha = sha256_file(checkpoint)
        training_step = int(saved['meta']['step'])
        checkpoint_sha = artifact_sha
        source_type = 'schema7_checkpoint'
        source_training_commit = (saved['meta'].get('source_provenance') or {}).get('git_commit')
        source_actor_sha = None
    if int(np.asarray(params['pi'][0]['w']).shape[0]) != OBS_DIM:
        raise ValueError('this examiner supports only the 220-input actor contract')
    configure_jax(device)
    import jax
    import jax.numpy as jnp
    require_backend(device=='cpu')
    uphill_axis=dict(uphill=(1.,0.),downhill=(-1.,0.),
                     cross_left=(0.,1.),cross_right=(0.,-1.))[direction]
    env=EmuMJXEnv(
        str(ROBOT_XML),num_envs=int(worlds),seed=int(seed),
        contact_slope_deg=float(angle_deg) if contact_slope else 0.,
        contact_uphill_xy=uphill_axis,
        max_episode_steps=700,push_max_force_n=0.,
        vx_hi_max=.6,vx_lo_min=-.2,vy_max_mps=.25,yaw_max_rad_s=1.5,
        phase_speed_gain=cfg.phase_speed_gain,
        attitude_profile=cfg.attitude_profile,
        pitch_command_max_deg=cfg.pitch_command_max_deg,
        sensor_noise_scale=0.,open_frontier=False,
        combo_mix_frac=0.,combo_probe_frac=0.,field_teach_recovery=False,
    )
    # Shared, nonrandom MJX gravity field. No robot XML modifications.
    actual_gravity=(np.asarray((0.,0.,-9.81),dtype=np.float32)
                    if contact_slope else g)
    if not contact_slope:
        env.mjx_model=env.mjx_model.tree_replace({'opt.gravity': jnp.asarray(g)})
    env.gravity_unit_world=jnp.tile(
        jnp.asarray(actual_gravity/np.linalg.norm(actual_gravity))[None,:],
        (int(worlds),1))
    trainer=make_trainer(
        env,seed=int(seed),lr=PPO_LR,rollout=ROLLOUT_STEPS,
        epochs=1,minibatches=1,entropy_cost=PPO_ENTROPY_COST,
        value_cost=PPO_VALUE_COST)
    task={
        'regime_frac_steady':1.,'regime_frac_transition':0.,
        'regime_frac_resilience':0.,'steady_difficulty':0.,
        'transition_difficulty':0.,'resilience_difficulty':0.,
        'posture_difficulty':0.,'probe_cmd_vx':float(speed),
        'probe_cmd_vy':float(lateral),'probe_cmd_yaw':float(yaw_rate),
        'probe_tilt_rad':0.,'probe_linvel_mps':0.,
        'probe_angvel_rad_s':0.,'probe_joint_rad':0.,
    }
    trace=trainer['evaluate_trace'](
        params,env.params_array(task),
        jnp.asarray(norm['mean']),jnp.asarray(actor_std),
        jax.random.PRNGKey(int(seed)),int(horizon),float(heading_kp),
        None,jnp.asarray(schedule),float(gyro_bias),float(gyro_noise),
        int(gyro_delay_frames))
    frames=jax.device_get(trace)
    evaluated=evaluate_trace_arrays(
        frames,np.tile(g,(int(worlds),1)),dt=.02,
        settle_frames=int(settle_frames))
    report={
        'schema':2,'source_commit':source_commit(),
        'source_artifact_type':source_type,
        'source_artifact_sha256':artifact_sha,
        'source_training_commit':source_training_commit,
        'actor_sha256':source_actor_sha,
        'checkpoint_sha256':checkpoint_sha,
        'checkpoint_training_step':training_step,
        'exam':dict(angle_deg=float(angle_deg),direction=direction,
                    ground_mode='inclined_contact_plane' if contact_slope else 'tilted_gravity',
                    actual_gravity_world=actual_gravity.tolist(),
                    grade_reference_gravity_world=g.tolist(),
                    gravity_world=actual_gravity.tolist(),worlds=int(worlds),
                    horizon=int(horizon),settle_frames=int(settle_frames),
                    speed=float(speed),lateral=float(lateral),
                    yaw_rate=float(yaw_rate),seed=int(seed),device=device,
                    heading_p_enabled=heading_kp>0,heading_kp=float(heading_kp),
                    heading_sensor='body gyro-Z integrated with optional bias/white noise/delay',
                    scenario=scenario,turn_rate=float(turn_rate),
                    gyro_bias_rad_s=float(gyro_bias),
                    gyro_noise_std_rad_s=float(gyro_noise),
                    gyro_delay_frames=int(gyro_delay_frames)),
        'worlds':evaluated,
    }
    output.mkdir(parents=True,exist_ok=True)
    output_json=output/'directional_exam.json'
    output_json.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    ok=sum(bool(r.get('course_pass')) for r in evaluated)
    gait=sum(bool(r.get('walking_pass')) for r in evaluated)
    lines=[
        '# EMU v5016 directional exam','',
        'Frozen actor source ('+source_type+') SHA256: '+artifact_sha,
        'Original training source: '+str(source_training_commit),
        'Original checkpoint SHA256: '+str(checkpoint_sha or 'not included in export'),
        'Exam source: '+report['source_commit'],
        'Incline: '+str(angle_deg)+' deg, '+direction,
        'Ground mode: '+report['exam']['ground_mode']+
            ' (real plane + vertical gravity)' if contact_slope else
            'Ground mode: gravity-equivalent flat plane',
        'Course success: '+str(ok)+'/'+str(worlds),
        'Contact-pair stride candidate success: '+str(gait)+'/'+str(worlds),
        '',
        'The contact-pair gait count is an estimate, not measured ground-reaction force.',
        'Per-servo torque, continuous-rating duty, thermal exposure proxy and shaft work are in each world physical_evidence.',
        '0.638 N m is a nominal rating, not a verified safe duty envelope for the finished robot.',
        'Actuator-force times joint-speed is BEFORE separate MuJoCo joint damping/friction/springs; do not treat it as net motor output or electrical draw.',
        'Thermal exposure is uncalibrated squared torque duty; it does not predict winding temperature.',
        'Contact-pair slip is geometric and spring potential variation does not prove recovered work.',
        'Scenario: '+scenario+'; turn rate: '+str(turn_rate)+' rad/s.',
        'Gyro bias/noise/delay: '+str(gyro_bias)+' / '+
          str(gyro_noise)+' rad/s / '+str(gyro_delay_frames)+' frames.',
        'Heading P gain: '+str(heading_kp)+' /s (0 means OFF).',
        'When ON, only the yaw-rate request delivered to PPO is corrected.',
        'Exam course is ALWAYS scored against original operator commands.',
        'Yaw estimate integrates body gyro with configured sensor imperfections.',
        'Original operator command schedule is not changed by heading P.',
        'No PPO retraining, reward or policy modification was performed.',
        'A rotated gravity field represents infinite uniform ground; not a physical ramp.',
        '',
        '| World | Horizon | Course | Gait | Signed m | Stride pairs | Servo over % | Peak Nm | Drive J/m | Brake latency s |',
        '|---|---|---|---|---:|---:|---:|---:|---:|---:|',
    ]
    def numeric(value, decimals=2):
        return '—' if value is None else f'{float(value):.{decimals}f}'
    for r in evaluated:
        p=r.get('physical_evidence', {})
        maneuvers=[m for m in r.get('maneuvers', [])
                   if m.get('kind') == 'brake_to_stop']
        latency=([m['response_latency_s'] for m in maneuvers
                  if m.get('response_latency_s') is not None] or [None])[0]
        peak=max(p['servo_peak_abs_torque_nm']) if p.get('servo_peak_abs_torque_nm') else None
        over=p.get('servo_all_sample_over_continuous_fraction')
        lines.append('| '+' | '.join([
            str(r['world']),
            'yes' if r['full_horizon_survival'] else 'no',
            'yes' if r.get('course_pass') else 'no',
            'yes' if r.get('walking_pass') else 'not proven',
            numeric(r.get('signed_course_progress_m'),3),
            str(r.get('contact_stride_pairs',0)),
            numeric(over*100 if over is not None else None,1),
            numeric(peak,2),
            numeric(r.get('positive_shaft_work_per_progress_m_j_m'),1),
            numeric(latency,2),
        ])+' |')
    lines+=['','A dash means not measured or not applicable, NOT a zero-load or zero-latency result.',
            'Servo over % is time/sample fraction above nominal continuous rating, not overheating.',
            'Drive J/m is defined only for >0.01 m signed commanded-direction progress.',
            'Per-servo p95/peak/duty/work and full maneuver events are in directional_exam.json.']
    output_md=output/'directional_exam.md'
    output_md.write_text('\n'.join(lines)+'\n')
    return output_json,output_md



def _summarize_worlds(worlds):
    """Keep survival and course/gait obedience distinct; do not merge them."""
    def mean_optional(key):
        values=[float(w[key]) for w in worlds if key in w]
        return sum(values)/len(values) if values else None
    def physical_optional(key):
        values=[float(w['physical_evidence'][key]) for w in worlds
                if w.get('physical_evidence',{}).get(key) is not None]
        return sum(values)/len(values) if values else None
    count=len(worlds)
    physical=[w['physical_evidence'] for w in worlds if 'physical_evidence' in w]
    peak=max((max(p['servo_peak_abs_torque_nm']) for p in physical),default=None)
    return dict(worlds=count,
                physical_worlds=len(physical),
                positive_progress_worlds=sum(
                    w.get('positive_shaft_work_per_progress_m_j_m') is not None
                    for w in worlds),
                servo_over_continuous_frac=physical_optional('servo_all_sample_over_continuous_fraction'),
                servo_peak_abs_torque_nm=peak,
                drive_work_j=physical_optional('step_integrated_drive_shaft_work_j'),
                abs_work_j=physical_optional('step_integrated_abs_shaft_work_j'),
                spring_potential_range_j=physical_optional('spring_potential_range_j'),
                contact_pair_slip_speed_m_s=physical_optional('contact_pair_slip_speed_mean_m_s'),
                drive_work_per_progress_j_m=mean_optional('positive_shaft_work_per_progress_m_j_m'),
                abs_work_per_progress_j_m=mean_optional('absolute_shaft_work_per_progress_m_j_m'),
                survived_full=sum(bool(w['full_horizon_survival']) for w in worlds)/count,
                course_pass=sum(bool(w.get('course_pass')) for w in worlds)/count,
                gait_candidate_pass=sum(bool(w.get('walking_pass')) for w in worlds)/count,
                progress_m=mean_optional('signed_course_progress_m'),
                maneuver_response_s=mean_optional('maneuver_mean_response_s'),
                maneuver_unobserved_count=sum(int(w.get('maneuver_unobserved_count', 0))
                                            for w in worlds),
                braking_path_first_0p5s_m=mean_optional('braking_path_first_0p5s_m'),
                heading_mae_deg=mean_optional('heading_mae_deg'),
                yaw_rate_mae_rad_s=mean_optional('yaw_rate_mae_rad_s'),
                yaw_abs_error_1s=mean_optional('yaw_abs_error_1s'),
                yaw_abs_error_3s=mean_optional('yaw_abs_error_3s'),
                gyro_heading_error_mae_deg=mean_optional('gyro_heading_error_mae_deg'),
                mean_abs_controller_correction_rad_s=
                    mean_optional('mean_abs_controller_correction_rad_s'))


def evaluate_directional_pair(run_dir=None, *, policy_file=None,
                              output, heading_kp=0.75, **kwargs):
    """Two matched read-only exams plus one human-readable delta report.

    The same saved policy, initial seeds, command and slope are evaluated once
    with the outer heading loop OFF, once ON. Never silently select a winner.
    """
    target=Path(output).expanduser().resolve()
    if (run_dir is None) == (policy_file is None):
        raise ValueError('specify either a run directory or exported policy')
    run=Path(policy_file if policy_file is not None else run_dir).expanduser().resolve()
    if target==run or run in target.parents or target in run.parents:
        raise ValueError('comparison output must be outside original run')
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise FileExistsError('comparison requires new/empty --output root')
    if not math.isfinite(heading_kp) or not 0 < heading_kp <= 2.0:
        raise ValueError('compare mode needs a positive heading-kp in (0,2]')
    source_kwargs = {'policy_file': run} if policy_file is not None else {'run_dir': run}
    off_json,_=evaluate_directional_run(output=target/'off',
                            heading_kp=0.0,**source_kwargs,**kwargs)
    on_json,_=evaluate_directional_run(output=target/'on',
                           heading_kp=heading_kp,**source_kwargs,**kwargs)
    off=json.loads(off_json.read_text())
    on=json.loads(on_json.read_text())
    if (off['source_artifact_sha256']!=on['source_artifact_sha256'] or
            off['checkpoint_training_step']!=on['checkpoint_training_step']):
        raise RuntimeError('paired evaluations did not use same checkpoint')
    for key in ('seed','worlds','horizon','speed','lateral','yaw_rate',
                'angle_deg','direction','settle_frames','scenario','turn_rate',
                'gyro_bias_rad_s','gyro_noise_std_rad_s','gyro_delay_frames',
                'ground_mode','actual_gravity_world'): 
        if off['exam'][key]!=on['exam'][key]:
            raise RuntimeError('paired examinations have unequal '+key)
    s_off=_summarize_worlds(off['worlds'])
    s_on=_summarize_worlds(on['worlds'])
    keys=('survived_full','course_pass','gait_candidate_pass','progress_m',
          'heading_mae_deg','yaw_rate_mae_rad_s','yaw_abs_error_1s','yaw_abs_error_3s',
          'gyro_heading_error_mae_deg',
          'mean_abs_controller_correction_rad_s',
          'maneuver_response_s','braking_path_first_0p5s_m',
          'servo_over_continuous_frac','servo_peak_abs_torque_nm',
          'drive_work_j','abs_work_j','spring_potential_range_j',
          'contact_pair_slip_speed_m_s','drive_work_per_progress_j_m',
          'abs_work_per_progress_j_m')
    delta={k: (s_on[k]-s_off[k]) if s_on[k] is not None and s_off[k] is not None else None
           for k in keys}
    result=dict(schema=2,checkpoint_sha256=off['checkpoint_sha256'],
                source_artifact_sha256=off['source_artifact_sha256'],
                source_artifact_type=off['source_artifact_type'],
                examined_at_revision=off['source_commit'],
                protocol={k:off['exam'][k] for k in off['exam'] if k not in
                          ('heading_p_enabled','heading_kp')},
                heading_kp=heading_kp,off=s_off,on=s_on,
                on_minus_off=delta,conclusion='comparison only; no automatic promotion')
    path=target/'heading_comparison.json'
    path.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    lines=['# Paired v5016 heading-hold evaluation','',
           'Source actor SHA256: '+result['source_artifact_sha256'],
           'Source type: '+result['source_artifact_type'],
           'Gain ON: '+str(heading_kp)+'/s. Both cases use same seeds and operator requests.',
           'The controller sees gyro-integrated relative yaw, not simulator true yaw.',
           'Course scoring always uses unmodified operator commands.',
           '',
           '| Measurement | OFF | ON | ON minus OFF |',
           '|---|---:|---:|---:|']
    for key in keys:
        vals=[s_off[key],s_on[key],delta[key]]
        lines.append('| '+key+' | '+' | '.join(
            '—' if x is None else f'{x:.4f}' for x in vals)+' |')
    lines+=['','Fractions are over all examined worlds. Other metrics are means',
            'of valid measured episodes, NOT conditionally matched survivor pairs.',
            'Physical summaries are per-world averages when observed, not pooled duty-weighted samples.',
            'Work per signed progress includes only worlds with >1 cm commanded-direction progress;',
            'inspect the positive_progress_worlds and physical_worlds counts before comparisons.',
            'Review individual world outcomes in off/ and on/ before any decision.',
            'A better heading MAE does not compensate for worse survival or course progress.',
            'This comparison did not retrain or alter the checkpoint.']
    md=target/'heading_comparison.md'
    md.write_text('\n'.join(lines)+'\n')
    return path,md


def directional_main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--run-dir',help='full schema-7 run directory')
    source.add_argument('--policy',help='exported read-only .emupolicy file')
    p.add_argument('--output',required=True)
    p.add_argument('--angle',type=float,default=0.)
    p.add_argument('--direction',choices=('uphill','downhill','cross_left','cross_right'),default='uphill')
    p.add_argument('--worlds',type=int,default=8)
    p.add_argument('--horizon',type=int,default=350)
    p.add_argument('--speed',type=float,default=.10)
    p.add_argument('--lateral',type=float,default=0.)
    p.add_argument('--yaw-rate',type=float,default=0.)
    p.add_argument('--settle-frames',type=int,default=25)
    p.add_argument('--scenario',choices=SCENARIOS,default='steady')
    p.add_argument('--turn-rate',type=float,default=.25)
    p.add_argument('--gyro-bias',type=float,default=0.)
    p.add_argument('--gyro-noise',type=float,default=0.)
    p.add_argument('--gyro-delay-frames',type=int,default=0)
    p.add_argument('--contact-slope',action='store_true',
                   help='rotate actual infinite floor plane; keep real gravity vertical')
    p.add_argument('--seed',type=int,default=1701)
    p.add_argument('--device',choices=('cpu','cuda','gpu','auto'),default='cuda')
    p.add_argument('--heading-kp',type=float,default=0.75,
                   help='P gain per second, used for ON/compare; default OFF unless selected')
    p.add_argument('--compare-heading',action='store_true',
                   help='evaluate matched OFF and ON rollouts into one separate bundle')
    a=p.parse_args(argv)
    kw=dict(angle_deg=a.angle,direction=a.direction,worlds=a.worlds,
            horizon=a.horizon,speed=a.speed,lateral=a.lateral,
            yaw_rate=a.yaw_rate,seed=a.seed,device=a.device,
            settle_frames=a.settle_frames,scenario=a.scenario,
            turn_rate=a.turn_rate,gyro_bias=a.gyro_bias,
            gyro_noise=a.gyro_noise,gyro_delay_frames=a.gyro_delay_frames,
            contact_slope=a.contact_slope)
    if a.compare_heading:
        outputs=evaluate_directional_pair(
                              run_dir=a.run_dir,policy_file=a.policy,
                              output=a.output,heading_kp=a.heading_kp,**kw)
    else:
        # Single-case invocation deliberately remains controller OFF by default.
        outputs=evaluate_directional_run(
                             run_dir=a.run_dir,policy_file=a.policy,
                             output=a.output,heading_kp=0.0,**kw)
    for path in outputs:
        print(path)
    return 0



# Operator-command courses: unmodified requests, never robot-derived.
import math
import numpy as np

SCENARIOS = ('steady', 'turn_stop_left', 'turn_stop_right',
             'sidestep_left', 'sidestep_right', 'reverse', 'stop_resume',
             'rest_start', 'speed_step', 'brake_stop')


def make_schedule(scenario, horizon, *, speed=.10, lateral=.06,
                  yaw_rate=0., turn_rate=.25):
    if scenario not in SCENARIOS:
        raise ValueError('unknown scenario: '+str(scenario))
    if not 30 <= int(horizon) <= 700:
        raise ValueError('schedule horizon must be between 30 and 700 frames')
    if not all(math.isfinite(x) for x in (speed, lateral, yaw_rate, turn_rate)):
        raise ValueError('commands must be finite')
    if abs(speed) > .6 or abs(lateral) > .25 or abs(yaw_rate) > 1.5 or not 0 < turn_rate <= 1.5:
        raise ValueError('command outside safe test envelope')
    n = int(horizon)
    cmds = np.tile(np.array([speed, 0., yaw_rate], dtype=np.float32), (n, 1))
    a = n//4
    b = n*2//3
    if scenario in ('turn_stop_left','turn_stop_right'):
        cmds[:,2] = 0.
        cmds[a:b,2] = turn_rate if scenario.endswith('left') else -turn_rate
    elif scenario in ('sidestep_left','sidestep_right'):
        cmds[:,2] = 0.
        cmds[a:b,0] = 0.
        cmds[a:b,1] = lateral if scenario.endswith('left') else -lateral
    elif scenario == 'reverse':
        cmds[:,2] = 0.
        cmds[a:b,0] = -abs(speed)
    elif scenario == 'stop_resume':
        cmds[:,2] = 0.
        cmds[a:b,0] = 0.
    elif scenario == 'rest_start':
        cmds[:,2] = 0.
        cmds[:a,0] = 0.
    elif scenario == 'speed_step':
        cmds[:,2] = 0.
        cmds[:a,0] = .4 * speed
    elif scenario == 'brake_stop':
        cmds[:,2] = 0.
        cmds[b:,0] = 0.
    return cmds


# Matched multi-branch GPU examination plans use the same physical examiner.

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
import zipfile


ARMS=('baseline','shadow','slope','pitch15')


def cases(profile):
    if profile=='physical':
        return [('champion', label, angle, direction, scenario, 350)
                for label,angle,direction,scenario in (
                    ('flat_steady',0.,'uphill','steady'),
                    ('rest_start',0.,'uphill','rest_start'),
                    ('accelerate',0.,'uphill','speed_step'),
                    ('brake_stop',0.,'uphill','brake_stop'),
                    ('stop_resume',0.,'uphill','stop_resume'),
                    ('reverse',0.,'uphill','reverse'),
                    ('turn_left',0.,'uphill','turn_stop_left'),
                    ('turn_right',0.,'uphill','turn_stop_right'),
                    ('sidestep_left',0.,'uphill','sidestep_left'),
                    ('sidestep_right',0.,'uphill','sidestep_right'),
                    # Unlike the legacy gravity-equivalent tests, these
                    # rotate the actual static collision plane. Do not
                    # mistake them for per-world terrain randomization.
                    ('contact_up_2deg',2.,'uphill','steady'),
                    ('contact_down_2deg',2.,'downhill','steady'),
                    ('contact_cross_2deg',2.,'cross_left','steady'))]
    if profile=='smoke':
        return [('baseline','flat',0.,'uphill','steady',40),
                ('baseline','turn_stop',0.,'uphill','turn_stop_left',160)]
    if profile=='pilot':
        tasks=[('turn_left',0.,'uphill','turn_stop_left'),
               ('turn_right',0.,'uphill','turn_stop_right'),
               ('up_025',.25,'uphill','steady'),
               ('up_050',.5,'uphill','steady'),
               ('cross_050',.5,'cross_left','steady')]
        return [(arm,name,angle,direction,scenario,350)
                for arm in ARMS for name,angle,direction,scenario in tasks]
    if profile=='sensor':
        return [(arm,name,angle,direction,scenario,350)
                for arm in ('baseline','slope')
                for name,angle,direction,scenario in (
                    ('noisy_turn',0.,'uphill','turn_stop_left'),
                    ('noisy_up_025',.25,'uphill','steady'))]
    if profile=='stretch':
        return [(arm,'up_100',1.,'uphill','steady',350) for arm in ARMS]
    raise ValueError('unknown profile')


def plan(profile,root,output,*,worlds=None,seed=1701,kp=.75,device='cuda',
         policy_file=None):
    if policy_file is not None:
        if root is not None or profile != 'physical':
            raise ValueError('exported policy requires physical profile and no run root')
        source=Path(policy_file).expanduser().resolve()
        if not source.is_file() or source.suffix != '.emupolicy':
            raise FileNotFoundError(source)
    else:
        if root is None or profile == 'physical':
            raise ValueError('specify training run root, or --policy for physical profile')
        source=Path(root).expanduser().resolve()
    dest=Path(output).expanduser().resolve()
    if source==dest or source in dest.parents or dest in source.parents:
        raise ValueError('outputs and training sources must be separate')
    n=int(worlds or (2 if profile=='smoke' else 4 if profile=='physical' else 8))
    if not 2<=n<=64 or not math.isfinite(kp) or not 0<kp<=2:
        raise ValueError('bounded worlds/gain required')
    bias,noise,delay=(.015,.02,2) if profile=='sensor' else (0.,0.,0)
    return [dict(arm=arm,label=label,angle=angle,direction=direction,
          scenario=scenario,horizon=horizon,worlds=n,seed=int(seed),
          heading_kp=kp,device=device,gyro_bias=bias,gyro_noise=noise,
          gyro_delay_frames=delay,
          run_dir=str(source if policy_file is not None else source/'round1'/arm),
          is_exported_policy=policy_file is not None,
          contact_slope=profile=='physical' and angle>0,
          output=str(dest/profile/arm/label))
          for arm,label,angle,direction,scenario,horizon in cases(profile)]


def verify_existing(path,job,revision,checkpoint_hash):
    if not path.exists():
        return False
    info=json.loads(path.read_text())
    source_hash=info.get('source_artifact_sha256',info.get('checkpoint_sha256'))
    if source_hash!=checkpoint_hash or (
        info.get('examined_at_revision')!=revision):
        raise ValueError('stale source or trained checkpoint in '+str(path))
    exam=info.get('protocol',{})
    wanted=dict(angle_deg=job['angle'],direction=job['direction'],
        scenario=job['scenario'],worlds=job['worlds'],horizon=job['horizon'],
        ground_mode='inclined_contact_plane' if job.get('contact_slope') else 'tilted_gravity',
        seed=job['seed'],gyro_bias_rad_s=job['gyro_bias'],
        gyro_noise_std_rad_s=job['gyro_noise'],
        gyro_delay_frames=job['gyro_delay_frames'])
    if any(exam.get(k)!=v for k,v in wanted.items()) or (
        info.get('heading_kp')!=job['heading_kp']):
        raise ValueError('stale/changed exam parameters: '+str(path))
    if not all((path.parent/mode/'directional_exam.json').is_file()
               for mode in ('off','on')):
        raise ValueError('incomplete paired evidence: '+str(path))
    return True


def summarize(jobs,profile_root,revision):
    rows=[]
    for job in jobs:
        p=Path(job['output'])/'heading_comparison.json'
        if p.is_file():
            data=json.loads(p.read_text())
            rows.append(dict(arm=job['arm'],case=job['label'],angle_deg=job['angle'],
                 checkpoint_sha256=data['checkpoint_sha256'],
                 protocol=data['protocol'],off=data['off'],on=data['on'],
                 on_minus_off=data['on_minus_off']))
    summary=dict(schema=1,profile=profile_root.name,source_commit=revision,
                 expected=len(jobs),completed=len(rows),results=rows)
    (profile_root/'results_data.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# v5016 '+profile_root.name+' — matched heading and terrain tests','',
        'Source commit: '+revision,
        str(len(rows))+'/'+str(len(jobs))+' cases complete.',
        'Same saved checkpoint and operator request for each ON/OFF pair.',
        'No PPO training or promotion performed. Gait contact flags are estimates.',
        'Servo duty is relative to nominal 0.638 Nm; shaft work is not electrical energy.',
        'Work per metre is conditional on positive signed course progress; null is not zero.',
        '',
        '| Policy | Case | OFF survival | ON survival | OFF course | ON course | OFF over-rated % | ON over-rated % | OFF J/m | ON J/m |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    def v(x,key,percent=False):
        value=x.get(key)
        if value is None:
            return '—'
        return format(100*value if percent else value,'.1f')+('%' if percent else '')
    for row in rows:
        a,b=row['off'],row['on']
        lines.append('| '+' | '.join((row['arm'],row['case'],
            v(a,'survived_full',True),v(b,'survived_full',True),
            v(a,'course_pass',True),v(b,'course_pass',True),
            v(a,'servo_over_continuous_frac',True),
            v(b,'servo_over_continuous_frac',True),
            v(a,'drive_work_per_progress_j_m'),
            v(b,'drive_work_per_progress_j_m')))+' |')
    lines+=['','Higher survival and actual requested-course progress must accompany',
            'any claimed heading benefit. Inspect the raw paired episodes.']
    (profile_root/'results.md').write_text('\n'.join(lines)+'\n')
    with (profile_root/'results.csv').open('w',newline='') as file:
        fields=('arm','case','mode','survived_full','course_pass',
                'gait_candidate_pass','progress_m','heading_mae_deg',
                'yaw_abs_error_1s','gyro_heading_error_mae_deg',
                'maneuver_response_s','maneuver_unobserved_count',
                'braking_path_first_0p5s_m',
                'physical_worlds','positive_progress_worlds',
                'servo_over_continuous_frac','servo_peak_abs_torque_nm',
                'drive_work_j','abs_work_j','spring_potential_range_j',
                'contact_pair_slip_speed_m_s','drive_work_per_progress_j_m',
                'abs_work_per_progress_j_m')
        w=csv.DictWriter(file,fieldnames=fields)
        w.writeheader()
        for r in rows:
            for mode in ('off','on'):
                item=r[mode]
                w.writerow(dict(arm=r['arm'],case=r['case'],mode=mode,
                    **{k:item.get(k) for k in fields[3:]}))
    temp=profile_root/'results_bundle.zip.tmp'
    bundle=profile_root/'results_bundle.zip'
    with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in ('results.md','results.csv','results_data.json'):
            z.write(profile_root/p,arcname=p)
        for job in jobs:
            base=Path(job['output'])
            for p in ('heading_comparison.md','heading_comparison.json',
                      'off/directional_exam.md','off/directional_exam.json',
                      'on/directional_exam.md','on/directional_exam.json'):
                f=base/p
                if f.is_file() and not f.is_symlink():
                    z.write(f,arcname=job['arm']+'/'+job['label']+'/'+p)
    temp.replace(bundle)
    return bundle,len(rows)


def execute(jobs,revision,*,actually_run=False):
    if not jobs:
        raise ValueError('no experiment cases')
    profile_root=Path(jobs[0]['output']).parents[1]
    for i,j in enumerate(jobs,1):
        print(f"{i:02d}/{len(jobs)} {j['arm']} {j['label']} "
              f"{j['angle']}deg {j['worlds']} worlds x {j['horizon']} steps "
              f"bias={j['gyro_bias']} noise={j['gyro_noise']} delay={j['gyro_delay_frames']}",
              flush=True)
    if not actually_run:
        print('PLAN ONLY. Add --run to execute (saved training is read-only).')
        return
    hashes={}
    for j in jobs:
        source=Path(j['run_dir'])
        artifact=source if j.get('is_exported_policy') else source/'checkpoints/final.pkl'
        if not artifact.is_file():
            raise FileNotFoundError(str(artifact))
        if j['arm'] not in hashes:
            hashes[j['arm']]=sha256_file(artifact)
    profile_root.mkdir(parents=True,exist_ok=True)
    for job in jobs:
        dest=Path(job['output'])
        report=dest/'heading_comparison.json'
        if verify_existing(report,job,revision,hashes[job['arm']]):
            print('Already validated: '+str(dest),flush=True)
            continue
        if dest.exists() and any(dest.iterdir()):
            raise FileExistsError('partial exam: inspect before retry '+str(dest))
        cmd=[sys.executable,'-m','emu.cli','directional-exam',
             '--policy' if job.get('is_exported_policy') else '--run-dir',
             job['run_dir'],
             '--output',str(dest),'--angle',str(job['angle']),
             '--direction',job['direction'],'--scenario',job['scenario'],
             '--worlds',str(job['worlds']),'--horizon',str(job['horizon']),
             '--seed',str(job['seed']),'--device',job['device'],
             '--heading-kp',str(job['heading_kp']),'--compare-heading',
             '--gyro-bias',str(job['gyro_bias']),
             '--gyro-noise',str(job['gyro_noise']),
             '--gyro-delay-frames',str(job['gyro_delay_frames']),
             *(['--contact-slope'] if job.get('contact_slope') else [])]
        subprocess.run(cmd,check=True)
        if not verify_existing(report,job,revision,hashes[job['arm']]):
            raise RuntimeError('no completed exam at '+str(report))
        summarize(jobs,profile_root,revision)
    zip_path,n=summarize(jobs,profile_root,revision)
    print('Completed '+str(n)+'/'+str(len(jobs))+'; upload '+str(zip_path))


def batch_main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--root',help='existing tournament run root')
    source.add_argument('--policy',help='frozen exported actor for physical profile')
    p.add_argument('--output',required=True)
    p.add_argument('--profile',choices=('smoke','pilot','sensor','stretch','physical'),default='smoke')
    p.add_argument('--worlds',type=int)
    p.add_argument('--seed',type=int,default=1701)
    p.add_argument('--heading-kp',type=float,default=.75)
    p.add_argument('--device',choices=('auto','cpu','cuda','gpu'),default='cuda')
    p.add_argument('--run',action='store_true')
    a=p.parse_args(argv)
    jobs=plan(a.profile,a.root,a.output,worlds=a.worlds,seed=a.seed,
              kp=a.heading_kp,device=a.device,policy_file=a.policy)
    execute(jobs,source_commit(),actually_run=a.run)
    return 0

if __name__ == '__main__':
    raise SystemExit(screen_main())
