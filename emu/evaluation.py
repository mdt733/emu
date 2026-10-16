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




if __name__ == '__main__':
    raise SystemExit(screen_main())
