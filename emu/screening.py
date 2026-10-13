"""Cheap held-out stress screen used by the v5013 tournament.

These probes are deliberately *not* curriculum inputs and are not part of the broad
score.  They are a longer ruler used only at tournament decision points so weak arms
can be culled before spending another 30--70M environment steps.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .checkpoint_io import load_checkpoint
from .contracts import task_param_defaults
from .environment import EmuMJXEnv
from .paths import ROBOT_XML
from .ppo import make_trainer
from .runtime import configure_jax, require_backend
from .training_config import (
    TrainingConfig, MAX_EPISODE_STEPS, PPO_LR, ROLLOUT_STEPS,
    PPO_ENTROPY_COST, PPO_VALUE_COST, EVAL_SEED,
)

SCREEN_FIELDS = ('survival', 'first_fall_step_mean', 'vx_mae', 'vy_mae', 'yaw_mae',
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


SCREEN_PANELS = {
    # Same conceptual coordinates that exposed the v5012 graph-vs-shadow trade.
    # Current v5013 hardware doctrine keeps its one-frame target delay; these are
    # therefore regression-style stress probes, not claims of physical limits.
    'resilience100': ('robust', 1103, _episode(.60, .65, 1.00, .65, (.08, .17, .75), .90)),
    'robust70': ('robust', 1207, _episode(.70, .78, .75, .60, (.20, .35, .45), .90)),
    'growth140': ('growth', 1301, _episode(1.0, .92, .35, .45, (.50, .38, .12), 1.0)),
    'growthcombo140': ('growth_combo', 1409, _episode(1.0, .92, .35, .45, (.50, .38, .12), 1.0)),
}


def _world(config: TrainingConfig, *, envs: int, seed_offset: int,
           limits: tuple[float, float, float, float], push_n: float,
           sensor_scale: float, combo_mix: float, terrain_radius: float = 0.0):
    vx_hi, vx_lo, vy, yaw = limits
    seed = EVAL_SEED + seed_offset
    from .domain_randomization import DomainRandomizationConfig
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
        'robust': _world(config,envs=envs,seed_offset=1447,
                         limits=(.80,-.32,.32,2.00),push_n=1.50,
                         sensor_scale=2.0,combo_mix=.35),
        'growth': _world(config,envs=envs,seed_offset=1211,
                         limits=(1.00,-.42,.40,2.40),push_n=.90,
                         sensor_scale=1.5,combo_mix=0.0),
        'growth_combo': _world(config,envs=envs,seed_offset=1319,
                               limits=(1.00,-.42,.40,2.40),push_n=.90,
                               sensor_scale=1.5,combo_mix=1.0),
    }
    panels = dict(SCREEN_PANELS)
    if eval_pitch_deg is not None:
        # Fixed slope-equivalent stress for v5015: one additional compiled
        # evaluation world for every arm, independent of training exposure.
        worlds['gentle_slope'] = _world(config,envs=envs,seed_offset=1555,
            limits=(.65,-.20,.24,1.50),push_n=.80,sensor_scale=1.5,
            combo_mix=.20,terrain_radius=.20)
        panels['gentle_slope'] = ('gentle_slope',1555,
            _episode(.70,.65,.35,.70,(.45,.40,.15),.80))
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


def main(argv=None) -> int:
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
    raise SystemExit(main())
