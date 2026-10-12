#!/usr/bin/env python3
"""EMU physical capability / recovery atlas (versioned by VERSION file).

This is deliberately not a single-number benchmark. It sweeps physically
interpretable command and recovery disturbances, writes the complete curves,
and reports 95/80/50% competence frontiers. Non-vx command sweeps retain a
+0.08 m/s forward baseline, so vy/yaw frontiers are explicitly conditional on
slow forward locomotion rather than static standing. This atlas is the primary
answer to "what can this policy actually recover from?".
"""
from __future__ import annotations

import argparse, json, math, time
from pathlib import Path

from .paths import VERSION, ROBOT_XML
import numpy as np
from .contracts import CONTROL_DT
from .tracking_contract import recovery_outcome_numpy

from .environment import EmuMJXEnv
from .runtime import configure_jax, require_backend
from .ppo import make_trainer
from .checkpoint_io import load_checkpoint
from .artifacts import write_rows_csv
from .contracts import task_param_defaults


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


def main(argv=None):
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

    env=EmuMJXEnv(str(ROBOT_XML), num_envs=args.envs, seed=args.seed,
                  max_episode_steps=max(args.horizon+50, 700), push_max_force_n=0.70, vx_hi_max=1.8, vx_lo_min=-0.9,
                  vy_max_mps=0.70, yaw_max_rad_s=4.5,
                  phase_speed_gain=1.25, sensor_noise_scale=1.5,
                  open_frontier=False, combo_mix_frac=0.0)
    tr=make_trainer(env, seed=args.seed, lr=2e-4, rollout=32, epochs=1, minibatches=1)
    obs_mean=jax.numpy.asarray(norm.get('mean', np.zeros((tr['obs_dim'],),np.float32)), dtype=jax.numpy.float32)
    obs_std=jax.numpy.asarray(np.sqrt(norm.get('var', np.ones((tr['obs_dim'],),np.float32))), dtype=jax.numpy.float32)

    eval_fields=('survival','first_fall_step_mean','vx_mae','vy_mae','yaw_mae','gravity_z','foot_mid_cone','step_evidence','support')
    rows=[]; summary={'checkpoint':str(ck),'envs':args.envs,'horizon_steps':args.horizon,'control_dt_s':CONTROL_DT,'commands':{},'recovery':{}}
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
    md += ['', 'Recovery means the robot remains unfallen and holds a near-upright, low-roll/pitch-rate, command-tracking, supported state for 0.20s continuously. Recovery time distributions are retained for every point; survival by itself is not treated as recovery.',
           '', f"Wall time: {summary['wall_seconds']:.1f}s", '']
    (out/'atlas.md').write_text('\n'.join(md),encoding='utf-8')
    print(f"[atlas] complete -> {out/'atlas.md'}",flush=True)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
