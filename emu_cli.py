#!/usr/bin/env python3
"""Small public CLI for EMU.

The research scripts remain available, but this is the intended front door for
normal use and for people who did not live through v5002-v5010 archaeology.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
VERSION = (HERE / 'VERSION').read_text(encoding='utf-8').strip()
DEFAULT_RUN_ROOT = Path.home() / 'robot_lab' / 'emu_runs' / VERSION

def _run_root_for_seed(seed: int) -> Path:
    return DEFAULT_RUN_ROOT if int(seed)==733 else Path.home()/'robot_lab'/'emu_runs'/f'{VERSION}_seed{int(seed)}'


def _runner(extra: list[str]) -> int:
    return subprocess.call([sys.executable, str(HERE/'run_ladder.py'), *extra], cwd=HERE)


def _recommended_train_args(args) -> list[str]:
    if int(args.steps) < 20_000_000:
        raise SystemExit('--steps must be at least 20,000,000; the cold nursery itself is 20M')
    out = ['--stage','trainer','--only','trainer_graph','--no-trainer-confirm-zero',
           '--seed',str(args.seed),'--trainer-target-steps',str(args.steps),
           '--device',args.device]
    if args.run_root:
        out += ['--run-root',str(Path(args.run_root).expanduser())]
    elif int(args.seed) != 733:
        out += ['--run-root',str(Path.home()/'robot_lab'/'emu_runs'/f'{VERSION}_seed{int(args.seed)}')]
    if args.results_dir:
        out += ['--results-dir',str(Path(args.results_dir).expanduser())]
    elif int(args.seed) != 733:
        out += ['--results-dir',str(Path.home()/'robot_lab'/'emu'/'results'/f'{VERSION}_seed{int(args.seed)}')]
    if args.envs:
        out += ['--envs',str(args.envs)]
    return out


def _find_checkpoint(run_root: Path) -> Path | None:
    rr = run_root / 'rung_runs'
    if not rr.exists(): return None
    # Semantic priority beats mtime: a terminal final checkpoint is preferred
    # over a newer intermediate/latest file from another branch/process.
    for name in ('final.pkl','best_eval.pkl','latest.pkl','latest_mjx.pkl'):
        candidates=[p for p in rr.glob(f'*/checkpoints/{name}') if p.is_file()]
        if candidates:
            return max(candidates,key=lambda p:p.stat().st_mtime)
    return None


def doctor(args) -> int:
    print(f'EMU {VERSION} doctor')
    print(f'Python: {sys.version.split()[0]}')
    failures=[]
    try:
        import numpy as np
        print(f'NumPy: {np.__version__}')
    except Exception as exc:
        failures.append(f'NumPy import: {exc}')
    try:
        # A diagnostic reset/step does not need JAX's large default VRAM
        # reservation; disabling preallocation here avoids scary-but-recoverable
        # OOM spam on display GPUs. This process-local setting does not affect training.
        os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
        import jax
        print(f'JAX: {jax.__version__}  backend={jax.default_backend()}  devices={jax.devices()}')
        if args.require_gpu and jax.default_backend() not in ('gpu','cuda'):
            failures.append('GPU required but JAX is not using a GPU backend')
    except Exception as exc:
        failures.append(f'JAX import/backend: {exc}')
    try:
        import mujoco
        m=mujoco.MjModel.from_xml_path(str(HERE/'robot.xml'))
        print(f'MuJoCo: {mujoco.__version__}  robot nq={m.nq} nv={m.nv} nu={m.nu}')
    except Exception as exc:
        failures.append(f'MuJoCo/XML: {exc}')
    try:
        import numpy as np
        from capability_field import CapabilityField, FIELD_VERSION
        f=CapabilityField(); T,N=96,8
        z=np.zeros((T,N),np.float32); o=np.ones((T,N),np.float32)
        age=(np.arange(T,dtype=np.float32)[:,None]+1)*.02+z
        metrics={'cmd_vx':z+.1,'cmd_vy':z,'cmd_yaw':z,'target_cmd_vx':z+.1,'target_cmd_vy':z,'target_cmd_yaw':z,
                 'transition_ramp_speed':z,'vx':z+.1,'vy':z,'yaw_rate':z,'gravity_z':o,'support':o*.3,'bad_contact':z,
                 'episode_fell':z,'episode_done':z,'age_s':age,'regime_resilience':z,'birth_resilience_d':z,'transition_frac':z,'cmd_resampled':z}
        d=f.update_from_rollout(metrics,step=65536)
        assert FIELD_VERSION==3 and d['field_node_events_update']>0
        CapabilityField.from_state_dict(f.state_dict())
        print(f'Capability field schema {FIELD_VERSION}: PASS ({int(d["field_node_events_update"])} semantic node events)')
    except Exception as exc:
        failures.append(f'Capability field: {exc}')
    if not args.no_mjx_smoke:
        try:
            import jax.numpy as jnp
            from environment import EmuMJXEnv
            from training import RegimeGovernor
            e=EmuMJXEnv(str(HERE/'robot.xml'),num_envs=8,seed=7,max_episode_steps=40,
                        actuator_realism=True,comm_latency_steps=1,disable_target_slew=True)
            p=RegimeGovernor().params_dict(); st,_=e.reset(p)
            out=e.step(st,jnp.zeros((8,6),dtype=jnp.float32),p)
            out.obs.block_until_ready()
            print(f'MJX reset/step: PASS obs={tuple(out.obs.shape)}')
        except Exception as exc:
            failures.append(f'MJX reset/step: {exc}')
    if failures:
        print('\nNOT READY:')
        for x in failures: print('  -',x)
        return 1
    print('\nReady to train.')
    return 0


def main(argv=None) -> int:
    p=argparse.ArgumentParser(prog='emu',description=f'EMU {VERSION} — self-mapping biped trainer')
    p.add_argument('--version',action='version',version=VERSION)
    sub=p.add_subparsers(dest='cmd',required=True)

    def training_parser(name,help_text):
        q=sub.add_parser(name,help=help_text)
        q.add_argument('--seed',type=int,default=733)
        q.add_argument('--steps',type=int,default=120_000_000,help='absolute training-step target (default 120M)')
        q.add_argument('--device',choices=['cuda','gpu','cpu','auto'],default='cuda')
        q.add_argument('--envs',type=int,default=0,help='override parallel environments; 0 keeps recommended 2048')
        q.add_argument('--run-root',default='')
        q.add_argument('--results-dir',default='')
        return q

    training_parser('train','cold-start the current recommended trainer')
    training_parser('resume','resume/reuse the current recommended run; safe to rerun after interruption')

    q=sub.add_parser('view',help='open the live read-only capability-field viewer')
    q.add_argument('--run-dir',default=''); q.add_argument('--run-root',default=''); q.add_argument('--seed',type=int,default=733); q.add_argument('--port',type=int,default=8765)

    q=sub.add_parser('evaluate',help='run the physical atlas for a checkpoint')
    q.add_argument('--checkpoint',default='',help='defaults to final, then best, then latest checkpoint')
    q.add_argument('--run-root',default=''); q.add_argument('--seed',type=int,default=733); q.add_argument('--device',choices=['cuda','gpu','cpu','auto'],default='cuda')
    q.add_argument('--envs',type=int,default=1024); q.add_argument('--horizon',type=int,default=300)

    q=sub.add_parser('doctor',help='validate Python, JAX/MJX, robot model and capability-field plumbing')
    q.add_argument('--require-gpu',action=argparse.BooleanOptionalAction,default=True)
    q.add_argument('--no-mjx-smoke',action='store_true')

    q=sub.add_parser('benchmark',help='run the recommended trainer from two independent cold-start seeds')
    q.add_argument('--seeds',default='733,1742'); q.add_argument('--steps',type=int,default=120_000_000)
    q.add_argument('--device',choices=['cuda','gpu','cpu','auto'],default='cuda')

    q=sub.add_parser('compare',help='run matched shadow-vs-graph trainer arms from exact 20M foundations')
    q.add_argument('--seeds',default='733,1742'); q.add_argument('--steps',type=int,default=120_000_000)
    q.add_argument('--device',choices=['cuda','gpu','cpu','auto'],default='cuda')

    q=sub.add_parser('experiment',help='pass advanced arguments directly to run_ladder.py')
    q.add_argument('args',nargs=argparse.REMAINDER)

    args=p.parse_args(argv)
    if args.cmd in ('train','resume'):
        if args.cmd == 'resume':
            root=Path(args.run_root).expanduser() if args.run_root else _run_root_for_seed(args.seed)
            checkpoints=list((root/'rung_runs').glob('*/checkpoints/latest_mjx.pkl')) if (root/'rung_runs').exists() else []
            if not checkpoints:
                raise SystemExit(f'No resumable run found under {root}; use `emu train` for a new cold start.')
        return _runner(_recommended_train_args(args))
    if args.cmd=='view':
        from field_viewer import main as view_main
        av=['--port',str(args.port)]
        if args.run_dir: av += ['--run-dir',args.run_dir]
        else: av += ['--run-root',str(Path(args.run_root).expanduser() if args.run_root else _run_root_for_seed(args.seed))]
        return int(view_main(av) or 0)
    if args.cmd=='doctor': return doctor(args)
    if args.cmd=='evaluate':
        root=Path(args.run_root).expanduser() if args.run_root else _run_root_for_seed(args.seed)
        ck=Path(args.checkpoint).expanduser() if args.checkpoint else _find_checkpoint(root)
        if ck is None or not ck.exists():
            raise SystemExit(f'No checkpoint found. Pass --checkpoint or train under {root}.')
        return _runner(['--stage','atlas','--start','checkpoint','--checkpoint',str(ck),'--device',args.device,
                        '--atlas-envs',str(args.envs),'--atlas-horizon',str(args.horizon)])
    if args.cmd=='compare':
        seeds=[int(x.strip()) for x in args.seeds.split(',') if x.strip()]
        if not seeds: raise SystemExit('--seeds must contain at least one integer')
        for seed in seeds:
            run_root=Path.home()/'robot_lab'/'emu_runs'/f'{VERSION}_compare_seed{seed}'
            results_dir=Path.home()/'robot_lab'/'emu'/'results'/f'{VERSION}_compare_seed{seed}'
            rc=_runner(['--stage','trainer','--only','trainer_shadow,trainer_graph','--no-trainer-confirm-zero',
                        '--seed',str(seed),'--trainer-target-steps',str(args.steps),'--device',args.device,
                        '--run-root',str(run_root),'--results-dir',str(results_dir)])
            if rc: return rc
        return 0
    if args.cmd=='benchmark':
        seeds=[int(x.strip()) for x in args.seeds.split(',') if x.strip()]
        if not seeds: raise SystemExit('--seeds must contain at least one integer')
        rc=0
        for seed in seeds:
            run_root=Path.home()/'robot_lab'/'emu_runs'/f'{VERSION}_benchmark_seed{seed}'
            results_dir=Path.home()/'robot_lab'/'emu'/'results'/f'{VERSION}_benchmark_seed{seed}'
            rc=_runner(['--stage','trainer','--only','trainer_graph','--no-trainer-confirm-zero',
                        '--seed',str(seed),'--trainer-target-steps',str(args.steps),'--device',args.device,
                        '--run-root',str(run_root),'--results-dir',str(results_dir)])
            if rc: return rc
        return rc
    if args.cmd=='experiment': return _runner(args.args)
    return 2

if __name__=='__main__':
    raise SystemExit(main())
