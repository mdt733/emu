#!/usr/bin/env python3
"""Public command line for modern EMU."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

from .paths import VERSION, ROBOT_XML, seed_root
from .checkpoints import find_evaluation


def _run(module: str, *args: str) -> int:
    return subprocess.call([sys.executable, '-m', f'emu.{module}', *args])


def doctor(args) -> int:
    print(f'EMU {VERSION} doctor')
    print(f'Python: {sys.version.split()[0]}')
    failures=[]
    try:
        import numpy as np; print(f'NumPy: {np.__version__}')
    except Exception as exc: failures.append(f'NumPy: {exc}')
    try:
        os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
        import jax; print(f'JAX: {jax.__version__}  backend={jax.default_backend()}  devices={jax.devices()}')
        if args.require_gpu and jax.default_backend() not in ('gpu','cuda'): failures.append('GPU required but JAX is not using GPU')
    except Exception as exc: failures.append(f'JAX: {exc}')
    try:
        import mujoco
        m=mujoco.MjModel.from_xml_path(str(ROBOT_XML))
        print(f'MuJoCo: {mujoco.__version__}  robot nq={m.nq} nv={m.nv} nu={m.nu}')
    except Exception as exc: failures.append(f'MuJoCo/XML: {exc}')
    try:
        import numpy as np
        from .capability_field import CapabilityField, FIELD_VERSION
        f=CapabilityField(); T,N=96,8; z=np.zeros((T,N),np.float32); o=np.ones((T,N),np.float32)
        age=(np.arange(T,dtype=np.float32)[:,None]+1)*.02+z
        m={'cmd_vx':z+.1,'cmd_vy':z,'cmd_yaw':z,'target_cmd_vx':z+.1,'target_cmd_vy':z,'target_cmd_yaw':z,
           'transition_ramp_speed':z,'vx':z+.1,'vy':z,'yaw_rate':z,'gravity_z':o,'support':o*.3,'bad_contact':z,
           'episode_fell':z,'episode_done':z,'age_s':age,'regime_resilience':z,'birth_resilience_d':z,'transition_frac':z,'cmd_resampled':z}
        d=f.update_from_rollout(m,step=65536); CapabilityField.from_state_dict(f.state_dict())
        if d['field_node_events_update']<=0: raise RuntimeError('no semantic node event')
        print(f'Capability field schema {FIELD_VERSION}: PASS')
    except Exception as exc: failures.append(f'Capability field: {exc}')
    if not args.no_mjx_smoke:
        try:
            import jax.numpy as jnp
            from .environment import EmuMJXEnv
            from .curriculum import RegimeGovernor
            e=EmuMJXEnv(str(ROBOT_XML),num_envs=8,seed=7,max_episode_steps=40,)
            p=RegimeGovernor().params_dict(); st,_=e.reset(p); out=e.step(st,jnp.zeros((8,6),dtype=jnp.float32),p); out.obs.block_until_ready()
            print(f'MJX reset/step: PASS obs={tuple(out.obs.shape)}')
        except Exception as exc: failures.append(f'MJX reset/step: {exc}')
    if args.domain_smoke:
        try:
            import jax
            import jax.numpy as jnp
            from .domain_randomization import DomainRandomizationConfig
            from .environment import EmuMJXEnv
            from .curriculum import RegimeGovernor
            cfg=DomainRandomizationConfig(enabled=True,robot_radius=.6,terrain_radius=.6,hardware_radius=.6)
            e=EmuMJXEnv(str(ROBOT_XML),num_envs=8,seed=5014,max_episode_steps=40,domain_config=cfg)
            p=RegimeGovernor().params_dict(); st,_=e.reset(p); out=e.step(st,jnp.zeros((8,6),dtype=jnp.float32),p); out.obs.block_until_ready()
            info=jax.device_get(e.domain_info)
            max_slope=max(float(abs(info['slope_roll_rad']).max()),float(abs(info['slope_pitch_rad']).max()))
            print(f'Domain MJX smoke: PASS obs={tuple(out.obs.shape)} max_sampled_slope={max_slope*57.2957795:.2f}deg')
        except Exception as exc: failures.append(f'Domain MJX smoke: {exc}')
    if failures:
        print('\nNOT READY:'); [print('  -',x) for x in failures]; return 1
    print('\nReady to train.'); return 0


def main(argv=None) -> int:
    p=argparse.ArgumentParser(prog='emu',description=f'EMU {VERSION} — dynamically unstable biped RL')
    p.add_argument('--version',action='version',version=VERSION)
    sub=p.add_subparsers(dest='cmd',required=True)

    for name,help_text in [('train','cold-start one trainer'),('compare','paired shadow-vs-graph experiment')]:
        q=sub.add_parser(name,help=help_text); q.add_argument('--seed',type=int,default=733); q.add_argument('--steps',type=int,default=120_000_000)
        q.add_argument('--device',choices=['auto','cpu','cuda','gpu'],default='cuda'); q.add_argument('--envs',type=int,default=2048)
        q.add_argument('--run-root',default=''); q.add_argument('--results-dir',default='')
        if name=='train': q.add_argument('--mode',choices=['shadow','graph'],default='graph')


    q=sub.add_parser('tournament',help='run the efficient 12-hypothesis v5013 successive-halving tournament')
    q.add_argument('--seed',type=int,default=733); q.add_argument('--confirm-seed',type=int,default=1742)
    q.add_argument('--device',choices=['auto','cpu','cuda','gpu'],default='cuda'); q.add_argument('--envs',type=int,default=2048)
    q.add_argument('--screen-envs',type=int,default=512); q.add_argument('--screen-horizon',type=int,default=350)
    q.add_argument('--run-root',default=''); q.add_argument('--results-dir',default='')
    q.add_argument('--stop-after',choices=['A','B','C','80','120','confirm'],default='confirm'); q.add_argument('--plan',action='store_true')

    q=sub.add_parser('view',help='open the read-only capability-field viewer'); q.add_argument('--seed',type=int,default=733); q.add_argument('--run-root',default=''); q.add_argument('--run-dir',default=''); q.add_argument('--port',type=int,default=8765)
    q=sub.add_parser('evaluate',help='run the physical capability atlas'); q.add_argument('--seed',type=int,default=733); q.add_argument('--run-root',default=''); q.add_argument('--checkpoint',default=''); q.add_argument('--device',choices=['auto','cpu','cuda','gpu'],default='cuda'); q.add_argument('--envs',type=int,default=1024); q.add_argument('--horizon',type=int,default=300); q.add_argument('--out-dir',default='')
    q=sub.add_parser('doctor',help='validate Python/JAX/MJX and core contracts'); q.add_argument('--require-gpu',action=argparse.BooleanOptionalAction,default=True); q.add_argument('--no-mjx-smoke',action='store_true'); q.add_argument('--domain-smoke',action='store_true',help='also compile/step an 8-world randomized MJX model')
    q=sub.add_parser('forge',help='train the single v5014 production lineage'); q.add_argument('--root',default=''); q.add_argument('--seed',type=int,default=733); q.add_argument('--envs',type=int,default=2048); q.add_argument('--device',choices=['auto','cpu','cuda','gpu'],default='cuda'); q.add_argument('--target',type=int,default=400_000_000,choices=[20_000_000,60_000_000,140_000_000,260_000_000,400_000_000]); q.add_argument('--plan',action='store_true')
    q=sub.add_parser('export-policy',help='export a checkpoint to browser-native .emupolicy'); q.add_argument('checkpoint'); q.add_argument('output'); q.add_argument('--pretty',action='store_true')
    q=sub.add_parser('viewer-build',help='build the static browser viewer into the package (requires npm once)')
    q=sub.add_parser('viewer',help='serve the local-browser MuJoCo viewer'); q.add_argument('policy',nargs='?',default=''); q.add_argument('--port',type=int,default=8765); q.add_argument('--no-open',action='store_true'); q.add_argument('--live',action='store_true',help='watch newest validated checkpoint from a training lineage'); q.add_argument('--root',default='',help='v5014 production run root to watch')

    args=p.parse_args(argv)
    if args.cmd=='doctor': return doctor(args)
    if args.cmd in ('train','compare'):
        av=[args.cmd,'--seed',str(args.seed),'--steps',str(args.steps),'--device',args.device,'--envs',str(args.envs)]
        if getattr(args,'mode',None): av += ['--mode',args.mode]
        if args.run_root: av += ['--run-root',args.run_root]
        if args.results_dir: av += ['--results-dir',args.results_dir]
        return _run('experiments',*av)
    if args.cmd=='tournament':
        av=['--seed',str(args.seed),'--confirm-seed',str(args.confirm_seed),'--device',args.device,
            '--envs',str(args.envs),'--screen-envs',str(args.screen_envs),'--screen-horizon',str(args.screen_horizon),
            '--stop-after',args.stop_after]
        if args.plan: av += ['--plan']
        if args.run_root: av += ['--run-root',args.run_root]
        if args.results_dir: av += ['--results-dir',args.results_dir]
        return _run('tournament',*av)
    if args.cmd=='view':
        from .field_viewer import main as view
        av=['--port',str(args.port)]
        if args.run_dir: av += ['--run-dir',args.run_dir]
        else: av += ['--run-root',str(Path(args.run_root).expanduser() if args.run_root else seed_root(args.seed))]
        return int(view(av) or 0)
    if args.cmd=='evaluate':
        root=Path(args.run_root).expanduser() if args.run_root else seed_root(args.seed)
        ck=Path(args.checkpoint).expanduser() if args.checkpoint else find_evaluation(root)
        if ck is None or not ck.exists(): raise SystemExit(f'no checkpoint found under {root}')
        out=Path(args.out_dir).expanduser() if args.out_dir else root/'atlas'
        return _run('atlas','--checkpoint',str(ck),'--out-dir',str(out),'--device',args.device,'--envs',str(args.envs),'--horizon',str(args.horizon),'--seed',str(args.seed))
    if args.cmd=='forge':
        av=['--seed',str(args.seed),'--envs',str(args.envs),'--device',args.device,'--target',str(args.target)]
        if args.root: av += ['--root',args.root]
        if args.plan: av += ['--plan']
        return _run('production',*av)
    if args.cmd=='export-policy':
        from .policy_export import build_bundle, write_bundle
        bundle=build_bundle(Path(args.checkpoint)); write_bundle(bundle,Path(args.output),pretty=args.pretty); print(Path(args.output).expanduser().resolve()); return 0
    if args.cmd=='viewer-build':
        from .viewer_server import build
        return build()
    if args.cmd=='viewer':
        from .viewer_server import serve
        pol=Path(args.policy) if args.policy else None
        return serve(port=args.port,policy=pol,live_root=(Path(args.root).expanduser() if args.root else Path.home()/'robot_lab'/'emu_runs'/'v5014_production') if args.live else None,open_browser=not args.no_open)
    return 2


if __name__=='__main__': raise SystemExit(main())
