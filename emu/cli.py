#!/usr/bin/env python3
"""Public command line for modern EMU."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

from .paths import VERSION, ROBOT_XML, seed_root
from .checkpoint_io import find_evaluation


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
    if getattr(args, 'v5015_smoke', False):
        # Explicit integration smoke: actually JIT-compile and step an
        # experimental four-command task, then compile the held-out summary.
        # No training/checkpoint/output files are created by this probe.
        try:
            import jax
            import jax.numpy as jnp
            from .environment import EmuMJXEnv
            from .curriculum import RegimeGovernor
            from .ppo import make_trainer
            from .contracts import OBS_DIM
            n = 4
            env = EmuMJXEnv(str(ROBOT_XML), num_envs=n, seed=5015,
                            max_episode_steps=40, attitude_profile='v5015',
                            pitch_command_max_deg=45.0)
            params = RegimeGovernor().params_dict()
            state, obs = env.reset(params)
            if state.cmd_target.shape != (n, 4) or obs.shape != (n, OBS_DIM):
                raise RuntimeError(f'v5015 reset contract: target={state.cmd_target.shape}, obs={obs.shape}')
            stepped = env.step(state, jnp.zeros((n, 6), dtype=jnp.float32), params)
            stepped.obs.block_until_ready()
            for key in ('body_pitch', 'cmd_pitch', 'robot_lean_forward',
                        'robot_lean_side', 'structural_attitude_ok', 'height_error'):
                arr = stepped.metrics[key]
                if arr.shape != (n,) or not bool(jax.device_get(jnp.all(jnp.isfinite(arr)))):
                    raise RuntimeError(f'v5015 invalid metric {key}: {arr.shape}')
            learner = make_trainer(env, seed=5015, lr=2e-4, rollout=4,
                                   epochs=1, minibatches=1)
            initial = learner['make_initial_state'](params)
            compact = learner['evaluate_summary'](
                initial.params, env.params_array(params),
                jnp.zeros((OBS_DIM,), dtype=jnp.float32),
                jnp.ones((OBS_DIM,), dtype=jnp.float32),
                jax.random.PRNGKey(5015), 3)
            compact.block_until_ready()
            if compact.shape != (11,) or not bool(jax.device_get(jnp.all(jnp.isfinite(compact)))):
                raise RuntimeError(f'v5015 held-out summary invalid: {compact.shape}')
            # Also JIT-compile one *real* PPO gradient/update, not just forward
            # inference: this catches reward/rollout/optimizer shape drift
            # before spending 20 million GPU environment steps.
            updated, rolled, diagnostics, _, _ = learner['train_update'](
                initial, env.params_array(params),
                jnp.zeros((OBS_DIM,), dtype=jnp.float32),
                jnp.ones((OBS_DIM,), dtype=jnp.float32),
                jnp.asarray(.25, dtype=jnp.float32))
            if not all(bool(jax.device_get(jnp.all(jnp.isfinite(leaf))))
                       for leaf in jax.tree_util.tree_leaves(updated.params)):
                raise RuntimeError('v5015 PPO smoke produced nonfinite parameters')
            if not bool(jax.device_get(jnp.isfinite(diagnostics['rollout_reward_mean']))):
                raise RuntimeError('v5015 PPO smoke produced nonfinite rollout reward')
            print('v5015 MJX + PPO update + evaluation smoke: PASS (4 envs, one optimizer update, 11 evaluation fields)')
        except Exception as exc:
            failures.append(f'v5015 MJX + PPO smoke: {exc}')
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
    p=argparse.ArgumentParser(
        prog='emu', description=f'EMU {VERSION} — dynamically unstable biped RL',
        epilog='Current recipe: emu ladder --stage plan. To train safely: bash scripts/run_ladder.sh')
    p.add_argument('--version',action='version',version=VERSION)
    sub=p.add_subparsers(dest='cmd',required=True)

    q=sub.add_parser('view',help='open the read-only capability-field viewer'); q.add_argument('--seed',type=int,default=733); q.add_argument('--run-root',default=''); q.add_argument('--run-dir',default=''); q.add_argument('--port',type=int,default=8765)
    q=sub.add_parser('evaluate',help='run the physical capability atlas'); q.add_argument('--seed',type=int,default=733); q.add_argument('--run-root',default=''); q.add_argument('--checkpoint',default=''); q.add_argument('--device',choices=['auto','cpu','cuda','gpu'],default='cuda'); q.add_argument('--envs',type=int,default=1024); q.add_argument('--horizon',type=int,default=300); q.add_argument('--out-dir',default='')
    q=sub.add_parser('doctor',help='validate Python/JAX/MJX and core contracts'); q.add_argument('--require-gpu',action=argparse.BooleanOptionalAction,default=True); q.add_argument('--no-mjx-smoke',action='store_true'); q.add_argument('--domain-smoke',action='store_true',help='also compile/step an 8-world randomized MJX model'); q.add_argument('--v5015-smoke',action='store_true',help='compile/step experimental pitch tasks and 11-field PPO evaluation (no training)')
    q=sub.add_parser('posture-report',help='read-only height drift report for saved metrics.csv'); q.add_argument('--root',default='')
    for ladder_command in ('ladder', 'lab5015'):
        q=sub.add_parser(ladder_command,help=('current experiment ladder — inspect with --stage plan'
                                             if ladder_command=='ladder'
                                             else argparse.SUPPRESS))
        q.add_argument('--stage',choices=['plan','foundation','round1','round2','round3','score','auto'],default='plan')
        q.add_argument('--round',choices=['round1','round2','round3'],default='round1')
        q.add_argument('--stop-after',choices=['round1','round2','round3'],default='round3',help='for --stage auto, stop after scoring this round')
        q.add_argument('--root',default='')
        q.add_argument('--seed',type=int,default=733)
        q.add_argument('--envs',type=int,default=2048)
        q.add_argument('--device',choices=['auto','cpu','cuda','gpu'],default='cuda')
        q.add_argument('--screen-envs',type=int,default=256)
        q.add_argument('--screen-horizon',type=int,default=350)
        q.add_argument('--arm',action='append',default=[])
        q.add_argument('--select',nargs='+',default=[])
    q=sub.add_parser('directional-exam',help='read-only real-MJX course and contact-stride exam of a saved policy')
    q.add_argument('--run-dir',required=True)
    q.add_argument('--output',required=True,help='unique examination directory, outside training root')
    q.add_argument('--angle',type=float,default=0.)
    q.add_argument('--direction',choices=['uphill','downhill','cross_left','cross_right'],default='uphill')
    q.add_argument('--worlds',type=int,default=8)
    q.add_argument('--horizon',type=int,default=350)
    q.add_argument('--speed',type=float,default=.10)
    q.add_argument('--lateral',type=float,default=0.)
    q.add_argument('--yaw-rate',type=float,default=0.)
    q.add_argument('--settle-frames',type=int,default=25)
    from .directional_exam import SCENARIOS
    q.add_argument('--scenario',choices=SCENARIOS,default='steady')
    q.add_argument('--turn-rate',type=float,default=.25)
    q.add_argument('--gyro-bias',type=float,default=0.)
    q.add_argument('--gyro-noise',type=float,default=0.)
    q.add_argument('--gyro-delay-frames',type=int,default=0)
    q.add_argument('--compare-heading',action='store_true',
                   help='matched read-only controller OFF/ON comparison')
    q.add_argument('--heading-kp',type=float,default=0.75)
    q.add_argument('--seed',type=int,default=1701)
    q.add_argument('--device',choices=['auto','cpu','cuda','gpu'],default='cuda')
    q=sub.add_parser('directional-batch',
        help='plan or run bounded v5016 GPU exams and compile a shareable ZIP')
    q.add_argument('--root',required=True,help='original read-only four-way run root')
    q.add_argument('--output',required=True,help='new separate exam root')
    q.add_argument('--profile',choices=['smoke','pilot','sensor','stretch'],
                   default='smoke')
    q.add_argument('--worlds',type=int)
    q.add_argument('--seed',type=int,default=1701)
    q.add_argument('--heading-kp',type=float,default=.75)
    q.add_argument('--device',choices=['auto','cpu','cuda','gpu'],default='cuda')
    q.add_argument('--run',action='store_true',help='execute rather than plan only')
    q=sub.add_parser('report',help='regenerate campaign results.md and evidence ZIP without GPU')
    q.add_argument('--root',required=True)
    q.add_argument('--no-bundle',action='store_true')
    q=sub.add_parser('train',help='train one protected experiment (no ladder)')
    q.add_argument('--run-dir',required=True,help='unique experiment directory OUTSIDE source checkout')
    q.add_argument('--steps',type=int,required=True,help='absolute target environment steps')
    q.add_argument('--envs',type=int,default=2048)
    q.add_argument('--seed',type=int,default=733)
    q.add_argument('--device',choices=['auto','cpu','cuda','gpu'],default='cuda')
    q.add_argument('--profile',choices=['foundation','mature'],default='foundation')
    q.add_argument('--resume',default='',help='auto for same-run checkpoint, or an explicit compatible parent .pkl')
    q.add_argument('--plan',action='store_true',help='print exact resolved config without training or writing files')
    q=sub.add_parser('export-policy',help='export a checkpoint to browser-native .emupolicy'); q.add_argument('checkpoint'); q.add_argument('output'); q.add_argument('--pretty',action='store_true')
    q=sub.add_parser('viewer-build',help='build the static browser viewer into the package (requires npm once)')
    q=sub.add_parser('viewer',help='serve the local-browser MuJoCo viewer'); q.add_argument('policy',nargs='?',default=''); q.add_argument('--port',type=int,default=8765); q.add_argument('--no-open',action='store_true'); q.add_argument('--live',action='store_true',help='watch newest validated checkpoint from a training lineage'); q.add_argument('--root',default='',help='experiment run root to watch')

    args=p.parse_args(argv)
    if args.cmd=='doctor': return doctor(args)
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
    if args.cmd=='posture-report':
        from .posture_report import report
        return report(Path(args.root) if args.root else Path.home()/'robot_lab'/'emu_runs')
    if args.cmd in ('ladder','lab5015'):
        av=['--stage',args.stage,'--round',args.round,'--stop-after',args.stop_after,'--seed',str(args.seed),
            '--envs',str(args.envs),'--device',args.device,
            '--screen-envs',str(args.screen_envs),'--screen-horizon',str(args.screen_horizon)]
        if args.root: av += ['--root',args.root]
        for arm in args.arm: av += ['--arm',arm]
        if args.select: av += ['--select',*args.select]
        return _run('branch_lab',*av)
    if args.cmd=='directional-exam':
        return _run('directional_exam','--run-dir',args.run_dir,'--output',args.output,
                    '--angle',str(args.angle),'--direction',args.direction,
                    '--worlds',str(args.worlds),'--horizon',str(args.horizon),
                    '--speed',str(args.speed),'--lateral',str(args.lateral),
                    '--yaw-rate',str(args.yaw_rate),'--settle-frames',str(args.settle_frames),
                    '--seed',str(args.seed),'--device',args.device,
                    '--heading-kp',str(args.heading_kp),
                    '--scenario',args.scenario,'--turn-rate',str(args.turn_rate),
                    '--gyro-bias',str(args.gyro_bias),'--gyro-noise',str(args.gyro_noise),
                    '--gyro-delay-frames',str(args.gyro_delay_frames),
                    *(['--compare-heading'] if args.compare_heading else []))
    if args.cmd=='directional-batch':
        from .v5016_batch import plan,execute
        from .directional_exam import source_commit
        jobs=plan(args.profile,args.root,args.output,worlds=args.worlds,
                  seed=args.seed,kp=args.heading_kp,device=args.device)
        execute(jobs,source_commit(),actually_run=args.run)
        return 0
    if args.cmd=='report':
        from .results import write_report
        for output in write_report(Path(args.root),bundle=not args.no_bundle):
            print(output)
        return 0
    if args.cmd=='train':
        from .single_run import train_one
        return train_one(args)
    if args.cmd=='export-policy':
        from .policy_export import build_bundle, write_bundle
        bundle=build_bundle(Path(args.checkpoint)); write_bundle(bundle,Path(args.output),pretty=args.pretty); print(Path(args.output).expanduser().resolve()); return 0
    if args.cmd=='viewer-build':
        from .viewer_server import build
        return build()
    if args.cmd=='viewer':
        from .viewer_server import serve
        pol=Path(args.policy) if args.policy else None
        return serve(port=args.port,policy=pol,live_root=(Path(args.root).expanduser() if args.root else Path.home()/'robot_lab'/'emu_runs'/'v5015_lab1') if args.live else None,open_browser=not args.no_open)
    return 2


if __name__=='__main__': raise SystemExit(main())
