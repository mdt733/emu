#!/usr/bin/env python3
"""Public command line for modern EMU."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

from .contracts import VERSION, ROBOT_XML, seed_root
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
        from .curriculum import CapabilityField, FIELD_VERSION
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
    if getattr(args, 'v5016_smoke', False):
        # Explicit sensor-schema acceptance smoke, no writes or long run.
        try:
            import jax
            import jax.numpy as jnp
            from .contracts import OBS_SCHEMA_SENSOR, SENSOR_OBS_DIM
            from .environment import EmuMJXEnv
            from .curriculum import RegimeGovernor
            from .ppo import make_trainer, build_obs_norm_mask
            n = 2
            env = EmuMJXEnv(str(ROBOT_XML), num_envs=n, seed=5016,
                            max_episode_steps=30, attitude_profile='v5015',
                            contact_slope_deg=2.,
                            observation_schema=OBS_SCHEMA_SENSOR)
            params = RegimeGovernor().params_dict()
            state, obs = env.reset(params)
            if obs.shape != (n, SENSOR_OBS_DIM) or state.hist.shape != (n, 4, 58):
                raise RuntimeError(f'v5016 sensor observation shape mismatch: {obs.shape}')
            if build_obs_norm_mask(OBS_SCHEMA_SENSOR).shape != (SENSOR_OBS_DIM,):
                raise RuntimeError('v5016 observation normalizer mask mismatch')
            out = env.step(state, jnp.zeros((n, 6), dtype=jnp.float32), params)
            out.obs.block_until_ready()
            if out.bootstrap_obs.shape != (n, SENSOR_OBS_DIM):
                raise RuntimeError('v5016 bootstrap observation shape mismatch')
            trainer = make_trainer(env, seed=5016, lr=2e-4, rollout=2,
                                   epochs=1, minibatches=1)
            init = trainer['make_initial_state'](params)
            packed = env.params_array(params)
            mean = jnp.zeros((SENSOR_OBS_DIM,), jnp.float32)
            var = jnp.ones((SENSOR_OBS_DIM,), jnp.float32)
            updated, _, diagnostics, _, _ = trainer['train_update'](
                init, packed, mean, var, jnp.asarray(.25, dtype=jnp.float32))
            if not all(bool(jax.device_get(jnp.all(jnp.isfinite(leaf))))
                       for leaf in jax.tree_util.tree_leaves(updated.params)):
                raise RuntimeError('v5016 PPO update produced nonfinite parameters')
            if not bool(jax.device_get(jnp.isfinite(diagnostics['rollout_reward_mean']))):
                raise RuntimeError('v5016 PPO update produced nonfinite rollout reward')
            print('v5016 sensor MJX + 232-input PPO gradient smoke: PASS '
                  '(2 worlds, 2-frame rollout, one optimizer update)')
        except Exception as exc:
            failures.append(f'v5016 sensor PPO smoke: {exc}')
    if args.domain_smoke:
        try:
            import jax
            import jax.numpy as jnp
            from .environment import DomainRandomizationConfig
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
        epilog='One continuous run: emu train --help. Matched studies: emu ladder --stage plan.')
    p.add_argument('--version',action='version',version=VERSION)
    sub=p.add_subparsers(dest='cmd',required=True)

    q=sub.add_parser('view',help='open the read-only capability-field viewer'); q.add_argument('--seed',type=int,default=733); q.add_argument('--run-root',default=''); q.add_argument('--run-dir',default=''); q.add_argument('--port',type=int,default=8765)
    q=sub.add_parser('evaluate',help='run the physical capability atlas'); q.add_argument('--seed',type=int,default=733); q.add_argument('--run-root',default=''); q.add_argument('--checkpoint',default=''); q.add_argument('--device',choices=['auto','cpu','cuda','gpu'],default='cuda'); q.add_argument('--envs',type=int,default=1024); q.add_argument('--horizon',type=int,default=300); q.add_argument('--out-dir',default='')
    q=sub.add_parser('doctor',help='validate Python/JAX/MJX and core contracts'); q.add_argument('--require-gpu',action=argparse.BooleanOptionalAction,default=True); q.add_argument('--no-mjx-smoke',action='store_true'); q.add_argument('--domain-smoke',action='store_true',help='also compile/step an 8-world randomized MJX model'); q.add_argument('--v5015-smoke',action='store_true',help='compile/step experimental pitch tasks and 11-field PPO evaluation (no training)'); q.add_argument('--v5016-smoke',action='store_true',help='compile the opt-in 232-input sensor MJX and one PPO update (no checkpoint/training)')
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
    source=q.add_mutually_exclusive_group(required=True)
    source.add_argument('--run-dir',help='read-only schema-7 training run')
    source.add_argument('--policy',help='frozen exported .emupolicy actor')
    q.add_argument('--output',required=True,help='unique examination directory, outside training root')
    q.add_argument('--angle',type=float,default=0.)
    q.add_argument('--direction',choices=['uphill','downhill','cross_left','cross_right'],default='uphill')
    q.add_argument('--worlds',type=int,default=8)
    q.add_argument('--horizon',type=int,default=350)
    q.add_argument('--speed',type=float,default=.10)
    q.add_argument('--lateral',type=float,default=0.)
    q.add_argument('--yaw-rate',type=float,default=0.)
    q.add_argument('--settle-frames',type=int,default=25)
    from .evaluation import SCENARIOS
    q.add_argument('--scenario',choices=SCENARIOS,default='steady')
    q.add_argument('--turn-rate',type=float,default=.25)
    q.add_argument('--gyro-bias',type=float,default=0.)
    q.add_argument('--gyro-noise',type=float,default=0.)
    q.add_argument('--gyro-delay-frames',type=int,default=0)
    q.add_argument('--contact-slope',action='store_true',help='real inclined floor with vertical gravity')
    q.add_argument('--reuse-completed',action='store_true',
                   help='reuse saved OFF/ON exam JSON to rebuild missing comparison')
    q.add_argument('--compare-heading',action='store_true',
                   help='matched read-only controller OFF/ON comparison')
    q.add_argument('--heading-kp',type=float,default=0.75)
    q.add_argument('--seed',type=int,default=1701)
    q.add_argument('--device',choices=['auto','cpu','cuda','gpu'],default='cuda')
    q=sub.add_parser('directional-batch',
        help='plan or run bounded v5016 GPU exams and compile a shareable ZIP')
    source=q.add_mutually_exclusive_group(required=True)
    source.add_argument('--root',help='original read-only four-way run root')
    source.add_argument('--policy',help='frozen exported actor, requires physical profile')
    q.add_argument('--output',required=True,help='new separate exam root')
    q.add_argument('--profile',choices=['smoke','pilot','sensor','stretch','physical'],
                   default='smoke')
    q.add_argument('--worlds',type=int)
    q.add_argument('--seed',type=int,default=1701)
    q.add_argument('--heading-kp',type=float,default=.75)
    q.add_argument('--device',choices=['auto','cpu','cuda','gpu'],default='cuda')
    q.add_argument('--run',action='store_true',help='execute rather than plan only')
    q=sub.add_parser('campaign5016',help='12h staged 232-input pinnacle; plan by default, --run to execute')
    q.add_argument('--root',default=str(Path.home()/'robot_lab'/'emu_runs'/'v5016_pinnacle'))
    q.add_argument('--foundation',default=str(Path.home()/'robot_lab'/'emu_runs'/'v5016_sensor_foundation_20m'))
    q.add_argument('--hours',type=float,default=12.)
    q.add_argument('--envs',type=int,default=2048)
    q.add_argument('--screen-envs',type=int,default=128)
    q.add_argument('--screen-horizon',type=int,default=300)
    q.add_argument('--seed',type=int,default=733)
    q.add_argument('--device',choices=['cuda','cpu','auto','gpu'],default='cuda')
    q.add_argument('--plan',action='store_true')
    q.add_argument('--status',action='store_true')
    q.add_argument('--run',action='store_true',help='run the GPU campaign (default only plans)')
    q.add_argument('--resume-after-screen-fix',action='store_true',
                   help='audit original training revision and resume failed Round-1 screen')
    q=sub.add_parser('report',help='regenerate campaign results.md and evidence ZIP without GPU')
    q.add_argument('--root',required=True)
    q.add_argument('--no-bundle',action='store_true')
    q=sub.add_parser('train',help='train one protected experiment (no ladder)')
    q.add_argument('--run-dir',required=True,help='unique experiment directory OUTSIDE source checkout')
    q.add_argument('--steps',type=int,required=True,help='absolute target environment steps')
    q.add_argument('--envs',type=int,default=2048)
    q.add_argument('--seed',type=int,default=733)
    q.add_argument('--device',choices=['auto','cpu','cuda','gpu'],default='cuda')
    q.add_argument('--profile',choices=['foundation','mature','sim2real'],default='foundation')
    q.add_argument('--pitch-deg',type=float,default=None,help='chassis pitch command limit in degrees (sim2real defaults to 30; legacy profiles to 8)')
    q.add_argument('--motor-guidance-weight',type=float,default=0.0,
                   help='optional motor effort teaching term in [0, 0.10], default 0 unchanged')
    q.add_argument('--contact-slope-deg',type=float,default=0.0,
                   help='opt-in shared real incline 0..5 degrees, sim2real only')
    q.add_argument('--contact-terrain-radius',type=float,default=0.0,
                   help='opt-in per-world real plane inclines (0..1 strength, max 5deg), sim2real only')
    q.add_argument('--terrain-box-height-mm',type=int,choices=[0,5,10],default=None,
                   help='v5018 isolated real-MJX step research: 0mm off-course control or 5/10mm step; sim2real + sensor actor only')
    q.add_argument('--terrain-height-ceiling-mm',type=int,default=0,
                   help='adaptive per-episode real steps 0..150mm; 0=off; sim2real sensor actor only')
    q.add_argument('--terrain-height-flat-control',action='store_true',
                   help='same adaptive mocap geometry, box always off course; matched flat PPO control')
    q.add_argument('--terrain-height-exposure',type=float,default=.60,
                   help='fraction of births on variable-height obstacle, rest on flat rehearsal')
    q.add_argument('--terrain-height-edge-fraction',type=float,default=.35,
                   help='fraction of obstacle births near the earned height frontier')
    q.add_argument('--terrain-height-growth-mm',type=float,default=1.0,
                   help='physical height growth per qualified rollout, millimetres')
    q.add_argument('--terrain-progress-weight',type=float,default=0.,
                   help='opt-in bounded reward-only obstacle approach/crossing coaching [0,0.10]')
    q.add_argument('--terrain-height-initial-mm',type=float,default=3.0,
                   help='initial adaptive obstacle upper height, millimetres')
    q.add_argument('--sensor-actor',action='store_true',
                   help='train opt-in 232-input ToF/optical-flow policy (foundation or sim2real profile)')
    q.add_argument('--forward-preview',action='store_true',
                   help='v5019 opt-in 280-input two-lane 3-zone ToF profile (preserves 220/232 policies)')
    q.add_argument('--forward-preview-masked',action='store_true',
                   help='same 280-input sensor geometry/timing with profile+validity channels zeroed (controlled baseline)')
    q.add_argument('--resume',default='',help='auto for same-run checkpoint, or an explicit compatible parent .pkl')
    q.add_argument('--plan',action='store_true',help='print exact resolved config without training or writing files')
    q=sub.add_parser('export-policy',help='export a checkpoint to browser-native .emupolicy'); q.add_argument('checkpoint'); q.add_argument('output'); q.add_argument('--pretty',action='store_true')
    q=sub.add_parser('viewer-build',help='build the static browser viewer into the package (requires npm once)')
    q=sub.add_parser('viewer',help='serve the local-browser MuJoCo viewer'); q.add_argument('policy',nargs='?',default=''); q.add_argument('--port',type=int,default=8765); q.add_argument('--no-open',action='store_true'); q.add_argument('--live',action='store_true',help='watch newest validated checkpoint from a training lineage'); q.add_argument('--root',default='',help='experiment run root to watch')

    args=p.parse_args(argv)
    if args.cmd=='doctor': return doctor(args)
    if args.cmd=='view':
        from .viewer_server import field_main as view
        av=['--port',str(args.port)]
        if args.run_dir: av += ['--run-dir',args.run_dir]
        else: av += ['--run-root',str(Path(args.run_root).expanduser() if args.run_root else seed_root(args.seed))]
        return int(view(av) or 0)
    if args.cmd=='evaluate':
        root=Path(args.run_root).expanduser() if args.run_root else seed_root(args.seed)
        ck=Path(args.checkpoint).expanduser() if args.checkpoint else find_evaluation(root)
        if ck is None or not ck.exists(): raise SystemExit(f'no checkpoint found under {root}')
        out=Path(args.out_dir).expanduser() if args.out_dir else root/'atlas'
        from .evaluation import atlas_main
        return atlas_main(['--checkpoint',str(ck),'--out-dir',str(out),'--device',args.device,'--envs',str(args.envs),'--horizon',str(args.horizon),'--seed',str(args.seed)])
    if args.cmd=='posture-report':
        from .results import posture_report
        return posture_report(Path(args.root) if args.root else Path.home()/'robot_lab'/'emu_runs')
    if args.cmd in ('ladder','lab5015'):
        av=['--stage',args.stage,'--round',args.round,'--stop-after',args.stop_after,'--seed',str(args.seed),
            '--envs',str(args.envs),'--device',args.device,
            '--screen-envs',str(args.screen_envs),'--screen-horizon',str(args.screen_horizon)]
        if args.root: av += ['--root',args.root]
        for arm in args.arm: av += ['--arm',arm]
        if args.select: av += ['--select',*args.select]
        return _run('branch_lab',*av)
    if args.cmd=='directional-exam':
        # Parse ONCE and call the examiner directly. Previously this rebuilt
        # strings for a second parser: a fragile seam that caused real WSL
        # crashes while lower-level tests remained green.
        from .evaluation import evaluate_directional_pair, evaluate_directional_run
        if args.reuse_completed and not args.compare_heading:
            p.error('--reuse-completed requires --compare-heading')
        request=dict(
            run_dir=args.run_dir,policy_file=args.policy,output=args.output,
            angle_deg=args.angle,direction=args.direction,
            worlds=args.worlds,horizon=args.horizon,speed=args.speed,
            lateral=args.lateral,yaw_rate=args.yaw_rate,
            settle_frames=args.settle_frames,seed=args.seed,device=args.device,
            scenario=args.scenario,turn_rate=args.turn_rate,
            gyro_bias=args.gyro_bias,gyro_noise=args.gyro_noise,
            gyro_delay_frames=args.gyro_delay_frames,
            contact_slope=args.contact_slope)
        if args.compare_heading:
            paths=evaluate_directional_pair(
                heading_kp=args.heading_kp,
                reuse_completed=args.reuse_completed,**request)
        else:
            paths=evaluate_directional_run(heading_kp=0.,**request)
        for path in paths:
            print(path)
        return 0
    if args.cmd=='directional-batch':
        from .evaluation import plan,execute
        from .evaluation import source_commit
        jobs=plan(args.profile,args.root,args.output,worlds=args.worlds,
                  seed=args.seed,kp=args.heading_kp,device=args.device,
                  policy_file=args.policy)
        execute(jobs,source_commit(),actually_run=args.run)
        return 0
    if args.cmd=='campaign5016':
        from .campaign5016 import main as pinnacle
        argv=['--root',args.root,'--foundation',args.foundation,
              '--hours',str(args.hours),'--envs',str(args.envs),
              '--screen-envs',str(args.screen_envs),
              '--screen-horizon',str(args.screen_horizon),
              '--seed',str(args.seed),'--device',args.device]
        if args.plan: argv.append('--plan')
        if args.status: argv.append('--status')
        if args.run: argv.append('--run')
        if args.resume_after_screen_fix: argv.append('--resume-after-screen-fix')
        return pinnacle(argv)
    if args.cmd=='report':
        from .results import write_report
        for output in write_report(Path(args.root),bundle=not args.no_bundle):
            print(output)
        return 0
    if args.cmd=='train':
        from .branch_lab import train_one
        return train_one(args)
    if args.cmd=='export-policy':
        from .viewer_server import build_bundle, write_bundle
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
