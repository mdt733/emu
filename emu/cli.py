#!/usr/bin/env python3
"""Public command line for modern EMU."""
from __future__ import annotations

import argparse
from pathlib import Path

from .contracts import VERSION, seed_root
from .doctor import doctor
from .checkpoint_io import find_evaluation


def main(argv=None) -> int:
    p=argparse.ArgumentParser(
        prog='emu', description=f'EMU {VERSION} — dynamically unstable biped RL',
        epilog='One source-pinned lineage: emu train --help. Unified GPU training remains launch-guarded.')
    p.add_argument('--version',action='version',version=VERSION)
    sub=p.add_subparsers(dest='cmd',required=True)

    q=sub.add_parser('view',help='open the read-only capability-field viewer'); q.add_argument('--seed',type=int,default=733); q.add_argument('--run-root',default=''); q.add_argument('--run-dir',default=''); q.add_argument('--port',type=int,default=8765)
    q=sub.add_parser('evaluate',help='run the physical capability atlas'); q.add_argument('--seed',type=int,default=733); q.add_argument('--run-root',default=''); q.add_argument('--checkpoint',default=''); q.add_argument('--device',choices=['auto','cpu','cuda','gpu'],default='cuda'); q.add_argument('--envs',type=int,default=1024); q.add_argument('--horizon',type=int,default=300); q.add_argument('--out-dir',default='')
    q=sub.add_parser('doctor',help='validate Python/JAX/MJX and core contracts'); q.add_argument('--require-gpu',action=argparse.BooleanOptionalAction,default=True); q.add_argument('--no-mjx-smoke',action='store_true'); q.add_argument('--domain-smoke',action='store_true',help='also compile/step an 8-world randomized MJX model')
    q=sub.add_parser('posture-report',help='read-only height drift report for saved metrics.csv'); q.add_argument('--root',default='')
    q=sub.add_parser('terrain-exam',help='frozen-policy continuous MJX terrain course: flat vs step up/down')
    q.add_argument('--policy',required=True,help='verified 220/232-input .emupolicy (diagnostic, read only)')
    q.add_argument('--output',required=True,help='new/empty directory OUTSIDE checkout')
    q.add_argument('--worlds',type=int,default=4,help='even worlds, paired up/down, max 32')
    q.add_argument('--horizon',type=int,default=260,help='control frames, 80..700')
    q.add_argument('--speed',type=float,default=.15,help='requested forward speed, m/s')
    q.add_argument('--heights-cm',type=float,nargs='+',default=[0.,2.,4.],
                   help='matched real collision heights in cm (default 0 2 4)')
    q.add_argument('--seed',type=int,default=1701)
    q.add_argument('--device',choices=['cpu','cuda','gpu','auto'],default='cpu')
    q=sub.add_parser('capability-exam',help='frozen 360-input unified physics: command, slope, step, recovery and control')
    q.add_argument('--policy',required=True,help='verified array-only 360 .emupolicy')
    q.add_argument('--output',required=True,help='new/empty directory separate from actor')
    q.add_argument('--worlds',type=int,default=4,help='even 2..32 paired physical worlds')
    q.add_argument('--horizon',type=int,default=300)
    q.add_argument('--speed',type=float,default=.10)
    q.add_argument('--seed',type=int,default=1701)
    q.add_argument('--device',choices=['cpu','cuda','gpu','auto'],default='cpu')
    q.add_argument('--quick',action='store_true',help='eight essential physical cases instead of the full nineteen')
    q.add_argument('--legacy-main-baseline',action='store_true',
                   help='explicit read-only sim2real_232 showcase on the SAME 360 two-slab physics')
    q.add_argument('--speed-ladder',action='store_true',
                   help='read-only flat 0.1..1.0m/s measured speed ladder')
    q.add_argument('--ladder-speeds',type=float,nargs='+',default=None,
                   help='optional requested forward speeds in m/s; requires --speed-ladder')
    q.add_argument('--terrain-matrix',action='store_true',
                   help='matched 1/2/3/5 degree four-direction slope and step matrix')
    q.add_argument('--matrix-grades-deg',type=float,nargs='+',default=None,
                   help='optional four-direction physical matrix, e.g. 5 8 10 12 15; require --terrain-matrix')
    q.add_argument('--matrix-heights-cm',type=float,nargs='+',default=None,
                   help='optional paired UP/DOWN physical steps in cm, e.g. 2 2.5 3 3.5 (max 4)')
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
    q=sub.add_parser('physical-quality',help='summarise independent frozen physical exams without GPU or policy changes')
    q.add_argument('--exam',type=Path,action='append',required=True,
                   help='unified_exam.json or its directory; repeat for independent seeds/test panels')
    q.add_argument('--output',type=Path,required=True,
                   help='new output directory for physical_quality.json and .md')
    q=sub.add_parser('report',help='regenerate campaign results.md and evidence ZIP without GPU')
    q.add_argument('--root',required=True)
    q.add_argument('--no-bundle',action='store_true')
    q=sub.add_parser('train',help='one source-pinned training lineage (foundation or unified research)')
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
    q.add_argument('--sensor-actor',action='store_true',
                   help='train opt-in 232-input ToF/optical-flow policy (foundation or sim2real profile)')
    q.add_argument('--terrain-actor',action='store_true',
                   help='train NEW 360-input 4x4 actual-depth ToF policy, not the 232 champion')
    q.add_argument('--terrain-step-max-m',type=float,default=0.,
                   help='fixed physical finite-step topology, 0.01..0.08m for terrain actor')
    q.add_argument('--unified-capabilities',action='store_true',help='single-policy capability specialists')
    q.add_argument('--terrain-carpet',action='store_true',
                   help='v5041: 9 real 2-triangle MJX heightfields recycled in all specialist worlds')
    q.add_argument('--carpet-transition-cap-deg',type=float,default=3.0,
                   help='maximum smoothly varying local slope-angle change between 50cm tiles')
    q.add_argument('--foundation-steps',type=int,choices=(10000000,20000000),default=20000000,
                   help='cold foundation length: experiment at 10M or reference 20M')
    q.add_argument('--fixed-eval-every-steps',type=int,default=60000000,
                   help='source-pinned frozen exam interval (overnight comparison: 10M)')
    q.add_argument('--teaching-schedule',choices=('balanced','staged','terrain-early','steering'),
                   default='balanced',help='per-episode attention: balanced, staged, or terrain-early')
    q.add_argument('--foundation-mix',choices=('main','command-heavy','curious'),default='main',
                   help='one learner; compare original regime allocation with command-heavy or gentle grace nursery')
    q.add_argument('--growth-recipe',choices=('legacy','main-mature'),default='legacy',
                   help='experiment: old 360 recipe or restore main mature commands/phase/rehearsal')
    q.add_argument('--athletic-rehearsal-share',type=float,default=None,
                   help='v5029: 0..0.5 of unified growth births keep full-speed flat steady gait; default inherits profile')
    q.add_argument('--temporal-vx-reward-share',type=float,default=0.,
                   help='v5030 B: 0..1 rolling 25-frame (0.5s) forward tracking blend; OFF by default')
    q.add_argument('--verified-ascent-reallocation',type=float,default=0.,
                   help='v5031 C: shift 0..4 existing uphill shaping points to verified upper-support transfer')
    q.add_argument('--retention-share',type=float,default=0.0,
                   help='0..0.4 of focused births reserved for dynamic old-skill rehearsal')
    q.add_argument('--terrain-lesson',choices=('mixed','coherent','coherent-direct','coherent-careful'),default='mixed',
                   help='mixed, coherent, coherent-direct, or bounded considered-step ascent')
    q.add_argument('--force-program',choices=('gentle','aggressive'),default='aggressive',
                   help='gentle 0.7N research push ceiling or aggressive 4N push frontier')
    q.add_argument('--velocity-i-ramp-s',type=float,default=0.4,
                   help='experimental vx/vy integral ramp seconds (0.1..5)')
    q.add_argument('--velocity-i-multiplier',type=float,default=1.0,
                   help='experimental command trim ceiling: 1=OFF, 1.5/2=trial')
    # From-scratch streaming STEP worlds, same actor/optimizer/world batch.
    q.add_argument('--academy-stream',action='store_true',
                   help='three recyclable real MJX slabs; unlimited random up/down lips')
    q.add_argument('--academy-intro-steps',type=int,default=60000000,
                   help='birth age for initial easy multi-lip nursery')
    q.add_argument('--academy-intro-share',type=float,default=.10,
                   help='Academy share at start of the main age ramp')
    q.add_argument('--academy-start-steps',type=int,default=80000000,
                   help='start increasing Academy allocation toward maximum')
    q.add_argument('--academy-ramp-steps',type=int,default=20000000,
                   help='world-step duration for smooth Academy ramp')
    q.add_argument('--academy-max-share',type=float,default=.50,
                   help='fraction of ALL new episode births (not PPO steps) at full ramp')
    q.add_argument('--academy-height-cap-m',type=float,default=.150,
                   help='maximum earned UP or DOWN frontier in streaming terrain')
    q.add_argument('--academy-course-episode-frames',type=int,default=1200,
                   help='streaming episode time; ordinary births retain 700 frames')
    q.add_argument('--resume',default='',help='auto for same-run checkpoint, or an authorised experiment fork')
    q.add_argument('--research-parent-source',default='',
                   help='explicit cross-source fork only: full 40-char parent Git SHA (requires parent SHA256)')
    q.add_argument('--research-parent-sha256',default='',
                   help='explicit cross-source fork only: SHA256 of immutable parent final.pkl')
    q.add_argument('--check-resume-only',action='store_true',
                   help='validate source/parent/checkpoint now, exit before any PPO training')
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
    if args.cmd=='terrain-exam':
        from .evaluation import evaluate_terrain_course
        paths=evaluate_terrain_course(
            policy=Path(args.policy),output=Path(args.output),
            worlds=args.worlds,horizon=args.horizon,speed=args.speed,
            heights_cm=tuple(args.heights_cm),seed=args.seed,device=args.device)
        for path in paths:
            print(path)
        return 0
    if args.cmd=='capability-exam':
        from .evaluation import evaluate_unified_capabilities
        paths=evaluate_unified_capabilities(
            policy=Path(args.policy),output=Path(args.output),
            worlds=args.worlds,horizon=args.horizon,speed=args.speed,
            seed=args.seed,device=args.device,quick=args.quick,
            terrain_matrix=args.terrain_matrix,
            matrix_grades_deg=args.matrix_grades_deg,
            matrix_heights_cm=args.matrix_heights_cm,
            speed_ladder=args.speed_ladder,
            ladder_speeds=args.ladder_speeds,
            legacy_main_baseline=args.legacy_main_baseline)
        for path in paths:
            print(path)
        return 0
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
    if args.cmd=='physical-quality':
        from .physical_quality import report_main
        av=[]
        for path in args.exam: av += ['--exam',str(path)]
        av += ['--output',str(args.output)]
        return report_main(av)
    if args.cmd=='report':
        from .results import write_report
        for output in write_report(Path(args.root),bundle=not args.no_bundle):
            print(output)
        return 0
    if args.cmd=='train':
        from .run import train_one
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
