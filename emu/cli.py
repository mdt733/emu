#!/usr/bin/env python3
"""Public command line for modern EMU."""
from __future__ import annotations

import argparse
from pathlib import Path

from .contracts import VERSION, seed_root
from .doctor import doctor


def main(argv=None) -> int:
    p=argparse.ArgumentParser(
        prog='emu', description=f'EMU {VERSION} — dynamically unstable biped RL',
        epilog='One source-pinned lineage: emu train --help. Unified GPU training remains launch-guarded.')
    p.add_argument('--version',action='version',version=VERSION)
    sub=p.add_subparsers(dest='cmd',required=True)

    q=sub.add_parser('view',help='open the read-only capability-field viewer'); q.add_argument('--seed',type=int,default=733); q.add_argument('--run-root',default=''); q.add_argument('--run-dir',default=''); q.add_argument('--port',type=int,default=8765)
    q=sub.add_parser('doctor',help='validate Python/JAX/MJX and core contracts')
    q.add_argument('--require-gpu',action=argparse.BooleanOptionalAction,default=True)
    q.add_argument('--no-mjx-smoke',action='store_true')
    q.add_argument('--domain-smoke',action='store_true',
                   help='also compile/step an 8-world randomized MJX model')
    q=sub.add_parser('posture-report',help='read-only height drift report for saved metrics.csv')
    q.add_argument('--root',default='')
    q=sub.add_parser('prism-exam',
        help='read-only frozen 360 actor on true MJX prisms: raw support, terrain and progress')
    q.add_argument('--policy',type=Path,required=True)
    q.add_argument('--output',type=Path,required=True,help='new or empty directory outside policy')
    q.add_argument('--worlds',type=int,default=4)
    q.add_argument('--horizon',type=int,default=300)
    q.add_argument('--speed',type=float,default=.12,help='positive forward command in m/s')
    q.add_argument('--seed',type=int,default=1701)
    q.add_argument('--device',choices=('cpu','cuda','gpu','auto'),default='cpu')
    q.add_argument('--mode',choices=('flat','slope','step','mixed'),default='mixed')
    q.add_argument('--grade-deg',type=float,default=3.,help='requested slope frontier, not a fixed fixture')
    q.add_argument('--lip-cm',type=float,default=1.,help='requested signed-step frontier, not a fixed fixture')
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
                   help='historical 232 input; unsupported by the v5045 prism-only trainer')
    q.add_argument('--terrain-actor',action='store_true',
                   help='required: current 360-input 4x4 depth-ToF policy')
    q.add_argument('--terrain-step-max-m',type=float,default=0.,
                   help='maximum signed procedural-prism step frontier, up to 0.08m')
    q.add_argument('--unified-capabilities',action='store_true',help='single-policy capability specialists')
    q.add_argument('--terrain-carpet',action='store_true',
                   help='required: one endless procedural 3x3 prism source carpet')
    q.add_argument('--carpet-prisms',action='store_true',
                   help='required: 18 real convex-prism MJX contact solids (GPU preflight required)')
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
                   help='required: Academy episodes share the same streaming prism carpet')
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
        from .capability_viewer import field_main as view
        av=['--port',str(args.port)]
        if args.run_dir: av += ['--run-dir',args.run_dir]
        else: av += ['--run-root',str(Path(args.run_root).expanduser() if args.run_root else seed_root(args.seed))]
        return int(view(av) or 0)
    if args.cmd=='posture-report':
        from .results import posture_report
        return posture_report(Path(args.root) if args.root else Path.home()/'robot_lab'/'emu_runs')
    if args.cmd=='prism-exam':
        from .prism_exam import evaluate_prism_policy
        paths=evaluate_prism_policy(
            policy=args.policy, output=args.output, worlds=args.worlds,
            horizon=args.horizon, speed=args.speed, seed=args.seed,
            device=args.device, mode=args.mode,
            grade_deg=args.grade_deg, lip_cm=args.lip_cm)
        for path in paths:
            print(path)
        return 0
    if args.cmd=='report':
        from .results import write_report
        for output in write_report(Path(args.root),bundle=not args.no_bundle):
            print(output)
        return 0
    if args.cmd=='train':
        from .run import train_one
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
