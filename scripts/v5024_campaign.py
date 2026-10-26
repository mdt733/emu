#!/usr/bin/env python3
"""v5024: controlled cold-start 3x60M foundation-and-steering study.

Default --plan prints without touching disk or CUDA. Requires --run to train.
Shares the proven source-pinned log, safe-resume and morning report utilities
with scripts/overnight_teaching.py; does not copy checkpoints across arms.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from overnight_teaching import (BRANCH as OLD_BRANCH, cmd_for,
                                execute, summary, campaign_eta, git)

ROOT=Path(__file__).resolve().parents[1]
BRANCH='feature/v5024-terrain-control'


def candidates(seed: int, steps: int):
    return [
        {'name':f'main_balanced_s{seed}','recipe':'main',
         'schedule':'balanced','steps':steps},
        {'name':f'commandheavy_balanced_s{seed}','recipe':'command-heavy',
         'schedule':'balanced','steps':steps},
        {'name':f'commandheavy_steering_s{seed}','recipe':'command-heavy',
         'schedule':'steering','steps':steps},
    ]


def has_checkpoint(directory: Path):
    ck=directory/'checkpoints'
    return (any((ck/name).is_file() for name in ('latest.pkl','final.pkl'))
            or any(ck.glob('snapshot_step_*.pkl')))


def collect_runtime(root: Path):
    for filename,command in [
        ('pip_freeze.txt',[sys.executable,'-m','pip','freeze']),
        ('nvidia_smi.txt',['nvidia-smi','-q']),
        ('disk_space.txt',['df','-h',str(root)]),
    ]:
        result=subprocess.run(command,text=True,capture_output=True,
                              cwd=ROOT,check=False)
        (root/filename).write_text(result.stdout+result.stderr,encoding='utf-8')
        if result.returncode:
            print('WARNING: unable to collect '+filename,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,
                        default=Path.home()/'robot_lab/emu_runs/v5024_cold_steering')
    parser.add_argument('--steps',type=int,default=60_000_000)
    parser.add_argument('--foundation-steps',type=int,choices=(10_000_000,20_000_000),
                        default=10_000_000)
    parser.add_argument('--seed',type=int,default=733)
    parser.add_argument('--run',action='store_true',help='START the CUDA experiments')
    parser.add_argument('--skip-smoke',action='store_true',
                        help='only after same-source stamped CPU and CUDA preflight')
    args=parser.parse_args()
    if args.steps<args.foundation_steps+40_000_000:
        parser.error('long runs must cover >=40M specialist development steps')
    if not 0<=args.seed<2**31:
        parser.error('seed must be in [0,2**31)')
    branch=git('branch','--show-current')
    if branch!=BRANCH:
        parser.error('expected branch '+BRANCH+', got '+branch)
    if git('status','--porcelain'):
        parser.error('dirty source; refuse ambiguous reproducibility')
    sha=git('rev-parse','HEAD')
    root=args.root.expanduser().resolve()
    if root==ROOT or ROOT in root.parents:
        parser.error('training output must be OUTSIDE repository')
    arms=candidates(args.seed,args.steps)
    print('SOURCE',sha,'\nROOT',root,'\n3 sequential independent cold policies')
    for arm in arms:
        print('  ',arm['name'],arm['steps'],'resume=',has_checkpoint(root/arm['name']))
        print('   ', ' '.join(cmd_for(arm,root,args.seed,args.foundation_steps,
                                     has_checkpoint(root/arm['name']))))
    print('TOTAL',sum(x['steps'] for x in arms),'world steps')
    if not args.run:
        print('PLAN ONLY. --run is required to start CUDA.')
        return 0
    manifest={'source_sha':sha,'seed':args.seed,
              'foundation_steps':args.foundation_steps,'runs':arms,
              'intent':'compare foundation (A/B) and steering (B/C)'}
    root.mkdir(parents=True,exist_ok=True)
    manifest_path=root/'campaign_manifest.json'
    if manifest_path.is_file():
        if json.loads(manifest_path.read_text(encoding='utf-8'))!=manifest:
            parser.error('run manifest changed; use a NEW run root')
    else:
        if any(root.iterdir()):
            parser.error('nonempty root without manifest; refuse to overwrite')
        manifest_path.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n',
                                 encoding='utf-8')
    env={**os.environ,
         'EMU_UNIFIED_CONTINUOUS':'cold-growth-approved',
         'XLA_PYTHON_CLIENT_PREALLOCATE':'false'}
    if args.skip_smoke:
        stamp=root/'preflight_pass.json'
        saved=json.loads(stamp.read_text()) if stamp.is_file() else {}
        if saved.get('source_sha')!=sha:
            parser.error('no matching CPU+CUDA preflight stamp on exact SHA')
    collect_runtime(root)
    print(f'FREE DISK {shutil.disk_usage(root).free/2**30:.1f} GiB',flush=True)
    outcome={}
    errors=[]
    try:
        if not args.skip_smoke:
            start=time.monotonic()
            execute([sys.executable,'scripts/smoke_unified_focus.py',
                     '--device','cpu','--envs','12'],
                    root/'logs/preflight_cpu.log',env)
            execute([sys.executable,'scripts/smoke_unified_focus.py',
                     '--device','cuda','--envs','2048','--full-update'],
                    root/'logs/preflight_cuda.log',env)
            (root/'preflight_pass.json').write_text(
                json.dumps({'source_sha':sha,
                            'elapsed_s':time.monotonic()-start},indent=2)+'\n')
        for arm in arms:
            print('CAMPAIGN ETA:',campaign_eta(root,arms),flush=True)
            dest=root/arm['name']
            if dest.exists() and not (dest/'training_config.json').is_file():
                raise RuntimeError('unexpected non-resumable run contents: '+str(dest))
            cmd=cmd_for(arm,root,args.seed,args.foundation_steps,
                        has_checkpoint(dest))
            try:
                execute(cmd,root/'logs'/(arm['name']+'.log'),env,
                        campaign=(root,arms))
                outcome[arm['name']]='completed'
            except Exception as exc:
                outcome[arm['name']]=f'failed: {exc}'
                errors.append(arm['name'])
                print(f'FAILED {arm["name"]}: {exc}; moving to next arm',
                      file=sys.stderr,flush=True)
            (root/'campaign_status.json').write_text(
                json.dumps(outcome,indent=2,sort_keys=True)+'\n')
    finally:
        summary(root,arms,sha)
    if errors:
        print('Failed arms:',', '.join(errors),file=sys.stderr)
        return 1
    print('All v5024 arms completed; held-out exams are in overnight_results.zip')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
