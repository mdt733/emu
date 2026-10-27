#!/usr/bin/env python3
"""v5025 exploration: one shared 20M cold foundation, then 12 matched 20M forks.

Explicit --run is required for CUDA. The previous source-pinned 180M study
and public showcase remain untouched. All branches inherit one *new*
same-source checkpoint, including policy, optimiser, normaliser and teacher.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
import v5024_campaign as base

ROOT=Path(__file__).resolve().parents[1]
BRANCH='feature/v5024-release-polish'
START=20_000_000
TARGET=40_000_000

def arms(seed=734):
    entries=[
        ('balanced_off','balanced',1.,.4),
        ('steering_off','steering',1.,.4),
        ('terrain_off','terrain-early',1.,.4),
        ('staged_off','staged',1.,.4),
        ('balanced_i02_15','balanced',1.5,.2),
        ('balanced_i04_15','balanced',1.5,.4),
        ('balanced_i08_15','balanced',1.5,.8),
        ('balanced_i02_20','balanced',2.,.2),
        ('balanced_i04_20','balanced',2.,.4),
        ('balanced_i08_20','balanced',2.,.8),
        ('steering_i04_20','steering',2.,.4),
        ('terrain_i04_20','terrain-early',2.,.4),
    ]
    return [dict(name=f'screen_{label}_s{seed}',recipe='main',
                 schedule=schedule,steps=TARGET,velocity_i_multiplier=mult,
                 velocity_i_ramp_s=ramp)
            for label,schedule,mult,ramp in entries]

def command(arm,root,seed,*,parent=None,resume=False):
    cmd=base.cmd_for(arm,root,seed,START,resume)
    cmd += ['--velocity-i-multiplier',str(arm.get('velocity_i_multiplier',1.)),
            '--velocity-i-ramp-s',str(arm.get('velocity_i_ramp_s',.4))]
    if parent is not None and not resume:
        cmd += ['--resume',str(parent)]
    return cmd

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path.home()/'robot_lab/emu_runs/v5025_screen_seed734')
    p.add_argument('--run',action='store_true',help='ACTUALLY RUN CUDA')
    p.add_argument('--skip-smoke',action='store_true')
    a=p.parse_args()
    if base.git('branch','--show-current')!=BRANCH or base.git('status','--porcelain'):
        p.error('requires clean '+BRANCH+' checkout')
    sha=base.git('rev-parse','HEAD')
    root=a.root.expanduser().resolve()
    if root==ROOT or ROOT in root.parents:
        p.error('results root must be outside source tree')
    seed=734
    shared=dict(name='shared_main_foundation_s734',recipe='main',schedule='balanced',
                steps=START,velocity_i_multiplier=1.,velocity_i_ramp_s=.4)
    screen=arms(seed)
    all_arms=[shared,*screen]
    parent=root/shared['name']/'checkpoints/final.pkl'
    manifest=dict(source_sha=sha,seed=seed,foundation_steps=START,
                  shared=shared,runs=screen,experiment='single source shared-20M plus 12 20M forks')
    print('SOURCE',sha,'ROOT',root)
    print('FOUNDATION',shared['name'],'0 -> 20M')
    for arm in screen:
        print(arm['name'],'20M -> 40M',arm['schedule'],
              'I multiplier',arm['velocity_i_multiplier'],
              'ramp',arm['velocity_i_ramp_s'])
    print('TOTAL 260,000,000 world steps')
    if not a.run:
        print('PLAN ONLY. --run is required to start CUDA.')
        return 0
    root.mkdir(parents=True,exist_ok=True)
    path=root/'campaign_manifest.json'
    if path.is_file():
        if json.loads(path.read_text())!=manifest:
            p.error('source/config changed: use NEW data root')
    else:
        if any(root.iterdir()):
            p.error('nonempty root without matching campaign manifest')
        path.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    env={**os.environ,'EMU_UNIFIED_CONTINUOUS':'cold-growth-approved',
         'XLA_PYTHON_CLIENT_PREALLOCATE':'false'}
    errors=[]
    outcomes={}
    try:
        if not a.skip_smoke:
            base.execute([sys.executable,'scripts/smoke_unified_focus.py',
                          '--device','cpu','--envs','12',
                          '--velocity-i-ramp-s','0.4',
                          '--velocity-i-multiplier','2'],
                         root/'logs/preflight_cpu.log',env)
            base.execute([sys.executable,'scripts/smoke_unified_focus.py',
                          '--device','cuda','--envs','2048','--full-update',
                          '--velocity-i-ramp-s','0.4',
                          '--velocity-i-multiplier','2'],
                         root/'logs/preflight_cuda.log',env)
            (root/'preflight_pass.json').write_text(
                json.dumps({'source_sha':sha,'study':'v5025_screen'})+'\n')
        else:
            stamp=root/'preflight_pass.json'
            saved=json.loads(stamp.read_text()) if stamp.is_file() else {}
            if saved.get('source_sha')!=sha:
                p.error('no exact-source CPU/CUDA smoke stamp')
        base.collect_runtime(root)
        dest=root/shared['name']
        base.execute(command(shared,root,seed,resume=base.has_checkpoint(dest)),
                     root/'logs/foundation.log',env,campaign=(root,all_arms))
        if not parent.is_file():
            raise RuntimeError('shared foundation did not produce final.pkl')
        outcomes[shared['name']]='completed'
        (root/'campaign_status.json').write_text(json.dumps(outcomes,indent=2)+'\n')
        for arm in screen:
            # Prevent updating the checkout while a source-pinned campaign
            # is active. A different Git commit invalidates parent lineage.
            if base.git('rev-parse','HEAD')!=sha or base.git('status','--porcelain'):
                raise RuntimeError('source changed during campaign; stop, do not fork')
            dest=root/arm['name']
            try:
                if dest.exists() and not (dest/'training_config.json').is_file():
                    raise RuntimeError('non-resumable directory: '+str(dest))
                resume=base.has_checkpoint(dest)
                fork_env=dict(env)
                if not resume:
                    fork_env['EMU_UNIFIED_SCREEN_FORK']='same-source-20m-v1'
                base.execute(command(arm,root,seed,parent=parent,resume=resume),
                             root/'logs'/(arm['name']+'.log'),fork_env,
                             campaign=(root,all_arms))
                outcomes[arm['name']]='completed'
            except Exception as exc:
                errors.append(arm['name'])
                outcomes[arm['name']]='failed: '+str(exc)
                print('FAILED',arm['name'],str(exc),file=sys.stderr,flush=True)
            (root/'campaign_status.json').write_text(json.dumps(outcomes,indent=2)+'\n')
    finally:
        base.summary(root,all_arms,sha)
    if errors:
        print('FAILED ARMS:',', '.join(errors),file=sys.stderr)
        return 1
    print('Completed 20M foundation plus 12 x 20M branches.')
    return 0

if __name__=='__main__':
    raise SystemExit(main())
