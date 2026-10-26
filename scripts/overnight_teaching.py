#!/usr/bin/env python3
"""Source-pinned overnight EMU curriculum tournament.

Plan is the default. --run explicitly opts into sequential CUDA campaigns.
No Git source transitions, external checkpoint forks, separate PPO brains,
silent run-dir overwrite, or unpublished champion/showcase modifications.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
BRANCH='feature/v5023-capability-growth'
FOCUS=('command','resilience','slope','step','transition','control')


def git(*args):
    return subprocess.check_output(
        ['git',*args],cwd=ROOT,text=True,stderr=subprocess.STDOUT).strip()


def cases(seed: int, foundation_steps: int, long_steps: int):
    # Cheap-ish 10M question: which lesson recipe develops walking best?
    screen=[
        {'name':f'screen_{mix}_s{seed}','recipe':mix,
         'schedule':'balanced','steps':foundation_steps}
        for mix in ('main','command-heavy','curious')]
    # Next question: after an equal cold foundation, which teaching
    # distribution across the whole development produces a better policy?
    # Deliberately INDEPENDENT cold runs; this does NOT fake checkpoint fork
    # equivalence by copying neural weights without optimiser/teacher state.
    mature=[
        {'name':f'growth_{schedule}_s{seed}','recipe':'curious',
         'schedule':schedule,'steps':long_steps}
        for schedule in ('balanced','staged','terrain-early')]
    return screen+mature


def cmd_for(case,root,seed,foundation_steps,resume):
    cmd=[
        sys.executable,'-m','emu.cli','train',
        '--run-dir',str(root/case['name']),
        '--profile','sim2real','--terrain-actor',
        '--terrain-step-max-m','0.04','--unified-capabilities',
        '--foundation-mix',case['recipe'],
        '--foundation-steps',str(foundation_steps),
        '--teaching-schedule',case['schedule'],
        '--steps',str(case['steps']),
        '--envs','2048','--seed',str(seed),'--device','cuda']
    if resume:
        cmd+=['--resume','auto']
    return cmd


def read_latest_csv(path):
    if not path.is_file():
        return {}
    with path.open(newline='',encoding='utf-8-sig') as inp:
        return next(reversed(list(csv.DictReader(inp))),{})


def summary(root,all_cases,sha):
    rows=[]
    for case in all_cases:
        run=root/case['name']
        metrics=read_latest_csv(run/'metrics.csv')
        exam=read_latest_csv(run/'fixed_eval.csv')
        item={
            'name':case['name'],'recipe':case['recipe'],
            'schedule':case['schedule'],'target':case['steps'],
            'step':metrics.get('step',''),
            'validated_final_present':(run/'checkpoints/final.pkl').is_file(),
            'reward':metrics.get('reward_mean',''),
            'fall_rate':metrics.get('fall_rate',''),
            'actual_vx':metrics.get('vx_mean',''),
            'requested_vx':metrics.get('cmd_vx_mean',''),
            'control_shaft_j_per_m':metrics.get('control_shaft_j_per_m',''),
            'fixed_exam_score':exam.get('eval_broad_score',''),
            'fixed_exam_core_survival':exam.get('eval_core_survival',''),
            'fixed_exam_core_vx_mae':exam.get('eval_core_vx_mae',''),
            'slope_near_attempts':sum(float(metrics.get('slope_near_attempts_'+k,0) or 0)
               for k in ('uphill','downhill','cross_left','cross_right')),
            'step_near_attempts':sum(float(metrics.get('terrain_near_attempts_'+k,0) or 0)
               for k in ('up','down')),
            'focus_episodes':{
                k:metrics.get('focus_episodes_'+k,'')
                for k in FOCUS},
            'focus_alloc':{
                k:metrics.get('focus_alloc_'+k,'')
                for k in FOCUS},
        }
        rows.append(item)
    result={'source_sha':sha,'runs':rows,
            'warning':('Training results are not proof of policy quality. '
                       'Use matched held-out exam, completed physical evidence, '
                       'and multiple seeds before ranking curricula.')}
    root.mkdir(parents=True,exist_ok=True)
    (root/'overnight_summary.json').write_text(
        json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    fieldnames=['name','recipe','schedule','target','step','validated_final_present',
                'reward','fall_rate','actual_vx','requested_vx',
                'control_shaft_j_per_m','fixed_exam_score',
                'fixed_exam_core_survival','fixed_exam_core_vx_mae',
                'slope_near_attempts','step_near_attempts']
    with (root/'overnight_summary.csv').open('w',newline='',encoding='utf-8') as fp:
        writer=csv.DictWriter(fp,fieldnames=fieldnames,extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)
    md=[
        '# EMU overnight teaching experiments','',
        f'Source: \`{sha}\`','',
        '| Run | Target | Actual steps | Final verified? | Falls | vx actual/cmd | Exam score |',
        '|---|---:|---:|:---:|---:|---:|---:|']
    for x in rows:
        md.append(f"| {x['name']} | {x['target']:,} | {x['step'] or '?'} "
                  f"| {'file' if x['validated_final_present'] else 'NO'} | "
                  f"{x['fall_rate'] or '?'} | {x['actual_vx'] or '?'}/"
                  f"{x['requested_vx'] or '?'} | {x['fixed_exam_score'] or '?'} |")
    md+=['','## Interpretation','',
         '- First compare matched held-out walking, support, tracking and falls.',
         '- Inspect real near-boundary command, slope and step completed attempts.',
         '- Shaft work per useful distance is a proxy, not battery energy.',
         '- Unobserved capabilities remain UNKNOWN; no automatic winner.',
         '- A single seed is discovery, not evidence of repeatability.',
         '']
    (root/'overnight_summary.md').write_text('\n'.join(md),encoding='utf-8')
    zip_path=root/'overnight_results.zip'
    with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name in ('campaign_manifest.json','overnight_summary.json',
                     'overnight_summary.csv','overnight_summary.md'):
            f=root/name
            if f.is_file():z.write(f,arcname=name)
        for case in all_cases:
            d=root/case['name']
            for filename in ('results.md','results_data.json','results_bundle.zip',
                             'metrics.csv','fixed_eval.csv','training_config.json',
                             'run_meta.json','envelope_summary.json'):
                f=d/filename
                if f.is_file(): z.write(f,arcname=case['name']+'/'+filename)
    print('Morning report:',root/'overnight_summary.md',flush=True)
    print('Shareable bundle:',zip_path,flush=True)


def execute(command,log_path,env):
    log_path.parent.mkdir(parents=True,exist_ok=True)
    print('\nRUN:',shlex.join(command),flush=True)
    with log_path.open('a',encoding='utf-8',buffering=1) as log:
        log.write('\n=== '+shlex.join(command)+' ===\n')
        with subprocess.Popen(command,cwd=ROOT,env=env,
             stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
             text=True,bufsize=1) as proc:
            for line in proc.stdout:
                sys.stdout.write(line);sys.stdout.flush()
                log.write(line)
            return_code=proc.wait()
    if return_code:
        raise RuntimeError(f'Command exited {return_code}; see {log_path}')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=
                    Path.home()/'robot_lab/emu_runs/overnight_teaching_v5023')
    ap.add_argument('--run',action='store_true',help='ACTUALLY launch all CUDA runs')
    ap.add_argument('--seed',type=int,default=733)
    ap.add_argument('--foundation-steps',type=int,choices=(10000000,20000000),
                    default=10000000)
    ap.add_argument('--long-steps',type=int,default=80000000)
    ap.add_argument('--skip-smoke',action='store_true',
                    help='explicitly skip CPU and full CUDA acceptance on resumed campaign')
    args=ap.parse_args()
    if args.long_steps<args.foundation_steps+40_000_000:
        ap.error('long run should span initial foundation and both staged shifts')
    sha=git('rev-parse','HEAD')
    branch=git('branch','--show-current')
    if branch!=BRANCH:
        ap.error('wrong branch: '+branch)
    if git('status','--porcelain'):
        ap.error('source tree is dirty; do not start an ambiguous campaign')
    root=args.root.expanduser().resolve()
    if ROOT==root or ROOT in root.parents:
        ap.error('experiment output must live outside the source checkout')
    entries=cases(args.seed,args.foundation_steps,args.long_steps)
    print('Source',sha,'\nCampaign',root,'\nCUDA 2048 worlds sequentially',flush=True)
    for entry in entries:
        resume=(root/entry['name']/'training_config.json').is_file()
        print(('RESUME' if resume else 'NEW')+':',shlex.join(
            cmd_for(entry,root,args.seed,args.foundation_steps,resume)))
    if not args.run:
        print('PLAN ONLY: run with --run after checking exact-SHA CI and CUDA readiness')
        return 0
    meta={'source_sha':sha,'seed':args.seed,
          'foundation_steps':args.foundation_steps,
          'long_steps':args.long_steps,'runs':entries}
    manifest=root/'campaign_manifest.json'
    if manifest.is_file():
        if json.loads(manifest.read_text(encoding='utf-8'))!=meta:
            ap.error('existing campaign manifest differs; new root or original settings required')
    else:
        if root.exists() and any(root.iterdir()):
            ap.error('nonempty campaign directory has no manifest; refuse overwrite')
        root.mkdir(parents=True,exist_ok=True)
        manifest.write_text(json.dumps(meta,indent=2,sort_keys=True)+'\n',
                            encoding='utf-8')
    env={**os.environ,'XLA_PYTHON_CLIENT_PREALLOCATE':'false',
         'EMU_UNIFIED_CONTINUOUS':'cold-growth-approved'}
    try:
        if not args.skip_smoke:
            execute([sys.executable,'scripts/smoke_unified_focus.py',
                     '--device','cpu','--envs','12'],
                    root/'logs/preflight_cpu.log',env)
            execute([sys.executable,'scripts/smoke_unified_focus.py',
                     '--device','cuda','--envs','2048','--full-update'],
                    root/'logs/preflight_cuda.log',env)
        for entry in entries:
            dest=root/entry['name']
            if dest.exists() and not (dest/'training_config.json').is_file():
                raise RuntimeError(f'Nonempty/uninitialized run root: {dest}')
            cmd=cmd_for(entry,root,args.seed,args.foundation_steps,
                        resume=dest.exists())
            execute(cmd,root/'logs'/(entry['name']+'.log'),env)
    finally:
        # On failure the finished runs still get an honest partial bundle.
        summary(root,entries,sha)
    print('Overnight tournament complete. All policy files remain local to run roots.')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
