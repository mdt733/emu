#!/usr/bin/env python3
"""Source-pinned cold-start comparison for the release-polish branch.

Plan only by default; --run explicitly starts sequential GPU experiments.
Uses a distinct default output root from the frozen training campaign.
This standalone script contains the results/reporting helpers it needs:
the obsolete v5023 tournament launcher is no longer a runtime dependency.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import shutil
import shlex
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BRANCH = 'feature/v5024-release-polish'
FOCUS = ('command', 'resilience', 'slope', 'step', 'transition', 'control')

def git(*args):
    return subprocess.check_output(
        ['git',*args],cwd=ROOT,text=True,stderr=subprocess.STDOUT).strip()


def cmd_for(case,root,seed,foundation_steps,resume):
    cmd=[
        sys.executable,'-m','emu.cli','train',
        '--run-dir',str(root/case['name']),
        '--profile','sim2real','--terrain-actor',
        '--terrain-step-max-m','0.04','--unified-capabilities',
        '--foundation-mix',case['recipe'],
        '--foundation-steps',str(foundation_steps),
        '--teaching-schedule',case['schedule'],
        '--fixed-eval-every-steps','10000000',
        '--steps',str(case['steps']),
        '--envs','2048','--seed',str(seed),'--device','cuda']
    if resume:
        cmd+=['--resume','auto']
    return cmd


def read_rows(path):
    if not path.is_file():
        return []
    with path.open(newline='',encoding='utf-8-sig') as inp:
        return list(csv.DictReader(inp))


def safe_float(value):
    try:
        number=float(value)
        return number if math.isfinite(number) else None
    except (TypeError,ValueError):
        return None


def near_attempt_total(rows,prefix,names):
    # Closed-episode counts are per-ROLLOUT. The teacher clears its moving
    # windows after decisions, but those counters must NOT be read from
    # only the final metrics row.
    return sum(safe_float(row.get(prefix+name)) or 0.
               for row in rows for name in names)


def campaign_eta(root,cases):
    # Actual observed wall throughput, sampled from completed/active CSV.
    # Exams/compilation are extra, so this is explicitly a lower bound.
    speeds=[]
    remaining=0.
    for case in cases:
        rows=read_rows(root/case['name']/'metrics.csv')
        seen=max((safe_float(r.get('step')) or 0. for r in rows),default=0.)
        remaining+=max(0.,case['steps']-seen)
        if rows:
            speed=safe_float(rows[-1].get('steps_per_sec'))
            if speed is not None and speed>0.:
                speeds.append(speed)
    if not speeds:
        return 'ETA unavailable until measured GPU training throughput'
    speed=sorted(speeds)[len(speeds)//2]
    return (f'Observed throughput ~{speed:,.0f} world steps/s; '
            f'remaining TRAINING at least {remaining/speed/3600:.1f} h '
            '(plus compilation, frozen exams and restarts)')


def read_latest_csv(path):
    if not path.is_file():
        return {}
    with path.open(newline='',encoding='utf-8-sig') as inp:
        return next(reversed(list(csv.DictReader(inp))),{})


def summary(root,all_cases,sha):
    rows=[]
    for case in all_cases:
        run=root/case['name']
        raw_rows=read_rows(run/'metrics.csv')
        metrics=raw_rows[-1] if raw_rows else {}
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
            'policy_requested_vx':metrics.get('policy_cmd_vx_mean',''),
            'policy_requested_vy':metrics.get('policy_cmd_vy_mean',''),
            'operator_requested_vy':metrics.get('operator_cmd_vy_mean',''),
            'velocity_i_x':metrics.get('velocity_i_x_mean',''),
            'velocity_i_y':metrics.get('velocity_i_y_mean',''),
            'control_shaft_j_per_m':metrics.get('control_shaft_j_per_m',''),
            'fixed_exam_score':exam.get('eval_broad_score',''),
            'fixed_exam_core_survival':exam.get('eval_core_survival',''),
            'fixed_exam_core_vx_mae':exam.get('eval_core_vx_mae',''),
            'slope_near_attempts':near_attempt_total(
                raw_rows,'slope_near_attempts_',
                ('uphill','downhill','cross_left','cross_right')),
            'step_near_attempts':near_attempt_total(
                raw_rows,'terrain_near_attempts_',('up','down')),
            'rollback_count':metrics.get('rollback_count',''),
            'focus_step_share':{
                k:metrics.get('focus_step_share_'+k,'') for k in FOCUS},
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
                'policy_requested_vx','policy_requested_vy',
                'operator_requested_vy','velocity_i_x','velocity_i_y',
                'control_shaft_j_per_m','fixed_exam_score',
                'fixed_exam_core_survival','fixed_exam_core_vx_mae',
                'slope_near_attempts','step_near_attempts','rollback_count']
    with (root/'overnight_summary.csv').open('w',newline='',encoding='utf-8') as fp:
        writer=csv.DictWriter(fp,fieldnames=fieldnames,extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)
    md=[
        '# EMU v5024 foundation and steering comparison','',
        f'Source: {sha}','',
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
         '- Per-run training reward/fall/vx are NOT matched across teaching mixes.',
         '- See frozen_exams_by_age.csv for comparable same-age measurements.',
         '']
    (root/'overnight_summary.md').write_text('\n'.join(md),encoding='utf-8')
    # Comparable windows, not only end-of-run scores. Every report row is
    # the actual most recent on-policy observation AT OR BEFORE that age.
    # Missing or incomplete phases remain unknown, never interpolated.
    stages=[]
    milestones=(10_000_000,20_000_000,40_000_000,60_000_000,80_000_000)
    for case in all_cases:
        path=root/case['name']/'metrics.csv'
        if not path.is_file():
            continue
        with path.open(newline='',encoding='utf-8-sig') as fp:
            series=list(csv.DictReader(fp))
        for target in milestones:
            if target>case['steps']:
                continue
            eligible=[]
            for record in series:
                try:
                    step=float(record.get('step','nan'))
                except (TypeError,ValueError):
                    continue
                if step<=target and step>=0.:
                    eligible.append((step,record))
            if not eligible:
                continue
            observed,record=max(eligible,key=lambda x:x[0])
            stages.append({
                'run':case['name'],'milestone':target,
                'actual_step':int(observed),
                'fall_rate':record.get('fall_rate',''),
                'vx_mean':record.get('vx_mean',''),
                'cmd_vx_mean':record.get('cmd_vx_mean',''),
                'frontier_vx_hi':record.get('frontier_vx_hi',''),
                'slope_max_deg':record.get('slope_max_deg',''),
                'terrain_height_m':record.get('terrain_height_m',''),
                'control_shaft_j_per_m':record.get('control_shaft_j_per_m',''),
                'focus_alloc_command':record.get('focus_alloc_command',''),
                'focus_alloc_slope':record.get('focus_alloc_slope',''),
                'focus_alloc_step':record.get('focus_alloc_step','')})
    if stages:
        with (root/'progress_by_age.csv').open('w',newline='',encoding='utf-8') as fp:
            out=csv.DictWriter(fp,fieldnames=list(stages[0]))
            out.writeheader();out.writerows(stages)
    # Matched FROZEN evaluation only. No on-policy curriculum statistics
    # masquerade as comparable test results. Keep actual steps and missing
    # cohorts visible; do not invent scores at unsampled ages.
    frozen=[]
    for case in all_cases:
        exams=read_rows(root/case['name']/'fixed_eval.csv')
        for rec in exams:
            measured=safe_float(rec.get('step'))
            if measured is None:
                continue
            frozen.append({
                'run':case['name'],'actual_step':int(measured),
                'eval_broad_score':rec.get('eval_broad_score',''),
                'eval_core_survival':rec.get('eval_core_survival',''),
                'eval_core_vx_mae':rec.get('eval_core_vx_mae',''),
                'eval_dynamic80_survival':rec.get('eval_dynamic80_survival',''),
                'eval_resilience75_survival':rec.get('eval_resilience75_survival',''),
                'eval_policy_step':rec.get('eval_policy_step','')})
    if frozen:
        with (root/'frozen_exams_by_age.csv').open('w',newline='',encoding='utf-8') as fp:
            writer=csv.DictWriter(fp,fieldnames=list(frozen[0]))
            writer.writeheader();writer.writerows(frozen)
    zip_path=root/'overnight_results.zip'
    with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name in ('campaign_manifest.json','overnight_summary.json',
                     'overnight_summary.csv','overnight_summary.md',
                     'progress_by_age.csv','frozen_exams_by_age.csv',
                     'pip_freeze.txt','nvidia_smi.txt','disk_space.txt',
                     'campaign_status.json','preflight_pass.json'):
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


def execute(command,log_path,env,*,campaign=None):
    log_path.parent.mkdir(parents=True,exist_ok=True)
    print('\nRUN:',shlex.join(command),flush=True)
    with log_path.open('a',encoding='utf-8',buffering=1) as log:
        log.write('\n=== '+shlex.join(command)+' ===\n')
        last_eta=0.0
        seen_header=False
        with subprocess.Popen(command,cwd=ROOT,env=env,
             stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
             text=True,bufsize=1,start_new_session=True) as proc:
            for line in proc.stdout:
                sys.stdout.write(line);sys.stdout.flush()
                log.write(line)
                if campaign is not None and ' step=' in line:
                    now=time.monotonic()
                    if now-last_eta>300.:
                        last_eta=now
                        folder=Path(command[command.index('--run-dir')+1])
                        data=folder/'metrics.csv'
                        if data.is_file() and not seen_header:
                            with data.open(newline='',encoding='utf-8-sig') as fp:
                                reader=csv.DictReader(fp)
                                fields=set(reader.fieldnames or [])
                            expected={'focus_alloc_command',
                                      'focus_episodes_command',
                                      'focus_step_share_command',
                                      'slope_near_attempts_uphill',
                                      'terrain_near_attempts_up',
                                      'rollback_count','eval_core_vx_mae'}
                            # eval_core_vx_mae is present only after the
                            # first frozen exam, not the first PPO summary.
                            expected.remove('eval_core_vx_mae')
                            missing=expected-fields
                            if missing:
                                # The train CLI launches another Python
                                # subprocess. Kill the whole candidate group
                                # before moving to the next GPU experiment.
                                os.killpg(proc.pid,signal.SIGTERM)
                                raise RuntimeError(
                                    f'Unrecognized training telemetry: {sorted(missing)}')
                            seen_header=True
                        eta=campaign_eta(*campaign)
                        print('CAMPAIGN ETA:',eta,flush=True)
                        log.write('CAMPAIGN ETA: '+eta+'\n')
            return_code=proc.wait()
    if return_code:
        raise RuntimeError(f'Command exited {return_code}; see {log_path}')



def candidates(seed: int, steps: int):
    return [
        {'name':f'main_balanced_s{seed}','recipe':'main',
         'schedule':'balanced','steps':steps},
        {'name':f'commandheavy_balanced_s{seed}','recipe':'command-heavy',
         'schedule':'balanced','steps':steps},
        {'name':f'commandheavy_steering_s{seed}','recipe':'command-heavy',
         'schedule':'steering','steps':steps},
    ]


def select_arms(all_arms, requested):
    """Select exactly A/B/C by name in canonical experiment order."""
    if len(requested) != len(set(requested)):
        raise ValueError('each --arms label may appear only once')
    return [arm for key, arm in zip('ABC', all_arms)
            if key in requested]


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
                        default=Path.home()/'robot_lab/emu_runs/v5024_polish_study')
    parser.add_argument('--steps',type=int,default=60_000_000)
    parser.add_argument('--foundation-steps',type=int,choices=(10_000_000,20_000_000),
                        default=10_000_000)
    parser.add_argument('--seed',type=int,default=733)
    parser.add_argument('--arms',nargs='+',choices=('A','B','C'),
                        default=('A','B','C'),
                        help='A main+balanced, B command+balanced, C command+steering; default all')
    parser.add_argument('--run',action='store_true',help='START the CUDA experiments')
    parser.add_argument('--skip-smoke',action='store_true',
                        help='only after same-source stamped CPU and CUDA preflight')
    args=parser.parse_args()
    if args.steps<args.foundation_steps+40_000_000:
        parser.error('long runs must cover >=40M specialist development steps')
    if not 0<=args.seed<2**31:
        parser.error('seed must be in [0,2**31)')
    if len(args.arms)!=len(set(args.arms)):
        parser.error('--arms must not repeat the same arm')
    branch=git('branch','--show-current')
    if branch!=BRANCH:
        parser.error('expected branch '+BRANCH+', got '+branch)
    if git('status','--porcelain'):
        parser.error('dirty source; refuse ambiguous reproducibility')
    sha=git('rev-parse','HEAD')
    root=args.root.expanduser().resolve()
    if root==ROOT or ROOT in root.parents:
        parser.error('training output must be OUTSIDE repository')
    arms=select_arms(candidates(args.seed,args.steps),args.arms)
    print('SOURCE',sha,'\nROOT',root,
          f'\n{len(arms)} sequential independent cold policies')
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
              'intent':('compare foundation (A/B) and steering (B/C)'
                        if len(arms)==3 else
                        'selected independent cold-start replication recipes')}
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
            try:
                if dest.exists() and not (dest/'training_config.json').is_file():
                    raise RuntimeError('unexpected non-resumable run contents: '+str(dest))
                cmd=cmd_for(arm,root,args.seed,args.foundation_steps,
                            has_checkpoint(dest))
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
    print('All selected arms completed; frozen exams are in overnight_results.zip')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
