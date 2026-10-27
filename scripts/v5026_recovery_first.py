#!/usr/bin/env python3
"""One-actor, recovery-first causal research campaign (P1/P2; no tmux).

One new 20M cold-start source, then three 20M matched P1 forks, then
three 20M matched P2 forks from the selected P1 checkpoint: 140M total.
All arms share 360 observations, six motors and a single PPO optimiser.
Experiments remain source-pinned, checkpoint-restorable, sequential and
in a foreground terminal. No main/showcase writes. Plan-only by default.

P0's cross-schema Pinnacle physical exam is deliberately NOT claimed here:
that needs the protected main source and the original 232 actor contract.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
import v5024_campaign as base

ROOT = Path(__file__).resolve().parents[1]
BRANCH = 'feature/v5024-release-polish'
SEED = 735
NURSERY = 20_000_000
P1_END = 40_000_000
P2_END = 60_000_000
DEFAULT_ROOT = Path.home()/'robot_lab/emu_runs/v5026_recovery_first_s735'

def cases():
    shared=dict(name='shared_20m',recipe='main',schedule='balanced',
                steps=NURSERY,growth='legacy',retain=0.,terrain='mixed',
                force='gentle',from_steps=0)
    p1=[
        dict(name='P1_A_360_control',recipe='main',schedule='balanced',
             steps=P1_END,from_steps=NURSERY,growth='legacy',
             retain=0.,terrain='mixed',force='gentle'),
        dict(name='P1_B_main_mature',recipe='main',schedule='balanced',
             steps=P1_END,from_steps=NURSERY,growth='main-mature',
             retain=0.,terrain='mixed',force='gentle'),
        dict(name='P1_C_mature_retention',recipe='main',schedule='balanced',
             steps=P1_END,from_steps=NURSERY,growth='main-mature',
             retain=.25,terrain='mixed',force='gentle'),
    ]
    return shared,p1

def phase2(parent_case):
    common=dict(recipe='main',schedule='balanced',steps=P2_END,
                from_steps=P1_END,growth=parent_case['growth'],
                retain=parent_case['retain'])
    return [
        dict(name='P2_D_mixed_gentle',terrain='mixed',force='gentle',**common),
        dict(name='P2_E_coherent_gentle',terrain='coherent',force='gentle',**common),
        dict(name='P2_F_coherent_aggressive',terrain='coherent',force='aggressive',**common),
    ]

def command(case,root,*,parent=None,resume=False):
    cmd=base.cmd_for(case,root,SEED,NURSERY,resume)
    cmd[cmd.index('--terrain-step-max-m')+1]='0.08'
    cmd+=['--growth-recipe',case['growth'],
          '--retention-share',str(case['retain']),
          '--terrain-lesson',case['terrain'],
          '--force-program',case['force']]
    if parent is not None and not resume:
        cmd+=['--resume',str(parent)]
    return cmd

def latest_exam(root,name):
    row=base.read_latest_csv(root/name/'fixed_eval.csv')
    def val(key):
        n=base.safe_float(row.get(key))
        return n if n is not None else float('nan')
    return {
        'name':name,
        'broad':val('eval_broad_score'),
        'core':val('eval_core_survival'),
        'dynamic':val('eval_dynamic80_survival'),
        'recovery':val('eval_resilience75_survival'),
        'vx_mae':val('eval_core_vx_mae')
    }

def select_phase1(root,p1):
    """Use fixed common evaluation, never on-policy reward/frontier.

    This is a reproducible triage choice, NOT a causal significance test.
    A candidate with no valid fixed exam or materially damaged familiar
    walking/recovery cannot become P2's shared parent. Fall back to the
    measured control; never choose a winner from absent evidence.
    """
    import math
    records=[latest_exam(root,x['name']) for x in p1]
    if not all(all(math.isfinite(x[k]) for k in
                   ('broad','core','dynamic','recovery','vx_mae'))
               for x in records):
        raise RuntimeError('P1 fixed evaluation missing or nonfinite: '+str(records))
    control=records[0]
    eligible=[records[0]]
    for item in records[1:]:
        if (item['core']>=control['core']-.03
            and item['dynamic']>=control['dynamic']-.05
            and item['recovery']>=control['recovery']-.05
            and item['vx_mae']<=max(.02,control['vx_mae']*1.15)
            and item['broad']>=control['broad']-.025):
            eligible.append(item)
    # Only promote a research recipe when it offers credible fixed-exam
    # upside; keep the old-skill control if all alternatives are worse.
    best=max(eligible,key=lambda item:(
        item['broad'] + .10*item['recovery'] + .05*item['dynamic']
        - .20*item['vx_mae']))
    winner=next(x for x in p1 if x['name']==best['name'])
    return winner,{'fixed_exams':records,
                   'passed_retention_screen':[x['name'] for x in eligible],
                   'selected':winner['name'],
                   'rule':'same-core within 3pp, dynamic/recovery within 5pp, '
                          'vx MAE <= +15% or .02m/s, broad within .025; '
                          'then weighted broad/recovery/dynamic/vx score',
                   'warning':'one seed; these fixed cohorts are a screen, '
                             'not a force-conditioned P0 exam or proof of causality'}

def guard_source(expected):
    if base.git('rev-parse','HEAD')!=expected or base.git('status','--porcelain'):
        raise RuntimeError('source checkout changed/dirty during pinned study')

def save_json(path,obj):
    path.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n',encoding='utf-8')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',action='store_true',help='perform CUDA preflight and research')
    p.add_argument('--skip-smoke',action='store_true',
                   help='use preflight stamp from exactly this source SHA')
    p.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    a=p.parse_args()
    if base.git('branch','--show-current')!=BRANCH or base.git('status','--porcelain'):
        p.error('requires a clean '+BRANCH+' checkout')
    sha=base.git('rev-parse','HEAD')
    root=a.root.expanduser().resolve()
    if root==ROOT or ROOT in root.parents:
        p.error('results must live outside Git checkout')
    shared,p1=cases()
    provisional=phase2(p1[2])
    manifest=dict(source_sha=sha,seed=SEED,foundation_steps=NURSERY,
                  mission='recovery-first controlled P1/P2 140M',
                  foundation=shared,p1=p1,
                  p2_design=[dict(name=x['name'],terrain=x['terrain'],
                                  force=x['force']) for x in provisional],
                  selection='fixed_exam_core+dynamic+recovery+vx_mae+broad',
                  prior_results='do not reuse v5025_290m or protected main checkpoints')
    print('SOURCE:',sha,'\nROOT:',root)
    print('P1: 20M cold shared foundation, then three matched 20->40M forks')
    for x in p1:print(' ',x['name'],x['growth'],'retain',x['retain'])
    print('P2: select a measured P1 parent, then three matched 40->60M forks')
    for x in provisional:print(' ',x['name'],x['terrain'],x['force'])
    print('TOTAL: 140,000,000 aggregate environment steps')
    print('NO tmux; sequential foreground CUDA; Ctrl+C terminates child.')
    print('NOTE: independent original-Pinnacle P0 exam remains outstanding.')
    if not a.run:
        print('PLAN ONLY; append --run to train.')
        return 0

    root.mkdir(parents=True,exist_ok=True)
    old_manifest=root/'campaign_manifest.json'
    if old_manifest.is_file():
        if json.loads(old_manifest.read_text(encoding='utf-8'))!=manifest:
            p.error('source/config mismatch; choose a NEW --root')
    else:
        if any(root.iterdir()):
            p.error('nonempty data root without campaign_manifest.json')
        save_json(old_manifest,manifest)
    env={**os.environ,'EMU_UNIFIED_CONTINUOUS':'cold-growth-approved',
         'XLA_PYTHON_CLIENT_PREALLOCATE':'false'}
    report_cases=[shared,*p1,*provisional]
    outcomes={}
    status=root/'campaign_status.json'
    if status.is_file():
        outcomes=json.loads(status.read_text(encoding='utf-8'))
    try:
        if not a.skip_smoke:
            for label,config in [
                ('nursery',('legacy','0','mixed','gentle')),
                ('retention',('main-mature','0.25','mixed','gentle')),
                ('terrain',('main-mature','0.25','coherent','aggressive'))]:
                growth,retention,terrain,force=config
                base.execute([sys.executable,'scripts/smoke_unified_focus.py',
                    '--device','cpu','--envs','12','--terrain-max-m','0.08',
                    '--growth-recipe',growth,'--retention-share',retention,
                    '--terrain-lesson',terrain,'--force-program',force],
                    root/'logs'/('preflight_'+label+'_cpu.log'),env)
            base.execute([sys.executable,'scripts/smoke_unified_focus.py',
                    '--device','cuda','--envs','2048','--full-update',
                    '--terrain-max-m','0.08','--growth-recipe','main-mature',
                    '--retention-share','0.25','--terrain-lesson','coherent',
                    '--force-program','aggressive'],
                    root/'logs/preflight_full_cuda.log',env)
            save_json(root/'preflight_pass.json',{'source_sha':sha,
                     'cpu_cases':3,'cuda_worlds':2048,
                     'note':'full-update acceptance, not proven long-run learning'})
        else:
            stamp=root/'preflight_pass.json'
            checked=json.loads(stamp.read_text()) if stamp.is_file() else {}
            if checked.get('source_sha')!=sha or checked.get('cuda_worlds')!=2048:
                p.error('missing matching full CUDA preflight stamp')

        base.collect_runtime(root)
        def train_case(x,parent=None):
            guard_source(sha)
            name=x['name']
            directory=root/name
            if directory.exists() and not (directory/'training_config.json').is_file():
                raise RuntimeError('non-resumable run directory: '+str(directory))
            resume=base.has_checkpoint(directory)
            base.execute(command(x,root,parent=parent,resume=resume),
                         root/'logs'/(name+'.log'),env,
                         campaign=(root,report_cases))
            result=directory/'checkpoints/final.pkl'
            if not result.is_file():
                raise RuntimeError('no final checkpoint for '+name)
            outcomes[name]='complete'
            save_json(status,outcomes)
            return result

        parent20=train_case(shared)
        for case in p1:
            train_case(case,parent20)

        winner,decision=select_phase1(root,p1)
        saved=root/'p1_selection.json'
        if saved.is_file() and json.loads(saved.read_text())!=decision:
            raise RuntimeError('P1 selection changed across resume')
        save_json(saved,decision)
        print('P1 SELECTION:',winner['name'],decision['rule'],flush=True)
        p2=phase2(winner)
        report_cases=[shared,*p1,*p2]
        parent40=root/winner['name']/'checkpoints/final.pkl'
        if not parent40.is_file():
            raise RuntimeError('selected P1 final absent')
        for case in p2:
            train_case(case,parent40)
        print('RESEARCH COMPLETE: review physical evidence before P3/400M.',
              flush=True)
        return 0
    except KeyboardInterrupt:
        print('\nINTERRUPTED; saved checkpoints remain. Re-run SAME command '
              'on SAME source to resume.',file=sys.stderr,flush=True)
        raise SystemExit(130)
    finally:
        # Do not hide training failure behind report generation.
        try:
            base.summary(root,report_cases,sha)
        except Exception as exc:
            print('WARNING: report generation failed:',exc,file=sys.stderr)
        print('Campaign root:',root,flush=True)
        print('Report:',root/'overnight_results.zip',flush=True)

if __name__=='__main__':
    raise SystemExit(main())
