#!/usr/bin/env python3
"""v5027: isolate coherent step-command attenuation and moving retention.

Source-pinned, from random 360 actor weights; same 20M nursery for ALL arms.
A original coherent-gentle, B coherent-direct-gentle, C B + 25% retention.
Three 20->60M forks (140M aggregate PPO world steps). Each arm has fixed
10M exams; after the training trials, run exactly matched CPU terrain
matrices on the final exported actor. No Pinnacle writes or main changes.

Plan-only by default; --run performs CPU/CUDA preflight and actual training.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import subprocess
import zipfile
from pathlib import Path
import v5024_campaign as base

HERE=Path(__file__).resolve().parents[1]
BRANCH='feature/v5027-coherent-retention'
SEED=735
FOUNDATION=20_000_000
FINAL=60_000_000
# Previous default was stamped by the aborted wrong-import preflight,
# with no learner checkpoint. Preserve that audit instead of rewriting it.
DEFAULT_ROOT=Path.home()/'robot_lab/emu_runs/v5027_coherent_retention_s735_r2'

def cases():
    first=dict(name='shared_foundation20',recipe='main',schedule='balanced',
               steps=FOUNDATION,from_steps=0,
               growth='legacy',terrain='mixed',retain=0.,force='gentle')
    common=dict(recipe='main',schedule='balanced',steps=FINAL,
                from_steps=FOUNDATION,growth='main-mature',force='gentle')
    arms=[
        dict(name='A_coherent_original',terrain='coherent',retain=0.,**common),
        dict(name='B_coherent_direct',terrain='coherent-direct',retain=0.,**common),
        dict(name='C_direct_retention',terrain='coherent-direct',retain=.25,**common),
    ]
    return first,arms

def args_for(c,root,*,parent=None,resume=False):
    cmd=base.cmd_for(c,root,SEED,FOUNDATION,resume)
    cmd[cmd.index('--terrain-step-max-m')+1]='0.08'
    cmd.extend([
        '--growth-recipe',c['growth'], '--terrain-lesson',c['terrain'],
        '--retention-share',str(c['retain']),'--force-program',c['force']])
    if parent is not None and not resume:
        cmd.extend(['--resume',str(parent)])
    return cmd

def source_guard(sha):
    if base.git('branch','--show-current')!=BRANCH or base.git('rev-parse','HEAD')!=sha:
        raise RuntimeError('branch/head changed inside source-pinned campaign')
    if base.git('status','--porcelain'):
        raise RuntimeError('the training source checkout became dirty')

def save(path,doc):
    path.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n',encoding='utf-8')

def physical_exams(root,arms,env,source_sha):
    # Read-only matched physical crossing/slope results are NOT PPO feedback.
    for arm in arms:
        source_guard(source_sha)
        policy=root/'physical_policies'/(arm['name']+'.emupolicy')
        checkpoint=root/arm['name']/'checkpoints/final.pkl'
        directory=root/'physical_exams'/arm['name']
        if not policy.is_file():
            base.execute([sys.executable,'-m','emu.cli','export-policy',
                          str(checkpoint),str(policy)],
                         root/'logs'/('export_'+arm['name']+'.log'),env)
        if (directory/'unified_exam.json').is_file() and (directory/'unified_exam.md').is_file():
            continue
        if directory.exists() and any(directory.iterdir()):
            raise RuntimeError('Incomplete physical exam output: '+str(directory))
        base.execute([sys.executable,'-m','emu.cli','capability-exam',
                      '--terrain-matrix','--policy',str(policy),
                      '--output',str(directory),'--worlds','8',
                      '--horizon','350','--speed','0.15','--device','cpu'],
                     root/'logs'/('physical_'+arm['name']+'.log'),env)
    report=['# v5027 same-seed, same-physics final 60M frozen exams','',
            'Two separate outcomes: physical mission/step crossing and strict',
            'all-axis heading precision. No policy was trained by these exams.','',
            '| Case | '+' | '.join(arm['name'] for arm in arms)+' |',
            '|---|'+'|'.join('---:' for _ in arms)+'|']
    docs=[json.loads((root/'physical_exams'/arm['name']/'unified_exam.json').read_text())
          for arm in arms]
    refs=[d['results'] for d in docs]
    if any([x['case']['name'] for x in records]!=
           [x['case']['name'] for x in refs[0]] for records in refs[1:]):
        raise RuntimeError('Physical exam condition list differs')
    for rows in zip(*refs):
        entry=rows[0]['case']
        name=entry['name']
        cells=[]
        for x in rows:
            n=len(x['worlds'])
            if entry['kind']=='step':
                value=f"{x['crossed']}/{n} cross; {x['mission_passed']}/{n} mission"
            elif entry['kind']=='slope':
                value=f"{x['stable']}/{n} stable; {x['mission_passed']}/{n} mission"
            else:
                value=f"{x['mission_passed']}/{n} mission; {x['passed']}/{n} strict"
            cells.append(value)
        report.append('| '+name+' | '+' | '.join(cells)+' |')
    report+=['','Each policy also has full per-world physical metrics in physical_exams/.',
             'Repeat with other seeds and more worlds before declaring mastery.']
    (root/'physical_summary.md').write_text('\n'.join(report)+'\n',encoding='utf-8')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',action='store_true',help='preflight, train and examine')
    p.add_argument('--skip-smoke',action='store_true',help='requires same-SHA preflight stamp')
    p.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    a=p.parse_args()
    if base.git('branch','--show-current')!=BRANCH or base.git('status','--porcelain'):
        p.error('run only from a clean '+BRANCH+' branch')
    sha=base.git('rev-parse','HEAD')
    root=a.root.expanduser().resolve()
    if root==HERE or HERE in root.parents:
        p.error('results must remain outside the Git checkout')
    foundation,arms=cases()
    all_cases=[foundation,*arms]
    manifest={
        'experiment':'v5027_coherent_forward_and_retention_causal_study',
        'source_sha':sha,'branch':BRANCH,'seed':SEED,
        'foundation':foundation,'arms':arms,
        'aggregate_target_steps':FOUNDATION+len(arms)*(FINAL-FOUNDATION),
        'note':'fresh source-pinned cold start; no v5026 checkpoint transplanted',
    }
    print('v5027: '+sha+'; '+str(root),flush=True)
    print('20M shared cold start, then THREE same-age 20->60M forks:',flush=True)
    for x in arms:
        print(' ',x['name'],x['terrain'],'retention='+str(x['retain']),flush=True)
    print('140M aggregate steps, with matched frozen 60M physical exams.',flush=True)
    if not a.run:
        print('PLAN ONLY: append --run to perform the campaign.',flush=True)
        return 0
    root.mkdir(parents=True,exist_ok=True)
    fp=root/'campaign_manifest.json'
    if fp.is_file():
        if json.loads(fp.read_text(encoding='utf-8'))!=manifest:
            p.error('different source or experiment; choose new --root')
    elif any(root.iterdir()):
        p.error('root not empty and no v5027 manifest')
    else:
        save(fp,manifest)
    # The operator reuses an editable venv installed in ~/emu-graduation.
    # Running "python scripts/..." otherwise puts scripts/ ahead of an
    # old editable site-packages mapping, silently importing the OLD
    # emu/environment.py despite using this new Git checkout.
    # Never rewrite/install another checkout into the operator's venv:
    # pin this source root ahead of editable mappings for ALL children.
    existing_pythonpath=os.environ.get('PYTHONPATH','')
    env={**os.environ,'XLA_PYTHON_CLIENT_PREALLOCATE':'false',
         'EMU_UNIFIED_CONTINUOUS':'cold-growth-approved',
         'PYTHONPATH':str(HERE)+(os.pathsep+existing_pythonpath
                                if existing_pythonpath else '')}
    # Check in a distinct child interpreter under the EXACT environment
    # used for CPU smoke, CUDA smoke, training, export and frozen exams.
    # This import is cheap: do not compile MJX before confirming source.
    check_source=(
        "import importlib.util, pathlib, sys; "
        "spec=importlib.util.find_spec('emu.environment'); "
        "actual=pathlib.Path(spec.origin).resolve() if spec else None; "
        "expected=pathlib.Path(sys.argv[1]).resolve(); "
        "print('EMU environment source:', actual, flush=True); "
        "sys.exit(0 if actual==expected else 29)"
    )
    result=subprocess.run(
        [sys.executable,'-c',check_source,str(HERE/'emu/environment.py')],
        cwd=HERE,env=env,check=False)
    if result.returncode:
        raise RuntimeError(
            'Refusing v5027: Python is importing EMU from a different '
            'checkout; no GPU work started. Expected '+str(HERE/'emu/environment.py'))
    try:
        if not a.skip_smoke:
            for label,terrain,retention in [
                ('original','coherent','0'),
                ('direct','coherent-direct','0'),
                ('direct_retained','coherent-direct','.25')]:
                base.execute([sys.executable,'scripts/smoke_unified_focus.py',
                    '--device','cpu','--envs','12','--terrain-max-m','.08',
                    '--growth-recipe','main-mature','--terrain-lesson',terrain,
                    '--retention-share',retention,'--force-program','gentle'],
                    root/'logs'/('preflight_'+label+'_cpu.log'),env)
            base.execute([sys.executable,'scripts/smoke_unified_focus.py',
                '--device','cuda','--envs','2048','--full-update',
                '--terrain-max-m','.08','--growth-recipe','main-mature',
                '--terrain-lesson','coherent-direct',
                '--retention-share','.25','--force-program','gentle'],
                root/'logs'/'preflight_2048_cuda.log',env)
            save(root/'preflight_pass.json',{'sha':sha,'cuda_worlds':2048,
                                              'cpu_variants':3})
        else:
            stamp=root/'preflight_pass.json'
            prev=json.loads(stamp.read_text()) if stamp.is_file() else {}
            if prev.get('sha')!=sha or prev.get('cuda_worlds')!=2048:
                p.error('no exact-source completed CUDA preflight; omit --skip-smoke')
        base.collect_runtime(root)
        def train_case(c,parent=None):
            source_guard(sha)
            d=root/c['name']
            if d.exists() and not (d/'training_config.json').is_file():
                raise RuntimeError('incomplete/foreign run directory: '+str(d))
            resume=base.has_checkpoint(d)
            base.execute(args_for(c,root,parent=parent,resume=resume),
                         root/'logs'/(c['name']+'.log'),env,
                         campaign=(root,all_cases))
            final=d/'checkpoints/final.pkl'
            if not final.is_file():
                raise RuntimeError('missing completed final checkpoint: '+str(final))
            return final
        ancestor=train_case(foundation)
        for arm in arms:
            train_case(arm,ancestor)
        (root/'physical_policies').mkdir(exist_ok=True)
        physical_exams(root,arms,env,sha)
        print('Complete. Study actual fixed physical missions and retained',
              'core/recovery before selecting any next policy.',flush=True)
    except KeyboardInterrupt:
        print('\nInterrupted; run same unchanged source to resume.',file=sys.stderr)
        raise SystemExit(130)
    finally:
        try:
            base.summary(root,all_cases,sha)
            phys=root/'physical_summary.md'
            if phys.is_file():
                with zipfile.ZipFile(root/'overnight_results.zip','a',
                                     compression=zipfile.ZIP_DEFLATED) as z:
                    z.write(phys,'physical_summary.md')
                    for arm in arms:
                        for name in ('unified_exam.json','unified_exam.md'):
                            path=root/'physical_exams'/arm['name']/name
                            if path.is_file():
                                z.write(path,'physical_exams/'+arm['name']+'/'+name)
        except Exception as exc:
            print('WARNING: report packing failed: '+str(exc),file=sys.stderr)
        print('Campaign root: '+str(root),flush=True)
        print('Results bundle: '+str(root/'overnight_results.zip'),flush=True)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
