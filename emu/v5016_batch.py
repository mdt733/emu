"""Read-only, resumable v5016 GPU comparisons; plan is stdlib-only."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
import zipfile

from .directional_exam import sha256_file, source_commit

ARMS=('baseline','shadow','slope','pitch15')


def cases(profile):
    if profile=='smoke':
        return [('baseline','flat',0.,'uphill','steady',40),
                ('baseline','turn_stop',0.,'uphill','turn_stop_left',160)]
    if profile=='pilot':
        tasks=[('turn_left',0.,'uphill','turn_stop_left'),
               ('turn_right',0.,'uphill','turn_stop_right'),
               ('up_025',.25,'uphill','steady'),
               ('up_050',.5,'uphill','steady'),
               ('cross_050',.5,'cross_left','steady')]
        return [(arm,name,angle,direction,scenario,350)
                for arm in ARMS for name,angle,direction,scenario in tasks]
    if profile=='sensor':
        return [(arm,name,angle,direction,scenario,350)
                for arm in ('baseline','slope')
                for name,angle,direction,scenario in (
                    ('noisy_turn',0.,'uphill','turn_stop_left'),
                    ('noisy_up_025',.25,'uphill','steady'))]
    if profile=='stretch':
        return [(arm,'up_100',1.,'uphill','steady',350) for arm in ARMS]
    raise ValueError('unknown profile')


def plan(profile,root,output,*,worlds=None,seed=1701,kp=.75,device='cuda'):
    source=Path(root).expanduser().resolve()
    dest=Path(output).expanduser().resolve()
    if source==dest or source in dest.parents or dest in source.parents:
        raise ValueError('outputs and training directories must be separate')
    n=int(worlds or (2 if profile=='smoke' else 8))
    if not 2<=n<=64 or not math.isfinite(kp) or not 0<kp<=2:
        raise ValueError('bounded worlds/gain required')
    bias,noise,delay=(.015,.02,2) if profile=='sensor' else (0.,0.,0)
    return [dict(arm=arm,label=label,angle=angle,direction=direction,
          scenario=scenario,horizon=horizon,worlds=n,seed=int(seed),
          heading_kp=kp,device=device,gyro_bias=bias,gyro_noise=noise,
          gyro_delay_frames=delay,run_dir=str(source/'round1'/arm),
          output=str(dest/profile/arm/label))
          for arm,label,angle,direction,scenario,horizon in cases(profile)]


def verify_existing(path,job,revision,checkpoint_hash):
    if not path.exists():
        return False
    info=json.loads(path.read_text())
    if info.get('checkpoint_sha256')!=checkpoint_hash or (
        info.get('examined_at_revision')!=revision):
        raise ValueError('stale source or trained checkpoint in '+str(path))
    exam=info.get('protocol',{})
    wanted=dict(angle_deg=job['angle'],direction=job['direction'],
        scenario=job['scenario'],worlds=job['worlds'],horizon=job['horizon'],
        seed=job['seed'],gyro_bias_rad_s=job['gyro_bias'],
        gyro_noise_std_rad_s=job['gyro_noise'],
        gyro_delay_frames=job['gyro_delay_frames'])
    if any(exam.get(k)!=v for k,v in wanted.items()) or (
        info.get('heading_kp')!=job['heading_kp']):
        raise ValueError('stale/changed exam parameters: '+str(path))
    if not all((path.parent/mode/'directional_exam.json').is_file()
               for mode in ('off','on')):
        raise ValueError('incomplete paired evidence: '+str(path))
    return True


def summarize(jobs,profile_root,revision):
    rows=[]
    for job in jobs:
        p=Path(job['output'])/'heading_comparison.json'
        if p.is_file():
            data=json.loads(p.read_text())
            rows.append(dict(arm=job['arm'],case=job['label'],angle_deg=job['angle'],
                 checkpoint_sha256=data['checkpoint_sha256'],
                 protocol=data['protocol'],off=data['off'],on=data['on'],
                 on_minus_off=data['on_minus_off']))
    summary=dict(schema=1,profile=profile_root.name,source_commit=revision,
                 expected=len(jobs),completed=len(rows),results=rows)
    (profile_root/'results_data.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# v5016 '+profile_root.name+' — matched heading and terrain tests','',
        'Source commit: '+revision,
        str(len(rows))+'/'+str(len(jobs))+' cases complete.',
        'Same saved checkpoint and operator request for each ON/OFF pair.',
        'No PPO training or promotion performed. Gait contact flags are estimates.',
        '',
        '| Policy | Case | OFF survival | ON survival | OFF course | ON course | OFF heading MAE | ON heading MAE |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    def v(x,key,percent=False):
        value=x.get(key)
        if value is None:
            return '—'
        return format(100*value if percent else value,'.1f')+('%' if percent else '')
    for row in rows:
        a,b=row['off'],row['on']
        lines.append('| '+' | '.join((row['arm'],row['case'],
            v(a,'survived_full',True),v(b,'survived_full',True),
            v(a,'course_pass',True),v(b,'course_pass',True),
            v(a,'heading_mae_deg'),v(b,'heading_mae_deg')))+' |')
    lines+=['','Higher survival and actual requested-course progress must accompany',
            'any claimed heading benefit. Inspect the raw paired episodes.']
    (profile_root/'results.md').write_text('\n'.join(lines)+'\n')
    with (profile_root/'results.csv').open('w',newline='') as file:
        fields=('arm','case','mode','survived_full','course_pass',
                'gait_candidate_pass','progress_m','heading_mae_deg',
                'yaw_abs_error_1s','gyro_heading_error_mae_deg')
        w=csv.DictWriter(file,fieldnames=fields)
        w.writeheader()
        for r in rows:
            for mode in ('off','on'):
                item=r[mode]
                w.writerow(dict(arm=r['arm'],case=r['case'],mode=mode,
                    **{k:item.get(k) for k in fields[3:]}))
    temp=profile_root/'results_bundle.zip.tmp'
    bundle=profile_root/'results_bundle.zip'
    with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in ('results.md','results.csv','results_data.json'):
            z.write(profile_root/p,arcname=p)
        for job in jobs:
            base=Path(job['output'])
            for p in ('heading_comparison.md','heading_comparison.json',
                      'off/directional_exam.md','off/directional_exam.json',
                      'on/directional_exam.md','on/directional_exam.json'):
                f=base/p
                if f.is_file() and not f.is_symlink():
                    z.write(f,arcname=job['arm']+'/'+job['label']+'/'+p)
    temp.replace(bundle)
    return bundle,len(rows)


def execute(jobs,revision,*,actually_run=False):
    if not jobs:
        raise ValueError('no experiment cases')
    profile_root=Path(jobs[0]['output']).parents[1]
    for i,j in enumerate(jobs,1):
        print(f"{i:02d}/{len(jobs)} {j['arm']} {j['label']} "
              f"{j['angle']}deg {j['worlds']} worlds x {j['horizon']} steps "
              f"bias={j['gyro_bias']} noise={j['gyro_noise']} delay={j['gyro_delay_frames']}",
              flush=True)
    if not actually_run:
        print('PLAN ONLY. Add --run to execute (saved training is read-only).')
        return
    hashes={}
    for j in jobs:
        checkpoint=Path(j['run_dir'])/'checkpoints/final.pkl'
        if not checkpoint.is_file():
            raise FileNotFoundError(str(checkpoint))
        if j['arm'] not in hashes:
            hashes[j['arm']]=sha256_file(checkpoint)
    profile_root.mkdir(parents=True,exist_ok=True)
    for job in jobs:
        dest=Path(job['output'])
        report=dest/'heading_comparison.json'
        if verify_existing(report,job,revision,hashes[job['arm']]):
            print('Already validated: '+str(dest),flush=True)
            continue
        if dest.exists() and any(dest.iterdir()):
            raise FileExistsError('partial exam: inspect before retry '+str(dest))
        cmd=[sys.executable,'-m','emu.directional_exam','--run-dir',job['run_dir'],
             '--output',str(dest),'--angle',str(job['angle']),
             '--direction',job['direction'],'--scenario',job['scenario'],
             '--worlds',str(job['worlds']),'--horizon',str(job['horizon']),
             '--seed',str(job['seed']),'--device',job['device'],
             '--heading-kp',str(job['heading_kp']),'--compare-heading',
             '--gyro-bias',str(job['gyro_bias']),
             '--gyro-noise',str(job['gyro_noise']),
             '--gyro-delay-frames',str(job['gyro_delay_frames'])]
        subprocess.run(cmd,check=True)
        if not verify_existing(report,job,revision,hashes[job['arm']]):
            raise RuntimeError('no completed exam at '+str(report))
        summarize(jobs,profile_root,revision)
    zip_path,n=summarize(jobs,profile_root,revision)
    print('Completed '+str(n)+'/'+str(len(jobs))+'; upload '+str(zip_path))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--profile',choices=('smoke','pilot','sensor','stretch'),default='smoke')
    p.add_argument('--worlds',type=int)
    p.add_argument('--seed',type=int,default=1701)
    p.add_argument('--heading-kp',type=float,default=.75)
    p.add_argument('--device',choices=('auto','cpu','cuda','gpu'),default='cuda')
    p.add_argument('--run',action='store_true')
    a=p.parse_args(argv)
    jobs=plan(a.profile,a.root,a.output,worlds=a.worlds,seed=a.seed,
              kp=a.heading_kp,device=a.device)
    execute(jobs,source_commit(),actually_run=a.run)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
