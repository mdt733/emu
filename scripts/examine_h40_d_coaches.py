#!/usr/bin/env python3
"""EMU D: one reproducible, read-only 16-policy science autopsy + uploadable ZIP.

Every frozen policy is evaluated with the SAME independent physical protocols,
never its own trainer-success label. No optimizer update, no branch selection.
Output is deliberately one manageable archive of source/config/learning curves,
per-world exact-prism exams, second-lip traces and source-grounded caveats.

Run smoke before full. A partial exam remains resumable, NEVER falsely
packaged as complete or confused with an independent PPO training seed.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

from emu.policy_export import load_exported_actor_for_exam

ARMS=(
    "00-control","01-progressive","02-free-phase","03-foothold",
    "04-commands","05-progressive-free-phase","06-progressive-foothold",
    "07-progressive-commands","08-free-phase-foothold",
    "09-free-phase-commands","10-foothold-commands",
    "11-progressive-free-phase-foothold",
    "12-progressive-free-phase-commands",
    "13-progressive-foothold-commands",
    "14-free-phase-foothold-commands","15-all-four",
)
FULL_PROGRAMS=("up","mixed","down","mixed-command","tof-masked","phase-held")
D_PARENT="11db21960d09b6cc3de924c04ab6e738a3ddf07313c0feacdbef9abc7f604600"


def sha(path:Path):
    h=hashlib.sha256()
    with path.open("rb") as fp:
        for data in iter(lambda:fp.read(1024*1024),b""):
            h.update(data)
    return h.hexdigest()


def json_write(path:Path,content):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(content,indent=2,sort_keys=True,
                               allow_nan=False)+"\n",encoding="utf-8")


def cmd(args, *, log:Path, env=None):
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open("w",encoding="utf-8") as f:
        f.write("COMMAND: "+" ".join(str(x) for x in args)+"\n")
        f.flush()
        process=subprocess.Popen(
            list(map(str,args)),stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,text=True,env=env,bufsize=1)
        for line in process.stdout:
            print(line,end="",flush=True)
            f.write(line)
        rc=process.wait()
        if rc:
            raise RuntimeError(
                f'Exam failed rc={rc}. Partial evidence kept in {log}. '
                'Fix cause and rerun; never silently declare complete.')


def run_metadata(root):
    p=root/"run_meta.json"
    if not p.is_file():
        raise FileNotFoundError(f'Missing source identity: {p}')
    return json.loads(p.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def detect_source_issues(root):
    """Source-truth notes; not invented numerical effectiveness estimates."""
    issues=[
      ("free-phase","The factor modifies phase synchronization *reward*, "
       "which is already multiplied by (1-outcome_walk_alpha). "
       "Once walk alpha=1 on Academy, its intended effect is zero. "
       "Study the saved training schedule; do not claim freer phase harmed gait."),
      ("foothold","The original new_foothold signal is gated by body-root "
       "tile movement. It cannot pay the intended first foot landing "
       "*before* the chassis crosses the edge."),
      ("progressive","It introduces early first-lip absorbing lessons and "
       "suppresses two-lip completion while alpha<0.95. Different duration "
       "and success denominators prevent a naive full-course comparison."),
      ("commands","It widens requested yaw/vy and birth heading while "
       "C's old continuation rule compares heading to transfer heading. "
       "Compare independent motion/turning and stage outcomes."),
      ("report","First-lip sequential foot evidence and second-lip "
       "simultaneous two-foot support use different original definitions. "
       "The old 50cm label can refer to first-lip travel.")
    ]
    return dict(schema='h40_d_coach_treatment_integrity_v1',
                note="Source-derived warnings, not physical exam measurements.",
                items=[dict(topic=k,caveat=v) for k,v in issues])


def validate_runs(root:Path,names:list[str]):
    original=root/'ancestor.sha256'
    parent=root/'foundation-60m'/'checkpoints'/'final.pkl'
    if not original.is_file() or not parent.is_file():
        raise FileNotFoundError('Missing completed D foundation and locked ancestor')
    declared=original.read_text(encoding='utf-8').strip()
    actual=sha(parent)
    if declared!=actual or declared!=D_PARENT:
        raise RuntimeError('Unexpected immutable 60M D parent SHA. Refuse exam.')
    source=None
    metadata={}
    for name in ('foundation-60m',*(f'arm-{x}' for x in names)):
        location=root/name
        cfg=location/'training_config.json'
        ck=location/'checkpoints'/'final.pkl'
        if not (cfg.is_file() and ck.is_file()):
            raise FileNotFoundError(f'Unfinished/missing final training run {name}')
        config=json.loads(cfg.read_text(encoding='utf-8'))
        if name.startswith('arm-'):
            if int(config['steps'])!=160000000:
                raise RuntimeError(f'{name} did not reach requested 160M')
            if config.get('stair_tournament_arm') in ('off',None):
                raise RuntimeError(f'{name} missing D study treatment identity')
        m=run_metadata(location)
        commit=(m.get('source_provenance') or {}).get('git_commit')
        if not commit:
            raise RuntimeError(f'{name} source SHA unrecorded')
        if source and commit!=source:
            raise RuntimeError('Study source SHAs differ: no matched factorial comparison')
        source=commit
        metadata[name]=dict(source_sha=commit,
                            checkpoint_sha256=sha(ck),
                            checkpoint_bytes=ck.stat().st_size,
                            training_config=config)
    return dict(parent_sha256=actual,training_source=source,runs=metadata)


def matrix_case(name,kind,*,smoke):
    if smoke:
        worlds=64; horizon=120
        if kind=='stairs':return dict(worlds=64,horizon=horizon,levels=8,trials_per_level=4,mode=kind)
        if kind=='flat':return dict(worlds=64,horizon=horizon,levels=1,trials_per_level=64,mode=kind)
        return dict(worlds=64,horizon=horizon,levels=1,trials_per_level=4,mode=kind)
    worlds=2048;horizon=420
    if kind=='stairs':return dict(worlds=worlds,horizon=horizon,levels=20,trials_per_level=50,mode=kind)
    if kind=='flat':return dict(worlds=worlds,horizon=horizon,levels=1,trials_per_level=2000,mode=kind)
    return dict(worlds=worlds,horizon=horizon,levels=20,trials_per_level=10,mode=kind)


def finished_exam(out:Path,*,kind:str, expected=None):
    header=out/('exam.json' if kind=='trace' else 'matrix_summary.json')
    evidence=out/('physical_worlds.csv' if kind=='trace' else 'matrix_worlds.csv')
    if not (header.is_file() and evidence.is_file() and evidence.stat().st_size>100):
        return False
    try:
        manifest=json.loads(header.read_text(encoding="utf-8"))
        if kind=='matrix':manifest=manifest['manifest']
        if not isinstance(manifest,dict):return False
        if expected:
            for k,value in expected.items():
                if manifest.get(k)!=value:
                    return False
    except (ValueError,KeyError,OSError):
        return False
    return True


def run_case(outdir, args, *,case, env,expected):
    typ='trace' if case.startswith('trace_') else 'matrix'
    if finished_exam(outdir,kind=typ,expected=expected):
        print(f'VERIFIED EXISTING: {outdir}',flush=True)
        return
    if outdir.exists() and any(outdir.iterdir()):
        raise RuntimeError(f'Incomplete output in {outdir}; keep for forensics, '
                           'manually rename it before rerun')
    logs=outdir.parent.parent.parent/'logs'/(
        outdir.parent.name+'_'+case+'.log')
    cmd(args,log=logs,env=env)
    if not finished_exam(outdir,kind=typ,expected=expected):
        raise RuntimeError(f'Exam returned without matching source/protocol evidence: {outdir}')


def summarize(exam_root:Path, names:list[str]):
    """Summarise real physical exams, never trained episode success rates."""
    table=[]
    for name in names:
        for path in sorted((exam_root/'exams'/name).glob('*')):
            if (path/'exam.json').is_file():
                d=json.loads((path/'exam.json').read_text())
                n=int(d['worlds'])
                table.append(dict(actor=name,exam=path.name,
                                  n=n,first=d['counts']['first_sequential_both'],
                                  second=d['counts']['second_sequential_both'],
                                  first_simul=d['counts']['first_simultaneous_both'],
                                  second_simul=d['counts']['second_simultaneous_both'],
                                  recovered=d['counts']['half_m_after_second'],
                                  verified=d['counts']['course_healthy_commanded'],
                                  zero_training_updates=True))
            elif (path/'matrix_summary.json').is_file():
                d=json.loads((path/'matrix_summary.json').read_text())
                for g in d['curves']:
                    table.append(dict(actor=name,exam=path.name,
                                      kind=g['kind'],level=g['level'],
                                      height_mm=g['requested_step_mm'],
                                      n=g['worlds'],
                                      lead=g['leading_top'],
                                      both=g['each_foot_top_five'],
                                      traversed=g['traversed_5frame'],
                                      completed=g['completed_destinations'],
                                      zero_training_updates=True))
    json_write(exam_root/'science_counts.json',
               dict(schema='h40_d_independent_exam_counts_v1',rows=table))
    # Markdown contains no selected winner or speculative significance test.
    byname={n:[] for n in names}
    for row in table:
        if 'kind' not in row and row['exam']=='trace_mixed_s1701':
            byname[row['actor']].append(row)
    lines=[
      '# EMU D — frozen-policy coaching science autopsy','',
      '**This is independent physical MJX evaluation, not training reward.**',
      'All 16 arms get identical tests; no finalists or auto-promotions.',
      'No policy weights, optimizer state or checkpoint bytes in ZIP.','',
      '## Identical two-lip mixed-sign tests (seed 1701)','',
      '| Actor | Worlds | First foot-by-foot verified | First simultaneous | Second foot-by-foot verified | Second simultaneous | Second +50cm | Commanded recovery |',
      '|---|---:|---:|---:|---:|---:|---:|---:|'
    ]
    for name,items in byname.items():
        if not items:continue
        r=items[0]
        lines.append(f"| {name} | {r['n']} | {r['first']} "
                     f"| {r['first_simul']} | {r['second']} | {r['second_simul']} "
                     f"| {r['recovered']} | {r['verified']} |")
    lines.extend([
      '', '## What to inspect before designing another teacher',
      '',
      '1. Compare each arm to the same exact signed-prism matrix, including',
      '   the paired zero-height seam, command/slope/recovery retention.',
      '2. Separate UP and DOWN and actual signed height, first-toe contact',
      '   before root crossing, each-foot support versus simultaneous stance.',
      '3. Contrast pure-command, mixed-command, ToF-masked and phase-held',
      '   INPUT probes. The latter two are interventions, not hardware tests.',
      '4. Inspect delivered ToF zone availability, motor near-rated torque,',
      '   real post-second 50cm distance and 20-frame velocity convergence.',
      '5. Verify what each factor changed in the source; do not interpret',
      '   a near-zero difference from a mathematically inactive treatment.',
      '',
      '### Guardrails',
      'These are frozen deterministic actor tests from **one training seed**.',
      'World repeats are not independent PPO seeds. The two-lip Academy',
      'samples authentic signed risers at its own initial difficulty; the',
      'exact height-grid matrix tests one seam, not a repeated-stair course.',
      'No static stance, shaft-work proxy or motor current model should be',
      'misreported as proven real-world performance.',
      '',
      'For detailed per-world evidence read `exams/*/physical_worlds.csv`',
      'and `exams/*/matrix_worlds.csv`; read `training/*` for',
      'the original full PPO histories and curriculum evolution.',
      'See `treatment_integrity.json` for known treatment defects.',
    ])
    (exam_root/'SCIENCE_README.md').write_text(
        '\n'.join(lines)+'\n',encoding='utf-8')
    return len(table)


def bundle(output:Path,*,smoke:bool,expected:list[str],metadata):
    required=[]
    for name in expected:
        required.extend(output/'exams'/name/t
                        for t in (('matrix_stairs','matrix_flat','matrix_balanced',
                                   'trace_up_s1701') if smoke
                                  else ('matrix_stairs','matrix_flat','matrix_balanced',
                                        'trace_up_s1701','trace_mixed_s1701',
                                        'trace_down_s1701','trace_mixed-command_s1701',
                                        'trace_tof-masked_s1701','trace_phase-held_s1701',
                                        'trace_up_s2701','trace_mixed_s2701',
                                        'trace_down_s2701')))
    absent=[str(p) for p in required if not finished_exam(
        p,kind='trace' if p.name.startswith('trace_') else 'matrix')]
    if absent:
        raise RuntimeError('Refusing misleading COMPLETE zip; missing:\n'+'\n'.join(absent))
    summary_rows=summarize(output,expected)
    documents=[]
    excluded={'science-bundle.zip','science-smoke.zip',
              'bundle_manifest.json'}
    for p in sorted(output.rglob('*')):
        if not p.is_file() or p.name in excluded or p.name.endswith('.part'):
            continue
        if 'actors' in p.relative_to(output).parts:
            # The user-requested ZIP must contain evidence ONLY.
            # Do not leak all sixteen trained policy weight blobs.
            continue
        if 'logs' in p.relative_to(output).parts:
            # Include logs only if user needs diagnosis, not verbose build spam.
            continue
        documents.append(p)
    index=[dict(path=str(p.relative_to(output)),bytes=p.stat().st_size,
                sha256=sha(p)) for p in documents]
    json_write(output/'bundle_manifest.json',
               dict(schema='h40_d_science_archive_v1',
                    complete=True,smoke=smoke,
                    policies=expected,case_count=len(required),
                    summary_rows=summary_rows,
                    trained_parent_sha=metadata['parent_sha256'],
                    original_training_git_sha=metadata['training_source'],
                    files=index))
    documents.append(output/'bundle_manifest.json')
    target=output/('science-smoke.zip' if smoke else 'science-bundle.zip')
    temporary=output/(target.name+'.part')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6,allowZip64=True) as z:
        for p in documents:z.write(p,p.relative_to(output).as_posix())
    temporary.replace(target)
    # Ensure ZIP is valid even on a file system with unusual rename semantics.
    with zipfile.ZipFile(target) as z:
        bad=z.testzip()
        if bad:raise RuntimeError(f'Corrupt ZIP entry: {bad}')
    print(f'\nCOMPLETE SCIENCE PACKAGE: {target} '
          f'({target.stat().st_size/1024**2:.1f} MiB), '
          f'{len(required)} independently checked test cases.',flush=True)
    print('Upload this ONE zip; no checkpoints or policy weights inside.',flush=True)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path.home()/
                    'emu-data/runs/h40-d-coach-lab-s735')
    ap.add_argument('--output',type=Path,default=Path.home()/
                    'emu-data/exams/h40-d-coach-science-v1')
    ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--only-arm',choices=ARMS,help='Run one arm while debugging, '
                    'never claim the full 16-arm result complete')
    ap.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    args=ap.parse_args(argv)
    root=args.root.expanduser().resolve()
    output=args.output.expanduser().resolve()
    if args.smoke:output=output.with_name(output.name+'-smoke')
    names=[args.only_arm] if args.only_arm else (
        [ARMS[0]] if args.smoke else list(ARMS))
    meta=validate_runs(root,names)
    checkout=Path(__file__).resolve().parent.parent
    exam_commit=subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=checkout,text=True).strip()
    if subprocess.check_output(
        ['git','status','--porcelain'],cwd=checkout,text=True).strip():
        raise RuntimeError('Examiner checkout must be SOURCE PINNED and clean')
    examiner_sources={
        p:sha(checkout/p) for p in (
          'emu/d_science_exam.py','emu/ppo.py',
          'scripts/h40_d_two_lip_exam.py',
          'scripts/examine_h40_d_coaches.py',
          'scripts/capability_matrix_exam.py',
          'emu/terrain_carpet_mjx.py','emu/carpet_prism_mjx.py',
          'emu/physical_variants.py','emu/robot.xml')}
    if (output==root or root in output.parents or output==Path.cwd()
            or output==checkout or checkout in output.parents):
        ap.error('Exam output must be separate from training and source')
    output.mkdir(parents=True,exist_ok=True)
    json_write(output/'provenance.json',dict(
        **meta,examiner_commit=exam_commit,
        examiner_source_hashes=examiner_sources))
    json_write(output/'treatment_integrity.json',detect_source_issues(root))
    env=dict(os.environ,EMU_PHYSICAL_PROFILE='h40',
             PYTHONPATH=str(Path(__file__).resolve().parent.parent))
    # Export weights into a dedicated evidence-only directory. Validate the
    # actual schema, normalizer, robot XML and final checkpoint lineage.
    for name in names:
        print(f'\n--- READ-ONLY AUDIT: {name} ---',flush=True)
        run=root/('arm-'+name)
        policy=output/'actors'/f'{name}.emupolicy'
        policy.parent.mkdir(parents=True,exist_ok=True)
        if not policy.is_file():
            cmd((sys.executable,'-m','emu.cli','export-policy',
                 run/'checkpoints'/'final.pkl',policy),
                log=output/'logs'/f'{name}_export.log',env=env)
        _,_,actor=load_exported_actor_for_exam(policy)
        if actor['source'].get('git_commit')!=meta['training_source']:
            raise RuntimeError('Exported actor source is not the frozen D study')
        for kind in ('stairs','flat','balanced'):
            spec=matrix_case(name,kind,smoke=args.smoke)
            target=output/'exams'/name/('matrix_'+kind)
            run_case(target,(
                sys.executable,'-m','scripts.capability_matrix_exam',
                '--policy',policy,'--output',target,
                '--device',args.device,'--seed','1701',
                '--worlds',str(spec['worlds']),
                '--horizon',str(spec['horizon']),
                '--levels',str(spec['levels']),
                '--trials-per-level',str(spec['trials_per_level']),
                '--mode',spec['mode']
            ),case='matrix_'+kind,env=env,
               expected=dict(policy_sha256=actor['artifact_sha256'],
                             seed=1701,worlds=spec['worlds'],
                             horizon=spec['horizon'],mode=kind))
        programs=('up',) if args.smoke else FULL_PROGRAMS
        seeds=(1701,) if args.smoke else (1701,2701)
        for seed in seeds:
            for program in programs if seed==1701 else ('up','mixed','down'):
                target=output/'exams'/name/f'trace_{program}_s{seed}'
                run_case(target,(
                    sys.executable,'-m','scripts.h40_d_two_lip_exam',
                    '--policy',policy,'--output',target,
                    '--program',program,'--seed',str(seed),
                    '--device',args.device,
                    '--worlds',str(4 if args.smoke else 128),
                    '--horizon',str(450 if args.smoke else 1050)
                ),case=f'trace_{program}_s{seed}',env=env,
                   expected=dict(
                       exported_policy_sha256=actor['artifact_sha256'],
                       seed=seed,worlds=(4 if args.smoke else 128),
                       horizon=(450 if args.smoke else 1050),
                       program=program))
        # Preserve complete original training evidence, not just the
        # distorted stage summary. Never include checkpoint or actor bytes.
        dest=output/'training'/name
        dest.mkdir(parents=True,exist_ok=True)
        for fname in ('training_config.json','run_meta.json','metrics.csv',
                      'fixed_eval.csv','results.md','results_data.json',
                      'envelope_summary.json'):
            src=run/fname
            if src.is_file():shutil.copyfile(src,dest/fname)
        # Saved exports remain local for resume, but are deliberately
        # excluded from the upload archive.
    # Preserve the 60M ancestor's learning curve too, without exporting
    # training weights or treating it as one of the 16 D treatments.
    source_root=root/'foundation-60m'
    foundation_out=output/'training'/'foundation-60m'
    foundation_out.mkdir(parents=True,exist_ok=True)
    for fname in ('training_config.json','run_meta.json','metrics.csv',
                  'fixed_eval.csv','results.md','results_data.json',
                  'envelope_summary.json'):
        src=source_root/fname
        if src.is_file():shutil.copyfile(src,foundation_out/fname)
    if not args.only_arm:
        bundle(output,smoke=args.smoke,expected=names,metadata=meta)
    else:
        print('ONE-ARM diagnostic complete. Re-run without --only-arm '
              'for a complete 16-arm scientific ZIP.',flush=True)


if __name__=='__main__':
    main()
