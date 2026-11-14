#!/usr/bin/env python3
"""Explain the 16-arm D coaching laboratory, NEVER rank or eliminate policies.

Every line is from the SAME training-source SHA. metrics.csv contains
periodically LOGGED PPO rollouts, not every environment episode, so rates
are explicitly labelled 'logged rollouts'. Use independent frozen prism
exams, with identical simulator/seed/protocol, before trusting any claim.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

FACTORS=('progressive','free-phase','foothold','commands')
COUNTS={
    'academy_first_lip_supported_episodes':'First real two-foot support',
    'academy_two_lip_crossed_episodes':'Second physical lip crossed',
    'academy_second_bilateral_episodes':'Second real two-foot support',
    'academy_posttransfer_50cm_episodes':'50cm after latest transfer',
    'academy_vx_window_episodes':'Valid 0.4s commanded vx at ending',
    'academy_two_lip_continuations':'Complete certified D course',
    'academy_intro_lesson_successes':'Introductory one-lip lessons',
}


def number(row,field,default=0.0):
    try:
        value=float(row.get(field,default))
    except (TypeError,ValueError):
        return default
    return value if math.isfinite(value) else default


def summarize_arm(run:Path, *, start:int=100_000_000)->dict|None:
    cfg=run/'training_config.json'
    csv_path=run/'metrics.csv'
    if not cfg.is_file() or not csv_path.is_file():
        return None
    config=json.loads(cfg.read_text(encoding='utf-8'))
    arm=config.get('stair_tournament_arm','off')
    if arm in ('off',None):
        return None
    factors=frozenset(arm.split('+')) if arm!='control' else frozenset()
    if not factors<=set(FACTORS):
        raise ValueError(f'Unknown factor combination in {run}: {arm}')
    with csv_path.open(newline='',encoding='utf-8') as stream:
        rows=[r for r in csv.DictReader(stream)
              if number(r,'step')>=start]
    if not rows:
        return dict(name=run.name,arm=arm,factors=factors,rows=0,episodes=0.,
                    fields={key:0. for key in COUNTS},rates={key:float('nan') for key in COUNTS},
                    summary={},finish=0,source=None)
    sample_count=sum(number(x,'academy_episodes') for x in rows)
    fields={key:sum(number(x,key) for x in rows) for key in COUNTS}
    rates={key:(100.*value/sample_count if sample_count>0 else float('nan'))
           for key,value in fields.items()}
    summary={}
    for key in ('fall_rate','servo_over_rated_frac','eval_core_vx_mae',
                'eval_resilience75_survival','shaft_j_per_m_forward',
                'frontier_precision_combo','physical_slope_world_fraction'):
        values=[number(row,key,default=float('nan')) for row in rows]
        values=[v for v in values if math.isfinite(v) and v>=0.]
        summary[key]=statistics.mean(values) if values else float('nan')
    meta=run/'run_meta.json'
    source=None
    if meta.is_file():
        source=json.loads(meta.read_text(encoding='utf-8')).get(
            'source_provenance',{}).get('git_commit')
    return dict(name=run.name,arm=arm,factors=factors,rows=len(rows),
                episodes=sample_count,fields=fields,rates=rates,
                summary=summary,finish=int(number(rows[-1],'step')),source=source)


def fmt(value,digits=2):
    return f'{value:.{digits}f}' if math.isfinite(value) else '—'


def report(root:Path,start:int=100_000_000)->str:
    raw=[summarize_arm(d,start=start) for d in sorted(root.glob('arm-*'))
         if d.is_dir()]
    arms=[r for r in raw if r is not None]
    lines=[
        '# EMU D — what did each stair TEACHER teach us?',
        '',
        'This is a **diagnostic field notebook**, not a policy leaderboard.',
        'No arm is disqualified, selected as a champion or granted more PPO',
        'steps because of its current score. An apparently poor performer',
        'can identify a useful partial skill or a broken learning signal.',
        '',
        f'Sampled PPO logging window: absolute step >= {start:,}.',
        'Numerators and denominators below are from **logged rollouts only**.',
        'They are not exhaustive trial counts, independent held-out exams,',
        'credible intervals or significance tests. Progression can end',
        'an assisted first-lip episode early, changing exposure denominators.',
        '',
        f'**Available research arms:** {len(arms)}/16.',
    ]
    if not arms:return '\n'.join(lines+['','No logged branch data yet.',''])
    sha={a['source'] for a in arms if a['source']}
    if len(sha)>1:
        raise ValueError('Arm source SHAs differ; refuse causal-effect summary')
    lines.extend(['',
        '| Branch / active teaching factors | PPO step | Academy episodes in logged rollouts | First 2-foot % | Second 2-foot % | 50cm % | Full course % |',
        '|---|---:|---:|---:|---:|---:|---:|'])
    for a in arms:
        r=a['rates']
        lines.append(
            f"| {a['name']} ({a['arm']}) | {a['finish']/1e6:.1f}M "
            f"| {int(a['episodes'])} "
            f"| {fmt(r['academy_first_lip_supported_episodes'])} "
            f"| {fmt(r['academy_second_bilateral_episodes'])} "
            f"| {fmt(r['academy_posttransfer_50cm_episodes'])} "
            f"| {fmt(r['academy_two_lip_continuations'])} |"
        )
    by_set={a['factors']:a for a in arms}
    lines.extend(['',
        '## Paired hypotheses — which coach factor helps in context?',
        '',
        'Pair each treatment arm with its EXACT matching control for the',
        'other three factors. Only equal-training-budget pairs are included;',
        'partial/incomplete arms are visible but not interpreted as negative',
        'evidence. These are descriptive percentage-point differences in',
        'logged complete-course events per completed Academy episode.',
        '',
        '| Teaching factor | Matched pairs | Mean complete-course Δ (pp) | Range Δ (pp) | Coaching question |',
        '|---|---:|---:|---:|---|'])
    questions={
        'progressive':'Does one-lip introduction improve later two-lip competence?',
        'free-phase':'Does removing stair-local phase shaping improve real supported transfer?',
        'foothold':'Does contact-grounded sparse shaping improve discovery?',
        'commands':'Does broader real operator command practice transfer to stairs?',
    }
    for factor in FACTORS:
        values=[]
        for fs,a in by_set.items():
            if factor in fs:continue
            matched=by_set.get(fs|{factor})
            if matched is None or min(a['finish'],matched['finish'])<160_000_000:
                continue
            x=matched['rates']['academy_two_lip_continuations']
            y=a['rates']['academy_two_lip_continuations']
            if math.isfinite(x) and math.isfinite(y):
                values.append(x-y)
        avg=statistics.mean(values) if values else float('nan')
        bounds=(f"{fmt(min(values))} to {fmt(max(values))}" if values else '—')
        lines.append(
            f"| {factor} | {len(values)}/8 | {fmt(avg)} | {bounds} | {questions[factor]} |"
        )
    lines.extend(['',
        '## The important next investigation',
        '',
        'Read the full trajectory/protocol and failure stages for every',
        'branch. Check the 0.4s velocity window, actual physical contact,',
        'sensor visibility, trajectory heading and first→second transition.',
        'Re-run matched independent MJX exams and verify flat/slope/turning',
        'retention and rated motor output. Do not infer causality from',
        'logged conditional rates alone; investigate differences in exposure.',
        '',
        'Finally combine compatible, demonstrated teaching mechanisms',
        'into one small curriculum and validate a fresh end-to-end walker.',
        'There is deliberately NO finalist or winner in this document.',
        '',
    ])
    return '\n'.join(lines)


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path.home()/
                        'emu-data/runs/h40-d-coach-lab-s735')
    parser.add_argument('--start-step',type=int,default=100_000_000)
    args=parser.parse_args()
    if args.start_step<60_000_000:parser.error('Study branches begin from 60M')
    root=args.root.expanduser().resolve()
    doc=report(root,start=args.start_step)
    root.mkdir(parents=True,exist_ok=True)
    out=root/'coach_research_summary.md'
    out.write_text(doc,encoding='utf-8')
    print(doc)
    print(f'\nWritten diagnostic notebook: {out}')


if __name__=='__main__':
    main()
