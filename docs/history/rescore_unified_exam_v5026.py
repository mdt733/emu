#!/usr/bin/env python3
"""HISTORICAL ONLY: read-only rescore of archived EMU v5026 terrain exams.

Retain the original strict all-axis pass; add provisional physical-mission
interpretation from saved per-world evidence. No PPO or simulation calls.\n\nThis script is archived with v5026 research; not a current release or\ntraining utility. Reproduce its exact 2026 research from the historical\nsource SHA before attributing results.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

VX_TOL = .07
VY_TOL = .06
YAW_BIAS_TOL = .08
YAW_JITTER_TOL = .20


def moving(w):
    fields = ('tracking_mae_vx_mps', 'tracking_mae_vy_mps',
              'tracking_signed_bias_yaw_rad_s', 'tracking_jitter_yaw_rad_s',
              'path_travel_m', 'requested_path_m',
              'supported_fraction', 'bad_contact_fraction')
    if any(not isinstance(w.get(k), (float, int)) for k in fields):
        return False
    return bool(w.get('survived') and not w.get('fell') and
                w['tracking_mae_vx_mps'] <= VX_TOL and
                w['tracking_mae_vy_mps'] <= VY_TOL and
                abs(w['tracking_signed_bias_yaw_rad_s']) <= YAW_BIAS_TOL and
                w['tracking_jitter_yaw_rad_s'] <= YAW_JITTER_TOL and
                w['supported_fraction'] >= .35 and
                w['bad_contact_fraction'] < .20 and
                (w['requested_path_m'] < .01 or
                 w['path_travel_m'] >= .30 * w['requested_path_m']))


def score_case(entry):
    case = entry['case']
    kind = case['kind']
    worlds = entry['worlds']
    def mission(w):
        if kind == 'slope':
            return bool(w.get('grade_verified') and
                        w.get('slope_stability_pass') and moving(w))
        if kind in ('step', 'mixed'):
            return bool(w.get('crossing_pass') and moving(w))
        if kind in ('flat', 'control'):
            return moving(w)
        return None  # yaw commands need full trace, not a mean-only regrade
    grades = [mission(w) for w in worlds]
    n = len(worlds)
    return {
        'name': case['name'], 'kind': kind, 'n': n,
        'strict': entry['passed'], 'survived': entry['survived'],
        'mission': None if None in grades else sum(grades),
        'crossed': (sum(w.get('crossing_pass') is True for w in worlds)
                    if kind in ('step','mixed') else None),
        'stable': (sum(w.get('slope_stability_pass') is True for w in worlds)
                   if kind == 'slope' else None),
        'vx': mean(w['tracking_mae_vx_mps'] for w in worlds),
        'vy': mean(w['tracking_mae_vy_mps'] for w in worlds),
        'yaw': mean(w['tracking_mae_yaw_rad_s'] for w in worlds),
        'bias': mean(w['tracking_signed_bias_yaw_rad_s'] for w in worlds),
        'jitter': mean(w['tracking_jitter_yaw_rad_s'] for w in worlds),
    }


def summarize(paths, labels=None):
    studies = []
    for p in paths:
        doc = json.loads(p.read_text(encoding='utf-8'))
        if doc.get('schema') != 'emu_unified_frozen_exam_v1':
            raise ValueError('Only archived v1 exams are supported: ' + str(p))
        if not doc.get('protocol', {}).get('terrain_matrix'):
            raise ValueError('Not a terrain-matrix exam: ' + str(p))
        studies.append({
            'label': labels[len(studies)] if labels else p.parent.name,
            'source': doc['source_commit'],
            'actor': doc['actor_artifact_sha256'],
            'protocol': doc['protocol'],
            'scores': [score_case(r) for r in doc['results']],
        })
    names = [row['name'] for row in studies[0]['scores']]
    for s in studies[1:]:
        if [r['name'] for r in s['scores']] != names or s['protocol'] != studies[0]['protocol']:
            raise ValueError('Mismatched cases/seeds/physical exam protocol')
    lines = [
        '# EMU archived terrain reanalysis', '',
        'Diagnostic only: original strict all-axis pass counts are unchanged.',
        'Mission allows bounded yaw-rate bias (0.08rad/s) and jitter (0.20rad/s)',
        'rather than requiring yaw-rate MAE <=0.08rad/s on straight gait.',
        'It also requires vx/vy MAE <=0.07/0.06m/s, movement, foot support,',
        'survival and task-specific verified crossing or stable grade.',
        'It does not certify long-range heading control.', '',
        '| Physical case | ' + ' | '.join(s['label'] for s in studies) + ' |',
        '|---|' + '|'.join('---:' for _ in studies) + '|',
    ]
    for name in names:
        rows = [next(r for r in s['scores'] if r['name'] == name) for s in studies]
        kind = rows[0]['kind']
        if kind in ('step','mixed'):
            suffix = ' (supported crossing / mission / strict)'
            cells = [f"{r['crossed']}/{r['n']} / {r['mission']}/{r['n']} / {r['strict']}/{r['n']}" for r in rows]
        elif kind == 'slope':
            suffix = ' (stable / mission / strict)'
            cells = [f"{r['stable']}/{r['n']} / {r['mission']}/{r['n']} / {r['strict']}/{r['n']}" for r in rows]
        else:
            suffix = ' (mission / strict)'
            cells = [f"{r['mission']}/{r['n']} / {r['strict']}/{r['n']}"
                     if r['mission'] is not None else f"— / {r['strict']}/{r['n']}"
                     for r in rows]
        lines.append('| ' + name + suffix + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', '## Flat command metrics', '',
              '| Policy | vx MAE | vy MAE | yaw MAE | yaw signed bias | yaw jitter |',
              '|---|---:|---:|---:|---:|---:|']
    for s in studies:
        r = next(x for x in s['scores'] if x['name'] == 'flat')
        lines.append(f"| {s['label']} | {r['vx']:.3f} m/s | {r['vy']:.3f} m/s | "
                     f"{r['yaw']:.3f} rad/s | {r['bias']:+.3f} rad/s | {r['jitter']:.3f} rad/s |")
    lines += ['', '## Provenance', '']
    for s in studies:
        lines.append(f"- {s['label']}: actor SHA256 {s['actor']}; exam source {s['source']}")
    lines += ['', 'Absolute path travel is NOT signed goal progress.',
              'No changes to archived JSON, physical simulation, policy, or PPO.',
              'Original strict pass and provisional mission pass are different questions.']
    return '\n'.join(lines) + '\n'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('json', nargs='+', type=Path)
    p.add_argument('--labels', nargs='+', help='name per file, in input order')
    p.add_argument('--output', type=Path, help='new destination file; never overwrite')
    a = p.parse_args()
    if a.labels is not None and len(a.labels) != len(a.json):
        p.error('--labels must match number of inputs')
    report = summarize(a.json, a.labels)
    if a.output:
        if a.output.exists():
            raise FileExistsError('Refusing to overwrite: ' + str(a.output))
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(report, encoding='utf-8')
        print(a.output)
    else:
        print(report, end='')


if __name__ == '__main__':
    main()
