#!/usr/bin/env python3
"""Produce a medium-sized, read-only report from two matched EMU physical exams.

Usage: python emu_physical_report.py v5030_temporal_physical_comparison.zip
       python emu_physical_report.py ~/robot_lab/emu_runs/v5030_temporal_reward_s747

Accepts either a completed comparison ZIP or an experiment run directory.
Writes Markdown and compact JSON beside the requested output location.  The
report never loads checkpoints, trains policies, or changes source files.
"""
from __future__ import annotations
import argparse
import csv
import io
import json
import math
import statistics
import zipfile
from pathlib import Path

SUITES = ('speed_ladder', 'terrain_matrix')
METRICS = (
    'achieved_mean_vx_mps', 'tracking_mae_vx_mps',
    'tracking_mae_vy_mps', 'tracking_signed_bias_yaw_rad_s',
    'tracking_jitter_yaw_rad_s', 'mechanical_j_per_travel_m',
    'first_episode_frames',
)


def avg(items):
    numbers = [float(v) for v in items if isinstance(v, (int, float)) and math.isfinite(float(v))]
    return statistics.mean(numbers) if numbers else None


def format_number(value, digits=3):
    return '—' if value is None else f'{value:.{digits}f}'


def fraction(a, b):
    return f'{a}/{b}' if b else '—'


def unpack(source):
    source = Path(source).expanduser()
    data = {}
    if source.is_file():
        with zipfile.ZipFile(source) as z:
            names = set(z.namelist())
            for suite in SUITES:
                for name in sorted(names):
                    suffix = '/' + suite + '/unified_exam.json'
                    if not name.endswith(suffix):
                        continue
                    parts = name.split('/')
                    if len(parts) == 3 or (len(parts) == 4 and parts[0] == 'physical_exams'):
                        arm = parts[-3]
                        data.setdefault(arm, {})[suite] = json.loads(z.read(name))
            for filename in ('campaign_manifest.json', 'overnight_summary.csv'):
                if filename in names:
                    data['_'+filename] = z.read(filename).decode('utf-8-sig')
    elif source.is_dir():
        base = source / 'physical_exams' if (source / 'physical_exams').exists() else source
        for arm_dir in base.iterdir():
            if not arm_dir.is_dir():
                continue
            for suite in SUITES:
                path = arm_dir / suite / 'unified_exam.json'
                if path.is_file():
                    data.setdefault(arm_dir.name, {})[suite] = json.loads(path.read_text())
        for filename in ('campaign_manifest.json', 'overnight_summary.csv'):
            p = source / filename
            if p.is_file():
                data['_'+filename] = p.read_text(encoding='utf-8-sig')
    else:
        raise FileNotFoundError(source)
    arms = {key: value for key, value in data.items() if not key.startswith('_')}
    if len(arms) != 2 or any(set(value) != set(SUITES) for value in arms.values()):
        raise ValueError('Expected exactly two complete arms, with speed_ladder and terrain_matrix each')
    return data, arms


def validate(arms, manifest):
    names = sorted(arms)
    shas = {suite_data['source_commit'] for by_suite in arms.values() for suite_data in by_suite.values()}
    if len(shas) != 1:
        raise ValueError('Physical examinations have different source commits')
    source_sha = next(iter(shas))
    if manifest and manifest.get('source_sha') != source_sha:
        raise ValueError('Campaign manifest and physical exam source mismatch')
    for suite in SUITES:
        left, right = (arms[n][suite] for n in names)
        if [row['case'] for row in left['results']] != [row['case'] for row in right['results']]:
            raise ValueError('Physical cases differ between arms in '+suite)
        for field in ('backend','contact','horizon','seed','speed','worlds','model',
                      'physical_fixture_revision','ladder_speeds','matrix_heights_cm','matrix_grades_deg'):
            if left['protocol'].get(field) != right['protocol'].get(field):
                raise ValueError(f'Unmatched {suite} protocol: {field}')
        for arm in names:
            if arms[arm][suite]['source_training_step'] != arms[arm][SUITES[0]]['source_training_step']:
                raise ValueError('Different checkpoint training step within arm '+arm)
            if arms[arm][suite]['actor_artifact_sha256'] != arms[arm][SUITES[0]]['actor_artifact_sha256']:
                raise ValueError('Different actor artifact within arm '+arm)
    return names, source_sha


def summarize_case(row):
    worlds = row['worlds']
    s = dict(name=row['case']['name'], case=row['case'], count=len(worlds),
             mission=row['mission_passed'], strict=row['passed'], survived=row['survived'],
             crossed=row['crossed'])
    for field in METRICS:
        values = [world.get(field) for world in worlds]
        if field == 'tracking_signed_bias_yaw_rad_s':
            s['mean_abs_yaw_bias'] = avg([abs(v) if v is not None else None for v in values])
        s[field] = avg(values)
    for field in ('movement_tracking_pass', 'heading_control_pass'):
        s[field] = sum(world.get(field) is True for world in worlds)
    if row['case'].get('kind') == 'step':
        s['by_direction'] = {}
        for direction in ('step-up', 'step-down'):
            w = [world for world in worlds if world.get('direction') == direction]
            s['by_direction'][direction] = {
                'worlds': len(w), 'crossed': sum(x.get('crossing_pass') is True for x in w),
                'mission': sum(x.get('mission_pass') is True for x in w),
                'survived': sum(x.get('survived') is True for x in w),
            }
    return s


def fnum(row, key, digits=3):
    return format_number(row.get(key), digits)


def render(arms, names, source_sha, campaign):
    a, b = names
    summ = {arm: {suite: [summarize_case(r) for r in arms[arm][suite]['results']] for suite in SUITES}
            for arm in names}
    title = '# EMU — matched physical comparison (decision report)'
    out = [title, '', f'**Source SHA:** `{source_sha}`  ',
           f'**Arms:** A = `{a}`, B = `{b}`.  ',
           '**Interpretation:** physical mission, supported crossing, survival and tracking are separate measures. This is a single-seed screen, not a statistical guarantee.', '']
    for arm in names:
        j=arms[arm]['speed_ladder']
        out.append(f'**{arm}:** {j["source_training_step"]:,} steps; actor SHA256 `{j["actor_artifact_sha256"]}`.  ')
    sample=arms[a]['speed_ladder']['protocol']
    out.extend(['',f'Fixture: {sample.get("physical_fixture_revision")}; {sample["worlds"]} worlds per speed, {sample["horizon"]} frames, seed {sample["seed"]}.', ''])
    if campaign:
        out.extend(['## Frozen evaluation (broad cohort)', '', '| Arm | Broad score | Fall rate (training) |', '|---|---:|---:|'])
        for arm in names:
            row = next((r for r in campaign if r.get('name') == arm), {})
            out.append(f'| {arm} | {row.get("fixed_exam_score") or "—"} | {row.get("fall_rate") or "—"} |')
        out.append('')
    out.extend(['## Actual forward walking — same commands', '',
        '| Request | A actual | B actual | A vx MAE | B vx MAE | A survived | B survived | A mission | B mission |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|'])
    for left,right in zip(summ[a]['speed_ladder'],summ[b]['speed_ladder']):
        speed=left['case'].get('vx')
        # At <100 frames mean achieved speed is a misleading transient rather than steady walking.
        def achieved(x):
            if (x.get('first_episode_frames') or 0) < 100: return 'early fall'
            return fnum(x,'achieved_mean_vx_mps')
        out.append(f'| {speed:.2f} | {achieved(left)} | {achieved(right)} | '
                   f'{fnum(left,"tracking_mae_vx_mps")} | {fnum(right,"tracking_mae_vx_mps")} | '
                   f'{fraction(left["survived"],left["count"])} | {fraction(right["survived"],right["count"])} | '
                   f'{fraction(left["mission"],left["count"])} | {fraction(right["mission"],right["count"])} |')
    out.extend(['', 'Average speed/MAE use available per-world values and include early-terminated worlds where reported. "Early fall" suppresses misleading average speed when the **mean episode lasts under 100 control frames**; it is not a sustained speed result.',
                '', '## Straight-line directional control — separate drift from oscillation', '',
                '| Requested vx | A abs yaw bias | B abs yaw bias | A yaw jitter | B yaw jitter | A heading pass | B heading pass |',
                '|---:|---:|---:|---:|---:|---:|---:|'])
    for left,right in zip(summ[a]['speed_ladder'],summ[b]['speed_ladder']):
        speed=left['case'].get('vx')
        def qualified(x,field):
            return fnum(x,field) if (x.get('first_episode_frames') or 0)>=100 else 'early fall'
        out.append(f'| {speed:.2f} | {qualified(left,"mean_abs_yaw_bias")} | {qualified(right,"mean_abs_yaw_bias")} | '
                   f'{qualified(left,"tracking_jitter_yaw_rad_s")} | {qualified(right,"tracking_jitter_yaw_rad_s")} | '
                   f'{fraction(left["heading_control_pass"],left["count"])} | {fraction(right["heading_control_pass"],right["count"])} |')
    out.extend(['', 'Yaw bias = mean of each world’s **absolute signed** yaw-rate bias (rad/s), not the mean of absolute instantaneous yaw rates. Jitter = the exam’s measured yaw-rate fluctuation (rad/s). Stride-scale jitter is a diagnostic, not an automatic reason to penalise the gait. Bias can accumulate into meaningful heading drift; verify with path traces.', ''])
    out.extend(['## Physically supported steps — UP versus DOWN', '',
                '| Height | A UP | B UP | A DOWN | B DOWN | A mission total | B mission total |',
                '|---:|---:|---:|---:|---:|---:|---:|'])
    for left,right in zip(summ[a]['terrain_matrix'],summ[b]['terrain_matrix']):
        if not left['name'].startswith('paired_steps_'):continue
        u1,d1=left['by_direction']['step-up'],left['by_direction']['step-down']
        u2,d2=right['by_direction']['step-up'],right['by_direction']['step-down']
        height=left['case'].get('height_cm')
        if height is None: height=left['name'].replace('paired_steps_','').replace('p','.').replace('cm','')
        out.append(f'| {height} cm | {fraction(u1["crossed"],u1["worlds"])} | {fraction(u2["crossed"],u2["worlds"])} | '
                   f'{fraction(d1["crossed"],d1["worlds"])} | {fraction(d2["crossed"],d2["worlds"])} | '
                   f'{fraction(left["mission"],left["count"])} | {fraction(right["mission"],right["count"])} |')
    out.extend(['', 'Supported crossings require real far-side contact and five supported frames. "Mission" also requires motion/heading tracking, and is deliberately different from physical crossing.', '',
                '## Slope / steering missions and survival', '',
                '| Test | A mission | B mission | A survived | B survived | A vx actual | B vx actual |',
                '|---|---:|---:|---:|---:|---:|---:|'])
    for left,right in zip(summ[a]['terrain_matrix'],summ[b]['terrain_matrix']):
        if left['name'].startswith('paired_steps_'):continue
        out.append(f'| {left["name"]} | {fraction(left["mission"],left["count"])} | '
                   f'{fraction(right["mission"],right["count"])} | '
                   f'{fraction(left["survived"],left["count"])} | '
                   f'{fraction(right["survived"],right["count"])} | '
                   f'{fnum(left,"achieved_mean_vx_mps")} | {fnum(right,"achieved_mean_vx_mps")} |')
    out.extend(['', '## Evaluation contract and limitations', '',
        '- The two arms share the same examined source commit, requested cases, seed, fixture and test durations; their checkpoint/actor hashes differ as expected.',
        '- No automatic treatment winner: compare actual speed, drift, crossings, falls and grade performance by mission. The broad frozen score alone is insufficient.',
        '- 0/16 mission at faster gait can result from yaw-jitter thresholds even when support, movement and speed are useful. Do not confuse gait oscillation with long-term directional error.',
        '- The 1.0m/s test may include rapid failure; do not cite transient speeds as evidence of sustained walking. Consider a progressive-speed exam.',
        '- Per-world evidence and uncertainty remain in the original JSON. Any 8/8 or 2/8 directional crossing count has wide uncertainty.',
        '- Mechanical shaft work per travel metre is only a proxy for energy usage. It is not electrical or battery energy.',
        '- This report does not touch training, rewards, actor data, checkpoints, or the main branch.', ''])
    return '\n'.join(out), summ


def run(source, destination=None):
    data, arms = unpack(source)
    manifest = json.loads(data['_campaign_manifest.json']) if '_campaign_manifest.json' in data else None
    names, sha=validate(arms,manifest)
    campaign=list(csv.DictReader(io.StringIO(data['_overnight_summary.csv']))) if '_overnight_summary.csv' in data else []
    md,summary=render(arms,names,sha,campaign)
    target=Path(destination).expanduser() if destination else Path(source).expanduser().parent
    target.mkdir(parents=True,exist_ok=True)
    markdown=target/'physical_comparison_report.md'
    small=target/'physical_comparison_report.json'
    markdown.write_text(md+'\n',encoding='utf-8')
    small.write_text(json.dumps({'source_sha':sha,'arms':names,'results':summary},indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return markdown,small


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path,help='physical comparison ZIP or experiment root')
    p.add_argument('--output-dir',type=Path)
    args=p.parse_args()
    for f in run(args.input,args.output_dir): print(f)

if __name__=='__main__':main()
