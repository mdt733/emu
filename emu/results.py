"""Offline EMU results.md + evidence ZIP, for shared-foundation and forked runs.

Never imports JAX, opens checkpoint pickle objects, or reruns evaluations.
Use python3 -m emu.results --root /path/to/campaign.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import zipfile
from pathlib import Path

FILES = ('metrics.csv', 'fixed_eval.csv', 'screen_eval.json', 'training_config.json',
         'run_meta.json', 'lab_spec.json', 'envelope_summary.json',
         'performance.json', 'capability_field_summary.json')
FIELDS = ('reward_mean', 'fall_rate', 'ep_len_mean', 'height_error_m', 'step_evidence',
          'vx_mean', 'cmd_vx_mean', 'servo_shaft_abs_power_w',
          'servo_shaft_drive_power_w', 'servo_shaft_brake_power_w',
          'servo_over_rated_frac', 'servo_peak_torque_limit_frac',
          'spring_potential_energy_j', 'shin_spring_energy_j',
          'steps_per_sec', 'ppo_approx_kl', 'nan_grad_skips_total')
PAYMENTS = ('alive', 'locomotion', 'posture', 'contact', 'anti_slip',
            'clearance', 'smoothness', 'joint_safety', 'fall_penalty')
SCREENS = ('resilience100', 'robust70', 'growth140', 'growthcombo140', 'gentle_slope')
FIXED = ('core', 'stress30', 'dynamic50', 'robust40', 'combo70',
         'resilience50', 'resilience75', 'dynamic80', 'growth110', 'growthcombo110')
MARK = chr(96)


def read_json(path):
    try:
        return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}
    except (OSError, ValueError, UnicodeError) as exc:
        return {'_read_error': str(exc)}


def read_csv(path):
    if not path.is_file():
        return []
    try:
        with path.open(newline='', encoding='utf-8-sig') as stream:
            return list(csv.DictReader(stream))
    except (OSError, csv.Error, UnicodeError):
        return []


def number(row, key):
    try:
        x = float(row.get(key, ''))
        return x if math.isfinite(x) else None
    except (TypeError, ValueError, AttributeError):
        return None


def pretty(x, factor=1):
    return '—' if x is None else f'{x * factor:.3f}'


def text(x):
    return str(x).replace('|', '/').replace('\r', ' ').replace('\n', ' ')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def summarize(run, expected=None):
    logged = [r for r in read_csv(run / 'metrics.csv') if number(r, 'step') is not None]
    unique = {int(number(r, 'step')): r for r in logged}
    logged = [unique[k] for k in sorted(unique)]
    count = len(logged)
    window = max(1, min(12, math.ceil(.1 * count))) if count else 0

    def median(rows, key):
        values = [number(r, key) for r in rows]
        values = [v for v in values if v is not None]
        return statistics.median(values) if values else None

    early, late = logged[:window], logged[-window:] if window else []
    changes = {k: {'early': median(early, k), 'late': median(late, k)} for k in FIELDS}
    rewards = {k: median(late, 'reward_weighted_' + k) for k in PAYMENTS}
    config = read_json(run / 'training_config.json')
    meta = read_json(run / 'run_meta.json')
    spec = read_json(run / 'lab_spec.json')
    screen = read_json(run / 'screen_eval.json')
    fixed = read_csv(run / 'fixed_eval.csv')
    final = logged[-1] if count else {}
    step, target = number(final, 'step'), number(config, 'steps')
    checkpoint = run / 'checkpoints' / 'final.pkl'
    sha = digest(checkpoint) if checkpoint.is_file() and not checkpoint.is_symlink() else None
    problems = []
    if not logged:
        problems.append('training metrics missing/empty')
    if not fixed:
        problems.append('fixed evaluation missing/empty')
    if fixed and step is not None:
        fixed_step = number(fixed[-1], 'step')
        if fixed_step is not None and fixed_step < step:
            problems.append('fixed evaluation predates latest logged policy')
    if not screen and run.name != 'foundation20':
        problems.append('independent screening details missing')
    if not sha:
        problems.append('checkpoint absent; identity cannot be verified')
    if (expected or {}).get('checkpoint_sha256') and (expected['checkpoint_sha256'] != sha):
        problems.append('SCORECARD CHECKPOINT HASH MISMATCH / MISSING')
    if target is not None and (step is None or step < target):
        problems.append('logged steps are below target')
    if any(d.get('_read_error') for d in (config, meta, spec, screen)):
        problems.append('a JSON evidence file is unreadable')
    if logged and any(v is None for v in rewards.values()):
        problems.append('reward payment columns are incomplete')
    return dict(path=str(run), step=int(step) if step is not None else None,
                target=int(target) if target is not None else None, log_rows=count,
                changes=changes, rewards=rewards, config=config,
                source=meta.get('source_provenance', {}), parent=spec.get('parent', {}),
                checkpoint_sha256=sha, screen=screen,
                fixed=fixed[-1] if fixed else {}, problems=problems)


def collect(root):
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f'No campaign directory: {root}')
    context = read_json(root / 'lab_context.json')
    boards = {p.stem.split('_')[0]: read_json(p)
              for p in sorted(root.glob('round?_scoreboard.json'))}
    dirs = [root/'foundation20'] if (root/'foundation20').is_dir() else []
    for stage in ('round1', 'round2', 'round3'):
        folder = root / stage
        if folder.is_dir():
            dirs.extend(sorted(p for p in folder.iterdir() if p.is_dir() and not p.is_symlink()))
    if not dirs and (root/'metrics.csv').is_file():
        dirs = [root]  # one-run emu train experiments
    if not dirs and not boards:
        raise ValueError('No campaign runs or scoreboards found under root')
    runs = {}
    for path in dirs:
        name = str(path.relative_to(root)) if path != root else 'single'
        stage, _, arm = name.partition('/')
        scored = next((r for r in boards.get(stage, {}).get('results', [])
                       if r.get('arm') == arm), None)
        runs[name] = summarize(path, scored)
    foundation = runs.get('foundation20', {}).get('checkpoint_sha256')
    for name, run in runs.items():
        original = context.get('git_commit')
        actual = run['source'].get('git_commit')
        if original and actual and original != actual:
            run['problems'].append('SOURCE COMMIT DIFFERS from campaign')
        if name.startswith('round1/') and foundation:
            parent = run['parent'].get('sha256')
            if parent and parent != foundation:
                run['problems'].append('PARENT HASH DIFFERS from foundation')
    return dict(schema=1, campaign=root.name, context=context, boards=boards, runs=runs)


def _table(lines, headers, rows):
    lines.extend(['| ' + ' | '.join(headers) + ' |',
                  '|' + '|'.join(['---'] * len(headers)) + '|'])
    for row in rows:
        lines.append('| ' + ' | '.join(map(text, row)) + ' |')
    lines.append('')


def _panel_table(lines, values, panels, prefix):
    rows = []
    for panel in panels:
        s = number(values, prefix + panel + '_survival')
        if s is None or s < 0:
            continue
        rows.append((panel, pretty(s, 100)+'%',
                     pretty(number(values, prefix+panel+'_vx_mae')),
                     pretty(number(values, prefix+panel+'_vy_mae')),
                     pretty(number(values, prefix+panel+'_yaw_mae')),
                     pretty(number(values, prefix+panel+'_height_shortfall_m'), 1000)))
    if rows:
        _table(lines, ('Panel', 'Survival', 'vx MAE', 'vy MAE', 'yaw MAE',
                       'Height shortfall mm'), rows)
    else:
        lines.extend(['No panel-level evaluation data available.', ''])


def markdown(info):
    ctx = info['context']
    lines = ['# EMU campaign results — ' + text(info['campaign']), '',
             'Offline report: no training, evaluation or policy changes were performed.', '',
             '## Provenance', '',
             '- Training Git commit: ' + MARK + text(ctx.get('git_commit', 'unknown')) + MARK,
             '- Seed: ' + text(ctx.get('seed', 'unknown')) + '; worlds: ' + text(ctx.get('envs', 'unknown')),
             '- Foundation checkpoint is shared. Per-policy steps are cumulative.', '']
    for stage, board in info['boards'].items():
        lines.extend(['## ' + stage + ': common held-out ranking', '',
                      'Matched exam: ' + text(board.get('screen_envs', '?')) + ' worlds × ' +
                      text(board.get('screen_horizon', '?')) + ' steps; pitch ' +
                      text(board.get('screen_pitch_deg', '?')) + ' degrees.', ''])
        _table(lines, ('Arm', 'Eligible', 'Score', 'Hard survival', 'Growth survival',
                       'Pitch MAE', 'Height shortfall mm'),
               [(r.get('arm', '?'), 'yes' if r.get('eligible') is True else 'no',
                 pretty(number(r, 'score')), pretty(number(r, 'hard_survival'),100)+'%',
                 pretty(number(r, 'growth_survival'),100)+'%',
                 pretty(number(r, 'pitch_mae_rad')),
                 pretty(number(r, 'height_shortfall_m'),1000))
                for r in board.get('results', [])])
        lines.extend(['Rank is an existing eligibility-weighted score, not proof of walking.', ''])
        for r in board.get('results', []):
            name = stage + '/' + str(r.get('arm', ''))
            if name in info['runs']:
                lines.extend(['**'+text(name)+' — detailed common exam**', ''])
                _panel_table(lines, info['runs'][name]['screen'], SCREENS, 'screen_')
    lines.extend(['## Training trends (not held-out comparisons)', '',
                  'Early/late values are medians of first/last 10% of logged rows (up to 12 rows).', ''])
    _table(lines, ('Run','Step','Reward early → late','Fall early → late',
                   'Height error early → late (mm)','Torque over rating late'),
           [(name, row['step'] or '—',
             pretty(row['changes']['reward_mean']['early'])+' → '+pretty(row['changes']['reward_mean']['late']),
             pretty(row['changes']['fall_rate']['early'],100)+' → '+pretty(row['changes']['fall_rate']['late'],100)+'%',
             pretty(row['changes']['height_error_m']['early'],1000)+' → '+pretty(row['changes']['height_error_m']['late'],1000),
             pretty(row['changes']['servo_over_rated_frac']['late'],100)+'%')
            for name, row in info['runs'].items()])
    for name, run in info['runs'].items():
        lines.extend(['## Run: '+text(name), '',
                      '- Logged/target steps: '+text(run['step'])+' / '+text(run['target']),
                      '- CSV observations: '+str(run['log_rows']),
                      '- Final policy SHA-256: '+MARK+text(run['checkpoint_sha256'] or 'unavailable')+MARK,
                      '- Parent SHA-256: '+MARK+text(run['parent'].get('sha256', 'unavailable'))+MARK,
                      '- Training commit: '+MARK+text(run['source'].get('git_commit', 'unavailable'))+MARK,
                      '- Training profile: teacher '+text(run['config'].get('capability_field_mode', '?'))+
                      ', terrain radius '+text(run['config'].get('domain_terrain_radius', '?'))+
                      ', pitch '+text(run['config'].get('pitch_command_max_deg', '?'))+' degrees.', '',
                      '### Reward payments (late on-policy samples)', ''])
        _table(lines, ('Payment','Contribution'),
               [(key, pretty(value)) for key,value in run['rewards'].items()])
        lines.extend(['### Measurement trends', ''])
        _table(lines, ('Measurement','Early','Late'),
               [(k, pretty(v['early']), pretty(v['late'])) for k,v in run['changes'].items()])
        lines.extend(['### Independent fixed evaluation', ''])
        _panel_table(lines, run['fixed'], FIXED, 'eval_')
        if run['problems']:
            lines.extend(['**Warnings / missing evidence:** ' + '; '.join(map(text, run['problems'])), ''])
    if not info['runs']:
        lines.extend(['Only scorecards were found; branch logs and physical diagnostics are unavailable.', ''])
    lines.extend(['## Interpretation and sharing', '',
                  '- Foot lift and foot-height support are proxies, not certified completed strides.',
                  '- Shaft torque times speed is mechanical, not electrical power; stall can draw current at zero shaft speed.',
                  '- Spring potential is stored energy, not necessarily returned useful work.',
                  '- On-policy effort/survival reflects different curricula; compare policies under matched exam conditions.',
                  '- A one-seed ranking is exploratory, not proof of a superior gait.', '',
                  'The results_bundle.zip archive contains the report, machine-readable results and underlying CSV/JSON',
                  'for every available branch. No checkpoints or console logs are included.', ''])
    return '\n'.join(lines)


def write_scorecard(root: Path, board: dict) -> Path:
    """Render one held-out study round; ranking/eligibility remain with lab.

    This reporter does not load policy checkpoints or influence promotion.
    The generated Markdown is intentionally independent of GPU or PPO.
    """
    round_name = board['round']
    rows = board['results']
    target = int(board['target'])
    pitch = float(board['screen_pitch_deg'])
    lines = [f'# EMU v5015 {round_name} held-out comparisons', '',
             f'Every policy at {target:,} absolute steps in the same {pitch:g} degree testing envelope.',
             'Only eligible arms auto-advance; manual review remains available. Screen scores are not proof of locomotion; stepping and support lack calibrated minimum gates.', '',
             '| arm | eligible | score | hard survival | growth survival | slope survival | motion MAE | step evidence | support | pitch MAE | height shortfall | reasons |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for row in rows:
        lines.append(f"| {row['arm']} | {row['eligible']} | {row['score']} | "
                     f"{row.get('hard_survival','—')} | {row.get('growth_survival','—')} | "
                     f"{row.get('gentle_slope_survival','—')} | "
                     f"{row.get('motion_mae','—')} | {row.get('step_evidence','—')} | "
                     f"{row.get('support','—')} | {row.get('pitch_mae_rad','—')} rad | "
                     f"{row.get('height_shortfall_m','—')} m | "
                     f"{', '.join(row['reasons'])} |")
    report = Path(root) / f'{round_name}_scorecard.md'
    temp = report.with_name(report.name + '.tmp')
    temp.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    temp.replace(report)
    return report


def write_report(root, *, bundle=True):
    root = Path(root).expanduser().resolve()
    data = collect(root)
    outputs = []
    for filename, content in (('results.md', markdown(data)),
                              ('results_data.json', json.dumps(data, indent=2, sort_keys=True, allow_nan=False))):
        dest = root / filename
        temp = root / (filename+'.tmp')
        temp.write_text(content+'\n', encoding='utf-8')
        temp.replace(dest)
        outputs.append(dest)
    if bundle:
        dest = root / 'results_bundle.zip'
        temp = root / 'results_bundle.zip.tmp'
        with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for path in outputs:
                z.write(path,arcname=path.name)
            for filename in ('lab_context.json', 'round1_scoreboard.json', 'round1_scorecard.md',
                             'round2_scoreboard.json', 'round2_scorecard.md',
                             'round3_scoreboard.json', 'round3_scorecard.md'):
                path = root / filename
                if path.is_file() and not path.is_symlink():
                    z.write(path,arcname=filename)
            for name,run in data['runs'].items():
                folder = Path(run['path'])
                for filename in FILES:
                    path = folder / filename
                    if path.is_file() and not path.is_symlink():
                        z.write(path,arcname=name+'/'+filename)
        temp.replace(dest)
        outputs.append(dest)
    return outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description='Offline EMU results.md and shareable evidence ZIP')
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--no-bundle',action='store_true')
    args = parser.parse_args(argv)
    for path in write_report(args.root,bundle=not args.no_bundle):
        print(path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
