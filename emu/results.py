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
         'performance.json', 'capability_field_summary.json', 'capability_evidence.json')
FIELDS = ('reward_mean', 'fall_rate', 'ep_len_mean', 'height_error_m', 'step_evidence',
          'vx_mean', 'cmd_vx_mean', 'servo_shaft_abs_power_w',
          'servo_shaft_drive_power_w', 'servo_shaft_brake_power_w',
          'servo_over_rated_frac', 'servo_peak_torque_limit_frac',
          'spring_potential_energy_j', 'shin_spring_energy_j',
          'steps_per_sec', 'ppo_approx_kl', 'nan_grad_skips_total')
PAYMENTS = ('alive', 'locomotion', 'posture', 'contact', 'anti_slip',
            'clearance', 'smoothness', 'joint_safety', 'fall_penalty', 'motor_cost')
SCREENS = ('resilience100', 'robust70', 'growth140', 'growthcombo140', 'gentle_slope')
FIXED = ('core', 'stress30', 'dynamic50', 'robust40', 'combo70',
         'resilience50', 'resilience75', 'dynamic80', 'growth110', 'growthcombo110')
STORY_FIELDS = ('field_known_nodes', 'field_known_edges', 'frontier_vx_hi')
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


def training_story(logged, changes):
    """Describe changing-task training metrics, NOT independent walking ability."""
    if len(logged) < 3:
        return ['Too few logged observations to describe an early-to-late trend.']

    def pair(key):
        values = changes.get(key, {})
        return values.get('early'), values.get('late')

    notes = []
    before, after = pair('reward_mean')
    if before is not None and after is not None:
        trend = 'rose' if after > before + .02 else ('fell' if after < before - .02 else 'changed little')
        notes.append(f'On-policy teaching reward {trend}: early {before:.3f}, late {after:.3f}.')
    before, after = pair('fall_rate')
    if before is not None and after is not None:
        trend = 'fell' if after < before - .02 else ('rose' if after > before + .02 else 'changed little')
        notes.append(f'Training-episode fall fraction {trend}: early {before:.1%}, late {after:.1%}.')
    for key, label in (('field_known_nodes', 'mapped command nodes'),
                       ('field_known_edges', 'mapped transition edges')):
        before, after = pair(key)
        if before is not None and after is not None and after > before + .5:
            notes.append(f'Capability evidence covered more {label}: {before:.0f} to {after:.0f}; this is coverage, not mastery.')
    before, after = pair('frontier_vx_hi')
    if before is not None and after is not None and after > before + .005:
        notes.append(f'Allowed forward-command frontier expanded from {before:.3f} to {after:.3f} m/s; allowed is not achieved speed.')
    if len(logged) >= 8:
        width = max(2, min(12, len(logged)//5))
        windows = []
        for start in range(len(logged)-width+1):
            rows = logged[start:start+width]
            vals = [number(row, 'reward_mean') for row in rows]
            if all(v is not None for v in vals):
                windows.append((statistics.median(vals), int(number(rows[-1], 'step') or 0)))
        if windows:
            peak, peak_step = max(windows)
            late_reward = pair('reward_mean')[1]
            if late_reward is not None and peak-late_reward > max(.15, abs(peak)*.20):
                notes.append(f'Possible training setback: best {width}-row reward median {peak:.3f} near step {peak_step:,}, versus late {late_reward:.3f}. Harder lessons may explain this.')
    return notes or ['Comparable training trends unavailable; inspect the frozen-policy exam.']


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
    changes = {k: {'early': median(early, k), 'late': median(late, k)} for k in (*FIELDS, *STORY_FIELDS)}
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
                changes=changes, story=training_story(logged, changes),
                rewards=rewards, config=config,
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


def comparison_story(board):
    """Read-only interpretation of a matched screen; never promote policies."""
    rows = board.get('results', [])
    eligible = sorted((row for row in rows if row.get('eligible') is True
                       and number(row, 'score') is not None),
                      key=lambda row: (-number(row, 'score'), str(row.get('arm', ''))))
    if not eligible:
        return ['No eligible policy with a finite common-screen score; do not infer a winner.']
    leader = eligible[0]
    if len(eligible) > 1:
        other = eligible[1]
        margin = number(leader, 'score') - number(other, 'score')
        story = [f"Under the common held-out screen, {leader['arm']} leads {other['arm']} by {margin:.3f} score points among {len(eligible)} eligible policies."]
    else:
        story = [f"Only {leader['arm']} has an eligible common-screen score; there is no eligible runner-up."]
    excluded = [row for row in rows if row.get('eligible') is not True
                or number(row, 'score') is None]
    if excluded:
        detail = '; '.join(
            f"{row.get('arm', '?')}: {', '.join(row.get('reasons', [])) or 'missing eligible score'}"
            for row in excluded)
        story.append(f'Not eligible for automatic promotion: {detail}.')
    story.append('Scores are from matched frozen-policy screens, not training rewards; one seed is not proof of walking.')
    return story


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
        lines.extend('- ' + text(line) for line in comparison_story(board))
        lines.append('')
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
                      '### Training narrative (on-policy, not a walking grade)', ''])
        lines.extend('- '+text(item) for item in run['story'])
        lines.append('')
        lines.extend(['### Reward payments (late on-policy samples)', ''])
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



# Run logging, metric tables and field evidence: same reporting owner.
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def ensure_run_dir(path: Path) -> None:
    (path / 'checkpoints').mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')



def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write heterogeneous row dicts using first-seen column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open('r', newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def append_metrics_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fields = list(row.keys())
    if exists:
        with path.open('r', newline='', encoding='utf-8') as f:
            old_header = next(csv.reader(f))
        if fields != old_header:
            missing = [x for x in old_header if x not in fields]
            added = [x for x in fields if x not in old_header]
            raise RuntimeError(f'metrics schema changed for {path}: added={added} missing={missing}')
    with path.open('a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        w.writerow(row)


def read_metrics_csv(path: Path) -> list[dict[str, str]]:
    return read_csv_rows(path)


def _fmt_len(v: float) -> str:
    # sentinel -1.0 from masked_mean means "no episodes of this kind this rollout".
    return '-' if v < 0 else f"{v:.0f}"


def format_line(version: str, step: int, reward: float, row: dict[str, float], report: dict[str, Any], train_fps: float,
                 wall_fps: float | None = None, nan_grad_skips: int = 0) -> str:
    fail = f"{100*row['fall_height']:.0f}/{100*row['fall_tilt']:.0f}/{100*row['fall_roll_rate']:.0f}/{100*row['fall_pitch_rate']:.0f}"
    line1 = (f"[{version}] step={step:,} rew={reward:+.3f} len={row['ep_len_mean']:.0f} "
             f"fall={100*row['fall_rate']:.1f}% train_fps={train_fps:.0f} wall_fps={(wall_fps if wall_fps is not None else train_fps):.0f} fail(h/t/r/p)={fail} "
             f"vx={row['vx_mean']:+.3f}/{row['cmd_vx_mean']:+.3f} cone={row['foot_mid_angle_deg']:.1f}deg "
             f"step={row['step_evidence']:.2f} sync={row['phase_sync']:.2f}")
    # Three regimes' difficulty + live env-allocation fraction, plus
    # posture's own global dial.
    line2 = (f"[{version}]   steady={report['steady_difficulty']:.2f}(env={report['regime_frac_steady']:.2f}) "
             f"transition={report['transition_difficulty']:.2f}(env={report['regime_frac_transition']:.2f}) "
             f"resilience={report['resilience_difficulty']:.2f}(env={report['regime_frac_resilience']:.2f}) "
             f"posture={report['posture_difficulty']:.2f} | "
             f"fall_ema steady/trans/res/post={report['fall_ema_steady']:.2f}/{report['fall_ema_transition']:.2f}/"
             f"{report['fall_ema_resilience']:.2f}/{report['fall_ema_posture']:.2f} target={report['fall_target']:.2f} | "
             f"fail_len={_fmt_len(row.get('fail_ep_len_mean', -1.0))} "
             f"timeout_len={_fmt_len(row.get('timeout_ep_len_mean', -1.0))} | "
             f"r_track={row['reward_command_tracking']:.2f} r_contact={row['reward_feet_only_contact']:.2f} | "
             f"nan_skips={nan_grad_skips} ev={row.get('explained_variance', 0.0):.2f} "
             f"expl_std={row.get('exploration_std_mean', 0.0):.3f}")
    line3 = (f"[{version}]   events steady/trans/res/post="
             f"{report['event_steady']}/{report['event_transition']}/{report['event_resilience']}/{report['event_posture']}"
             f"{' GLOBAL-RETREAT' if report.get('global_retreat') else ''} "
             f"comp steady/trans/res={report['competence_steady']:.2f}/{report['competence_transition']:.2f}/{report['competence_resilience']:.2f} "
             f"core_comp={report['core_competence']:.2f} core_bad_streak={int(report['core_bad_streak'])} "
             f"core_fall={row.get('core_fall_rate', -1.0):.2f} core_len={_fmt_len(row.get('core_ep_len_mean', -1.0))} "
             f"precision_vx={report['precision_vx']:.2f} combo={report.get('precision_combo',-1.0):.2f} kl_stress={report['kl_stopped_frac_seen']:.2f}"
             f"{'(CHRONIC-IGNORED)' if report.get('kl_stress_chronic', 0.0) else ''} "
             f"kl_end={row.get('ppo_approx_kl_final', 0.0):.3f} "
             f"eff_lr={row.get('ppo_effective_lr', 0.0):.2e} "
             f"a_clip={row.get('ppo_action_clip_frac', 0.0):.2f}")
    line4 = (f"[{version}]   frontier vx=[{report.get('frontier_vx_lo',0.0):+.3f},{report.get('frontier_vx_hi',0.0):+.3f}] "
             f"vy=±{report.get('frontier_vy',0.0):.3f} yaw=±{report.get('frontier_yaw',0.0):.3f} "
             f"combo={report.get('frontier_combo_radius',1.0):.3f} push={report.get('frontier_push_n',0.0):.3f}N "
             f"sensor={report.get('frontier_sensor_scale',0.0):.2f}x hold={report.get('frontier_transition_hold_s',0.0):.3f}s "
             f"ramp={report.get('frontier_ramp_steps',0.0):.2f}frames")
    line5 = (f"[{version}]   field nodes={row.get('field_known_nodes',0.0):.0f}/{row.get('field_competent_nodes',0.0):.0f} "
             f"frontier={row.get('field_frontier_nodes',0.0):.0f} q={row.get('field_mean_quality_known',0.0):.2f} "
             f"edges={row.get('field_known_edges',0.0):.0f}/{row.get('field_competent_edges',0.0):.0f} "
             f"recovery={row.get('field_known_recovery_cells',0.0):.0f}/{row.get('field_competent_recovery_cells',0.0):.0f} "
             f"events(n/r/e)+={row.get('field_node_events_update',0.0):.0f}/{row.get('field_recovery_events_update',0.0):.0f}/{row.get('field_edge_events_update',0.0):.0f} "
             f"teach(node/edge)={report.get('field_node_influence',0.0):.2f}/{report.get('field_edge_influence',0.0):.2f}")
    return line1 + "\n" + line2 + "\n" + line3 + "\n" + line4 + "\n" + line5



def attach_field_influence(report: dict[str, Any], params: dict[str, Any]) -> None:
    """Attach the teaching influence that generated the next task packet.

    Periodic and final metrics rows must share one schema. Keeping this in one
    helper prevents the final-row path from silently omitting these fields.
    """
    report['field_node_influence'] = float(params.get('field_node_frac', 0.0))
    report['field_edge_influence'] = float(params.get('field_edge_frac', 0.0))


def make_metrics_row(*, step: int, elapsed: float, wall_fps: float, train_fps: float,
                     reward: float, row: dict[str, Any], report: dict[str, Any], aux: dict[str, Any],
                     nan_grad_skips: int, num_envs: int, jax_backend: str, perf_row: dict[str, Any],
                     rollout_steps: int, base_lr: float) -> dict[str, Any]:
    """Canonical metrics.csv schema for periodic and final rows."""
    return {
        'step': step, 'elapsed_s': elapsed, 'steps_per_sec': wall_fps, 'train_steps_per_sec': train_fps,
        'reward_mean': reward, **row, **report,
        'backend': 'mjx', 'jax_backend': jax_backend, 'num_envs': num_envs, 'rollout': rollout_steps,
        'ppo_policy_loss': float(np.asarray(aux.get('policy_loss', 0.0))),
        'ppo_value_loss': float(np.asarray(aux.get('value_loss', 0.0))),
        'ppo_entropy': float(np.asarray(aux.get('entropy', 0.0))),
        'ppo_approx_kl': float(np.asarray(aux.get('approx_kl', 0.0))),
        'ppo_policy_loss_finite_frac': float(np.asarray(aux.get('policy_loss_finite', 1.0))),
        'ppo_value_loss_finite_frac': float(np.asarray(aux.get('value_loss_finite', 1.0))),
        'ppo_kl_stopped_frac': float(np.asarray(aux.get('kl_stopped', 0.0))),
        'ppo_approx_kl_final': float(np.asarray(aux.get('approx_kl_final', 0.0))),
        'ppo_lr_scale': float(np.asarray(aux.get('lr_scale', 1.0))),
        'ppo_effective_lr': base_lr * float(np.asarray(aux.get('lr_scale', 1.0))),
        'ppo_action_clip_frac': float(np.asarray(aux.get('action_clip_frac', 0.0))),
        'ppo_adv_tail_frac': float(np.asarray(aux.get('adv_tail_frac', 0.0))),
        'explained_variance': float(np.asarray(aux.get('explained_variance', 0.0))),
        'exploration_std_mean': float(np.asarray(aux.get('exploration_std_mean', 0.0))),
        'nan_grad_skips_total': nan_grad_skips,
        **perf_row,
    }


def append_training_observation(
    run_dir: Path, *, step: int, params: dict[str, Any],
    report: dict[str, Any], metrics_row: dict[str, Any],
) -> None:
    """Identical metrics/envelope output for intermediate and final records."""
    append_metrics_csv(run_dir / 'metrics.csv', metrics_row)
    write_json(run_dir / 'envelope_summary.json', {
        'step': step,
        'params': {k: v for k, v in params.items()
                   if not k.startswith('field_') or k.endswith('_frac')},
        'report': report,
        'last_row': metrics_row,
    })


def write_capability_field_artifacts(
    run_dir: Path, field: Any, *, step: int, mode: str,
    params: dict[str, Any], label: str,
) -> None:
    """Write latest/final field snapshot with one canonical summary."""
    if label not in ('latest', 'final'):
        raise ValueError(f"unknown field artifact label: {label}")
    field.save_npz(run_dir / f'capability_field_{label}.npz')
    field.write_summary(
        run_dir / 'capability_field_summary.json', step=step, mode=mode,
        node_frac=float(params.get('field_node_frac', 0.0)),
        edge_frac=float(params.get('field_edge_frac', 0.0)),
    )
    # Report-only projection of the EXISTING capability events. Nothing in
    # trainer or governor consumes this JSON as a lesson or PPO reward.
    write_json(run_dir / 'capability_evidence.json',
               capability_field_evidence_snapshot(field))


# PPO rollout observations: report facts, never pay training reward.
from typing import Any

import numpy as np

from .contracts import ACTION_DIM, COMBO_P, SEED_VX_LO, SEED_VX_HI, task_param_defaults
from .rewards import PAYMENT_NAMES, FALL_PAYMENT_NAME, MOTOR_PAYMENT_NAME
from .contracts import (
    FRONTIER_VX_ABS_TOL, FRONTIER_VX_REL_TOL, FRONTIER_VY_ABS_TOL, FRONTIER_VY_REL_TOL,
    FRONTIER_YAW_ABS_TOL, FRONTIER_YAW_REL_TOL, FRONTIER_EDGE_FRAC, FRONTIER_SETTLE_S,
)

def capability_evidence_report(*, family: str, source: str,
                               success: np.ndarray, failure: np.ndarray,
                               lower: np.ndarray, quality_sum: np.ndarray | None = None,
                               min_events: int = 20) -> dict[str, Any]:
    """Read-only evidence summary across an existing capability family.

    Each bucket must supply independent SUCCESS/FAILURE counts and a
    conservative lower confidence bound (e.g. CapabilityField.node_lower).
    This does not invent an overall quality score, mutate curriculum or
    treat unsampled conditions as either success or failure. Reward is
    intentionally not an input. Mechanical grace is not claimed unless
    its own quality measurements actually exist.
    """
    if source not in ("training_rollout", "held_out_exam"):
        raise ValueError("evidence source must distinguish training and held-out")
    if not family or min_events < 1:
        raise ValueError("family required; min_events must be >= 1")
    yes,no,lo=(np.asarray(x, dtype=np.float64) for x in (success,failure,lower))
    if yes.shape!=no.shape or yes.shape!=lo.shape or not yes.size:
        raise ValueError("success/failure/lower must have matching nonempty shapes")
    if any(not np.isfinite(x).all() for x in (yes,no,lo)):
        raise ValueError("evidence values must be finite")
    if (yes<0).any() or (no<0).any() or (lo<0).any() or (lo>1).any():
        raise ValueError("evidence counts/bounds must be nonnegative and bounded")
    samples=yes+no
    quality=None
    if quality_sum is not None:
        quality=np.asarray(quality_sum, dtype=np.float64)
        if (quality.shape!=yes.shape or not np.isfinite(quality).all()
                or (quality<0).any() or (quality>samples+1e-5).any()):
            raise ValueError("observed quality must be in [0, sample count]")
    known=samples>=min_events
    count=int(known.sum())
    attempted=float(samples[known].sum())
    return dict(
        family=family,source=source,min_events=min_events,
        total_bins=int(yes.size),known_bins=count,unknown_bins=int(yes.size-count),
        known_fraction=float(count/yes.size),
        independent_trials_in_known_bins=attempted,
        success_rate_known=float(yes[known].sum()/attempted) if attempted else None,
        conservative_success_lower_mean=float(lo[known].mean()) if count else None,
        observed_quality_known=float(quality[known].sum()/attempted)
        if quality is not None and attempted else None,
        # Deliberately no aggregate grade/rank or motor efficiency claim here.
    )


def capability_field_evidence_snapshot(field, min_events: int = 20) -> dict[str, Any]:
    """Report, but do not change, main's three evidence families.

    Command nodes have tracking/upright/support *observed* quality. Existing
    recovery/transition records track success only; their quality is not
    fabricated. Terrain and motor grace will require their OWN truth.
    """
    return {
        "schema": "emu_capability_evidence_1",
        "training_only": True,
        "qualification": "descriptive diagnostics, not independent exam results",
        "command_nodes": capability_evidence_report(
            family="command_nodes",source="training_rollout",
            success=field.node_success,failure=field.node_failure,
            lower=field.node_lower(),quality_sum=field.node_quality_sum,
            min_events=min_events),
        "recovery": capability_evidence_report(
            family="recovery",source="training_rollout",
            success=field.rec_success,failure=field.rec_failure,
            lower=field.recovery_lower(),min_events=min_events),
        "transitions": capability_evidence_report(
            family="transitions",source="training_rollout",
            success=field.edge_success,failure=field.edge_failure,
            lower=field.edge_lower(),min_events=min_events),
    }


def mean_metric(metrics: dict[str, Any], name: str) -> float:
    if name not in metrics:
        return 0.0
    return float(np.asarray(metrics[name]).mean())


def done_mean(metrics: dict[str, Any], name: str, default: float = 0.0) -> float:
    # A thin wrapper over masked_mean(metrics, name, 'episode_done',
    # default) -- kept as its own function since "done_mean" reads better
    # than the fully spelled-out call at its 9 call sites.
    return masked_mean(metrics, name, 'episode_done', default)


def masked_mean(metrics: dict[str, Any], value_name: str, mask_name: str, default: float = 0.0) -> float:
    """Mean of `value_name` restricted to steps where `mask_name` > 0.5.

    Used to split ep_len_mean by *why* the episode ended (fall vs. timeout),
    since a single blended ep_len_mean can't distinguish "mostly instant
    deaths" from "mostly late stumbles right before the time limit" -- and
    those call for very different next actions.
    """
    if value_name not in metrics or mask_name not in metrics:
        return default
    mask = np.asarray(metrics[mask_name]) > 0.5
    vals = np.asarray(metrics[value_name])
    if mask.sum() <= 0:
        return default
    return float(vals[mask].mean())


def masked_mean_in_band(metrics: dict[str, Any], value_name: str, mask_name: str, band_name: str,
                         lo: float, hi: float, default: float = 0.0) -> float:
    """The core-holdout source: "core" is a fixed band on the actual
    sampled command, independent of curriculum difficulty and independent
    of which regime produced it. The reference band is SEED_VX_LO..
    SEED_VX_HI (the same constants _sample_task births the robot into) --
    the one capability that must never be lost regardless of how far any
    regime's own frontier has pushed. Filtering by the actual per-episode
    mean commanded vx means this keeps working correctly at any curriculum
    position, from any regime -- any episode that happens to land in the
    seed band counts, any episode outside it doesn't, regardless of label.
    """
    if value_name not in metrics or mask_name not in metrics or band_name not in metrics:
        return default
    band = np.asarray(metrics[band_name])
    mask = (np.asarray(metrics[mask_name]) > 0.5) & (band >= lo) & (band <= hi)
    vals = np.asarray(metrics[value_name])
    if mask.sum() <= 0:
        return default
    return float(vals[mask].mean())


def masked_mean_regime(metrics: dict[str, Any], value_name: str, regime_mask_name: str, default: float = -1.0) -> float:
    """Mean of an episode-outcome field (e.g. episode_fell,
    episode_length) restricted to done steps where `regime_mask_name`
    (regime_steady/regime_transition/regime_resilience, from environment.py's
    _metrics) was active for that env this step. Same masking discipline as
    masked_mean_in_band -- episode_done combined with a second filter --
    just a boolean regime mask instead of a numeric command band. Regime is
    assigned per-env at that env's own reset and held for the whole
    episode, so "was this env in regime X this step" and "was this env in
    regime X when its episode ended" are the same question -- no ambiguity
    from a regime that could change mid-episode."""
    if value_name not in metrics or regime_mask_name not in metrics or 'episode_done' not in metrics:
        return default
    done = np.asarray(metrics['episode_done']) > 0.5
    regime_mask = np.asarray(metrics[regime_mask_name]) > 0.5
    mask = done & regime_mask
    vals = np.asarray(metrics[value_name])
    if mask.sum() <= 0:
        return default
    return float(vals[mask].mean())


def _dynamic_tol(cmd: np.ndarray, abs_floor: float, rel_frac: float) -> np.ndarray:
    return np.maximum(abs_floor, rel_frac * np.abs(cmd))


def _settled_frontier_mask(metrics: dict[str, Any]) -> np.ndarray | None:
    """Stable command-hold steps suitable for frontier competence.

    Frontier expansion is intentionally judged only on the non-rehearsal
    STEADY cohort after an initial physical settling window. TRANSITION has its
    own acceleration frontier and RESILIENCE has its own push frontier; mixing
    those transients into an axis-boundary measurement would corrupt evidence.
    """
    needed = ('regime_steady_frontier', 'age_s')
    if any(k not in metrics for k in needed):
        return None
    steady = np.asarray(metrics['regime_steady_frontier']) > 0.5
    age = np.asarray(metrics['age_s']) >= FRONTIER_SETTLE_S
    return steady & age


def _precision_on_mask(actual: np.ndarray, cmd: np.ndarray, mask: np.ndarray,
                       abs_floor: float, rel_frac: float, default: float = -1.0) -> float:
    """Fraction of environments whose *window-mean* tracking error is in tolerance.

    A healthy stepping gait oscillates around its target, so instantaneous frames
    are not independent trials. Aggregate each environment first; compare its mean
    absolute error with its mean dynamic tolerance.
    """
    actual=np.asarray(actual); cmd=np.asarray(cmd); mask=np.asarray(mask,bool)
    if mask.sum() <= 0:
        return default
    tol=_dynamic_tol(cmd,abs_floor,rel_frac)
    if actual.ndim < 2:
        return float(np.mean(np.abs(actual-cmd)[mask]) <= np.mean(tol[mask]))
    counts=mask.sum(axis=0)
    valid=counts>0
    if not np.any(valid):
        return default
    err_sum=(np.abs(actual-cmd)*mask).sum(axis=0)
    tol_sum=(tol*mask).sum(axis=0)
    good=(err_sum/np.maximum(counts,1)) <= (tol_sum/np.maximum(counts,1))
    return float(good[valid].mean())


def _combo_precision_on_mask(vx,cvx,vy,cvy,yw,cyw,mask) -> float:
    mask=np.asarray(mask,bool)
    if mask.sum()<=0: return -1.0
    arrs=((vx,cvx,FRONTIER_VX_ABS_TOL,FRONTIER_VX_REL_TOL),
          (vy,cvy,FRONTIER_VY_ABS_TOL,FRONTIER_VY_REL_TOL),
          (yw,cyw,FRONTIER_YAW_ABS_TOL,FRONTIER_YAW_REL_TOL))
    if np.asarray(vx).ndim < 2:
        return float(all(np.mean(np.abs(np.asarray(a)-np.asarray(c))[mask]) <= np.mean(_dynamic_tol(np.asarray(c),af,rf)[mask]) for a,c,af,rf in arrs))
    counts=mask.sum(axis=0); valid=counts>0
    if not np.any(valid): return -1.0
    goods=[]
    for a,c,af,rf in arrs:
        a=np.asarray(a); c=np.asarray(c); tol=_dynamic_tol(c,af,rf)
        goods.append(((np.abs(a-c)*mask).sum(axis=0)/np.maximum(counts,1)) <= ((tol*mask).sum(axis=0)/np.maximum(counts,1)))
    return float((goods[0]&goods[1]&goods[2])[valid].mean())


def frontier_tracking_metrics(metrics: dict[str, Any], frontier_params: dict[str, float] | None,
                              combo_p: float = COMBO_P) -> dict[str, float]:
    """Settled edge-tracking diagnostics used by the physical frontier.

    The ordinary ``precision_*`` fields are settled whole-envelope fidelity.
    ``frontier_precision_*`` uses only samples in the outer 28% of the current
    axis/radius, which makes expansion evidence actually refer to the boundary.
    """
    needed = ('vx','vy','yaw_rate','cmd_vx','cmd_vy','cmd_yaw')
    if any(k not in metrics for k in needed):
        return {}
    base = _settled_frontier_mask(metrics)
    if base is None:
        return {}
    vx=np.asarray(metrics['vx']); vy=np.asarray(metrics['vy']); yw=np.asarray(metrics['yaw_rate'])
    cvx=np.asarray(metrics['cmd_vx']); cvy=np.asarray(metrics['cmd_vy']); cyw=np.asarray(metrics['cmd_yaw'])
    p = task_param_defaults()
    if frontier_params:
        p.update(frontier_params)
    hi=max(0.05, float(p['frontier_vx_hi']))
    lo=min(-0.02, float(p['frontier_vx_lo']))
    vym=max(0.02, abs(float(p['frontier_vy'])))
    ywm=max(0.04, abs(float(p['frontier_yaw'])))
    radius=max(0.20, abs(float(p['frontier_combo_radius'])))
    edge=FRONTIER_EDGE_FRAC

    out = {
        'precision_vx': _precision_on_mask(vx, cvx, base, FRONTIER_VX_ABS_TOL, FRONTIER_VX_REL_TOL),
        'precision_vy': _precision_on_mask(vy, cvy, base, FRONTIER_VY_ABS_TOL, FRONTIER_VY_REL_TOL),
        'precision_yaw': _precision_on_mask(yw, cyw, base, FRONTIER_YAW_ABS_TOL, FRONTIER_YAW_REL_TOL),
    }
    m_hi=base & (cvx >= edge*hi)
    m_lo=base & (cvx <= edge*lo)
    m_vy=base & (np.abs(cvy) >= edge*vym)
    m_yw=base & (np.abs(cyw) >= edge*ywm)
    out['frontier_precision_vx_forward'] = _precision_on_mask(vx, cvx, m_hi, FRONTIER_VX_ABS_TOL, FRONTIER_VX_REL_TOL)
    out['frontier_precision_vx_reverse'] = _precision_on_mask(vx, cvx, m_lo, FRONTIER_VX_ABS_TOL, FRONTIER_VX_REL_TOL)
    out['frontier_precision_vy'] = _precision_on_mask(vy, cvy, m_vy, FRONTIER_VY_ABS_TOL, FRONTIER_VY_REL_TOL)
    out['frontier_precision_yaw'] = _precision_on_mask(yw, cyw, m_yw, FRONTIER_YAW_ABS_TOL, FRONTIER_YAW_REL_TOL)
    out['frontier_edge_frac_vx_forward'] = float(m_hi.mean())
    out['frontier_edge_frac_vx_reverse'] = float(m_lo.mean())
    out['frontier_edge_frac_vy'] = float(m_vy.mean())
    out['frontier_edge_frac_yaw'] = float(m_yw.mean())

    # Mixed-command edge in normalized physical coordinates. Positive and
    # reverse vx use their own scales because the envelope is asymmetric.
    nx=np.where(cvx>=0.0, cvx/hi, cvx/abs(lo))
    ny=cvy/vym; nz=cyw/ywm
    pp=max(1.0, float(combo_p))
    norm=np.power(np.abs(nx)**pp + np.abs(ny)**pp + np.abs(nz)**pp, 1.0/pp)
    active=(np.abs(nx)>0.18).astype(np.int32)+(np.abs(ny)>0.18).astype(np.int32)+(np.abs(nz)>0.18).astype(np.int32)
    cmask=base & (active>=2) & (norm >= edge*radius)
    out['frontier_precision_combo']=_combo_precision_on_mask(vx,cvx,vy,cvy,yw,cyw,cmask)
    out['frontier_edge_frac_combo']=float(cmask.mean())

    # Acceleration competence is deliberately separate: score the ramp itself
    # with relaxed (2x) dynamic tolerances and average the three axis success
    # fractions rather than requiring all three to be perfect simultaneously.
    if 'transition_frac' in metrics:
        rmask=np.asarray(metrics['transition_frac'])>0.5
        if rmask.sum()>0:
            scores=[]
            for actual,cmd,af,rf in ((vx,cvx,FRONTIER_VX_ABS_TOL,FRONTIER_VX_REL_TOL),
                                     (vy,cvy,FRONTIER_VY_ABS_TOL,FRONTIER_VY_REL_TOL),
                                     (yw,cyw,FRONTIER_YAW_ABS_TOL,FRONTIER_YAW_REL_TOL)):
                scores.append(_precision_on_mask(actual,cmd,rmask,2.0*af,2.0*rf))
            out['transition_precision']=float(sum(scores)/len(scores))
        else:
            out['transition_precision']=-1.0
    return out


# Rollout provenance belongs with measured diagnostics, not training policy.
# Values refer to the exact task distribution and policy that made the rollout.
def augment_rollout_row(
    row: dict[str, Any], aux_now: dict[str, Any], *,
    rollout_params_for_metrics: dict[str, Any],
    rollout_lr_scale_for_metrics: float,
    norm_diag: dict[str, Any], latest_eval: dict[str, Any],
    obs_norm_mode: str,
) -> None:
    """Attach diagnostics with the exact rollout provenance.

    `row` describes the rollout collected by rollout_policy_state, under
    rollout_params_for_metrics/evaluated_obs_norm -- not the policy and
    curriculum that exist after the gradient/governor update. Keeping
    these aligned is what makes collapse forensics trustworthy.
    """
    for axis in ('steady', 'transition', 'resilience', 'posture'):
        cur = float(rollout_params_for_metrics.get(f'{axis}_difficulty', 0.0))
        born = float(row.get(f'birth_{axis}_d', -1.0))
        row[f'rollout_{axis}_d'] = cur
        row[f'birth_lag_{axis}'] = (cur - born) if born >= 0.0 else -1.0
    # What allocation generated the samples, distinct from the governor's
    # allocation for NEXT update and from measured per-step occupancy.
    for rn in ('steady', 'transition', 'resilience'):
        row[f'rollout_regime_frac_{rn}'] = float(rollout_params_for_metrics.get(f'regime_frac_{rn}', 0.0))
    row['rollout_athletic_rehearsal_frac'] = float(rollout_params_for_metrics.get('athletic_rehearsal_frac', 0.0))
    row['rollout_lr_scale'] = float(rollout_lr_scale_for_metrics)
    row['obs_norm_mode'] = obs_norm_mode
    row.update(norm_diag)
    row.update(latest_eval)
    scalar_aux = {
        'ppo_kl_stopped_frac': 'kl_stopped',
        'explained_variance': 'explained_variance',
        'exploration_std_mean': 'exploration_std_mean',
        'ppo_approx_kl_final': 'approx_kl_final',
        'ppo_action_clip_frac': 'action_clip_frac',
        'ppo_adv_tail_frac': 'adv_tail_frac',
        'ppo_grad_norm_policy_preclip': 'grad_norm_policy_preclip',
        'ppo_grad_norm_value_preclip': 'grad_norm_value_preclip',
        'ppo_update_norm_policy': 'update_norm_policy',
        'ppo_update_norm_value': 'update_norm_value',
    }
    for dst, src in scalar_aux.items():
        row[dst] = float(np.asarray(aux_now.get(src, 0.0)))
    for ji in range(ACTION_DIM):
        row[f'ppo_action_clip_j{ji}'] = float(np.asarray(aux_now.get(f'action_clip_j{ji}', 0.0)))
    for rn in ('steady', 'transition', 'resilience'):
        for key in ('adv_mean', 'adv_std', 'value_ev'):
            row[f'ppo_{key}_{rn}'] = float(np.asarray(aux_now.get(f'{key}_{rn}', 0.0)))


def summarize_rollout(metrics: dict[str, Any], frontier_params: dict[str, float] | None = None, combo_p: float = COMBO_P) -> dict[str, float]:
    """Report episode outcomes and physical evidence, separate from payment.

    Episodes are measured at their end, while gait, motion, power and
    occupancy are step-averaged. Missing episodes use explicit sentinels
    where zero would misleadingly imply perfect or catastrophic behaviour.
    """
    done = np.asarray(metrics.get('episode_done', 0.0)) > 0.5
    episodes = int(done.sum())
    row = {
        'episodes': float(episodes),
        'ep_len_mean': done_mean(metrics, 'episode_length', default=0.0),
        'ep_rew_mean': done_mean(metrics, 'episode_return', default=0.0),
        'fall_rate': done_mean(metrics, 'episode_fell', default=mean_metric(metrics, 'terminal')),
        'timeout_rate': done_mean(metrics, 'episode_timeout', default=0.0),
        'vx_mean': done_mean(metrics, 'episode_vx_mean', default=mean_metric(metrics, 'vx')),
        'cmd_vx_mean': done_mean(metrics, 'episode_cmd_vx_mean', default=mean_metric(metrics, 'cmd_vx')),
        'bad_contact': done_mean(metrics, 'episode_bad_contact', default=mean_metric(metrics, 'bad_contact')),
        'support': done_mean(metrics, 'episode_support', default=mean_metric(metrics, 'support')),
        'fall_height': mean_metric(metrics, 'fall_height'),
        'fall_tilt': mean_metric(metrics, 'fall_tilt'),
        'fall_roll_rate': mean_metric(metrics, 'fall_roll_rate'),
        'fall_pitch_rate': mean_metric(metrics, 'fall_pitch_rate'),
        'height': mean_metric(metrics, 'height'),
        'height_error_m': mean_metric(metrics, 'height_error'),
        'gravity_z': mean_metric(metrics, 'gravity_z'),
        'foot_mid_angle_deg': mean_metric(metrics, 'foot_mid_angle_deg'),
        'foot_mid_cone': mean_metric(metrics, 'foot_mid_cone'),
        'phase_sync': mean_metric(metrics, 'phase_sync'),
        'step_evidence': mean_metric(metrics, 'step_evidence'),
        'organic_step_shape': mean_metric(metrics, 'organic_step_shape'),
        'reward_valid_alive': mean_metric(metrics, 'reward_valid_alive'),
        'reward_command_tracking': mean_metric(metrics, 'reward_command_tracking'),
        'reward_upright_height': mean_metric(metrics, 'reward_upright_height'),
        'reward_feet_only_contact': mean_metric(metrics, 'reward_feet_only_contact'),
        'reward_swing_clearance': mean_metric(metrics, 'reward_swing_clearance'),
        # Read-only nominal servo mechanics: no effect on PPO or governance.
        'servo_shaft_abs_power_w': mean_metric(metrics, 'servo_shaft_abs_power_w'),
        'servo_shaft_drive_power_w': mean_metric(metrics, 'servo_shaft_drive_power_w'),
        'servo_shaft_brake_power_w': mean_metric(metrics, 'servo_shaft_brake_power_w'),
        'servo_over_rated_frac': mean_metric(metrics, 'servo_over_rated_frac'),
        'servo_peak_torque_limit_frac': mean_metric(metrics, 'servo_peak_torque_limit_frac'),
        'spring_potential_energy_j': mean_metric(metrics, 'spring_potential_energy_j'),
        'shin_spring_energy_j': mean_metric(metrics, 'shin_spring_energy_j'),
        # The reward ledger owns payment names. These remain teaching
        # payments; unlike measured progress or torque they are NOT exams.
        **{f'reward_weighted_{name}': mean_metric(metrics, f'reward_weighted_{name}')
           for name in (*PAYMENT_NAMES, FALL_PAYMENT_NAME, MOTOR_PAYMENT_NAME)},
        # Fall-conditioned vs. timeout-conditioned episode length.
        # default=-1.0 (not 0.0) means "no episodes of this kind ended this
        # rollout" so it's never confused with a real, alarming zero.
        'fail_ep_len_mean': masked_mean(metrics, 'episode_length', 'episode_fell', default=-1.0),
        'timeout_ep_len_mean': masked_mean(metrics, 'episode_length', 'episode_timeout', default=-1.0),
        # "core" is a fixed band on the actual commanded speed
        # (SEED_VX_LO..SEED_VX_HI), not a regime label -- see
        # masked_mean_in_band's own docstring for why.
        'core_fall_rate': masked_mean_in_band(metrics, 'episode_fell', 'episode_done', 'episode_cmd_vx_mean',
                                               SEED_VX_LO, SEED_VX_HI, default=-1.0),
        'core_ep_len_mean': masked_mean_in_band(metrics, 'episode_length', 'episode_done', 'episode_cmd_vx_mean',
                                                 SEED_VX_LO, SEED_VX_HI, default=-1.0),
        # Precision fields are attached below by
        # frontier_tracking_metrics(), which scores settled physical tracking
        # and separately measures the actual frontier edge.
        # Requested regime fractions are curriculum controller outputs; these
        # are the ACTUAL per-step occupancies PPO consumed this rollout.
        'occupancy_steady': mean_metric(metrics, 'regime_steady'),
        'occupancy_transition': mean_metric(metrics, 'regime_transition'),
        'occupancy_resilience': mean_metric(metrics, 'regime_resilience'),
        'occupancy_athletic_rehearsal': mean_metric(metrics, 'athletic_rehearsal'),
        'occupancy_field_taught': mean_metric(metrics, 'field_taught'),
        # Difficulty attached to each episode at birth. Because 700-step
        # episodes can span ~22 rollout=32 PPO updates, these may materially
        # lag the current governor position when the episode finally ends.
        'birth_steady_d': done_mean(metrics, 'birth_steady_d', default=-1.0),
        'birth_transition_d': done_mean(metrics, 'birth_transition_d', default=-1.0),
        'birth_resilience_d': done_mean(metrics, 'birth_resilience_d', default=-1.0),
        'birth_posture_d': done_mean(metrics, 'birth_posture_d', default=-1.0),
    }
    row.update(frontier_tracking_metrics(metrics, frontier_params, combo_p=combo_p))
    # Command-transition tracking (see environment.py's out_metrics).
    # transition_*_err = mean |vel - cmd| over the steps that were MID-RAMP;
    # holding_*_err = same over the steps that were not (named to avoid
    # colliding with the STEADY REGIME, a different concept -- this is
    # "not currently mid-ramp", not "in the steady regime"). -1.0 means
    # "no such steps this rollout" (e.g. min_hold_s still long so commands
    # almost never change) -- a sentinel, never a real zero-error reading.
    tf = mean_metric(metrics, 'transition_frac')
    row['transition_frac'] = tf
    for axis in ('vx', 'vy', 'yaw'):
        row[f'transition_{axis}_err'] = mean_metric(metrics, f'transition_{axis}_err') / tf if tf > 1e-4 else -1.0
        row[f'holding_{axis}_err'] = mean_metric(metrics, f'steady_{axis}_err') / (1.0 - tf) if (1.0 - tf) > 1e-4 else -1.0
    # Per-REGIME breakdown of the same fields the governor's competence
    # formula reads globally -- ep_len_mean/fall_rate/vx_mean/cmd_vx_mean
    # are done-masked AND regime-masked (via masked_mean_regime, same
    # discipline as core_fall_rate's done+band masking above); foot_mid_cone
    # is a plain per-step regime-masked mean, matching how the GLOBAL
    # foot_mid_cone field itself is computed (mean_metric, not done-masked).
    # -1.0 = no episodes of that regime ended this rollout, never a real zero.
    for regime in ('steady', 'transition', 'resilience'):
        rmask = f'regime_{regime}_frontier'
        row[f'fall_rate_{regime}'] = masked_mean_regime(metrics, 'episode_fell', rmask, default=-1.0)
        row[f'ep_len_mean_{regime}'] = masked_mean_regime(metrics, 'episode_length', rmask, default=-1.0)
        row[f'vx_mean_{regime}'] = masked_mean_regime(metrics, 'episode_vx_mean', rmask, default=-1.0)
        row[f'cmd_vx_mean_{regime}'] = masked_mean_regime(metrics, 'episode_cmd_vx_mean', rmask, default=-1.0)
        row[f'foot_mid_cone_{regime}'] = masked_mean(metrics, 'foot_mid_cone', rmask, default=0.0)
        # Raw counts, so RegimeGovernor can pool episodes across updates
        # (see its _pooled_fall_ema) instead of averaging per-rollout
        # ratios whose denominators can differ by 100x.
        if 'episode_done' in metrics and rmask in metrics:
            ended = (np.asarray(metrics['episode_done']) > 0.5) & (np.asarray(metrics[rmask]) > 0.5)
            row[f'episodes_{regime}'] = float(ended.sum())
            row[f'falls_{regime}'] = float((np.asarray(metrics['episode_fell'])[ended] > 0.5).sum()) if 'episode_fell' in metrics else 0.0
    return row


# Optional read-only posture drift analysis from previously saved metrics.
import argparse
import csv
import math
from pathlib import Path
from statistics import median

MIN_DROP_M = 0.015  # diagnostic threshold, not a proven failure boundary

def _number(row: dict, key: str) -> float | None:
    try:
        x = float(row.get(key, ''))
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None

def _mid(rows, field):
    numbers = [_number(r, field) for r in rows]
    valid = [v for v in numbers if v is not None]
    return median(valid) if valid else None

def summarize_posture(path: Path) -> dict | None:
    with path.open(newline='', encoding='utf-8-sig') as f:
        rows = [r for r in csv.DictReader(f) if _number(r, 'height') is not None]
    if not rows:
        return None
    n = len(rows)
    width = min(5, max(1, n // 3)) if n >= 3 else 1
    old, new = rows[:width], rows[-width:]
    a, b = _mid(old, 'height'), _mid(new, 'height')
    ea, eb = _mid(old, 'height_error_m'), _mid(new, 'height_error_m')
    tracking_delta = eb - ea if ea is not None and eb is not None else None
    return dict(
        path=str(path), rows=n, step=_number(rows[-1], 'step'),
        early_height_m=a, late_height_m=b, height_change_m=b - a,
        early_height_error_m=ea, late_height_error_m=eb,
        tracking_error_change_m=tracking_delta,
        warning=n >= 6 and (tracking_delta < -MIN_DROP_M if tracking_delta is not None
                             else b - a < -MIN_DROP_M),
    )

def posture_report(root: Path) -> int:
    root = root.expanduser()
    files = ([root] if root.is_file() and root.name == 'metrics.csv' else
             sorted(root.rglob('metrics.csv')) if root.is_dir() else [])
    if not files:
        print(f'No metrics.csv under {root}. Try ~/robot_lab/emu_runs or --root /path/to/run.')
        return 1
    found = 0
    for file in files:
        try:
            item = summarize_posture(file)
        except (OSError, UnicodeError, csv.Error) as exc:
            print(f'Could not read {file}: {exc}')
            continue
        if item is None:
            continue
        found += 1
        step = f"{int(item['step']):,}" if item['step'] is not None else '?'
        print(f"\n{file}\n  step {step}; {item['rows']} rows")
        print(f"  torso height: {item['early_height_m']*1000:.1f} → "
              f"{item['late_height_m']*1000:.1f} mm "
              f"(Δ {item['height_change_m']*1000:+.1f} mm)")
        d = item['tracking_error_change_m']
        if d is None:
            print('  target tracking unavailable (v5014); absolute height cannot establish cause')
        else:
            print(f"  measured minus commanded: {item['early_height_error_m']*1000:+.1f} → "
                  f"{item['late_height_error_m']*1000:+.1f} mm (Δ {d*1000:+.1f} mm)")
        if item['warning']:
            print('  CHECK: downward drift >=15mm. Examine fall rate and command distribution.')
    if found == 0:
        print('No finite height readings found.')
        return 1
    print('\nRead-only trend screening; lower height does not prove reward exploitation.')
    return 0

def posture_main(argv=None):
    p = argparse.ArgumentParser(description='Read-only posture-height review for old EMU training runs')
    p.add_argument('--root', type=Path, default=Path.home() / 'robot_lab' / 'emu_runs')
    return posture_report(p.parse_args(argv).root)


if __name__ == '__main__':
    raise SystemExit(main())
