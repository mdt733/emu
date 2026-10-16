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

if __name__ == '__main__':
    raise SystemExit(main())
