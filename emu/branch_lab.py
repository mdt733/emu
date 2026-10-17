"""Checkpoint-safe v5015 successive-halving experiments: manual or autopilot.

One 20M foundation, four 80M forks (+60M each), optional two 100M finalists, one 120M follow-up.
All raw checkpoints remain outside Git; never touches the public 60M brain.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from .checkpoint_io import checkpoint_step, git_provenance, resolve_target_checkpoint, run_training
from .contracts import source_repo_root
from .training_config import _phase_config, verify_existing_phase_config
from .training_config import apply_experiment_knobs


# Human-editable experiment recipe: the complete study lives beside its executor.
# All step targets are ABSOLUTE environment steps in a single policy lineage.
# Round1 adds 60M to the common 20M foundation: each fork reaches 80M total.
FOUNDATION = 20_000_000
ROUNDS = {
    'round1': 80_000_000,  # four comparable +60M continuations from 20M
    'round2': 100_000_000, # optional: +20M to two eligible finalists
    'round3': 120_000_000, # optional: +20M to best eligible finalist
}
# A succeeding round continues this many eligible policies from the last
# common frozen-policy exam. Baseline/forks/finallists share ONE recipe here.
PROMOTION_SLOTS = {'round2': 2, 'round3': 1}
SCREEN_PITCH_DEG = 15.0  # common evaluation task, not a training command


@dataclass(frozen=True)
class Arm:
    """One hypothesis: changes to named, validated TrainingConfig knobs only.

    Each (field, value) pair becomes an explicit field override to the common
    mature baseline. No PPO-engine edits or new CLI switches are needed.
    """
    name: str
    hypothesis: str
    knobs: tuple[tuple[str, bool | int | float | str | None], ...] = ()


# Each challenger differs from the baseline in one understandable dimension.
# All knobs are existing fields of emu/training_config.py::TrainingConfig.
ARMS = (
    Arm('baseline', '10% graph teaching, level-floor control'),
    Arm('shadow', 'No capability-field teaching', (
        ('capability_field_mode', 'shadow'),
        ('field_max_influence', 0.0),
    )),
    Arm('slope', 'Gentle gravity-equivalent slope practice', (
        ('domain_randomization', True),
        ('domain_terrain_radius', 0.12),
    )),
    Arm('pitch15', '15 degree chassis command ceiling', (
        ('pitch_command_max_deg', 15.0),
    )),
)
BY_NAME = {arm.name: arm for arm in ARMS}


def plan_dict(seed=733, envs=2048):
    """Human-readable hypothesis/budget plan, derived from this recipe only.

    The executor imports this function but does not define study budgets.
    A complete round costs (target - previous target) per continuing arm.
    """
    round_names = list(ROUNDS)
    prior_target = FOUNDATION
    nominal_steps = FOUNDATION
    for stage in round_names:
        contenders = len(ARMS) if stage == 'round1' else PROMOTION_SLOTS[stage]
        nominal_steps += contenders * (ROUNDS[stage] - prior_target)
        prior_target = ROUNDS[stage]
    first_increment = len(ARMS) * (ROUNDS['round1'] - FOUNDATION)
    return {'version': 'v5015-lab1', 'seed': seed, 'envs': envs,
            'foundation': FOUNDATION, 'rounds': ROUNDS,
            'arms': [asdict(a) for a in ARMS],
            'nominal_steps': nominal_steps,
            'review_points': ['after common 20M foundation', 'after four 80M branches (+60M each)',
                              'optional: two 100M finalists', 'optional: one 120M finalist'],
            'recommended': 'emu ladder --stage foundation, inspect, then emu ladder --stage auto --stop-after round1',
            'round1_incremental_steps': first_increment,
            'through_round1_total_steps': FOUNDATION + first_increment,
            'automatic_promotion': 'eligible-only, score-ranked in --stage auto; manual by default',
            'auto_stop_if_ineligible': True}



def _read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    temporary.replace(path)


def _hash(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _context(root, seed, envs, provenance):
    repo = source_repo_root()
    if repo is not None and (root == repo or repo in root.parents):
        raise RuntimeError('Raw experiment runs must live outside the Git checkout')
    # An unknown Git identity is not a valid reproducible campaign source.
    # The single-run launcher already fails closed here; the ladder must too.
    if not provenance.get('available') or not provenance.get('commit'):
        raise RuntimeError('Ladder requires an identifiable clean Git source commit')
    if provenance.get('dirty'):
        raise RuntimeError('Git checkout has uncommitted changes; freeze a clean source commit before training')
    wanted = {'version': 'v5015-lab1', 'seed': seed, 'envs': envs,
              'git_commit': provenance.get('commit', '')}
    path = root / 'lab_context.json'
    if path.exists():
        if _read(path) != wanted:
            raise RuntimeError('Source commit, seed or environment count changed. Use pinned source or a NEW --root.')
    else:
        _save(path, wanted)


def config(run, *, arm, target, seed, envs, device, resume):
    """Pure recipe -> training configuration. No GPU, filesystem or PPO side effects."""
    # Foundation is common to every hypothesis. No arm knobs apply before fork.
    if arm is None:
        return _phase_config(run.name, FOUNDATION, 0, 0, 0, root=run.parent,
                             seed=seed, device=device, envs=envs, resume=resume,
                             profile='v5015', pitch_deg=8.0)
    baseline = _phase_config(run.name, target, 0, 0, 0, root=run.parent,
                             seed=seed, device=device, envs=envs, resume=resume,
                             profile='v5015', pitch_deg=8.0)
    return apply_experiment_knobs(baseline, arm.knobs)


def validate_recipe(*, seed: int, envs: int, device: str, root: Path) -> None:
    """Fail on bad ladder definitions BEFORE starting the shared foundation."""
    import re

    if envs < 1:
        raise ValueError('Experiment envs must be positive')
    if list(ROUNDS) != ['round1', 'round2', 'round3']:
        raise ValueError('Ladder requires ordered round1/round2/round3 stages')
    if list(PROMOTION_SLOTS) != ['round2', 'round3'] or not (
            len(ARMS) >= PROMOTION_SLOTS['round2'] > PROMOTION_SLOTS['round3'] > 0):
        raise ValueError('Ladder promotion sizes must form a strict smaller finalist pool')
    targets = [FOUNDATION, *ROUNDS.values()]
    if any(type(v) is not int or v <= 0 for v in targets) or targets != sorted(set(targets)):
        raise ValueError('Ladder targets must be strictly increasing integer environment steps')
    if len(ARMS) < 2 or len(BY_NAME) != len(ARMS):
        raise ValueError('Ladder needs at least two uniquely named experiment arms')
    if ARMS[0].name != 'baseline' or ARMS[0].knobs:
        raise ValueError('First arm must be an unmodified baseline')
    for arm in ARMS:
        if not re.fullmatch(r'[a-z][a-z0-9_]*', arm.name):
            raise ValueError(f'Unsafe ladder arm name: {arm.name}')
        # Prove every arm maps to a typed TrainingConfig. This is pure:
        # no JAX compile, file writes, subprocesses, or policy mutation.
        config(root/'round1'/arm.name, arm=arm, target=ROUNDS['round1'],
               seed=seed, envs=envs, device=device, resume=None)


def completed(root, stage, name, envs):
    if stage == 'foundation':
        run, target = root / 'foundation20', FOUNDATION
    else:
        run, target = root / stage / name, ROUNDS[stage]
    complete, partial, expected = resolve_target_checkpoint(
        run, target_steps=target, envs=envs, label=str(run))
    if complete is None or checkpoint_step(complete) != expected:
        raise RuntimeError(f'Missing valid completed {stage} checkpoint at {run}; expected {expected:,} steps')
    return complete


def ensure_run(run, *, arm, parent, target, seed, envs, device, provenance):
    ancestor = ({'source': str(parent.resolve()), 'step': checkpoint_step(parent),
                 'sha256': _hash(parent)}
                if parent is not None else {'source': 'new foundation', 'step': 0, 'sha256': None})
    spec = {'arm': asdict(arm) if arm is not None else None,
            'parent': ancestor, 'target': target, 'seed': seed, 'envs': envs}
    spec_file = run / 'lab_spec.json'
    if spec_file.exists():
        if _read(spec_file) != spec:
            raise RuntimeError(f'Resumption would change checkpoint parent or hypothesis for {run}')
    else:
        _save(spec_file, spec)
    expected_config = config(run, arm=arm, target=target, seed=seed,
                             envs=envs, device=device, resume=None)
    # A wrong phase name used to launch a complete 20M foundation into
    # 00_foundation/ while ensure_run inspected foundation20/. Check BEFORE
    # launching the trainer so a misplaced checkpoint can never happen again.
    configured_dir = Path(expected_config.run_dir).resolve()
    if configured_dir != run.resolve():
        raise RuntimeError(
            f'Runner and trainer disagree on output folder: '
            f'expected {run.resolve()}, got {configured_dir}')
    verify_existing_phase_config(run, expected_config)
    final, partial, expected = resolve_target_checkpoint(
        run, target_steps=target, envs=envs, label=run.name)
    if final is not None:
        print(f'{run.name}: validated checkpoint already complete at {expected:,}')
        return final
    resume = partial or parent
    print(f'{run.name}: {checkpoint_step(resume) if resume else 0:,} -> {expected:,} steps', flush=True)
    run_training(config(run, arm=arm, target=target, seed=seed, envs=envs,
                        device=device, resume=resume), provenance=provenance)
    final, partial, expected = resolve_target_checkpoint(
        run, target_steps=target, envs=envs, label=run.name)
    if final is None:
        raise RuntimeError(f'{run.name} did not reach {expected:,}; resumable checkpoint: {partial}')
    return final


def _selection(root, stage, selected, previous, envs):
    amount = PROMOTION_SLOTS[stage]
    if len(selected) != amount or len(set(selected)) != amount or any(x not in BY_NAME for x in selected):
        raise ValueError(f'{stage} requires exactly {amount} different valid --select names')
    file = root / f'{previous}_scoreboard.json'
    if not file.is_file():
        raise RuntimeError(f'Score {previous} first; missing {file}')
    board = _read(file)
    if board['target'] != ROUNDS[previous]:
        raise RuntimeError('Stale scorecard target')
    scored = {r['arm']: r for r in board['results']}
    for name in selected:
        if name not in scored or not scored[name]['eligible']:
            raise RuntimeError(f'Arm {name} has no eligible held-out score for {previous}')
        ck = completed(root, previous, name, envs)
        if scored[name].get('checkpoint_sha256') != _hash(ck):
            raise RuntimeError(f'Checkpoint for {name} differs from the one evaluated; rescore {previous}')
    if stage == 'round3':
        if selected[0] not in _read(root / 'round2_selection.json')['selected']:
            raise RuntimeError('Round3 must continue one of the round2 finalists')
    selection = {'selected': sorted(selected), 'from_round': previous}
    locked = root / f'{stage}_selection.json'
    if locked.exists():
        if _read(locked) != selection:
            raise RuntimeError('Selection has already been locked; use a new root to revise an experiment')
    else:
        _save(locked, selection)


def _number(x):
    try:
        n = float(x)
        return n if math.isfinite(n) else None
    except (ValueError, TypeError):
        return None


def score_row(name, source):
    """No automatic winner. Missing data and excessive crouching fail closed."""
    panels = ['resilience100', 'robust70', 'growth140', 'growthcombo140']
    surv = [_number(source.get(f'screen_{p}_survival')) for p in panels]
    pitch = [_number(source.get(f'screen_{p}_pitch_mae')) for p in panels]
    height = [_number(source.get(f'screen_{p}_height_shortfall_m')) for p in panels]
    motion = [_number(source.get(f'screen_{p}_{axis}_mae'))
              for p in panels for axis in ('vx', 'vy', 'yaw')]
    steps = [_number(source.get(f'screen_{p}_step_evidence')) for p in panels]
    support = [_number(source.get(f'screen_{p}_support')) for p in panels]
    slope = _number(source.get('screen_gentle_slope_survival'))
    slope_height = _number(source.get('screen_gentle_slope_height_shortfall_m'))
    if slope is None or slope_height is None or any(n is None for n in surv + pitch + height + motion + steps + support):
        return {'arm': name, 'eligible': False, 'score': None,
                'reasons': ['missing/nonfinite held-out metrics']}
    hard = sum(surv[:2])/2
    growth = sum(surv[2:])/2
    pe = sum(pitch)/4
    hs = sum(height)/4
    me = sum(motion)/12
    step_evidence = sum(steps)/len(steps)
    mean_support = sum(support)/len(support)
    reasons = []
    if hard < .20: reasons.append('hard survival below 20%')
    if hs > .035: reasons.append('height shortfall exceeds 35mm')
    if pe > .45: reasons.append('pitch tracking error exceeds .45rad')
    if slope_height > .045: reasons.append('gentle slope height shortfall exceeds 45mm')
    score = (.43*hard + .26*growth + .13*slope - .08*min(1, pe/.45)
             - .07*min(1, hs/.035) - .03*min(1, me/.5))
    return {'arm': name, 'eligible': not reasons, 'reasons': reasons,
            'score': round(score, 6), 'hard_survival': round(hard, 5),
            'growth_survival': round(growth, 5), 'gentle_slope_survival': round(slope, 5),
            'pitch_mae_rad': round(pe, 5),
            'height_shortfall_m': round(hs, 5), 'motion_mae': round(me, 5),
            'step_evidence': round(step_evidence, 5), 'support': round(mean_support, 5)}


def score(root, round_name, *, envs, screen_envs, screen_horizon, device):
    if round_name not in ROUNDS:
        raise ValueError('Invalid scoring round')
    if round_name == 'round1':
        names = list(BY_NAME)
    else:
        names = _read(root / f'{round_name}_selection.json')['selected']
    run_dirs = []
    for name in names:
        completed(root, round_name, name, envs)
        run_dirs.append(root / round_name / name)
    # All candidates are judged in the SAME deterministic v5015 environment:
    # independent of their own teacher, domain severity or training pitch limit.
    from .evaluation import evaluate_runs
    raw = evaluate_runs(run_dirs, envs=screen_envs, horizon=screen_horizon,
                        device=device, eval_pitch_deg=SCREEN_PITCH_DEG)
    rows = [score_row(name, raw[name]) for name in names]
    for row in rows:
        ck = root / round_name / row['arm'] / 'checkpoints' / 'final.pkl'
        row['checkpoint_sha256'] = _hash(ck)
    if round_name == 'round1' and rows[0]['score'] is not None:
        baseline_hard = rows[0]['hard_survival']
        for item in rows[1:]:
            if item['score'] is not None and item['hard_survival'] < baseline_hard - .08:
                item['reasons'].append('hard survival >8 points below baseline')
                item['eligible'] = False
    rows.sort(key=lambda r: (r['eligible'], r['score'] if r['score'] is not None else -999),
              reverse=True)
    result = {'round': round_name, 'target': ROUNDS[round_name],
              'screen_pitch_deg': SCREEN_PITCH_DEG, 'screen_envs': screen_envs,
              'screen_horizon': screen_horizon, 'results': rows}
    _save(root / f'{round_name}_scoreboard.json', result)
    # Reporting formats the examined evidence; ranking stays above.
    from .results import write_scorecard, write_report
    report = write_scorecard(root, result)
    for row in rows:
        print(f"{row['arm']:<10} eligible={row['eligible']} score={row['score']}, {row['reasons']}")
    # Package the existing held-out score and all run histories for human review.
    # This reads checkpoints ONLY to hash their identity; no policy load or PPO.
    try:
        for path in write_report(root):
            print(f'Campaign report: {path}')
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f'WARNING: offline results export failed ({exc}). '
              f'Retry safely with: emu report --root {root}')
    print(f'Saved {report}. Auto mode promotes eligible contenders; manual review is also supported.')


def promotion_choices(board: dict, slots: int) -> list[str]:
    """Choose only qualified arms, ordered deterministically by held-out score.

    No eligibility bypass. With insufficient qualified contenders the autopilot
    stops, preserving every checkpoint for inspection/manual intervention.
    """
    if slots not in set(PROMOTION_SLOTS.values()):
        raise ValueError('Promotion supports 1 or 2 places only')
    ordered = sorted((row for row in board['results']
                      if row.get('arm') in BY_NAME and row.get('eligible') is True
                      and _number(row.get('score')) is not None),
                     key=lambda row: (-float(row['score']),
                                      list(BY_NAME).index(row['arm'])))
    chosen = [row['arm'] for row in ordered[:slots]]
    if len(chosen) != slots:
        raise RuntimeError(
            f"Only {len(chosen)} of {slots} finalists are eligible in "
            f"{board.get('round', 'unknown round')}. Stopping safely; read "
            f"the scorecard and examine the checkpoint evidence.")
    return chosen


def score_is_current(root: Path, round_name: str, *, envs: int,
                     screen_envs: int, screen_horizon: int) -> bool:
    """Avoid repeated GPU screens after an interrupted/restarted autopilot.

    Fingerprints, cohort size, horizon, and report must all match.
    """
    path = root / f'{round_name}_scoreboard.json'
    if not path.is_file() or not (root / f'{round_name}_scorecard.md').is_file():
        return False
    board = _read(path)
    if (board.get('round') != round_name
        or board.get('target') != ROUNDS[round_name]
        or board.get('screen_pitch_deg') != SCREEN_PITCH_DEG
        or board.get('screen_envs') != screen_envs
        or board.get('screen_horizon') != screen_horizon):
        return False
    expected_names = set(BY_NAME) if round_name == 'round1' else set(
        _read(root / f'{round_name}_selection.json')['selected'])
    rows = board.get('results', [])
    if len(rows) != len(expected_names) or {r.get('arm') for r in rows} != expected_names:
        return False
    for row in rows:
        try:
            ck = completed(root, round_name, row['arm'], envs)
        except RuntimeError:
            return False
        if _hash(ck) != row.get('checkpoint_sha256'):
            return False
    return True


def autopilot(root: Path, *, seed: int, envs: int, device: str,
              screen_envs: int, screen_horizon: int,
              provenance: dict, stop_after: str = 'round3') -> None:
    """Sequential stages; each fork restarts from one fixed validated ancestor.

    Pause on failed training or a failed/ambiguous evaluation. Never promote an
    ineligible policy and never alter the published champion.
    """
    print(f'[EMU ladder] Auto mode: 20M foundation, {len(ARMS)}x80M forks '
          '(+60M each); optional 2x100M then 1x120M.', flush=True)
    ensure_run(root/'foundation20', arm=None, parent=None, target=FOUNDATION,
               seed=seed, envs=envs, device=device, provenance=provenance)
    for round_name in ROUNDS:
        if round_name == 'round1':
            names = list(BY_NAME)
            prior = 'foundation'
        else:
            prior = 'round1' if round_name == 'round2' else 'round2'
            selection_file = root / f'{round_name}_selection.json'
            if selection_file.exists():
                names = _read(selection_file)['selected']
                print(f'[EMU lab5015] Reusing locked {round_name} '
                      f'selection: {", ".join(names)}', flush=True)
            else:
                scored = _read(root / f'{prior}_scoreboard.json')
                slots = PROMOTION_SLOTS[round_name]
                names = promotion_choices(scored, slots)
                print(f'[EMU lab5015] Auto-selected from {prior}: '
                      f'{", ".join(names)}', flush=True)
            _selection(root, round_name, names, prior, envs)
        for name in names:
            parent = completed(root, prior, name if prior != 'foundation' else None, envs)
            ensure_run(root/round_name/name, arm=BY_NAME[name], parent=parent,
                       target=ROUNDS[round_name], seed=seed, envs=envs,
                       device=device, provenance=provenance)
        if score_is_current(root, round_name, envs=envs, screen_envs=screen_envs,
                            screen_horizon=screen_horizon):
            print(f'[EMU lab5015] Reusing validated {round_name} '
                  'scoreboard, not rerunning GPU evaluation', flush=True)
        else:
            print(f'[EMU lab5015] Running common held-out {round_name} '
                  'evaluation...', flush=True)
            score(root, round_name, envs=envs, screen_envs=screen_envs,
                  screen_horizon=screen_horizon, device=device)
        print(f'[EMU lab5015] Completed {round_name} at '
              f'{ROUNDS[round_name]:,} policy steps. '
              f'Scorecard: {root / (round_name + "_scorecard.md")}', flush=True)
        if stop_after == round_name:
            print(f'[EMU lab5015] Finished at requested {stop_after} boundary.', flush=True)
            break


def main(argv=None):
    parser = argparse.ArgumentParser(description='EMU v5015: review-gated common-ancestor experiment lab')
    parser.add_argument('--stage', choices=['plan','foundation','round1','round2','round3','score','auto'],
                        default='plan')
    parser.add_argument('--round', choices=list(ROUNDS), default='round1')
    parser.add_argument('--stop-after', choices=list(ROUNDS), default='round3',
                        help='when --stage auto, stop after this round and its score')
    parser.add_argument('--root', type=Path, default=Path.home()/'robot_lab'/'emu_runs'/'v5015_lab1')
    parser.add_argument('--seed', type=int, default=733)
    parser.add_argument('--envs', type=int, default=2048)
    parser.add_argument('--device', choices=['auto','cpu','cuda','gpu'], default='cuda')
    parser.add_argument('--screen-envs', type=int, default=256)
    parser.add_argument('--screen-horizon', type=int, default=350)
    parser.add_argument('--arm', action='append', default=[], help='optional first-round arm, repeatable')
    parser.add_argument('--select', nargs='+', default=[], help='manually promote 2 or 1 arms')
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    validate_recipe(seed=args.seed, envs=args.envs, device=args.device, root=root)
    if args.stage == 'plan':
        print(json.dumps(plan_dict(args.seed,args.envs),indent=2))
        return 0
    if args.screen_envs < 1 or args.screen_horizon < 1:
        parser.error('screen env count and horizon must be positive')
    if args.stage == 'score':
        if args.arm or args.select: parser.error('score accepts no --arm or --select')
        context = root / 'lab_context.json'
        if not context.is_file():
            raise RuntimeError('No experiment context; complete a foundation first')
        saved = _read(context)
        if saved['envs'] != args.envs or saved['seed'] != args.seed:
            raise RuntimeError('Scoring must use the experiment original env count and seed')
        provenance = git_provenance()
        if saved['git_commit'] != provenance.get('commit', '') or provenance.get('dirty'):
            raise RuntimeError('Scoring source differs from pinned experiment source')
        score(root,args.round,envs=args.envs,screen_envs=args.screen_envs,
              screen_horizon=args.screen_horizon,device=args.device)
        return 0
    if args.stage != 'round1' and args.arm:
        parser.error('--arm is only valid for round1')
    if args.stage == 'auto' and args.select:
        parser.error('autopilot ranks eligible candidates; no --select permitted')
    if args.stage in ('round1','foundation') and args.select:
        parser.error('--select only applies to round2 and round3')
    provenance = git_provenance()
    _context(root, args.seed, args.envs, provenance)
    if args.stage == 'auto':
        autopilot(root, seed=args.seed, envs=args.envs, device=args.device,
                  screen_envs=args.screen_envs, screen_horizon=args.screen_horizon,
                  provenance=provenance, stop_after=args.stop_after)
        return 0
    if args.stage == 'foundation':
        ensure_run(root/'foundation20',arm=None,parent=None,target=FOUNDATION,
                   seed=args.seed,envs=args.envs,device=args.device,provenance=provenance)
        from .results import write_report
        write_report(root)
        return 0
    if args.stage == 'round1':
        names = args.arm or list(BY_NAME)
        if len(names) != len(set(names)) or any(n not in BY_NAME for n in names):
            parser.error('unknown or duplicate arm')
        parent_stage = 'foundation'
    else:
        names = args.select
        prior = 'round1' if args.stage == 'round2' else 'round2'
        _selection(root,args.stage,names,prior,args.envs)
        parent_stage = prior
    for name in names:
        ancestor = completed(root,parent_stage, name if parent_stage!='foundation' else None,args.envs)
        ensure_run(root/args.stage/name,arm=BY_NAME[name],parent=ancestor,
                   target=ROUNDS[args.stage],seed=args.seed,envs=args.envs,
                   device=args.device,provenance=provenance)
    return 0



# One-run profile is a small specialised path through the same experiment driver.
import json
from dataclasses import asdict, replace
from pathlib import Path

from .training_config import verify_existing_phase_config
from .contracts import source_repo_root
from .checkpoint_io import (git_provenance, resolve_target_checkpoint,
                          run_training, update_boundary_at_or_after)
from .training_config import TrainingConfig, foundation_config, mature_config


def one_run_config(args) -> TrainingConfig:
    """Resolve a named, existing training profile without inventing PPO flags."""
    if args.envs <= 0 or args.steps <= 0:
        raise ValueError('--envs and --steps must both be positive')
    run_dir = Path(args.run_dir).expanduser().resolve()
    if args.profile == 'foundation':
        cfg = foundation_config(run_dir=run_dir, seed=args.seed,
                                device=args.device, envs=args.envs)
    elif args.profile in ('mature', 'sim2real'):
        # v5016 reuses the stable learning engine: pitch is a SHARED task
        # distribution, not a pitch-only experiment arm. Future trials can
        # vary acquisition, motor cost and terrain with this range fixed.
        cfg = mature_config('graph', run_dir=run_dir, steps=args.steps,
                            seed=args.seed, device=args.device, envs=args.envs,
                            resume=None)
    else:
        raise ValueError(f'unknown profile: {args.profile}')
    requested_pitch=getattr(args, 'pitch_deg', None)
    pitch_deg=float((30.0 if args.profile == 'sim2real' else 8.0)
                    if requested_pitch is None else requested_pitch)
    if not math.isfinite(pitch_deg) or not 0.0 < pitch_deg <= 30.0:
        raise ValueError('--pitch-deg must be within (0, 30] degrees')
    return replace(cfg, steps=args.steps, attitude_profile='v5015',
                   pitch_command_max_deg=pitch_deg)


def inspect_run(config: TrainingConfig, *, provenance: dict, resume: str) -> tuple[Path | None, bool]:
    """Fail closed on incompatible runs, unclean source, and unsafe paths.

    Returns (validated checkpoint to resume from, already complete?).
    A resume from another experiment does not rewrite that ancestor.
    """
    root = Path(config.run_dir).resolve()
    repo = source_repo_root()
    if repo and (root == repo.resolve() or repo.resolve() in root.parents):
        raise RuntimeError('Choose --run-dir OUTSIDE the Git checkout; raw training outputs are not source files')
    if root == Path.home().resolve():
        raise RuntimeError('Do not use your home directory directly as a run directory')
    if not provenance.get('available') or not provenance.get('commit') or provenance.get('dirty'):
        raise RuntimeError('A clean Git source commit is required for reproducible training')

    cfg_path = root / 'training_config.json'
    meta_path = root / 'run_meta.json'
    if root.exists() and any(root.iterdir()) and not cfg_path.is_file():
        raise RuntimeError(f'{root} contains files but no training_config.json; refusing to overwrite them')
    if cfg_path.is_file():
        previous = asdict(TrainingConfig.load(cfg_path))
        wanted = asdict(config)
        for variable in ('resume', 'device'):
            previous.pop(variable, None)
            wanted.pop(variable, None)
        if previous != wanted:
            raise RuntimeError('Existing training configuration differs; choose a fresh --run-dir')
        if not meta_path.is_file():
            # run_support writes config and console.log *before* the trainer
            # checks CUDA. A driver/setup failure leaves these two harmless
            # files but no policy state. Allow retry only in that precise case.
            unexpected = {p.name for p in root.iterdir()} - {
                'training_config.json', 'console.log',
            }
            if unexpected:
                raise RuntimeError(
                    'Saved config has no run_meta.json but other run contents exist: '
                    f'{sorted(unexpected)}; refusing ambiguous lineage')
        else:
            metadata = json.loads(meta_path.read_text(encoding='utf-8'))
            ancestor = metadata.get('source_provenance', {}).get('git_commit')
            if ancestor != provenance['commit']:
                raise RuntimeError('Source commit differs from prior run; use the matching source or a new --run-dir')
        verify_existing_phase_config(root, config)

    finished, partial, expected = resolve_target_checkpoint(
        root, target_steps=config.steps, envs=config.envs, label=root.name)
    if finished is not None:
        if resume not in ('', 'auto'):
            raise RuntimeError('Run already complete; external --resume is not appropriate here')
        return finished, True

    if partial is not None:
        if resume != 'auto':
            raise RuntimeError('Existing partial checkpoint detected: use --resume auto, or a new --run-dir')
        return partial, False
    if resume == 'auto':
        raise RuntimeError('--resume auto requested but no valid checkpoint exists in this run directory')
    if resume:
        from .checkpoint_io import load_checkpoint
        parent = Path(resume).expanduser().resolve()
        if not parent.is_file():
            raise FileNotFoundError(f'Parent checkpoint not found: {parent}')
        payload = load_checkpoint(parent, validate=True)
        source = payload['meta'].get('source_provenance', {}).get('git_commit')
        if source != provenance['commit']:
            raise RuntimeError('Parent checkpoint source commit differs; refuse cross-source warm start')
        parent_step = int(payload['meta']['step'])
        if parent_step >= expected:
            raise RuntimeError(f'Parent step {parent_step:,} is at/beyond target {expected:,}')
        return parent, False
    return None, False


def train_one(args) -> int:
    """A small CLI wrapper; the existing training module performs the updates."""
    config = one_run_config(args)
    target = update_boundary_at_or_after(config.steps, config.envs)
    print(f'One-run EMU: {args.profile} | {config.envs} worlds | '
          f'{config.steps:,} requested steps (PPO boundary {target:,})')
    print(f'Run directory: {config.run_dir}')
    if args.plan:
        print(json.dumps(asdict(config), sort_keys=True, indent=2))
        print('Planning only; no checkpoint opened or training launched.')
        return 0
    provenance = git_provenance()
    parent, completed = inspect_run(config, provenance=provenance, resume=args.resume)
    if completed:
        print(f'Already complete: validated {parent} (no PPO update).')
        return 0
    actual = replace(config, resume=str(parent) if parent is not None else '')
    print(f'Training {parent or "from scratch"} -> {target:,} steps on source {provenance["commit"][:12]}')
    run_training(actual, provenance=provenance)
    # Do not announce success until the trainer's own validated final is present.
    final, _, _ = resolve_target_checkpoint(
        Path(config.run_dir), target_steps=config.steps, envs=config.envs,
        label=Path(config.run_dir).name)
    if final is None:
        raise RuntimeError('Trainer returned but no validated final checkpoint reached the target')
    print(f'Completed validated run: {final}')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
