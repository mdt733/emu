#!/usr/bin/env python3
"""EMU v5007 repaired-frontier experiment runner.

v5007 is deliberately narrower than v5006 in *questions* but not in useful
compute.  The generation fixes the broken command-frontier measurement, makes
no-software-slew the actuator baseline, preserves the proven v5005 athletic
command distribution, strengthens fixed evaluation, and then tests the few
remaining causal levers with enough branches/seeds to distinguish real effects
from one-seed PPO wandering.

Typical use:
    python run_ladder.py --list
    python run_ladder.py --device cuda
    python run_ladder.py --only slew_static,noslew_static,frontier_repaired --skip-confirm --skip-phase2

Default bootstrap behaviour:
    1. train one shared foundation policy from scratch to 20M steps;
    2. use that exact checkpoint as the warm start for every phase-1 branch;
    3. confirm the top families on a second seed;
    4. continue the selected family in phase 2.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAINING_PY = HERE / 'training.py'
VERSION_FILE = HERE / 'VERSION'


def read_version() -> str:
    try:
        value = VERSION_FILE.read_text(encoding='utf-8').strip()
    except OSError as exc:
        raise SystemExit(f'Missing VERSION file: {VERSION_FILE}') from exc
    if not re.fullmatch(r'v[0-9]+', value):
        raise SystemExit(f'Invalid VERSION value {value!r}; expected e.g. v5007')
    return value


VERSION = read_version()
RAW_RUN_BASE = Path(os.environ.get('EMU_RUN_ROOT', str(HERE.parent / 'emu_runs'))).expanduser()
DEFAULT_RUN_ROOT = RAW_RUN_BASE / VERSION
DEFAULT_RESULTS_DIR = HERE / 'results' / VERSION
PHASE2_SUFFIX = '__phase2'
CONFIRM_SUFFIX = '__confirm'
LOG_PREFIX = f'[{VERSION}]'


def git_provenance(root: Path) -> dict:
    """Capture source identity without requiring Git to exist."""
    def git(*args: str) -> tuple[int, str]:
        try:
            proc = subprocess.run(['git', '-C', str(root), *args], text=True,
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            return proc.returncode, (proc.stdout or '').strip()
        except (OSError, subprocess.SubprocessError):
            return 1, ''

    rc, commit = git('rev-parse', 'HEAD')
    if rc:
        return {'available': False, 'commit': '', 'short_commit': '', 'branch': '',
                'describe': '', 'dirty': None, 'source_dirty': None}
    _, branch = git('branch', '--show-current')
    _, describe = git('describe', '--tags', '--always', '--dirty')
    _, status = git('status', '--porcelain')
    source_files = ['VERSION', 'robot.xml', 'environment.py', 'training.py', 'viewer.py', 'run_ladder.py']
    _, source_status = git('status', '--porcelain', '--', *source_files)
    return {
        'available': True,
        'commit': commit,
        'short_commit': commit[:12],
        'branch': branch,
        'describe': describe,
        'dirty': bool(status),
        'source_dirty': bool(source_status),
    }



def foundation_flags() -> dict:
    """Conservative shared cold-start curriculum used only to recover a common locomotor.

    The foundation deliberately avoids every v5007 experimental lever: no open
    frontier, no combo blob, no push/sensor/accel frontier and no special
    rehearsal.  It *does* retain the target hardware/observation contract
    (no software target slew, actuator realism, one-step comm latency and the
    deployable sensor-noise scale), so phase-1 branches do not begin with a
    gratuitous actuator-domain switch.  Its command envelope is intentionally
    easier than the old v5005 athletic envelope; the job here is simply to
    recover a competent shared dynamic stepping/catching policy.
    """
    return {
        '--freeze-obs-norm-after-steps': '0',
        '--obs-norm-mode': 'continuous',
        '--difficulty-base-step': '0.0014',
        '--vx-hi-max': '0.40',
        '--vx-lo-min': '-0.12',
        '--vy-max-mps': '0.14',
        '--yaw-max-rad-s': '0.90',
        '--disable-target-slew': None,
        '--phase-speed-gain': '0.75',
        '--regime-command-frac': '0.60',
        '--push-max-force-n': '0.70',
        '--actuator-realism': None,
        '--comm-latency-steps': '1',
        '--sensor-noise-scale': '1.5',
    }


def preferred_checkpoint(run_dir: Path) -> Path | None:
    """Return the best reusable checkpoint available for a run."""
    for name in ('best_broad_eval.pkl', 'best_eval.pkl', 'latest_mjx.pkl'):
        ck = run_dir / 'checkpoints' / name
        if ck.exists():
            return ck
    return None

def build_branches() -> list[dict]:
    """Purposeful v5007 matrix around the v5005 athletic champion."""
    def br(name: str, flags: dict, group: str, rationale: str) -> dict:
        return {'name': name, 'flags': dict(flags), 'tier': group, 'rationale': rationale}

    # Shared hardware-ish baseline. Most v5007 branches remove the explicit
    # software slew clamp so the simulated geared servo decides whether a
    # requested position can physically be reached at that instant. A single
    # clean fixed-envelope slew_static control retains the old 4 rad/s clamp.
    base = {
        '--freeze-obs-norm-after-steps': '25000000', '--obs-norm-mode': 'continuous',
        '--difficulty-base-step': '0.0024', '--vx-hi-max': '0.60', '--vx-lo-min': '-0.20',
        '--vy-max-mps': '0.22', '--yaw-max-rad-s': '1.50', '--disable-target-slew': None,
        '--phase-speed-gain': '1.25', '--regime-command-frac': '0.80', '--push-max-force-n': '3.0',
        '--actuator-realism': None, '--comm-latency-steps': '1', '--sensor-noise-scale': '1.5',
    }
    slew_static = {k: v for k, v in base.items() if k != '--disable-target-slew'}
    slew_static['--target-rate-limit-rad-s'] = '4.0'

    frontier_axis = {
        **base, '--open-frontier': None, '--frontier-growth-frac': '0.0015',
        '--frontier-retreat-frac': '0.0040', '--athletic-rehearsal-frac': '0.20',
    }
    repaired = {
        **frontier_axis, '--combo-mix-frac': '0.65', '--combo-p': '1.35', '--combo-probe-frac': '0.08',
    }

    return [
        br('slew_static', slew_static, 'control',
           'Clean actuator-interface control: fixed v5005 athletic envelope with the old 4 rad/s software target slew retained.'),
        br('noslew_static', base, 'control',
           'No artificial target slew, but otherwise keep the proven v5005 athletic envelope fixed. Clean actuator-interface baseline.'),
        br('frontier_axis', frontier_axis, 'frontier',
           'Repaired settled edge-tracking frontier without combo sampling. Tests the governor itself before mixed-command geometry.'),
        br('frontier_repaired', repaired, 'frontier',
           'Primary v5007 command experiment: repaired edge frontier + p=1.35 mixed commands + 20% full athletic-envelope rehearsal.'),
        br('frontier_no_rehearsal', {**repaired, '--athletic-rehearsal-frac': '0.0'}, 'ablation',
           'Ablates full-envelope rehearsal. If late yaw/dynamic forgetting returns, rehearsal is causal rather than decorative.'),
        br('frontier_push', {**repaired, '--push-frontier': None}, 'resilience',
           'Repaired command frontier plus the v5006 lever that genuinely expanded: competence-driven push force.'),
        br('frontier_accel', {**repaired, '--accel-frontier': None}, 'transitions',
           'Repaired command frontier plus transition timing controlled by direct ramp-tracking competence.'),
        br('frontier_push_accel', {**repaired, '--push-frontier': None, '--accel-frontier': None}, 'combined',
           'Tests whether the two v5006 frontiers that actually worked compose once command-frontier measurement is repaired.'),
        br('frontier_lower_kl', {**repaired, '--lr-target-kl': '0.025'}, 'optimizer',
           'Tests the 0.025 KL hypothesis without changing the curriculum or actuator interface.'),
        br('resilience_focus', {**repaired, '--push-frontier': None, '--regime-command-frac': '0.90', '--fall-target': '0.34'}, 'resilience',
           'Allows curriculum to tolerate a harder fall band while manoeuvring under stronger pushes; asks whether resilience75 can rise without trading away command fidelity.'),
        br('sensor_frontier_repaired', {**repaired, '--sensor-frontier': None}, 'sensing',
           'Re-tests deployable sensor-margin growth now that vy/yaw precision no longer starts below its own measurement tolerance.'),
        br('frontier_beast2', {**repaired, '--push-frontier': None, '--sensor-frontier': None, '--accel-frontier': None,
                               '--regime-command-frac': '0.90'}, 'combined',
           'Retest the broad v5006 combination only after fixing the governor; determines whether beast-style training was inherently conflicting or merely poisoned by the old frontier gate.'),
    ]


def flags_to_cli(flags: dict) -> list[str]:
    out: list[str] = []
    for k, v in flags.items():
        out.append(k)
        if v is not None:
            out.append(str(v))
    return out


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        with path.open(newline='', encoding='utf-8') as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def num(row: dict, key: str, default: float | None = None) -> float | None:
    try:
        x = float(row.get(key, ''))
        return x if x == x else default
    except (TypeError, ValueError):
        return default


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def summarize_rung(run_dir: Path) -> dict:
    rows = read_csv(run_dir / 'metrics.csv')
    ev = read_csv(run_dir / 'fixed_eval.csv')
    if not rows:
        return {'n_rows': 0, 'final_step': -1, 'final_ep_len': -1.0, 'final_fall_rate': -1.0,
                'eval_core_survival': -1.0, 'eval_stress30_survival': -1.0,
                'durability_score': -1e9}
    last = rows[-1]
    valid = [r for r in ev if num(r, 'eval_core_survival') is not None]
    elast = valid[-1] if valid else {}
    tail = valid[-4:]
    if tail:
        def avg(key: str, default=0.0):
            return _mean([num(r, key, default) or 0.0 for r in tail])
        survival = (0.05*avg('eval_core_survival') + 0.08*avg('eval_stress30_survival') +
                    0.10*avg('eval_dynamic50_survival') + 0.16*avg('eval_dynamic80_survival') +
                    0.12*avg('eval_combo70_survival') + 0.12*avg('eval_resilience50_survival') +
                    0.14*avg('eval_resilience75_survival') + 0.11*avg('eval_robust40_survival') +
                    0.06*avg('eval_growth110_survival') + 0.06*avg('eval_growthcombo110_survival'))
        tracking_pen = (0.10*min(1.0, avg('eval_dynamic80_vx_mae', 1.0)/0.12) +
                        0.10*min(1.0, avg('eval_dynamic80_yaw_mae', 2.0)/0.40) +
                        0.08*min(1.0, avg('eval_combo70_vy_mae', 1.0)/0.12) +
                        0.10*min(1.0, avg('eval_combo70_yaw_mae', 2.0)/0.40) +
                        0.06*min(1.0, avg('eval_growthcombo110_yaw_mae', 2.0)/0.55))
        # Penalise non-monotonic tail collapse. This is deliberately a tail
        # ruler, not a final-checkpoint beauty contest.
        keys = ('eval_dynamic80_survival','eval_resilience75_survival','eval_robust40_survival','eval_growthcombo110_survival')
        drop = max((max(num(r,k,0.0) or 0.0 for r in tail) - min(num(r,k,0.0) or 0.0 for r in tail)) for k in keys)
        durability = survival - tracking_pen - 0.15*drop
    else:
        durability = -1e9
    return {
        'n_rows': len(rows), 'final_step': int(num(last, 'step', -1) or -1),
        'final_ep_len': num(last, 'ep_len_mean', -1.0), 'final_fall_rate': num(last, 'fall_rate', -1.0),
        **{k: num(elast, k, -1.0) for k in (
            'eval_core_survival','eval_stress30_survival','eval_dynamic50_survival','eval_dynamic80_survival',
            'eval_combo70_survival','eval_resilience50_survival','eval_resilience75_survival','eval_robust40_survival',
            'eval_growth110_survival','eval_growthcombo110_survival','eval_dynamic80_vx_mae','eval_dynamic80_yaw_mae',
            'eval_combo70_vy_mae','eval_combo70_yaw_mae','eval_growthcombo110_yaw_mae','eval_broad_score')},
        'frontier_vx_hi': num(last,'frontier_vx_hi',-1.0), 'frontier_vx_lo': num(last,'frontier_vx_lo',-1.0),
        'frontier_vy': num(last,'frontier_vy',-1.0), 'frontier_yaw': num(last,'frontier_yaw',-1.0),
        'frontier_combo_radius': num(last,'frontier_combo_radius',-1.0), 'frontier_push_n': num(last,'frontier_push_n',-1.0),
        'frontier_sensor_scale': num(last,'frontier_sensor_scale',-1.0),
        'durability_score': durability,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def _fmt(v, digits=3):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v if v not in (None, '') else '-')
    if x <= -1e8:
        return '-'
    return f'{x:.{digits}f}'


def safe_report(run_root: Path, results_dir: Path) -> None:
    """Refresh compact Git-tracked evidence from the external raw run store.

    Raw checkpoints, full metrics and console logs remain under ``run_root``.
    Only small durable evidence is copied/written under ``results/<VERSION>/``.
    """
    try:
        results_dir.mkdir(parents=True, exist_ok=True)
        rows = read_csv(run_root / 'ladder_summary.csv')
        meta_path = run_root / 'run_meta.json'
        meta = json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.exists() else {}
        git = meta.get('git') or {}

        lines = [
            f'# EMU {VERSION} results',
            '',
            f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')}",
            '',
            f"- Source version: `{VERSION}`",
            f"- Git commit: `{git.get('commit') or 'not available'}`",
            f"- Git branch: `{git.get('branch') or 'not available'}`",
            f"- Source dirty at launch: `{git.get('source_dirty')}`",
            f"- Raw run store: `{run_root}`",
            '',
            'Fixed held-out evaluators are the main comparison ruler. Training fall rate and live frontier values are supporting evidence, not branch-selection scores by themselves.',
            '',
        ]

        foundation = meta.get('foundation') or {}
        if foundation:
            lines += [
                '## Shared foundation', '',
                f"- run: `{foundation.get('name','')}`",
                f"- target steps: `{int(foundation.get('steps',0)):,}`",
                f"- seed: `{foundation.get('seed','')}`",
                f"- checkpoint used by all phase-1 branches: `{foundation.get('checkpoint','')}`",
                '',
            ]

        if rows:
            lines += [
                '## Run summary', '',
                '| run | phase | seed | ok | step | durability | dyn80 | combo70 | res75 | robust40 | growth110 | growthcombo110 | vx MAE | yaw MAE | vy frontier | yaw frontier | combo radius | push N |',
                '|---|---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
            ]
            for r in rows:
                lines.append(
                    f"| {r.get('name','')} | {r.get('phase','')} | {r.get('seed','')} | "
                    f"{'yes' if str(r.get('ok','')).lower() in ('true','1','yes') else 'no'} | "
                    f"{r.get('final_step','')} | {_fmt(r.get('durability_score'))} | "
                    f"{_fmt(r.get('eval_dynamic80_survival'))} | {_fmt(r.get('eval_combo70_survival'))} | "
                    f"{_fmt(r.get('eval_resilience75_survival'))} | {_fmt(r.get('eval_robust40_survival'))} | "
                    f"{_fmt(r.get('eval_growth110_survival'))} | {_fmt(r.get('eval_growthcombo110_survival'))} | "
                    f"{_fmt(r.get('eval_dynamic80_vx_mae'),4)} | {_fmt(r.get('eval_dynamic80_yaw_mae'),4)} | "
                    f"{_fmt(r.get('frontier_vy'),4)} | {_fmt(r.get('frontier_yaw'),4)} | "
                    f"{_fmt(r.get('frontier_combo_radius'),3)} | {_fmt(r.get('frontier_push_n'),3)} |"
                )
            lines.append('')

        family_scores = meta.get('family_scores') or {}
        if family_scores:
            lines += ['## Cross-seed family scores', '', '| family | mean durability |', '|---|---:|']
            for name, score in sorted(family_scores.items(), key=lambda kv: float(kv[1]), reverse=True):
                lines.append(f'| {name} | {_fmt(score)} |')
            lines.append('')

        if meta.get('phase2_selected'):
            lines += [
                '## Phase 2 selection', '',
                f"- family: `{meta.get('phase2_selected')}`",
                f"- source run: `{meta.get('phase2_source_run','')}`",
                f"- selection: {meta.get('phase2_selection','')}",
                '',
            ]

        errors = meta.get('errors') or []
        lines += ['## Errors', '']
        if errors:
            for e in errors:
                lines.append(f"- `{e.get('branch','unknown')}` failed. Full console/error evidence remains in the external raw run directory.")
        else:
            lines.append('No branch errors recorded.')
        lines.append('')

        lines += [
            '## Raw evidence location', '',
            f'- `{run_root / "ladder_summary.csv"}` — one-row-per-run comparison.',
            f'- `{run_root / "run_meta.json"}` — full launch flags, provenance, selection and error metadata.',
            f'- `{run_root / "rung_runs" / "<name>" / "fixed_eval.csv"}` — complete held-out trajectory.',
            f'- `{run_root / "rung_runs" / "<name>" / "metrics.csv"}` — full training/frontier telemetry.',
            f'- `{run_root / "rung_runs" / "<name>" / "checkpoints" / "best_broad_eval.pkl"}` — preferred policy checkpoint.',
            '',
            'The raw paths above are intentionally outside Git. This directory contains the compact evidence worth preserving with the source history.',
            '',
        ]
        report_path = results_dir / 'results.md'
        report_path.write_text('\n'.join(lines), encoding='utf-8')

        # These two files are small and useful for future cross-version analysis.
        if (run_root / 'ladder_summary.csv').exists():
            shutil.copy2(run_root / 'ladder_summary.csv', results_dir / 'ladder_summary.csv')
        if meta_path.exists():
            shutil.copy2(meta_path, results_dir / 'run_meta.json')

        print(f'    [report] refreshed tracked evidence: {report_path}', flush=True)
    except Exception:
        print(f'{LOG_PREFIX} WARNING: tracked results generation failed (training continues):\n' + traceback.format_exc(), flush=True)

def run_preflight(out_dir: Path) -> None:
    """Cheap CPU-side sanity checks before an overnight GPU batch."""
    smoke = (
        "import jax, optax, mujoco; from mujoco import mjx; "
        "m=mujoco.MjModel.from_xml_path('robot.xml'); "
        "print('jax', jax.__version__, 'backend', jax.default_backend(), "
        "'mujoco', mujoco.__version__, 'nq', m.nq, 'nv', m.nv, 'nu', m.nu)"
    )
    commands = [
        [sys.executable, '-m', 'py_compile', 'environment.py', 'training.py', 'run_ladder.py', 'viewer.py'],
        [sys.executable, '-c', smoke],
    ]
    records=[]; logs=[]
    env=dict(os.environ); env['CUDA_VISIBLE_DEVICES']=''; env['JAX_PLATFORMS']='cpu'
    for cmd in commands:
        proc=subprocess.run(cmd,cwd=HERE,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        logs.append(f'$ {" ".join(cmd)}\n{proc.stdout or ""}\n')
        records.append({'command':cmd,'returncode':proc.returncode,'ok':proc.returncode==0})
        if proc.returncode:
            (out_dir/'preflight.log').write_text('\n'.join(logs),encoding='utf-8')
            (out_dir/'preflight.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
            raise SystemExit(f'Preflight failed: {" ".join(cmd)}\nSee {out_dir / "preflight.log"}')
    (out_dir/'preflight.log').write_text('\n'.join(logs),encoding='utf-8')
    (out_dir/'preflight.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    print(f'{LOG_PREFIX} preflight passed: core Python compile + CPU JAX/MuJoCo/XML smoke test', flush=True)


def run_rung(args, run_dir: Path, steps: int, flags: dict, *, seed: int, provenance: dict,
             resume: str = '', warm_start: str = '') -> tuple[int, str]:
    cmd=[sys.executable,str(TRAINING_PY),
         '--steps',str(steps),'--envs',str(args.envs),'--rollout',str(args.rollout),
         '--epochs',str(args.epochs),'--minibatches',str(args.minibatches),'--lr',str(args.lr),
         '--device',args.device,'--seed',str(seed),'--max-episode-steps',str(args.max_episode_steps),
         '--run-dir',str(run_dir),'--run-name',run_dir.name,
         '--checkpoint-every',str(args.checkpoint_every),'--log-every',str(args.log_every),
         '--eval-every-steps',str(args.eval_every_steps),'--eval-horizon',str(args.eval_horizon),
         '--eval-cohorts',str(args.eval_cohorts),'--obs-norm-anchor-steps',str(args.obs_norm_anchor_steps)]
    if resume: cmd += ['--resume',resume]
    elif warm_start: cmd += ['--warm-start-weights',warm_start]
    if args.allow_cpu_mjx: cmd.append('--allow-cpu-mjx')
    cmd += flags_to_cli(flags)
    run_dir.mkdir(parents=True,exist_ok=True)
    tail=[]
    with (run_dir/'console.log').open('w',encoding='utf-8') as log:
        child_env = dict(os.environ)
        child_env.update({
            'EMU_SOURCE_VERSION': VERSION,
            'EMU_GIT_COMMIT': str(provenance.get('commit') or ''),
            'EMU_GIT_BRANCH': str(provenance.get('branch') or ''),
            'EMU_GIT_DESCRIBE': str(provenance.get('describe') or ''),
            'EMU_GIT_DIRTY': 'true' if provenance.get('source_dirty') else 'false',
        })
        proc=subprocess.Popen(cmd,cwd=HERE,env=child_env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line); log.flush(); tail.append(line); tail=tail[-80:]
            t=line.strip()
            if (t.startswith(f'[emu_{VERSION}] step=') or 'fixed-eval step=' in t or 'AUTO-RECOVERY' in t or
                    'WARNING:' in t or 'Traceback' in t or 'training complete' in t):
                print('    '+t[:240],flush=True)
        rc=proc.wait()
    return rc, ''.join(tail[-50:]) if rc else ''


def print_plan(branches: list[dict], args) -> None:
    print(f'{VERSION} plan: shared foundation -> {len(branches)} branches x {args.steps_per_branch:,} steps, '
          f'{args.eval_cohorts} fixed evaluator cohorts')
    if args.base_checkpoint:
        print(f'  foundation:      skipped (explicit --base-checkpoint supplied)')
    else:
        print(f'  foundation:      one shared cold-start run to {args.foundation_steps:,} steps ({args.foundation_name})')
    print(f'  raw runs:       {Path(args.run_root).expanduser()}')
    print(f'  tracked results:{Path(args.results_dir).expanduser()}')
    for i,b in enumerate(branches,1):
        print(f'  {i:2d}. {b["name"]:24s} [{b["tier"]:12s}] {b["rationale"]}')
    if not args.skip_confirm:
        print(f'  confirm: top {args.confirm_top_k} branch families rerun at seed +{args.confirm_seed_offset}')
    if not args.skip_phase2:
        print(f'  phase 2: selected family continued from best-broad checkpoint to {args.phase2_steps:,} absolute steps')


def main(argv=None) -> int:
    p=argparse.ArgumentParser(description=f'EMU {VERSION} repaired-frontier capability matrix')
    p.add_argument('--foundation-steps',type=int,default=20_000_000,
                   help='absolute steps for the one shared cold-start foundation run')
    p.add_argument('--foundation-name',default='foundation20',
                   help='raw-run directory name for the shared foundation policy')
    p.add_argument('--rebuild-foundation',action='store_true',
                   help='delete and retrain the shared foundation even if a reusable checkpoint exists')
    p.add_argument('--skip-foundation',action='store_true',
                   help='do not train/reuse a foundation; requires --base-checkpoint')
    p.add_argument('--steps-per-branch',type=int,default=60_000_000)
    p.add_argument('--base-checkpoint',default='',
                   help='optional explicit warm-start checkpoint; bypasses the automatic shared foundation')
    p.add_argument('--phase2-steps',type=int,default=120_000_000)
    p.add_argument('--phase2-branch',default='auto',help='auto or explicit branch family')
    p.add_argument('--skip-phase2',action='store_true'); p.add_argument('--phase2-only',action='store_true')
    p.add_argument('--confirm-top-k',type=int,default=2)
    p.add_argument('--confirm-seed-offset',type=int,default=1009)
    p.add_argument('--skip-confirm',action='store_true')
    p.add_argument('--envs',type=int,default=2048); p.add_argument('--rollout',type=int,default=32)
    p.add_argument('--epochs',type=int,default=4); p.add_argument('--minibatches',type=int,default=8)
    p.add_argument('--lr',type=float,default=2e-4); p.add_argument('--max-episode-steps',type=int,default=700)
    p.add_argument('--checkpoint-every',type=int,default=20); p.add_argument('--log-every',type=int,default=10)
    p.add_argument('--eval-every-steps',type=int,default=10_000_000); p.add_argument('--eval-horizon',type=int,default=350)
    p.add_argument('--eval-cohorts',type=int,default=2); p.add_argument('--obs-norm-anchor-steps',type=int,default=20_000_000)
    p.add_argument('--seed',type=int,default=733)
    p.add_argument('--device',choices=['auto','cpu','cuda','gpu'],default='cuda'); p.add_argument('--allow-cpu-mjx',action='store_true')
    p.add_argument('--run-root', default=str(DEFAULT_RUN_ROOT), help='external heavyweight run directory for this VERSION')
    p.add_argument('--results-dir', default=str(DEFAULT_RESULTS_DIR), help='small Git-tracked evidence directory for this VERSION')
    p.add_argument('--only',default='')
    p.add_argument('--skip-preflight',action='store_true'); p.add_argument('--list',action='store_true')
    args=p.parse_args(argv)

    all_branches=build_branches(); branches=all_branches
    if args.only:
        wanted={x.strip() for x in args.only.split(',') if x.strip()}; branches=[b for b in all_branches if b['name'] in wanted]
        missing=wanted-{b['name'] for b in branches}
        if missing: raise SystemExit(f'Unknown branch(es): {sorted(missing)}')
    print_plan(branches,args)
    if args.list: return 0

    run_root=Path(args.run_root).expanduser().resolve()
    results_dir=Path(args.results_dir).expanduser().resolve()
    if HERE.resolve() == run_root or HERE.resolve() in run_root.parents:
        raise SystemExit(f'Raw run root must be outside the Git repository: {run_root}')
    rung_root=run_root/'rung_runs'; rung_root.mkdir(parents=True,exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    provenance = git_provenance(HERE)
    if provenance.get('available'):
        print(f"{LOG_PREFIX} source {provenance.get('short_commit')} branch={provenance.get('branch') or '(detached)'} "
              f"dirty={provenance.get('dirty')} source_dirty={provenance.get('source_dirty')}", flush=True)
        if provenance.get('source_dirty'):
            print(f'{LOG_PREFIX} WARNING: executable source files have uncommitted changes. The run will continue, but provenance is not perfectly reconstructible.', flush=True)
    else:
        print(f'{LOG_PREFIX} WARNING: this folder is not yet a Git repository; commit provenance will be unavailable.', flush=True)
    print(f'{LOG_PREFIX} raw runs -> {run_root}', flush=True)
    print(f'{LOG_PREFIX} tracked evidence -> {results_dir}', flush=True)

    if args.skip_foundation and not args.base_checkpoint and not args.phase2_only:
        raise SystemExit(f'{LOG_PREFIX} --skip-foundation requires --base-checkpoint.')
    if args.foundation_steps <= 0:
        raise SystemExit('--foundation-steps must be > 0')
    if not args.skip_preflight and not args.phase2_only: run_preflight(run_root)

    meta_path=run_root/'run_meta.json'; summary_path=run_root/'ladder_summary.csv'
    try: meta=json.loads(meta_path.read_text()) if meta_path.exists() and (args.phase2_only or args.only) else {}
    except Exception: meta={}
    meta.setdefault('branches',{}); meta.setdefault('errors',[]); meta['version']=VERSION; meta['git']=provenance; meta['args']=vars(args); meta['raw_run_root']=str(run_root); meta['tracked_results_dir']=str(results_dir)
    rows=read_csv(summary_path) if (args.phase2_only or args.only) else []
    start_all=time.time()

    def record(name:str, branch:dict, phase:str, ok:bool, elapsed:float, tail:str, seed:int) -> None:
        nonlocal rows
        sm=summarize_rung(rung_root/name)
        bm={'ok':ok,'phase':phase,'seconds':round(elapsed,2),'flags':' '.join(flags_to_cli(branch['flags'])),
            'tier':branch['tier'],'rationale':branch['rationale'],'seed':seed,**sm}
        meta['branches'][name]=bm
        if not ok: meta['errors'].append({'branch':name,'tail':tail[-5000:]})
        rows=[r for r in rows if r.get('name')!=name]; rows.append({'name':name,**bm}); write_csv(summary_path,rows)
        meta['last_updated']=time.strftime('%Y-%m-%d %H:%M:%S'); meta_path.write_text(json.dumps(meta,indent=2,default=str),encoding='utf-8')
        safe_report(run_root, results_dir)

    # Phase 0: recover exactly one shared locomotor from scratch.  Every
    # phase-1 comparison then begins from the same actor + observation
    # normalizer via --warm-start-weights, while optimizer/curriculum/frontier
    # state starts fresh per branch.  An explicit --base-checkpoint bypasses
    # this phase for advanced/recovery use.
    if not args.phase2_only and not args.base_checkpoint:
        foundation_dir = rung_root / args.foundation_name
        ckpt = preferred_checkpoint(foundation_dir)
        foundation_summary = summarize_rung(foundation_dir) if foundation_dir.exists() else {}
        foundation_complete = int(foundation_summary.get('final_step', -1) or -1) >= args.foundation_steps and ckpt is not None

        if args.rebuild_foundation and foundation_dir.exists():
            print(f'{LOG_PREFIX} rebuilding shared foundation: deleting {foundation_dir}', flush=True)
            shutil.rmtree(foundation_dir)
            ckpt = None
            foundation_complete = False

        if foundation_complete:
            print(f'{LOG_PREFIX} reusing completed shared foundation: {ckpt}', flush=True)
        else:
            foundation_dir.mkdir(parents=True, exist_ok=True)
            rationale = (
                f'Shared {args.foundation_steps:,}-step cold-start locomotion foundation. '
                'All v5007 phase-1 branches warm-start from this exact policy; '
                'this run is not itself an experimental branch.\n'
            )
            (foundation_dir / 'rationale.txt').write_text(rationale, encoding='utf-8')
            resume_ckpt = preferred_checkpoint(foundation_dir)
            if resume_ckpt:
                print(f'\n{LOG_PREFIX} === phase 0 foundation: resuming {args.foundation_name} -> {args.foundation_steps:,} ===', flush=True)
                t0=time.time(); rc,tail=run_rung(args,foundation_dir,args.foundation_steps,foundation_flags(),
                                                  seed=args.seed,provenance=provenance,resume=str(resume_ckpt))
            else:
                print(f'\n{LOG_PREFIX} === phase 0 foundation: cold start {args.foundation_name} -> {args.foundation_steps:,} ===', flush=True)
                t0=time.time(); rc,tail=run_rung(args,foundation_dir,args.foundation_steps,foundation_flags(),
                                                  seed=args.seed,provenance=provenance)
            foundation_branch={'name':args.foundation_name,'flags':foundation_flags(),'tier':'foundation',
                               'rationale':rationale.strip()}
            record(args.foundation_name, foundation_branch, 'foundation', rc==0, time.time()-t0, tail, args.seed)
            if rc:
                raise SystemExit(f'{LOG_PREFIX} shared foundation failed; phase-1 branches were not started.\n{tail}')
            ckpt = preferred_checkpoint(foundation_dir)
            if ckpt is None:
                raise SystemExit(f'{LOG_PREFIX} foundation completed but no reusable checkpoint was found in {foundation_dir / "checkpoints"}')

        args.base_checkpoint = str(ckpt)
        meta['foundation'] = {
            'name': args.foundation_name,
            'steps': args.foundation_steps,
            'checkpoint': str(ckpt),
            'seed': args.seed,
            'flags': flags_to_cli(foundation_flags()),
        }
        meta_path.write_text(json.dumps(meta,indent=2,default=str),encoding='utf-8')
        safe_report(run_root, results_dir)
    elif not args.phase2_only:
        print(f'{LOG_PREFIX} using explicit shared warm start: {args.base_checkpoint}', flush=True)
        if not Path(args.base_checkpoint).expanduser().exists():
            raise SystemExit(f'{LOG_PREFIX} --base-checkpoint does not exist: {args.base_checkpoint}')

    if not args.phase2_only:
        for i,branch in enumerate(branches,1):
            name=branch['name']; d=rung_root/name
            if d.exists(): shutil.rmtree(d)
            d.mkdir(parents=True); (d/'rationale.txt').write_text(branch['rationale']+'\n',encoding='utf-8')
            print(f'\n{LOG_PREFIX} === branch {i}/{len(branches)}: {name} ===\n{LOG_PREFIX} {branch["rationale"]}',flush=True)
            t0=time.time(); rc,tail=run_rung(args,d,args.steps_per_branch,branch['flags'],seed=args.seed,provenance=provenance,warm_start=args.base_checkpoint)
            record(name,branch,'phase1',rc==0,time.time()-t0,tail,args.seed)
            if rc: print(f'{LOG_PREFIX} branch {name} failed; continuing.\n{tail}',flush=True)

        # Confirmation seed: rerun top K branch FAMILIES from the same warm-start.
        if not args.skip_confirm and args.confirm_top_k>0:
            scored=[]
            for b in branches:
                sm=summarize_rung(rung_root/b['name'])
                if sm['durability_score']>-1e8: scored.append((sm['durability_score'],b))
            for _,branch in sorted(scored,key=lambda x:x[0],reverse=True)[:args.confirm_top_k]:
                seed2=args.seed+args.confirm_seed_offset; name=branch['name']+CONFIRM_SUFFIX; d=rung_root/name
                if d.exists(): shutil.rmtree(d)
                d.mkdir(parents=True); (d/'rationale.txt').write_text(branch['rationale']+f'\nConfirmation seed {seed2}.\n',encoding='utf-8')
                print(f'\n{LOG_PREFIX} === confirmation: {branch["name"]} seed={seed2} ===',flush=True)
                t0=time.time(); rc,tail=run_rung(args,d,args.steps_per_branch,branch['flags'],seed=seed2,provenance=provenance,warm_start=args.base_checkpoint)
                record(name,branch,'confirm',rc==0,time.time()-t0,tail,seed2)

    # Family score = mean across available primary/confirmation seeds.
    lookup={b['name']:b for b in all_branches}
    family_scores={}
    for b in all_branches:
        vals=[]
        for rn in (b['name'], b['name']+CONFIRM_SUFFIX):
            sm=summarize_rung(rung_root/rn)
            if sm['durability_score']>-1e8: vals.append(sm['durability_score'])
        if vals: family_scores[b['name']]=sum(vals)/len(vals)
    meta['family_scores']=family_scores

    if not args.skip_phase2:
        chosen=args.phase2_branch
        if chosen=='auto': chosen=max(family_scores,key=family_scores.get) if family_scores else ''
        if chosen not in lookup:
            print(f'{LOG_PREFIX} phase 2 skipped: no valid selected branch family ({chosen!r}).',flush=True)
        else:
            branch=lookup[chosen]
            # Continue whichever seed instance of this family scored better,
            # from its best held-out checkpoint rather than blindly from final.
            sources=[]
            for rn in (chosen, chosen+CONFIRM_SUFFIX):
                d=rung_root/rn; sm=summarize_rung(d)
                ck=(d/'checkpoints'/'best_broad_eval.pkl')
                if not ck.exists(): ck=d/'checkpoints'/'latest_mjx.pkl'
                if ck.exists() and sm['durability_score']>-1e8: sources.append((sm['durability_score'],rn,ck))
            if not sources:
                print(f'{LOG_PREFIX} phase 2 skipped: no checkpoint for {chosen}.',flush=True)
            else:
                _,source_name,ckpt=max(sources,key=lambda x:x[0])
                meta['phase2_selected']=chosen; meta['phase2_source_run']=source_name
                meta['phase2_selection']='mean family durability across available training seeds' if args.phase2_branch=='auto' else 'explicit user/CLI selection'
                name2=chosen+PHASE2_SUFFIX; d2=rung_root/name2
                if d2.exists(): shutil.rmtree(d2)
                d2.mkdir(parents=True); (d2/'rationale.txt').write_text(
                    f'Long continuation of {chosen} from {source_name} best-broad checkpoint to {args.phase2_steps:,} absolute steps.\n',encoding='utf-8')
                source_seed=int(meta.get('branches',{}).get(source_name,{}).get('seed',args.seed))
                print(f'\n{LOG_PREFIX} === phase 2: {chosen} via {source_name} -> {args.phase2_steps:,} ===',flush=True)
                t0=time.time(); rc,tail=run_rung(args,d2,args.phase2_steps,branch['flags'],seed=source_seed,provenance=provenance,resume=str(ckpt))
                record(name2,branch,'phase2',rc==0,time.time()-t0,tail,source_seed)
                if rc: print(f'{LOG_PREFIX} phase 2 failed; completed evidence is still preserved.\n{tail}',flush=True)

    meta['wall_seconds_this_invocation']=round(time.time()-start_all,2); meta['last_updated']=time.strftime('%Y-%m-%d %H:%M:%S')
    meta_path.write_text(json.dumps(meta,indent=2,default=str),encoding='utf-8'); safe_report(run_root, results_dir)
    print(f'\n{LOG_PREFIX} complete. Tracked result: {results_dir / "results.md"}',flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
