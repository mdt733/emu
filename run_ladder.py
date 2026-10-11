#!/usr/bin/env python3
"""EMU v5012 self-mapping trainer runner.

v5012 keeps older diagnostic routes, but its primary product is the event-based graph cold-start trainer.  Stage ``profile`` is deliberately small: it
runs one short continuation with detailed wall-time attribution so performance
work is evidence-driven.  Stage ``capability`` runs the focused v5008 matrix,
adds harder held-out rulers and confirms close winners across extra seeds.

Typical use:
    python run_ladder.py --stage trainer --list
    python run_ladder.py --stage trainer --device cuda
    python run_ladder.py --list
    python run_ladder.py --stage profile --device cuda
    python run_ladder.py --stage capability --device cuda
    python run_ladder.py --stage capability --start zero --device cuda
    python run_ladder.py --stage capability --start checkpoint --checkpoint /path/model.pkl
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
REFERENCE_BASE = Path(os.environ.get('EMU_REFERENCE_ROOT', str(HERE.parent / 'emu_references'))).expanduser()
DEFAULT_REFERENCE_DIR = REFERENCE_BASE / 'v5007_120m'
DEFAULT_REFERENCE_CHECKPOINT = DEFAULT_REFERENCE_DIR / 'latest_mjx.pkl'
PHASE2_SUFFIX = '__phase2'
THIRD_SEED_SUFFIX = '__confirm2'
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
    source_files = ['VERSION', 'robot.xml', 'environment.py', 'training.py', 'viewer.py', 'run_ladder.py', 'atlas.py', 'capability_cloud.py', 'capability_field.py', 'field_viewer.py', 'emu_cli.py', 'pyproject.toml', 'README.md', 'docs/v5012-design.md']
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

    The foundation deliberately avoids every v5008 experimental lever: no open
    frontier, no combo blob, no push/sensor/accel frontier and no special
    rehearsal.  It *does* retain the target hardware/observation contract
    (no software target slew, actuator realism, one-step comm latency and the
    deployable sensor-noise scale), so phase-1 branches do not begin with a
    gratuitous actuator-domain switch.  Its command envelope is intentionally
    easier than the old v5005 athletic envelope; the job here is simply to
    recover a competent shared dynamic stepping/catching policy for zero-start regression runs.
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
        # Zero-start foundation remains a cheap locomotion bootstrap; capability
        # cloud telemetry begins on the experimental branches, not during seed recovery.
        '--cloud-probes-per-eval': '0',
        '--cloud-final-probes': '0',
    }



def trainer_foundation_flags() -> dict:
    """0->20M nursery for trainer qualification.

    The live capability field observes from step zero in shadow mode so every
    fork inherits the same evidence, but it cannot steer the nursery.  Physical
    recovery maxima already match the mature trainer so the 20M handoff does
    not silently change what a resilience difficulty means.
    """
    f=foundation_flags()
    f.update({
        '--capability-field-mode':'shadow',
        '--field-half-life-steps':'20000000',
        '--field-influence-start-steps':'10000000',
        '--field-influence-full-steps':'20000000',
        '--field-max-influence':'0.65',
        '--recovery-tilt-max-deg':'28.0', '--recovery-position-max-m':'0.025',
        '--recovery-joint-max-deg':'14.0', '--recovery-linvel-max-mps':'0.65',
        '--recovery-angvel-max-rad-s':'3.6',
    })
    return f


def build_trainer_branches() -> list[dict]:
    """Historical ablations plus the v5012 recommended graph trainer from one exact 20M ancestor."""
    def br(name, mode, rationale):
        common={
            '--freeze-obs-norm-after-steps':'25000000', '--obs-norm-mode':'continuous',
            '--difficulty-base-step':'0.0024', '--vx-hi-max':'0.60', '--vx-lo-min':'-0.20',
            '--vy-max-mps':'0.22', '--yaw-max-rad-s':'1.50', '--disable-target-slew':None,
            '--phase-speed-gain':'1.25', '--regime-command-frac':'0.85', '--push-max-force-n':'3.2',
            '--actuator-realism':None, '--comm-latency-steps':'1', '--sensor-noise-scale':'1.5',
            '--open-frontier':None, '--frontier-growth-frac':'0.0015', '--frontier-retreat-frac':'0.0040',
            '--athletic-rehearsal-frac':'0.10', '--combo-mix-frac':'0.88', '--combo-p':'1.35',
            '--combo-probe-frac':'0.12', '--sensor-frontier':None,
            '--frontier-vx-hi-safety':'1.20', '--frontier-vx-lo-safety':'0.60',
            '--frontier-vy-safety':'0.60', '--frontier-yaw-safety':'4.50', '--frontier-combo-safety':'2.00',
            '--recovery-tilt-max-deg':'28.0', '--recovery-position-max-m':'0.025',
            '--recovery-joint-max-deg':'14.0', '--recovery-linvel-max-mps':'0.65',
            '--recovery-angvel-max-rad-s':'3.6', '--push-frontier':None,
            '--frontier-push-safety-n':'3.2', '--fall-target':'0.34', '--accel-frontier':None, '--legacy-hard-probes':None,
            '--capability-field-mode':mode, '--field-half-life-steps':'20000000',
            '--field-influence-start-steps':'20000000', '--field-influence-full-steps':'30000000',
            '--field-max-influence':'0.65',
        }
        return {'name':name,'flags':common,'tier':'trainer','rationale':rationale}
    return [
        br('trainer_shadow','shadow',
           'Ablation control: builds the v5012 event-based field but never lets it steer births or transitions.'),
        br('trainer_nodes','nodes',
           'Ablation: event-based command/recovery nodes steer bounded births; maneuver edges are observed but do not steer.'),
        br('trainer_graph','graph',
           'Recommended v5012 trainer: statistically honest node/recovery events plus complete maneuver-edge events steer practice; governor retains global safety/frontiers.'),
    ]


def trainer_rank(summary: dict, control: dict | None = None) -> tuple:
    """Transparent scheduler rule, not a claimed scalar capability score."""
    def f(k,d=-1.0):
        try:
            x=float(summary.get(k,d)); return x if x==x else d
        except Exception: return d
    dur=f('durability_score',-1e9)
    base=f('cloud_baseline_competent_frac',-1.0)
    rec=f('cloud_valid_recovery_mean',-1.0)
    worst=f('cloud_valid_worst_direction_mean',-1.0)
    cmd=f('field_competent_nodes',0.0)
    edges=f('field_competent_edges',0.0)
    known_edges=f('field_known_edges',0.0)
    try: valid_n=int(float(summary.get('cloud_valid_n',0)))
    except Exception: valid_n=0
    if control:
        try: cd=float(control.get('durability_score',dur))
        except Exception: cd=dur
        try: cb=float(control.get('cloud_baseline_competent_frac',base))
        except Exception: cb=base
        eligible=(dur >= cd-0.05 and base >= cb-0.10 and valid_n >= 4)
    else:
        eligible=(dur>-1e8 and valid_n>=4)
    # Final held-out panels are identical across branches. Preserve command
    # competence/durability as eligibility gates, then prefer recovery and its
    # worst-direction behavior. Training-derived field coverage only breaks
    # ties; it is never treated as independent exam evidence.
    return (1 if eligible else 0, rec, worst, base, edges, known_edges, cmd, dur)

def preferred_checkpoint(run_dir: Path) -> Path | None:
    """Return the best reusable checkpoint available for a run."""
    for name in ('best_broad_eval.pkl', 'best_eval.pkl', 'best_v5008_eval.pkl', 'latest_mjx.pkl'):
        ck = run_dir / 'checkpoints' / name
        if ck.exists():
            return ck
    return None

def build_branches() -> list[dict]:
    """v5008 stage-5 recovery/capability campaign around the v5007 120M champion."""
    def br(name: str, flags: dict, group: str, rationale: str) -> dict:
        return {'name': name, 'flags': dict(flags), 'tier': group, 'rationale': rationale}

    reference = {
        '--freeze-obs-norm-after-steps': '25000000', '--obs-norm-mode': 'continuous',
        '--difficulty-base-step': '0.0024', '--vx-hi-max': '0.60', '--vx-lo-min': '-0.20',
        '--vy-max-mps': '0.22', '--yaw-max-rad-s': '1.50', '--disable-target-slew': None,
        '--phase-speed-gain': '1.25', '--regime-command-frac': '0.80', '--push-max-force-n': '3.0',
        '--actuator-realism': None, '--comm-latency-steps': '1', '--sensor-noise-scale': '1.5',
        '--open-frontier': None, '--frontier-growth-frac': '0.0015', '--frontier-retreat-frac': '0.0040',
        '--athletic-rehearsal-frac': '0.20', '--combo-mix-frac': '0.65', '--combo-p': '1.35',
        '--combo-probe-frac': '0.08', '--sensor-frontier': None,
    }

    extended = {
        '--frontier-vx-hi-safety': '1.20', '--frontier-vx-lo-safety': '0.60',
        '--frontier-vy-safety': '0.60', '--frontier-yaw-safety': '4.50',
        '--frontier-combo-safety': '2.00',
    }

    # One resilience difficulty scalar remains authoritative. These branches
    # change only the physical envelope that scalar expands into.
    pose = {
        '--reset-resilience-difficulty': '0.20',
        '--recovery-tilt-max-deg': '38.0', '--recovery-position-max-m': '0.025',
        '--recovery-joint-max-deg': '20.0', '--recovery-linvel-max-mps': '0.12',
        '--recovery-angvel-max-rad-s': '1.2',
    }
    momentum = {
        '--reset-resilience-difficulty': '0.20',
        '--recovery-tilt-max-deg': '10.0', '--recovery-position-max-m': '0.015',
        '--recovery-joint-max-deg': '5.0', '--recovery-linvel-max-mps': '0.90',
        '--recovery-angvel-max-rad-s': '5.0',
    }
    impact = {
        '--reset-resilience-difficulty': '0.20',
        '--recovery-tilt-max-deg': '12.0', '--recovery-position-max-m': '0.015',
        '--recovery-joint-max-deg': '6.0', '--recovery-linvel-max-mps': '0.20',
        '--recovery-angvel-max-rad-s': '1.5',
        '--push-frontier': None, '--push-max-force-n': '4.0',
        '--frontier-push-safety-n': '4.0', '--fall-target': '0.34',
    }
    chaos = {
        '--reset-resilience-difficulty': '0.16',
        '--recovery-tilt-max-deg': '28.0', '--recovery-position-max-m': '0.025',
        '--recovery-joint-max-deg': '14.0', '--recovery-linvel-max-mps': '0.65',
        '--recovery-angvel-max-rad-s': '3.6',
        '--push-frontier': None, '--push-max-force-n': '3.2',
        '--frontier-push-safety-n': '3.2', '--fall-target': '0.34',
    }

    return [
        br('reference_continue', reference, 'control',
           'Exact v5007 champion recipe. Control for forgetting, PPO drift and cloud measurement.'),

        br('combo_maneuver', {**reference, **extended,
                              '--combo-mix-frac': '0.85', '--combo-probe-frac': '0.12',
                              '--accel-frontier': None, '--athletic-rehearsal-frac': '0.10'},
           'command',
           'Expands genuine vx+vy+yaw combinations and rapid command transitions without adding new recovery disorder.'),

        br('pose_recovery', {**reference, **pose, '--combo-mix-frac': '0.75',
                             '--regime-command-frac': '0.78'},
           'recovery',
           'Trains recovery from plausible body tilt and independent joint-state disorder while preserving mixed locomotion.'),

        br('momentum_recovery', {**reference, **momentum, '--combo-mix-frac': '0.75',
                                 '--regime-command-frac': '0.78'},
           'recovery',
           'Trains recovery from linear and angular momentum mismatch, including already-moving bad births.'),

        br('impact_recovery', {**reference, **impact, '--combo-mix-frac': '0.75',
                               '--regime-command-frac': '0.85'},
           'recovery',
           'Focuses on recoverable external impacts with a competence-earned physical push frontier.'),

        br('chaos_recovery', {**reference, **chaos, '--combo-mix-frac': '0.80',
                              '--regime-command-frac': '0.82'},
           'recovery',
           'Combines moderate pose, momentum and impact disorder under one recovery difficulty scalar.'),

        br('chaos_rehearsal10', {**reference, **chaos, '--combo-mix-frac': '0.80',
                                  '--regime-command-frac': '0.82',
                                  '--athletic-rehearsal-frac': '0.10'},
           'distribution',
           'Tests whether 10% clean athletic rehearsal preserves old competence while freeing more births for recovery learning.'),

        br('chaos_lower_kl', {**reference, **chaos, '--combo-mix-frac': '0.80',
                              '--regime-command-frac': '0.82',
                              '--athletic-rehearsal-frac': '0.10', '--lr-target-kl': '0.025'},
           'optimizer',
           'Tests the promising lower-KL controller specifically under the harder mixed-recovery distribution.'),

        br('integrated_maneuver_recovery', {**reference, **extended, **chaos,
                                             '--combo-mix-frac': '0.88', '--combo-probe-frac': '0.12',
                                             '--regime-command-frac': '0.85',
                                             '--athletic-rehearsal-frac': '0.10',
                                             '--accel-frontier': None},
           'integration',
           'Final integration hypothesis: wide mixed commands + rapid transitions + moderate chaos recovery in one policy.'),
    ]

def profile_branch() -> dict:
    """Exact winning recipe for a short profiling probe."""
    return build_branches()[0]


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



def summarize_cloud_final(run_dir: Path) -> dict:
    rows = read_csv(run_dir / 'heldout_capability_cloud.csv')
    if not rows:
        return {
            'cloud_baseline_competent_frac': -1.0,
            'cloud_baseline_recovery_mean': -1.0,
            'cloud_valid_recovery_mean': -1.0,
            'cloud_valid_worst_direction_mean': -1.0,
            'cloud_valid_survival_mean': -1.0,
            'cloud_valid_n': 0,
            'cloud_final_points': 0,
        }
    # Prefer the forced final panel; if a partial run lacks it, use the latest
    # policy-step panel so interrupted campaigns still retain useful evidence.
    final = [r for r in rows if str(r.get('forced_final','0')).lower() in ('1','true','yes')]
    if not final:
        steps=[int(float(r.get('policy_step',-1) or -1)) for r in rows]
        last=max(steps) if steps else -1
        final=[r for r in rows if int(float(r.get('policy_step',-1) or -1)) == last]
    controls=[r for r in final if r.get('family') == 'none']
    valid=[r for r in final if r.get('family') != 'none' and
           str(r.get('resilience_valid','0')).lower() in ('1','true','yes')]
    def mean(key, vals, default=-1.0):
        xs=[]
        for r in vals:
            try:
                x=float(r.get(key,''))
                if x==x: xs.append(x)
            except Exception:
                pass
        return sum(xs)/len(xs) if xs else default
    cal=[]
    for r in controls:
        try:
            a=float(r.get('field_predicted_competence','nan')); b=float(r.get('node_window_rate','nan'))
            if a==a and b==b: cal.append((a,b))
        except Exception:
            pass
    return {
        'cloud_baseline_competent_frac': mean('baseline_competent', controls),
        'cloud_baseline_recovery_mean': mean('recovery_rate', controls),
        'cloud_valid_recovery_mean': mean('recovery_rate', valid),
        'cloud_valid_worst_direction_mean': mean('recovery_worst_direction', valid),
        'cloud_valid_survival_mean': mean('survival', valid),
        'cloud_valid_n': len(valid),
        'cloud_final_points': len(final),
        'field_calibration_mae': (sum(abs(a-b) for a,b in cal)/len(cal)) if cal else -1.0,
        'field_calibration_brier': (sum((a-b)**2 for a,b in cal)/len(cal)) if cal else -1.0,
        'field_calibration_points': len(cal),
        'field_calibration_target': 'node_window_rate',
    }


def campaign_rank(summary: dict, reference: dict | None = None) -> tuple:
    """Transparent lexicographic campaign ranking.

    First preserve old longitudinal locomotion and undisturbed command coverage.
    Among policies that do, prefer actual held-out disturbed recovery, then
    worst-direction recovery. This is deliberately not presented as a physical
    capability score; it is only the automation rule for buying extra seeds.
    """
    def fv(obj, key, default):
        try:
            x=float(obj.get(key, default))
            return x if x == x else default
        except Exception:
            return default
    dur=fv(summary,'durability_score',-1e9)
    base=fv(summary,'cloud_baseline_competent_frac',-1.0)
    rec=fv(summary,'cloud_valid_recovery_mean',-1.0)
    worst=fv(summary,'cloud_valid_worst_direction_mean',-1.0)
    try: n=int(float(summary.get('cloud_valid_n',0)))
    except Exception: n=0
    if reference:
        ref_dur=fv(reference,'durability_score',dur)
        ref_base=fv(reference,'cloud_baseline_competent_frac',base)
        eligible = (dur >= ref_dur - 0.04 and base >= ref_base - 0.05 and n >= 4)
    else:
        eligible = (dur > -1e8 and n >= 4)
    return (1 if eligible else 0, rec, worst, base, dur)

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
    cloud = summarize_cloud_final(run_dir)
    try:
        field_summary=json.loads((run_dir/'capability_field_summary.json').read_text()) if (run_dir/'capability_field_summary.json').exists() else {}
    except Exception:
        field_summary={}
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
        'eval_dynamic100_survival': num(elast,'eval_dynamic100_survival',-1.0),
        'eval_resilience100_survival': num(elast,'eval_resilience100_survival',-1.0),
        'eval_robust70_survival': num(elast,'eval_robust70_survival',-1.0),
        'eval_growth140_survival': num(elast,'eval_growth140_survival',-1.0),
        'eval_growthcombo140_survival': num(elast,'eval_growthcombo140_survival',-1.0),
        'eval_v5008_score': num(elast,'eval_v5008_score',durability),
        'durability_score': durability,
        **cloud,
        **{k: field_summary.get(k, -1.0) for k in (
            'field_schema','field_known_nodes','field_competent_nodes','field_frontier_nodes','field_stale_nodes',
            'field_mean_competence_known','field_mean_lower_known','field_mean_quality_known',
            'field_known_recovery_cells','field_competent_recovery_cells','field_known_edges','field_competent_edges',
            'field_mean_edge_competence','field_mean_edge_rate','field_effective_samples',
            'field_node_events_total','field_recovery_events_total','field_edge_events_total')},
        # Kept for backward-compatible CSV readers. Stage-5 confirmation and
        # phase-2 selection use campaign_rank(), not this scalar.
        'selection_score': durability,
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
            ('v5012 records gait-window node events plus explicit one-frame maneuvers. Held-out node calibration and recovery exams remain quarantined and never update the live field.'
             if (meta.get('args') or {}).get('stage') == 'trainer' else
             'Fixed held-out evaluators are the main comparison ruler. Training fall rate and live frontier values are supporting evidence, not branch-selection scores by themselves.'),
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
                '| run | phase | seed | ok | step | durability | heldout recovery | worst-dir | cal MAE | nodes known/mastered | edges known/mastered | vx+ | vx- | seconds |',
                '|---|---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
            ]
            for r in rows:
                lines.append(
                    f"| {r.get('name','')} | {r.get('phase','')} | {r.get('seed','')} | "
                    f"{'yes' if str(r.get('ok','')).lower() in ('true','1','yes') else 'no'} | "
                    f"{r.get('final_step','')} | {_fmt(r.get('durability_score'))} | "
                    f"{_fmt(r.get('cloud_valid_recovery_mean'))} | {_fmt(r.get('cloud_valid_worst_direction_mean'))} | "
                    f"{_fmt(r.get('field_calibration_mae'))} | "
                    f"{_fmt(r.get('field_known_nodes'),0)}/{_fmt(r.get('field_competent_nodes'),0)} | "
                    f"{_fmt(r.get('field_known_edges'),0)}/{_fmt(r.get('field_competent_edges'),0)} | "
                    f"{_fmt(r.get('frontier_vx_hi'),3)} | {_fmt(r.get('frontier_vx_lo'),3)} | {_fmt(r.get('seconds'),1)} |"
                )
            if (meta.get('args') or {}).get('stage') == 'trainer':
                lines += ['', '* Field node/edge counts are **training-derived** self-map evidence, useful for trainer efficiency and diagnosis. Held-out recovery/worst-direction columns remain independent exam evidence.', '']
            else:
                lines += ['', '* Old fixed dyn/resilience/hard probes remain regression coordinates only.', '']

        family_scores = meta.get('family_scores') or {}
        if family_scores:
            lines += ['## Cross-seed family scores', '', '| family | mean longitudinal durability |', '|---|---:|']
            for name, score in sorted(family_scores.items(), key=lambda kv: float(kv[1]), reverse=True):
                lines.append(f'| {name} | {_fmt(score)} |')
            lines.append('')

        family_capability = meta.get('family_capability') or {}
        if family_capability:
            lines += [
                '## Cross-seed paired capability evidence', '',
                '| family | seeds | durability | baseline command OK | valid disturbed recovery | worst-direction recovery | valid n |',
                '|---|---:|---:|---:|---:|---:|---:|',
            ]
            ref = family_capability.get('reference_continue') or {}
            ordered = sorted(
                family_capability.items(),
                key=lambda kv: campaign_rank(kv[1], ref),
                reverse=True)
            for name, d in ordered:
                lines.append(
                    f"| {name} | {d.get('seeds','')} | {_fmt(d.get('durability_score'))} | "
                    f"{_fmt(d.get('cloud_baseline_competent_frac'))} | "
                    f"{_fmt(d.get('cloud_valid_recovery_mean'))} | "
                    f"{_fmt(d.get('cloud_valid_worst_direction_mean'))} | {d.get('cloud_valid_n','')} |"
                )
            lines += ['', 'Selection is lexicographic and explicit: preserve reference durability (within 0.04) and undisturbed command competence (within 0.05), require at least four valid paired disturbance points, then prefer held-out disturbed recovery and worst-direction recovery.', '']

        if meta.get('trainer_selected'):
            lines += [
                '## Trainer selection', '',
                f"- selected trainer: `{meta.get('trainer_selected')}`",
                f"- rule: {meta.get('trainer_selection','')}",
                '',
            ]
            ranking=meta.get('trainer_ranking') or []
            if ranking:
                lines += ['| trainer | rank tuple | heldout recovery | competent nodes | competent edges | durability |',
                          '|---|---|---:|---:|---:|---:|']
                for r in ranking:
                    sm=r.get('summary') or {}
                    lines.append(f"| {r.get('name','')} | `{r.get('rank_tuple',[])}` | {_fmt(sm.get('cloud_valid_recovery_mean'))} | {_fmt(sm.get('field_competent_nodes'),0)} | {_fmt(sm.get('field_competent_edges'),0)} | {_fmt(sm.get('durability_score'))} |")
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
            f'- `{run_root / "rung_runs" / "<name>" / "performance.json"}` — cumulative wall-time attribution when profiling is enabled.',
            f'- `{run_root / "rung_runs" / "<name>" / "heldout_capability_cloud.csv"}` — held-out sparse combined-command probe points accumulated during training.',
            f'- `{run_root / "rung_runs" / "<name>" / "capability_field_latest.npz"}` — continuously updated training-derived node/recovery/maneuver field.',
            f'- `{run_root / "rung_runs" / "<name>" / "capability_field_summary.json"}` — compact current field diagnostics.',
            f'- `{run_root / "rung_runs" / "<name>" / "checkpoints" / "best_eval.pkl"}` — public alias for the preferred fixed-regression checkpoint; historical filenames remain readable.',
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
    recovery_smoke = (
        "import jax, jax.numpy as jnp; "
        "from environment import EmuMJXEnv; from training import RegimeGovernor; "
        "e=EmuMJXEnv('robot.xml',num_envs=8,seed=7,max_episode_steps=80,"
        "actuator_realism=True,comm_latency_steps=1,disable_target_slew=True,"
        "recovery_tilt_max_deg=28.0,recovery_linvel_max_mps=0.65,"
        "recovery_angvel_max_rad_s=3.6,recovery_joint_max_deg=14.0); "
        "p=RegimeGovernor().params_dict(); s,_obs=e.reset(p); "
        "o=e.step(s,jnp.zeros((8,6),dtype=jnp.float32),p); "
        "jax.block_until_ready(o.obs); print('recovery-smoke',o.obs.shape)"
    )
    cloud_batch_smoke = (
        "import jax, jax.numpy as jnp; "
        "from environment import EmuMJXEnv; from training import RegimeGovernor, make_trainer; "
        "from capability_cloud import make_panel, evaluate_specs_batch; "
        "e=EmuMJXEnv('robot.xml',num_envs=8,seed=9,max_episode_steps=40,actuator_realism=True,comm_latency_steps=1,disable_target_slew=True); "
        "tr=make_trainer(e,seed=9,lr=2e-4,rollout=4,epochs=1,minibatches=1,obs_norm_mode='continuous'); "
        "rp=RegimeGovernor().params_dict(); st=tr['make_initial_state'](rp); specs=make_panel(5)[:2]; "
        "keys=[jax.random.PRNGKey(91),jax.random.PRNGKey(92)]; "
        "rows=evaluate_specs_batch(tr,e,st.params,jnp.zeros((tr['obs_dim'],),jnp.float32),jnp.ones((tr['obs_dim'],),jnp.float32),keys,8,specs,stable_steps=3,chunk_size=2); "
        "print('cloud-batch-smoke',len(rows),rows[0]['family'],rows[1]['family'])"
    )
    field_smoke = (
        "import numpy as np; from capability_field import CapabilityField,FIELD_VERSION; "
        "f=CapabilityField(); T,N=96,8; z=np.zeros((T,N),np.float32); o=np.ones((T,N),np.float32); "
        "age=(np.arange(T,dtype=np.float32)[:,None]+1)*.02+np.zeros((T,N),np.float32); "
        "m={'cmd_vx':z+.10,'cmd_vy':z,'cmd_yaw':z,'target_cmd_vx':z+.10,'target_cmd_vy':z,'target_cmd_yaw':z,"
        "'transition_ramp_speed':z,'vx':z+.10,'vy':z,'yaw_rate':z,'gravity_z':o,'support':o*.30,'bad_contact':z,"
        "'episode_fell':z,'episode_done':z,'age_s':age,'regime_resilience':z,'birth_resilience_d':z,'transition_frac':z,'cmd_resampled':z}; "
        "d=f.update_from_rollout(m,step=65536); p=f.teaching_packet(step=30000000,mode='graph',start=20000000,full=30000000,maximum=.65); "
        "assert FIELD_VERSION==3 and d['field_node_events_update']>0 and p['field_node_anchors'].shape==(24,5) and p['field_edge_anchors'].shape==(24,7); "
        "g=CapabilityField.from_state_dict(f.state_dict()); assert g.summary()['field_node_events_total']==f.summary()['field_node_events_total']; "
        "print('field-v3-smoke',d['field_node_events_update'],d['field_mean_competence_known'],p['field_node_frac'])"
    )
    commands = [
        [sys.executable, '-m', 'py_compile', 'environment.py', 'training.py', 'run_ladder.py', 'viewer.py', 'atlas.py', 'capability_cloud.py', 'capability_field.py', 'field_viewer.py', 'emu_cli.py'],
        [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
        [sys.executable, '-c', field_smoke],
        [sys.executable, '-c', smoke],
        [sys.executable, '-c', recovery_smoke],
        [sys.executable, '-c', cloud_batch_smoke],
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
    print(f'{LOG_PREFIX} preflight passed: compile + semantic unit tests + continuous-field + CPU JAX/MuJoCo/XML + recovery + batched-cloud smoke tests', flush=True)


def run_rung(args, run_dir: Path, steps: int, flags: dict, *, seed: int, provenance: dict,
             resume: str = '', warm_start: str = '') -> tuple[int, str]:
    cmd=[sys.executable,str(TRAINING_PY),
         '--steps',str(steps),'--envs',str(args.envs),'--rollout',str(args.rollout),
         '--epochs',str(args.epochs),'--minibatches',str(args.minibatches),'--lr',str(args.lr),
         '--device',args.device,'--seed',str(seed),'--max-episode-steps',str(args.max_episode_steps),
         '--run-dir',str(run_dir),'--run-name',run_dir.name,
         '--checkpoint-every',str(args.checkpoint_every),'--log-every',str(args.log_every),
         '--eval-every-steps',str(args.eval_every_steps),'--eval-horizon',str(args.eval_horizon),
         '--eval-cohorts',str(args.eval_cohorts),'--final-eval-cohorts',str(args.final_eval_cohorts),
         '--obs-norm-anchor-steps',str(args.obs_norm_anchor_steps),
         '--hard-eval-every-steps',str(args.hard_eval_every_steps),
         '--cloud-eval-every-steps',str(args.cloud_eval_every_steps),
         '--cloud-probes-per-eval',str(args.cloud_probes_per_eval if args.stage in ('capability','trainer') else 0),
         '--cloud-final-probes',str(args.cloud_final_probes if args.stage in ('capability','trainer') else 0),
         '--cloud-envs',str(args.cloud_envs),'--cloud-horizon',str(args.cloud_horizon),
         '--cloud-batch-points',str(args.cloud_batch_points)]
    if args.profile_performance: cmd.append('--profile-performance')
    if resume: cmd += ['--resume',resume]
    elif warm_start:
        cmd += ['--warm-start-weights',warm_start]
        if args.seed_reference_curriculum and getattr(args, 'start', 'reference') != 'zero':
            cmd.append('--warm-start-curriculum')
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
            if (t.startswith(f'[emu_{VERSION}] step=') or 'fixed-eval step=' in t or 'capability-cloud step=' in t or 'AUTO-RECOVERY' in t or
                    'WARNING:' in t or 'Traceback' in t or 'training complete' in t):
                print('    '+t[:240],flush=True)
        rc=proc.wait()
    return rc, ''.join(tail[-50:]) if rc else ''


def resolve_imported_phase2(args) -> tuple[str, Path, int] | None:
    """Resolve a v5008 phase-2 checkpoint for a clean v5009 continuation.

    This is intentionally an import, not an in-place resume: v5009 gets a new
    run directory and evaluator semantics while preserving policy/optimizer/
    governor state from the actual v5008 checkpoint.
    """
    explicit = str(getattr(args, 'import_phase2_checkpoint', '') or '').strip()
    if not getattr(args, 'import_v5008_phase2', False) and not explicit:
        return None
    if explicit:
        ck = Path(explicit).expanduser().resolve()
        candidates = [ck]
    else:
        legacy_root = RAW_RUN_BASE / 'v5008' / 'rung_runs'
        candidates = sorted(legacy_root.glob(f'*{PHASE2_SUFFIX}/checkpoints/latest_mjx.pkl'))
    candidates = [c for c in candidates if c.exists()]
    if not candidates:
        raise SystemExit(f'{LOG_PREFIX} no importable v5008 phase-2 latest_mjx.pkl found. '
                         f'Use --import-phase2-checkpoint /path/to/latest_mjx.pkl if needed.')
    if len(candidates) > 1:
        raise SystemExit(f'{LOG_PREFIX} multiple v5008 phase-2 checkpoints found: '
                         + ', '.join(str(c) for c in candidates)
                         + '. Use --import-phase2-checkpoint explicitly.')
    ck = candidates[0]
    phase_dir = ck.parent.parent
    inferred = phase_dir.name[:-len(PHASE2_SUFFIX)] if phase_dir.name.endswith(PHASE2_SUFFIX) else ''
    chosen = args.phase2_branch if args.phase2_branch != 'auto' else inferred
    if chosen not in {b['name'] for b in build_branches()}:
        raise SystemExit(f'{LOG_PREFIX} could not infer a valid branch from {phase_dir.name!r}; '
                         f'use --phase2-branch explicitly.')
    seed = int(args.seed)
    run_meta = phase_dir / 'run_meta.json'
    if run_meta.exists():
        try:
            meta = json.loads(run_meta.read_text(encoding='utf-8'))
            seed = int((meta.get('args') or {}).get('seed', seed))
        except Exception:
            pass
    return chosen, ck, seed


def print_plan(branches: list[dict], args) -> None:
    print(f'{VERSION} stage={args.stage} start={args.start} branches={len(branches)} '
          f'eval_cohorts={args.eval_cohorts} final_eval_cohorts={args.final_eval_cohorts} profile={args.profile_performance}')
    if args.start == 'zero':
        if args.stage == 'trainer':
            print(f'  start policy:    genuine cold start -> {args.foundation_steps:,}-step trainer nursery (seed {args.seed})')
        else:
            print(f'  start policy:    cold start -> shared {args.foundation_steps:,}-step foundation ({args.foundation_name})')
    else:
        print(f'  start policy:    {args.base_checkpoint}')
        print(f'  seed frontier:   {args.seed_reference_curriculum}')
    if args.stage == 'profile':
        print(f'  profile steps:   {args.profile_steps:,} (one branch, no confirm/phase2)')
    elif args.stage == 'atlas':
        print(f'  atlas:           {args.atlas_envs} envs/point, {args.atlas_horizon} control steps/point; no training')
    elif args.stage == 'cloud':
        print(f'  cloud:           {args.cloud_explore_points}+{args.cloud_refine_points} sparse points, {args.cloud_envs} envs/point, {args.cloud_horizon} steps/point; no training')
    elif args.stage == 'trainer':
        print(f'  cold foundation: 0 -> {args.foundation_steps:,} steps, field observes in shadow mode')
        print(f'  trainer contest: {args.foundation_steps:,} -> {args.trainer_target_steps:,} absolute steps from the exact same policy/optimizer/governor/field checkpoint')
        print(f'  field:           21x15x25 command lattice; event-based competence + separate quality; 20M-step half-life')
        print(f'  held-out check:  paired calibration cloud every {args.cloud_eval_every_steps:,} steps; {args.cloud_probes_per_eval} requested / {args.cloud_final_probes} final')
        print(f'  independent cold confirmation: {args.trainer_confirm_zero}; optional winner extension: {args.trainer_extend_winner} -> {args.trainer_extended_steps:,}')
    else:
        print(f'  branch steps:    {args.steps_per_branch:,}')
        print(f'  capability mapping: control-gated paired cloud every {args.cloud_eval_every_steps:,} steps; request up to {args.cloud_probes_per_eval} routine / {args.cloud_final_probes} final; batch={args.cloud_batch_points} points')
        print('  final-boss ruler: dense physical-unit atlas; legacy fixed hard probes are regression-only')
    print(f'  raw runs:        {Path(args.run_root).expanduser()}')
    print(f'  tracked results: {Path(args.results_dir).expanduser()}')
    for i,b in enumerate(branches,1):
        print(f'  {i:2d}. {b["name"]:24s} [{b["tier"]:12s}] {b["rationale"]}')
    if args.stage == 'capability' and not args.skip_confirm:
        print(f'  confirm: top {args.confirm_top_k} families + reference control at seed +{args.confirm_seed_offset}; '
              f'third seed for top-two recovery tie < {args.tie_threshold:.3f}')
    if args.stage == 'capability' and not args.skip_phase2:
        print(f'  phase 2: selected family continued to {args.phase2_steps:,} absolute branch steps')


def main(argv=None) -> int:
    p=argparse.ArgumentParser(description=f'EMU {VERSION} performance hardening + focused capability matrix')
    p.add_argument('--stage', choices=['profile','atlas','cloud','capability','trainer'], default='profile',
                   help='trainer = v5012 cold-start self-mapping trainer qualification; profile/atlas/cloud/capability retain diagnostic compatibility')
    p.add_argument('--start', choices=['reference','zero','checkpoint'], default='reference',
                   help='reference uses protected v5007 120M policy; zero rebuilds from scratch; checkpoint uses --checkpoint')
    p.add_argument('--reference-checkpoint', default=str(DEFAULT_REFERENCE_CHECKPOINT),
                   help='protected exact-120M v5007 checkpoint used by --start reference')
    p.add_argument('--checkpoint', default='', help='checkpoint used by --start checkpoint')
    p.add_argument('--seed-reference-curriculum', action=argparse.BooleanOptionalAction, default=True,
                   help='when starting from reference/checkpoint, seed demonstrated frontier/difficulty scalars while keeping optimizer + governor histories fresh')
    p.add_argument('--foundation-steps',type=int,default=20_000_000)
    p.add_argument('--foundation-name',default='foundation20')
    p.add_argument('--rebuild-foundation',action='store_true')
    p.add_argument('--profile-steps',type=int,default=12_000_000)
    p.add_argument('--atlas-envs',type=int,default=1024)
    p.add_argument('--atlas-horizon',type=int,default=300)
    p.add_argument('--cloud-envs',type=int,default=256)
    p.add_argument('--cloud-horizon',type=int,default=250)
    p.add_argument('--cloud-eval-every-steps',type=int,default=20_000_000)
    p.add_argument('--cloud-batch-points',type=int,default=4)
    p.add_argument('--cloud-explore-points',type=int,default=100)
    p.add_argument('--cloud-refine-points',type=int,default=40)
    p.add_argument('--cloud-probes-per-eval',type=int,default=15,
                   help='capability stage: sparse cloud points added at each regular held-out evaluation')
    p.add_argument('--cloud-final-probes',type=int,default=50,
                   help='capability stage: sparse cloud points added at final forced evaluation')
    p.add_argument('--steps-per-branch',type=int,default=60_000_000)
    p.add_argument('--trainer-target-steps',type=int,default=120_000_000,
                   help='absolute step target for each 20M->120M trainer branch')
    p.add_argument('--trainer-confirm-zero',action=argparse.BooleanOptionalAction,default=True,
                   help='after comparing trainer branches, build a second independent cold-start foundation and run the winner to the same target')
    p.add_argument('--trainer-extend-winner',action=argparse.BooleanOptionalAction,default=False,
                   help='optionally continue the winning confirmed trainer to --trainer-extended-steps')
    p.add_argument('--trainer-extended-steps',type=int,default=220_000_000)
    p.add_argument('--phase2-steps',type=int,default=120_000_000)
    p.add_argument('--phase2-branch',default='auto',help='auto or explicit branch family')
    p.add_argument('--skip-phase2',action='store_true'); p.add_argument('--phase2-only',action='store_true')
    p.add_argument('--import-v5008-phase2', action='store_true',
                    help='v5009 transition: locate the existing v5008 *__phase2/latest_mjx.pkl, infer its branch + seed, and continue it under v5009 evaluation rules')
    p.add_argument('--import-phase2-checkpoint', default='',
                    help='explicit checkpoint for a cross-version phase-2 import; branch is inferred from the parent *__phase2 directory unless --phase2-branch is explicit')
    p.add_argument('--confirm-top-k',type=int,default=3)
    p.add_argument('--confirm-seed-offset',type=int,default=1009)
    p.add_argument('--third-seed-offset',type=int,default=2017)
    p.add_argument('--tie-threshold',type=float,default=0.030)
    p.add_argument('--skip-confirm',action='store_true')
    p.add_argument('--envs',type=int,default=2048); p.add_argument('--rollout',type=int,default=32)
    p.add_argument('--epochs',type=int,default=4); p.add_argument('--minibatches',type=int,default=8)
    p.add_argument('--lr',type=float,default=2e-4); p.add_argument('--max-episode-steps',type=int,default=700)
    p.add_argument('--checkpoint-every',type=int,default=20); p.add_argument('--log-every',type=int,default=10)
    p.add_argument('--eval-every-steps',type=int,default=30_000_000); p.add_argument('--eval-horizon',type=int,default=350)
    p.add_argument('--eval-cohorts',type=int,default=1); p.add_argument('--final-eval-cohorts',type=int,default=2)
    p.add_argument('--hard-eval-every-steps',type=int,default=0)
    p.add_argument('--obs-norm-anchor-steps',type=int,default=20_000_000)
    p.add_argument('--profile-performance', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--seed',type=int,default=733)
    p.add_argument('--device',choices=['auto','cpu','cuda','gpu'],default='cuda'); p.add_argument('--allow-cpu-mjx',action='store_true')
    p.add_argument('--run-root', default=str(DEFAULT_RUN_ROOT), help='external heavyweight run directory for this VERSION')
    p.add_argument('--results-dir', default=str(DEFAULT_RESULTS_DIR), help='small Git-tracked evidence directory for this VERSION')
    p.add_argument('--only',default='')
    p.add_argument('--skip-preflight',action='store_true'); p.add_argument('--list',action='store_true')
    args=p.parse_args(argv)
    if args.stage == 'trainer':
        # Trainer qualification is intentionally checkpoint-independent.  The
        # shared 20M ancestor is built inside this invocation from random
        # weights; protected references remain irrelevant to the experiment.
        args.start = 'zero'
        args.phase2_only = False
        args.skip_phase2 = True
        args.skip_confirm = True
        args.foundation_steps = 20_000_000
        # The continuously updated field replaces frequent legacy examinations.
        # Keep only one mid-run regression check plus the forced final; the
        # smaller held-out calibration exam remains every 20M.
        args.eval_every_steps = 60_000_000
        args.cloud_eval_every_steps = 40_000_000
        args.cloud_probes_per_eval = min(int(args.cloud_probes_per_eval), 10)

    # Resolve the common ancestor explicitly. The zero route intentionally
    # remains first-class; protected checkpoints accelerate experiments but are
    # never allowed to become an architectural dependency.
    if args.start == 'reference':
        args.base_checkpoint = str(Path(args.reference_checkpoint).expanduser())
    elif args.start == 'checkpoint':
        if not args.checkpoint:
            raise SystemExit('--start checkpoint requires --checkpoint /path/model.pkl')
        args.base_checkpoint = str(Path(args.checkpoint).expanduser())
    else:
        args.base_checkpoint = ''

    imported_phase2 = None if args.stage == 'trainer' else resolve_imported_phase2(args)
    if imported_phase2 is not None:
        args.phase2_only = True
        args.phase2_branch = imported_phase2[0]
        args.skip_confirm = True
        args.start = 'checkpoint'
        args.base_checkpoint = str(imported_phase2[1])
        print(f'{LOG_PREFIX} cross-version import: {imported_phase2[0]} from {imported_phase2[1]} seed={imported_phase2[2]}', flush=True)

    all_branches = ([profile_branch()] if args.stage == 'profile' else
                    ([] if args.stage in ('atlas','cloud') else
                     (build_trainer_branches() if args.stage == 'trainer' else build_branches())))
    branches=all_branches
    if args.only:
        wanted={x.strip() for x in args.only.split(',') if x.strip()}; branches=[b for b in all_branches if b['name'] in wanted]
        missing=wanted-{b['name'] for b in branches}
        if missing: raise SystemExit(f'Unknown branch(es): {sorted(missing)}')
    if args.stage in ('profile','atlas','cloud'):
        args.skip_confirm = True
        args.skip_phase2 = True
        args.hard_eval_every_steps = 0
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
            print(f'{LOG_PREFIX} WARNING: executable source files have uncommitted changes. Commit before a serious ladder.', flush=True)
    else:
        print(f'{LOG_PREFIX} WARNING: this folder is not yet a Git repository; commit provenance will be unavailable.', flush=True)
    print(f'{LOG_PREFIX} raw runs -> {run_root}', flush=True)
    print(f'{LOG_PREFIX} tracked evidence -> {results_dir}', flush=True)

    if args.foundation_steps <= 0: raise SystemExit('--foundation-steps must be > 0')
    if args.start != 'zero' and not args.phase2_only and not Path(args.base_checkpoint).exists():
        raise SystemExit(f'{LOG_PREFIX} start checkpoint does not exist: {args.base_checkpoint}\n'
                         f'Expected protected v5007 reference at {DEFAULT_REFERENCE_CHECKPOINT} for --start reference.')
    if not args.skip_preflight: run_preflight(run_root)

    if args.stage == 'atlas':
        atlas_out = run_root / 'atlas_reference'
        if atlas_out.exists(): shutil.rmtree(atlas_out)
        atlas_out.mkdir(parents=True, exist_ok=True)
        cmd=[sys.executable, str(HERE/'atlas.py'), '--checkpoint', args.base_checkpoint,
             '--out-dir', str(atlas_out), '--device', args.device,
             '--envs', str(args.atlas_envs), '--horizon', str(args.atlas_horizon), '--seed', str(args.seed)]
        if args.allow_cpu_mjx: cmd.append('--allow-cpu-mjx')
        print(f'{LOG_PREFIX} building physical capability/recovery atlas...', flush=True)
        rc=subprocess.call(cmd,cwd=HERE)
        if rc: raise SystemExit(f'{LOG_PREFIX} atlas failed with exit code {rc}')
        for name in ('atlas.md','capability_atlas.csv','capability_atlas.json'):
            src=atlas_out/name
            if src.exists(): shutil.copy2(src, results_dir/name)
        print(f'{LOG_PREFIX} atlas complete. Tracked result: {results_dir / "atlas.md"}', flush=True)
        return 0

    if args.stage == 'cloud':
        cloud_out = run_root / 'cloud_reference'
        if cloud_out.exists(): shutil.rmtree(cloud_out)
        cloud_out.mkdir(parents=True, exist_ok=True)
        cmd=[sys.executable, str(HERE/'capability_cloud.py'), '--checkpoint', args.base_checkpoint,
             '--out-dir', str(cloud_out), '--device', args.device, '--envs', str(args.cloud_envs),
             '--horizon', str(args.cloud_horizon), '--explore-points', str(args.cloud_explore_points),
             '--refine-points', str(args.cloud_refine_points), '--seed', str(args.seed)]
        if args.allow_cpu_mjx: cmd.append('--allow-cpu-mjx')
        print(f'{LOG_PREFIX} building sparse combined-command capability cloud...', flush=True)
        rc=subprocess.call(cmd,cwd=HERE)
        if rc: raise SystemExit(f'{LOG_PREFIX} capability cloud failed with exit code {rc}')
        for name in ('capability_cloud.md','capability_cloud.csv','capability_cloud.json'):
            src=cloud_out/name
            if src.exists(): shutil.copy2(src, results_dir/name)
        print(f'{LOG_PREFIX} cloud complete. Tracked result: {results_dir / "capability_cloud.md"}', flush=True)
        return 0

    meta_path=run_root/'run_meta.json'; summary_path=run_root/'ladder_summary.csv'
    try: meta=json.loads(meta_path.read_text()) if meta_path.exists() else {}
    except Exception: meta={}
    meta.setdefault('branches',{}); meta.setdefault('errors',[]); meta['version']=VERSION; meta['git']=provenance; meta['args']=vars(args)
    meta['raw_run_root']=str(run_root); meta['tracked_results_dir']=str(results_dir); meta['start_mode']=args.start
    if imported_phase2 is not None:
        meta['cross_version_import']={'from_version':'v5008','branch':imported_phase2[0],
                                      'checkpoint':str(imported_phase2[1]),'seed':int(imported_phase2[2])}
    meta['reference_checkpoint']=args.base_checkpoint if args.start != 'zero' else ''
    rows=read_csv(summary_path) if summary_path.exists() else []
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

    if args.stage == 'trainer':
        # ---------------------------------------------------------------
        # v5012 trainer qualification: one genuine 0->20M ancestor, then
        # three 20->120M teaching strategies from the exact same checkpoint.
        # The fork uses --resume, not warm-start, so optimizer, normalizer,
        # governor AND capability-field state are identical at the split.
        # ---------------------------------------------------------------
        foundation_name=f'trainer_foundation20_seed{args.seed}'
        foundation_dir=rung_root/foundation_name
        foundation_dir.mkdir(parents=True,exist_ok=True)
        rationale=('Cold-start v5012 nursery. Random actor/critic, fresh optimizer/governor; '
                   'continuous capability field observes in shadow mode through 20M.')
        (foundation_dir/'rationale.txt').write_text(rationale+'\n',encoding='utf-8')
        latest=foundation_dir/'checkpoints'/'latest_mjx.pkl'
        fsm=summarize_rung(foundation_dir)
        if int(fsm.get('final_step',-1) or -1) < args.foundation_steps or not latest.exists():
            print(f'\n{LOG_PREFIX} === TRAINER FOUNDATION seed={args.seed}: 0 -> {args.foundation_steps:,} ===',flush=True)
            t0=time.time(); rc,tail=run_rung(args,foundation_dir,args.foundation_steps,trainer_foundation_flags(),
                                              seed=args.seed,provenance=provenance,
                                              resume=str(latest) if latest.exists() else '')
            record(foundation_name,{'name':foundation_name,'flags':trainer_foundation_flags(),'tier':'foundation','rationale':rationale},
                   'trainer_foundation',rc==0,time.time()-t0,tail,args.seed)
            if rc: raise SystemExit(f'{LOG_PREFIX} trainer foundation failed.\n{tail}')
        else:
            print(f'{LOG_PREFIX} reusing completed trainer foundation {latest}',flush=True)
        if not latest.exists(): raise SystemExit(f'{LOG_PREFIX} missing exact foundation checkpoint {latest}')
        meta['foundation']={'name':foundation_name,'steps':args.foundation_steps,'checkpoint':str(latest),'seed':args.seed,
                            'flags':flags_to_cli(trainer_foundation_flags()),'fork_semantics':'exact resume including optimizer, governor, normalizer and capability field'}

        for i,branch in enumerate(branches,1):
            name=branch['name']; d=rung_root/name; d.mkdir(parents=True,exist_ok=True)
            (d/'rationale.txt').write_text(branch['rationale']+'\n',encoding='utf-8')
            own_latest=d/'checkpoints'/'latest_mjx.pkl'
            sm=summarize_rung(d)
            if int(sm.get('final_step',-1) or -1) >= args.trainer_target_steps and own_latest.exists():
                print(f'{LOG_PREFIX} trainer branch already complete: {name} step={sm.get("final_step")}',flush=True)
                continue
            source=own_latest if own_latest.exists() else latest
            print(f'\n{LOG_PREFIX} === TRAINER {i}/{len(branches)}: {name} -> {args.trainer_target_steps:,} ===\n{LOG_PREFIX} {branch["rationale"]}',flush=True)
            t0=time.time(); rc,tail=run_rung(args,d,args.trainer_target_steps,branch['flags'],seed=args.seed,
                                              provenance=provenance,resume=str(source))
            record(name,branch,'trainer_20_to_120',rc==0,time.time()-t0,tail,args.seed)
            if rc: print(f'{LOG_PREFIX} trainer branch {name} failed; continuing.\n{tail}',flush=True)

        control=summarize_rung(rung_root/'trainer_shadow')
        ranked=[]
        for b in branches:
            sm=summarize_rung(rung_root/b['name'])
            if sm.get('durability_score',-1e9)>-1e8:
                ranked.append((trainer_rank(sm,control),b,sm))
        ranked.sort(key=lambda x:x[0],reverse=True)
        meta['trainer_ranking']=[{'name':b['name'],'rank_tuple':list(rk),'summary':sm} for rk,b,sm in ranked]
        print(f'\n{LOG_PREFIX} trainer ranking (eligible, heldout-recovery, worst-dir, baseline-command, competent-edges, known-edges, competent-nodes, durability):',flush=True)
        for rk,b,_ in ranked: print(f'    {b["name"]}: {rk}',flush=True)
        winner=ranked[0][1] if ranked else None
        if winner is not None:
            meta['trainer_selected']=winner['name']
            meta['trainer_selection']='regression-safe lexicographic scheduler: held-out recovery first; continuous field coverage is training-derived tie/efficiency evidence, not an exam score'

        confirm_dir=None
        if winner is not None and args.trainer_confirm_zero:
            seed2=args.seed+args.confirm_seed_offset
            qf_name=f'trainer_foundation20_seed{seed2}'
            qf=rung_root/qf_name; qf.mkdir(parents=True,exist_ok=True)
            (qf/'rationale.txt').write_text(f'Independent cold-start foundation for winner confirmation seed {seed2}.\n',encoding='utf-8')
            qlatest=qf/'checkpoints'/'latest_mjx.pkl'
            qsm=summarize_rung(qf)
            if int(qsm.get('final_step',-1) or -1)<args.foundation_steps or not qlatest.exists():
                print(f'\n{LOG_PREFIX} === INDEPENDENT ZERO FOUNDATION seed={seed2}: 0 -> {args.foundation_steps:,} ===',flush=True)
                t0=time.time(); rc,tail=run_rung(args,qf,args.foundation_steps,trainer_foundation_flags(),seed=seed2,provenance=provenance,
                                                  resume=str(qlatest) if qlatest.exists() else '')
                record(qf_name,{'name':qf_name,'flags':trainer_foundation_flags(),'tier':'foundation','rationale':'Independent zero-start trainer confirmation foundation.'},
                       'trainer_confirm_foundation',rc==0,time.time()-t0,tail,seed2)
                if rc: raise SystemExit(f'{LOG_PREFIX} independent cold foundation failed.\n{tail}')
            confirm_name=winner['name']+'__zero_confirm'; confirm_dir=rung_root/confirm_name; confirm_dir.mkdir(parents=True,exist_ok=True)
            (confirm_dir/'rationale.txt').write_text(winner['rationale']+f'\nIndependent complete cold-start confirmation seed {seed2}.\n',encoding='utf-8')
            clatest=confirm_dir/'checkpoints'/'latest_mjx.pkl'; csm=summarize_rung(confirm_dir)
            if int(csm.get('final_step',-1) or -1)<args.trainer_target_steps or not clatest.exists():
                source=clatest if clatest.exists() else qlatest
                print(f'\n{LOG_PREFIX} === WINNER ZERO CONFIRM: {winner["name"]} seed={seed2} -> {args.trainer_target_steps:,} ===',flush=True)
                t0=time.time(); rc,tail=run_rung(args,confirm_dir,args.trainer_target_steps,winner['flags'],seed=seed2,provenance=provenance,resume=str(source))
                record(confirm_name,winner,'trainer_zero_confirm',rc==0,time.time()-t0,tail,seed2)
                if rc: print(f'{LOG_PREFIX} winner zero confirmation failed.\n{tail}',flush=True)
            meta['trainer_zero_confirmation']={'winner':winner['name'],'seed':seed2,'foundation':str(qlatest),'run':confirm_name}

        if winner is not None and args.trainer_extend_winner:
            source_dir=confirm_dir if confirm_dir is not None and (confirm_dir/'checkpoints'/'latest_mjx.pkl').exists() else rung_root/winner['name']
            source_ck=source_dir/'checkpoints'/'latest_mjx.pkl'
            source_seed=(args.seed+args.confirm_seed_offset) if source_dir==confirm_dir else args.seed
            ext_name=winner['name']+'__extended220'; ext=rung_root/ext_name; ext.mkdir(parents=True,exist_ok=True)
            ext_latest=ext/'checkpoints'/'latest_mjx.pkl'; esm=summarize_rung(ext)
            if int(esm.get('final_step',-1) or -1)<args.trainer_extended_steps or not ext_latest.exists():
                source=ext_latest if ext_latest.exists() else source_ck
                print(f'\n{LOG_PREFIX} === OPTIONAL ASYMPTOTIC EXTENSION: {winner["name"]} -> {args.trainer_extended_steps:,} ===',flush=True)
                t0=time.time(); rc,tail=run_rung(args,ext,args.trainer_extended_steps,winner['flags'],seed=source_seed,provenance=provenance,resume=str(source))
                record(ext_name,winner,'trainer_extended',rc==0,time.time()-t0,tail,source_seed)
            meta['trainer_extension']={'run':ext_name,'target_steps':args.trainer_extended_steps}

        meta['wall_seconds_this_invocation']=round(time.time()-start_all,2)
        meta['last_updated']=time.strftime('%Y-%m-%d %H:%M:%S')
        meta_path.write_text(json.dumps(meta,indent=2,default=str),encoding='utf-8')
        safe_report(run_root,results_dir)
        print(f'\n{LOG_PREFIX} trainer qualification complete. Tracked result: {results_dir / "results.md"}',flush=True)
        return 0

    # --start zero: rebuild/reuse one shared basic locomotor, then compare all
    # branches from exactly that checkpoint. Reference/checkpoint starts bypass
    # this phase and preserve the zero route for periodic bootstrap sanity tests.
    if not args.phase2_only and args.start == 'zero':
        foundation_dir = rung_root / args.foundation_name
        ckpt = preferred_checkpoint(foundation_dir)
        foundation_summary = summarize_rung(foundation_dir) if foundation_dir.exists() else {}
        foundation_complete = int(foundation_summary.get('final_step', -1) or -1) >= args.foundation_steps and ckpt is not None
        if args.rebuild_foundation and foundation_dir.exists():
            print(f'{LOG_PREFIX} rebuilding shared foundation: deleting {foundation_dir}', flush=True)
            shutil.rmtree(foundation_dir); ckpt=None; foundation_complete=False
        if not foundation_complete:
            foundation_dir.mkdir(parents=True, exist_ok=True)
            rationale=(f'Shared {args.foundation_steps:,}-step v5008 cold-start locomotion foundation. '
                       'Used only for --start zero regression/capability runs.')
            (foundation_dir/'rationale.txt').write_text(rationale+'\n',encoding='utf-8')
            resume_ckpt=preferred_checkpoint(foundation_dir)
            print(f'\n{LOG_PREFIX} === zero-start foundation -> {args.foundation_steps:,} ===',flush=True)
            t0=time.time(); rc,tail=run_rung(args,foundation_dir,args.foundation_steps,foundation_flags(),seed=args.seed,
                                              provenance=provenance,resume=str(resume_ckpt) if resume_ckpt else '')
            record(args.foundation_name,{'name':args.foundation_name,'flags':foundation_flags(),'tier':'foundation','rationale':rationale},
                   'foundation',rc==0,time.time()-t0,tail,args.seed)
            if rc: raise SystemExit(f'{LOG_PREFIX} shared foundation failed.\n{tail}')
            ckpt=preferred_checkpoint(foundation_dir)
        if ckpt is None: raise SystemExit(f'{LOG_PREFIX} no reusable foundation checkpoint found')
        args.base_checkpoint=str(ckpt)
        meta['foundation']={'name':args.foundation_name,'steps':args.foundation_steps,'checkpoint':str(ckpt),'seed':args.seed,
                            'flags':flags_to_cli(foundation_flags())}
    elif not args.phase2_only:
        print(f'{LOG_PREFIX} using protected/common warm start: {args.base_checkpoint}',flush=True)

    if not args.phase2_only:
        run_steps = args.profile_steps if args.stage == 'profile' else args.steps_per_branch
        for i,branch in enumerate(branches,1):
            name='perf_probe2' if args.stage=='profile' else branch['name']; d=rung_root/name
            if d.exists(): shutil.rmtree(d)
            d.mkdir(parents=True); (d/'rationale.txt').write_text(branch['rationale']+'\n',encoding='utf-8')
            print(f'\n{LOG_PREFIX} === branch {i}/{len(branches)}: {name} ===\n{LOG_PREFIX} {branch["rationale"]}',flush=True)
            t0=time.time(); rc,tail=run_rung(args,d,run_steps,branch['flags'],seed=args.seed,provenance=provenance,warm_start=args.base_checkpoint)
            record(name,branch,'profile' if args.stage=='profile' else 'phase1',rc==0,time.time()-t0,tail,args.seed)
            if rc: print(f'{LOG_PREFIX} branch {name} failed; continuing.\n{tail}',flush=True)

        if args.stage == 'capability' and not args.skip_confirm and args.confirm_top_k>0:
            reference_sm=summarize_rung(rung_root/'reference_continue')
            scored=[]
            for b in branches:
                sm=summarize_rung(rung_root/b['name'])
                if sm['durability_score']>-1e8:
                    scored.append((campaign_rank(sm, reference_sm), b, sm))
            ranked=sorted(scored,key=lambda x:x[0],reverse=True)
            print(f"{LOG_PREFIX} confirmation ranking (eligible, recovery, worst-dir, baseline, durability):", flush=True)
            for rk,b,sm in ranked:
                print(f"    {b['name']}: {rk}", flush=True)
            confirm_branches=[b for _,b,_ in ranked[:args.confirm_top_k]]
            ref_branch=next((b for b in branches if b['name']=='reference_continue'), None)
            if ref_branch is not None and all(b['name'] != 'reference_continue' for b in confirm_branches):
                confirm_branches.append(ref_branch)
            for branch in confirm_branches:
                seed2=args.seed+args.confirm_seed_offset; name=branch['name']+CONFIRM_SUFFIX; d=rung_root/name
                if d.exists(): shutil.rmtree(d)
                d.mkdir(parents=True); (d/'rationale.txt').write_text(branch['rationale']+f'\nConfirmation seed {seed2}.\n',encoding='utf-8')
                print(f'\n{LOG_PREFIX} === confirmation: {branch["name"]} seed={seed2} ===',flush=True)
                t0=time.time(); rc,tail=run_rung(args,d,args.steps_per_branch,branch['flags'],seed=seed2,provenance=provenance,warm_start=args.base_checkpoint)
                record(name,branch,'confirm',rc==0,time.time()-t0,tail,seed2)

            # If the top two family means are practically tied, buy one more
            # seed for each instead of pretending a 0.002 PPO wobble is causal.
            provisional={}
            for b in branches:
                sms=[]
                for rn in (b['name'],b['name']+CONFIRM_SUFFIX):
                    sm=summarize_rung(rung_root/rn)
                    if sm['durability_score']>-1e8: sms.append(sm)
                if sms:
                    provisional[b['name']]={
                        'durability_score': _mean([float(x['durability_score']) for x in sms]),
                        'cloud_baseline_competent_frac': _mean([float(x['cloud_baseline_competent_frac']) for x in sms]),
                        'cloud_valid_recovery_mean': _mean([float(x['cloud_valid_recovery_mean']) for x in sms]),
                        'cloud_valid_worst_direction_mean': _mean([float(x['cloud_valid_worst_direction_mean']) for x in sms]),
                        'cloud_valid_n': int(round(_mean([float(x['cloud_valid_n']) for x in sms]))),
                    }
            ref_prov=provisional.get('reference_continue', reference_sm)
            top2=sorted(provisional.items(),key=lambda kv:campaign_rank(kv[1],ref_prov),reverse=True)[:2]
            if len(top2)==2 and abs(float(top2[0][1].get('cloud_valid_recovery_mean',-1)) -
                                     float(top2[1][1].get('cloud_valid_recovery_mean',-1))) < args.tie_threshold:
                for family,_ in top2:
                    branch=next(b for b in branches if b['name']==family)
                    seed3=args.seed+args.third_seed_offset; name=family+THIRD_SEED_SUFFIX; d=rung_root/name
                    if d.exists(): shutil.rmtree(d)
                    d.mkdir(parents=True); (d/'rationale.txt').write_text(branch['rationale']+f'\nTie-break confirmation seed {seed3}.\n',encoding='utf-8')
                    print(f'\n{LOG_PREFIX} === tie-break confirmation: {family} seed={seed3} ===',flush=True)
                    t0=time.time(); rc,tail=run_rung(args,d,args.steps_per_branch,branch['flags'],seed=seed3,provenance=provenance,warm_start=args.base_checkpoint)
                    record(name,branch,'confirm2',rc==0,time.time()-t0,tail,seed3)

    lookup={b['name']:b for b in build_branches()}
    family_scores={}
    family_capability={}
    if args.stage == 'capability' or args.phase2_only:
        for b in build_branches():
            sms=[]
            for rn in (b['name'],b['name']+CONFIRM_SUFFIX,b['name']+THIRD_SEED_SUFFIX):
                sm=summarize_rung(rung_root/rn)
                if sm['durability_score']>-1e8: sms.append(sm)
            if sms:
                family_scores[b['name']]=_mean([float(x['durability_score']) for x in sms])
                family_capability[b['name']]={
                    'durability_score': family_scores[b['name']],
                    'cloud_baseline_competent_frac': _mean([float(x['cloud_baseline_competent_frac']) for x in sms]),
                    'cloud_valid_recovery_mean': _mean([float(x['cloud_valid_recovery_mean']) for x in sms]),
                    'cloud_valid_worst_direction_mean': _mean([float(x['cloud_valid_worst_direction_mean']) for x in sms]),
                    'cloud_valid_n': int(round(_mean([float(x['cloud_valid_n']) for x in sms]))),
                    'seeds': len(sms),
                }
        meta['family_scores']=family_scores
        meta['family_capability']=family_capability

    if args.stage == 'capability' and not args.skip_phase2:
        chosen=args.phase2_branch
        if chosen=='auto':
            ref_family=family_capability.get('reference_continue')
            chosen=max(family_capability,key=lambda n:campaign_rank(family_capability[n],ref_family)) if family_capability else ''
        if chosen not in lookup:
            print(f'{LOG_PREFIX} phase 2 skipped: no valid selected branch family ({chosen!r}).',flush=True)
        else:
            branch=lookup[chosen]; sources=[]
            imported = imported_phase2 if 'imported_phase2' in locals() else None
            if imported is not None:
                source_name=f'v5008_import::{imported[0]}'
                ckpt=imported[1]
                sources=[((1,1,1,1,1), source_name, ckpt)]
            else:
                for rn in (chosen,chosen+CONFIRM_SUFFIX,chosen+THIRD_SEED_SUFFIX):
                    d=rung_root/rn; sm=summarize_rung(d)
                    # Stage 5 ranks the FINAL paired capability panel, so phase 2
                    # must continue the exact terminal state that produced that panel.
                    # Falling back to an earlier best-broad checkpoint would break
                    # selection/checkpoint provenance.
                    ck=d/'checkpoints'/'latest_mjx.pkl'
                    if not ck.exists(): ck=d/'checkpoints'/'best_broad_eval.pkl'
                    if not ck.exists(): ck=d/'checkpoints'/'best_v5008_eval.pkl'
                    if ck.exists() and sm['durability_score']>-1e8:
                        ref_sm=summarize_rung(rung_root/'reference_continue')
                        sources.append((campaign_rank(sm,ref_sm),rn,ck))
            if sources:
                _,source_name,ckpt=max(sources,key=lambda x:x[0])
                meta['phase2_selected']=chosen; meta['phase2_source_run']=source_name
                meta['phase2_selection']='regression-safe paired capability ranking: preserve durability + undisturbed command competence, then prefer held-out recovery and worst-direction recovery across available seeds'
                name2=chosen+PHASE2_SUFFIX; d2=rung_root/name2
                source_seed=(int(imported_phase2[2]) if imported_phase2 is not None else int(meta.get('branches',{}).get(source_name,{}).get('seed',args.seed)))
                existing_latest=d2/'checkpoints'/'latest_mjx.pkl'
                if existing_latest.exists():
                    resume_ckpt=existing_latest
                    existing_summary=summarize_rung(d2)
                    existing_step=int(existing_summary.get('final_step',-1) or -1)
                    print(f'\n{LOG_PREFIX} === phase 2 resume-in-place: {chosen} from existing step~{existing_step:,} -> {args.phase2_steps:,} ===',flush=True)
                else:
                    if d2.exists(): shutil.rmtree(d2)
                    d2.mkdir(parents=True,exist_ok=True)
                    (d2/'rationale.txt').write_text(
                        f'v5009 continuation of {chosen} from {source_name} checkpoint to {args.phase2_steps:,} absolute branch steps under the v5009 evaluator-throughput rules.\n',encoding='utf-8')
                    resume_ckpt=ckpt
                    print(f'\n{LOG_PREFIX} === phase 2: {chosen} via {source_name} -> {args.phase2_steps:,} ===',flush=True)
                t0=time.time(); rc,tail=run_rung(args,d2,args.phase2_steps,branch['flags'],seed=source_seed,provenance=provenance,resume=str(resume_ckpt))
                record(name2,branch,'phase2',rc==0,time.time()-t0,tail,source_seed)
            else:
                print(f'{LOG_PREFIX} phase 2 skipped: no checkpoint for {chosen}.',flush=True)

    meta['wall_seconds_this_invocation']=round(time.time()-start_all,2); meta['last_updated']=time.strftime('%Y-%m-%d %H:%M:%S')
    meta_path.write_text(json.dumps(meta,indent=2,default=str),encoding='utf-8'); safe_report(run_root, results_dir)
    print(f'\n{LOG_PREFIX} complete. Tracked result: {results_dir / "results.md"}',flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
