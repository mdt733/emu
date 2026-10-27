"""EMU single-run reports, training telemetry, and held-out evidence.

Training and examiner facts are kept separate. No Python/JAX training is
invoked when making a report; checkpoints are never packaged.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import zipfile
from pathlib import Path

def _last_finite_metrics(path: Path) -> dict:
    if not path.is_file():
        return {}
    with path.open(newline='',encoding='utf-8-sig') as stream:
        rows=list(csv.DictReader(stream))
    def step(row):
        try:
            value=float(row.get('step','nan'))
            return value if math.isfinite(value) else -1.
        except (TypeError,ValueError):
            return -1.
    return max(rows,key=step,default={})

def _read_optional_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        value=json.loads(path.read_text(encoding='utf-8'))
        return value if isinstance(value,dict) else {}
    except (OSError,UnicodeError,ValueError):
        return {}

def write_report(root: Path, *, bundle: bool = True) -> list[Path]:
    """Report ONE source-pinned run, not a superseded four-arm tournament.

    Raw steps, motor work and physical contact counts are not independent
    walking competence. Fixed-policy examination is a separate artefact.
    """
    root=Path(root).expanduser().resolve()
    row=_last_finite_metrics(root/'metrics.csv')
    if not row:
        raise ValueError(f'No metrics.csv with training data in {root}')
    cfg=_read_optional_json(root/'training_config.json')
    meta=_read_optional_json(root/'run_meta.json')
    source=(meta.get('source_provenance') or {})
    keys=(
        ('reward_mean','Training reward'),
        ('fall_rate','Completed-episode fall fraction'),
        ('vx_mean','Achieved mean vx (m/s)'),
        ('cmd_vx_mean','Requested mean vx (m/s)'),
        ('control_shaft_j_per_m','Control mechanical J/m (proxy)'),
        ('terrain_attempts_up','Step-up attempts'),
        ('terrain_successes_up','Supported step-up successes'),
        ('terrain_attempts_down','Step-down attempts'),
        ('terrain_successes_down','Supported step-down successes'),
        ('physical_slope_deg_max','Sampled real slope max (deg)'),
    )
    values={label:row.get(key,'') for key,label in keys if row.get(key,'') not in ('',None)}
    step=row.get('step','?')
    lines=[
        '# EMU — single-run observation report','',
        f"- Run: `{root.name}`",
        f"- Source SHA: `{source.get('git_commit','unavailable')}`",
        f"- Recorded environment steps: {step}",
        f"- Observation contract: {cfg.get('observation_schema','unknown')}",
        f"- Unified specialists: {cfg.get('unified_capability_focus',False)}",
        '','## Last recorded on-policy measurements','',
        '| Measurement | Value |','| --- | ---: |',
        *[f'| {label} | {value} |' for label,value in values.items()],
        '','These are training-rollout observations, not held-out walking scores.',
        'A requested velocity is not an achieved velocity. Mechanical J/m is',
        'not verified battery consumption. Successful steps require actual',
        'foot-supported upper-platform contact; compare direction-matched exams.',
        '','## Independent evidence','',
        'Inspect fixed_eval.csv and separate frozen physical terrain examinations',
        'before increasing difficulty or promoting a policy.',''
    ]
    report={'run_dir':str(root),'step':step,'training_config':cfg,
            'source_provenance':source,'latest_metrics':row,
            'independent_exam_required':True}
    txt=root/'results.md'
    data=root/'results_data.json'
    txt.write_text('\n'.join(lines),encoding='utf-8')
    data.write_text(json.dumps(report,indent=2,sort_keys=True),encoding='utf-8')
    outputs=[txt,data]
    if bundle:
        dest=root/'results_bundle.zip'; tmp=root/'results_bundle.tmp'
        with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for name in ('results.md','results_data.json','metrics.csv',
                         'training_config.json','run_meta.json','fixed_eval.csv',
                         'envelope_summary.json','capability_evidence.json',
                         'capability_field_summary.json','performance.json'):
                path=root/name
                if path.is_file() and not path.is_symlink():
                    z.write(path,arcname=name)
        tmp.replace(dest)
        outputs.append(dest)
    return outputs

def main(argv=None) -> int:
    parser=argparse.ArgumentParser(description='Read-only single-run report or held-out comparison')
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--root',type=Path)
    mode.add_argument('--compare-fixed',type=Path,nargs=2,
                      metavar=('REFERENCE_CSV','CANDIDATE_CSV'))
    parser.add_argument('--no-bundle',action='store_true')
    parser.add_argument('--reference-protocol')
    parser.add_argument('--candidate-protocol')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args(argv)
    if args.root:
        for path in write_report(args.root,bundle=not args.no_bundle):
            print(path)
        return 0
    if not args.reference_protocol or not args.candidate_protocol:
        parser.error('--compare-fixed requires two matching exam protocol fingerprints')
    reference,candidate=args.compare_fixed
    a,b=read_csv_rows(reference),read_csv_rows(candidate)
    if not a or not b:
        raise ValueError('no fixed-eval evidence in one or both inputs')
    result=compare_fixed_exam_evidence(a[-1],b[-1],
                reference_protocol=args.reference_protocol,
                candidate_protocol=args.candidate_protocol)
    if args.output:
        write_json(args.output,result)
        print(args.output)
    else:
        print(json.dumps(result,indent=2,sort_keys=True))
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
                wall_fps: float | None = None, nan_grad_skips: int = 0,
                *, slope_deg: float = 0.0, contact_terrain_radius: float = 0.0) -> str:
    """Readable first line; detailed diagnostics remain in the run's console.log.

    The \u201cstep\u201d height is an exposure ceiling, not the sampled ledge height.
    Shaft J/m is simulated positive mechanical work, not battery energy.
    """
    if 'slope_max_deg' in report:
        # Real-world sampled slope, not a static compiled terrain setting.
        # The fraction is the ACTUAL physical slant-world occupancy seen.
        slope = (f"≤{report['slope_max_deg']:.1f}°"
                 f"({100*row.get('physical_slope_world_fraction',0.):.0f}%)")
    elif contact_terrain_radius > 0.0:
        slope = f"≤{5.0*contact_terrain_radius:.1f}°"
    else:
        slope = f"{slope_deg:.1f}°"
    exposure = 100.0 * float(report.get('terrain_exposure', 0.0))
    ceiling_cm = 100.0 * float(report.get('terrain_height_m', 0.0))
    jpm = float(row.get('shaft_j_per_m_forward', -1.0))
    efficiency = f"{jpm:.1f}" if math.isfinite(jpm) and jpm >= 0.0 else "–"
    # Make the physical asymmetry visible during GPU training, not only in
    # offline CSV reports. Omit fractions when a direction has no completed
    # terrain episodes in this rollout.
    def directional_course(label):
        attempts = int(row.get(f'terrain_attempts_{label}', 0.))
        passes = int(row.get(f'terrain_successes_{label}', 0.))
        return f"{passes}/{attempts}" if attempts else "-"
    terrain_pair = (
        f" up={directional_course('up')} down={directional_course('down')}"
        if float(row.get('terrain_frames_up', 0.)) + float(row.get('terrain_frames_down', 0.)) > 0.
        else ""
    )
    line1 = (
        f"[{version}] step={step:,} rew={reward:+.3f} "
        f"len={row['ep_len_mean']:.0f} fall={100*row['fall_rate']:.1f}% "
        f"fps={(wall_fps if wall_fps is not None else train_fps):.0f} "
        f"vx=[{report.get('frontier_vx_lo',0.0):+.2f},{report.get('frontier_vx_hi',0.0):+.2f}] "
        f"act/cmd={row['vx_mean']:+.2f}/{row['cmd_vx_mean']:+.2f} "
        f"slope={slope} step≤{ceiling_cm:.1f}cm({exposure:.0f}%) "
        f"res={report['resilience_difficulty']:.2f} shaft≈{efficiency}J/m"
        f"{terrain_pair}"
        f"{' NAN-SKIPS='+str(nan_grad_skips) if nan_grad_skips else ''}"
    )
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
    # A concise physical lesson pulse belongs in the live terminal, not
    # buried exclusively in metrics.csv. Counts are completed episodes
    # actually seen by the governor, not claimed mastery or future targets.
    if 'focus_worlds_command' in row:
        labels=('command','resilience','slope','step','transition','control')
        births='/'.join(str(int(row.get('focus_worlds_'+k,0))) for k in labels)
        ended='/'.join(str(int(row.get('focus_episodes_'+k,0))) for k in labels)
        slope_near=sum(row.get('slope_near_attempts_'+k,0)
                       for k in ('uphill','downhill','cross_left','cross_right'))
        step_near=sum(row.get('terrain_near_attempts_'+k,0)
                      for k in ('up','down'))
        line6=(f"[{version}]   lessons active(cmd/res/sl/step/tr/ctl)={births} "
               f"done={ended} "
               f"near(cmd+/-)={int(row.get('command_probe_attempts_vx_hi',0))}/"
               f"{int(row.get('command_probe_attempts_vx_lo',0))} "
               f"slope={int(slope_near)} step={int(step_near)} "
               f"ctl_Jm={row.get('control_shaft_j_per_m',-1.):.1f}")
        return line1 + "\n" + line2 + "\n" + line3 + "\n" + line4 + "\n" + line5 + "\n" + line6
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
    # Report-only projection for the real capability field. The old public
    # writer contract also accepts minimal save_npz/write_summary fields
    # (for example independent delivery/export test doubles). Don't demand
    # new private array methods from those objects.
    from .curriculum import CapabilityField
    if isinstance(field, CapabilityField):
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


# Provisional engineering guardrails for comparable FROZEN-policy panels.
# These are tolerances for regression alerts, not statistically justified
# significance bounds or reward coefficients. Core practice must survive a
# new capability (such as stepping over terrain).
FIXED_EXAM_GUARDS = {
    "eval_core_survival": ("floor", 0.05),
    "eval_core_support": ("floor", 0.08),
    "eval_core_step_evidence": ("floor", 0.15),
    "eval_core_vx_mae": ("ceiling", 0.025),
    "eval_core_vy_mae": ("ceiling", 0.025),
    "eval_core_yaw_mae": ("ceiling", 0.12),
    "eval_core_pitch_mae": ("ceiling", 0.07),
    "eval_core_height_shortfall_m": ("ceiling", 0.007),
    "eval_dynamic80_survival": ("floor", 0.075),
    "eval_combo70_survival": ("floor", 0.075),
    "eval_resilience75_survival": ("floor", 0.10),
    "eval_robust40_survival": ("floor", 0.08),
}


def compare_fixed_exam_evidence(
    reference: dict[str, Any], candidate: dict[str, Any], *,
    reference_protocol: str, candidate_protocol: str,
) -> dict[str, Any]:
    """Independent matched-exam regression report; NEVER a PPO lesson.

    Both inputs must be REAL fixed-evaluation aggregate rows from the same
    frozen environment/commands/sensor support. The caller must supply a
    nonempty verifiable exam-protocol fingerprint for each run. Missing data
    and changed exam setup are **not comparisons**. This is a conservative
    regression alert, not a new champion rank or a confidence interval.
    """
    if (not reference_protocol or not candidate_protocol
            or reference_protocol != candidate_protocol):
        raise ValueError("fixed exams require identical nonempty protocol fingerprints")
    required=set(FIXED_EXAM_GUARDS)|{
        "eval_policy_step","perf_eval_cohorts_used","perf_eval_env_steps"}
    def checked(name: str, row: dict[str, Any]) -> dict[str,float]:
        missing=sorted(required-row.keys())
        if missing:
            raise ValueError(f"{name} missing held-out fields: {', '.join(missing)}")
        result={}
        for key in required:
            try:
                result[key]=float(row[key])
            except (TypeError,ValueError) as exc:
                raise ValueError(f"{name} has invalid {key}") from exc
            if not math.isfinite(result[key]):
                raise ValueError(f"{name} has nonfinite {key}")
        if (result["perf_eval_cohorts_used"]<1
                or result["perf_eval_env_steps"]<1
                or result["eval_policy_step"]<0):
            raise ValueError(f"{name} has insufficient fixed-exam provenance")
        if any(not (0 <= result[k] <= 1) for k in (
              "eval_core_survival","eval_core_support","eval_core_step_evidence",
              "eval_dynamic80_survival","eval_combo70_survival",
              "eval_resilience75_survival","eval_robust40_survival")):
            raise ValueError(f"{name} has out-of-range success/support evidence")
        if any(result[k]<0 for k in required if k.endswith("_mae") or k.endswith("_m")):
            raise ValueError(f"{name} has negative physical errors")
        return result
    before=checked("reference",reference)
    after=checked("candidate",candidate)
    if (before["perf_eval_cohorts_used"]!=after["perf_eval_cohorts_used"]
            or before["perf_eval_env_steps"]!=after["perf_eval_env_steps"]):
        raise ValueError("fixed exam cohort/size mismatch; no valid comparison")

    metrics={}
    regressed=[]
    for field,(direction,tolerance) in FIXED_EXAM_GUARDS.items():
        change=after[field]-before[field]
        # Guardrails have an explicit inclusive boundary, irrespective of
        # binary float representation of values loaded from fixed_eval.csv.
        bad=(change < -tolerance-1e-10) if direction=="floor" else (change > tolerance+1e-10)
        metrics[field]=dict(
            reference=before[field],candidate=after[field],delta=change,
            direction=direction,allowed_worsening=tolerance,
            regressed=bool(bad))
        if bad:regressed.append(field)
    return {
        "schema":"emu_held_out_core_regression_1",
        "source":"held_out_exam",
        "protocol_fingerprint":reference_protocol,
        "reference_step":int(before["eval_policy_step"]),
        "candidate_step":int(after["eval_policy_step"]),
        "matched_cohorts":int(before["perf_eval_cohorts_used"]),
        "matched_eval_env_steps":int(before["perf_eval_env_steps"]),
        "guardrail_status":"regression" if regressed else "no_guardrail_regression",
        "regressed_fields":regressed,
        "metrics":metrics,
        "qualification":(
            "Alert-only matched physical panel comparison. Thresholds are "
            "provisional and cohort averages are not independent confidence "
            "intervals. No ranking, curriculum teaching, or promotion."
        ),
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
        # G4 physically completed finite-terrain episodes, counted only
        # from true MJX contact/trajectory evidence (NOT proxy foot lift).
        'terrain_attempts': float(np.asarray(metrics['terrain_episode_attempt']).sum())
            if 'terrain_episode_attempt' in metrics else 0.0,
        'terrain_successes': float(np.asarray(metrics['terrain_episode_success']).sum())
            if 'terrain_episode_success' in metrics else 0.0,
        'terrain_crossings': float(np.asarray(metrics['terrain_cross_new']).sum())
            if 'terrain_cross_new' in metrics else 0.0,
        'terrain_ascent_guidance_per_frame': mean_metric(metrics, 'terrain_ascent_guidance'),
        'ep_len_mean': done_mean(metrics, 'episode_length', default=0.0),
        'ep_rew_mean': done_mean(metrics, 'episode_return', default=0.0),
        'fall_rate': done_mean(metrics, 'episode_fell', default=mean_metric(metrics, 'terminal')),
        'timeout_rate': done_mean(metrics, 'episode_timeout', default=0.0),
        'vx_mean': done_mean(metrics, 'episode_vx_mean', default=mean_metric(metrics, 'vx')),
        'cmd_vx_mean': done_mean(metrics, 'episode_cmd_vx_mean', default=mean_metric(metrics, 'cmd_vx')),
        # Assisted control: distinguish OPERATOR demand, what the policy
        # was actually asked to do, and accumulated correction.
        'operator_cmd_vy_mean': mean_metric(metrics,'cmd_vy'),
        'policy_cmd_vx_mean': mean_metric(metrics,'policy_cmd_vx'),
        'policy_cmd_vy_mean': mean_metric(metrics,'policy_cmd_vy'),
        'velocity_i_x_mean': mean_metric(metrics,'velocity_i_x'),
        'velocity_i_y_mean': mean_metric(metrics,'velocity_i_y'),
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
    # True mixed-geometry physical slope: never infer gradient from a
    # curriculum target or fold downhill into successful uphill evidence.
    # All four directional counters are COMPLETE episode facts.
    if 'physical_slope_deg' in metrics:
        raw_slope=np.asarray(metrics['physical_slope_deg'])
        active=(raw_slope>=.10)
        row['physical_slope_deg_mean']=float(raw_slope[active].mean()) if active.any() else 0.
        row['physical_slope_deg_max']=float(raw_slope.max())
        row['physical_slope_world_fraction']=float(active.mean())
        for name in ('uphill','downhill','cross_left','cross_right'):
            for outcome,word in (('attempt','attempts'),('success','successes'),
                                 ('stable','stable_successes')):
                key=f'slope_episode_{outcome}_{name}'
                row[f'slope_{word}_{name}']=(
                    float(np.asarray(metrics[key]).sum()) if key in metrics else 0.)
            for band in ('near','easy'):
                for outcome,word in (('attempt','attempts'),('success','successes'),
                                     ('stable','stability_successes')):
                    key=f'slope_episode_{band}_{outcome}_{name}'
                    row[f'slope_{band}_{word}_{name}']=(
                        float(np.asarray(metrics[key]).sum()) if key in metrics else 0.)
    # Direction-separated terrain evidence. A quality metric without any
    # active frames is UNKNOWN (-1), not success/zero error.
    for direction in ('up', 'down'):
        for outcome, plural in (('attempt', 'attempts'), ('success', 'successes'),
                                ('fall', 'falls')):
            name = f'terrain_episode_{outcome}_{direction}'
            row[f'terrain_{plural}_{direction}'] = (
                float(np.asarray(metrics[name]).sum()) if name in metrics else 0.0)
        for band in ('near','easy'):
            for outcome,plural in (('attempt','attempts'),('success','successes')):
                key=f'terrain_episode_{band}_{outcome}_{direction}'
                row[f'terrain_{band}_{plural}_{direction}']=(
                    float(np.asarray(metrics[key]).sum()) if key in metrics else 0.0)
        mask_name = f'terrain_active_{direction}'
        mask = (np.asarray(metrics[mask_name]) > 0.5 if mask_name in metrics
                else np.zeros_like(np.asarray(metrics.get('vx', 0.0)), dtype=bool))
        row[f'terrain_frames_{direction}'] = float(mask.sum())
        for key, label in (('support', 'support'), ('bad_contact', 'bad_contact'),
                           ('servo_shaft_drive_power_w', 'shaft_drive_power_w')):
            row[f'terrain_{label}_{direction}'] = (
                float(np.asarray(metrics[key])[mask].mean())
                if mask.any() and key in metrics else -1.0)
        row[f'terrain_vx_mae_{direction}'] = (
            float(np.abs(np.asarray(metrics['vx'])[mask] -
                         np.asarray(metrics['cmd_vx'])[mask]).mean())
            if mask.any() and 'vx' in metrics and 'cmd_vx' in metrics else -1.0)

    # CPU-only arithmetic on rollout metrics already transferred from JAX.
    # Positive mechanical actuator work divided by NET forward displacement
    # in frames requesting forward walking. Never infer battery consumption,
    # and refuse an unstable ratio for near-zero or backwards displacement.
    row['shaft_j_per_m_forward'] = -1.0
    if all(name in metrics for name in ('servo_shaft_drive_power_w', 'vx', 'cmd_vx')):
        cmd = np.asarray(metrics['cmd_vx'], dtype=float)
        actual = np.asarray(metrics['vx'], dtype=float)
        drive = np.asarray(metrics['servo_shaft_drive_power_w'], dtype=float)
        forward = ((cmd >= 0.075) & np.isfinite(cmd) &
                   np.isfinite(actual) & np.isfinite(drive))
        if np.count_nonzero(forward) >= max(1, cmd.size // 20):
            forward_vx = float(actual[forward].mean())
            if forward_vx >= 0.04:
                row['shaft_j_per_m_forward'] = max(
                    0.0, float(drive[forward].sum() / actual[forward].sum()))
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
    # Keep separate true up-crossing evidence by commanded speed; no pooled
    # DOWN result or metric of mere lip approach can certify this ability.
    for speed_band in ('slow','medium','fast'):
        for outcome,plural in (('attempt','attempts'),('success','successes'),('fall','falls')):
            key = f'terrain_episode_{outcome}_up_{speed_band}'
            row[f'terrain_{plural}_up_{speed_band}'] = (
                float(np.asarray(metrics[key]).sum()) if key in metrics else 0.0)

    # Outcome ledgers use COMPLETED episodes only. Fast/slow partial frames
    # and a short-lived lucky speed spike never count as full competence.
    # Focus is a teacher label; old experiments expose -1 for all worlds.
    if 'specialist_focus' in metrics:
        focus=np.asarray(metrics['specialist_focus'],dtype=np.int32)
        for i,name in enumerate(('command','resilience','slope','step','transition','control')):
            row[f'focus_worlds_{name}'] = float(np.count_nonzero(focus[0]==i))
            # Birth allocation is NOT exposure: episodes can finish at
            # different lengths and roles turn on gradually at the boundary.
            # Count actual sampled PPO frames in each specialty.
            row[f'focus_step_share_{name}'] = float(np.mean(focus==i))
            world_done=done & (focus==i)
            row[f'focus_episodes_{name}']=float(world_done.sum())
            row[f'focus_falls_{name}']=float((
                world_done & (np.asarray(metrics['episode_fell'])>.5)).sum())
        cmd=done & (focus==0)
        quality=np.asarray(metrics['episode_command_quality'])
        radius=np.asarray(metrics['command_probe_radius'])
        xshare=np.asarray(metrics['command_probe_weight_x'])
        yshare=np.asarray(metrics['command_probe_weight_y'])
        zshare=np.asarray(metrics['command_probe_weight_z'])
        vx=np.asarray(metrics['cmd_vx'])
        vy=np.asarray(metrics['cmd_vy'])
        yaw=np.asarray(metrics['cmd_yaw'])
        near=cmd & (radius>=.88) & (radius<=1.14)
        easier=cmd & (radius>=.50) & (radius<.80)
        success=(np.asarray(metrics['episode_fell'])<.5) & (
            np.asarray(metrics['episode_length'])>=120.) & (quality>=.60) & (
            np.asarray(metrics['episode_bad_contact'])<.3) & (
            np.asarray(metrics['episode_support'])>.10)
        sectors={
            'vx_hi':(xshare>=.80)&(vx>0.),
            'vx_lo':(xshare>=.80)&(vx<0.),
            'vy_pos':(yshare>=.80)&(vy>0.),
            'vy_neg':(yshare>=.80)&(vy<0.),
            'yaw_pos':(zshare>=.80)&(yaw>0.),
            'yaw_neg':(zshare>=.80)&(yaw<0.),
            'mixed':(np.maximum(np.maximum(xshare,yshare),zshare)<.80)
        }
        for label,mask in sectors.items():
            tested=near & mask
            row[f'command_probe_attempts_{label}']=float(tested.sum())
            row[f'command_probe_successes_{label}']=float((tested & success).sum())
            easier_tested=easier & mask
            row[f'command_easy_attempts_{label}']=float(easier_tested.sum())
            row[f'command_easy_successes_{label}']=float((easier_tested & success).sum())
        ctl=(focus==5)
        # Control starts with real but slow forward walking (typically
        # 0.04..0.06 m/s). Requiring 0.075 m/s here previously made the
        # entire efficiency cohort report UNKNOWN even when it was moving.
        # Keep a positive actual-progress floor: standing still cannot
        # produce an artificial zero-energy achievement.
        moving=(ctl & (np.asarray(metrics['cmd_vx'])>=.045) &
                (np.asarray(metrics['vx'])>=.025))
        if np.any(moving):
            row['control_shaft_j_per_m'] = float(
                np.asarray(metrics['servo_shaft_drive_power_w'])[moving].sum() /
                max(.01,np.asarray(metrics['vx'])[moving].sum()))
        else:
            row['control_shaft_j_per_m']=-1.
        ctl_done=done & ctl
        row['control_episode_quality']=float(quality[ctl_done].mean()) if ctl_done.any() else -1.
        smooth=np.asarray(metrics.get('reward_weighted_smoothness',0.))
        row['control_smoothness_payment']=float(smooth[ctl].mean()) if ctl.any() else -1.
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
