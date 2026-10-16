"""Training-run artifact and human-readable logging helpers."""
from __future__ import annotations

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
