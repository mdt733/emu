"""Summarise what the robot actually did; do NOT calculate its reward.

The universe provides physical measurements and the reward ledger's component
payments. This module reduces the JAX arrays on the host into per-rollout
facts: survival, command obedience, step proxies, fall causes, servo mechanics
and curriculum frontiers.

`summarize_rollout` does not alter the robot, PPO or the curriculum. It
produces evidence *for* their respective consumers. True signed uphill
progress and load-bearing walking require separate controlled exams.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .contracts import ACTION_DIM, COMBO_P, SEED_VX_LO, SEED_VX_HI, task_param_defaults
from .rewards import PAYMENT_NAMES, FALL_PAYMENT_NAME
from .tracking_contract import (
    FRONTIER_VX_ABS_TOL, FRONTIER_VX_REL_TOL, FRONTIER_VY_ABS_TOL, FRONTIER_VY_REL_TOL,
    FRONTIER_YAW_ABS_TOL, FRONTIER_YAW_REL_TOL, FRONTIER_EDGE_FRAC, FRONTIER_SETTLE_S,
)

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
           for name in (*PAYMENT_NAMES, FALL_PAYMENT_NAME)},
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
