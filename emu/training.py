#!/usr/bin/env python3
"""The *one* training run: coordinate worlds, teacher, PPO and checkpoints.

Read this file in order:

1. `train(config)` restores (or starts) the learner and creates MJX worlds.
2. PPO collects batches of real physics experience across `config.envs` worlds.
3. Rollout metrics and independent exam snapshots describe what happened.
4. Curriculum/health adjust the *next* task distribution and recovery policy.
5. Durable logs and checkpoints preserve the policy and its learning state.
6. The final checkpoint and report are published only after validation.

Training DOES NOT define the reward mathematics (see rewards.py) or choose the
winning training strategy (see ladder_recipe.py / branch_lab.py).

KNOWN DEBT: most of these stages still inhabit a long `train` body.
See docs/SIMPLIFICATION.md Pass D before extracting code; avoid wrapper mazes.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

from .run_support import resume_training_lineage, promote_checkpoint
from .environment import EmuMJXEnv
from .domain_randomization import DomainRandomizationConfig, doctrine_dict as domain_doctrine_dict
from .capability_field import CapabilityField
from .contracts import ACTION_DIM, OBS_DIM, CONTROL_DT, FIELD_NODE_ANCHORS, FIELD_EDGE_ANCHORS
from .training_config import (
    TrainingConfig, doctrine_dict,
    ROLLOUT_STEPS, PPO_EPOCHS, PPO_MINIBATCHES, PPO_LR, PPO_ENTROPY_COST, PPO_VALUE_COST,
    LOG_EVERY_UPDATES, CHECKPOINT_EVERY_UPDATES, SNAPSHOT_EVERY_STEPS, SNAPSHOT_KEEP,
    PROFILE_PERFORMANCE, MAX_EPISODE_STEPS,
    OBS_NORM_MODE, PPO_TARGET_KL, PPO_LR_SCALE_MIN, PPO_LR_SCALE_INIT, PPO_LR_SCALE_MAX, PPO_LR_ADJUST, FOUNDATION_STEPS,
    EVAL_EVERY_STEPS, EVAL_SEED, EVAL_COHORTS, FINAL_EVAL_COHORTS,
    CLOUD_EVAL_EVERY_STEPS, CLOUD_ENVS, CLOUD_HORIZON, CLOUD_BATCH_POINTS,
    FIELD_HALF_LIFE_STEPS, SENSOR_NOISE_SCALE,
    RECOVERY_TILT_MAX_DEG, RECOVERY_POSITION_MAX_M, RECOVERY_LINVEL_MAX_MPS,
    RECOVERY_ANGVEL_MAX_RAD_S, RECOVERY_JOINT_MAX_DEG,
)
from .curriculum import RegimeGovernor
from .health import TrainingHealth
from .ppo import make_trainer, update_running_stats
from .rollout_metrics import summarize_rollout, augment_rollout_row
from .runtime import PerfBook, configure_jax, require_backend
from .artifacts import ensure_run_dir, write_json, write_rows_csv, read_csv_rows, append_metrics_csv, read_metrics_csv, format_line
from .evaluation import (
    build_worlds as build_eval_worlds, initial_metrics as initial_eval_metrics, evaluate_fixed,
    EVAL_PERF_TIMER_NAMES, PolicySnapshot, outranks_best, checkpoint_eval_score,
)
from .paths import ROBOT_XML, VERSION
from .checkpoint_io import (save_checkpoint, save_rotating_snapshot, load_checkpoint,
                            payload_train_state_is_finite, source_provenance_from_env)

RUN_VERSION = f'emu_{VERSION}'
LOG_PREFIX = f'[{RUN_VERSION}]'

TRAIN_PERF_TIMER_NAMES = (
    'train_update', 'host_metric_transfer', 'capability_field_update',
    'host_rollout_summary', 'governor_and_params', 'obs_norm_and_sync',
    'log_io', 'fixed_eval_total', 'capability_cloud',
    'checkpoint_best_device_get', 'checkpoint_best_pickle_write', 'checkpoint_best_total',
    'checkpoint_latest_device_get', 'checkpoint_latest_pickle_write', 'checkpoint_latest_total',
    'checkpoint_snapshot_device_get', 'checkpoint_snapshot_pickle_write', 'checkpoint_snapshot_total',
    'checkpoint_final_device_get', 'checkpoint_final_pickle_write', 'checkpoint_final_total',
    *EVAL_PERF_TIMER_NAMES,
)

# ---------------------------------------------------------------------------
# Utilities / artifacts.
# ---------------------------------------------------------------------------

def env_kwargs_from_args(args) -> dict:
    """One source of truth for the physical/sensor envelope used by training."""
    return dict(
        push_max_force_n=args.push_max_force_n,
        vx_hi_max=args.vx_hi_max, vx_lo_min=args.vx_lo_min,
        vy_max_mps=args.vy_max_mps, yaw_max_rad_s=args.yaw_max_rad_s,
        phase_speed_gain=args.phase_speed_gain,
        attitude_profile=args.attitude_profile,
        pitch_command_max_deg=args.pitch_command_max_deg,
        sensor_noise_scale=SENSOR_NOISE_SCALE, open_frontier=args.open_frontier,
        combo_mix_frac=args.combo_mix_frac, combo_probe_frac=args.combo_probe_frac,
        recovery_tilt_max_deg=RECOVERY_TILT_MAX_DEG,
        recovery_position_max_m=RECOVERY_POSITION_MAX_M,
        recovery_linvel_max_mps=RECOVERY_LINVEL_MAX_MPS,
        recovery_angvel_max_rad_s=RECOVERY_ANGVEL_MAX_RAD_S,
        recovery_joint_max_deg=RECOVERY_JOINT_MAX_DEG,
        field_teach_recovery=bool(args.field_teach_recovery),
        domain_config=DomainRandomizationConfig(
            enabled=bool(args.domain_randomization),
            robot_radius=float(args.domain_robot_radius),
            terrain_radius=float(args.domain_terrain_radius),
            hardware_radius=float(args.domain_hardware_radius),
        ),
    )


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
                     base_lr: float = PPO_LR) -> dict[str, Any]:
    """Canonical metrics.csv schema for periodic and final rows."""
    return {
        'step': step, 'elapsed_s': elapsed, 'steps_per_sec': wall_fps, 'train_steps_per_sec': train_fps,
        'reward_mean': reward, **row, **report,
        'backend': 'mjx', 'jax_backend': jax_backend, 'num_envs': num_envs, 'rollout': ROLLOUT_STEPS,
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

# ---------------------------------------------------------------------------
# Run modes.
# ---------------------------------------------------------------------------

def train(args: TrainingConfig) -> int:
    """Run one checkpoint lineage; no branch-selection logic lives here.

    The main loop is organised in evidence time:
    1. examine the policy that produced the last rollout;
    2. reduce facts, revise next curriculum, and checkpoint if needed;
    3. collect the next rollout and update PPO.
    The first rollout is collected before entering the loop.
    """
    allow_cpu_mjx = args.device == 'cpu'
    configure_jax(args.device)
    import jax
    require_backend(allow_cpu_mjx)
    version = RUN_VERSION
    if not args.run_dir:
        raise ValueError('training config requires run_dir')
    run_dir = Path(args.run_dir).resolve()
    run_name = run_dir.name
    ensure_run_dir(run_dir)

    # Restore schema-7 learning state; never update an already-complete run.
    resumed_payload, resumed_step, already_complete = resume_training_lineage(
        run_dir, resume=getattr(args, 'resume', ''), target_steps=args.steps, envs=args.envs)
    if already_complete:
        return 0

    meta = {'version': version, 'source_provenance': source_provenance_from_env(),
            'config': vars(args), 'doctrine': {**doctrine_dict(), **{f'DOMAIN_{k}': v for k, v in domain_doctrine_dict().items()}},
            'resumed_from_step': resumed_step if resumed_payload is not None else None}
    write_json(run_dir / 'run_meta.json', meta)
    perf = PerfBook(enabled=bool(PROFILE_PERFORMANCE), known_names=TRAIN_PERF_TIMER_NAMES)

    env = EmuMJXEnv(str(ROBOT_XML), num_envs=args.envs, seed=args.seed,
                    max_episode_steps=MAX_EPISODE_STEPS, **env_kwargs_from_args(args))
    trainer = make_trainer(env, seed=args.seed, lr=args.ppo_lr, rollout=ROLLOUT_STEPS, epochs=PPO_EPOCHS, minibatches=PPO_MINIBATCHES,
                           entropy_cost=args.ppo_entropy_cost, value_cost=args.ppo_value_cost)
    fixed_eval_worlds = build_eval_worlds(args)

    # Sparse capability cloud: intentionally much smaller than the fixed
    # continuity evaluator. Each cloud point is one real vx+vy+yaw command
    # combination plus a categorical disturbance family and one severity
    # scalar. Nuisance direction/sign is randomized across the vector batch
    # rather than becoming another map dimension.
    cloud_enabled = int(args.cloud_probes_per_eval) > 0 or int(args.cloud_final_probes) > 0
    cloud_eval_env = cloud_eval_trainer = None
    cloud_rows: list[dict] = []
    if cloud_enabled:
        from .capability_cloud import (make_panel as cloud_make_panel,
                                      make_calibration_panel as cloud_make_calibration_panel,
                                      make_common_recovery_panel as cloud_make_common_recovery_panel,
                                      evaluate_specs_batch as cloud_evaluate_specs_batch, summarize_rows as cloud_summarize_rows)
        cloud_eval_env = EmuMJXEnv(str(ROBOT_XML), num_envs=int(CLOUD_ENVS),
                    seed=EVAL_SEED + 44000, max_episode_steps=max(MAX_EPISODE_STEPS, int(CLOUD_HORIZON)+50),
                    push_max_force_n=max(4.0, float(args.push_max_force_n or 0.35)),
                    vx_hi_max=1.20, vx_lo_min=-0.65, vy_max_mps=0.65, yaw_max_rad_s=5.0,
                    phase_speed_gain=args.phase_speed_gain,
                    attitude_profile=args.attitude_profile,
                    pitch_command_max_deg=args.pitch_command_max_deg,
                    sensor_noise_scale=SENSOR_NOISE_SCALE, open_frontier=False, combo_mix_frac=0.0)
        cloud_eval_trainer = make_trainer(cloud_eval_env, seed=EVAL_SEED + 44000, lr=args.ppo_lr, rollout=ROLLOUT_STEPS,
                           epochs=1, minibatches=1, entropy_cost=args.ppo_entropy_cost, value_cost=args.ppo_value_cost)
        cloud_rows = read_csv_rows(run_dir / 'heldout_capability_cloud.csv')

    # No track-level flags to validate here -- a regime's difficulty is judged directly by RegimeGovernor, with no scheduling layer in between.
    governor = RegimeGovernor(fall_target=args.fall_target,
                              difficulty_base_step=args.difficulty_base_step,
                              regime_command_frac=args.regime_command_frac,
                              athletic_rehearsal_frac=args.athletic_rehearsal_frac,
                              open_frontier=args.open_frontier,
                              frontier_vx_hi_safety=args.frontier_vx_hi_safety,
                              frontier_vx_lo_safety=args.frontier_vx_lo_safety,
                              frontier_vy_safety=args.frontier_vy_safety, frontier_yaw_safety=args.frontier_yaw_safety,
                              frontier_combo_safety=args.frontier_combo_safety, frontier_push_safety_n=args.frontier_push_safety_n)
    # When adaptive physical frontiers are disabled, report the actual fixed
    # command envelope rather than dormant RegimeParams defaults.
    if not args.open_frontier and resumed_payload is None:
        governor.params.frontier_vx_hi = float(args.vx_hi_max)
        governor.params.frontier_vx_lo = float(args.vx_lo_min)
        governor.params.frontier_vy = float(args.vy_max_mps)
        governor.params.frontier_yaw = float(args.yaw_max_rad_s)
    health = TrainingHealth()
    if resumed_payload is not None:
        try:
            governor.load_state_dict(resumed_payload['curriculum_state'])
            health.load_state_dict(resumed_payload['health_state'])
            print(f"{LOG_PREFIX} --resume: restored curriculum + training health "
                  f"(updates={governor._updates_done_total}, core_bad_streak={governor.core_bad_streak}, "
                  f"kl_history={len(health.kl_stress_history)}, collapse_streak={health.stepping_collapse_streak}).", flush=True)
        except Exception as exc:
            raise RuntimeError(f"--resume curriculum/health restore failed: {exc}") from exc

    # The capability field is built from the SAME rollout
    # metrics already transferred for PPO/governor diagnostics; there is no
    # extra simulation and no per-death host callback.  Held-out cloud results
    # never enter this object.
    field = CapabilityField(half_life_steps=float(FIELD_HALF_LIFE_STEPS))
    if resumed_payload is not None:
        try:
            field = CapabilityField.from_state_dict(resumed_payload['field_state'])
            print(f"{LOG_PREFIX} --resume: restored capability field "
                  f"({field.summary()['field_known_nodes']:.0f} known nodes, "
                  f"{field.summary()['field_known_edges']:.0f} known edges)", flush=True)
        except Exception as exc:
            raise RuntimeError(f"--resume capability-field restore failed: {exc}") from exc

    def params_with_field(step: int) -> dict[str, Any]:
        d = governor.params_dict()
        packet = field.teaching_packet(
            step=int(step), mode=str(args.capability_field_mode),
            start=int(args.field_influence_start_steps), full=int(args.field_influence_full_steps),
            fade_start=int(args.field_influence_fade_start_steps), zero=int(args.field_influence_zero_steps),
            maximum=float(args.field_max_influence),
            teach_nodes=bool(args.field_teach_nodes), teach_edges=bool(args.field_teach_edges),
            node_count=FIELD_NODE_ANCHORS, edge_count=FIELD_EDGE_ANCHORS)
        d.update(packet)
        return d

    params = params_with_field(resumed_step)
    state = trainer['make_initial_state'](params)
    if resumed_payload is not None:
        try:
            loaded_state = resumed_payload['state']
            state = state._replace(params=jax.device_put(loaded_state['params']),
                                    opt_state=jax.device_put(loaded_state['opt_state']))
            print(f"{LOG_PREFIX} --resume: restored policy/value params + optimizer state + curriculum "
                  f"(steady={governor.params.steady_difficulty:.2f} transition={governor.params.transition_difficulty:.2f} "
                  f"resilience={governor.params.resilience_difficulty:.2f} posture={governor.params.posture_difficulty:.2f} "
                  f"env-alloc={governor.params.regime_frac_steady:.2f}/{governor.params.regime_frac_transition:.2f}/"
                  f"{governor.params.regime_frac_resilience:.2f}); episode buffers re-initialize fresh.", flush=True)
        except Exception as exc:
            raise RuntimeError(f"--resume network/optimizer restore failed: {exc}") from exc
    p = env.params_array(params)

    print(f"{LOG_PREFIX} training start run={run_name} backend=mjx steps={args.steps:,} envs={args.envs} rollout={ROLLOUT_STEPS} epochs={PPO_EPOCHS} device={args.device} jax_backend={jax.default_backend()}", flush=True)
    print(f"{LOG_PREFIX} obs_dim={OBS_DIM} action_dim={ACTION_DIM} regimes=steady/transition/resilience + posture", flush=True)
    print(f"{LOG_PREFIX} compiling MJX/JAX update...", flush=True)
    import jax.numpy as jnp
    obs_mean = jnp.zeros((OBS_DIM,), dtype=jnp.float32)
    obs_var = jnp.ones((OBS_DIM,), dtype=jnp.float32)
    obs_count = 1e-4  # small epsilon, standard init for a running-variance accumulator
    if resumed_payload is not None and resumed_payload.get('obs_norm') is not None:
        try:
            saved_norm = resumed_payload['obs_norm']
            obs_mean = jnp.asarray(saved_norm['mean'], dtype=jnp.float32)
            obs_var = jnp.asarray(saved_norm['var'], dtype=jnp.float32)
            obs_count = float(saved_norm['count'])
            print(f"{LOG_PREFIX} --resume: restored obs normalization (count={obs_count:,.0f})", flush=True)
        except Exception as exc:
            raise RuntimeError(f"--resume observation-normalizer restore failed: {exc}") from exc
    # KL-adaptive learning-rate controller (standard form: Schulman
    # adaptive-KL PPO, rl_games, Brax) -- a soft, multiplicative, self-
    # tuning complement to the hard KL_HARD_STOP ceiling above. KL cost
    # scales as 1/sigma^2 and sigma moves over a run, so a fixed step size
    # alone can't track that. Starts at 0.25x (not 1.0x), not 1.0x full
    # scale: entering at full lr would be a large, untested jump in
    # per-rollout policy movement before the controller has had a chance
    # to find the right scale on its own.
    LR_TARGET_UPDATE_KL = float(PPO_TARGET_KL)   # desired total KL from rollout policy per update
    LR_SCALE_MIN, LR_SCALE_MAX = float(PPO_LR_SCALE_MIN), float(PPO_LR_SCALE_MAX)
    lr_scale = float(PPO_LR_SCALE_INIT)
    if resumed_payload is not None:
        lr_scale = float(resumed_payload['meta']['lr_scale'])
        print(f"{LOG_PREFIX} --resume: restored lr_scale={lr_scale:.4f} "
              f"(effective lr={PPO_LR * lr_scale:.3e})", flush=True)
    # Associate rollout-derived health metrics with the exact policy,
    # normalizer, curriculum and governor state that GENERATED them.
    evaluated_obs_norm = {'mean': obs_mean, 'var': obs_var, 'count': obs_count}
    rollout_params_for_metrics = params_with_field(resumed_step)
    rollout_curriculum_state_for_metrics = governor.state_dict()
    rollout_health_state_for_metrics = health.state_dict()
    rollout_field_state_for_metrics = field.state_dict()
    rollout_lr_scale_for_metrics = lr_scale
    norm_diag = {'obs_norm_update_mean_l2': 0.0, 'obs_norm_update_std_l2': 0.0,
                 'obs_norm_anchor_mean_l2': 0.0, 'obs_norm_anchor_std_l2': 0.0,
                 'obs_norm_frozen': 0.0}
    obs_anchor = None

    t0 = time.time()
    old_mean, old_std = obs_mean, obs_var ** 0.5
    state, rollout_metrics, aux, raw_obs, rollout_policy_state = trainer['train_update'](state, p, obs_mean, old_std, lr_scale)
    warmup_end_steps = resumed_step + args.envs * ROLLOUT_STEPS
    norm_frozen = args.freeze_obs_norm_after_steps > 0 and resumed_step >= args.freeze_obs_norm_after_steps
    if not norm_frozen:
        obs_mean, obs_var, obs_count = update_running_stats(obs_mean, obs_var, obs_count, raw_obs)
    norm_diag['obs_norm_update_mean_l2'] = float(np.linalg.norm(np.asarray(obs_mean - old_mean)))
    norm_diag['obs_norm_update_std_l2'] = float(np.linalg.norm(np.asarray(obs_var ** 0.5 - old_std)))
    norm_diag['obs_norm_frozen'] = float(norm_frozen)
    if warmup_end_steps >= FOUNDATION_STEPS:
        obs_anchor = (np.asarray(obs_mean).copy(), np.asarray(obs_var ** 0.5).copy())
    state.obs.block_until_ready()
    metric_transfer_t0 = time.perf_counter()
    rollout_metrics, aux = jax.device_get((rollout_metrics, aux))
    perf.add('host_metric_transfer', time.perf_counter() - metric_transfer_t0)
    print(f"{LOG_PREFIX} compile+first_update={time.time()-t0:.2f}s", flush=True)

    # --steps is the absolute target step count, not "how many more steps
    # to run" -- so a resumed run's warm-up update above is counted against
    # the restored step total, and log/save cadence continues from where
    # it left off instead of immediately firing at time zero.
    total_steps = resumed_step + args.envs * ROLLOUT_STEPS
    start = time.time()
    last_log = resumed_step
    last_save = resumed_step
    last_snapshot = resumed_step
    last_seen_nan_skips = [0]  # mutable box so the log block above can update it
    # Canonical rollback candidate: never overwrite a stronger existing
    # recovery snapshot after resume/fork.
    recovery_path = run_dir / 'checkpoints' / 'recovery.pkl'
    best_recovery_rank = (-1, -1.0)
    if recovery_path.exists():
        try:
            existing_recovery = load_checkpoint(recovery_path)
            saved_rank = existing_recovery.get('meta', {}).get('recovery_rank', [-1, -1.0])
            best_recovery_rank = (int(saved_rank[0]), float(saved_rank[1]))
        except Exception:
            pass
    comp_window: list[tuple[float, float]] = []
    recent_rewards: list[float] = []
    log_every_steps = args.envs * ROLLOUT_STEPS * max(1, int(LOG_EVERY_UPDATES))
    save_every_steps = args.envs * ROLLOUT_STEPS * max(1, int(CHECKPOINT_EVERY_UPDATES))
    snapshot_every_steps = max(0, int(SNAPSHOT_EVERY_STEPS))

    # Held-out evaluation state is reporting/artifact state only; it never feeds
    # PPO, curriculum, health or capability-field updates.
    latest_eval = initial_eval_metrics()
    fixed_history = read_metrics_csv(run_dir / 'fixed_eval.csv')
    last_eval_step = int(float(fixed_history[-1]['step'])) if fixed_history else resumed_step
    if fixed_history:
        for key in latest_eval:
            if key in fixed_history[-1]:
                latest_eval[key] = float(fixed_history[-1][key])
    broad_eval_history = [float(row['eval_broad_score']) for row in fixed_history[-2:]
                          if row.get('eval_broad_score') not in (None, '')]
    last_cloud_step = max(
        [int(float(row.get('policy_step', resumed_step))) for row in cloud_rows] or [resumed_step])
    best_eval_score = -1e9

    # Preserve evaluation cadence/history and best-checkpoint monotonicity across
    # same-run process restarts. External forks have no local eval artifacts.
    best_eval_path = run_dir / 'checkpoints' / 'best.pkl'
    if best_eval_path.exists():
        try:
            best_eval_score = checkpoint_eval_score(load_checkpoint(best_eval_path).get('meta', {}))
        except Exception:
            pass

    def maybe_fixed_eval(step: int, force: bool = False) -> None:
        nonlocal last_eval_step, latest_eval, best_eval_score
        if not force and step - last_eval_step < EVAL_EVERY_STEPS:
            return
        cohorts = FINAL_EVAL_COHORTS if force else EVAL_COHORTS
        snapshot = PolicySnapshot(step, state, obs_mean, obs_var, obs_count)
        out = evaluate_fixed(
            fixed_eval_worlds, snapshot.params, snapshot.norm(),
            step=snapshot.step, cohorts=cohorts, perf=perf)
        broad_eval_history.append(float(out['eval_broad_score']))
        out['eval_broad_tail3'] = float(sum(broad_eval_history[-3:]) / len(broad_eval_history[-3:]))
        last_eval_step = step

        best_ckpt_timing = {'device_get_s': 0.0, 'pickle_write_s': 0.0, 'total_s': 0.0, 'bytes': 0.0, 'saved': 0.0}
        if outranks_best(float(out['eval_broad_score']), best_eval_score):
            best_eval_score = float(out['eval_broad_score'])
            best_ckpt_timing = save_checkpoint(
                best_eval_path, snapshot.state,
                {'step': snapshot.step, 'policy_step': snapshot.step, 'version': version,
                 'best_eval_score': best_eval_score,
                 'instant_eval_score': out['eval_broad_score'], 'evaluated_policy': True,
                 'lr_scale': lr_scale},
                snapshot.norm(),
                curriculum_state=governor.state_dict(),
                health_state=health.state_dict(),
                field_state=field.state_dict(), require_finite_state=True)
            if not best_ckpt_timing.get('saved', 0.0):
                raise RuntimeError('Best-policy publication refused: non-finite actor or optimizer state')
            perf.add('checkpoint_best_device_get', best_ckpt_timing['device_get_s'])
            perf.add('checkpoint_best_pickle_write', best_ckpt_timing['pickle_write_s'])
            perf.add('checkpoint_best_total', best_ckpt_timing['total_s'])

        perf.add('fixed_eval_total', out['perf_fixed_eval_total_s'])
        out['perf_best_checkpoint_total_s'] = float(best_ckpt_timing['total_s'])
        out['perf_best_checkpoint_device_get_s'] = float(best_ckpt_timing['device_get_s'])
        out['perf_best_checkpoint_pickle_s'] = float(best_ckpt_timing['pickle_write_s'])
        latest_eval.update(out)
        append_metrics_csv(run_dir / 'fixed_eval.csv', {'step': step, **out})
        if perf.enabled:
            write_json(run_dir / 'performance.json', {'step': step, 'cumulative': perf.row(prefix='')})

        print(f"{LOG_PREFIX} fixed-eval step={step:,} score={out['eval_broad_score']:.3f} tail3={out['eval_broad_tail3']:.3f} "
              f"core={out['eval_core_survival']:.3f} dyn80={out['eval_dynamic80_survival']:.3f} "
              f"combo70={out['eval_combo70_survival']:.3f}/{out['eval_combo70_yaw_mae']:.3f}yaw "
              f"res75={out['eval_resilience75_survival']:.3f} robust40={out['eval_robust40_survival']:.3f} "
              f"growth={out['eval_growth110_survival']:.3f}/{out['eval_growthcombo110_survival']:.3f} "
              f"eval_wall={out.get('perf_fixed_eval_total_s',0.0):.1f}s", flush=True)

    def maybe_cloud_eval(step: int, force: bool = False) -> None:
        nonlocal last_cloud_step, cloud_rows
        if not cloud_enabled or cloud_eval_env is None or cloud_eval_trainer is None:
            return
        every = max(0, int(CLOUD_EVAL_EVERY_STEPS))
        if not force and (every <= 0 or step - last_cloud_step < every):
            return
        count = int(args.cloud_final_probes if force else args.cloud_probes_per_eval)
        if count <= 0:
            return
        t0_cloud = time.perf_counter()
        field_before_exam = field.fingerprint()

        # Screen no-disturbance controls first; only spend expensive disturbed
        # rollouts on command anchors that are already demonstrably competent.
        panel_id = 0 if force else max(0, int(step // max(1, every)) - 1)
        # Let the training-derived field choose
        # *questions* spanning its predicted competence bands, but never let
        # held-out answers update the field. If the map is still too young,
        # fall back to the deterministic paired panel.
        bundles=max(1,int(np.ceil(count/5)))
        if force:
            # Final comparison is a COMMON deterministic panel across every
            # trainer branch.  Field-selected calibration commands are useful
            # longitudinally, but ranking policies on different questions would
            # be scientifically invalid.  We still annotate the common controls
            # with the live field's prediction, so the same final exam also
            # measures self-map calibration.
            requested = cloud_make_common_recovery_panel(count, source='final_common_recovery')
            for spec in requested:
                pred,ev = field.predict_command(float(spec['cmd_vx']),float(spec['cmd_vy']),float(spec['cmd_yaw']))
                spec['field_predicted_competence']=pred
                spec['field_effective_evidence']=ev
        else:
            calibration_cmds = field.calibration_commands(bundles)
            if calibration_cmds:
                requested = cloud_make_calibration_panel(calibration_cmds, panel_id=panel_id,
                                                          source='field_calibration')
            else:
                requested = cloud_make_panel(count, panel_id=panel_id, source='routine_panel')
        controls = [x for x in requested if x.get('family') == 'none']
        snapshot = PolicySnapshot(step, state, obs_mean, obs_var, obs_count)
        base_key = jax.random.PRNGKey(int(EVAL_SEED) + 44000)
        control_keys = [jax.random.fold_in(base_key, int(x.get('probe_index', i)))
                        for i, x in enumerate(controls)]
        control_rows = cloud_evaluate_specs_batch(
            cloud_eval_trainer, cloud_eval_env, snapshot.params,
            snapshot.obs_mean, snapshot.obs_var ** 0.5,
            control_keys, int(CLOUD_HORIZON), controls, chunk_size=int(CLOUD_BATCH_POINTS))
        for r in control_rows:
            r['policy_step'] = int(step); r['forced_final'] = int(bool(force))

        competent_bundles = {int(r.get('bundle_id', -1)) for r in control_rows
                             if float(r.get('recovery_rate', 0.0)) >= 0.80
                             and float(r.get('survival', 0.0)) >= 0.95}
        disturbed = [x for x in requested if x.get('family') != 'none'
                     and int(x.get('bundle_id', -1)) in competent_bundles]
        disturb_keys = [jax.random.fold_in(base_key, int(x.get('probe_index', i)) + 1000003)
                        for i, x in enumerate(disturbed)]
        disturbed_rows = cloud_evaluate_specs_batch(
            cloud_eval_trainer, cloud_eval_env, snapshot.params,
            snapshot.obs_mean, snapshot.obs_var ** 0.5,
            disturb_keys, int(CLOUD_HORIZON), disturbed, chunk_size=int(CLOUD_BATCH_POINTS)) if disturbed else []
        for r in disturbed_rows:
            r['policy_step'] = int(step); r['forced_final'] = int(bool(force))

        new_rows = control_rows + disturbed_rows
        cloud_rows.extend(new_rows)
        summary = cloud_summarize_rows(cloud_rows)  # also annotates matched controls in-place
        write_rows_csv(run_dir / 'heldout_capability_cloud.csv', cloud_rows)
        cal_pairs=[(float(r.get('field_predicted_competence',np.nan)),float(r.get('node_window_rate',np.nan)))
                   for r in control_rows if np.isfinite(float(r.get('field_predicted_competence',np.nan)))
                   and np.isfinite(float(r.get('node_window_rate',np.nan)))]
        cal_mae=float(np.mean([abs(a-b) for a,b in cal_pairs])) if cal_pairs else -1.0
        cal_brier=float(np.mean([(a-b)**2 for a,b in cal_pairs])) if cal_pairs else -1.0
        summary.update({'policy_step': int(step), 'points': len(cloud_rows),
                        'requested_panel_points': len(requested), 'executed_panel_points': len(new_rows),
                        'envs_per_point': int(CLOUD_ENVS), 'horizon_steps': int(CLOUD_HORIZON),
                        'field_calibration_points':len(cal_pairs),'field_calibration_mae':cal_mae,
                        'field_calibration_brier':cal_brier,'field_calibration_target':'node_window_rate'})
        latest_eval['field_calibration_mae']=cal_mae
        latest_eval['field_calibration_brier']=cal_brier
        latest_eval['field_calibration_points']=float(len(cal_pairs))
        write_json(run_dir / 'heldout_capability_cloud.json', summary)
        last_cloud_step = int(step)
        elapsed = time.perf_counter() - t0_cloud
        perf.add('capability_cloud', elapsed)
        field_after_exam = field.fingerprint()
        if field_after_exam != field_before_exam:
            raise RuntimeError('held-out capability exam mutated the live capability field; quarantine invariant violated')

        valid=[r for r in new_rows if int(float(r.get('resilience_valid',0) or 0)) == 1]
        baseline_ok=float(len(competent_bundles) / max(1, len(control_rows)))
        mean_rec=float(np.mean([float(r['recovery_rate']) for r in valid])) if valid else 0.0
        worst=float(np.mean([float(r.get('recovery_worst_direction',0)) for r in valid])) if valid else 0.0
        print(f"{LOG_PREFIX} capability-cloud step={step:,}: controls={len(control_rows)} "
              f"disturbed={len(disturbed_rows)}/{len(requested)-len(controls)} baseline_ok={baseline_ok:.3f} "
              f"valid_recovery={mean_rec:.3f} worst_dir={worst:.3f} calibration_mae={cal_mae:.3f} wall={elapsed:.1f}s", flush=True)

    while total_steps < args.steps:
        maybe_fixed_eval(total_steps)
        maybe_cloud_eval(total_steps)
        host_summary_t0 = time.perf_counter()
        row = summarize_rollout(rollout_metrics, rollout_params_for_metrics)
        # Continuous capability review: ordinary PPO experience updates the
        # decaying command/recovery/maneuver field every rollout.  This is the
        # broad review loop; no simulator pause is required.
        field_t0 = time.perf_counter()
        field_diag = field.update_from_rollout(
            rollout_metrics, step=total_steps, control_dt=CONTROL_DT,
            exclude_field_taught=not bool(args.field_evidence_from_taught))
        perf.add('capability_field_update', time.perf_counter() - field_t0)
        row.update(field_diag)
        reward_mean = float(np.asarray(aux.get('rollout_reward_mean', 0.0)))
        recent_rewards.append(reward_mean)
        recent_rewards = recent_rewards[-64:]
        # Add PPO, normalisation, evaluator and birth-cohort diagnostics
        # before the governor sees the row. Everything here describes the
        # SAME rollout/policy/task distribution.
        augment_rollout_row(
            row, aux, rollout_params_for_metrics=rollout_params_for_metrics,
            rollout_lr_scale_for_metrics=rollout_lr_scale_for_metrics,
            norm_diag=norm_diag, latest_eval=latest_eval, obs_norm_mode=OBS_NORM_MODE)
        perf.add('host_rollout_summary', time.perf_counter() - host_summary_t0)
        governor_t0 = time.perf_counter()
        health_ppo = health.observe_ppo(row)
        report = governor.update(row, kl_ok_to_expand=bool(health_ppo['kl_ok_to_expand']))
        health_collapse = health.observe_collapse(float(report['fall_ema_posture']))
        report.update({k: v for k, v in health_ppo.items() if not k.startswith('kl_') or k in ('kl_stopped_frac_seen', 'kl_stress_chronic')})
        report.update(health_collapse)
        if health_ppo['kl_became_chronic']:
            print(f"{LOG_PREFIX} WARNING: ppo_kl_stopped_frac stuck above {health.KL_STRESS_EXPAND_FLOOR:.2f} for "
                  f"{health.KL_STRESS_CHRONIC_WINDOW} updates -- ignoring the KL-stress gate so it can't freeze "
                  "expansion indefinitely. Retreat is unaffected (driven by fall rate/competence only).", flush=True)
        params = params_with_field(total_steps)
        attach_field_influence(report, params)
        p = env.params_array(params)
        perf.add('governor_and_params', time.perf_counter() - governor_t0)

        # Canonical recovery snapshot.  This is behavior-equivalent to the old
        # best-known-good / best-effort / best-seen files, but stores only the
        # one checkpoint rollback would actually choose.
        overall_competence = min(report['competence_steady'], report['competence_transition'], report['competence_resilience'])
        comp_window.append((overall_competence, float(report['core_competence'])))
        comp_window = comp_window[-health.RECOVERY_WINDOW:]
        if len(comp_window) >= health.RECOVERY_WINDOW:
            comp_med = float(np.median([c for c, _ in comp_window]))
            core_med = float(np.median([c for _, c in comp_window]))
        else:
            comp_med = core_med = -1.0

        recovery_rank, recovery_tier = health.recovery_candidate(comp_med, core_med)

        if recovery_rank > best_recovery_rank:
            best_recovery_rank = recovery_rank
            save_checkpoint(
                run_dir / 'checkpoints' / 'recovery.pkl', rollout_policy_state,
                {'step': total_steps, 'run_name': run_name,
                 'competence': comp_med, 'core_competence': core_med,
                 'recovery_tier': recovery_tier, 'recovery_rank': list(recovery_rank),
                 'lr_scale': rollout_lr_scale_for_metrics, 'evaluated_policy': True},
                evaluated_obs_norm,
                curriculum_state=rollout_curriculum_state_for_metrics,
                health_state=rollout_health_state_for_metrics,
                field_state=rollout_field_state_for_metrics)
            print(f"{LOG_PREFIX} recovery snapshot <- {recovery_tier} at step={total_steps:,} "
                  f"(median-of-{health.RECOVERY_WINDOW} competence={comp_med:.2f}, core={core_med:.2f})", flush=True)

        # Auto-recovery. The governor can only see the symptom (stepping-
        # collapse sustained past ordinary retreat's ability to fix it); it
        # has no access to network weights, so the actual rollback happens
        # here. This guards against a real failure mode: a policy getting
        # stuck for tens of millions of steps after apparently learning to
        # balance-and-avoid-stepping rather than walk, with ordinary
        # per-regime retreat unable to pull it back out on its own.
        if args.allow_rollback and report['needs_recovery']:
            recovered = False
            if recovery_path.exists():
                try:
                    best_payload = load_checkpoint(recovery_path)
                    if payload_train_state_is_finite(best_payload):
                        loaded_state = best_payload['state']
                        state = state._replace(params=jax.device_put(loaded_state['params']),
                                                opt_state=jax.device_put(loaded_state['opt_state']))
                        # Full temporal rewind: network, optimiser, curriculum,
                        # governor histories/EMAs, capability field, observation
                        # normaliser and LR controller. Current-schema recovery
                        # points are complete or rejected; partial rewind is
                        # not a supported state.
                        governor.load_state_dict(best_payload['curriculum_state'])
                        health.load_state_dict(best_payload['health_state'])
                        field = CapabilityField.from_state_dict(best_payload['field_state'])
                        if best_payload.get('obs_norm') is not None:
                            obs_mean = jnp.asarray(best_payload['obs_norm']['mean'], dtype=jnp.float32)
                            obs_var = jnp.asarray(best_payload['obs_norm']['var'], dtype=jnp.float32)
                            obs_count = float(best_payload['obs_norm']['count'])
                        lr_scale = float(best_payload['meta']['lr_scale'])
                        # Revert to the previous good state AND shift the RNG --
                        # rolling back only params/opt_state/curriculum would
                        # leave every env's live physical state exactly as the
                        # COLLAPSED policy left it (mid-fall, tangled, whatever
                        # it was), for the just-reverted policy to inherit.
                        # env.reset() alone wouldn't fix this either: it always
                        # resets from the same fixed self.key0, so calling it
                        # naively here would replay IDENTICAL initial conditions
                        # every time recovery fires, not shift anything.
                        # jax.random.fold_in with total_steps (monotonically
                        # increasing, so never coincidentally repeats) derives a
                        # genuinely different key each time, deterministically.
                        fresh_key = jax.random.fold_in(env.key0, total_steps)
                        fresh_params_array = env.params_array(params_with_field(total_steps))
                        fresh_env_state = env.reset_jit(fresh_key, fresh_params_array)
                        state = state._replace(env_state=fresh_env_state, obs=env.obs(fresh_env_state))
                        best_step = best_payload.get('meta', {}).get('step', '?')
                        print(f"{LOG_PREFIX} AUTO-RECOVERY: stepping-collapse sustained for "
                              f"{int(report['stepping_collapse_streak'])} updates despite repeated retreats -- "
                              f"rolled back network + curriculum to the {best_payload.get('meta', {}).get('recovery_tier', 'recovery')} snapshot from "
                              f"step={best_step} (competence={best_payload.get('meta', {}).get('competence', '?')}). "
                              f"Every env RESET with a fresh RNG derived from step={total_steps:,} (not the run's "
                              f"original seed) -- a genuinely different attempt, not a replay of the same luck "
                              f"that led here, and no env inherits whatever physical state the collapsed policy "
                              f"left it in.", flush=True)
                        recovered = True
                    else:
                        print(f"{LOG_PREFIX} AUTO-RECOVERY triggered but {recovery_path.name} contains non-finite "
                              f"params -- skipping recovery, continuing with ordinary retreat only.", flush=True)
                except Exception as exc:
                    print(f"{LOG_PREFIX} AUTO-RECOVERY triggered but failed to load {recovery_path.name} ({exc!r}) "
                          f"-- continuing with ordinary retreat only.", flush=True)
            else:
                print(f"{LOG_PREFIX} AUTO-RECOVERY triggered after {int(report['stepping_collapse_streak'])} "
                      f"collapsed updates, but recovery.pkl does not exist yet -- curriculum retreat continues.", flush=True)
            # Only clear the streak when a rollback actually happened.
            # Clearing it unconditionally would mean a FAILED attempt (e.g.
            # no snapshot to load, the case just above) forces a blind wait
            # for a whole new streak before trying again, even though
            # nothing about the situation changed. Leaving the streak alone
            # lets update()'s own normal per-update bookkeeping (+1 if
            # still collapsed, reset to 0 the moment a rollout genuinely
            # isn't) decide when to stop retrying.
            if recovered:
                health.note_recovery_performed()
            params = params_with_field(total_steps)
            p = env.params_array(params)
            if recovered:
                # don't immediately re-save a "best" from the state we just
                # loaded: drop the trailing window so the next save has to be
                # earned from health.RECOVERY_WINDOW fresh post-recovery rollouts.
                comp_window = []

        if total_steps - last_log >= log_every_steps:
            last_log = total_steps
            elapsed = time.time() - start
            # fps must be steps-*this-process* / elapsed-*this-process*.
            # Using absolute total_steps (which starts at resumed_step, not
            # 0, after --resume) divided by this process's short elapsed
            # time would produce nonsense fps spikes right after a resume.
            wall_fps = (total_steps - resumed_step) / max(1e-9, elapsed)
            train_seconds = float(perf.total.get('train_update', 0.0)) if perf.enabled else 0.0
            train_fps = ((total_steps - resumed_step) / max(1e-9, train_seconds)) if train_seconds > 0.0 else wall_fps
            # Surfaces optax.apply_if_finite's running counter, so a
            # skipped (non-finite) gradient update shows up in the log
            # instead of silently vanishing. Any nonzero value here means
            # something upstream produced a NaN/Inf gradient this run --
            # worth investigating even though the skip itself kept the
            # network safe.
            nan_grad_skips = int(np.asarray(state.opt_state.total_notfinite))
            if nan_grad_skips > last_seen_nan_skips[0]:
                print(f"{LOG_PREFIX} WARNING: optimizer skipped a non-finite gradient update "
                      f"(total_notfinite={nan_grad_skips}, was {last_seen_nan_skips[0]}) at step={total_steps:,}. "
                      f"Network was NOT corrupted -- the update was discarded instead of applied. "
                      f"Worth checking whether this correlates with a specific curriculum/task extreme.", flush=True)
                last_seen_nan_skips[0] = nan_grad_skips
            # Surfaces the KL-early-stopping guard firing. A nonzero
            # fraction here means at least one minibatch this update
            # exceeded KL_HARD_STOP and further updates were skipped for
            # the rest of this rollout's epochs -- the policy was kept
            # closer to the data it was actually collected under. Worth
            # watching: if it fires repeatedly and steadily, that's a sign
            # of real, ongoing instability (lr/epochs/minibatches too
            # aggressive for the current reward/task regime), not a one-off.
            kl_stopped_frac = float(np.asarray(aux.get('kl_stopped', 0.0)))
            if kl_stopped_frac > 0.0:
                print(f"{LOG_PREFIX} WARNING: KL early-stopping fired this update (stopped_frac={kl_stopped_frac:.2f}, "
                      f"approx_kl={float(np.asarray(aux.get('approx_kl', 0.0))):.4f}) at step={total_steps:,}. "
                      f"Further updates were skipped this rollout to avoid walking the policy too far from "
                      f"the data it was collected under. Fine as a one-off; investigate lr/epochs/minibatches "
                      f"if this keeps firing.", flush=True)
            log_row = make_metrics_row(
                step=total_steps, elapsed=elapsed, wall_fps=wall_fps, train_fps=train_fps,
                reward=float(np.mean(recent_rewards)), row=row, report=report, aux=aux,
                nan_grad_skips=nan_grad_skips, num_envs=args.envs,
                jax_backend=jax.default_backend(), perf_row=perf.row(), base_lr=args.ppo_lr)
            log_io_t0 = time.perf_counter()
            append_metrics_csv(run_dir / 'metrics.csv', log_row)
            write_json(run_dir / 'envelope_summary.json', {'step': total_steps, 'params': {k:v for k,v in params.items() if not k.startswith('field_') or k.endswith('_frac')}, 'report': report, 'last_row': log_row})
            field.save_npz(run_dir / 'capability_field_latest.npz')
            field.write_summary(run_dir / 'capability_field_summary.json', step=total_steps,
                                mode=args.capability_field_mode,
                                node_frac=float(params.get('field_node_frac',0.0)),
                                edge_frac=float(params.get('field_edge_frac',0.0)))
            perf.add('log_io', time.perf_counter() - log_io_t0)
            if perf.enabled:
                write_json(run_dir / 'performance.json', {'step': total_steps, 'cumulative': perf.row(prefix='')})
            print(format_line(version, total_steps, float(np.mean(recent_rewards)), row, report, train_fps, wall_fps, nan_grad_skips), flush=True)

        if save_every_steps > 0 and total_steps - last_save >= save_every_steps:
            last_save = total_steps
            obs_norm = {'mean': obs_mean, 'var': obs_var, 'count': obs_count}
            # apply_if_finite should make this unreachable in practice (a
            # non-finite update is skipped before it ever reaches
            # state.params), but checking again here, right at the point of
            # writing the checkpoint, costs almost nothing and closes the
            # loop completely rather than relying on a single guard.
            ck_timing = save_checkpoint(
                run_dir / 'checkpoints' / 'latest.pkl', state,
                {'step': total_steps, 'run_name': run_name, 'lr_scale': lr_scale}, obs_norm,
                curriculum_state=governor.state_dict(), health_state=health.state_dict(),
                field_state=field.state_dict(), require_finite_state=True)
            perf.add('checkpoint_latest_device_get', ck_timing['device_get_s'])
            perf.add('checkpoint_latest_pickle_write', ck_timing['pickle_write_s'])
            perf.add('checkpoint_latest_total', ck_timing['total_s'])
            if not ck_timing.get('saved', 0.0):
                print(f"{LOG_PREFIX} WARNING: refusing to overwrite checkpoints/latest.pkl at step={total_steps:,} "
                      f"-- host copy contains non-finite params despite the optimizer guard. "
                      f"The same device->host transfer used for checkpointing performed the validation, "
                      f"so capability bookkeeping adds no duplicate rollout transfer.", flush=True)

        if snapshot_every_steps > 0 and total_steps - last_snapshot >= snapshot_every_steps:
            last_snapshot = total_steps
            obs_norm = {'mean': obs_mean, 'var': obs_var, 'count': obs_count}
            snap_timing = save_rotating_snapshot(
                run_dir / 'checkpoints', state,
                {'step': total_steps, 'run_name': run_name, 'lr_scale': lr_scale}, obs_norm,
                curriculum_state=governor.state_dict(), health_state=health.state_dict(),
                field_state=field.state_dict(), step=total_steps,
                keep=max(0, int(SNAPSHOT_KEEP)))
            perf.add('checkpoint_snapshot_device_get', snap_timing['device_get_s'])
            perf.add('checkpoint_snapshot_pickle_write', snap_timing['pickle_write_s'])
            perf.add('checkpoint_snapshot_total', snap_timing['total_s'])

        # Freeze provenance for the rollout we are ABOUT to collect. The
        # next loop iteration's row must be attributed to these exact values,
        # even though PPO/governor/normalizer may all move afterwards.
        rollout_params_for_metrics = params_with_field(total_steps)
        rollout_curriculum_state_for_metrics = governor.state_dict()
        rollout_health_state_for_metrics = health.state_dict()
        rollout_field_state_for_metrics = field.state_dict()
        rollout_lr_scale_for_metrics = float(lr_scale)
        evaluated_obs_norm = {'mean': obs_mean, 'var': obs_var, 'count': obs_count}
        old_mean, old_std = obs_mean, obs_var ** 0.5
        train_update_t0 = time.perf_counter()
        state, rollout_metrics, aux, raw_obs, rollout_policy_state = trainer['train_update'](
            state, p, obs_mean, old_std, lr_scale)
        if perf.enabled:
            # JAX dispatch is asynchronous; block one small result so this timer
            # means actual completed GPU work instead of Python enqueue time.
            state.obs.block_until_ready()
            perf.add('train_update', time.perf_counter() - train_update_t0)
        metric_transfer_t0 = time.perf_counter()
        # All PPO/rollout diagnostics are host-side from here onward. Fetch the
        # pytree once instead of issuing dozens of tiny np.asarray device gets.
        rollout_metrics, aux = jax.device_get((rollout_metrics, aux))
        perf.add('host_metric_transfer', time.perf_counter() - metric_transfer_t0)
        # Controller step. `approx_kl_final` is the last minibatch of the
        # last epoch, i.e. total divergence from the policy this rollout
        # was collected under -- the quantity a trust region is about. The
        # dead band (0.5x-2x of target) stops the lr oscillating on noise.
        kl_final = float(np.asarray(aux.get('approx_kl_final', 0.0)))
        if np.isfinite(kl_final):
            if kl_final > 2.0 * LR_TARGET_UPDATE_KL:
                lr_scale /= PPO_LR_ADJUST
            elif kl_final < 0.5 * LR_TARGET_UPDATE_KL:
                lr_scale *= PPO_LR_ADJUST
            lr_scale = float(min(LR_SCALE_MAX, max(LR_SCALE_MIN, lr_scale)))
        # Observation-normalizer experiment. A curriculum can change the
        # meaning of normalised coordinates even with frozen policy weights;
        # branches may freeze this transformation after a chosen step.
        norm_t0 = time.perf_counter()
        norm_frozen = (args.freeze_obs_norm_after_steps > 0 and
                       total_steps >= args.freeze_obs_norm_after_steps)
        if not norm_frozen:
            obs_mean, obs_var, obs_count = update_running_stats(obs_mean, obs_var, obs_count, raw_obs)
        new_std = obs_var ** 0.5
        norm_diag = {
            'obs_norm_update_mean_l2': float(np.linalg.norm(np.asarray(obs_mean - old_mean))),
            'obs_norm_update_std_l2': float(np.linalg.norm(np.asarray(new_std - old_std))),
            'obs_norm_frozen': float(norm_frozen),
            'obs_norm_anchor_mean_l2': 0.0,
            'obs_norm_anchor_std_l2': 0.0,
        }
        next_total_steps = total_steps + args.envs * ROLLOUT_STEPS
        if obs_anchor is None and next_total_steps >= FOUNDATION_STEPS:
            obs_anchor = (np.asarray(obs_mean).copy(), np.asarray(new_std).copy())
        if obs_anchor is not None:
            norm_diag['obs_norm_anchor_mean_l2'] = float(np.linalg.norm(np.asarray(obs_mean) - obs_anchor[0]))
            norm_diag['obs_norm_anchor_std_l2'] = float(np.linalg.norm(np.asarray(new_std) - obs_anchor[1]))
        state.obs.block_until_ready()
        perf.add('obs_norm_and_sync', time.perf_counter() - norm_t0)
        total_steps = next_total_steps

    # final write. Evaluate and report the exact policy that generated the
    # final rollout before allowing that rollout to update governor stats.
    maybe_fixed_eval(total_steps, force=True)
    maybe_cloud_eval(total_steps, force=True)
    row = summarize_rollout(rollout_metrics, rollout_params_for_metrics)
    row.update(field.update_from_rollout(
        rollout_metrics, step=total_steps, control_dt=CONTROL_DT,
        exclude_field_taught=not bool(args.field_evidence_from_taught)))
    augment_rollout_row(
            row, aux, rollout_params_for_metrics=rollout_params_for_metrics,
            rollout_lr_scale_for_metrics=rollout_lr_scale_for_metrics,
            norm_diag=norm_diag, latest_eval=latest_eval, obs_norm_mode=OBS_NORM_MODE)
    health_ppo = health.observe_ppo(row)
    report = governor.update(row, kl_ok_to_expand=bool(health_ppo['kl_ok_to_expand']))
    report.update({k: v for k, v in health_ppo.items() if not k.startswith('kl_') or k in ('kl_stopped_frac_seen', 'kl_stress_chronic')})
    report.update(health.observe_collapse(float(report['fall_ema_posture'])))
    params = params_with_field(total_steps)
    attach_field_influence(report, params)
    elapsed = time.time() - start
    wall_fps = (total_steps - resumed_step) / max(1e-9, elapsed)
    train_seconds = float(perf.total.get('train_update', 0.0)) if perf.enabled else 0.0
    train_fps = ((total_steps - resumed_step) / max(1e-9, train_seconds)) if train_seconds > 0.0 else wall_fps
    nan_grad_skips = int(np.asarray(state.opt_state.total_notfinite))
    final_row = make_metrics_row(
        step=total_steps, elapsed=elapsed, wall_fps=wall_fps, train_fps=train_fps,
        reward=float(np.mean(recent_rewards) if recent_rewards else 0.0), row=row, report=report, aux=aux,
        nan_grad_skips=nan_grad_skips, num_envs=args.envs,
        jax_backend=jax.default_backend(), perf_row=perf.row(), base_lr=args.ppo_lr)
    append_metrics_csv(run_dir / 'metrics.csv', final_row)
    write_json(run_dir / 'envelope_summary.json', {'step': total_steps, 'params': {k:v for k,v in params.items() if not k.startswith('field_') or k.endswith('_frac')}, 'report': report, 'last_row': final_row})
    final_ck_timing = save_checkpoint(
        run_dir / 'checkpoints' / 'latest.pkl', state,
        {'step': total_steps, 'run_name': run_name, 'lr_scale': lr_scale},
        {'mean': obs_mean, 'var': obs_var, 'count': obs_count},
        curriculum_state=governor.state_dict(), health_state=health.state_dict(),
        field_state=field.state_dict(), require_finite_state=True)
    if not final_ck_timing.get('saved', 0.0):
        raise RuntimeError('Refusing to publish a non-finite final checkpoint; prior validated checkpoint remains intact')
    perf.add('checkpoint_final_device_get', final_ck_timing['device_get_s'])
    perf.add('checkpoint_final_pickle_write', final_ck_timing['pickle_write_s'])
    perf.add('checkpoint_final_total', final_ck_timing['total_s'])
    promote_checkpoint(run_dir / 'checkpoints' / 'latest.pkl',
                       run_dir / 'checkpoints' / 'final.pkl')
    field.save_npz(run_dir / 'capability_field_final.npz')
    field.write_summary(run_dir / 'capability_field_summary.json', step=total_steps,
                        mode=args.capability_field_mode,
                        node_frac=float(params.get('field_node_frac',0.0)),
                        edge_frac=float(params.get('field_edge_frac',0.0)))
    if perf.enabled:
        write_json(run_dir / 'performance.json', {'step': total_steps, 'cumulative': perf.row(prefix='')})
    print(format_line(version, total_steps, final_row['reward_mean'], row, report, train_fps, wall_fps, nan_grad_skips), flush=True)
    print(f'{LOG_PREFIX} training complete', flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"EMU {VERSION} internal trainer")
    parser.add_argument("--config", required=True, help="serialized TrainingConfig JSON")
    ns = parser.parse_args(argv)
    return train(TrainingConfig.load(Path(ns.config)))


if __name__ == '__main__':
    raise SystemExit(main())
