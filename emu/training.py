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
stable cold-start training followed by evidence-driven capability growth.

KNOWN DEBT: most of these stages still inhabit a long `train` body.
Keep extraction local to existing modules; avoid new wrapper mazes.
"""
from __future__ import annotations

import argparse
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .checkpoint_io import (
    resume_training_lineage, promote_checkpoint, try_restore_recovery,
    save_checkpoint, save_rotating_snapshot, load_checkpoint,
    source_provenance_from_env,
)
from .environment import EmuMJXEnv
from .domain_randomization import DomainRandomizationConfig, doctrine_dict as domain_doctrine_dict
from .capability_field import CapabilityField
from .contracts import ACTION_DIM, OBS_SCHEMA_TERRAIN, TERRAIN_OBS_DIM, CONTROL_DT, FIELD_NODE_ANCHORS, FIELD_EDGE_ANCHORS
from .training_config import (
    TrainingConfig, doctrine_dict, training_sensor_contract,
    ROLLOUT_STEPS, PPO_EPOCHS, PPO_MINIBATCHES, PPO_LR, PPO_ENTROPY_COST, PPO_VALUE_COST,
    LOG_EVERY_UPDATES, CHECKPOINT_EVERY_UPDATES, SNAPSHOT_EVERY_STEPS, SNAPSHOT_KEEP,
    PROFILE_PERFORMANCE, MAX_EPISODE_STEPS,
    OBS_NORM_MODE, PPO_TARGET_KL, PPO_LR_SCALE_MIN, PPO_LR_SCALE_INIT, PPO_LR_SCALE_MAX, PPO_LR_ADJUST, FOUNDATION_STEPS,
    EVAL_EVERY_STEPS, EVAL_SEED, EVAL_COHORTS, FINAL_EVAL_COHORTS,
    FIELD_HALF_LIFE_STEPS, SENSOR_NOISE_SCALE,
    RECOVERY_TILT_MAX_DEG, RECOVERY_POSITION_MAX_M, RECOVERY_LINVEL_MAX_MPS,
    RECOVERY_ANGVEL_MAX_RAD_S, RECOVERY_JOINT_MAX_DEG,
)
from .curriculum import RegimeGovernor
from .training_health import TrainingHealth
from .capability_focus import (teaching_profile, attention_weights, FOCUS_NAMES,
                               academy_birth_allocation, academy_focus_mix,
                               academy_conditional_birth_share)
from .ppo import (make_trainer, update_running_stats, build_obs_norm_mask,
                  PerfBook, configure_jax, require_backend)
from .results import (
    summarize_rollout, augment_rollout_row,
    ensure_run_dir, write_json, read_csv_rows, format_line,
    attach_field_influence, make_metrics_row, append_training_observation,
    write_capability_field_artifacts, StairCreditAudit,
)
from .evaluation import (
    build_worlds as build_eval_worlds, initial_metrics as initial_eval_metrics, evaluate_fixed,
    EVAL_PERF_TIMER_NAMES, PolicySnapshot, checkpoint_eval_score, record_fixed_exam,
)
from .contracts import ROBOT_XML, VERSION

RUN_VERSION = f'emu_{VERSION}'
LOG_PREFIX = f'[{RUN_VERSION}]'

TRAIN_PERF_TIMER_NAMES = (
    'train_update', 'host_metric_transfer', 'capability_field_update',
    'host_rollout_summary', 'governor_and_params', 'obs_norm_and_sync',
    'log_io', 'fixed_eval_total',
    'checkpoint_best_device_get', 'checkpoint_best_pickle_write', 'checkpoint_best_total',
    'checkpoint_latest_device_get', 'checkpoint_latest_pickle_write', 'checkpoint_latest_total',
    'checkpoint_snapshot_device_get', 'checkpoint_snapshot_pickle_write', 'checkpoint_snapshot_total',
    'checkpoint_final_device_get', 'checkpoint_final_pickle_write', 'checkpoint_final_total',
    *EVAL_PERF_TIMER_NAMES,
)

# ---------------------------------------------------------------------------
# Utilities / artifacts.
# ---------------------------------------------------------------------------

def compatible_academy_evidence_recipe(parent: TrainingConfig,
                                       current: TrainingConfig) -> bool:
    """Same measured stair-success semantics and physical lesson distribution.

    A new *source SHA alone* is not evidence of a changed episode objective:
    a logging-only fork must preserve its completed near/easy evidence.
    This check is necessary but cannot prove code equivalence; a deliberate
    new success-definition implementation must explicitly reset evidence.
    """
    if not parent.science_step_finish_on_transfer:
        return False
    return all(getattr(parent, name) == getattr(current, name)
               for name in (
                   'science_step_finish_on_transfer',
                   'science_step_goal_weight', 'science_step_approach_m',
                   'academy_stream', 'academy_height_cap_m',
                   'academy_course_episode_frames',
                   'terrain_carpet', 'terrain_carpet_prisms',
                   'terrain_step_max_m', 'terrain_lesson',
                   'solver_iterations', 'solver_ls_iterations'))


def env_kwargs_from_args(args) -> dict:
    """One source of truth for the physical/sensor envelope used by training."""
    return dict(
        push_max_force_n=args.push_max_force_n,
        vx_hi_max=args.vx_hi_max, vx_lo_min=args.vx_lo_min,
        vy_max_mps=args.vy_max_mps, yaw_max_rad_s=args.yaw_max_rad_s,
        phase_speed_gain=args.phase_speed_gain,
        motor_guidance_weight=args.motor_guidance_weight,
        terrain_step_max_m=args.terrain_step_max_m,
        unified_capability_focus=args.unified_capability_focus,
        velocity_i_ramp_s=args.velocity_i_ramp_s,
        velocity_i_multiplier=args.velocity_i_multiplier,
        retention_share=args.retention_share,
        terrain_lesson=args.terrain_lesson,
        temporal_vx_reward_share=args.temporal_vx_reward_share,
        stillness_birth_frac=args.stillness_birth_frac,
        science_gate_mode=args.science_gate_mode,
        science_phase_lowcmd_scale=args.science_phase_lowcmd_scale,
        science_phase_sync_strength=args.science_phase_sync_strength,
        science_crouch_allowance_m=args.science_crouch_allowance_m,
        science_stride_efficiency_weight=args.science_stride_efficiency_weight,
        science_supported_cross_bonus=args.science_supported_cross_bonus,
        science_foothold_bonus=args.science_foothold_bonus,
        science_bilateral_support_bonus=args.science_bilateral_support_bonus,
        science_sensor_teacher_bonus=args.science_sensor_teacher_bonus,
        science_followthrough_bonus=args.science_followthrough_bonus,
        science_step_goal_weight=args.science_step_goal_weight,
        science_step_velocity_relief=args.science_step_velocity_relief,
        science_step_approach_m=args.science_step_approach_m,
        science_step_finish_on_transfer=args.science_step_finish_on_transfer,
        outcome_graduation=args.outcome_graduation,
        academy_stream=args.academy_stream,
        terrain_carpet=args.terrain_carpet,
        terrain_carpet_prisms=args.terrain_carpet_prisms,
        solver_iterations=args.solver_iterations,
        solver_ls_iterations=args.solver_ls_iterations,
        carpet_transition_cap_deg=args.carpet_transition_cap_deg,
        academy_height_cap_m=args.academy_height_cap_m,
        academy_course_episode_frames=args.academy_course_episode_frames,
        attitude_profile=args.attitude_profile,
        pitch_command_max_deg=args.pitch_command_max_deg,
        sensor_noise_scale=SENSOR_NOISE_SCALE, observation_schema=args.observation_schema,
        open_frontier=args.open_frontier,
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
            hardware_radius=float(args.domain_hardware_radius),
        ),
    )


# ---------------------------------------------------------------------------
# Run modes.
# ---------------------------------------------------------------------------

def unified_launch_mode(args: TrainingConfig) -> str:
    """Explicit authorisation boundary; deliberately independent of JAX/GPU.

    Prefer one clean cold-start source and run directory across the entire
    training target. The runner independently enforces Git and checkpoint
    provenance; this gate accepts research forks; inspect_run verifies compatibility.
    """
    # Fail before JAX compilation if a caller bypasses the public
    # prism-only CLI and supplies a retired configuration directly.
    if not (args.unified_capability_focus
            and args.observation_schema == OBS_SCHEMA_TERRAIN
            and args.terrain_carpet and args.terrain_carpet_prisms
            and args.academy_stream and args.attitude_profile == 'v5015'):
        raise ValueError('v5045 trainer requires one unified v5015 '
                         '360-input prism carpet with streaming Academy')
    core = (args.device == 'cuda'
            and args.terrain_step_max_m >= .001
            and args.open_frontier)
    if (core and os.environ.get('EMU_UNIFIED_ACCEPTANCE') == 'bounded-ppo-lifecycle'
            and args.envs in (8, 12, 2048)
            and 0 < args.steps <= 2 * args.envs * ROLLOUT_STEPS):
        return 'bounded'
    # One cold run across 20M, 40M, 100M, and beyond. No checkpoint-era
    # unlock: the specialists gradually become active inside this run.
    # The CLI checks identical Git SHA + saved config for same-run resume.
    # Deliberate research must not require rewriting this function for
    # each sensible knob. The CLI/config/env validate physically meaningful
    # ranges and compatibility; a CUDA preflight tests resource constraints.
    # These are REAL launch prerequisites, not a whitelist of permitted
    # learning ideas or arbitrary birth/development ages.
    if (core and os.environ.get('EMU_UNIFIED_CONTINUOUS') == 'cold-growth-approved'
            and args.steps >= 1_000_000 and args.envs > 0
            and args.foundation_steps in (10_000_000,20_000_000)
            and args.foundation_mix in ('main','command-heavy','curious')
            and 0 <= args.seed < 2**31):
        parent = Path(args.resume).resolve() if args.resume else None
        own_dir = Path(args.run_dir).resolve() / 'checkpoints'
        explicit_parent = (
            parent is not None and
            parent.suffix == '.pkl' and parent.parent.name == 'checkpoints')
        if not args.resume or args.resume == 'auto' or explicit_parent:
            return 'continuous'
    raise RuntimeError(
        'PRE-LAUNCH: Unified live training requires the explicit bounded CUDA smoke or '
        'EMU_UNIFIED_CONTINUOUS=cold-growth-approved for a source-pinned '
        '360-input CUDA learner. Explicit '
        'same-source compatible research forks are allowed using --resume <checkpoint>.'
    )


def train(args: TrainingConfig) -> int:
    """Run one checkpoint lineage; no branch-selection logic lives here.

    The main loop is organised in evidence time:
    1. examine the policy that produced the last rollout;
    2. reduce facts, revise next curriculum, and checkpoint if needed;
    3. collect the next rollout and update PPO.
    The first rollout is collected before entering the loop.
    """
    launch_mode = unified_launch_mode(args)
    bounded_acceptance = launch_mode == 'bounded'
    continuous_growth = launch_mode == 'continuous'
    # Main-style 20M cold birth, then a gradual exposure blend. A specialist
    # adjusts its physical target independently once its lessons are active.
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

    # One exact 360-input actor and normalizer/checkpoint identity.
    # Historical 220/232 schemas cannot enter this executable trainer.
    sensor_contract = training_sensor_contract(args)
    if sensor_contract is None:
        raise RuntimeError('v5045 trainer has no 360-input sensor contract')
    obs_dim = TERRAIN_OBS_DIM
    obs_meta = dict(
        checkpoint_schema=9, observation_schema=OBS_SCHEMA_TERRAIN,
        sensor_contract=sensor_contract, obs_dim=obs_dim)
    resumed_payload, resumed_step, already_complete = resume_training_lineage(
        run_dir, resume=getattr(args, 'resume', ''), target_steps=args.steps, envs=args.envs,
        expected_observation_schema=args.observation_schema,
        expected_sensor_contract=sensor_contract)
    if already_complete:
        return 0

    parent_meta = ((resumed_payload or {}).get('meta') or {})
    meta = {'version': version, 'source_provenance': source_provenance_from_env(),
            'ancestor_checkpoint_source': ((parent_meta.get('source_provenance') or {}).get('git_commit')),
            'ancestor_checkpoint_step': parent_meta.get('step'),
            'ancestor_checkpoint_path': args.resume if resumed_payload is not None else None,
            'config': vars(args), 'doctrine': {**doctrine_dict(), **{f'DOMAIN_{k}': v for k, v in domain_doctrine_dict().items()}},
            'resumed_from_step': resumed_step if resumed_payload is not None else None}
    write_json(run_dir / 'run_meta.json', meta)
    perf = PerfBook(enabled=bool(PROFILE_PERFORMANCE), known_names=TRAIN_PERF_TIMER_NAMES)

    env = EmuMJXEnv(str(ROBOT_XML), num_envs=args.envs, seed=args.seed,
                    max_episode_steps=MAX_EPISODE_STEPS,
                    **env_kwargs_from_args(args))
    # A physics topology change is NOT an observation-contract change.
    # Record the actual compiled model so resumed policies and physical
    # fixed-evaluation reports cannot silently confuse the old torso-contact
    # world with the new two-ball-foot-only world.
    meta['mjx_physics'] = {
        'feet_only_contacts': env.feet_only_contacts,
        'reset_forward_capacity': env.reset_forward_capacity,
        'solver_iterations': int(env.mj_model.opt.iterations),
        'solver_ls_iterations': int(env.mj_model.opt.ls_iterations),
        'ngeom': int(env.mj_model.ngeom),
        'padded_contacts_per_world': int(
            env.mjx.make_data(env.mjx_model_template)._impl.contact.geom1.shape[0]),
    }
    write_json(run_dir / 'run_meta.json', meta)
    if env.obs_dim != obs_dim or env.sensor_contract != sensor_contract:
        raise RuntimeError('Observation contract mismatch between trainer and sensor environment')
    trainer = make_trainer(env, seed=args.seed, lr=args.ppo_lr, rollout=ROLLOUT_STEPS, epochs=PPO_EPOCHS, minibatches=PPO_MINIBATCHES,
                           entropy_cost=args.ppo_entropy_cost, value_cost=args.ppo_value_cost)
    # A two-update CUDA acceptance verifies PPO, physical telemetry,
    # complete checkpoint state, warm resume and actor export. The full
    # 10-panel/350-frame fixed exam is for measuring trained walking, and
    # took ~8 minutes on an untrained eight-world smoke. Never pretend
    # this bounded protocol is a held-out walking evaluation.
    # Held-out exams must not allocate five additional 2048-world MJX
    # environments alongside the 2048-world learner on an 8GB GPU.
    # Keep the frozen 10-panel protocol/350-frame horizon/two final cohorts;
    # only independent evaluation batch size is capped for this 20M run.
    eval_args = replace(args, envs=32) if continuous_growth else args
    fixed_eval_worlds = {} if bounded_acceptance else build_eval_worlds(eval_args)
    if continuous_growth:
        print(f'{LOG_PREFIX} one cold-start PPO lineage to {args.steps:,} steps; '
              f'foundation={args.foundation_mix} until {args.foundation_steps:,} steps; '
              'specialists blend in over the next 10M; frozen evaluations '
              'use 32 worlds rather than risking 8GB GPU exhaustion.',
              flush=True)
    if bounded_acceptance:
        print(f'{LOG_PREFIX} bounded lifecycle acceptance: expensive final '
              'walking exams disabled; use independent frozen policy exam '
              'after checkpoint/resume/export verification', flush=True)

    # No track-level flags to validate here -- a regime's difficulty is judged directly by RegimeGovernor, with no scheduling layer in between.
    governor = RegimeGovernor(fall_target=args.fall_target,
                              difficulty_base_step=args.difficulty_base_step,
                              regime_command_frac=args.regime_command_frac,
                              athletic_rehearsal_frac=args.athletic_rehearsal_frac,
                              open_frontier=args.open_frontier,
                              frontier_vx_hi_safety=args.frontier_vx_hi_safety,
                              frontier_vx_lo_safety=args.frontier_vx_lo_safety,
                              frontier_vy_safety=args.frontier_vy_safety, frontier_yaw_safety=args.frontier_yaw_safety,
                              frontier_combo_safety=args.frontier_combo_safety, frontier_push_safety_n=args.frontier_push_safety_n,
                              unified_capability_focus=args.unified_capability_focus,
                              science_step_promotion_mm=args.science_step_promotion_mm,
                              science_step_retreat_mm=args.science_step_retreat_mm,
                              science_step_min_attempts=args.science_step_min_attempts,
                              outcome_mode=args.outcome_graduation)
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
            # A fork must not spend 260M's partially collected old-threshold
            # evidence. Preserve heights and all learned weights. Do NOT
            # erase evidence on later same-source crash recovery.
            ancestor_sha = ((resumed_payload.get('meta', {}).get('source_provenance') or {})
                            .get('git_commit'))
            # Evidence is keyed to the physical lesson's success definition,
            # NOT merely the Python source SHA. v5054 changed that definition,
            # so a legacy source fork must discard incomparable evidence.
            # Conversely a logging-only v5055 research fork from an already
            # v5054-trained actor must NOT silently empty its curriculum.
            # Cross-source preservation requires the parent's saved training
            # recipe to prove matching active step-lesson semantics.
            if (args.science_step_finish_on_transfer and
                    ancestor_sha != source_provenance_from_env().get('git_commit')):
                safe_evidence_fork = False
                if args.resume and args.resume.lower() != 'auto':
                    parent_cfg_path = (
                        Path(args.resume).expanduser().resolve().parent.parent
                        / 'training_config.json')
                    if parent_cfg_path.is_file():
                        try:
                            parent_cfg = TrainingConfig.load(parent_cfg_path)
                            safe_evidence_fork = (
                                compatible_academy_evidence_recipe(
                                    parent_cfg, args))
                        except (ValueError, TypeError, OSError) as exc:
                            print(f'{LOG_PREFIX} parent recipe not verifiable: '
                                  f'{exc}; clearing incompatible evidence',
                                  flush=True)
                if safe_evidence_fork:
                    print(f'{LOG_PREFIX} source-only diagnostic fork: '
                          'preserved already-compatible Academy/STEP evidence',
                          flush=True)
                else:
                    governor.academy_evidence = governor._empty_academy_evidence()
                    governor.terrain_evidence = governor._empty_terrain_evidence()
                    print(f'{LOG_PREFIX} new or unverified step-success recipe: '
                          'reset inherited Academy/STEP evidence '
                          '(height frontiers retained)',flush=True)
            health.load_state_dict(resumed_payload['health_state'])
            print(f"{LOG_PREFIX} --resume: restored curriculum + training health "
                  f"(updates={governor._updates_done_total}, core_bad_streak={governor.core_bad_streak}, "
                  f"kl_history={len(health.kl_stress_history)}, collapse_streak={health.stepping_collapse_streak}).", flush=True)
        except Exception as exc:
            raise RuntimeError(f"--resume curriculum/health restore failed: {exc}") from exc

    # Source-pinned curriculum fork: no 3mm warm-up for the 50mm nursery.
    if args.outcome_graduation:
        governor.params.academy_height_up_m=min(args.academy_height_cap_m,
            max(.050,governor.params.academy_height_up_m))
        governor.params.academy_height_down_m=min(args.academy_height_cap_m,
            governor.params.academy_height_down_m)
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

    def params_with_field(
        step: int, *, teaching_field: CapabilityField | None = None,
    ) -> dict[str, Any]:
        """Build the next lesson for the live OR restored capability field."""
        d = governor.params_dict()
        if args.unified_capability_focus:
            profile = teaching_profile(
                step=step, foundation_steps=args.foundation_steps,
                foundation_mix=args.foundation_mix,
                steady_difficulty=float(governor.params.steady_difficulty))
            d['specialist_mix'] = profile.specialist_fraction
            # The 20M foundation uses original main-style births only.
            # Afterward, honour the explicit run configuration even when
            # resuming a cold-foundation checkpoint whose saved governor
            # still contains athletic_rehearsal_frac=0. This value controls
            # actual training births; it is not a checkpoint network field.
            d['athletic_rehearsal_frac']=(
                args.athletic_rehearsal_frac if profile.adaptive_specialists
                else 0.0)
            d['steering_attention']=(1.0 if (
                args.teaching_schedule=='steering' and
                step>args.foundation_steps) else 0.0)
            weights=list(attention_weights(
                schedule=args.teaching_schedule,step=step,
                foundation_steps=args.foundation_steps))
            # Reserve a deterministic share of all focused births for
            # useful moving, low-disturbance retention (CONTROL role).
            # The ordinary control role remains unchanged in baseline arms.
            if step>args.foundation_steps and args.retention_share>0.:
                base_control=weights[-1]
                other_total=1.-base_control
                target_control=max(base_control,args.retention_share)
                weights=[w*(1.-target_control)/other_total for w in weights[:-1]]
                weights.append(target_control)
            # Academy is an additional episode birth role, not a separate
            # optimizer or policy. Its share is relative to ALL worlds.
            # Original role fractions survive inside non-Academy births.
            academy_probability=0.0
            if args.academy_stream and profile.adaptive_specialists:
                academy_share=academy_birth_allocation(
                    step=step,intro_steps=args.academy_intro_steps,
                    start_steps=args.academy_start_steps,
                    ramp_steps=args.academy_ramp_steps,
                    intro_share=args.academy_intro_share,
                    max_share=args.academy_max_share)
                if args.outcome_graduation:
                    # The mature fork begins meaningful physical lessons immediately.
                    if step>=args.outcome_walk_start_steps:
                        academy_share=max(academy_share,.60)
                # Fast rehearsal was reserved BEFORE focus assignment.
                # Honour 50% of ALL births rather than accidentally 50% of
                # the remaining 75% when 25% of worlds rehearse fast gait.
                conditional,actual_share=academy_conditional_birth_share(
                    academy_share,float(d['athletic_rehearsal_frac']))
                weights,academy_probability=academy_focus_mix(
                    weights,conditional)
            d['academy_episode_probability']=academy_probability
            if args.outcome_graduation:
                competence=max(0.,min(1.,(governor.core_competence-.40)/.25))
                d['outcome_walk_alpha']=(min(1.,max(0.,(step-args.outcome_walk_start_steps)/10_000_000.))*competence)
                d['outcome_stair_alpha']=(min(1.,max(0.,(step-args.outcome_approval_start_steps)/10_000_000.)) if args.outcome_stair_approved else 0.)
                d['outcome_down_share']=(args.outcome_down_share if args.outcome_stair_approved else 0.)
            for name,weight in zip(FOCUS_NAMES,weights):
                d['focus_weight_'+name]=weight
            # Keep the 70-update steady-only bootstrap exactly where the
            # original foundation started. Afterwards the candidate recipe
            # may request a different balance of the SAME ordinary regimes.
            after_bootstrap=(governor._updates_done_total>governor.BOOTSTRAP_UPDATES)
            d['foundation_control_mix']=(
                profile.foundation_control_mix if after_bootstrap else 0.0)
            if profile.regime_mix is not None and after_bootstrap:
                for key,portion in zip(
                    ('regime_frac_steady','regime_frac_transition',
                     'regime_frac_resilience'),profile.regime_mix):
                    d[key]=portion
            # A zero specialist allocation is not a separate PPO or a
            # synthetic zero-gradient loss. It simply sends all worlds
            # through main's ordinary gait sampler. Terrain stays dormant.
            if not profile.adaptive_specialists:
                d['terrain_exposure'] = 0.0
                d['terrain_height_m'] = 0.0
                d['slope_exposure'] = 0.0
                d['slope_max_deg'] = 0.0
        elif step < args.foundation_steps:
            d['terrain_exposure'] = 0.0
            d['terrain_height_m'] = 0.0
        source_field = field if teaching_field is None else teaching_field
        packet = source_field.teaching_packet(
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
    print(f"{LOG_PREFIX} obs_dim={obs_dim} schema={args.observation_schema} action_dim={ACTION_DIM} regimes=steady/transition/resilience + posture", flush=True)
    if args.terrain_step_max_m:
        print(f"{LOG_PREFIX} finite terrain topology allocated (ceiling={args.terrain_step_max_m:.3f}m), "
              "but physical exposure/height remain ZERO until independently proven crossing-based "
              "curriculum activation is implemented.", flush=True)
    print(f"{LOG_PREFIX} compiling MJX/JAX update...", flush=True)
    import jax.numpy as jnp
    obs_mean = jnp.zeros((obs_dim,), dtype=jnp.float32)
    obs_var = jnp.ones((obs_dim,), dtype=jnp.float32)
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
                 'obs_norm_frozen': 0.0, 'obs_norm_terrain_clip_frac': -1.0,
                 'obs_norm_terrain_clip_measured_step': -1.0}
    obs_anchor = None
    terrain_clip_last = -1.0
    terrain_clip_step = -1.0

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
    if warmup_end_steps >= args.foundation_steps:
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
    # The first rollout is real training. Starting the wall timer AFTER it
    # made one-update lifecycle smoke report impossible multi-million FPS.
    # Include the compilation/first-update wall time conservatively; steady
    # training throughput is separately available in perf diagnostics.
    start = t0
    last_log = resumed_step
    last_save = resumed_step
    last_snapshot = resumed_step
    last_seen_nan_skips = [0]  # mutable box so the log block above can update it
    previous_rows=read_csv_rows(run_dir / 'metrics.csv')
    rollback_count=(int(float(previous_rows[-1].get('rollback_count',0)))
                    if previous_rows else 0)
    # Canonical rollback candidate: never overwrite a stronger existing
    # recovery snapshot after resume/fork.
    recovery_path = run_dir / 'checkpoints' / 'recovery.pkl'
    best_recovery_rank = (-1, -1.0)
    if recovery_path.exists():
        try:
            existing_recovery = load_checkpoint(
                recovery_path,
                expected_observation_schema=args.observation_schema,
                expected_sensor_contract=sensor_contract)
            saved_rank = existing_recovery.get('meta', {}).get('recovery_rank', [-1, -1.0])
            best_recovery_rank = (int(saved_rank[0]), float(saved_rank[1]))
        except Exception:
            pass
    comp_window: list[tuple[float, float]] = []
    recent_rewards: list[float] = []
    # Audit EVERY PPO rollout, not just the one in each periodic metrics row.
    # This only aggregates already-transferred scalars; no GPU work or PPO
    # behavior changes. The count is since THIS process began.
    stair_audit = StairCreditAudit()
    log_every_steps = args.envs * ROLLOUT_STEPS * max(1, int(LOG_EVERY_UPDATES))
    save_every_steps = args.envs * ROLLOUT_STEPS * max(1, int(CHECKPOINT_EVERY_UPDATES))
    snapshot_every_steps = max(0, int(SNAPSHOT_EVERY_STEPS))

    # Held-out evaluation state is reporting/artifact state only; it never feeds
    # PPO, curriculum, health or capability-field updates.
    latest_eval = initial_eval_metrics()
    fixed_history = read_csv_rows(run_dir / 'fixed_eval.csv')
    last_eval_step = int(float(fixed_history[-1]['step'])) if fixed_history else resumed_step
    if fixed_history:
        for key in latest_eval:
            if key in fixed_history[-1]:
                latest_eval[key] = float(fixed_history[-1][key])
    broad_eval_history = [float(row['eval_broad_score']) for row in fixed_history[-2:]
                          if row.get('eval_broad_score') not in (None, '')]
    best_eval_score = -1e9

    # Preserve evaluation cadence/history and best-checkpoint monotonicity across
    # same-run process restarts. External forks have no local eval artifacts.
    best_eval_path = run_dir / 'checkpoints' / 'best.pkl'
    if best_eval_path.exists():
        try:
            best_eval_score = checkpoint_eval_score(load_checkpoint(
                best_eval_path, expected_observation_schema=args.observation_schema,
                expected_sensor_contract=sensor_contract).get('meta', {}))
        except Exception:
            pass

    def maybe_fixed_eval(step: int, force: bool = False) -> None:
        nonlocal last_eval_step, best_eval_score
        if bounded_acceptance:
            return
        if not force and step - last_eval_step < args.fixed_eval_every_steps:
            return
        # A full frozen physical exam compiles multiple JAX worlds and can
        # take minutes after the last training progress line. Announce the
        # phase BEFORE the first compile, never leave the console looking hung.
        print(f'{LOG_PREFIX} fixed-eval starting at step={step:,} '
              f'({"final" if force else "scheduled"}; '
              f'{FINAL_EVAL_COHORTS if force else EVAL_COHORTS} cohorts; '
              'training updates paused while independent MJX worlds run)',flush=True)
        snapshot = PolicySnapshot(step, state, obs_mean, obs_var, obs_count)
        out, best_eval_score = record_fixed_exam(
            snapshot, worlds=fixed_eval_worlds,
            cohorts=FINAL_EVAL_COHORTS if force else EVAL_COHORTS,
            run_dir=run_dir, broad_history=broad_eval_history,
            best_score=best_eval_score, best_path=best_eval_path,
            curriculum_state=governor.state_dict, health_state=health.state_dict,
            field_state=field.state_dict, lr_scale=lr_scale, version=version,
            checkpoint_meta=obs_meta, perf=perf, log_prefix=LOG_PREFIX,
            progress=lambda message: print(f'{LOG_PREFIX} {message}',flush=True))
        last_eval_step = step
        latest_eval.update(out)

    while total_steps < args.steps:
        maybe_fixed_eval(total_steps)
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
        stair_audit.observe(row, aux)
        perf.add('host_rollout_summary', time.perf_counter() - host_summary_t0)
        governor_t0 = time.perf_counter()
        health_ppo = health.observe_ppo(row)
        report = governor.advance_curriculum(
            row, step=total_steps, max_height_m=args.terrain_step_max_m,
            foundation_steps=args.foundation_steps,
            kl_ok_to_expand=bool(health_ppo['kl_ok_to_expand']),
            academy_max_height_m=(args.academy_height_cap_m if args.academy_stream else 0.))
        health_collapse = health.observe_collapse(float(report['fall_ema_posture']))
        report.update({k: v for k, v in health_ppo.items() if not k.startswith('kl_') or k in ('kl_stopped_frac_seen', 'kl_stress_chronic')})
        report.update(health_collapse)
        if health_ppo['kl_became_chronic']:
            print(f"{LOG_PREFIX} WARNING: ppo_kl_stopped_frac stuck above {health.KL_STRESS_EXPAND_FLOOR:.2f} for "
                  f"{health.KL_STRESS_CHRONIC_WINDOW} updates -- ignoring the KL-stress gate so it can't freeze "
                  "expansion indefinitely. Retreat is unaffected (driven by fall rate/competence only).", flush=True)
        params = params_with_field(total_steps)
        if args.unified_capability_focus:
            # The fixed experimental nursery mix may override the ordinary
            # governor's NEXT-birth regime allocation. Report what worlds
            # will really sample, never only the governor's unused proposal.
            for name in ('steady','transition','resilience'):
                key='regime_frac_'+name
                report[key]=float(params[key])
            report['foundation_control_mix']=float(params['foundation_control_mix'])
            report['academy_expected_birth_fraction']=float(
                (1.-params['athletic_rehearsal_frac'])*
                params['specialist_mix']*params['focus_weight_step']*
                params['academy_episode_probability'])
            for name in FOCUS_NAMES:
                report['focus_alloc_'+name]=float(params['focus_weight_'+name])
            report['teaching_profile']=teaching_profile(
                step=total_steps,foundation_steps=args.foundation_steps,
                foundation_mix=args.foundation_mix,
                steady_difficulty=governor.params.steady_difficulty).name
        report['rollback_count']=rollback_count
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
                {'step': total_steps, 'run_name': run_name, **obs_meta,
                 'competence': comp_med, 'core_competence': core_med,
                 'recovery_tier': recovery_tier, 'recovery_rank': list(recovery_rank),
                 'lr_scale': rollout_lr_scale_for_metrics, 'evaluated_policy': True},
                evaluated_obs_norm,
                curriculum_state=rollout_curriculum_state_for_metrics,
                health_state=rollout_health_state_for_metrics,
                field_state=rollout_field_state_for_metrics)
            print(f"{LOG_PREFIX} recovery snapshot <- {recovery_tier} at step={total_steps:,} "
                  f"(median-of-{health.RECOVERY_WINDOW} competence={comp_med:.2f}, core={core_med:.2f})", flush=True)

        # One explicit temporal rewind: policy/optimiser + teacher + health +
        # field + observation coordinates + LR, then fresh physics for EVERY
        # world. If recovery fails, retain the collapse streak and retry later.
        # The historical first-20M foundation deliberately disabled
        # checkpoint rollbacks; resume and later unified recovery remain.
        if (args.allow_rollback and
                (not args.unified_capability_focus or total_steps >= args.foundation_steps)
                and report['needs_recovery']):
            restored = try_restore_recovery(
                recovery_path, state=state, governor=governor,
                health=health, env=env, step=total_steps, report=report,
                obs_mean=obs_mean, obs_var=obs_var, obs_count=obs_count,
                task_params_for_field=lambda new_field: params_with_field(
                    total_steps, teaching_field=new_field),
                log_prefix=LOG_PREFIX)
            if restored is not None:
                rollback_count += 1
                print(f'{LOG_PREFIX} rollback_count={rollback_count} '
                      f'at step={total_steps:,} (step clock is NOT rewound)',
                      flush=True)
                state, field = restored.state, restored.field
                obs_mean, obs_var = restored.obs_mean, restored.obs_var
                obs_count, lr_scale = restored.obs_count, restored.lr_scale
                # A rollback must earn a fresh complete recovery window.
                comp_window = []
            params = params_with_field(total_steps)
            p = env.params_array(params)

        if total_steps - last_log >= log_every_steps:
            last_log = total_steps
            elapsed = time.time() - start
            # fps must be steps-*this-process* / elapsed-*this-process*.
            # Using absolute total_steps (which starts at resumed_step, not
            # 0, after --resume) divided by this process's short elapsed
            # time would produce nonsense fps spikes right after a resume.
            wall_fps = (total_steps - resumed_step) / max(1e-9, elapsed)
            train_seconds = float(perf.total.get('train_update', 0.0)) if perf.enabled else 0.0
            # train_update perf only measures subsequent updates, never the
            # compile+first update. Its numerator must cover the SAME steps.
            measured_train_steps = max(0, total_steps - resumed_step - args.envs * ROLLOUT_STEPS)
            train_fps = (measured_train_steps / train_seconds
                         if train_seconds > 0.0 and measured_train_steps > 0
                         else wall_fps)
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
                reward=float(np.mean(recent_rewards)),
                row={**row, **stair_audit.summary()}, report=report, aux=aux,
                nan_grad_skips=nan_grad_skips, num_envs=args.envs,
                jax_backend=jax.default_backend(), perf_row=perf.row(), rollout_steps=ROLLOUT_STEPS, base_lr=args.ppo_lr)
            log_io_t0 = time.perf_counter()
            append_training_observation(
                run_dir, step=total_steps, params=params, report=report, metrics_row=log_row)
            write_capability_field_artifacts(
                run_dir, field, step=total_steps, mode=args.capability_field_mode,
                params=params, label='latest')
            perf.add('log_io', time.perf_counter() - log_io_t0)
            if perf.enabled:
                write_json(run_dir / 'performance.json', {'step': total_steps, 'cumulative': perf.row(prefix='')})
            print(format_line(version, total_steps, float(np.mean(recent_rewards)),
                              row, report, train_fps, wall_fps, nan_grad_skips),
                  flush=True)
            # No new synchronization: reuse existing cumulative PerfBook timers.
            # The console's primary fps includes first-use JIT; this separates
            # compiled full-PPO rate from host reporting/metrics overhead.
            if perf.enabled and measured_train_steps > 0:
                print(
                    f"{LOG_PREFIX}   speed steady_full_ppo={train_fps:.0f}/s "
                    f"wall_since_start={wall_fps:.0f}/s "
                    f"cumulative_s(ppo/transfer/summary/field/governor/norm)="
                    f"{perf.total.get('train_update',0.):.1f}/"
                    f"{perf.total.get('host_metric_transfer',0.):.1f}/"
                    f"{perf.total.get('host_rollout_summary',0.):.1f}/"
                    f"{perf.total.get('capability_field_update',0.):.1f}/"
                    f"{perf.total.get('governor_and_params',0.):.1f}/"
                    f"{perf.total.get('obs_norm_and_sync',0.):.1f}",
                    flush=True)

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
                {'step': total_steps, 'run_name': run_name, 'lr_scale': lr_scale, **obs_meta}, obs_norm,
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
                {'step': total_steps, 'run_name': run_name, 'lr_scale': lr_scale, **obs_meta}, obs_norm,
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
        # Only training-derived KL adjusts the NEXT update's LR; frozen
        # examination results are never consulted by this controller.
        lr_scale = health.adjust_learning_rate(
            lr_scale,
            approx_kl=float(np.asarray(aux.get('approx_kl_final', 0.0))),
            target_kl=LR_TARGET_UPDATE_KL, factor=PPO_LR_ADJUST,
            min_scale=LR_SCALE_MIN, max_scale=LR_SCALE_MAX)
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
            'obs_norm_terrain_clip_frac': terrain_clip_last,
            'obs_norm_terrain_clip_measured_step': terrain_clip_step,
        }
        # Read-only sample at the handover: does 10M of flat terrain
        # make raw ToF ranges / grades saturate the +/-10 normalized clip?
        # Sample only 128 observations every ~2M steps around the boundary.
        # We do NOT silently alter the observation contract or optimizer.
        norm_probe_interval=2_000_000
        if (args.unified_capability_focus
                and args.foundation_steps-2_000_000 <= total_steps
                    <= args.foundation_steps+6_000_000
                and total_steps // norm_probe_interval !=
                    (total_steps-args.envs*ROLLOUT_STEPS)//norm_probe_interval):
            from .contracts import (
                TERRAIN_OBS_FIELDS, TERRAIN_RANGE_FIELDS, HISTORY_LEN)
            sensor_names=set(TERRAIN_RANGE_FIELDS)|{
                'ground_forward_grade_rad','ground_sideways_grade_rad'}
            base=np.asarray(
                [i for i,name in enumerate(TERRAIN_OBS_FIELDS)
                 if name in sensor_names],dtype=np.int32)
            indices=np.concatenate([
                base+i*len(TERRAIN_OBS_FIELDS) for i in range(HISTORY_LEN)])
            mask=build_obs_norm_mask(args.observation_schema)
            indices=indices[mask[indices]>.5]
            if len(indices):
                sampled=np.asarray(raw_obs[:128,indices])
                std=np.asarray(old_std)[indices]
                centre=np.asarray(old_mean)[indices]
                z=(sampled-centre)/np.maximum(std,1e-8)
                terrain_clip_last=float(np.mean(np.abs(z)>=10.))
                terrain_clip_step=float(total_steps)
                norm_diag['obs_norm_terrain_clip_frac']=terrain_clip_last
                norm_diag['obs_norm_terrain_clip_measured_step']=terrain_clip_step
                if norm_diag['obs_norm_terrain_clip_frac']>.10:
                    print(f'{LOG_PREFIX} WARNING: '
                          'terrain observation normalization clips '
                          f'{100*norm_diag["obs_norm_terrain_clip_frac"]:.1f}% '
                          f'around step {total_steps:,}',flush=True)
        next_total_steps = total_steps + args.envs * ROLLOUT_STEPS
        if obs_anchor is None and next_total_steps >= args.foundation_steps:
            obs_anchor = (np.asarray(obs_mean).copy(), np.asarray(new_std).copy())
        if obs_anchor is not None:
            norm_diag['obs_norm_anchor_mean_l2'] = float(np.linalg.norm(np.asarray(obs_mean) - obs_anchor[0]))
            norm_diag['obs_norm_anchor_std_l2'] = float(np.linalg.norm(np.asarray(new_std) - obs_anchor[1]))
        state.obs.block_until_ready()
        perf.add('obs_norm_and_sync', time.perf_counter() - norm_t0)
        total_steps = next_total_steps

    # final write. The last rollout was not reduced by the while-loop; in a
    # one-update smoke the reward history is otherwise EMPTY and prints 0.000,
    # although PPO really saw nonzero rewards. Include the last rollout once.
    recent_rewards.append(float(np.asarray(aux['rollout_reward_mean'])))
    recent_rewards = recent_rewards[-64:]
    # Evaluate and report the exact policy that generated the final rollout
    # before allowing that rollout to update governor stats.
    print(f'{LOG_PREFIX} PPO TARGET REACHED at step={total_steps:,}; '
          'starting final read-only physics exams and checkpoint publication. '
          'The next log lines report each exam panel; no new PPO updates '
          'are running during this phase.',flush=True)
    maybe_fixed_eval(total_steps, force=True)
    print(f'{LOG_PREFIX} final physical exams finished; '
          'summarizing final rollout and writing validated final checkpoint...',flush=True)
    row = summarize_rollout(rollout_metrics, rollout_params_for_metrics)
    row.update(field.update_from_rollout(
        rollout_metrics, step=total_steps, control_dt=CONTROL_DT,
        exclude_field_taught=not bool(args.field_evidence_from_taught)))
    augment_rollout_row(
            row, aux, rollout_params_for_metrics=rollout_params_for_metrics,
            rollout_lr_scale_for_metrics=rollout_lr_scale_for_metrics,
            norm_diag=norm_diag, latest_eval=latest_eval, obs_norm_mode=OBS_NORM_MODE)
    stair_audit.observe(row, aux)
    health_ppo = health.observe_ppo(row)
    report = governor.advance_curriculum(
        row, step=total_steps, max_height_m=args.terrain_step_max_m,
        foundation_steps=args.foundation_steps,
        kl_ok_to_expand=bool(health_ppo['kl_ok_to_expand']),
        academy_max_height_m=(args.academy_height_cap_m if args.academy_stream else 0.))
    report.update({k: v for k, v in health_ppo.items() if not k.startswith('kl_') or k in ('kl_stopped_frac_seen', 'kl_stress_chronic')})
    report.update(health.observe_collapse(float(report['fall_ema_posture'])))
    params = params_with_field(total_steps)
    if args.unified_capability_focus:
        for name in ('steady','transition','resilience'):
            key='regime_frac_'+name
            report[key]=float(params[key])
        report['foundation_control_mix']=float(params['foundation_control_mix'])
        report['academy_expected_birth_fraction']=float(
            (1.-params['athletic_rehearsal_frac'])*
            params['specialist_mix']*params['focus_weight_step']*
            params['academy_episode_probability'])
        for name in FOCUS_NAMES:
            report['focus_alloc_'+name]=float(params['focus_weight_'+name])
        report['teaching_profile']=teaching_profile(
            step=total_steps,foundation_steps=args.foundation_steps,
            foundation_mix=args.foundation_mix,
            steady_difficulty=governor.params.steady_difficulty).name
    report['rollback_count']=rollback_count
    attach_field_influence(report, params)
    elapsed = time.time() - start
    wall_fps = (total_steps - resumed_step) / max(1e-9, elapsed)
    train_seconds = float(perf.total.get('train_update', 0.0)) if perf.enabled else 0.0
    measured_train_steps = max(0, total_steps - resumed_step - args.envs * ROLLOUT_STEPS)
    train_fps = (measured_train_steps / train_seconds
                 if train_seconds > 0.0 and measured_train_steps > 0
                 else wall_fps)
    nan_grad_skips = int(np.asarray(state.opt_state.total_notfinite))
    final_row = make_metrics_row(
        step=total_steps, elapsed=elapsed, wall_fps=wall_fps, train_fps=train_fps,
        reward=float(np.mean(recent_rewards)),
        row={**row, **stair_audit.summary()}, report=report, aux=aux,
        nan_grad_skips=nan_grad_skips, num_envs=args.envs,
        jax_backend=jax.default_backend(), perf_row=perf.row(), rollout_steps=ROLLOUT_STEPS, base_lr=args.ppo_lr)
    append_training_observation(
        run_dir, step=total_steps, params=params, report=report, metrics_row=final_row)
    final_ck_timing = save_checkpoint(
        run_dir / 'checkpoints' / 'latest.pkl', state,
        {'step': total_steps, 'run_name': run_name, 'lr_scale': lr_scale, **obs_meta},
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
    write_capability_field_artifacts(
        run_dir, field, step=total_steps, mode=args.capability_field_mode,
        params=params, label='final')
    if perf.enabled:
        write_json(run_dir / 'performance.json', {'step': total_steps, 'cumulative': perf.row(prefix='')})
    print(format_line(version, total_steps, final_row['reward_mean'],
                      row, report, train_fps, wall_fps, nan_grad_skips),
          flush=True)
    print(f'{LOG_PREFIX} training complete', flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"EMU {VERSION} internal trainer")
    parser.add_argument("--config", required=True, help="serialized TrainingConfig JSON")
    ns = parser.parse_args(argv)
    return train(TrainingConfig.load(Path(ns.config)))


if __name__ == '__main__':
    raise SystemExit(main())
