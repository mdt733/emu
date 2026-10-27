"""One source-pinned training lineage, from cold foundation to earned growth.

The previous multi-arm tournament driver is archived in Git history.
This module owns only the stable single-run CLI, provenance and resume.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, replace
from pathlib import Path

from .checkpoint_io import (git_provenance, resolve_target_checkpoint,
                            run_training, update_boundary_at_or_after)
from .contracts import (source_repo_root, OBS_SCHEMA_LEGACY,
                        OBS_SCHEMA_SENSOR, OBS_SCHEMA_TERRAIN)
from .training_config import (TrainingConfig, foundation_config, mature_config,
                              training_sensor_contract, verify_existing_phase_config)


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
    unified = bool(getattr(args, 'unified_capabilities', False))
    foundation_steps=int(getattr(args,'foundation_steps',20_000_000))
    foundation_mix=str(getattr(args,'foundation_mix','main'))
    teaching_schedule=str(getattr(args,'teaching_schedule','balanced'))
    velocity_i_ramp_s=float(getattr(args,'velocity_i_ramp_s',0.4))
    velocity_i_multiplier=float(getattr(args,'velocity_i_multiplier',1.0))
    if (not math.isfinite(velocity_i_ramp_s) or
            not 0.1 <= velocity_i_ramp_s <= 5.0 or
            not math.isfinite(velocity_i_multiplier) or
            not 1.0 <= velocity_i_multiplier <= 3.0 or
            (velocity_i_multiplier > 1.0 and not unified)):
        raise ValueError('velocity I requires unified learning, ramp 0.1..5s, multiplier 1..3')
    fixed_eval_every_steps=int(getattr(args,'fixed_eval_every_steps',60_000_000))
    if fixed_eval_every_steps < 1_000_000:
        raise ValueError('fixed eval cadence must be >=1M world steps')
    if foundation_steps not in (10_000_000,20_000_000):
        raise ValueError('foundation must be 10M or 20M world steps')
    if foundation_mix not in ('main','command-heavy','curious'):
        raise ValueError('unknown foundation mix')
    if teaching_schedule not in ('balanced','staged','terrain-early','steering'):
        raise ValueError('unknown teaching schedule')
    if not unified and (foundation_steps!=20_000_000 or foundation_mix!='main'):
        raise ValueError('foundation experiments require --unified-capabilities')
    requested_pitch=getattr(args, 'pitch_deg', None)
    # Main-style gentle cold posture, without changing mature sim2real runs.
    pitch_deg=float((8.0 if unified or args.profile != 'sim2real' else 30.0)
                    if requested_pitch is None else requested_pitch)
    if not math.isfinite(pitch_deg) or not 0.0 < pitch_deg <= 30.0:
        raise ValueError('--pitch-deg must be within (0, 30] degrees')
    weight=float(getattr(args,'motor_guidance_weight',0.0))
    if not math.isfinite(weight) or not 0.0 <= weight <= 0.10:
        raise ValueError('--motor-guidance-weight must be in [0, 0.10]')
    grade=float(getattr(args,'contact_slope_deg',0.0))
    if not math.isfinite(grade) or not 0.0 <= grade <= 5.0:
        raise ValueError('--contact-slope-deg must be in [0, 5] degrees')
    sensor_actor = bool(getattr(args, 'sensor_actor', False))
    terrain_actor = bool(getattr(args, 'terrain_actor', False))
    terrain_max = float(getattr(args,'terrain_step_max_m',0.0))
    if unified and (not terrain_actor or args.profile != 'sim2real'):
        raise ValueError('unified capability learning requires sim2real and terrain_360')

    if sensor_actor and terrain_actor:
        raise ValueError('--sensor-actor and --terrain-actor are mutually exclusive')
    if not math.isfinite(terrain_max) or not 0.0 <= terrain_max <= .08:
        raise ValueError('--terrain-step-max-m must be between 0 and 0.08')
    if terrain_actor and not 0.001 <= terrain_max <= .08:
        raise ValueError('--terrain-actor needs --terrain-step-max-m above zero')
    if terrain_max > 0 and (grade > 0 or getattr(args,'contact_terrain_radius',0.0)>0):
        raise ValueError('finite steps and inclined floor are not yet compatible')
    if (sensor_actor or terrain_actor) and args.profile not in ('foundation', 'sim2real'):
        raise ValueError('--sensor-actor requires --profile foundation or sim2real')
    if grade > 0 and args.profile != 'sim2real':
        raise ValueError('real contact slope requires --profile sim2real')
    terrain_radius=float(getattr(args,'contact_terrain_radius',0.0))
    if not math.isfinite(terrain_radius) or not 0.0 <= terrain_radius <= 1.0:
        raise ValueError('--contact-terrain-radius must be in [0, 1]')
    if terrain_radius > 0 and args.profile != 'sim2real':
        raise ValueError('per-world real contact terrain requires --profile sim2real')
    if terrain_radius > 0 and grade > 0:
        raise ValueError('shared real incline and per-world real terrain are mutually exclusive')
    return replace(cfg, steps=args.steps, attitude_profile='v5015',
                   observation_schema=(OBS_SCHEMA_TERRAIN if terrain_actor else
                                       OBS_SCHEMA_SENSOR if sensor_actor else OBS_SCHEMA_LEGACY),
                   terrain_step_max_m=terrain_max, unified_capability_focus=unified,
                   foundation_steps=foundation_steps, foundation_mix=foundation_mix,
                   teaching_schedule=teaching_schedule,
                   velocity_i_ramp_s=velocity_i_ramp_s,
                   velocity_i_multiplier=velocity_i_multiplier,
                   fixed_eval_every_steps=fixed_eval_every_steps,
                   # Source main foundation: steady-first low-gain gait. The
                   # mature caps remain physical *headroom*, not birth tasks.
                   difficulty_base_step=.0014 if unified else cfg.difficulty_base_step,
                   fall_target=.30 if unified else cfg.fall_target,
                   # Keep the proven 20M command/phase economics rather than
                   # accidentally inheriting mature speed limits.
                   vx_hi_max=.40 if unified else cfg.vx_hi_max,
                   vx_lo_min=-.12 if unified else cfg.vx_lo_min,
                   vy_max_mps=.14 if unified else cfg.vy_max_mps,
                   yaw_max_rad_s=.90 if unified else cfg.yaw_max_rad_s,
                   freeze_obs_norm_after_steps=0 if unified else cfg.freeze_obs_norm_after_steps,
                   cloud_probes_per_eval=0 if unified else cfg.cloud_probes_per_eval,
                   cloud_final_probes=0 if unified else cfg.cloud_final_probes,
                   combo_probe_frac=.08 if unified else cfg.combo_probe_frac,
                   phase_speed_gain=.75 if unified else cfg.phase_speed_gain,
                   regime_command_frac=.60 if unified else cfg.regime_command_frac,
                   athletic_rehearsal_frac=0.0 if unified else cfg.athletic_rehearsal_frac,
                   combo_mix_frac=0.0 if unified else cfg.combo_mix_frac,
                   frontier_vx_hi_safety=1.4 if unified else cfg.frontier_vx_hi_safety,
                   frontier_vx_lo_safety=.55 if unified else cfg.frontier_vx_lo_safety,
                   frontier_vy_safety=.55 if unified else cfg.frontier_vy_safety,
                   frontier_yaw_safety=3.5 if unified else cfg.frontier_yaw_safety,
                   frontier_combo_safety=2.0 if unified else cfg.frontier_combo_safety,
                   frontier_push_safety_n=4.0 if unified else cfg.frontier_push_safety_n,
                   # Main's foundation still sees ≤0.70N; the unified
                   # governor may later earn real larger pushes.
                   push_max_force_n=4.0 if unified else cfg.push_max_force_n,
                   capability_field_mode='shadow' if unified else cfg.capability_field_mode,
                   pitch_command_max_deg=pitch_deg, motor_guidance_weight=weight,
                   contact_slope_deg=grade,domain_randomization=terrain_radius > 0,
                   domain_terrain_radius=terrain_radius,
                   domain_terrain_mode='contact' if terrain_radius > 0 else 'gravity')


def _validated_same_source_extension(config: TrainingConfig, old: TrainingConfig,
                                    root: Path, *, resume: str,
                                    provenance: dict) -> bool:
    """Extend only a FINISHED, identical-source unified run's target.

    No changes to the policy, optimizer, observations, reward or teacher.
    The only allowed mutation is increasing the requested total world-steps.
    Require a complete original final checkpoint, not merely an optimistic
    console message or a newer snapshot from an unfinished campaign.
    """
    if (resume != 'auto' or not config.unified_capability_focus or
            os.environ.get('EMU_UNIFIED_CONTINUOUS') != 'cold-growth-approved'
            or config.steps <= old.steps):
        return False
    a,b=asdict(old),asdict(config)
    for ignored in ('resume','device','steps'):
        a.pop(ignored,None);b.pop(ignored,None)
    if a != b:
        return False
    meta_file=root/'run_meta.json'
    final=root/'checkpoints/final.pkl'
    if not meta_file.is_file() or not final.is_file():
        return False
    prior_meta=json.loads(meta_file.read_text(encoding='utf-8'))
    if (prior_meta.get('source_provenance') or {}).get('git_commit') != provenance['commit']:
        return False
    from .checkpoint_io import load_checkpoint
    payload=load_checkpoint(
        final, validate=True,
        expected_observation_schema=config.observation_schema,
        expected_sensor_contract=training_sensor_contract(config))
    return (int(payload['meta']['step']) ==
            update_boundary_at_or_after(old.steps,old.envs)
            and ((payload['meta'].get('source_provenance') or {})
                 .get('git_commit')) == provenance['commit'])


def verify_experiment_fork(parent: TrainingConfig, child: TrainingConfig) -> None:
    """Keep real physical/actor contracts stable, not arbitrary research choices.

    Teaching emphasis, reward/controller economics, PPO settings, seed,
    experiment duration and the parent checkpoint age are freely selectable.
    """
    # Only observation/command MEANING is an immutable checkpoint
    # dependency. Command envelopes, hardware/domain variation, phase
    # economics, pushes and teaching are exactly what researchers may
    # deliberately vary. Saved configs/provenance describe those differences.
    fixed=('observation_schema','unified_capability_focus',
           'attitude_profile')
    bad={name:(getattr(parent,name),getattr(child,name))
         for name in fixed if getattr(parent,name)!=getattr(child,name)}
    if bad:
        raise RuntimeError(f'Incompatible research fork physical/actor contract: {bad}')


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
    target_extended=False
    if cfg_path.is_file():
        old_config=TrainingConfig.load(cfg_path)
        previous = asdict(old_config)
        wanted = asdict(config)
        for variable in ('resume', 'device'):
            previous.pop(variable, None)
            wanted.pop(variable, None)
        if previous != wanted:
            target_extended=_validated_same_source_extension(
                config,old_config,root,resume=resume,provenance=provenance)
            if not target_extended:
                raise RuntimeError(
                    'Existing training configuration differs; only a validated '
                    'same-source increase of a FINISHED continuous run target '
                    'with --resume auto is allowed')
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
        verify_existing_phase_config(
            root, config, allow_target_extension=target_extended)
        if target_extended:
            print(f'Same-source target extension approved: '
                  f'{old_config.steps:,} -> {config.steps:,} world steps, '
                  'continuing from the validated completed final.pkl')

    finished, partial, expected = resolve_target_checkpoint(
        root, target_steps=config.steps, envs=config.envs, label=root.name,
        expected_observation_schema=config.observation_schema,
        expected_sensor_contract=training_sensor_contract(config))
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
        payload = load_checkpoint(
            parent, validate=True, expected_observation_schema=config.observation_schema,
            expected_sensor_contract=training_sensor_contract(config))
        source = payload['meta'].get('source_provenance', {}).get('git_commit')
        if source != provenance['commit']:
            raise RuntimeError(
                'External checkpoint source differs from current Git commit. '
                'Use one cold-start run with --resume auto; do not transplant '
                'old learning state across code revisions.')
        # A newly created study is allowed to fork a source-matched
        # checkpoint at ANY training age; verify that it uses the same
        # robot/sensor/physical envelope, not the same lesson recipe.
        if parent.parent != root/'checkpoints':
            if parent.parent.name != 'checkpoints':
                raise RuntimeError('research parent must be inside checkpoints/')
            parent_cfg=parent.parent.parent/'training_config.json'
            if not parent_cfg.is_file():
                raise RuntimeError('research parent has no training_config.json')
            verify_experiment_fork(TrainingConfig.load(parent_cfg), config)
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
        label=Path(config.run_dir).name,
        expected_observation_schema=config.observation_schema,
        expected_sensor_contract=training_sensor_contract(config))
    if final is None:
        raise RuntimeError('Trainer returned but no validated final checkpoint reached the target')
    print(f'Completed validated run: {final}')
    # This report reads recorded learning evidence without importing PPO or
    # changing the validated checkpoint. Independent physical exams remain separate.
    try:
        from .results import write_report
        for report_path in write_report(Path(config.run_dir)):
            print(f'Run report: {report_path}')
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f'WARNING: checkpoint is validated but offline report failed: {exc}; '
              f'run python -m emu.results --root {config.run_dir}')
    return 0
