"""One source-pinned training lineage, from cold foundation to earned growth.

The previous multi-arm tournament driver is archived in Git history.
This module owns only the stable single-run CLI, provenance and resume.
"""
from __future__ import annotations

import json
import math
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
                   # Source main foundation: steady-first low-gain gait. The
                   # mature caps remain physical *headroom*, not birth tasks.
                   difficulty_base_step=.0014 if unified else cfg.difficulty_base_step,
                   fall_target=.30 if unified else cfg.fall_target,
                   combo_probe_frac=.08 if unified else cfg.combo_probe_frac,
                   phase_speed_gain=.75 if unified else cfg.phase_speed_gain,
                   regime_command_frac=.60 if unified else cfg.regime_command_frac,
                   athletic_rehearsal_frac=0.0 if unified else cfg.athletic_rehearsal_frac,
                   combo_mix_frac=0.0 if unified else cfg.combo_mix_frac,
                   capability_field_mode='shadow' if unified else cfg.capability_field_mode,
                   pitch_command_max_deg=pitch_deg, motor_guidance_weight=weight,
                   contact_slope_deg=grade,domain_randomization=terrain_radius > 0,
                   domain_terrain_radius=terrain_radius,
                   domain_terrain_mode='contact' if terrain_radius > 0 else 'gravity')


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
        label=Path(config.run_dir).name,
        expected_observation_schema=config.observation_schema,
        expected_sensor_contract=training_sensor_contract(config))
    if final is None:
        raise RuntimeError('Trainer returned but no validated final checkpoint reached the target')
    print(f'Completed validated run: {final}')
    # A finished standalone experiment deserves the same human-readable
    # evidence as tournament arms. This only reads CSV/JSON and hashes the
    # finished checkpoint; reporting cannot invalidate successful training.
    try:
        from .results import write_report
        for report_path in write_report(Path(config.run_dir)):
            print(f'Run report: {report_path}')
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f'WARNING: checkpoint is validated but offline report failed: {exc}; '
              f'run python -m emu.results --root {config.run_dir}')
    return 0
