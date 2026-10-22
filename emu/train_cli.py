"""A single protected EMU training run: parse profile, validate, resume, launch.

This module owns only the human-facing `emu train` command and its safety
checks. PPO and physics remain in training.py; tournament selection is
in campaign_refork.py. No historical experiment recipes live here.
"""
from __future__ import annotations

import json
import math
import hashlib
import re
from dataclasses import asdict, replace
from pathlib import Path

from .checkpoint_io import (git_provenance, resolve_target_checkpoint,
                            run_training, update_boundary_at_or_after)
from .contracts import (OBS_SCHEMA_LEGACY, OBS_SCHEMA_SENSOR,
                        OBS_SCHEMA_PREVIEW, source_repo_root)
from .training_config import (TrainingConfig, foundation_config,
                              mature_config, training_sensor_contract,
                              verify_existing_phase_config)


def one_run_config(args) -> TrainingConfig:
    """Resolve a named, existing training profile without inventing PPO flags."""
    if args.envs <= 0 or args.steps <= 0:
        raise ValueError('--envs and --steps must both be positive')
    run_dir = Path(args.run_dir).expanduser().resolve()
    if args.profile in ('foundation', 'terrain-bridge'):
        cfg = foundation_config(run_dir=run_dir, seed=args.seed,
                                device=args.device, envs=args.envs)
    elif args.profile in ('mature', 'sim2real'):
        # The pitch envelope is a shared run setting, not a tournament arm.
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
    weight=float(getattr(args,'motor_guidance_weight',0.0))
    if not math.isfinite(weight) or not 0.0 <= weight <= 0.10:
        raise ValueError('--motor-guidance-weight must be in [0, 0.10]')
    grade=float(getattr(args,'contact_slope_deg',0.0))
    if not math.isfinite(grade) or not 0.0 <= grade <= 5.0:
        raise ValueError('--contact-slope-deg must be in [0, 5] degrees')
    preview_actor = bool(getattr(args, 'forward_preview', False))
    preview_masked = bool(getattr(args, 'forward_preview_masked', False))
    if preview_masked and not preview_actor:
        raise ValueError('--forward-preview-masked requires --forward-preview')
    if preview_actor and args.profile not in ('foundation', 'sim2real', 'terrain-bridge'):
        raise ValueError('--forward-preview requires foundation or sim2real')
    sensor_actor = bool(getattr(args, 'sensor_actor', False) or preview_actor)
    if sensor_actor and args.profile not in ('foundation', 'sim2real', 'terrain-bridge'):
        raise ValueError('--sensor-actor requires --profile foundation or sim2real')
    if args.profile == 'terrain-bridge' and not preview_actor:
        raise ValueError('terrain-bridge requires --forward-preview (312-input)')
    if grade > 0 and args.profile != 'sim2real':
        raise ValueError('real contact slope requires --profile sim2real')
    terrain_radius=float(getattr(args,'contact_terrain_radius',0.0))
    if not math.isfinite(terrain_radius) or not 0.0 <= terrain_radius <= 1.0:
        raise ValueError('--contact-terrain-radius must be in [0, 1]')
    if terrain_radius > 0 and args.profile != 'sim2real' and not (
            args.profile == 'foundation' and preview_actor):
        raise ValueError('--contact-terrain-radius requires --profile sim2real, or --profile foundation with --forward-preview (312-input)')
    if terrain_radius > 0 and grade > 0:
        raise ValueError('shared real incline and per-world real terrain are mutually exclusive')
    box_raw=getattr(args,'terrain_box_height_mm',None)
    box_height=-1 if box_raw is None else box_raw
    if type(box_height) is not int or box_height not in (-1,0,5,10,15,20,30,40,50):
        raise ValueError('fixed terrain box height must be -1 or one of 0/5/10/15/20/30/40/50mm')
    if box_height >= 0 and (args.profile != 'sim2real' or not sensor_actor):
        raise ValueError('terrain box PPO research requires sim2real and --sensor-actor')
    if box_height >= 0 and (grade != 0 or terrain_radius != 0):
        raise ValueError('fixed physical box course cannot mix with inclined/random terrain')
    varied=bool(getattr(args, 'terrain_varied_placement', False))
    elevation_mode=str(getattr(args,'terrain_elevation_mode','hurdle'))
    if elevation_mode not in ('hurdle','up','down','mixed'):
        raise ValueError('unknown terrain_elevation_mode')
    adaptive_limit=int(getattr(args, 'terrain_height_ceiling_mm', 0))
    adaptive_flat=bool(getattr(args, 'terrain_height_flat_control',False))
    exposure=float(getattr(args, 'terrain_height_exposure', .60))
    edge=float(getattr(args, 'terrain_height_edge_fraction', .35))
    grow=float(getattr(args, 'terrain_height_growth_mm', 1.))
    initial=float(getattr(args, 'terrain_height_initial_mm', 3.))
    terrain_reward=float(getattr(args, 'terrain_progress_weight', 0.))
    if not 0 <= adaptive_limit <= 150:
        raise ValueError('adaptive terrain height safety must be 0..150mm')
    if not all(math.isfinite(x) for x in (exposure, edge, grow, initial, terrain_reward)):
        raise ValueError('nonfinite adaptive terrain setting')
    if not 0. <= terrain_reward <= .10:
        raise ValueError('terrain-progress-weight must be [0,.10]')
    if terrain_reward and not adaptive_limit:
        raise ValueError('terrain progress teaching requires adaptive real step terrain')
    if not (.05 <= exposure <= .95 and .05 <= edge <= .95 and
            .1 <= grow <= 3. and 0. <= initial <= 15.):
        raise ValueError('adaptive terrain exposure/edge/growth/initial outside lab bounds')
    if adaptive_flat and adaptive_limit:
        raise ValueError('adaptive flat control must have zero height ceiling')
    if varied and (not preview_actor or not (adaptive_limit or adaptive_flat)):
        raise ValueError('varied placement requires 312-input preview + adaptive physical step topology')
    if elevation_mode != 'hurdle' and initial != 0.:
        raise ValueError('real elevation starts at precisely 0mm, not a nonzero jump')
    if elevation_mode != 'hurdle' and (
            not preview_actor or not varied or adaptive_limit <= 0
            or box_height >= 0 or grade != 0 or terrain_radius > 0
            or args.profile not in ('sim2real', 'terrain-bridge')):
        raise ValueError('raised-platform training requires sim2real or terrain-bridge 312-input '
                         'adaptive step heights, varied placement and initially flat contact ground')
    # Mixed contact slopes and adaptive physical boxes require the verified
    # 312-input sensor contract; older actor schemes remain incompatible.
    if (adaptive_limit or adaptive_flat) and (
            box_height >= 0 or grade > 0 or args.profile not in ('sim2real', 'terrain-bridge')
            or not sensor_actor or (terrain_radius > 0 and not preview_actor)):
        raise ValueError('adaptive steps require sim2real; slope+step coupling requires 312-input preview')
    parent_commit = str(getattr(args, 'approve_parent_commit', '') or '')
    parent_sha = str(getattr(args, 'approve_parent_sha256', '') or '')
    if (bool(parent_commit) != bool(parent_sha) or
            (parent_commit and (not re.fullmatch(r'[0-9a-f]{40}', parent_commit)
                                or not re.fullmatch(r'[0-9a-f]{64}', parent_sha)))):
        raise ValueError('cross-source fork requires exact parent source commit and checkpoint SHA256')
    if parent_commit and args.profile != 'terrain-bridge':
        raise ValueError('cross-source parent approval requires terrain-bridge profile')
    return replace(cfg, steps=args.steps, attitude_profile='v5015',
                   approved_parent_git_commit=parent_commit,
                   approved_parent_sha256=parent_sha,
                   observation_schema=(OBS_SCHEMA_PREVIEW if preview_actor else
                                       OBS_SCHEMA_SENSOR if sensor_actor else OBS_SCHEMA_LEGACY),
                   forward_preview_masked=preview_masked,
                   pitch_command_max_deg=pitch_deg, motor_guidance_weight=weight,
                   contact_slope_deg=grade,terrain_box_height_mm=box_height,
                   terrain_random_placement=varied,
                   terrain_elevation_mode=elevation_mode,
                   terrain_height_safety_m=adaptive_limit / 1000.,
                   terrain_adaptive_flat_control=adaptive_flat,
                   terrain_height_initial_m=min(initial, adaptive_limit if adaptive_limit else initial) / 1000.,
                   terrain_height_exposure=exposure,
                   terrain_height_edge_fraction=edge,
                   terrain_height_growth_m=grow / 1000.,
                   terrain_progress_weight=terrain_reward,
                   domain_randomization=terrain_radius > 0,
                   domain_terrain_radius=terrain_radius,
                   domain_terrain_mode='contact' if terrain_radius > 0 else 'gravity')


def verify_flat_parent_terrain_bridge(parent_config: TrainingConfig,
                                     bridge_config: TrainingConfig) -> None:
    """No implicit 'mature' curriculum when a real 0mm axis is introduced.

    For a flat -> elevation handoff, the sole allowed differences are the
    physical terrain sampler, its rating safety/geometry, absolute run target,
    file location and the explicitly pinned historical checkpoint identity.
    Everything that changes the learned walking task, rewards, optimizer,
    observation, pushes and capabilities MUST be identical to the 20M run.
    """
    if (parent_config.terrain_height_safety_m > 0. or
            parent_config.terrain_adaptive_flat_control or
            parent_config.terrain_box_height_mm >= 0):
        raise ValueError('terrain-bridge requires a genuinely flat foundation parent')
    from dataclasses import asdict
    allowed = {
        'run_dir', 'steps', 'resume', 'device',
        'approved_parent_git_commit', 'approved_parent_sha256',
        'terrain_random_placement', 'terrain_elevation_mode',
        'terrain_height_safety_m', 'terrain_height_initial_m',
        'terrain_height_exposure', 'terrain_height_edge_fraction',
        'terrain_height_growth_m', 'terrain_progress_weight',
    }
    original=asdict(parent_config)
    desired=asdict(bridge_config)
    unexpected={key:(original[key],desired[key])
                for key in original if key not in allowed
                and original[key]!=desired[key]}
    if unexpected:
        raise RuntimeError(
            'Non-continuous flat-to-terrain learning envelope; unexpected '
            f'configuration changes: {unexpected}')
    if bridge_config.terrain_height_initial_m != 0.:
        raise RuntimeError('Untrained physical axis must start rated at 0mm')
    if bridge_config.terrain_progress_weight != parent_config.terrain_progress_weight:
        raise RuntimeError('Terrain handoff must not silently change reward weights')


def verify_rated_parent_continuation(parent_config: TrainingConfig,
                                     next_config: TrainingConfig) -> None:
    """A rated step-up stage is another checkpoint, NOT a new lesson.

    Allow only output location, absolute target steps and explicitly pinned
    source-parent identity to change. All training inputs/economics and the
    sampled physical height distribution must remain exactly the same.
    The saved curriculum's *earned* height is restored by training.py,
    never replaced by this static initial-height setting.
    """
    from dataclasses import asdict
    if (parent_config.terrain_elevation_mode != 'up' or
            parent_config.terrain_height_safety_m <= 0. or
            parent_config.observation_schema != OBS_SCHEMA_PREVIEW):
        raise ValueError('overnight rated step-up continuation requires a real 312-input up-stage parent')
    allowed={'run_dir','steps','resume','device',
             'approved_parent_git_commit','approved_parent_sha256'}
    old=asdict(parent_config)
    new=asdict(next_config)
    changed={key:(old[key],new[key]) for key in old
             if key not in allowed and old[key]!=new[key]}
    if changed:
        raise RuntimeError(
            'Non-continuous rated-elevation stage handoff. Refusing to alter '
            f'command, rewards, push, sensor, optimizer, terrain or curriculum: {changed}')


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
    if (root / 'UNSAFE_RESUME.json').is_file():
        raise RuntimeError(
            f'NON-RESUMABLE catastrophic terrain collapse: {root / "UNSAFE_RESUME.json"}. '
            'This checkpoint lineage is quarantined. Do not retry it; '
            'reassess the parent foundation and create a fresh run directory.')
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
            if (not config.approved_parent_git_commit or
                    config.approved_parent_git_commit != source):
                raise RuntimeError('Parent source commit differs; an exact, checksum-pinned terrain-bridge parent approval is required')
        if config.approved_parent_git_commit:
            checksum = hashlib.sha256()
            with parent.open('rb') as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b''):
                    checksum.update(chunk)
            if checksum.hexdigest() != config.approved_parent_sha256:
                raise RuntimeError('Approved parent checkpoint SHA256 mismatch')

        if (config.terrain_elevation_mode != 'hurdle' and
                config.terrain_height_safety_m > 0. and
                config.terrain_height_initial_m == 0.):
            old_config=TrainingConfig.load(
                parent.parent.parent/'training_config.json')
            if old_config.terrain_height_safety_m == 0.:
                verify_flat_parent_terrain_bridge(old_config, config)
            else:
                verify_rated_parent_continuation(old_config, config)
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
    return 0

