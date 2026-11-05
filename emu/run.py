"""One source-pinned training lineage, from cold foundation to earned growth.

The previous multi-arm tournament driver is archived in Git history.
This module owns only the stable single-run CLI, provenance and resume.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, replace
from pathlib import Path

from .checkpoint_io import (git_provenance, resolve_target_checkpoint,
                            run_training, update_boundary_at_or_after)
from .contracts import source_repo_root, OBS_SCHEMA_TERRAIN
from .training_config import (TrainingConfig, prism_base_config,
                              training_sensor_contract, verify_existing_phase_config)


def one_run_config(args) -> TrainingConfig:
    """Resolve a named, existing training profile without inventing PPO flags."""
    if args.envs <= 0 or args.steps <= 0:
        raise ValueError('--envs and --steps must both be positive')
    run_dir = Path(args.run_dir).expanduser().resolve()
    if args.profile != 'sim2real':
        raise ValueError(
            'v5045 prism-only training requires --profile sim2real '
            '(archived foundation/mature profiles are Git-history-only)')
    # One current 360-input profile. Foundation is an *age* within the same
    # run, never a separate obsolete 220-input training configuration.
    cfg = prism_base_config(run_dir=run_dir, steps=args.steps,
                            seed=args.seed, device=args.device, envs=args.envs,
                            resume=None)
    solver_iterations=getattr(args,"solver_iterations",None)
    solver_ls_iterations=getattr(args,"solver_ls_iterations",None)
    for key,value in (("solver_iterations",solver_iterations),
                      ("solver_ls_iterations",solver_ls_iterations)):
        if value is not None and (type(value) is not int or not 1<=value<=100):
            raise ValueError(f"--{key.replace('_','-')} requires integer 1..100")
    unified = bool(getattr(args, 'unified_capabilities', False))
    foundation_steps=int(getattr(args,'foundation_steps',20_000_000))
    foundation_mix=str(getattr(args,'foundation_mix','main'))
    teaching_schedule=str(getattr(args,'teaching_schedule','balanced'))
    velocity_i_ramp_s=float(getattr(args,'velocity_i_ramp_s',0.4))
    velocity_i_multiplier=float(getattr(args,'velocity_i_multiplier',1.0))
    growth_recipe=str(getattr(args,'growth_recipe','legacy'))
    retention_share=float(getattr(args,'retention_share',0.0))
    rehearsal_arg=getattr(args,'athletic_rehearsal_share',None)
    if rehearsal_arg is not None and (
            not unified or not math.isfinite(rehearsal_arg)
            or not 0.0<=rehearsal_arg<=0.5):
        raise ValueError('--athletic-rehearsal-share requires unified learning and 0..0.5')
    athletic_share=(float(rehearsal_arg) if rehearsal_arg is not None
                    else .10 if growth_recipe=='main-mature' else 0.0)
    terrain_lesson=str(getattr(args,'terrain_lesson','mixed'))
    temporal_share=float(getattr(args,'temporal_vx_reward_share',0.))
    stillness_birth_frac=float(getattr(args,'stillness_birth_frac',0.))
    if not (math.isfinite(stillness_birth_frac) and 0. <= stillness_birth_frac <= .25):
        raise ValueError('--stillness-birth-frac must be in [0,0.25]')
    science_promotion=float(getattr(args,'science_step_promotion_mm',0.))
    science_retreat=float(getattr(args,'science_step_retreat_mm',0.))
    science_attempts=getattr(args,'science_step_min_attempts',4)
    if (not math.isfinite(science_promotion) or not 0.<=science_promotion<=10.
        or not math.isfinite(science_retreat) or not 0.<=science_retreat<=10.
        or type(science_attempts) is not int or not 4<=science_attempts<=32):
        raise ValueError('step frontier science requires 0..10mm and 4..32 attempts')
    if (not math.isfinite(temporal_share) or not 0.<=temporal_share<=1.0
            or (temporal_share>0. and not unified)):
        raise ValueError('--temporal-vx-reward-share requires unified actor and 0..1')
    force_program=str(getattr(args,'force_program','aggressive'))
    if growth_recipe not in ('legacy','main-mature') or terrain_lesson not in ('mixed','coherent','coherent-direct','coherent-careful'):
        raise ValueError('invalid experimental lesson selection')
    if force_program not in ('gentle','aggressive') or not (0.<=retention_share<=.4):
        raise ValueError('invalid force programme or retention share')
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
    requested_pitch=getattr(args, 'pitch_deg', None)
    # Main-style gentle cold posture, without changing mature sim2real runs.
    pitch_deg=float(8.0 if requested_pitch is None else requested_pitch)
    if not math.isfinite(pitch_deg) or not 0.0 < pitch_deg <= 30.0:
        raise ValueError('--pitch-deg must be within (0, 30] degrees')
    weight=float(getattr(args,'motor_guidance_weight',0.0))
    if not math.isfinite(weight) or not 0.0 <= weight <= 0.10:
        raise ValueError('--motor-guidance-weight must be in [0, 0.10]')
    terrain_actor = bool(getattr(args, 'terrain_actor', False))
    terrain_max = float(getattr(args,'terrain_step_max_m',0.0))
    if unified and (not terrain_actor or args.profile != 'sim2real'):
        raise ValueError('unified capability learning requires sim2real and terrain_360')

    if not math.isfinite(terrain_max) or not 0.0 <= terrain_max <= .08:
        raise ValueError('--terrain-step-max-m must be between 0 and 0.08')
    if terrain_actor and not 0.001 <= terrain_max <= .08:
        raise ValueError('--terrain-actor needs --terrain-step-max-m above zero')
    terrain_carpet=bool(getattr(args,'terrain_carpet',False))
    carpet_prisms=bool(getattr(args,'carpet_prisms',False))
    if carpet_prisms and not terrain_carpet:
        raise ValueError('--carpet-prisms requires --terrain-carpet')
    carpet_transition_cap_deg=float(getattr(args,'carpet_transition_cap_deg',3.0))
    if (not math.isfinite(carpet_transition_cap_deg) or
            not 0.0<=carpet_transition_cap_deg<=15.0 or
            (not terrain_carpet and carpet_transition_cap_deg!=3.0)):
        raise ValueError('--carpet-transition-cap-deg requires --terrain-carpet and value 0..15')
    if terrain_carpet and not (unified and terrain_actor and args.profile=='sim2real'):
        raise ValueError('--terrain-carpet requires unified sim2real terrain_360')
    academy_stream=bool(getattr(args,'academy_stream',False))
    intro=int(getattr(args,'academy_intro_steps',60_000_000))
    start=int(getattr(args,'academy_start_steps',80_000_000))
    ramp=int(getattr(args,'academy_ramp_steps',20_000_000))
    intro_share=float(getattr(args,'academy_intro_share',.10))
    academy_share=float(getattr(args,'academy_max_share',.50))
    height_cap=float(getattr(args,'academy_height_cap_m',.150))
    course_frames=int(getattr(args,'academy_course_episode_frames',1200))
    if academy_stream and terrain_lesson not in (
            'coherent','coherent-direct','coherent-careful'):
        raise ValueError(
            'multi-lip Academy requires a coherent straight-ahead STEP lesson; '
            'set --terrain-lesson coherent-careful (recommended)')
    if ((academy_stream and not (unified and terrain_actor)) or
        intro<0 or start<intro or ramp<=0 or
        not (0.<=intro_share<=academy_share<=.85) or
        not math.isfinite(height_cap) or not .003<=height_cap<=.25 or
        not 700<=course_frames<=1500):
        raise ValueError('streaming Academy requires unified terrain_360 and a valid age/share/height contract')
    # Only the single policy / real prism carpet is executable in v5045.
    # Prior foundation, 232-input and nine-box lineages are source-pinned
    # to their historical Git commits; do not accept fake --plan recipes.
    if not (unified and terrain_actor and terrain_carpet and carpet_prisms
            and academy_stream):
        raise ValueError(
            "v5045 prism-only training requires --profile sim2real "
            "--terrain-actor --unified-capabilities --terrain-carpet "
            "--carpet-prisms --academy-stream (no legacy plane/232/box modes)")
    return replace(cfg, steps=args.steps, attitude_profile='v5015',
                   solver_iterations=solver_iterations,
                   solver_ls_iterations=solver_ls_iterations,
                   observation_schema=OBS_SCHEMA_TERRAIN,
                   terrain_step_max_m=terrain_max, unified_capability_focus=unified,
                   foundation_steps=foundation_steps, foundation_mix=foundation_mix,
                   teaching_schedule=teaching_schedule,
                   velocity_i_ramp_s=velocity_i_ramp_s,
                   velocity_i_multiplier=velocity_i_multiplier,
                   growth_recipe=growth_recipe, retention_share=retention_share,
                   academy_stream=academy_stream,
                   terrain_carpet=terrain_carpet,
                   terrain_carpet_prisms=carpet_prisms,
                   carpet_transition_cap_deg=carpet_transition_cap_deg,
                   academy_intro_steps=intro,academy_start_steps=start,
                   academy_ramp_steps=ramp,academy_intro_share=intro_share,
                   academy_max_share=academy_share,
                   academy_height_cap_m=height_cap,
                   academy_course_episode_frames=course_frames,
                   terrain_lesson=terrain_lesson, force_program=force_program,
                   temporal_vx_reward_share=temporal_share,
                   stillness_birth_frac=stillness_birth_frac,
                   science_step_promotion_mm=science_promotion,
                   science_step_retreat_mm=science_retreat,
                   science_step_min_attempts=science_attempts,
                   fixed_eval_every_steps=fixed_eval_every_steps,
                   # Source main foundation: steady-first low-gain gait. The
                   # mature caps remain physical *headroom*, not birth tasks.
                   difficulty_base_step=(.0024 if growth_recipe=='main-mature' else .0014),
                   fall_target=(.34 if growth_recipe=='main-mature' else .30),
                   # Keep the proven 20M command/phase economics rather than
                   # accidentally inheriting mature speed limits.
                   vx_hi_max=(.60 if growth_recipe=='main-mature' else .40),
                   vx_lo_min=(-.20 if growth_recipe=='main-mature' else -.12),
                   vy_max_mps=(.22 if growth_recipe=='main-mature' else .14),
                   yaw_max_rad_s=(1.50 if growth_recipe=='main-mature' else .90),
                   freeze_obs_norm_after_steps=0,
                   cloud_probes_per_eval=0, cloud_final_probes=0,
                   combo_probe_frac=(.12 if growth_recipe=='main-mature' else .08),
                   phase_speed_gain=(1.25 if growth_recipe=='main-mature' else .75),
                   regime_command_frac=(.85 if growth_recipe=='main-mature' else .60),
                   athletic_rehearsal_frac=athletic_share,
                   combo_mix_frac=(.88 if growth_recipe=='main-mature' else 0.0),
                   frontier_vx_hi_safety=1.4,
                   frontier_vx_lo_safety=.55,
                   frontier_vy_safety=.55,
                   frontier_yaw_safety=3.5,
                   frontier_combo_safety=2.0,
                   frontier_push_safety_n=(.70 if force_program=='gentle' else 4.0),
                   # Main's foundation still sees ≤0.70N; the unified
                   # governor may later earn real larger pushes.
                   push_max_force_n=4.0, capability_field_mode='shadow',
                   pitch_command_max_deg=pitch_deg, motor_guidance_weight=weight,
                   domain_randomization=False)


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



def _validate_cross_source_research_parent(
        parent: Path, *, payload: dict, expected_source: str, expected_sha256: str,
        provenance: dict) -> None:
    """Explicitly approve ONE immutable external research ancestor.

    Shape, normalizer, optimizer and sensor identity are already checked by
    load_checkpoint. This additionally binds the *entire checkpoint bytes*
    and its recorded source to the operator's exact declaration. Never use
    for automatic resume; checkpoints in an interrupted study must retain
    their new source SHA and resume via --resume auto.
    """
    source = (payload['meta'].get('source_provenance') or {}).get('git_commit')
    if (len(expected_source) != 40 or len(expected_sha256) != 64
            or any(c not in '0123456789abcdef' for c in expected_source)
            or any(c not in '0123456789abcdef' for c in expected_sha256)):
        raise RuntimeError('Research ancestor requires full lower-case source SHA and SHA256')
    if (source != expected_source or source == provenance['commit']
            or parent.name != 'final.pkl' or parent.parent.name != 'checkpoints'):
        raise RuntimeError('Research parent must be completed final.pkl from exact declared earlier source')
    digest = hashlib.sha256()
    with parent.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != expected_sha256:
        raise RuntimeError('Research parent checkpoint SHA256 changed: refusing cross-source fork')
    print('EXPLICIT CROSS-SOURCE RESEARCH FORK APPROVED: '
          f'{source[:12]} -> {provenance["commit"][:12]}; '
          f'parent SHA256={expected_sha256[:16]}...; original unchanged',
          flush=True)


def inspect_run(config: TrainingConfig, *, provenance: dict, resume: str,
                research_parent_source: str = '', research_parent_sha256: str = '') -> tuple[Path | None, bool]:
    """Fail closed on incompatible runs, unclean source, and unsafe paths.

    Returns (validated checkpoint to resume from, already complete?).
    A resume from another experiment does not rewrite that ancestor.
    """
    if bool(research_parent_source) != bool(research_parent_sha256):
        raise RuntimeError('Research fork requires BOTH exact source SHA and checkpoint SHA256')
    if research_parent_source and resume in ('', 'auto'):
        raise RuntimeError('Research parent hashes require an explicit external --resume checkpoint')
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
            if not (research_parent_source and research_parent_sha256):
                raise RuntimeError(
                    'External checkpoint source differs from current Git commit. '
                    'Cross-source research requires explicit full ancestor source SHA '
                    'and checkpoint SHA256, or start a new cold training lineage.')
            _validate_cross_source_research_parent(
                parent, payload=payload, expected_source=research_parent_source,
                expected_sha256=research_parent_sha256, provenance=provenance)
        elif research_parent_source:
            raise RuntimeError('Research cross-source flags are not allowed for same-source resume')
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
    parent, completed = inspect_run(
        config, provenance=provenance, resume=args.resume,
        research_parent_source=getattr(args,'research_parent_source',''),
        research_parent_sha256=getattr(args,'research_parent_sha256',''))
    if getattr(args, 'check_resume_only', False):
        print(f'RESUME VERIFICATION PASS: {parent or "cold start"}'
              f' (already complete={completed}); no PPO run or output created',
              flush=True)
        return 0
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
