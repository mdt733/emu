"""One experiment, one existing trainer, and a deliberate checkpoint lineage.

No PPO, evaluator, curriculum, or checkpoint serialization is implemented here.
The existing training engine and run_support own those responsibilities.
"""
from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

from .phase_config import verify_existing_phase_config
from .paths import source_repo_root
from .run_support import (git_provenance, resolve_target_checkpoint,
                          run_training, update_boundary_at_or_after)
from .training_config import TrainingConfig, foundation_config, mature_config


def one_run_config(args) -> TrainingConfig:
    """Resolve a named, existing training profile without inventing PPO flags."""
    if args.envs <= 0 or args.steps <= 0:
        raise ValueError('--envs and --steps must both be positive')
    run_dir = Path(args.run_dir).expanduser().resolve()
    if args.profile == 'foundation':
        cfg = foundation_config(run_dir=run_dir, seed=args.seed,
                                device=args.device, envs=args.envs)
    elif args.profile == 'mature':
        cfg = mature_config('graph', run_dir=run_dir, steps=args.steps,
                            seed=args.seed, device=args.device, envs=args.envs,
                            resume=None)
    else:
        raise ValueError(f'unknown profile: {args.profile}')
    return replace(cfg, steps=args.steps, attitude_profile='v5015',
                   pitch_command_max_deg=8.0)


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
            raise RuntimeError('Saved training config exists without run_meta.json; refusing ambiguous lineage')
        metadata = json.loads(meta_path.read_text(encoding='utf-8'))
        ancestor = metadata.get('source_provenance', {}).get('git_commit')
        if ancestor != provenance['commit']:
            raise RuntimeError('Source commit differs from prior run; use the matching source or a new --run-dir')
        verify_existing_phase_config(root, config)

    finished, partial, expected = resolve_target_checkpoint(
        root, target_steps=config.steps, envs=config.envs, label=root.name)
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
        payload = load_checkpoint(parent, validate=True)
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
        label=Path(config.run_dir).name)
    if final is None:
        raise RuntimeError('Trainer returned but no validated final checkpoint reached the target')
    print(f'Completed validated run: {final}')
    return 0
