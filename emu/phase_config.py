"""Shared phase configs and restart safety, independent of historic forge recipes.

Used by both the current experiment ladder and legacy single-lineage runs.
Keep the exact config and checkpoint compatibility semantics stable.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

from .training_config import foundation_config, mature_config


def _phase_config(name, target, r, t, h, *, root, seed, device, envs, resume, profile='v5014', pitch_deg=8.0214):
    run_dir = root / name
    if target <= 20_000_000:
        return replace(foundation_config(run_dir=run_dir, seed=seed, device=device, envs=envs, resume=resume),
                       attitude_profile=profile, pitch_command_max_deg=pitch_deg)
    cfg = mature_config('graph', run_dir=run_dir, steps=target, seed=seed, device=device, envs=envs, resume=resume)
    return replace(
        cfg,
        attitude_profile=profile, pitch_command_max_deg=pitch_deg,
        capability_field_mode='graph',
        field_max_influence=0.10,
        field_teach_nodes=True, field_teach_edges=True, field_teach_recovery=True,
        field_evidence_from_taught=True,
        # v5013's clearest reproducible lesson: low dose. Keep teacher and
        # physical-domain severity independent.
        field_influence_start_steps=20_000_000,
        field_influence_full_steps=30_000_000,
        domain_randomization=max(r,t,h) > 0.0,
        domain_robot_radius=r, domain_terrain_radius=t, domain_hardware_radius=h,
    )


def verify_existing_phase_config(run_dir: Path, expected) -> None:
    """Refuse a same-path continuation with a different physical/learning contract.

    A perfectly valid schema-7 checkpoint can still be from a different seed,
    env count, field teacher or hidden physical distribution. Prevent accidental
    reuse before `resolve_target_checkpoint` decides that an old phase is done.
    Hardware choice, source checkout location and resume pathname are not
    learning-contract parameters and may legitimately differ on restart.
    """
    ck = run_dir / 'checkpoints'
    if not ck.is_dir() or not any(ck.glob('*.pkl')):
        return
    from .training_config import TrainingConfig
    cfg_path = run_dir / 'training_config.json'
    if not cfg_path.is_file():
        raise RuntimeError(f'Found checkpoints in {run_dir} without training_config.json; refusing ambiguous lineage reuse')
    actual = asdict(TrainingConfig.load(cfg_path))
    wanted = asdict(expected)
    for ignored in ('resume', 'device', 'run_dir'):
        actual.pop(ignored, None)
        wanted.pop(ignored, None)
    differing = {key: (actual.get(key), wanted[key]) for key in wanted if actual.get(key) != wanted[key]}
    if differing:
        raise RuntimeError(f'Existing phase {run_dir.name} has a different training configuration: {differing}. '
                           'Use the original options, or choose a fresh --root; do not mix checkpoint lineages.')

