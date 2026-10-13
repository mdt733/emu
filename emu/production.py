#!/usr/bin/env python3
"""EMU v5014 production lineage: one brain, progressively less trustworthy worlds.

This is deliberately not a tournament. Each phase resumes the exact validated
checkpoint from the preceding phase, preserving policy/value weights, optimizer,
normalizer, curriculum/health and capability field. Only the hidden physical-domain
distribution broadens between phases, so XLA recompiles a small number of times
instead of whenever a curriculum scalar moves.
"""
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, replace
from pathlib import Path

from .artifacts import write_json
from .checkpoints import find_valid_resume
from .experiments import git_provenance, resolve_target_checkpoint, run_training
from .policy_export import build_bundle, write_bundle
from .training_config import foundation_config, mature_config

PHASES = (
    # name, absolute target, robot R, terrain T, hardware H
    ("00_foundation", 20_000_000, 0.00, 0.00, 0.00),
    ("10_adapt",      60_000_000, 0.10, 0.08, 0.12),
    ("20_breadth",   140_000_000, 0.30, 0.25, 0.35),
    ("30_robust",    260_000_000, 0.55, 0.50, 0.60),
    ("40_generalist",400_000_000, 0.75, 0.70, 0.75),
)

def _phase_config(name, target, r, t, h, *, root, seed, device, envs, resume):
    run_dir = root / name
    if target <= 20_000_000:
        return foundation_config(run_dir=run_dir, seed=seed, device=device, envs=envs, resume=resume)
    cfg = mature_config('graph', run_dir=run_dir, steps=target, seed=seed, device=device, envs=envs, resume=resume)
    return replace(
        cfg,
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

def run(root: Path, *, seed: int, device: str, envs: int, target: int = 400_000_000) -> Path:
    root = root.expanduser().resolve(); root.mkdir(parents=True, exist_ok=True)
    provenance = git_provenance()
    manifest_path = root / 'lineage.json'
    phase_records=[]
    previous=None
    for name, phase_target, r, t, h in PHASES:
        if phase_target > target: break
        run_dir=root/name
        verify_existing_phase_config(
            run_dir, _phase_config(name, phase_target, r, t, h,
                                   root=root, seed=seed, device=device, envs=envs, resume=None))
        complete, partial, expected = resolve_target_checkpoint(run_dir, target_steps=phase_target, envs=envs, label=name)
        if complete is not None:
            previous=complete
            print(f'[emu_v5014] {name}: complete @ {expected:,}', flush=True)
        else:
            resume = partial or previous
            cfg=_phase_config(name, phase_target, r,t,h, root=root, seed=seed, device=device, envs=envs, resume=resume)
            print(f'[emu_v5014] {name}: -> {expected:,}  domain R/T/H={r:.2f}/{t:.2f}/{h:.2f}', flush=True)
            run_training(cfg, provenance=provenance)
            complete, partial, expected = resolve_target_checkpoint(run_dir, target_steps=phase_target, envs=envs, label=name)
            if complete is None:
                raise RuntimeError(f'{name} did not produce target checkpoint; resumable={partial}')
            previous=complete
        phase_records.append({'name':name,'target':phase_target,'update_boundary':expected,'robot_radius':r,'terrain_radius':t,'hardware_radius':h,'checkpoint':str(previous)})
        write_json(manifest_path, {'version':'v5014','seed':seed,'envs':envs,'git':provenance,'target':target,'phases':phase_records})
    if previous is None: raise RuntimeError('no production phase ran')
    # Canonical output handles: immutable phase checkpoint remains source of truth.
    out_ck = root/'final.pkl'
    if out_ck.exists(): out_ck.unlink()
    try: out_ck.hardlink_to(previous)
    except OSError: shutil.copy2(previous,out_ck)
    bundle=build_bundle(previous,label='EMU v5014 production')
    write_bundle(bundle, root/'final.emupolicy')
    write_json(root/'lineage.json', {'version':'v5014','seed':seed,'envs':envs,'git':provenance,'target':target,'phases':phase_records,'final_checkpoint':str(out_ck),'final_policy':str(root/'final.emupolicy')})
    print(f'[emu_v5014] complete: {out_ck}', flush=True)
    print(f'[emu_v5014] browser policy: {root/"final.emupolicy"}', flush=True)
    return out_ck

def main(argv=None):
    ap=argparse.ArgumentParser(description='Train the single EMU v5014 production lineage')
    ap.add_argument('--root',type=Path,default=Path.home()/'robot_lab'/'emu_runs'/'v5014_production')
    ap.add_argument('--seed',type=int,default=733); ap.add_argument('--envs',type=int,default=2048)
    ap.add_argument('--device',choices=['auto','cpu','cuda','gpu'],default='cuda')
    ap.add_argument('--target',type=int,default=400_000_000,choices=[20_000_000,60_000_000,140_000_000,260_000_000,400_000_000])
    ap.add_argument('--plan',action='store_true')
    ns=ap.parse_args(argv)
    if ns.plan:
        print(json.dumps({'version':'v5014','seed':ns.seed,'target':ns.target,'teacher':'full graph @ 10% after 20M','phases':[{'name':n,'target':t,'R':r,'T':te,'H':h} for n,t,r,te,h in PHASES if t<=ns.target]},indent=2)); return 0
    run(ns.root,seed=ns.seed,device=ns.device,envs=ns.envs,target=ns.target); return 0

if __name__=='__main__': raise SystemExit(main())
