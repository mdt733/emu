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
from pathlib import Path

from .artifacts import write_json
from .experiments import git_provenance, resolve_target_checkpoint, run_training
from .policy_export import build_bundle, write_bundle
from .phase_config import _phase_config, verify_existing_phase_config

PHASES = (
    # name, absolute target, robot R, terrain T, hardware H
    ("00_foundation", 20_000_000, 0.00, 0.00, 0.00),
    ("10_adapt",      60_000_000, 0.10, 0.08, 0.12),
    ("20_breadth",   140_000_000, 0.30, 0.25, 0.35),
    ("30_robust",    260_000_000, 0.55, 0.50, 0.60),
    ("40_generalist",400_000_000, 0.75, 0.70, 0.75),
)

# v5015 is separate from the original v5014 learning recipe.
PHASES_5015 = (
    # phase, target steps, robot R, terrain T, hardware H, maximum chassis pitch
    ('00_foundation', 20_000_000, 0.00, 0.00, 0.00, 8.0),
    ('10_attitude', 60_000_000, 0.10, 0.08, 0.12, 15.0),
    ('20_breadth', 140_000_000, 0.30, 0.25, 0.35, 25.0),
    ('30_robust', 260_000_000, 0.55, 0.50, 0.60, 35.0),
    ('40_generalist', 400_000_000, 0.75, 0.70, 0.75, 45.0),
)


def run(root: Path, *, seed: int, device: str, envs: int, target: int = 400_000_000,
        profile: str = 'v5014') -> Path:
    if profile not in ('v5014', 'v5015'):
        raise ValueError(f'Unknown production profile: {profile}')
    root = root.expanduser().resolve(); root.mkdir(parents=True, exist_ok=True)
    provenance = git_provenance()
    manifest_path = root / 'lineage.json'
    phase_records=[]
    previous=None
    phases = PHASES if profile == 'v5014' else PHASES_5015
    for phase in phases:
        name, phase_target, r, t, h = phase[:5]
        pitch_deg = phase[5] if profile == 'v5015' else 8.0214
        if phase_target > target: break
        run_dir=root/name
        verify_existing_phase_config(
            run_dir, _phase_config(name, phase_target, r, t, h,
                                   root=root, seed=seed, device=device, envs=envs, resume=None,
                                   profile=profile, pitch_deg=pitch_deg))
        complete, partial, expected = resolve_target_checkpoint(run_dir, target_steps=phase_target, envs=envs, label=name)
        if complete is not None:
            previous=complete
            print(f'[emu_{profile}] {name}: complete @ {expected:,}', flush=True)
        else:
            resume = partial or previous
            cfg=_phase_config(name, phase_target, r,t,h, root=root, seed=seed, device=device, envs=envs, resume=resume,
                              profile=profile, pitch_deg=pitch_deg)
            print(f'[emu_{profile}] {name}: -> {expected:,}  R/T/H={r:.2f}/{t:.2f}/{h:.2f} pitch_max={pitch_deg:.1f}°', flush=True)
            run_training(cfg, provenance=provenance)
            complete, partial, expected = resolve_target_checkpoint(run_dir, target_steps=phase_target, envs=envs, label=name)
            if complete is None:
                raise RuntimeError(f'{name} did not produce target checkpoint; resumable={partial}')
            previous=complete
        phase_records.append({'name':name,'target':phase_target,'update_boundary':expected,
                              'robot_radius':r,'terrain_radius':t,'hardware_radius':h,
                              'attitude_profile':profile,'pitch_max_deg':pitch_deg,'checkpoint':str(previous)})
        write_json(manifest_path, {'version':profile,'seed':seed,'envs':envs,'git':provenance,'target':target,'phases':phase_records})
    if previous is None: raise RuntimeError('no production phase ran')
    # Canonical output handles: immutable phase checkpoint remains source of truth.
    out_ck = root/'final.pkl'
    if out_ck.exists(): out_ck.unlink()
    try: out_ck.hardlink_to(previous)
    except OSError: shutil.copy2(previous,out_ck)
    bundle=build_bundle(previous,label=f'EMU {profile} production')
    write_bundle(bundle, root/'final.emupolicy')
    write_json(root/'lineage.json', {'version':profile,'seed':seed,'envs':envs,'git':provenance,'target':target,'phases':phase_records,'final_checkpoint':str(out_ck),'final_policy':str(root/'final.emupolicy')})
    print(f'[emu_{profile}] complete: {out_ck}', flush=True)
    print(f'[emu_{profile}] browser policy: {root/"final.emupolicy"}', flush=True)
    return out_ck

def main(argv=None):
    ap=argparse.ArgumentParser(description='Train the single EMU v5014 production lineage')
    ap.add_argument('--root',type=Path,default=None)
    ap.add_argument('--profile',choices=['v5014','v5015'],default='v5014')
    ap.add_argument('--seed',type=int,default=733); ap.add_argument('--envs',type=int,default=2048)
    ap.add_argument('--device',choices=['auto','cpu','cuda','gpu'],default='cuda')
    ap.add_argument('--target',type=int,default=400_000_000,choices=[20_000_000,60_000_000,140_000_000,260_000_000,400_000_000])
    ap.add_argument('--plan',action='store_true')
    ns=ap.parse_args(argv)
    phases = PHASES if ns.profile == 'v5014' else PHASES_5015
    root = ns.root or Path.home()/'robot_lab'/'emu_runs'/f'{ns.profile}_production'
    if ns.plan:
        print(json.dumps({'version':ns.profile,'root':str(root),'seed':ns.seed,'target':ns.target,
            'teacher':'full graph @ 10% after 20M',
            'phases':[{'name':phase[0],'target':phase[1],'R':phase[2],'T':phase[3],
                       'H':phase[4], **({'chassis_pitch_max_deg':phase[5]} if len(phase)>5 else {})}
                      for phase in phases if phase[1]<=ns.target]},indent=2)); return 0
    run(root,seed=ns.seed,device=ns.device,envs=ns.envs,target=ns.target,profile=ns.profile); return 0

if __name__=='__main__': raise SystemExit(main())
