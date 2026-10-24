#!/usr/bin/env python3
"""Bounded GPU acceptance: PPO → checkpoint → resume → exported actor.

Pass --unified for the NEW 360-input actor, guarded to a maximum TWO PPO
updates. First/continue/verify use independent fresh directories outside Git;
source hashes and old checkpoints are protected. No overnight unlock.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

from emu.checkpoint_io import load_checkpoint, payload_train_state_is_finite
from emu.contracts import source_repo_root
from emu.checkpoint_io import git_provenance, run_training
from emu.training_config import ROLLOUT_STEPS, TrainingConfig, foundation_config

SEED = 733
ENVS = 8
UPDATE_STEPS = ENVS * ROLLOUT_STEPS


def configs(root: Path, *, unified: bool = False,
            envs: int = ENVS) -> tuple[TrainingConfig, TrainingConfig]:
    updates = envs * ROLLOUT_STEPS
    if unified:
        # The EXACT public unified configuration, only with bounded total
        # steps. Never duplicate the profile by reconstructing handpicked
        # training defaults in the smoke script.
        from types import SimpleNamespace
        from emu.run import one_run_config
        first = one_run_config(SimpleNamespace(
            profile="sim2real", run_dir=str(root / "first"),
            steps=updates, envs=envs, seed=SEED, device="cuda",
            terrain_actor=True, unified_capabilities=True,
            terrain_step_max_m=.04))
    else:
        first = replace(
            foundation_config(run_dir=root / "first", seed=SEED,
                              device="cuda", envs=envs),
            steps=updates, attitude_profile="v5015",
            pitch_command_max_deg=8.0)
    second = replace(first, run_dir=str(root / "continued"),
                     resume=str((root / "first/checkpoints/final.pkl").resolve()),
                     steps=2 * updates)
    return first, second


def protected_root(value: str) -> Path:
    root = Path(value).expanduser().resolve()
    source = source_repo_root()
    if source and (root == source.resolve() or root.is_relative_to(source.resolve())):
        raise ValueError("smoke run root must be OUTSIDE the Git checkout")
    if root == Path.home().resolve():
        raise ValueError("choose a new subdirectory under your run-data directory")
    return root


def current_source() -> dict:
    result = git_provenance()
    if not result.get("available") or not result.get("commit"):
        raise RuntimeError("Cannot identify this source checkout's Git commit.")
    if result.get("dirty"):
        status = subprocess.run(
            ["git", "status", "--short", "--untracked-files=normal"],
            cwd=source_repo_root(), text=True, capture_output=True, check=True,
        ).stdout.strip()
        raise RuntimeError(
            "Refusing to train from a modified Git checkout. Current changes:\n"
            + (status or "(Git reported changes; unable to list them)")
            + "\nInspect the paths with git diff / git status before continuing. "
              "Do not bypass the source-provenance gate or delete unreviewed files."
        )
    return result


def require_pinned_source(run_dir: Path, git_commit: str) -> None:
    """Refuse acceptance/checkpoint export from a different source checkout."""
    metadata = run_dir / 'run_meta.json'
    try:
        source = json.loads(metadata.read_text(encoding='utf-8'))['source_provenance']['git_commit']
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise RuntimeError(f'Invalid/missing acceptance source metadata: {metadata}') from exc
    if source != git_commit:
        raise RuntimeError(
            f'Acceptance source commit differs for {run_dir.name}: '
            f'checkpoint={source} checkout={git_commit}; '
            'return to the original pinned source or choose a NEW run root.')


def checkpoint(path: Path, expected_step: int) -> dict:
    if not path.is_file():
        raise RuntimeError(f"expected checkpoint not found: {path}")
    payload = load_checkpoint(path, validate=True)
    observed = int(payload["meta"]["step"])
    if observed != expected_step:
        raise RuntimeError(f"{path}: expected step {expected_step}, got {observed}")
    if not payload_train_state_is_finite(payload):
        raise RuntimeError(f"{path}: nonfinite policy or optimiser state")
    return payload


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixed_exam_exists(run_dir: Path, step: int) -> None:
    file = run_dir / "fixed_eval.csv"
    with file.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows or int(float(rows[-1]["step"])) != step:
        raise RuntimeError(f"no fixed evaluation for step {step:,} in {file}")
    if not (run_dir / "checkpoints/best.pkl").is_file():
        raise RuntimeError(f"fixed evaluation did not save best.pkl in {run_dir}")


PHYSICAL_DIAGNOSTICS = (
    'servo_shaft_abs_power_w', 'servo_shaft_drive_power_w',
    'servo_shaft_brake_power_w', 'servo_over_rated_frac',
    'servo_peak_torque_limit_frac', 'spring_potential_energy_j',
    'shin_spring_energy_j',
)


def physical_telemetry_exists(run_dir: Path) -> None:
    """Fail hardware smoke if physical instrumentation is absent or nonfinite."""
    path = run_dir / 'metrics.csv'
    with path.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f'No PPO metrics to inspect in {path}')
    latest = rows[-1]
    for key in PHYSICAL_DIAGNOSTICS:
        try:
            value = float(latest[key])
        except (KeyError, ValueError, TypeError) as exc:
            raise RuntimeError(f'Physical metric {key} is missing/invalid in {path}') from exc
        if not math.isfinite(value) or value < -1e-7:
            raise RuntimeError(f'Physical metric {key} is nonfinite or negative: {value}')
    if not 0 <= float(latest['servo_over_rated_frac']) <= 1:
        raise RuntimeError('Rated-torque fraction must be in [0, 1]')
    if float(latest['shin_spring_energy_j']) > float(latest['spring_potential_energy_j']) + 1e-6:
        raise RuntimeError('Passive shin spring potential cannot exceed total spring potential')


def first_stage(root: Path, *, unified: bool = False, envs: int = ENVS) -> None:
    updates = envs * ROLLOUT_STEPS
    provenance = current_source()
    if root.exists():
        raise RuntimeError(f"refusing to overwrite existing acceptance root: {root}")
    root.mkdir(parents=True)
    first, _ = configs(root, unified=unified, envs=envs)
    print(f"Acceptance stage 1: one PPO update ({updates} steps), {envs} MJX worlds", flush=True)
    run_training(first, provenance=provenance)
    checkpoint(root / "first/checkpoints/final.pkl", updates)
    fixed_exam_exists(root / "first", updates)
    physical_telemetry_exists(root / "first")
    print("PASS stage 1: trained checkpoint and read-only held-out exam", flush=True)


def continue_stage(root: Path, *, unified: bool = False, envs: int = ENVS) -> None:
    updates = envs * ROLLOUT_STEPS
    provenance = current_source()
    first, second = configs(root, unified=unified, envs=envs)
    parent = root / "first/checkpoints/final.pkl"
    checkpoint(parent, updates)
    first_config = TrainingConfig.load(root / "first/training_config.json")
    if asdict(first_config) != asdict(first):
        raise RuntimeError("stage-1 saved config does not match this smoke recipe")
    require_pinned_source(root / 'first', provenance['commit'])
    if (root / "continued").exists():
        raise RuntimeError("continued/ already exists: inspect it; never overwrite")
    parent_hash = digest(parent)
    print(f"Acceptance stage 2: warm-resume at {updates} and train to {2*updates}", flush=True)
    run_training(second, provenance=provenance)
    if digest(parent) != parent_hash:
        raise RuntimeError("the parent checkpoint changed during continuation")
    checkpoint(root / "continued/checkpoints/final.pkl", 2 * updates)
    fixed_exam_exists(root / "continued", 2 * updates)
    physical_telemetry_exists(root / "continued")
    print("PASS stage 2: fresh continuation from untouched validated ancestor", flush=True)


def verify_stage(root: Path, *, unified: bool = False, envs: int = ENVS) -> None:
    updates = envs * ROLLOUT_STEPS
    # Verify both original commit identities before opening pickle checkpoints.
    provenance = current_source()
    for name in ('first', 'continued'):
        require_pinned_source(root / name, provenance['commit'])
    first, second = configs(root, unified=unified, envs=envs)
    p1 = checkpoint(root / "first/checkpoints/final.pkl", updates)
    p2 = checkpoint(root / "continued/checkpoints/final.pkl", 2 * updates)
    for folder, step in (("first", updates), ("continued", 2 * updates)):
        fixed_exam_exists(root / folder, step)
        physical_telemetry_exists(root / folder)
    run_meta = json.loads((root / "continued/run_meta.json").read_text(encoding="utf-8"))
    if int(run_meta.get("resumed_from_step") or -1) != updates:
        raise RuntimeError("continuation did not report restoring the previous step")
    first_count = float(p1["obs_norm"]["count"])
    second_count = float(p2["obs_norm"]["count"])
    if not second_count > first_count:
        raise RuntimeError("observation normaliser count failed to advance")

    # An already-completed warm resume must not make another PPO update.
    before = digest(root / "continued/checkpoints/final.pkl")
    completed_config = root / "complete_resume_config.json"
    replace(second, resume="auto").save(completed_config)
    proc = subprocess.run([sys.executable, "-m", "emu.training",
                           "--config", str(completed_config)], check=True,
                          text=True, capture_output=True)
    if "already complete" not in proc.stdout:
        raise RuntimeError("completed resume did not take the no-op path")
    if digest(root / "continued/checkpoints/final.pkl") != before:
        raise RuntimeError("completed resume modified final checkpoint")

    # An actual deployable actor export is part of the acceptance contract.
    from emu.viewer_server import build_bundle, write_bundle
    exported = root / "tiny_smoke_actor.emupolicy"
    bundle = build_bundle(root / "continued/checkpoints/final.pkl",
                          label="Two-update smoke (not trained to walk)")
    if int(bundle["source"]["step"]) != 2 * updates:
        raise RuntimeError("exported actor has unexpected training step")
    if unified:
        from emu.contracts import TERRAIN_OBS_DIM
        from emu.viewer_server import load_exported_actor_for_exam
        if (bundle.get("schema") != 3 or
                bundle.get("policy",{}).get("obs_dim") != TERRAIN_OBS_DIM):
            raise RuntimeError("exported unified actor is not a 360-input schema-3 policy")
    write_bundle(bundle, exported)
    if unified:
        # Decode exactly what the frozen physical examiner will consume;
        # this checks array-only ToF semantics, model hash, and normalizer.
        _, _, exam_meta=load_exported_actor_for_exam(exported)
        if not exam_meta["array_only"] or exam_meta["obs_dim"]!=TERRAIN_OBS_DIM:
            raise RuntimeError("exported 360 actor failed array-only replay contract")
    print(f"PASS checkpoint lineage: {updates} -> {2*updates}", flush=True)
    print(f"PASS observation normaliser: count {first_count:.0f} -> {second_count:.0f}", flush=True)
    print("PASS held-out examinations, completed-resume no-op, and browser actor export", flush=True)
    print(f"Smoke export (NOT walking trained): {exported}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("first", "continue", "verify"))
    parser.add_argument("--root", required=True, help="new scratch folder OUTSIDE this Git checkout")
    parser.add_argument("--unified", action="store_true",
                        help="bounded 360-input unified acceptance; NEVER starts a campaign")
    parser.add_argument("--envs", type=int, default=ENVS,
                        help="8 default, or 12/2048 with --unified")
    args = parser.parse_args()
    if args.unified:
        if args.envs not in (8, 12, 2048):
            parser.error("--unified accepts 8, 12 or 2048 worlds only")
        # Intentionally opt in for this process (and completed-resume child)
        # only; trainer still refuses more than 2 total PPO updates.
        os.environ["EMU_UNIFIED_ACCEPTANCE"] = "bounded-ppo-lifecycle"
    elif args.envs != ENVS:
        parser.error("legacy smoke has a fixed eight-world reference recipe")
    root = protected_root(args.root)
    if args.stage == "first":
        first_stage(root, unified=args.unified, envs=args.envs)
    elif args.stage == "continue":
        continue_stage(root, unified=args.unified, envs=args.envs)
    else:
        verify_stage(root, unified=args.unified, envs=args.envs)


if __name__ == "__main__":
    main()
