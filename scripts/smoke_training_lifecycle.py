#!/usr/bin/env python3
"""Bounded GPU acceptance test: PPO → checkpoint → warm resume → export.

The three explicit stages never launch the full experiment ladder. They use two
independent run folders outside Git; no existing checkpoint is overwritten.
Run the stages in order using the commands in docs/validation.md.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

from emu.checkpoint_io import load_checkpoint, payload_train_state_is_finite
from emu.paths import source_repo_root
from emu.run_support import git_provenance, run_training
from emu.training_config import ROLLOUT_STEPS, TrainingConfig, foundation_config

SEED = 733
ENVS = 8
UPDATE_STEPS = ENVS * ROLLOUT_STEPS


def configs(root: Path) -> tuple[TrainingConfig, TrainingConfig]:
    first = replace(
        foundation_config(run_dir=root / "first", seed=SEED, device="cuda", envs=ENVS),
        steps=UPDATE_STEPS,
        attitude_profile="v5015",
        pitch_command_max_deg=8.0,
    )
    second = replace(first, run_dir=str(root / "continued"),
                     resume=str((root / "first/checkpoints/final.pkl").resolve()),
                     steps=2 * UPDATE_STEPS)
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


def first_stage(root: Path) -> None:
    provenance = current_source()
    if root.exists():
        raise RuntimeError(f"refusing to overwrite existing acceptance root: {root}")
    root.mkdir(parents=True)
    first, _ = configs(root)
    print(f"Acceptance stage 1: one PPO update ({UPDATE_STEPS} steps), {ENVS} MJX worlds", flush=True)
    run_training(first, provenance=provenance)
    checkpoint(root / "first/checkpoints/final.pkl", UPDATE_STEPS)
    fixed_exam_exists(root / "first", UPDATE_STEPS)
    physical_telemetry_exists(root / "first")
    print("PASS stage 1: trained checkpoint and read-only held-out exam", flush=True)


def continue_stage(root: Path) -> None:
    provenance = current_source()
    first, second = configs(root)
    parent = root / "first/checkpoints/final.pkl"
    checkpoint(parent, UPDATE_STEPS)
    first_config = TrainingConfig.load(root / "first/training_config.json")
    if asdict(first_config) != asdict(first):
        raise RuntimeError("stage-1 saved config does not match this smoke recipe")
    require_pinned_source(root / 'first', provenance['commit'])
    if (root / "continued").exists():
        raise RuntimeError("continued/ already exists: inspect it; never overwrite")
    parent_hash = digest(parent)
    print(f"Acceptance stage 2: warm-resume at {UPDATE_STEPS} and train to {2*UPDATE_STEPS}", flush=True)
    run_training(second, provenance=provenance)
    if digest(parent) != parent_hash:
        raise RuntimeError("the parent checkpoint changed during continuation")
    checkpoint(root / "continued/checkpoints/final.pkl", 2 * UPDATE_STEPS)
    fixed_exam_exists(root / "continued", 2 * UPDATE_STEPS)
    physical_telemetry_exists(root / "continued")
    print("PASS stage 2: fresh continuation from untouched validated ancestor", flush=True)


def verify_stage(root: Path) -> None:
    # Verify both original commit identities before opening pickle checkpoints.
    provenance = current_source()
    for name in ('first', 'continued'):
        require_pinned_source(root / name, provenance['commit'])
    first, second = configs(root)
    p1 = checkpoint(root / "first/checkpoints/final.pkl", UPDATE_STEPS)
    p2 = checkpoint(root / "continued/checkpoints/final.pkl", 2 * UPDATE_STEPS)
    for folder, step in (("first", UPDATE_STEPS), ("continued", 2 * UPDATE_STEPS)):
        fixed_exam_exists(root / folder, step)
        physical_telemetry_exists(root / folder)
    run_meta = json.loads((root / "continued/run_meta.json").read_text(encoding="utf-8"))
    if int(run_meta.get("resumed_from_step") or -1) != UPDATE_STEPS:
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
    from emu.policy_export import build_bundle, write_bundle
    exported = root / "tiny_smoke_actor.emupolicy"
    bundle = build_bundle(root / "continued/checkpoints/final.pkl",
                          label="Two-update smoke (not trained to walk)")
    if int(bundle["source"]["step"]) != 2 * UPDATE_STEPS:
        raise RuntimeError("exported actor has unexpected training step")
    write_bundle(bundle, exported)
    print(f"PASS checkpoint lineage: {UPDATE_STEPS} -> {2*UPDATE_STEPS}", flush=True)
    print(f"PASS observation normaliser: count {first_count:.0f} -> {second_count:.0f}", flush=True)
    print("PASS held-out examinations, completed-resume no-op, and browser actor export", flush=True)
    print(f"Smoke export (NOT walking trained): {exported}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("first", "continue", "verify"))
    parser.add_argument("--root", required=True, help="new scratch folder OUTSIDE this Git checkout")
    args = parser.parse_args()
    root = protected_root(args.root)
    if args.stage == "first":
        first_stage(root)
    elif args.stage == "continue":
        continue_stage(root)
    else:
        verify_stage(root)


if __name__ == "__main__":
    main()
