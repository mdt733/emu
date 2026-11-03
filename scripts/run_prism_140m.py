#!/usr/bin/env python3
"""v5045: one source-pinned, 140M-step, lean 18-prism GPU campaign.

This is a launch recipe for the existing one-run CLI, NOT a second trainer.
The gate is a separate two-update prism checkpoint/resume/export acceptance.
No run state is stored inside the Git checkout; main is never selected.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

STEPS = 140_000_000
ENVS = 2048
DEFAULT_SEED = 735
EXPECTED_BRANCH = "feature/v5045-core-skeleton-audit"
ACCEPTANCE_END_STEP = 2 * 8 * 32


def train_arguments(run_dir: Path, seed: int, mode: str) -> list[str]:
    """Only one immutable physical/curriculum recipe across start and resume."""
    if mode not in ("plan", "start", "resume", "verify"):
        raise ValueError(f"Unknown mode: {mode}")
    if not 0 <= seed < 2**31:
        raise ValueError("seed must be an unsigned signed-32-bit-safe integer")
    out = [
        sys.executable, "-u", "-m", "emu.cli", "train",
        "--run-dir", str(run_dir.resolve()),
        "--steps", str(STEPS),
        "--envs", str(ENVS),
        "--seed", str(seed),
        "--device", "cuda",
        "--profile", "sim2real",
        "--pitch-deg", "8",
        "--terrain-actor",
        "--terrain-step-max-m", ".08",
        "--unified-capabilities",
        "--terrain-carpet",
        "--carpet-prisms",
        "--carpet-transition-cap-deg", "3",
        "--foundation-steps", "20000000",
        "--foundation-mix", "main",
        "--growth-recipe", "main-mature",
        "--teaching-schedule", "terrain-early",
        "--terrain-lesson", "coherent-careful",
        "--force-program", "gentle",
        "--athletic-rehearsal-share", ".10",
        "--retention-share", ".15",
        "--academy-stream",
        "--academy-intro-steps", "60000000",
        "--academy-intro-share", ".10",
        "--academy-start-steps", "80000000",
        "--academy-ramp-steps", "20000000",
        "--academy-max-share", ".50",
        "--academy-height-cap-m", ".15",
        "--academy-course-episode-frames", "1200",
        "--fixed-eval-every-steps", "10000000",
    ]
    if mode == "plan":
        out.append("--plan")
    if mode in ("resume", "verify"):
        out.extend(("--resume", "auto"))
    if mode == "verify":
        out.append("--check-resume-only")
    return out


def protected_root(value: str, source: Path) -> Path:
    root = Path(value).expanduser().resolve()
    source = source.resolve()
    if root == source or source in root.parents:
        raise ValueError("run root must live OUTSIDE the source Git checkout")
    if root in (Path.home().resolve(), Path("/")):
        raise ValueError("use a unique experiment subdirectory, not home or /")
    return root


def source_identity(source: Path) -> str:
    """Fail closed before an actual run if source is not the expected feature branch."""
    def git(*parts: str) -> str:
        return subprocess.check_output(
            ("git", *parts), cwd=source, text=True, stderr=subprocess.DEVNULL
        ).strip()
    branch = git("branch", "--show-current")
    sha = git("rev-parse", "HEAD")
    if branch != EXPECTED_BRANCH:
        raise RuntimeError(f"Expected {EXPECTED_BRANCH}, found {branch or 'detached HEAD'}")
    if git("status", "--porcelain"):
        raise RuntimeError("Source checkout is dirty; commit or restore changes before training")
    if len(sha) != 40:
        raise RuntimeError("No valid source SHA")
    return sha


def validate_acceptance(root: Path, *, source_sha: str, seed: int) -> None:
    """Require a completed same-source two-update prism lifecycle, not an old box smoke."""
    from emu.checkpoint_io import load_checkpoint, payload_train_state_is_finite
    from emu.training_config import TrainingConfig, training_sensor_contract
    from emu.contracts import OBS_SCHEMA_TERRAIN

    root = root.resolve()
    first_path = root / "first" / "checkpoints" / "final.pkl"
    continued_path = root / "continued" / "checkpoints" / "final.pkl"
    for name, expected_step, path in (
        ("first", 8 * 32, first_path),
        ("continued", ACCEPTANCE_END_STEP, continued_path),
    ):
        phase = root / name
        cfg = TrainingConfig.load(phase / "training_config.json")
        if not (cfg.terrain_carpet and cfg.terrain_carpet_prisms
                and cfg.academy_stream and cfg.unified_capability_focus
                and cfg.observation_schema == OBS_SCHEMA_TERRAIN
                and cfg.seed == seed and cfg.envs == 8):
            raise RuntimeError(f"{name} has the wrong prism/actor/seed contract")
        meta = json.loads((phase / "run_meta.json").read_text(encoding="utf-8"))
        recorded_sha = (meta.get("source_provenance") or {}).get("git_commit")
        if recorded_sha != source_sha:
            raise RuntimeError(
                f"{name} acceptance source {recorded_sha} != checkout {source_sha}"
            )
        if not path.is_file():
            raise FileNotFoundError(f"Missing completed prism acceptance checkpoint {path}")
        checkpoint = load_checkpoint(
            path, validate=True, expected_observation_schema=cfg.observation_schema,
            expected_sensor_contract=training_sensor_contract(cfg),
        )
        if int(checkpoint["meta"]["step"]) != expected_step:
            raise RuntimeError(f"{name} acceptance checkpoint step mismatch")
        if not payload_train_state_is_finite(checkpoint):
            raise RuntimeError(f"{name} acceptance state contains nonfinite values")
    exported_actor = root / "tiny_smoke_actor.emupolicy"
    if not exported_actor.is_file():
        raise RuntimeError("Acceptance must complete verify and export a prism actor")
    bundle = json.loads(exported_actor.read_text(encoding="utf-8"))
    training = bundle.get("training") or {}
    if training.get("terrain_carpet_prisms") is not True:
        raise RuntimeError("Exported acceptance actor must identify 18 real prism colliders")
    # Prove that the NEW frozen examiner can read the actual accepted actor
    # and take an independent true-prism trace. The tiny actor need not walk.
    import hashlib
    exam_path = root / "prism_exam_smoke" / "prism_exam.json"
    if not exam_path.is_file():
        raise RuntimeError("Missing real-prism frozen-actor examination smoke; see v5045 runbook")
    exam = json.loads(exam_path.read_text(encoding="utf-8"))
    if (exam.get("schema") != "emu_prism_frozen_exam_v1" or
            exam.get("source_training_sha") != source_sha or
            exam.get("source_step") != ACCEPTANCE_END_STEP or
            exam.get("source_actor_sha256") != hashlib.sha256(
                exported_actor.read_bytes()).hexdigest() or
            (exam.get("protocol") or {}).get(
                "actual_collision") != "18 convex prism meshes / nine 50cm source tiles" or
            (exam.get("protocol") or {}).get("no_optimizer_or_curriculum_updates") is not True):
        raise RuntimeError("Frozen actor exam smoke does not match this exact accepted prism source")
    print(f"PRISM LIFECYCLE + FROZEN EXAM PASS: 256 -> 512; source={source_sha[:12]}", flush=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("plan", "start", "resume", "verify"))
    p.add_argument("--run-dir", default="~/emu_runs/v5045_prism_140m_s735",
                   help="dedicated permanent run directory, outside Git (default seed 735)")
    p.add_argument("--acceptance-root", default="~/emu_runs/v5045_prism_lifecycle_s735",
                   help="same-source 8-world prism first/continue/verify acceptance")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    a = p.parse_args(argv)

    source = Path(__file__).resolve().parent.parent
    root = protected_root(a.run_dir, source)
    acceptance = protected_root(a.acceptance_root, source)
    if root == acceptance or root in acceptance.parents or acceptance in root.parents:
        p.error("training and acceptance must occupy separate directories")
    if not 0 <= a.seed < 2**31:
        p.error("seed must be 0..2147483647")
    if a.mode == "plan":
        print(f"Prism 140M cold-start recipe | source checkout: {source}")
        return subprocess.call(train_arguments(root, a.seed, "plan"), cwd=source)

    # Confirm this script is executed with its OWN editable checkout installed.
    from emu.contracts import source_repo_root
    if source_repo_root().resolve() != source:
        raise RuntimeError(
            "Wrong editable emu checkout active; run python -m pip install -e . here"
        )
    source_sha = source_identity(source)
    if a.mode == "start":
        if root.exists() and any(root.iterdir()):
            raise RuntimeError(
                f"{root} is not empty. Never overwrite a previous run. "
                "Use resume for an interrupted matching run."
            )
        validate_acceptance(acceptance, source_sha=source_sha, seed=a.seed)
    else:
        if not (root / "training_config.json").is_file():
            raise RuntimeError(f"No prism training lineage found in {root}")
    cmd = train_arguments(root, a.seed, a.mode)
    env = os.environ.copy()
    env["EMU_UNIFIED_CONTINUOUS"] = "cold-growth-approved"
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    print(f"Source {source_sha} | {a.mode} | 18 true prisms | 2048 worlds | 140M steps", flush=True)
    print("Launch: " + " ".join(cmd[3:]), flush=True)
    # Foreground; retains the existing canonical console.log / checkpoint manager.
    return subprocess.call(cmd, cwd=source, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
