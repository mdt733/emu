#!/usr/bin/env python3
"""Source-pinned KS150 teaching-hypothesis tournament (plans/locks/one fork).

No new network, robot, PPO algorithm or actor is introduced. The 120M parent
is immutable; every 160M fork inherits its full PPO/normalizer/teacher state.
Each treatment changes ONE existing, documented TrainingConfig knob.

Subcommands:
  list : inspect the control and 12 experimental lessons (no files touched)
  lock : pre-register source/parent/checkpoint SHA, seed and run recipes (no GPU)
  plan : show one or all exact commands from the immutable lock (no GPU)
  run  : launch ONE explicitly selected arm, checking lineage first

Separate evals/lesson synthesis happen outside this training launcher, never
inside the teacher or PPO loop. No automatic champion promotion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PARENT_SOURCE = "4048e8450c36882501cbb624f4ca656e11da9419"
PARENT_STEPS = 120_061_952
TARGET_STEPS = 160_000_000  # absolute, at next full PPO-update boundary
PROFILE = "h40-spring-only"
EXPERIMENT_ID = "ks150-stair-teaching-lessons-v1"
DEFAULT_PARENT = Path.home() / "emu-data/runs/h40-school-cold-120m-s736/checkpoints/final.pkl"
DEFAULT_ROOT = Path.home() / f"emu-data/studies/{EXPERIMENT_ID}"

# The current cold-school recipe, copied as an explicit continuation control.
# A one-variable branch overrides exactly one --flag value below.
REFERENCE = (
    ("--profile", "sim2real"), ("--pitch-deg", "8"),
    ("--terrain-actor", None), ("--terrain-step-max-m", ".055"),
    ("--unified-capabilities", None), ("--terrain-carpet", None),
    ("--carpet-prisms", None), ("--carpet-transition-cap-deg", "3"),
    ("--academy-stream", None), ("--stair-survival-curriculum", None),
    ("--terrain-lesson", "coherent-careful"),
    ("--foundation-steps", "20000000"), ("--foundation-mix", "main"),
    ("--growth-recipe", "main-mature"),
    ("--teaching-schedule", "terrain-early"),
    ("--force-program", "gentle"),
    ("--athletic-rehearsal-share", ".15"), ("--retention-share", ".30"),
    ("--academy-intro-steps", "60000000"),
    ("--academy-start-steps", "80000000"),
    ("--academy-ramp-steps", "20000000"),
    ("--academy-intro-share", ".10"),
    ("--academy-max-share", ".28"),
    ("--academy-height-cap-m", ".055"),
    ("--academy-course-episode-frames", "1200"),
    ("--fixed-eval-every-steps", "20000000"),
    ("--science-step-goal-weight", ".12"),
    ("--science-step-velocity-relief", ".55"),
    ("--science-step-approach-m", ".22"),
    ("--science-bilateral-support-bonus", "6"),
    ("--science-followthrough-bonus", "7"),
)

# ID -> (single changed flag, new setting, mechanism, predicted effect).
# An ablation is just as valuable as an improvement: it can expose a bad
# premise in the original teaching recipe.
ARMS = {
    "C00_control": (None, None,
        "Additional learning time", "Distinguishes any treatment from 40M more PPO steps"),
    "H01_direct_commands": ("--terrain-lesson", "coherent-direct",
        "Coherent step command fidelity", "Less attenuation improves prompt transfer"),
    "H02_more_step_goal": ("--science-step-goal-weight", ".22",
        "Supported progress credit", "More completed two-foot transfer without loss of support"),
    "H03_no_velocity_relief": ("--science-step-velocity-relief", "0",
        "Ablate stair velocity relief", "Tests whether relief encourages lingering/stalling"),
    "H04_nearer_lip_birth": ("--science-step-approach-m", ".15",
        "First-lip encounter distance", "More frequent timely lip approach vs missed starts"),
    "H05_first_foothold_credit": ("--science-foothold-bonus", "4",
        "First upper foothold credit", "Acquisition improves; may worsen two-foot continuation"),
    "H06_more_bilateral_credit": ("--science-bilateral-support-bonus", "9",
        "Five-frame bilateral credit", "Higher two-foot/lead-top conversion"),
    "H07_no_followthrough_bonus": ("--science-followthrough-bonus", "0",
        "Ablate post-transfer reward", "Tests whether the bonus distracts from safe first transfer"),
    "H08_more_academy_dose": ("--academy-max-share", ".40",
        "More stair episode births", "Stair learning rises; monitor core skill interference"),
    "H09_less_academy_dose": ("--academy-max-share", ".16",
        "Less stair episode births", "Tests whether 28% is already overtraining narrow stairs"),
    "H10_more_core_retention": ("--retention-share", ".38",
        "Preserve old core skills", "Better general retention; possible stair learning trade-off"),
    "H11_shorter_courses": ("--academy-course-episode-frames", "900",
        "Ablate prolonged first-life exposure", "More completed safe lessons; test for fragile transfer"),
    "H12_zero_only_gate": ("--science-gate-mode", "zero-only",
        "Ablate posture/contact gate", "Tests if gate suppresses useful step preparation"),
}


def _capture(*argv: str) -> str:
    return subprocess.check_output(argv, cwd=REPO, text=True).strip()


def _source_identity() -> str:
    source = _capture("git", "rev-parse", "HEAD")
    if _capture("git", "status", "--porcelain"):
        raise RuntimeError("Git checkout is dirty: source-pinned training is forbidden")
    if len(source) != 40:
        raise RuntimeError("Cannot identify the immutable training source")
    return source


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fd:
        for block in iter(lambda: fd.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parent_identity(path: Path) -> dict:
    parent = path.expanduser().resolve()
    if not parent.is_file() or parent.name != "final.pkl" or parent.parent.name != "checkpoints":
        raise RuntimeError("Need the complete, frozen checkpoints/final.pkl, not an .emupolicy")
    run_dir = parent.parent.parent
    cfg = json.loads((run_dir / "training_config.json").read_text())
    meta = json.loads((run_dir / "run_meta.json").read_text())
    if (meta.get("source_provenance", {}).get("git_commit") != PARENT_SOURCE or
            cfg.get("physical_profile") != PROFILE or
            cfg.get("seed") != 736 or cfg.get("envs") != 2048 or
            cfg.get("steps") != 120_000_000 or
            not cfg.get("stair_survival_curriculum")):
        raise RuntimeError("Parent does not match preregistered 120M seed-736 KS150 school")
    # Refuse a subtly different 120M recipe: an apparent single-knob fork
    # would otherwise be a many-factor treatment before it even begins.
    same_recipe = {
        "terrain_lesson": "coherent-careful",
        "foundation_steps": 20_000_000,
        "foundation_mix": "main",
        "growth_recipe": "main-mature",
        "teaching_schedule": "terrain-early",
        "force_program": "gentle",
        "athletic_rehearsal_frac": .15,
        "retention_share": .30,
        "academy_intro_steps": 60_000_000,
        "academy_start_steps": 80_000_000,
        "academy_ramp_steps": 20_000_000,
        "academy_intro_share": .10,
        "academy_max_share": .28,
        "academy_height_cap_m": .055,
        "academy_course_episode_frames": 1200,
        "science_step_goal_weight": .12,
        "science_step_velocity_relief": .55,
        "science_step_approach_m": .22,
        "science_bilateral_support_bonus": 6.,
        "science_followthrough_bonus": 7.,
        "science_gate_mode": "legacy",
    }
    differing = {key: (cfg.get(key), want) for key,want in same_recipe.items()
                 if cfg.get(key) != want}
    if differing:
        raise RuntimeError(f"Parent is not the preregistered baseline: {differing}")
    # The parent *file*, not just its directory label, must be the precise
    # completed 120,061,952-step study. Later, the existing train CLI will
    # independently validate the full PPO/sensor/normalizer contract before
    # any updates. No inference export or accidental earlier snapshot fits.
    with parent.open("rb") as fd:
        payload = pickle.load(fd)
    checkpoint_meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    if (checkpoint_meta.get("step") != PARENT_STEPS or
            (checkpoint_meta.get("source_provenance") or {}).get("git_commit") != PARENT_SOURCE or
            checkpoint_meta.get("robot_model_sha256") !=
            "21b5856b7a248e5d5b988e2735255d57ec79bae1a7cab0bcb72f481bfc36f470"):
        raise RuntimeError("Parent checkpoint bytes are not the exact finished seed-736 KS150 study")
    del payload
    # Git-authorized checkpoint validation remains the duty of emu.cli train:
    # it checks source, full state, sensor, optimizer and signed bytes.
    return {"path": str(parent), "sha256": _sha256(parent),
            "source_commit": PARENT_SOURCE, "step": PARENT_STEPS,
            "training_seed": 736, "physical_profile": PROFILE}


def _recipe(arm: str) -> list[str]:
    if arm not in ARMS:
        raise ValueError(f"Unknown hypothesis {arm!r}")
    flag, setting, _, _ = ARMS[arm]
    recipe = []
    for k, v in REFERENCE:
        if k == flag:
            v = setting
        recipe.append(k)
        if v is not None:
            recipe.append(v)
    if flag is not None and flag not in dict(REFERENCE):
        # Nonbaseline hypothesis flags are allowed only if known to the CLI
        # and explicitly listed in the pre-registration registry.
        recipe.extend((flag, setting))
    return recipe


def _lock_path(root: Path) -> Path:
    return root / "study_lock.json"


def _lock(root: Path, parent: Path, seed: int) -> Path:
    root = root.expanduser().resolve()
    if seed < 0:
        raise ValueError("Branch seed must be a nonnegative integer")
    parent_run = parent.expanduser().resolve().parent.parent
    if (REPO == root or REPO in root.parents or root == Path.home() or
            root.exists() or root == parent_run or parent_run in root.parents):
        raise RuntimeError("Choose a NEW study root outside Git and the parent lineage")
    source = _source_identity()
    parent_info = _parent_identity(parent)
    if source == PARENT_SOURCE:
        raise RuntimeError("The experiment requires a separate, immutable post-parent source")
    manifest = {
        "schema": "emu_stair_teaching_lesson_lab_v1",
        "id": EXPERIMENT_ID,
        "source_commit": source,
        "parent": parent_info,
        "branch_seed": seed,
        "target_absolute_steps": TARGET_STEPS,
        "inherited_parent_steps": PARENT_STEPS,
        "envs": 2048,
        "model_profile": PROFILE,
        "baseline_recipe": list(REFERENCE),
        "arms": {key: dict(flag=v[0],value=v[1],question=v[2],
                          prediction=v[3]) for key,v in ARMS.items()},
        "evaluation": dict(
            discovery_seeds=[6901, 6902],
            sealed_final_seed=7901,
            measurements_at_absolute_steps=[PARENT_STEPS, 130_000_000,
                                            140_000_000, TARGET_STEPS],
            release_qualification=False),
        "rules": ["Control and 12 one-knob variants share complete parent state",
                  "Never rank on Academy frontier alone",
                  "Core command, slope and recovery retention are co-primary",
                  "No auto-promotion or publication", "Save every 2M for audit"],
    }
    root.mkdir(parents=True)
    output = _lock_path(root)
    with output.open("x", encoding="utf-8") as fd:
        json.dump(manifest, fd, indent=2, sort_keys=True)
        fd.write("\n")
    return output


def _read_lock(root: Path) -> dict:
    lock = json.loads(_lock_path(root.expanduser().resolve()).read_text())
    if (lock.get("schema") != "emu_stair_teaching_lesson_lab_v1" or
            lock.get("id") != EXPERIMENT_ID or
            lock.get("target_absolute_steps") != TARGET_STEPS or
            lock.get("baseline_recipe") != list(REFERENCE) or
            set(lock.get("arms", {})) != set(ARMS) or
            any(lock["arms"][k]["flag"] != v[0] or
                lock["arms"][k]["value"] != v[1] for k,v in ARMS.items())):
        raise RuntimeError("Locked recipe differs from this executable's pre-registered design")
    if _source_identity() != lock["source_commit"]:
        raise RuntimeError("Source commit drift: check out exact study_lock source")
    parent = Path(lock["parent"]["path"])
    if not parent.is_file() or _sha256(parent) != lock["parent"]["sha256"]:
        raise RuntimeError("Parent checkpoint missing or changed since study lock")
    return lock


def _command(lock: dict, root: Path, arm: str) -> list[str]:
    run_dir = root / "runs" / arm
    # Resumed children MUST restore their own frozen config; external parent
    # proof is for the first fork only, never a mid-run source override.
    has_child_checkpoint = (run_dir / "checkpoints").is_dir() and any(
        (run_dir / "checkpoints").glob("*.pkl"))
    if has_child_checkpoint:
        resume = ["--resume", "auto"]
    else:
        resume = ["--resume", lock["parent"]["path"],
                  "--research-parent-source", lock["parent"]["source_commit"],
                  "--research-parent-sha256", lock["parent"]["sha256"]]
    return [sys.executable, "-m", "emu.cli", "train",
            "--run-dir", str(run_dir), "--steps", str(TARGET_STEPS),
            "--envs", "2048", "--seed", str(lock["branch_seed"]),
            "--device", "cuda", *_recipe(arm), *resume]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("list", "lock", "plan", "run"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--seed", type=int, default=8200)
    parser.add_argument("--arm", choices=tuple(ARMS), default=None)
    args = parser.parse_args(argv)
    if args.action == "list":
        for name,(flag,value,question,prediction) in ARMS.items():
            print(f"{name:27} {flag or 'CONTROL':39} {value or '-':15} {question}")
        return 0
    if args.action == "lock":
        print("Registered immutable 13-arm study:",
              _lock(args.root, args.parent, args.seed))
        return 0
    lock = _read_lock(args.root)
    arms = (args.arm,) if args.arm else tuple(ARMS)
    if args.action == "run" and args.arm is None:
        parser.error("A training launch must select exactly one --arm")
    root = args.root.expanduser().resolve()
    for arm in arms:
        cmd = _command(lock, root, arm)
        print(arm, ":", shlex.join(cmd), flush=True)
        if args.action == "run":
            env = os.environ.copy()
            env.update(EMU_PHYSICAL_PROFILE=PROFILE,
                       XLA_PYTHON_CLIENT_PREALLOCATE="false",
                       PYTHONPATH=str(REPO))
            # First validate full checkpoint/source/physical contract without
            # creating a new training run or performing a PPO update.
            subprocess.run([*cmd, "--check-resume-only"],
                           cwd=REPO, env=env, check=True)
            subprocess.run(cmd, cwd=REPO, env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
