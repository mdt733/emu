#!/usr/bin/env python3
"""Source-pinned KS150 teaching-hypothesis tournament (plans/locks/one fork).

No new network, robot, PPO algorithm or actor is introduced. The 120M parent
is immutable; every 160M fork inherits its full PPO/normalizer/teacher state.
Each treatment changes ONE existing, documented TrainingConfig knob.

Subcommands:
  list : inspect the control and 12 experimental lessons (no files touched)
  lock : pre-register source/parent/checkpoint SHA, seed and run recipes (no GPU)
  plan      : preview a selected absolute step boundary (no GPU)
  preflight : authenticate complete checkpoint and source (no GPU/PPO)
  run       : preflight then train ONE arm to a registered boundary
  examine   : frozen stair, signed-stair and core-retention exams for ONE arm
  go        : ONE command, all 13 arms sequentially, staged frozen exams
  results   : package a shareable results.zip, including partial campaigns

Separate evals/lesson synthesis happen outside this training launcher, never
inside the teacher or PPO loop. No automatic champion promotion.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import pickle
import shlex
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PARENT_SOURCE = "4048e8450c36882501cbb624f4ca656e11da9419"
PARENT_STEPS = 120_061_952
PARENT_ACTOR_SHA256 = "f1e808b2c1fd3cb6433ddbcea9f1c9e1afce69c69eb768fabf678907e7deecbc"
TARGET_STEPS = 160_000_000  # ultimate absolute requested target
STAGES = (130_000_000, 140_000_000, TARGET_STEPS)
PPO_BATCH_STEPS = 2048 * 32  # 2,048 worlds x 32 rollout frames
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


def _actor_weights_sha256(payload:dict) -> str:
    """The original immutable export's exact float32 actor byte contract.

    Only invoked when locking a complete parent checkpoint, never to
    inspect running actors or rewrite their weights.
    """
    import numpy as np
    sha=hashlib.sha256()
    layers=payload["state"]["params"]["pi"]
    if len(layers)!=3:
        raise RuntimeError("Parent checkpoint has an unexpected actor layout")
    for layer in layers:
        for key in ("w","b"):
            sha.update(np.asarray(layer[key],dtype="<f4").tobytes(order="C"))
    return sha.hexdigest()


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
    actor_sha=_actor_weights_sha256(payload)
    if actor_sha!=PARENT_ACTOR_SHA256:
        raise RuntimeError(
            "Parent actor weights differ from the independently examined "
            "frozen 120M seed-736 actor; do not fork a mislabeled checkpoint")
    del payload
    # Git-authorized checkpoint validation remains the duty of emu.cli train:
    # it checks source, full state, sensor, optimizer and signed bytes.
    return {"path": str(parent), "sha256": _sha256(parent),
            "source_commit": PARENT_SOURCE, "step": PARENT_STEPS,
            "training_seed": 736, "physical_profile": PROFILE,
            "actor_sha256": actor_sha}


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
        "milestone_targets": list(STAGES),
        "milestone_ppo_boundaries": [
            (step + PPO_BATCH_STEPS - 1)//PPO_BATCH_STEPS*PPO_BATCH_STEPS
            for step in STAGES],
        "inherited_parent_steps": PARENT_STEPS,
        "envs": 2048,
        "model_profile": PROFILE,
        "baseline_recipe": [list(pair) for pair in REFERENCE],
        "arms": {key: dict(flag=v[0],value=v[1],question=v[2],
                          prediction=v[3]) for key,v in ARMS.items()},
        "evaluation": dict(
            discovery_seeds=[6901, 6902],
            sealed_final_seed=7901,
            measurements_at_absolute_steps=[PARENT_STEPS, *STAGES],
            all_arms_stop_for_review=True,
            do_not_advance_on_training_reward=True,
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
            lock.get("milestone_targets") != list(STAGES) or
            lock.get("milestone_ppo_boundaries") != [
                (step+PPO_BATCH_STEPS-1)//PPO_BATCH_STEPS*PPO_BATCH_STEPS
                for step in STAGES] or
            lock.get("baseline_recipe") != [list(pair) for pair in REFERENCE] or
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


def _go_lock(root:Path,parent:Path,seed:int) -> dict:
    """One-button registration, including recovery of an UNUSED older lock.

    An earlier draft locked source before the GO runner existed. Once the
    user fast-forwarded Git, the source changed and GO refused to start.
    Never repair that by weakening source validation or rewriting the
    ancestry of *any* run that has already begun.

    Only a registration-only lock, with zero run/exam/review/journal data,
    may be replaced. The original bytes are archived and SHA-verified;
    the complete saved parent is reauthenticated. Source-pinning is then
    strict again for every stage and every physical examination.
    """
    root=root.expanduser().resolve()
    lockfile=_lock_path(root)
    if not lockfile.is_file():
        _lock(root,parent,seed)
        return _read_lock(root)
    try:
        return _read_lock(root)
    except RuntimeError as error:
        if str(error)!="Source commit drift: check out exact study_lock source":
            raise
    # Only after the exact error above. A dirty checkout still fails here.
    current_sha=_source_identity()
    old=json.loads(lockfile.read_text(encoding="utf-8"))
    old_sha=old.get("source_commit")
    if (not isinstance(old_sha,str) or len(old_sha)!=40 or
            old_sha==current_sha):
        raise RuntimeError("Unrecognized older lock; refuse automatic source repair")
    # The lock has already passed the full original preregistered recipe
    # check in _read_lock. Verify it is genuinely unused by scanning all
    # published study namespaces, including incomplete/failed attempts.
    used=("runs","evidence","milestones","reviews",
          "go_events.jsonl","lesson_ledger.md")
    occupied=[name for name in used if (root/name).exists()]
    allowed={"study_lock.json","go.log","go.running.lock",
             "retired_study_locks"}
    unexpected=[entry.name for entry in root.iterdir()
                if entry.name not in allowed]
    if occupied or unexpected:
        raise RuntimeError(
            "Existing study contains training/evidence or unrecognized files; "
            f"never rebind a started study. Occupied={occupied}, "
            f"unexpected={unexpected}. Check out locked source {old_sha} "
            "or select a fresh --root.")
    # Prevent two concurrent GO shells from simultaneously re-registering
    # the same unused study. A process already holding this lock wins.
    guard=root/"go.running.lock"
    with guard.open("a+") as guard_fd:
        try:
            fcntl.flock(guard_fd.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("GO already running; study lock not changed") from exc
        # Re-evaluate within lock, and never rebind if another launch has
        # created a journal or begun to write checkpoints.
        if any((root/name).exists() for name in used):
            raise RuntimeError("Study began during registration; refuse source rebind")
        if json.loads(lockfile.read_text(encoding="utf-8"))!=old:
            raise RuntimeError("Study lock changed during registration")
        parent_info=_parent_identity(Path(old["parent"]["path"]))
        if parent_info!=old["parent"]:
            raise RuntimeError(
                "Saved full parent identity no longer matches old study lock")
        retired=root/"retired_study_locks"
        retired.mkdir(exist_ok=True)
        archive=retired/f"study_lock-{old_sha}.json"
        original=lockfile.read_bytes()
        if archive.exists():
            if archive.read_bytes()!=original:
                raise RuntimeError("Archived original lock does not match; abort")
        else:
            with archive.open("xb") as fd:
                fd.write(original)
                fd.flush()
                os.fsync(fd.fileno())
        new={**old,
             "source_commit":current_sha,
             "unused_lock_upgrade":dict(
                 prior_source_commit=old_sha,
                 prior_lock_sha256=hashlib.sha256(original).hexdigest(),
                 reason="pre-training source fast-forward; no study data existed")}
        staged=root/".study_lock.json.next"
        if staged.exists():
            staged.unlink()  # only an unpublished registration tempfile
        try:
            with staged.open("x",encoding="utf-8") as fd:
                json.dump(new,fd,indent=2,sort_keys=True)
                fd.write("\n")
                fd.flush()
                os.fsync(fd.fileno())
            os.replace(staged,lockfile)
        finally:
            staged.unlink(missing_ok=True)
        print("GO: archived unused lock pinned to "+old_sha[:12]+
              "; registered current source "+current_sha[:12]+
              " without changing the original parent.",flush=True)
    return _read_lock(root)


def _attach_locked_training_checkout(root:Path) -> None:
    """Keep the TRAINER on the study's original Git SHA across supervisor fixes.

    This script is a read-only/launch supervisor running from the current
    checkout. If an existing study is locked to an older commit, create a
    clean detached Git worktree of THAT exact source, then run all training
    and CUDA exams from it. No checkpoint or study-lock source is changed.
    The worker reattaches on restart. The supervisor revision is separately
    reported rather than mislabelled as the PPO training provenance.
    """
    global REPO
    root=root.expanduser().resolve()
    lockfile=_lock_path(root)
    if not lockfile.is_file():
        return
    lock=json.loads(lockfile.read_text(encoding="utf-8"))
    source=lock.get("source_commit")
    if (not isinstance(source,str) or len(source)!=40 or
            any(ch not in "0123456789abcdef" for ch in source)):
        raise RuntimeError("Malformed locked training Git SHA")
    current=_source_identity()
    if current==source:
        return
    original_repo=REPO
    pinned=root/("training_source_"+source[:12])
    if pinned.exists():
        if not (pinned/".git").exists():
            raise RuntimeError("Locked-source checkout exists but is not a Git worktree")
    else:
        # The old commit must be available in the user's local Git history.
        # No network access, checkout switch, or source rewrite is required.
        subprocess.run(
            ["git","-C",str(original_repo),"cat-file","-e",source+"^{commit}"],
            check=True)
        subprocess.run(
            ["git","-C",str(original_repo),"worktree","add","--detach",
             str(pinned),source],
            check=True)
    REPO=pinned
    if _source_identity()!=source:
        REPO=original_repo
        raise RuntimeError("Detached training source is not the exact clean locked Git commit")
    print(f"GO: supervisor runs from {original_repo}, while every PPO run "
          f"and physical exam stays at locked source {source[:12]} in {pinned}.",
          flush=True)


def _stage_extension_allowed(root: Path, arm: str, to_steps: int) -> bool:
    """Reject stage skips and incomplete intermediate milestones.

    All arms must complete, inspect and deliberately advance in the same
    130M -> 140M -> 160M sequence; no terminal 40M gamble. The core
    inspect_run path also validates the prior final.pkl before extending.
    """
    if arm not in ARMS or to_steps not in STAGES:
        raise ValueError("Unknown arm or non-preregistered stage")
    run_dir = root / "runs" / arm
    cfg_path = run_dir / "training_config.json"
    if not cfg_path.is_file():
        if to_steps != STAGES[0]:
            raise RuntimeError("First fork must stop at 130M for matched review")
        return False
    old_steps = json.loads(cfg_path.read_text(encoding="utf-8")).get("steps")
    if old_steps not in STAGES:
        raise RuntimeError("Existing child has an unregistered training target")
    if to_steps == old_steps:
        return False  # an interrupted stage resumes only itself
    if to_steps < old_steps or to_steps != STAGES[STAGES.index(old_steps)+1]:
        raise RuntimeError("Do not skip stages, reverse targets or exceed 160M")
    prior_final = run_dir / "checkpoints" / "final.pkl"
    if not prior_final.is_file():
        raise RuntimeError("Finish the previous stage and its physical review first")
    return True


def _command(lock: dict, root: Path, arm: str, to_steps: int) -> list[str]:
    if arm not in ARMS or to_steps not in STAGES:
        raise ValueError("Unknown arm or target stage")
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
            "--run-dir", str(run_dir), "--steps", str(to_steps),
            "--envs", "2048", "--seed", str(lock["branch_seed"]),
            "--device", "cuda", *_recipe(arm), *resume]


def _exam_commands(lock:dict,root:Path,arm:str,to_steps:int,seed:int) -> tuple[Path,list[list[str]]]:
    """Four immutable outputs for one completed arm, stage and discovery seed.

    Exam worlds, speeds, horizon and physical geometry are the SAME for
    control and treatments. No champion output and no PPO training.
    """
    if (arm not in ARMS or to_steps not in STAGES or
            seed not in lock["evaluation"]["discovery_seeds"]):
        raise ValueError("Exams require one registered arm, stage and discovery seed")
    stage=f"{to_steps//1_000_000}m-s{seed}"
    output=root/"evidence"/arm/stage
    checkpoint=root/"runs"/arm/"checkpoints"/"final.pkl"
    actor=output/"actor.emupolicy"
    envpython=sys.executable
    commands=[
        [envpython,"-m","emu.cli","export-policy",str(checkpoint),str(actor)],
        [envpython,str(REPO/"scripts"/"v5053_matrix_exam.py"),
         "--policy",str(actor),"--output",str(output/"signed_stairs"),
         "--mode","stairs","--worlds","512","--levels","20",
         "--trials-per-level","10","--horizon","350","--seed",str(seed),
         "--device","cuda"],
        [envpython,str(REPO/"scripts"/"examine_h40_specialties.py"),
         "--policy",str(actor),"--output",str(output/"generalist_pulse.zip"),
         "--worlds","2048","--horizon","600","--seed",str(seed),
         "--device","cuda"],
        [envpython,str(REPO/"scripts"/"v5053_matrix_exam.py"),
         "--school-diagnostic-snapshot-suite",
         "--lab-study-lock",str(_lock_path(root)),
         "--lab-arm",arm,
         "--policy",str(actor),"--output",str(output/"stair_speed_horizon"),
         "--seed",str(seed),"--device","cuda"],
    ]
    return output,commands


def _validate_completed_arm(lock:dict,root:Path,arm:str,to_steps:int) -> None:
    """Inspect the exact finalized all-state checkpoint, before examination."""
    from emu.checkpoint_io import load_checkpoint
    from emu.training_config import TrainingConfig,training_sensor_contract
    from emu.checkpoint_io import update_boundary_at_or_after
    run_dir=root/"runs"/arm
    cfgpath=run_dir/"training_config.json"
    checkpoint=run_dir/"checkpoints"/"final.pkl"
    if not cfgpath.is_file() or not checkpoint.is_file():
        raise RuntimeError("Finish this arm and stage before requesting physical exams")
    cfg=TrainingConfig.load(cfgpath)
    if (cfg.steps!=to_steps or cfg.envs!=2048 or
            cfg.seed!=lock["branch_seed"] or cfg.physical_profile!=PROFILE):
        raise RuntimeError("Requested arm/stage differs from its saved experiment configuration")
    payload=load_checkpoint(
        checkpoint,validate=True,
        expected_observation_schema=cfg.observation_schema,
        expected_sensor_contract=training_sensor_contract(cfg))
    got=payload["meta"]
    if (got.get("step")!=update_boundary_at_or_after(to_steps,2048) or
            got.get("run_name")!=arm or
            (got.get("source_provenance") or {}).get("git_commit")!=lock["source_commit"]):
        raise RuntimeError("Completed checkpoint age, arm or source is not this study")


def _launch_env() -> dict[str, str]:
    """Use the trainer's explicit full-CUDA authorization for EVERY stage.

    A first cross-source continuation requires the same 'continuous'
    launch gate as a later same-source extension. A previous launcher set
    this only on extensions: check-resume-only passed, real train failed.
    """
    env = os.environ.copy()
    env.pop("EMU_UNIFIED_ACCEPTANCE", None)
    env.update(
        EMU_UNIFIED_CONTINUOUS="cold-growth-approved",
        EMU_PHYSICAL_PROFILE=PROFILE,
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        PYTHONPATH=str(REPO))
    return env



# --- Unattended, reproducible thirteen-arm study ---------------------------
#
# One GO means: 13 separate forks, each at 130M -> 140M -> 160M;
# two locked physical exam seeds at every milestone. The only automatic
# early stop is a prespecified large decline in real generalist capability,
# never an Academy training reward. No fork promotes a champion.
#
# Crash recovery uses existing validated full checkpoints and immutable
# signed evidence, never overwrites a completed exam or learns from a
# different source. The JSONL journal is not itself an authority.
GO_CORE = ("command", "control", "transition", "slope", "resilience")
GO_CAUTION_DROP = .05    # descriptive risk warning, NOT significance
GO_HARD_STOP_DROP = .15  # strong loss reproduced on BOTH exam seeds


def _event(root:Path, event:str, **details) -> None:
    item=dict(utc=datetime.now(timezone.utc).isoformat(),
              event=event,**details)
    with (root/"go_events.jsonl").open("a",encoding="utf-8") as fd:
        fd.write(json.dumps(item,sort_keys=True)+"\n")
        fd.flush()
        os.fsync(fd.fileno())
    print("LESSON LAB:",json.dumps(item,sort_keys=True),flush=True)


def _assert_actor(lock:dict,root:Path,arm:str,steps:int,actor_path:Path,
                  exam_seed:int) -> dict:
    from scripts.v5053_matrix_exam import _verified_lesson_lab_actor
    actor=json.loads(actor_path.read_text(encoding="utf-8"))
    source=actor.get("source")
    if not isinstance(source,dict):
        raise RuntimeError(f"Invalid frozen actor: {actor_path}")
    verified=_verified_lesson_lab_actor(
        _lock_path(root),arm,source,exam_seed=exam_seed)
    expected_step=lock["milestone_ppo_boundaries"][
        lock["milestone_targets"].index(steps)]
    if verified["step"]!=expected_step:
        raise RuntimeError("Frozen actor is from an unexpected milestone")
    return actor


def _exam_valid(kind:str,path:Path,actor:dict,seed:int) -> bool:
    """Treat an existing output as complete only after provenance checking."""
    if kind=="signed_stairs" or kind=="stair_speed_horizon":
        file=(path/"matrix_summary.json" if kind=="signed_stairs"
              else path/"school_diagnostic.json")
        if not file.is_file():
            return False
        meta=json.loads(file.read_text(encoding="utf-8"))
        if kind=="signed_stairs":
            m=meta["manifest"]
            if (m.get("seed")!=seed or m.get("mode")!="stairs" or
                    m.get("assigned_worlds")!=400 or m.get("horizon")!=350 or
                    m.get("actor_sha256")!=actor["source"]["actor_sha256"] or
                    m.get("robot_model_sha256")!=actor["source"]["robot_model_sha256"] or
                    m.get("actor_training_source")!=actor["source"]):
                raise RuntimeError(f"Signed-stair exam provenance mismatch: {path}")
            return (path/"matrix_worlds.csv").is_file()
        if (meta.get("seed")!=seed or
                meta.get("actor_sha256")!=actor["source"]["actor_sha256"] or
                meta.get("model_sha256")!=actor["source"]["robot_model_sha256"] or
                not meta.get("same_trajectory_prefix_verified") or
                not meta.get("early_horizon_captured_from_same_rollout") or
                meta.get("horizon_prefix_disagreement_count")!=0):
            raise RuntimeError(f"Same-trajectory exam provenance mismatch: {path}")
        return (path/"school_diagnostic_heights.csv").is_file()
    if kind=="generalist_pulse":
        if not path.is_file():
            return False
        with zipfile.ZipFile(path) as archive:
            m=json.loads(archive.read("manifest.json"))
            keys=set(archive.namelist())
        if (m.get("seed")!=seed or
                m.get("worlds")!=2048 or m.get("horizon_frames")!=600 or
                m.get("actor_sha256")!=actor["source"]["actor_sha256"] or
                m.get("actor_source")!=actor["source"] or
                m.get("robot_model_sha256")!=actor["source"]["robot_model_sha256"] or
                not {"summary.json","pulse.csv","worlds.csv"}.issubset(keys)):
            raise RuntimeError(f"Generalist pulse provenance mismatch: {path}")
        return True
    raise ValueError(f"Unknown physical exam {kind}")


def _alternate_output(base:Path) -> Path:
    """Never remove or overwrite raw evidence from a failed GPU attempt."""
    if not base.exists():
        return base
    for i in range(1,100):
        trial=base.with_name(base.stem+f"-retry-{i:02d}"+base.suffix)
        if not trial.exists():
            return trial
    raise RuntimeError("Exceeded preserved exam retry attempts")


def _find_completed_exam(kind:str,base:Path,actor:dict,seed:int) -> Path|None:
    candidates=[base]
    candidates+=sorted(base.parent.glob(
        base.stem+"-retry-*"+base.suffix))
    for candidate in candidates:
        if _exam_valid(kind,candidate,actor,seed):
            return candidate
    return None


def _physical_evidence(lock:dict,root:Path,arm:str,steps:int,seed:int,
                       *,env:dict) -> dict:
    """Finish or reuse exact exams with provenance; never repeat PPO."""
    if seed not in lock["evaluation"]["discovery_seeds"]:
        raise ValueError("Unregistered physical exam seed")
    output,commands=_exam_commands(lock,root,arm,steps,seed)
    receipt=output/"exam_receipt.json"
    # Once later training advances, its current final.pkl is no longer
    # the preceding-stage checkpoint. Immutable earlier receipts and actors
    # remain the source of those earlier examination facts.
    if not receipt.is_file():
        _validate_completed_arm(lock,root,arm,steps)
    output.mkdir(parents=True,exist_ok=True)
    actor_path=output/"actor.emupolicy"
    if not actor_path.exists():
        # The existing export helper writes JSON directly; a power loss
        # midway could leave a truncated .emupolicy. Export to a temporary
        # name, authenticate the complete actor, then publish atomically.
        partial=actor_path.with_name(actor_path.name+".partial")
        if partial.exists():
            partial.unlink()  # never a published/frozen exam artifact
        print(f"Frozen export: {arm} {steps:,} seed={seed}",flush=True)
        export_cmd=list(commands[0])
        export_cmd[-1]=str(partial)
        subprocess.run(export_cmd,cwd=REPO,env=env,check=True)
        _assert_actor(lock,root,arm,steps,partial,seed)
        os.replace(partial,actor_path)
    actor=_assert_actor(lock,root,arm,steps,actor_path,seed)
    chosen={}
    for kind,cmd in zip(
            ("signed_stairs","generalist_pulse","stair_speed_horizon"),
            commands[1:]):
        base=Path(cmd[cmd.index("--output")+1])
        found=_find_completed_exam(kind,base,actor,seed)
        if found is None and kind=="stair_speed_horizon" and base.is_dir():
            # If both long GPU cases finished but host aggregation failed,
            # recover from their immutable CSVs without another GPU run.
            all_long=all((base/label/"matrix_summary.json").is_file()
                         and (base/label/"matrix_worlds.csv").is_file()
                         for label in ("slow_1200","fast_1200"))
            if all_long and not (base/"school_diagnostic.json").exists():
                subprocess.run(
                    [sys.executable,str(REPO/"scripts"/"v5053_matrix_exam.py"),
                     "--school-diagnostic-summarize","--output",str(base)],
                    cwd=REPO,env=env,check=True)
                found=_find_completed_exam(kind,base,actor,seed)
        if found is None:
            target=_alternate_output(base)
            revised=list(cmd)
            revised[revised.index("--output")+1]=str(target)
            print(f"GPU exam {kind}: {arm} {steps:,}, seed {seed}",flush=True)
            subprocess.run(revised,cwd=REPO,env=env,check=True)
            if not _exam_valid(kind,target,actor,seed):
                raise RuntimeError(f"GPU exam finished without valid {kind}: {target}")
            found=target
        chosen[kind]=str(found.relative_to(output))
    data=dict(schema="emu_lesson_lab_exam_receipt_v1",arm=arm,
              target_steps=steps,seed=seed,source=lock["source_commit"],
              actor_sha256=actor["source"]["actor_sha256"],
              actor_path="actor.emupolicy",evidence=chosen)
    if receipt.exists():
        prior=json.loads(receipt.read_text(encoding="utf-8"))
        if prior!=data:
            raise RuntimeError(f"Refusing to rewrite changed exam receipt: {receipt}")
    else:
        with receipt.open("x",encoding="utf-8") as fd:
            json.dump(data,fd,indent=2,sort_keys=True)
            fd.write("\n")
            fd.flush();os.fsync(fd.fileno())
    return data


def _read_pulse(root:Path,arm:str,steps:int,seed:int) -> dict[str,dict]:
    evidence=root/"evidence"/arm/f"{steps//1_000_000}m-s{seed}"
    receipt=json.loads((evidence/"exam_receipt.json").read_text(encoding="utf-8"))
    if (receipt.get("arm")!=arm or receipt.get("target_steps")!=steps or
            receipt.get("seed")!=seed):
        raise RuntimeError("Mismatched physical exam receipt")
    archive=evidence/receipt["evidence"]["generalist_pulse"]
    with zipfile.ZipFile(archive) as zip_file:
        summary=json.loads(zip_file.read("summary.json"))
    result={}
    for r in summary["headlines"]:
        result[r["specialty"]]=dict(
            passed=int(r["passed"]),attempted=int(r["attempted"]),
            survived=int(r["survived"]),
            pass_fraction=float(r["pass_fraction"]),
            survival_fraction=int(r["survived"])/int(r["attempted"]))
    if set(result)!=set((*GO_CORE,"step")):
        raise RuntimeError("Frozen pulse did not cover all six specialties")
    return result


def _read_stair_funnel(root:Path,arm:str,steps:int,seed:int) -> dict:
    """Extract immutable physical gates, not school reward/frontier scores."""
    evidence=root/"evidence"/arm/f"{steps//1_000_000}m-s{seed}"
    receipt=json.loads((evidence/"exam_receipt.json").read_text(encoding="utf-8"))
    signed=evidence/receipt["evidence"]["signed_stairs"]/"matrix_summary.json"
    snapshot=evidence/receipt["evidence"]["stair_speed_horizon"]/"school_diagnostic.json"
    matrix=json.loads(signed.read_text(encoding="utf-8"))
    diagnostic=json.loads(snapshot.read_text(encoding="utf-8"))
    if (matrix["manifest"].get("seed")!=seed or
            diagnostic.get("seed")!=seed or
            not diagnostic.get("early_horizon_captured_from_same_rollout")):
        raise RuntimeError("Stair funnel lacks matched-seed physical evidence")
    outcomes={}
    for kind,label in (("step_up","uphill"),("step_down","downhill")):
        curve=[r for r in matrix["curves"] if r["kind"]==kind]
        if len(curve)!=20:
            raise RuntimeError(f"Missing signed stair heights for {kind}")
        denom=sum(int(r["worlds"]) for r in curve)
        if denom!=200:
            raise RuntimeError("Stair diagnostic changed the assigned exam cohort")
        for gate,alias in (("leading_top","first_foot"),
                           ("each_foot_top_five","both_feet"),
                           ("traversed_5frame","traversal")):
            count=sum(int(r[gate]) for r in curve)
            if not 0<=count<=denom:
                raise RuntimeError("Invalid signed stair gate")
            outcomes[f"{label}_{alias}"]=count/denom
    # The previous 120M investigation showed an almost saturated small-
    # height first foothold but a separate 32mm+ acquisition cliff.
    # Never hide both phenomena behind one 4..80mm mean.
    uphill=[r for r in matrix["curves"] if r["kind"]=="step_up"]
    for band,low,high,expected in (
            ("uphill_4_28mm",4.,28.,70),
            ("uphill_32_80mm",32.,80.,130)):
        rows=[r for r in uphill if low<=float(r["requested_step_mm"])<=high]
        n=sum(int(r["worlds"]) for r in rows)
        if n!=expected:
            raise RuntimeError(f"Stair height band {band} is incomplete")
        for gate,alias in (("leading_top","first_foot"),
                           ("each_foot_top_five","both_feet"),
                           ("traversed_5frame","traversal")):
            outcomes[f"{band}_{alias}"]=sum(int(r[gate]) for r in rows)/n
    t=diagnostic["totals"]
    if t.get("worlds_per_condition")!=200:
        raise RuntimeError("Changed same-trajectory diagnostic cohort")
    for label in ("slow_350","slow_1200","fast_350","fast_1200"):
        for gate in ("lead_foot_top","both_feet_top_five",
                     "traversed_5frame","survived"):
            count=int(t[label+"_"+gate])
            if not 0<=count<=200:
                raise RuntimeError("Invalid first-life stair gate")
            outcomes[label+"_"+gate]=count/200
    return outcomes


def _review_stage(lock:dict,root:Path,arm:str,steps:int) -> dict:
    """Objective retention comparison against SAME-AGE control, both seeds.

    A >5 percentage point decline is documented; a >15-point decline in
    the same core specialty on BOTH independently seeded exam cohorts
    automatically stops further training of THIS arm. It does not stop
    other independent arms. No training reward, frontier or rank is used.
    """
    seeds=tuple(lock["evaluation"]["discovery_seeds"])
    samples={seed:_read_pulse(root,arm,steps,seed) for seed in seeds}
    stair={seed:_read_stair_funnel(root,arm,steps,seed) for seed in seeds}
    warnings=[]
    severe=[]
    contrasts={}
    stair_deltas={}
    if arm!="C00_control":
        controls={seed:_read_pulse(root,"C00_control",steps,seed)
                  for seed in seeds}
        control_stair={seed:_read_stair_funnel(
            root,"C00_control",steps,seed) for seed in seeds}
        for gate in stair[seeds[0]]:
            stair_deltas[gate]=round(
                sum(stair[seed][gate]-control_stair[seed][gate]
                    for seed in seeds)/len(seeds)*100.,3)
        for specialty in GO_CORE:
            diffs={}
            for gate in ("pass_fraction","survival_fraction"):
                values=[samples[seed][specialty][gate]-
                        controls[seed][specialty][gate] for seed in seeds]
                diffs[gate]=[round(v,6) for v in values]
                if min(values)<-GO_CAUTION_DROP:
                    warnings.append(dict(specialty=specialty,gate=gate,
                                         deltas=diffs[gate]))
                if all(v<=-GO_HARD_STOP_DROP for v in values):
                    severe.append(dict(specialty=specialty,gate=gate,
                                       deltas=diffs[gate]))
            contrasts[specialty]=diffs
    outcome=dict(schema="emu_lesson_lab_stage_review_v1",
                 arm=arm,target_steps=steps,seeds=list(seeds),
                 reference="same-age C00_control",
                 caution_threshold=GO_CAUTION_DROP,
                 automatic_hard_stop_threshold=GO_HARD_STOP_DROP,
                 warnings=warnings,hard_stops=severe,
                 advancement="stop_this_arm" if severe else "continue",
                 contrast_vs_control=contrasts,
                 stair_gate_deltas_pp_vs_control=stair_deltas,
                 stair_gate_fractions={
                     gate:round(sum(stair[seed][gate] for seed in seeds)/
                                len(seeds),6)
                     for gate in stair[seeds[0]]})
    location=root/"reviews"/arm
    location.mkdir(parents=True,exist_ok=True)
    dest=location/f"{steps//1_000_000}m.json"
    if dest.exists():
        if json.loads(dest.read_text(encoding="utf-8"))!=outcome:
            raise RuntimeError(f"Stage review changed; refusing overwrite: {dest}")
    else:
        with dest.open("x",encoding="utf-8") as fd:
            json.dump(outcome,fd,indent=2,sort_keys=True)
            fd.write("\n")
            fd.flush();os.fsync(fd.fileno())
    return outcome


def _write_lesson_progress(root:Path) -> None:
    """Synthesize factual reviews; never decide which policy to publish."""
    lines=["# Stair-school lesson ledger (automatic, provisional)",
           "",
           "Single-seed-parent explanatory screen. All effects are relative "
           "to same-stage, same-physical-world control, averaged across "
           "discovery seeds 6901 and 6902.",
           "A treatment can hurt or help; negative results are retained.",
           "No automatic champion selection, newborn teaching change or "
           "release claim.",
           "",
           "| Treatment | 130M | 140M | 160M | Small uphill 2-foot Δpp | "
           "All uphill traversal Δpp | Core retention cautions |",
           "|---|---|---|---|---:|---:|---|"]
    for arm in ARMS:
        cells=[]
        risks=[]
        latest=None
        for steps in STAGES:
            path=root/"reviews"/arm/f"{steps//1_000_000}m.json"
            if not path.exists():
                cells.append("not measured")
                continue
            review=json.loads(path.read_text(encoding="utf-8"))
            latest=review
            cells.append("STOP" if review["hard_stops"] else "measured")
            risks.extend(f"{w['specialty']} {w['gate']} @"
                         f" {steps//1_000_000}M"
                         for w in review["warnings"])
        stair=(latest or {}).get("stair_gate_deltas_pp_vs_control",{})
        bilateral=f"{stair['uphill_4_28mm_both_feet']:+.2f}" if stair else "—"
        traversal=f"{stair['uphill_traversal']:+.2f}" if stair else "—"
        lines.append(f"| {arm} | "+" | ".join(cells)+
                     f" | {bilateral} | {traversal} | "+
                     (", ".join(sorted(set(risks))) or "none flagged")+" |")
    lines+=["","Deltas are **percentage points**, not policy rankings. "
            "Each latest-stage contrast is calculated at the same "
            "requested training age, with identical signed 4–80mm "
            "physical fixtures and discovery seeds.",
            "The raw evidence also preserves downward transfer, "
            "slow/fast 350/1200-frame first-life gates and exact "
            "per-height stair counts.",
            "",
            "This is a descriptive dashboard, not twelve confirmed "
            "developmental lessons. For every arm, examine the frozen "
            "signed-height curves, two-foot transfer funnel, 350/1200-frame "
            "same-trajectory snapshots, and actual teacher birth counts.",
            "A newborn-school lesson remains a hypothesis until replicated "
            "in fresh from-zero training. The prior 120M student was already "
            "trained on stairs."]
    tmp=root/"lesson_ledger.md.tmp"
    tmp.write_text("\n".join(lines)+"\n",encoding="utf-8")
    os.replace(tmp,root/"lesson_ledger.md")


def _save_stage_ancestor(root:Path,lock:dict,arm:str,steps:int) -> Path:
    """Retain the FULL original optimizer/teacher checkpoint at every stage.

    Future same-run extensions replace checkpoints/final.pkl atomically.
    Link into the milestone archive before that replacement (copy across
    filesystems) so 130M/140M rewind points remain research-reproducible.
    This is a distinct immutable checkpoint, not an inference-only actor.
    """
    source=root/"runs"/arm/"checkpoints"/"final.pkl"
    directory=root/"milestones"/arm/f"{steps//1_000_000}m"
    directory.mkdir(parents=True,exist_ok=True)
    final=directory/"final.pkl"
    manifest=directory/"milestone.json"
    if final.exists():
        if not manifest.is_file():
            raise RuntimeError(f"Milestone final exists without immutable manifest: {final}")
        previous=json.loads(manifest.read_text(encoding="utf-8"))
        if (previous.get("source_commit")!=lock["source_commit"] or
                previous.get("arm")!=arm or previous.get("requested_steps")!=steps or
                _sha256(final)!=previous.get("sha256")):
            raise RuntimeError(f"Milestone ancestor bytes/provenance changed: {final}")
        return final
    if manifest.exists():
        raise RuntimeError(f"Milestone manifest exists without its checkpoint: {manifest}")
    digest=_sha256(source)
    temporary=directory/".final.pkl.partial"
    if temporary.exists():
        temporary.unlink()  # unpublished crashed copy, never an exam
    try:
        try:
            os.link(source,temporary)
        except OSError:
            with source.open("rb") as inp, temporary.open("xb") as out:
                shutil.copyfileobj(inp,out,1024*1024)
                out.flush();os.fsync(out.fileno())
        if _sha256(temporary)!=digest:
            raise RuntimeError("Milestone full-state checkpoint copy mismatch")
        os.replace(temporary,final)
    finally:
        temporary.unlink(missing_ok=True)
    receipt=dict(schema="emu_school_milestone_full_state_v1",arm=arm,
                 source_commit=lock["source_commit"],requested_steps=steps,
                 sha256=digest,checkpoint="final.pkl")
    with manifest.open("x",encoding="utf-8") as out:
        json.dump(receipt,out,indent=2,sort_keys=True)
        out.write("\n")
        out.flush();os.fsync(out.fileno())
    return final


# --- Portable evidence, not a second copy of the PPO checkpoints ----------
BUNDLE_EXTS=frozenset((".json",".jsonl",".md",".csv",".zip"))
BUNDLE_RUN_EXTS=frozenset((".json",".jsonl",".md",".csv"))


def _bundle_results(root:Path) -> Path:
    """Make/update one atomic upload-ready results.zip, including partial runs."""
    root=root.expanduser().resolve()
    if not root.is_dir() or not _lock_path(root).is_file():
        raise FileNotFoundError("Bundle needs an existing locked study root")
    output=root/"results.zip"
    temporary=root/".results.zip.partial"
    with (root/".results_bundle.lock").open("a+") as guard:
        fcntl.flock(guard.fileno(),fcntl.LOCK_EX)
        chosen={}
        for name in ("study_lock.json","go_events.jsonl","lesson_ledger.md"):
            file=root/name
            if file.is_file() and not file.is_symlink():
                chosen[name]=file
        for folder in ("runs","evidence","reviews","milestones"):
            parent=root/folder
            if not parent.is_dir() or parent.is_symlink():
                continue
            for file in parent.rglob("*"):
                if not file.is_file() or file.is_symlink():
                    continue
                relative=file.relative_to(root)
                if any(part.startswith(".") or part.endswith(".partial")
                       for part in relative.parts):
                    continue
                if folder=="milestones" and file.name!="milestone.json":
                    continue
                exts=BUNDLE_RUN_EXTS if folder=="runs" else BUNDLE_EXTS
                if file.suffix.lower() not in exts:
                    continue
                if folder=="runs" and "checkpoints" in relative.parts:
                    continue
                chosen[relative.as_posix()]=file
        events=[]
        journal=root/"go_events.jsonl"
        if journal.is_file():
            for line in journal.read_text(encoding="utf-8").splitlines():
                try:
                    item=json.loads(line)
                except json.JSONDecodeError:
                    continue
                if item.get("event") in ("study_started","study_failed","study_finished"):
                    events.append(item["event"])
        complete=bool(events and events[-1]=="study_finished")
        lock=json.loads(_lock_path(root).read_text(encoding="utf-8"))
        readme=(
            "# EMU stair-school evidence bundle\n\n"
            f"Status: {'COMPLETE' if complete else 'PARTIAL / RUNNING / STOPPED'}\n"
            f"Training source: {lock.get('source_commit','unknown')}\n"
            f"Parent SHA256: {lock.get('parent',{}).get('sha256','unknown')}\n\n"
            "Included: per-world signed stair and same-trajectory exams, "
            "six-specialty physical pulse ZIPs, original training metrics, "
            "control-relative stage reviews, stage manifests and lesson ledger "
            "(when available), with file SHA256 hashes in bundle_manifest.json.\n"
            "A missing arm or stage is UNTESTED, not a negative result. "
            "One-parent design-time research is not newborn validation.\n"
            "Omitted: enormous PPO .pkl checkpoints, frozen .emupolicy actors, "
            "source worktrees and unpublished .partial files. Their training "
            "and actor hashes remain in the included provenance metadata.\n")
        files=[]
        try:
            with zipfile.ZipFile(temporary,"w",
                                 compression=zipfile.ZIP_DEFLATED,
                                 compresslevel=6,allowZip64=True) as archive:
                for name,file in sorted(chosen.items()):
                    before=file.stat()
                    archive.write(
                        file,arcname=name,
                        compress_type=(zipfile.ZIP_STORED if file.suffix==".zip"
                                       else zipfile.ZIP_DEFLATED))
                    after=file.stat()
                    if (before.st_size!=after.st_size or
                            before.st_mtime_ns!=after.st_mtime_ns):
                        raise RuntimeError(f"Evidence changed during ZIP: {file}")
                    files.append(dict(path=name,size=before.st_size,
                                      sha256=_sha256(file)))
                log=root/"go.log"
                if log.is_file() and not log.is_symlink():
                    with log.open("rb") as inp:
                        inp.seek(0,os.SEEK_END)
                        total=inp.tell()
                        inp.seek(max(0,total-2_000_000))
                        archive.writestr("go_log_tail.txt",inp.read())
                archive.writestr("README.md",readme)
                manifest=dict(
                    schema="emu_stair_school_results_bundle_v1",
                    generated_utc=datetime.now(timezone.utc).isoformat(),
                    complete=complete,
                    training_source_commit=lock.get("source_commit"),
                    parent_sha256=lock.get("parent",{}).get("sha256"),
                    stage_reviews=sum(1 for r in files if
                                      r["path"].startswith("reviews/") and
                                      r["path"].endswith(".json")),
                    file_count=len(files),files=files)
                archive.writestr("bundle_manifest.json",
                                 json.dumps(manifest,indent=2,sort_keys=True)+"\n")
            with zipfile.ZipFile(temporary) as check:
                bad=check.testzip()
                if bad:
                    raise RuntimeError(f"Corrupt ZIP member {bad}")
            os.replace(temporary,output)
        finally:
            temporary.unlink(missing_ok=True)
    print(f"RESULTS ZIP: {output} ({len(files)} evidence files, "
          f"{'complete' if complete else 'partial'})",flush=True)
    return output


def _go_run(root:Path,lock:dict) -> int:
    """Run one ordered study in this LOCAL process, guarded against duplicates."""
    root=root.resolve()
    with (root/"go.running.lock").open("a+") as fd:
        try:
            fcntl.flock(fd.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("The study is already running; inspect go.log") from exc
        env=_launch_env()
        _event(root,"study_started",total_arms=len(ARMS),
               source=lock["source_commit"],stages=list(STAGES))
        for arm in ARMS:
            stopped=False
            for steps in STAGES:
                # Prevent source drift across a multi-day campaign before
                # doing another expensive PPO update.
                if _source_identity()!=lock["source_commit"]:
                    raise RuntimeError("Study Git source drift during unattended GO")
                existing_cfg=root/"runs"/arm/"training_config.json"
                if existing_cfg.is_file():
                    age=json.loads(existing_cfg.read_text(encoding="utf-8")).get("steps")
                    if age not in STAGES:
                        raise RuntimeError(f"{arm} has unknown saved target age {age}")
                    if age>steps:
                        # The earlier stage's immutable receipt is mandatory.
                        for seed in lock["evaluation"]["discovery_seeds"]:
                            archive=root/"evidence"/arm/f"{steps//1_000_000}m-s{seed}"/"exam_receipt.json"
                            if not archive.is_file():
                                raise RuntimeError(f"{arm} advanced before prior exams: {archive}")
                        milestone=root/"milestones"/arm/f"{steps//1_000_000}m"
                        if not (milestone/"milestone.json").is_file():
                            raise RuntimeError(
                                f"{arm} advanced without full-stage archival: {milestone}")
                        _save_stage_ancestor(root,lock,arm,steps)
                        review=_review_stage(lock,root,arm,steps)
                        _write_lesson_progress(root)
                        if review["hard_stops"]:
                            raise RuntimeError(f"{arm} advanced after a serious core regression")
                        continue
                prior_review=(root/"reviews"/arm/
                              f"{STAGES[STAGES.index(steps)-1]//1_000_000}m.json"
                              if steps!=STAGES[0] else None)
                if prior_review is not None:
                    if not prior_review.is_file():
                        raise RuntimeError(f"{arm} has no preceding physical review")
                    if json.loads(prior_review.read_text())["hard_stops"]:
                        stopped=True
                        break
                _event(root,"arm_stage_start",arm=arm,requested_steps=steps)
                milestone=(root/"milestones"/arm/
                           f"{steps//1_000_000}m"/"milestone.json")
                if milestone.is_file():
                    # A previous attempt already completed the ENTIRE PPO
                    # stage. Never re-run 10M updates or its final fixed
                    # exams just because a later evidence validator failed.
                    _validate_completed_arm(lock,root,arm,steps)
                    archive=_save_stage_ancestor(root,lock,arm,steps)
                    _event(root,"arm_stage_reused",arm=arm,
                           requested_steps=steps,
                           preserved_full_state=str(archive))
                else:
                    cmd=_command(lock,root,arm,steps)
                    _stage_extension_allowed(root,arm,steps)
                    subprocess.run([*cmd,"--check-resume-only"],
                                   cwd=REPO,env=env,check=True)
                    subprocess.run(cmd,cwd=REPO,env=env,check=True)
                    _validate_completed_arm(lock,root,arm,steps)
                    archive=_save_stage_ancestor(root,lock,arm,steps)
                    _event(root,"arm_stage_trained",arm=arm,
                           requested_steps=steps,
                           preserved_full_state=str(archive))
                for seed in lock["evaluation"]["discovery_seeds"]:
                    receipt=_physical_evidence(lock,root,arm,steps,seed,env=env)
                    _event(root,"arm_stage_examined",arm=arm,
                           requested_steps=steps,seed=seed,
                           actor_sha256=receipt["actor_sha256"])
                review=_review_stage(lock,root,arm,steps)
                _write_lesson_progress(root)
                _event(root,"stage_reviewed",arm=arm,requested_steps=steps,
                       result=review["advancement"],
                       warnings=len(review["warnings"]),
                       hard_stops=len(review["hard_stops"]))
                _bundle_results(root)
                if review["hard_stops"]:
                    stopped=True
                    break
            _event(root,"arm_finished",arm=arm,
                   outcome="safety_stopped" if stopped else "all_stages_complete")
        _write_lesson_progress(root)
        _event(root,"study_finished",arms=len(ARMS))
        _bundle_results(root)
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("list", "lock", "plan", "preflight", "run", "examine", "go", "results"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--seed", type=int, default=8200)
    parser.add_argument("--arm", choices=tuple(ARMS), default=None)
    parser.add_argument("--to-steps", type=int, choices=STAGES,
                        default=STAGES[0],
                        help="one matched stopping boundary: 130M, 140M, 160M")
    parser.add_argument("--detach",action="store_true",
                        help="GO: continue sequential training in an independent "
                             "local process; stdout/stderr in go.log")
    parser.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    parser.add_argument("--exam-seed", type=int, choices=(6901,6902),
                        default=6901,help="preregistered discovery exam cohort")
    parser.add_argument("--reviewed", action="store_true",
                        help="acknowledge matched physical/retention review "
                             "before advancing beyond the first stage")
    args = parser.parse_args(argv)
    if args.action == "list":
        for name,(flag,value,question,prediction) in ARMS.items():
            print(f"{name:27} {flag or 'CONTROL':39} {value or '-':15} {question}")
        return 0
    if args.action == "lock":
        print("Registered immutable 13-arm study:",
              _lock(args.root, args.parent, args.seed))
        return 0
    if args.action == "results":
        _bundle_results(args.root)
        return 0
    if args.action == "go":
        root=args.root.expanduser().resolve()
        _attach_locked_training_checkout(root)
        lock=_go_lock(root,args.parent,args.seed)
        if args.detach and not args.worker:
            logfile=root/"go.log"
            argv_worker=[sys.executable,"-u",str(Path(__file__).resolve()),
                         "go","--root",str(root),"--worker"]
            with logfile.open("ab",buffering=0) as stream:
                process=subprocess.Popen(
                    argv_worker,cwd=REPO,stdin=subprocess.DEVNULL,
                    stdout=stream,stderr=subprocess.STDOUT,
                    start_new_session=True,env=_launch_env())
            print(f"GO STARTED on this computer: PID {process.pid}. "
                  f"Follow {logfile}. Study is source/checkpoint locked.",
                  flush=True)
            return 0
        if args.detach and args.worker:
            raise ValueError("Invalid detached worker argument")
        if args.worker and not _lock_path(root).exists():
            raise RuntimeError("Worker cannot create an unverified study")
        try:
            return _go_run(root,lock)
        except Exception as exc:
            _event(root,"study_failed",failure_type=type(exc).__name__,
                   message=str(exc))
            try:
                _bundle_results(root)
            except Exception as bundle_error:
                print(f"RESULTS ZIP also failed: {bundle_error!r}",flush=True)
            raise
    if args.detach or args.worker:
        parser.error("--detach belongs only to the one-button GO action")
    lock = _read_lock(args.root)
    arms = (args.arm,) if args.arm else tuple(ARMS)
    if args.action in ("preflight", "run", "examine") and args.arm is None:
        parser.error("Select exactly one --arm for preflight, launch or exams")
    root = args.root.expanduser().resolve()
    if args.action == "examine":
        if args.reviewed:
            parser.error("--reviewed is for TRAINING stage advancement, not exams")
        arm=args.arm
        _validate_completed_arm(lock,root,arm,args.to_steps)
        output,commands=_exam_commands(
            lock,root,arm,args.to_steps,args.exam_seed)
        if output.exists():
            raise FileExistsError(
                f"Evidence already exists; never overwrite a prior exam: {output}")
        for cmd in commands:
            print("FROZEN EXAM:",shlex.join(cmd),flush=True)
        output.mkdir(parents=True)
        env=_launch_env()
        for cmd in commands:
            subprocess.run(cmd,cwd=REPO,env=env,check=True)
        print(f"Frozen evidence complete for {arm} stage={args.to_steps:,} "
              f"seed={args.exam_seed} at {output}",flush=True)
        return 0
    for arm in arms:
        cmd = _command(lock, root, arm, args.to_steps)
        print(arm, ":", shlex.join(cmd), flush=True)
        if args.action in ("preflight", "run"):
            extending = _stage_extension_allowed(root, arm, args.to_steps)
            if extending and not args.reviewed:
                parser.error("Advance requires --reviewed after independent "
                             "stair and general-skill retention exams")
            env = _launch_env()
            # The existing source/parent/optimizer/normalizer/teacher validator
            # is run before we enter the CUDA learner. No folder or PPO update
            # is created by this check, including during stage extension.
            subprocess.run([*cmd, "--check-resume-only"],
                           cwd=REPO, env=env, check=True)
            print(f"LAB CHECKPOINT PREFLIGHT PASS: {arm} -> {args.to_steps:,}",
                  flush=True)
            if args.action == "preflight":
                continue
            subprocess.run(cmd, cwd=REPO, env=env, check=True)
            print(f"STOP: {arm} reached requested stage {args.to_steps:,}; "
                  "retain frozen physical and core-skill evidence before advancing.",
                  flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
