#!/usr/bin/env python3
"""v5018 adaptive step-height experiment lab.

One 25M cold start -> six matched 35M forks -> best two 50M continuations.
The obstacle height is an earned CONTINUOUS physical frontier, not a fixed
5/10mm switch. All branches retain vx/vy/yaw/transition/resilience curriculum.
Default mode is PLAN_ONLY; --run deliberately starts GPU work.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import os
import subprocess
import sys
from datetime import datetime

# Allow direct script invocation and python -m execution alike.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.v5018_overnight_campaign import (
    BRANCH, ROOT, DEFAULT_BASE, preflight, run_command, archive,
    validated_checkpoint, export_one, exam, summarize_crossings, summarize_training,
    log_stage, required_target,
)

VERSION = "v5018-height-envelope-1"
MANIFEST = "height_envelope_manifest.json"
# Exactly one ancestor state. Every screen arm runs the SAME 10M additional
# environment steps with the SAME RNG seed / actor / optimizer / normalizer.
# A parameter in each row is a real intervention, not a promotional label.
ARMS = {
    "flat_control": dict(flat_control=True),
    "height_cautious": dict(limit_mm=80, exposure=.45, edge=.25,
                            growth_mm=.6, initial_mm=3., progress=0.),
    "height_balanced": dict(limit_mm=120, exposure=.60, edge=.35,
                            growth_mm=1., initial_mm=3., progress=0.),
    "height_edge": dict(limit_mm=150, exposure=.75, edge=.60,
                        growth_mm=1.4, initial_mm=3., progress=0.),
    "height_guided": dict(limit_mm=120, exposure=.60, edge=.35,
                          growth_mm=1., initial_mm=3., progress=.04),
    "height_guided_edge": dict(limit_mm=150, exposure=.75, edge=.60,
                               growth_mm=1.4, initial_mm=3., progress=.04),
}
# Physical exam is independent, real MuJoCo contacts and independent from the
# TRAINING height sampler. A score is not a real crossing until it passes the
# world-frame gait/contact/survival examiner's explicit crossing_pass.
SCREEN_HEIGHTS = (0, 20, 40, 80, 120)
FINAL_HEIGHTS = (0, 20, 40, 80, 120, 150)


def config(args):
    return dict(
        version=VERSION, seed=args.seed, envs=args.envs,
        foundation_steps=args.foundation_steps,
        screen_increment=args.screen_increment,
        final_increment=args.final_increment,
        exam_worlds=args.exam_worlds,
        exam_horizon=args.exam_horizon,
        pitch_deg=8.,
        arms=ARMS,
        screen_heights=list(SCREEN_HEIGHTS), final_heights=list(FINAL_HEIGHTS),
    )


def assert_args(args):
    if not 128 <= args.envs <= 2048 or args.envs % 128:
        raise ValueError("Use a multiple of 128 GPU worlds from 128..2048")
    if not (20_000_000 <= args.foundation_steps <= 30_000_000):
        raise ValueError("25M startup default; configurable only in 20..30M")
    if not (2_000_000 <= args.screen_increment <= 15_000_000):
        raise ValueError("Screen stage must have useful multi-million-step budget")
    if not (5_000_000 <= args.final_increment <= 30_000_000):
        raise ValueError("Final stage must have 5..30M additional steps")
    if not 2 <= args.exam_worlds <= 8 or not 250 <= args.exam_horizon <= 700:
        raise ValueError("Exam: 2..8 worlds, at least 250 control frames")
    if args.seed < 0:
        raise ValueError("Seed must be nonnegative")


def branch_args(arm: str) -> list[str]:
    hypothesis = ARMS[arm]
    if hypothesis.get("flat_control"):
        return ["--terrain-height-flat-control"]
    return [
        "--terrain-height-ceiling-mm", str(hypothesis["limit_mm"]),
        "--terrain-height-exposure", str(hypothesis["exposure"]),
        "--terrain-height-edge-fraction", str(hypothesis["edge"]),
        "--terrain-height-growth-mm", str(hypothesis["growth_mm"]),
        "--terrain-height-initial-mm", str(hypothesis["initial_mm"]),
        "--terrain-progress-weight", str(hypothesis["progress"]),
    ]


def training(root: Path, name: str, *, args, target: int,
             parent: Path | None, arm: str | None) -> Path:
    run = root / name
    ck = run / "checkpoints" / "final.pkl"
    if ck.exists():
        return validated_checkpoint(run, target, args.envs)
    # No stage writes inside the Git repo. Existing invalid/incomplete stage
    # cannot silently start fresh and diverge from its predecessor.
    cks = list((run / "checkpoints").glob("*.pkl"))
    resume = "auto" if cks else str(parent) if parent is not None else ""
    cmd = [
        "emu", "train", "--run-dir", str(run),
        "--steps", str(target), "--seed", str(args.seed),
        "--envs", str(args.envs), "--device", "cuda",
        "--sensor-actor", "--pitch-deg", "8",
        "--profile", "foundation" if arm is None else "sim2real",
    ]
    if arm is not None:
        cmd += branch_args(arm)
    if resume:
        cmd += ["--resume", resume]
    run_command(root, "train_" + name, cmd)
    return validated_checkpoint(run, target, args.envs)


def inspect_actor(root: Path, label: str, checkpoint: Path,
                  heights: tuple[int, ...], seeds: tuple[int, ...],
                  args) -> dict:
    # Idempotent on restart: rerun the read-only exams if requested. No
    # source/checkpoint weights are touched. A complete final checkpoint is
    # identified before export, not by a stale stage exit code.
    actor = export_one(root, label, checkpoint)
    files = []
    for seed in seeds:
        for height in heights:
            tag = f"{label}_height{height:03d}_seed{seed}"
            path = root / (tag + ".json")
            if not path.exists():
                exam(root, tag, actor, height=height, seed=seed,
                     worlds=args.exam_worlds, horizon=args.exam_horizon)
            files.append(path)
    physical = summarize_crossings(files)
    train_run = root / ("foundation25" if label == "foundation25" else
                         label.replace("_export", ""))
    return {
        "arm": label, "physical": physical,
        "train": summarize_training(train_run),
        "policy": str(actor), "checkpoint": str(checkpoint),
        "note": "Physical no-detour crossings; reward/progress are not success.",
    }


def pass_count(score: dict, height: int) -> int:
    return int(score["physical"]["by_height_mm"].get(str(height), {}).get("passed", 0))


def choose_finalists(scores: dict[str, dict], slots: int = 2) -> list[str]:
    """Rank primarily by proven step crossing while protecting flat gait.

    If ZERO models cross, select two exploratory candidates using measured
    contact/progress, marking the outcome INCONCLUSIVE; no false 'winner'.
    Flat-control remains a comparator and isn't itself a height hypothesis.
    """
    flat_ref = pass_count(scores["flat_control"], 0)
    candidates = []
    for arm, report in scores.items():
        if arm == "flat_control":
            continue
        scored = report["physical"]
        control = pass_count(report, 0)
        flat_retained = control >= max(0, flat_ref - 1)
        crossings = sum(pass_count(report, h) for h in (20, 40, 80, 120))
        weighted = (8*pass_count(report, 120) +
                    5*pass_count(report, 80) +
                    3*pass_count(report, 40) +
                    1*pass_count(report, 20))
        contact = int(scored.get("step_contact_world_frames", 0))
        progress = scored.get("mean_signed_progress_m")
        progress = float(progress) if progress is not None else -1.
        candidates.append((arm, flat_retained, crossings,
                           weighted, contact, progress))
    retained = [c for c in candidates if c[1]]
    pool = retained or candidates
    # No held-out crossing? Keep *exploratory* arms rather than wasting the
    # remaining night on a fake 0mm/10mm 'step success' celebration.
    # If nobody actually crosses, prioritise demonstrable forward progress
    # over the number of frames pushing against a riser. A policy stuck
    # grinding its feet on the wall should NOT win on contact-frame volume.
    pool.sort(key=lambda c: (c[2] > 0, c[3], c[5], c[4], c[0]), reverse=True)
    chosen = [c[0] for c in pool[:slots]]
    if len(chosen) < slots:
        raise RuntimeError("Not enough candidates for matched continuations")
    return chosen


def save_json(path: Path, payload: dict):
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=list) + "\n")
    os.replace(temp, path)


def stage_summary(root: Path, arm: str, value: dict):
    save_json(root / f"{arm}_score.json", value)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=733)
    ap.add_argument("--envs", type=int, default=1024)
    ap.add_argument("--foundation-steps", type=int, default=25_000_000)
    ap.add_argument("--screen-increment", type=int, default=10_000_000)
    ap.add_argument("--final-increment", type=int, default=15_000_000)
    ap.add_argument("--exam-worlds", type=int, default=4)
    ap.add_argument("--exam-horizon", type=int, default=350)
    args = ap.parse_args(argv)
    assert_args(args)
    doc = config(args)
    foundation_target = required_target(args.foundation_steps, args.envs)
    screen_target = required_target(args.foundation_steps + args.screen_increment, args.envs)
    final_target = required_target(args.foundation_steps + args.screen_increment +
                                   args.final_increment, args.envs)

    if args.self_test:
        assert foundation_target >= args.foundation_steps
        assert len(ARMS) == 6 and "flat_control" in ARMS
        assert branch_args("height_guided")[-1] == ".04"
        print("v5018 continuous-height campaign static contract: PASS")
        return 0
    if not (args.run or args.resume):
        print(json.dumps({
            "mode": "PLAN_ONLY", "branch": BRANCH,
            "steps": {"cold_start": foundation_target,
                      "screen_each": screen_target, "final_each": final_target,
                      "nominal_total_training_steps": (foundation_target +
                           len(ARMS) * (screen_target-foundation_target) +
                           2 * (final_target-screen_target))},
            "arms": ARMS,
            "design": "one full 25M PPO state, matched 6-way adaptive-height branches, then 2 ranked continuations",
            "height_exams_mm": {"screen": SCREEN_HEIGHTS, "final": FINAL_HEIGHTS},
            "promotion": "Real crossing first; true foot-box contact and progress exploratory tie-breaks; preserve flat gait",
            "checkpoint_rule": "All roots outside Git; never publish or overwrite protected champion",
        }, indent=2, default=list))
        print("Use --run explicitly to launch; --resume --root ACTUAL_EXPERIMENT_ROOT for recovery.")
        return 0
    sha = preflight()
    root = args.root.expanduser().resolve() if args.root is not None else (
        DEFAULT_BASE / ("v5018_height_envelope_" +
                        datetime.now().strftime("%Y%m%d_%H%M%S")))
    if root == ROOT or ROOT in root.parents:
        raise RuntimeError("Experiment root cannot be inside repository")
    if args.resume:
        if not (root / MANIFEST).is_file():
            raise RuntimeError("Missing pinned campaign manifest for resume")
        previous = json.loads((root / MANIFEST).read_text())
        if previous != {"source_commit": sha, "config": doc}:
            raise RuntimeError("Resume denied: source commit / training arm config changed")
    else:
        if root.exists():
            raise RuntimeError(f"New campaign root already exists: {root}")
        root.mkdir(parents=True)
        save_json(root / MANIFEST, {"source_commit": sha, "config": doc})
        (root / "champion_sha256_before.txt").write_text(hashlib.sha256(
            (ROOT / "showcase" / "champion.emupolicy").read_bytes()).hexdigest() + "\n")
    print(f"Source {sha}; physical envelopes, root {root}", flush=True)
    try:
        run_command(root, "physical_contract_tests", [
            sys.executable, "-m", "pytest", "-q",
            "tests/test_v5018_adaptive_height.py",
            "tests/test_v5018_terrain_training.py",
            "tests/test_v5018_overnight_campaign.py",
            "tests/test_direction.py::test_v5018_step_pilot_xml_yaw_and_declared_height",
        ])
        foundation = training(root, "foundation25", args=args, target=args.foundation_steps,
                              parent=None, arm=None)
        # The 25M foundation alone is not expected to know obstacle height.
        stage_summary(root, "foundation25", {
            "checkpoint": str(foundation), "train": summarize_training(root/"foundation25"),
            "note": "Independent from-zero 232-input gait foundation. No prior policy loaded.",
        })
        scores = {}
        for arm in ARMS:
            if preflight(source_commit=sha) != sha:
                raise RuntimeError("Study source changed during experiment")
            ck = training(root, f"screen_{arm}", args=args,
                          target=args.foundation_steps + args.screen_increment,
                          parent=foundation, arm=arm)
            result = inspect_actor(root, f"screen_{arm}", ck,
                                   SCREEN_HEIGHTS, (1701,), args)
            stage_summary(root, f"screen_{arm}", result)
            scores[arm] = result
            archive(root)

        select_path = root / "finalist_selection.json"
        if select_path.exists():
            chosen = json.loads(select_path.read_text())["selected"]
            if len(chosen) != 2 or any(c not in scores or c == "flat_control" for c in chosen):
                raise RuntimeError("Saved finalist selection not consistent with arms")
        else:
            chosen = choose_finalists(scores)
            evidence = {
                "selected": chosen,
                "all_arms_crossing_20mm": {
                    arm: pass_count(score, 20) for arm, score in scores.items()},
                "caution": "Crossing count 0 means exploratory promotion, not obstacle mastery",
            }
            save_json(select_path, evidence)
        print(f"Chosen 2 continuations: {chosen}", flush=True)
        for arm in chosen:
            ck = training(root, f"final_{arm}", args=args,
                          target=args.foundation_steps + args.screen_increment +
                                 args.final_increment,
                          parent=(root / f"screen_{arm}" / "checkpoints" / "final.pkl"),
                          arm=arm)
            result = inspect_actor(root, f"final_{arm}", ck,
                                   FINAL_HEIGHTS, (1702,1703), args)
            stage_summary(root, f"final_{arm}", result)
            archive(root)
        log_stage(root, "campaign_complete", "PASS", 0,
                  "see actual physical crossing JSON; no deployment promotion")
    finally:
        before = (root / "champion_sha256_before.txt").read_text().strip()
        after = hashlib.sha256(
            (ROOT / "showcase" / "champion.emupolicy").read_bytes()).hexdigest()
        log_stage(root, "protected_champion_hash", "PASS" if before == after else "FAIL",
                  int(before != after))
        archive(root)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        print(f"ABORTING HEIGHT STUDY: {exc}", file=sys.stderr)
        raise SystemExit(1)
