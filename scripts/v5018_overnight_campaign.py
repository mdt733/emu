#!/usr/bin/env python3
"""v5018 overnight research: independent cold starts -> matched low-step forks.

RUN only after a plan review. No GPU is needed for --plan or --self-test.
* Each seed starts from RANDOM 232-input weights; no showcase file is loaded.
* A 0mm off-course box and a 5mm real collidable box inherit exactly the
  SAME complete foundation checkpoint, optimizer, normalizer and teacher.
* Never call a zero-exit examiner an obstacle-crossing success.
* 10mm is gated on observed held-out 5mm crossing. A progress-only score
  cannot silently unlock it.
* Resume is explicit and requires unchanged Git source + parameters.
Raw .pkl files stay outside Git and out of compact .tar.gz evidence archives.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
BRANCH = "feature/v5018-terrain-awareness"
ROLLOUT = 32  # verified contract in emu/training_config.py; checked in preflight
DEFAULT_BASE = Path.home() / "robot_lab" / "emu_experiments"
FILENAME = "v5018_overnight_manifest.json"
STAGES = "stages.tsv"


def required_target(steps: int, envs: int) -> int:
    block = envs * ROLLOUT
    return (steps + block - 1) // block * block


def safe_ratio(num: int, den: int) -> float:
    return num / den if den > 0 else 0.


def summarize_training(run: Path) -> dict:
    """Read actual CSV evidence; never convert absent measurements into zero."""
    csv_path = run / "metrics.csv"
    rows = list(csv.DictReader(csv_path.open(newline=""))) if csv_path.is_file() else []
    fixed_path = run / "fixed_eval.csv"
    fixed = list(csv.DictReader(fixed_path.open(newline=""))) if fixed_path.is_file() else []
    def number(row, key):
        try:
            v = float((row or {}).get(key, ""))
            return v if math.isfinite(v) else None
        except (ValueError, TypeError, AttributeError):
            return None
    def last(key):
        return number(rows[-1], key) if rows else None
    contacts = [number(r, "terrain_box_foot_contact_any") for r in rows]
    contacts = [v for v in contacts if v is not None]
    return {
        "run": str(run),
        "step": last("step"),
        "last_fall_rate": last("fall_rate"),
        "last_servo_over_rated_frac": last("servo_over_rated_frac"),
        "last_box_foot_contact_fraction": last("terrain_box_foot_contact_any"),
        "max_logged_box_foot_contact_fraction": max(contacts) if contacts else None,
        "logged_box_contact_positive_rows": sum(v > 0 for v in contacts),
        "last_core_survival": number(fixed[-1], "eval_core_survival") if fixed else None,
        "last_core_vx_mae": number(fixed[-1], "eval_core_vx_mae") if fixed else None,
        "last_fixed_eval_step": number(fixed[-1], "step") if fixed else None,
        "metrics_rows": len(rows),
        "fixed_eval_rows": len(fixed),
    }


def summarize_crossings(paths: list[Path]) -> dict:
    trials = 0
    passed = 0
    box_frames = 0
    per_course = {}
    reasons = Counter()
    progresses = []
    for path in paths:
        if not path.is_file():
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        rows = doc.get("per_world", [])
        n = int(doc.get("worlds", -1))
        if len(rows) != n or n < 1:
            raise RuntimeError(f"Malformed independently scored crossing file: {path}")
        count = sum(bool(r.get("crossing_pass")) for r in rows)
        if count != int(doc.get("crossing_success_worlds", -1)):
            raise RuntimeError(f"Contradictory crossing score: {path}")
        key = str(doc["step_height_mm"])
        item = per_course.setdefault(key, {"passed": 0, "worlds": 0, "box_contact_frames": 0})
        item["passed"] += count
        item["worlds"] += n
        item["box_contact_frames"] += int(doc.get("step_contact_world_frames", 0))
        for row in rows:
            reasons.update(row.get("rejection_reasons", []))
            if row.get("final_signed_progress_m") is not None:
                progresses.append(float(row["final_signed_progress_m"]))
        trials += n
        passed += count
        box_frames += int(doc.get("step_contact_world_frames", 0))
    return {
        "evaluated_worlds": trials, "actual_crossings": passed,
        "step_contact_world_frames": box_frames,
        "by_height_mm": per_course,
        "mean_signed_progress_m": (sum(progresses)/len(progresses) if progresses else None),
        "reject_reasons": dict(reasons),
        "NOT_a_PPO_reward_metric": True,
    }


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args],
                                   text=True, stderr=subprocess.PIPE).strip()


def preflight(*, source_commit: str | None = None) -> str:
    branch = git("branch", "--show-current")
    if branch != BRANCH:
        raise RuntimeError(f"Must run on {BRANCH}, not {branch}")
    sha = git("rev-parse", "HEAD")
    if source_commit is not None and sha != source_commit:
        raise RuntimeError(f"Source changed since campaign start: {sha} != {source_commit}")
    dirty = git("status", "--porcelain")
    if dirty:
        raise RuntimeError(f"Dirty source; keep experiments outside Git:\n{dirty}")
    if not (ROOT / ".venv" / "bin" / "python").is_file():
        raise RuntimeError("Missing v5018 checkout .venv; see docs/INSTALL.md")
    return sha


def log_stage(root: Path, stage: str, status: str, exit_code: int,
              details: str = "") -> None:
    path = root / STAGES
    if not path.exists():
        path.write_text("stage\tstatus\texit_code\tdetail\n")
    with path.open("a") as f:
        f.write(f"{stage}\t{status}\t{exit_code}\t{details.replace(chr(9),' ').replace(chr(10),' ')}\n")


def run_command(root: Path, stage: str, cmd: list[str], *, allow_failure=False) -> bool:
    """Record the actual process exit; never infer scientific success."""
    print("\n==========", stage, "==========", flush=True)
    print("command:", " ".join(map(str, cmd)), flush=True)
    log_path = root / f"{stage}.log"
    env = os.environ.copy()
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    with log_path.open("w") as f:
        process = subprocess.Popen(cmd, cwd=ROOT, env=env, text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            f.write(line)
            print(line, end="", flush=True)
        rc = process.wait()
    log_stage(root, stage, "PASS" if rc == 0 else "FAIL", rc)
    if rc and not allow_failure:
        raise RuntimeError(f"{stage} failed (code {rc}), inspect {log_path}")
    return rc == 0


def validated_checkpoint(run_dir: Path, expected_steps: int, envs: int) -> Path:
    ck = run_dir / "checkpoints" / "final.pkl"
    if not ck.is_file() or ck.stat().st_size < 1000:
        raise RuntimeError(f"Missing saved nontrivial final checkpoint: {ck}")
    data = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    cfg = json.loads((run_dir / "training_config.json").read_text(encoding="utf-8"))
    if cfg["steps"] != expected_steps or cfg["envs"] != envs:
        raise RuntimeError("Checkpoint root has wrong target steps/envs")
    if data.get("source_provenance", {}).get("git_commit") != git("rev-parse", "HEAD"):
        raise RuntimeError("Run/checkpoint source provenance mismatch")
    if cfg.get("terrain_box_height_mm", -1) >= 0:
        xml = run_dir / "v5018_training_course.xml"
        if not xml.is_file() or hashlib.sha256(xml.read_bytes()).hexdigest() != data.get("terrain_course_sha256"):
            raise RuntimeError("Training course XML/hash mismatch")
    return ck


def train_one(root: Path, stage: str, *, seed: int, envs: int, steps: int,
              profile: str, pitch: float, box_mm: int | None = None,
              parent: Path | None = None) -> Path:
    run_dir = root / stage
    if run_dir.exists() and not (run_dir / "training_config.json").is_file():
        raise RuntimeError(f"Ambiguous existing training directory: {run_dir}")
    checkpoint_files = list((run_dir / "checkpoints").glob("*.pkl"))
    # Same-run failure recovery is ALWAYS explicit --resume auto. The first
    # invocation forks only from the chosen, validated full parent checkpoint.
    resume = "auto" if checkpoint_files else (str(parent) if parent else "")
    cmd = [
        "emu", "train", "--run-dir", str(run_dir), "--steps", str(steps),
        "--envs", str(envs), "--seed", str(seed), "--device", "cuda",
        "--profile", profile, "--sensor-actor", "--pitch-deg", str(pitch),
    ]
    if box_mm is not None:
        cmd += ["--terrain-box-height-mm", str(box_mm)]
    if resume:
        cmd += ["--resume", resume]
    run_command(root, stage, cmd)
    return validated_checkpoint(run_dir, steps, envs)


def export_one(root: Path, stage: str, checkpoint: Path) -> Path:
    output = root / f"{stage}.emupolicy"
    # Export even on --resume so exported bytes correspond to the
    # validated current checkpoint, never a stale success marker.
    run_command(root, f"export_{stage}", ["emu", "export-policy", str(checkpoint), str(output)])
    if not output.is_file() or output.stat().st_size < 100:
        raise RuntimeError(f"Missing exported candidate actor: {output}")
    return output


def exam(root: Path, stage: str, policy: Path, *, height: int, seed: int,
         worlds: int, horizon: int) -> Path:
    output = root / f"{stage}.json"
    cmd = [
        sys.executable, "scripts/v5018_champion_step_pilot.py",
        "--policy", str(policy), "--height-mm", str(height),
        "--heading-kp", ".75", "--speed", ".10",
        "--seed", str(seed), "--worlds", str(worlds),
        "--horizon", str(horizon), "--output", str(output),
    ]
    run_command(root, stage, cmd)
    if not output.is_file():
        raise RuntimeError(f"Exam returned successfully without result: {output}")
    return output


def save_result(root: Path, seed: int, stage: str, result: dict) -> None:
    dest = root / f"seed{seed}_{stage}_summary.json"
    dest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def archive(root: Path) -> None:
    """Compact, copy-safe evidence only; never store huge optimizer pickle."""
    dest = root.with_suffix(".tar.gz")
    tmp = dest.with_name(dest.name + ".partial")
    if tmp.exists():
        tmp.unlink()
    with tarfile.open(tmp, mode="w:gz", dereference=False) as tar:
        for path in root.rglob("*"):
            if not path.is_file() or path.is_symlink():
                continue
            if path.suffix in (".pkl", ".npz") or path.name.endswith(".emupolicy"):
                continue
            if path.stat().st_size > 20_000_000:
                continue
            tar.add(path, arcname=str(root.name / path.relative_to(root)) if False else
                    str(Path(root.name) / path.relative_to(root)))
    os.replace(tmp, dest)
    print(f"\nEvidence archive (no giant checkpoints/policies): {dest}", flush=True)


def request_config(args: argparse.Namespace) -> dict:
    return {
        "envs": args.envs, "seeds": args.seeds,
        "foundation_steps": args.foundation_steps,
        "screen_more_steps": args.screen_more_steps,
        "extend_more_steps": args.extend_more_steps,
        "pitch_deg": args.pitch_deg,
        "exam_worlds": args.exam_worlds,
        "exam_horizon": args.exam_horizon,
        "study_revision": 1,
    }


def validate_config(config: dict) -> None:
    if not (128 <= config["envs"] <= 2048):
        raise ValueError("GPU research supports 128..2048 worlds")
    if not (5_000_000 <= config["foundation_steps"] <= 30_000_000):
        raise ValueError("Foundation must be 5..30M steps")
    if not (1_000_000 <= config["screen_more_steps"] <= 5_000_000):
        raise ValueError("Screen increment must be 1..5M steps")
    if not (2_000_000 <= config["extend_more_steps"] <= 12_000_000):
        raise ValueError("Extension increment must be 2..12M steps")
    if not config["seeds"] or len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("Require unique, nonempty independent fresh seeds")
    if any(n < 0 for n in config["seeds"]) or len(config["seeds"]) > 3:
        raise ValueError("Use 1..3 nonnegative seeds")
    if not 0 < config["pitch_deg"] <= 30:
        raise ValueError("Pitch command must be within (0, 30]")
    if not (2 <= config["exam_worlds"] <= 8 and 80 <= config["exam_horizon"] <= 700):
        raise ValueError("Exam must use 2..8 worlds and 80..700 frames")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group()
    group.add_argument("--run", action="store_true", help="actually execute CUDA and create a new root")
    group.add_argument("--resume", action="store_true", help="resume an EXISTING --root, same options and source")
    group.add_argument("--self-test", action="store_true", help="stdlib logic-only test, no GPU")
    p.add_argument("--root", type=Path, help="new root for --run, existing one for --resume")
    p.add_argument("--envs", type=int, default=1024)
    p.add_argument("--seeds", type=int, nargs="+", default=[733, 734])
    p.add_argument("--foundation-steps", type=int, default=20_000_000)
    p.add_argument("--screen-more-steps", type=int, default=2_100_000)
    p.add_argument("--extend-more-steps", type=int, default=6_000_000)
    p.add_argument("--pitch-deg", type=float, default=8.)
    p.add_argument("--exam-worlds", type=int, default=4)
    p.add_argument("--exam-horizon", type=int, default=350)
    args = p.parse_args(argv)
    config = request_config(args)
    validate_config(config)
    if args.self_test:
        assert required_target(20_000_000, 1024) == 20_021_248
        assert safe_ratio(0, 0) == 0.
        assert safe_ratio(2, 8) == .25
        print("Overnight stdlib contract: PASS")
        return 0
    if args.resume and args.root is None:
        p.error("--resume requires an explicit --root from the earlier campaign")
    if args.root is not None:
        args.root = args.root.expanduser().resolve()
        if ROOT == args.root or ROOT in args.root.parents:
            p.error("Experiment root must be OUTSIDE source checkout")
    if not (args.run or args.resume):
        print(json.dumps({"mode": "PLAN_ONLY", "config": config,
                          "per_seed": [
                              "Cold-start 232-input foundation: 20M steps, 8deg",
                              "From identical foundation PPO state: blind 0mm vs physical 5mm, +2.1M steps each",
                              "Held-out 0/5mm no-detour crossing exams; preserve 0 vs 5 physics/control",
                              "Extend BOTH matched 0/5mm arms +6M only if real 5mm contact and acceptable flat retention",
                              "Allow 10mm fork only if held-out 5mm crossing actually passed",
                          ],
                          "boundaries": {
                              "foundation": required_target(args.foundation_steps,args.envs),
                              "screen": required_target(args.foundation_steps + args.screen_more_steps,args.envs),
                              "extension": required_target(args.foundation_steps + args.screen_more_steps + args.extend_more_steps,args.envs),
                          }}, indent=2))
        print("\nUse --run [--root NEW_DIRECTORY] to execute. --resume --root EXISTING_DIRECTORY after interruption.")
        return 0

    sha = preflight()
    root = args.root or DEFAULT_BASE / f"v5018_overnight_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if args.resume:
        manifest_path = root / FILENAME
        if not manifest_path.is_file():
            raise RuntimeError(f"Cannot resume without original manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("source_commit") != sha or manifest.get("config") != config:
            raise RuntimeError("Resume source/config mismatch; never mix lineages")
    else:
        if root.exists():
            raise RuntimeError(f"New --run root already exists: {root}. Use --resume, or new root.")
        root.mkdir(parents=True)
        manifest = {
            "schema": 1, "source_commit": sha, "source_branch": BRANCH,
            "created_at": datetime.now().astimezone().isoformat(),
            "config": config, "evidence_note": "PPO training, not independently certified walking",
        }
        (root / FILENAME).write_text(json.dumps(manifest, indent=2) + "\n")
        (root / "champion_sha256_before.txt").write_text(
            hashlib.sha256((ROOT / "showcase" / "champion.emupolicy").read_bytes()).hexdigest() + "\n")

    print("Overnight root:", root, "\nGit:", sha, flush=True)
    envs = args.envs
    foundation_step = required_target(args.foundation_steps, envs)
    screen_step = required_target(args.foundation_steps + args.screen_more_steps, envs)
    extended_step = required_target(args.foundation_steps + args.screen_more_steps + args.extend_more_steps, envs)
    try:
        run_command(root, "preflight_contracts", [
            sys.executable, "-m", "pytest", "-q",
            "tests/test_v5018_terrain_training.py",
            "tests/test_v5018_cold_start_contract.py",
            "tests/test_direction.py::test_v5018_step_pilot_xml_yaw_and_declared_height",
        ])
        for seed in args.seeds:
            preflight(source_commit=sha)
            label = f"s{seed}"
            print(f"\n========= Independent RANDOM-INITIALIZED seed {seed} =========", flush=True)
            foundation = train_one(root, f"{label}_foundation", seed=seed, envs=envs,
                                   steps=args.foundation_steps, profile="foundation",
                                   pitch=args.pitch_deg)
            foundation_summary = summarize_training(root / f"{label}_foundation")
            save_result(root, seed, "foundation", foundation_summary)
            # No fabricated thresholds for actual crossing, which only the
            # held-out examiner can score. A completely failed foundation
            # should not burn GPU on obstacle forks.
            core = foundation_summary.get("last_core_survival")
            if core is None or core < .35:
                log_stage(root, f"{label}_obstacle_forks", "SKIP", 0,
                          f"foundation core survival {core!r} < .35 (or unmeasured)")
                print("No reliable flat foundation; skipping obstacle branches for this seed.")
                archive(root)
                continue

            children = {}
            for mm in (0, 5):
                arm = f"{label}_screen_{mm}mm"
                ck = train_one(root, arm, seed=seed, envs=envs,
                               steps=args.foundation_steps + args.screen_more_steps,
                               profile="sim2real", pitch=args.pitch_deg,
                               box_mm=mm, parent=foundation)
                policy = export_one(root, arm, ck)
                reports = [exam(root, f"{arm}_exam_{h}mm_seed1701", policy,
                                height=h, seed=1701, worlds=args.exam_worlds,
                                horizon=args.exam_horizon) for h in (0, 5)]
                score = summarize_crossings(reports)
                metrics = summarize_training(root / arm)
                children[mm] = {"checkpoint": ck, "policy": policy,
                                "score": score, "metrics": metrics}
                save_result(root, seed, f"screen_{mm}mm", {"metrics": metrics, "crossing": score})

            baseline_flat = children[0]["score"]["by_height_mm"].get("0", {}).get("passed", 0)
            step_flat = children[5]["score"]["by_height_mm"].get("0", {}).get("passed", 0)
            step_contact = children[5]["metrics"].get("max_logged_box_foot_contact_fraction")
            step_survival = children[5]["metrics"].get("last_core_survival")
            step_overload = children[5]["metrics"].get("last_servo_over_rated_frac")
            extended = (
                step_contact is not None and step_contact > 0
                and step_survival is not None and step_survival >= .35
                and step_overload is not None and step_overload < .85
                and step_flat >= max(0, baseline_flat - 1)
            )
            reason = (f"contact={step_contact}, core_survival={step_survival}, "
                      f"overload={step_overload}, step_flat={step_flat}, baseline_flat={baseline_flat}")
            if not extended:
                log_stage(root, f"{label}_extensions", "SKIP", 0, reason)
                print("Extensions SKIPPED, insufficient physical contact/retention evidence:", reason)
                archive(root)
                continue
            # Matched learning budget: do NOT extend 5mm alone and declare a
            # winner against an earlier 0mm checkpoint.
            final = {}
            for mm in (0, 5):
                arm = f"{label}_extend_{mm}mm"
                ck = train_one(root, arm, seed=seed, envs=envs,
                               steps=args.foundation_steps + args.screen_more_steps + args.extend_more_steps,
                               profile="sim2real", pitch=args.pitch_deg, box_mm=mm,
                               parent=children[mm]["checkpoint"])
                policy = export_one(root, arm, ck)
                reports = [
                    exam(root, f"{arm}_exam_{h}mm_seed{test_seed}",
                         policy, height=h, seed=test_seed,
                         worlds=args.exam_worlds, horizon=args.exam_horizon)
                    for h in (0, 5, 10) for test_seed in (1702, 1703)
                ]
                final[mm] = {"checkpoint": ck, "policy": policy,
                             "metrics": summarize_training(root / arm),
                             "crossing": summarize_crossings(reports)}
                save_result(root, seed, f"extend_{mm}mm", final[mm] | {
                    "checkpoint": str(ck), "policy": str(policy)})
            # Only advance obstacle frontier if a supported far-side crossing
            # was actually observed on 5mm in held-out worlds.
            five = final[5]["crossing"]["by_height_mm"].get("5", {})
            baseline_five = final[0]["crossing"]["by_height_mm"].get("5", {})
            if five.get("passed", 0) <= 0 or five.get("passed", 0) < baseline_five.get("passed", 0):
                log_stage(root, f"{label}_10mm", "SKIP", 0,
                          "No positive 5mm crossing evidence beyond matched baseline")
                archive(root)
                continue

            ten_arm = f"{label}_10mm_frontier"
            ck = train_one(root, ten_arm, seed=seed, envs=envs,
                           steps=args.foundation_steps + args.screen_more_steps +
                                 args.extend_more_steps + args.screen_more_steps,
                           profile="sim2real", pitch=args.pitch_deg,
                           box_mm=10, parent=final[5]["checkpoint"])
            policy = export_one(root, ten_arm, ck)
            scores = [
                exam(root, f"{ten_arm}_exam_{h}mm_seed{test_seed}", policy,
                     height=h, seed=test_seed,
                     worlds=args.exam_worlds, horizon=args.exam_horizon)
                for h in (0, 5, 10) for test_seed in (1702, 1703)
            ]
            save_result(root, seed, "10mm", {
                "metrics": summarize_training(root / ten_arm),
                "crossing": summarize_crossings(scores),
                "qualification": "10mm exposure only, not certified stepping",
            })
            archive(root)
        log_stage(root, "campaign_complete", "PASS", 0,
                  "See summaries for physical crossing outcomes and skipped frontier gates")
    finally:
        try:
            original = (root / "champion_sha256_before.txt").read_text().strip()
            now = hashlib.sha256((ROOT / "showcase" / "champion.emupolicy").read_bytes()).hexdigest()
            if original != now:
                log_stage(root, "protected_policy_integrity", "FAIL", 1, "champion hash changed")
                print("WARNING: protected champion changed hash; investigate!", file=sys.stderr)
            else:
                log_stage(root, "protected_policy_integrity", "PASS", 0)
            archive(root)
        except Exception as exc:
            print(f"Evidence archive error: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"CAMPAIGN STOPPED: {exc}", file=sys.stderr)
        raise SystemExit(1)
