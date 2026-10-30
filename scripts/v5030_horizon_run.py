#!/usr/bin/env python3
"""v5030b: bounded, same-source 60->80M (optional 80->100M) A/B continuation.

This file deliberately lives on the reporting branch. Run from a CLEAN checkout
of feature/v5030-reward-lab at SOURCE_SHA; download with git show, outside Git.
It NEVER modifies the original source tree, original 60M policies or main.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

SOURCE_SHA = "a7be11b587f0912e642e100b22ec6d617f548f5b"
TRAIN_BRANCH = "feature/v5030-reward-lab"
SEED, ENVS = 747, 2048
ARMS = {"A_original25": 0.0, "B_halfsecond25": 0.5}
ORIGINAL = Path.home() / "robot_lab/emu_runs/v5030_temporal_reward_s747"
OUTPUT = Path.home() / "robot_lab/emu_runs/v5030_ab_learning_horizon_s747"

def read_json(p):
    return json.loads(p.read_text())

def git(repo, *args):
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()

def sha256(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for piece in iter(lambda: f.read(1024 * 1024), b""):
            h.update(piece)
    return h.hexdigest()

def check_repo(repo):
    if not (repo / "scripts/campaign_support.py").is_file():
        raise RuntimeError(f"Not an EMU checkout: {repo}")
    if git(repo, "rev-parse", "HEAD") != SOURCE_SHA:
        raise RuntimeError("Wrong training source SHA; do NOT pull or merge.")
    if git(repo, "branch", "--show-current") != TRAIN_BRANCH:
        raise RuntimeError("Training checkout must be feature/v5030-reward-lab")
    if git(repo, "status", "--porcelain"):
        raise RuntimeError("Training checkout is dirty. Keep it unchanged.")

def check_original(root):
    mf = read_json(root / "campaign_manifest.json")
    if any(mf.get(k) != v for k, v in [
        ("source_sha", SOURCE_SHA), ("seed", SEED), ("envs", ENVS)]):
        raise RuntimeError("Original A/B manifest provenance mismatch")
    stamp = read_json(root / "preflight_pass.json")
    expected = {"sha": SOURCE_SHA, "cpu": True,
                "full_cuda_2048": True, "shares_tested": [0.0, 0.5]}
    if stamp != expected:
        raise RuntimeError("Missing exact-source completed CPU/GPU acceptance")
    for arm, share in ARMS.items():
        p = root / arm / "checkpoints/final.pkl"
        if not p.is_file():
            raise RuntimeError(f"Missing original 60M final checkpoint: {p}")
        cfg = read_json(root / arm / "training_config.json")
        expected_cfg = {
            "seed": SEED, "envs": ENVS, "temporal_vx_reward_share": share,
            "foundation_steps": 20_000_000, "athletic_rehearsal_frac": .25,
            "growth_recipe": "main-mature", "terrain_lesson": "coherent-careful",
            "force_program": "gentle"}
        for k, v in expected_cfg.items():
            if cfg.get(k) != v:
                raise RuntimeError(f"{arm} original config {k}: {cfg.get(k)!r} != {v!r}")
        meta = read_json(root / arm / "run_meta.json")
        source = (meta.get("source_provenance") or {}).get("git_commit")
        if source != SOURCE_SHA:
            raise RuntimeError(f"{arm} original run source mismatch")

def train_command(stage, arm, share, resume, target):
    return [sys.executable, "-m", "emu.cli", "train",
        "--run-dir", str(stage / arm), "--profile", "sim2real",
        "--device", "cuda", "--seed", str(SEED), "--envs", str(ENVS),
        "--steps", str(target * 1_000_000), "--terrain-actor",
        "--terrain-step-max-m", ".08", "--unified-capabilities",
        "--foundation-steps", "20000000", "--foundation-mix", "main",
        "--teaching-schedule", "balanced", "--growth-recipe", "main-mature",
        "--terrain-lesson", "coherent-careful", "--force-program", "gentle",
        "--athletic-rehearsal-share", ".25",
        "--temporal-vx-reward-share", str(share),
        "--fixed-eval-every-steps", "20000000",
        "--resume", str(resume)]

def get_ancestor(original, output, target, arm):
    if target == 80:
        return original / arm / "checkpoints/final.pkl"
    return output / "at80M" / arm / "checkpoints/final.pkl"

def create_or_check_manifest(stage, data):
    p = stage / "campaign_manifest.json"
    if p.is_file():
        if read_json(p) != data:
            raise RuntimeError(f"Existing manifest differs: {p}")
    else:
        if stage.exists() and any(stage.iterdir()):
            raise RuntimeError(f"Unmanifested nonempty stage; refusing to overwrite: {stage}")
        stage.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")

def exam_flags():
    return {
        "speed_ladder": ["--speed-ladder", "--worlds", "16", "--horizon", "700",
            "--speed", ".15", "--ladder-speeds", ".1", ".2", ".3", ".4",
            ".5", ".6", ".8", "1.0"],
        "terrain_matrix": ["--terrain-matrix", "--worlds", "16",
            "--horizon", "350", "--speed", ".15",
            "--matrix-grades-deg", "5", "8", "12", "14.9",
            "--matrix-heights-cm", "2", "2.5", "3", "3.5"],
    }

def curve_report(original, output, stage):
    series = {}
    for arm in ARMS:
        bucket = {}
        for src in (original / arm / "fixed_eval.csv",
                    output / "at80M" / arm / "fixed_eval.csv",
                    output / "at100M" / arm / "fixed_eval.csv"):
            if not src.is_file():
                continue
            with src.open(newline="", encoding="utf-8-sig") as f:
                for r in csv.DictReader(f):
                    try:
                        step = int(float(r["step"]))
                        score = float(r["eval_broad_score"])
                    except (ValueError, TypeError, KeyError):
                        continue
                    age = round(step / 20_000_000) * 20
                    if age in (20, 40, 60, 80, 100):
                        bucket[age] = {"arm": arm, "age_M": age,
                                       "actual_step": step, "broad_score": score}
        series[arm] = bucket
    rows = [series[a][age] for age in sorted(set().union(*(set(v) for v in series.values())))
            for a in ARMS if age in series[a]]
    if rows:
        with (stage / "learning_curve.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    md = [
        "# EMU v5030b: learning horizon A/B", "",
        "Original training SHA: " + SOURCE_SHA, "",
        "| Age | A broad | B broad | B - A | A gain | B gain |",
        "|---:|---:|---:|---:|---:|---:|"]
    previous = {}
    for age in sorted(set(series["A_original25"]) | set(series["B_halfsecond25"])):
        aa, bb = (series[k].get(age, {}).get("broad_score") for k in ARMS)
        def fmt(v):
            return "—" if v is None else f"{v:+.4f}"
        md.append(f"| {age}M | {fmt(aa)} | {fmt(bb)} | "
                  f"{fmt(None if aa is None or bb is None else bb-aa)} | "
                  f"{fmt(None if aa is None or previous.get('A') is None else aa-previous['A'])} | "
                  f"{fmt(None if bb is None or previous.get('B') is None else bb-previous['B'])} |")
        if aa is not None:
            previous["A"] = aa
        if bb is not None:
            previous["B"] = bb
    md += ["", "Broad score alone is not enough. Check matched physical speed, yaw bias,",
           "survival, supported step UP/DOWN, slope tracking, and shaft J/m.", "",
           "Single-seed comparison; do not extrapolate indefinitely.", ""]
    (stage / "learning_curve.md").write_text("\n".join(md))

def run():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", type=int, choices=[80, 100], default=80)
    ap.add_argument("--run", action="store_true", help="Train A and B; default is safe preview")
    ap.add_argument("--repo", type=Path, default=Path.cwd())
    ap.add_argument("--original-root", type=Path, default=ORIGINAL)
    ap.add_argument("--output-root", type=Path, default=OUTPUT)
    args = ap.parse_args()
    repo = args.repo.expanduser().resolve()
    original = args.original_root.expanduser().resolve()
    output = args.output_root.expanduser().resolve()
    check_repo(repo)
    check_original(original)
    if output == repo or repo in output.parents or output == original or original in output.parents:
        raise RuntimeError("Output must be outside repo and original results")
    stage = output / f"at{args.target}M"
    prior = output / "at80M"
    if args.target == 100:
        for arm in ARMS:
            if not (prior / arm / "checkpoints/final.pkl").is_file():
                raise RuntimeError("80M A/B must both complete before 100M")
        if not (prior / "physical_comparison_report.md").is_file():
            raise RuntimeError("Review finished 80M exams before considering 100M")
    ancestor = {arm: get_ancestor(original, output, args.target, arm) for arm in ARMS}
    for p in ancestor.values():
        if not p.is_file():
            raise RuntimeError(f"Missing ancestor checkpoint: {p}")
    manifest = {
        "version": "v5030_ab_learning_horizon_v2",
        "source_sha": SOURCE_SHA, "branch": TRAIN_BRANCH,
        "seed": SEED, "envs": ENVS, "target_M": args.target,
        "origin": str(original),
        "ancestors": {arm: {"path": str(p), "sha256": sha256(p)}
                      for arm, p in ancestor.items()},
    }
    print(f"EMU 5030 BOUNDED A/B  {args.target-20}M -> {args.target}M")
    print("Source:", SOURCE_SHA, "  output:", stage)
    for arm, share in ARMS.items():
        print(arm, "reward share", share, "resume", ancestor[arm])
        print(" ".join(train_command(stage, arm, share, ancestor[arm], args.target)))
    if not args.run:
        print("PLAN ONLY: add --run to perform the continuation.")
        return 0
    create_or_check_manifest(stage, manifest)
    sys.path.insert(0, str(repo / "scripts"))
    import campaign_support as base
    env = dict(os.environ, PYTHONPATH=str(repo),
               XLA_PYTHON_CLIENT_PREALLOCATE="false",
               EMU_UNIFIED_CONTINUOUS="cold-growth-approved")
    cases = [dict(name=arm, recipe="main", schedule="balanced",
                  steps=args.target*1_000_000,
                  from_steps=(args.target-20)*1_000_000)
             for arm in ARMS]
    for arm, share in ARMS.items():
        final = stage / arm / "checkpoints/final.pkl"
        if final.is_file():
            meta = read_json(stage / arm / "run_meta.json")
            if (meta.get("source_provenance") or {}).get("git_commit") != SOURCE_SHA:
                raise RuntimeError(f"Existing final source mismatch: {final}")
            print("Already completed; preserving:", final, flush=True)
            continue
        own = stage / arm
        if own.exists() and any(own.iterdir()):
            if not base.has_checkpoint(own):
                raise RuntimeError(f"Partial run without valid resume checkpoint: {own}")
            resume = "auto"
        else:
            resume = str(ancestor[arm])
        base.execute(train_command(stage, arm, share, resume, args.target),
                     stage / "logs" / f"{arm}.log", env, campaign=(stage, cases))
        if not final.is_file():
            raise RuntimeError(f"No completed checkpoint for {arm}")
    for arm in ARMS:
        policy = stage / f"{arm}.emupolicy"
        if not policy.is_file():
            base.execute([sys.executable, "-m", "emu.cli", "export-policy",
                          str(stage/arm/"checkpoints/final.pkl"), str(policy)],
                         stage/"logs"/f"{arm}_export.log", env)
        for suite, flags in exam_flags().items():
            destination = stage/"physical_exams"/arm/suite
            if (destination/"unified_exam.json").is_file() and (destination/"unified_exam.md").is_file():
                print("Already examined:", arm, suite)
                continue
            if destination.exists() and any(destination.iterdir()):
                raise RuntimeError(f"Partial exam: {destination}")
            base.execute([sys.executable, "-m", "emu.cli", "capability-exam",
                          "--policy", str(policy), "--output", str(destination),
                          "--device", "cpu", *flags],
                         stage/"logs"/f"{arm}_{suite}.log", env)
    base.summary(stage, cases, SOURCE_SHA)
    curve_report(original, output, stage)
    reporter = Path(__file__).with_name("v5030_physical_report_tool.py")
    if reporter.is_file():
        import importlib.util
        spec = importlib.util.spec_from_file_location("v5030_report", reporter)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for p in module.run(stage, stage):
            print("READABLE PHYSICAL REPORT:", p)
    else:
        print("No sidecar reporter: raw exam files are intact")
    package = stage/"v5030_horizon_physical_comparison.zip"
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for filename in ["campaign_manifest.json", "learning_curve.md", "learning_curve.csv",
                         "physical_comparison_report.md", "physical_comparison_report.json",
                         "overnight_summary.md", "overnight_summary.csv"]:
            p = stage/filename
            if p.is_file():
                z.write(p, arcname=filename)
        for arm in ARMS:
            for suite in exam_flags():
                for name in ("unified_exam.md", "unified_exam.json"):
                    p = stage/"physical_exams"/arm/suite/name
                    if p.is_file():
                        z.write(p, arcname=f"{arm}/{suite}/{name}")
    print("COMPLETE:", stage, "\nCompact report:", stage/"physical_comparison_report.md",
          "\nArchive:", package)
    return 0

if __name__ == "__main__":
    raise SystemExit(run())
