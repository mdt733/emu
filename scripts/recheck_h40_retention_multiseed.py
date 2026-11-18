#!/usr/bin/env python3
"""Read-only multi-seed physical validation of the original H40 140->180M study.

This does NOT retrain a policy, reset an optimiser, or modify an old result.
Only scripts/ is changed on the dedicated work/emu-retention-research branch.
Usage:
  python scripts/recheck_h40_retention_multiseed.py plan
  python scripts/recheck_h40_retention_multiseed.py run
  python scripts/recheck_h40_retention_multiseed.py summarize
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import zipfile

SEEDS = (1701, 1711, 1729)
POLICIES = ("original-140", "original-160", "150", "165", "175", "180")
MODES = ("stairs", "flat")
EXPECTED_FINGERPRINT = "5d5fcefce1b24f0fd39d5467ed6894ea7098ddb9baddfb66ec03c89d8c0f5bfc"
TRAINED_SOURCE = "0e6e07d45e95363867220b3cabba10b4be8cdd63"
ORIGINAL_SOURCE = "22266821bdbfc82a529ef708e5171c82c6c9f29c"
STUDY_BRANCH = "work/emu-retention-research"
REPO = Path(__file__).resolve().parents[1]
# File-path invocation ("python scripts/recheck_...py") otherwise puts only
# scripts/ on sys.path, hiding the parent scripts package and emu package.
# Keep both file-path and module invocation working from any current directory.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def paths():
    home = Path.home()
    original = Path(os.environ.get("EMU_COMMITMENT_SOURCE_RUN",
                   home / "emu-data/runs/h40-holistic-cold-s735")).expanduser()
    experiment = Path(os.environ.get("EMU_COMMITMENT_ROOT",
                     home / "emu-data/runs/h40-commitment-140to180-s735")).expanduser()
    return original, experiment, experiment / "retention_recheck"


def actor_path(policy: str, original: Path, experiment: Path) -> Path:
    if policy.startswith("original-"):
        age = policy.split("-")[-1]
        return original / "stages" / f"{age}m" / "actor.emupolicy"
    return experiment / "actors" / f"commitment-{policy}m.emupolicy"


def result_dir(policy: str, seed: int, mode: str, original: Path,
               experiment: Path, study: Path) -> Path:
    if seed == 1701:
        if policy.startswith("original-"):
            return original / "heldout" / f"{policy.split('-')[-1]}m" / mode
        return experiment / "heldout" / f"{policy}m" / mode
    return study / f"seed-{seed}" / policy / mode


def _get_exam(policy, seed, mode, original, experiment, study):
    directory = result_dir(policy, seed, mode, original, experiment, study)
    summary = directory / "matrix_summary.json"
    worlds = directory / "matrix_worlds.csv"
    if not summary.is_file() or not worlds.is_file():
        raise FileNotFoundError(f"Missing completed physical exam: {directory}")
    meta = json.loads(summary.read_text(encoding="utf-8"))["manifest"]
    actor = actor_path(policy, original, experiment)
    expected_source = ORIGINAL_SOURCE if policy.startswith("original-") else TRAINED_SOURCE
    required = {
        "schema": "h40_true_prism_heterogeneous_matrix_v5_signed_joint_diagnostics",
        "exam_source_sha256": EXPECTED_FINGERPRINT,
        "seed": seed, "worlds": 2048, "horizon": 350, "mode": mode,
        "zero_optimizer_updates": True, "first_life_only": True,
        "policy_sha256": _sha256(actor),
    }
    for key, value in required.items():
        if meta.get(key) != value:
            raise ValueError(f"{directory}: {key} mismatch, {meta.get(key)!r} != {value!r}")
    if meta.get("actor_training_source", {}).get("git_commit") != expected_source:
        raise ValueError(f"{directory}: source commit mismatch")
    if mode == "stairs" and (meta.get("levels") != 20 or
                             meta.get("trials_per_level") != 50):
        raise ValueError(f"{directory}: stair fixture changed")
    if mode == "flat" and meta.get("trials_per_level") != 2000:
        raise ValueError(f"{directory}: flat fixture changed")
    return directory, meta, worlds


def _bool(value: str) -> bool:
    if value not in ("True", "False", "true", "false"):
        raise ValueError(f"Invalid literal physical gate: {value}")
    return value.lower() == "true"


def fraction(rows, key: str) -> float:
    if not rows:
        raise ValueError(f"Empty real-world cohort: {key}")
    return 100 * sum(_bool(row[key]) for row in rows) / len(rows)


def _cohort(stairs, direction: str, lo: float, hi: float):
    kind = "step_up" if direction == "up" else "step_down"
    return [r for r in stairs if r["kind"] == kind and
            lo - 1e-6 <= abs(float(r["exact_step_m"])) <= hi + 1e-6]


def _physical_metrics(stairs, flat):
    medium = _cohort(stairs, "up", .044, .060)
    hard = _cohort(stairs, "up", .064, .080)
    down = _cohort(stairs, "down", .044, .080)
    straight = [r for r in flat if r["kind"] == "flat_control"]
    # Do not silently reduce sample sizes or accidentally compare different fixtures.
    if len(straight) != 2000 or len(stairs) != 2000:
        raise ValueError(f"Unexpected cohort sizes: stairs={len(stairs)}, flat={len(straight)}")
    return {
        "flat_control": fraction(straight, "traversed_5frame"),
        "uphill_44_60": fraction(medium, "traversed_upright_5frame"),
        "uphill_64_80": fraction(hard, "traversed_upright_5frame"),
        "downhill_44_80": fraction(down, "traversed_upright_5frame"),
        "medium_both_feet": fraction(medium, "both_feet_top_five"),
        "medium_strict_heading_lane": fraction(medium, "traversed_5frame"),
        "flat_survival": fraction(straight, "survived"),
        "flat_lateral_peak_cm": 100 * statistics.mean(
            float(r["max_abs_lateral_offset_m"]) for r in straight),
        "downhill_both_feet": fraction(down, "both_feet_top_five"),
    }


METRICS = (
    "flat_control", "uphill_44_60", "uphill_64_80", "downhill_44_80",
    "medium_both_feet", "medium_strict_heading_lane",
    "flat_survival", "flat_lateral_peak_cm", "downhill_both_feet",
)


def _read_worlds(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _preflight(original: Path, experiment: Path):
    from scripts.v5053_matrix_exam import exam_source_fingerprint
    actual = exam_source_fingerprint()
    if actual != EXPECTED_FINGERPRINT:
        raise RuntimeError(
            "Physics/exam source changed, refusing mismatched comparison. "
            f"Expected {EXPECTED_FINGERPRINT}, got {actual}")
    if not (experiment / "timeline_sources.json").is_file():
        raise RuntimeError("180M experiment timeline_sources.json missing")
    timeline = json.loads((experiment / "timeline_sources.json").read_text())
    if (timeline.get("source_git_commit") != TRAINED_SOURCE or
            timeline.get("parent_git_commit") != ORIGINAL_SOURCE or
            timeline.get("exam_source_sha256") != EXPECTED_FINGERPRINT):
        raise RuntimeError("Original experiment source identity disagrees")
    for policy in POLICIES:
        actor = actor_path(policy, original, experiment)
        if not actor.is_file():
            raise FileNotFoundError(f"Missing frozen trained actor: {actor}")
        for mode in MODES:
            _get_exam(policy, 1701, mode, original, experiment,
                      experiment / "retention_recheck")
    return True


def _run(original, experiment, study, *, plan_only: bool):
    for seed in SEEDS:
        for policy in POLICIES:
            actor = actor_path(policy, original, experiment)
            for mode in MODES:
                dst = result_dir(policy, seed, mode, original, experiment, study)
                if seed == 1701:
                    print(f"REUSE   {policy:13s} seed={seed} {mode}: {dst}", flush=True)
                    continue
                if dst.exists():
                    if not dst.is_dir() or not (dst / "matrix_worlds.csv").is_file() or \
                       not (dst / "matrix_summary.json").is_file():
                        raise RuntimeError(f"Incomplete exam; inspect without overwriting: {dst}")
                    _get_exam(policy, seed, mode, original, experiment, study)
                    print(f"REUSE   {policy:13s} seed={seed} {mode}: {dst}", flush=True)
                    continue
                print(f"{'PLAN   ' if plan_only else 'EXAM   '} "
                      f"{policy:13s} seed={seed} {mode}: {dst}", flush=True)
                if plan_only:
                    continue
                args = [
                    sys.executable, "scripts/v5053_matrix_exam.py",
                    "--policy", str(actor), "--output", str(dst),
                    "--mode", mode, "--worlds", "2048", "--horizon", "350",
                    "--seed", str(seed), "--device", "cuda",
                ]
                if mode == "stairs":
                    args += ["--levels", "20", "--trials-per-level", "50",
                             "--max-step-m", ".08"]
                else:
                    args += ["--trials-per-level", "2000"]
                env = dict(os.environ)
                env["EMU_PHYSICAL_PROFILE"] = "h40-spring-v1"
                env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
                env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
                subprocess.run(args, cwd=REPO, env=env, check=True)
                _get_exam(policy, seed, mode, original, experiment, study)


def _summarize(original, experiment, study):
    sample_rows, manifests = [], {}
    for seed in SEEDS:
        for policy in POLICIES:
            exams = {}
            for mode in MODES:
                directory, manifest, csvfile = _get_exam(
                    policy, seed, mode, original, experiment, study)
                exams[mode] = _read_worlds(csvfile)
                manifests[(seed, policy, mode)] = dict(
                    exam_source_sha256=manifest["exam_source_sha256"],
                    policy_sha256=manifest["policy_sha256"],
                    actor_sha256=manifest["actor_sha256"],
                    robot_model_sha256=manifest["robot_model_sha256"],
                    actor_training_source=manifest["actor_training_source"],
                    directory=str(directory),
                )
            values = _physical_metrics(exams["stairs"], exams["flat"])
            sample_rows.append(dict(policy=policy, seed=seed, **{
                key: round(values[key], 4) for key in METRICS}))
    grouped = {policy: [r for r in sample_rows if r["policy"] == policy]
               for policy in POLICIES}
    summary = []
    for policy in POLICIES:
        row = dict(policy=policy)
        for key in METRICS:
            values = [r[key] for r in grouped[policy]]
            row[f"{key}_mean"] = round(statistics.mean(values), 3)
            row[f"{key}_min"] = round(min(values), 3)
            row[f"{key}_max"] = round(max(values), 3)
        summary.append(row)
    study.mkdir(parents=True, exist_ok=True)
    _write_csv(study / "per_seed.csv", sample_rows)
    _write_csv(study / "aggregate.csv", summary)

    def paired(a, b, metric):
        aa = {r["seed"]: r[metric] for r in grouped[a]}
        bb = {r["seed"]: r[metric] for r in grouped[b]}
        differences = [aa[seed] - bb[seed] for seed in SEEDS]
        return (round(statistics.mean(differences), 2),
                round(min(differences), 2),
                round(max(differences), 2),
                sum(x > 0 for x in differences))

    pairs = []
    for a, b in (("150", "original-140"), ("150", "original-160"),
                 ("165", "original-140"), ("175", "original-140"),
                 ("180", "original-160")):
        for metric in ("flat_control", "uphill_44_60",
                       "uphill_64_80", "downhill_44_80"):
            mean, low, high, wins = paired(a, b, metric)
            pairs.append(dict(candidate=a, baseline=b, metric=metric,
                              mean_delta_pp=mean, min_delta_pp=low,
                              max_delta_pp=high, seed_wins=wins,
                              evaluation_seed_count=len(SEEDS)))
    _write_csv(study / "paired_differences.csv", pairs)

    lines = [
        "# EMU H40: physical skill retention recheck",
        "",
        "No PPO updates. Three paired evaluation seeds (1701, 1711, 1729);",
        "same first-life true prism physics, sensor model, 2048-world batches,",
        "350-frame horizon, robot/actor source provenance and checkpoint ages.",
        "The old 1701 results are reused as-is; other seeds are newly examined.",
        "All checkpoints share original PPO training seed 735: this recheck tests",
        "**initial-condition robustness**, NOT training-seed replicability.",
        "",
        "## Independent physical success, mean [lowest–highest seed]",
        "",
        "| Actor | Flat lane | Uphill 44–60mm | Uphill 64–80mm | Downhill 44–80mm |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in summary:
        cells = []
        for metric in ("flat_control", "uphill_44_60",
                       "uphill_64_80", "downhill_44_80"):
            cells.append(f"{r[metric+'_mean']:.1f}% "
                         f"[{r[metric+'_min']:.1f}–{r[metric+'_max']:.1f}]")
        lines.append(f"| {r['policy']} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Seed-paired differences (percentage points; positive favours candidate)",
        "",
        "| Candidate vs original | Outcome | Mean Δ | Range Δ | Positive seeds |",
        "|---|---|---:|---:|---:|",
    ]
    for r in pairs:
        lines.append(
            f"| {r['candidate']} vs {r['baseline']} | {r['metric']} | "
            f"{r['mean_delta_pp']:+.1f} pp | "
            f"{r['min_delta_pp']:+.1f} to {r['max_delta_pp']:+.1f} pp | "
            f"{r['seed_wins']}/3 |")
    lines += [
        "",
        "Interpretation: large consistent differences across seeds are stronger",
        "evidence that a frozen *policy* has that skill. One training seed cannot",
        "prove that a teaching curriculum replicates; this does not justify main.",
        "The 175M uphill policy is not a safe replacement for a flat walker",
        "unless its control and downhill retention are also adequate.",
        "",
        "The complete per-world physical data, exam manifests, and CSV summaries",
        "are included in the ZIP, including the untouched seed 1701 originals.",
    ]
    report = study / "report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = dict(
        scheme="h40_retention_recheck_v1",
        seeds=list(SEEDS), policies=list(POLICIES), modes=list(MODES),
        original_training_source=ORIGINAL_SOURCE,
        experimental_training_source=TRAINED_SOURCE,
        exam_source_sha256=EXPECTED_FINGERPRINT,
        training_seed=735, zero_training_updates=True, physics_trials=36,
        result_sources=[
            dict(policy=policy, seed=seed, mode=mode, **manifests[(seed, policy, mode)])
            for seed in SEEDS for policy in POLICIES for mode in MODES
        ],
    )
    (study / "provenance.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    bundle = study / "retention_multiseed_results.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as zipped:
        for name in ("report.md", "per_seed.csv", "aggregate.csv",
                     "paired_differences.csv", "provenance.json"):
            zipped.write(study / name, arcname=name)
        for seed in SEEDS:
            for policy in POLICIES:
                for mode in MODES:
                    directory = result_dir(policy, seed, mode, original, experiment, study)
                    for name in ("matrix_summary.json", "matrix_curve.csv",
                                 "matrix_worlds.csv"):
                        zipped.write(directory / name,
                                     arcname=f"exams/seed-{seed}/{policy}/{mode}/{name}")
    print("\n".join(lines), flush=True)
    print("\nUPLOAD RESULTS:", bundle, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run", "summarize"))
    args = parser.parse_args(argv)
    from scripts.v5053_matrix_exam import exam_source_fingerprint
    original, experiment, study = paths()
    # Baseline data and examiner identity are checked BEFORE any new GPU work.
    _preflight(original, experiment)
    if args.action == "plan":
        _run(original, experiment, study, plan_only=True)
    elif args.action == "run":
        branch = subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=REPO, text=True).strip()
        if branch != STUDY_BRANCH:
            raise RuntimeError(f"Read-only study must run on {STUDY_BRANCH}, got {branch}")
        _run(original, experiment, study, plan_only=False)
        _summarize(original, experiment, study)
    else:
        _summarize(original, experiment, study)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
