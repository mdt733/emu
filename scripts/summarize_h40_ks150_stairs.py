#!/usr/bin/env python3
"""Immutable stage archiver and source-matched H40-KS150 stair scorecard.

This NEVER trains, merges, chooses a new champion or modifies an old actor.
Accept only real first-life MJX exams on identical scenario/source fixtures.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import zipfile

MODEL_SHA = "21b5856b7a248e5d5b988e2735255d57ec79bae1a7cab0bcb72f481bfc36f470"
BASE_ACTOR_SHA = "cc1f0ab126126815f6cea907dbc3a8a2b3faa6caff8ccbe2f05133d7c090771d"
PARENT_SOURCE = "5235ea687d04948b3dce88880e644eac5bb74cee"
AGENUM = (130, 140, 150, 160)
PULSE_GROUPS = ("command", "control", "transition", "slope", "step", "resilience")
# Predeclared tolerances: policy research choices, NOT statistical confidence.
# Ordinary skills are non-negotiable; a +3pp step gain alone is insufficient.
LOSS_LIMIT_PP = {"command": 5., "control": 2., "transition": 3.,
                 "slope": 5., "resilience": 5.}
MIN_STEP_GAIN_PP = 3.
MIN_UP_GAIN_PP = 5.
MIN_DOWN_RETENTION_PP = -5.


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new_json(path: Path, data: dict) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to replace immutable stage manifest {path}")
    temporary = path.with_name(path.name + ".partial")
    if temporary.exists():
        raise FileExistsError(f"Interrupted manifest creation: {temporary}")
    try:
        temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def freeze_stage(root: Path, age: int, parent: Path, parent_sha: str) -> None:
    """Copy entire exact PPO temporal state before the next same-source age."""
    from emu.checkpoint_io import load_checkpoint, update_boundary_at_or_after
    from emu.training_config import TrainingConfig, training_sensor_contract
    from emu.contracts import ROBOT_XML

    assert age in AGENUM and len(parent_sha) == 64
    assert sha256_file(parent) == parent_sha, "Frozen 120M parent changed"
    stage = root / "stages" / f"{age}m"
    source = root / "checkpoints" / "final.pkl"
    target = stage / "final.pkl"
    marker = stage / "stage_manifest.json"
    source_git = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1], text=True).strip()
    stage.mkdir(parents=True, exist_ok=True)
    if marker.exists():
        manifest = json.loads(marker.read_text())
        assert manifest["age_m"] == age
        assert manifest["source_git_commit"] == source_git
        assert manifest["physical_sha256"] == MODEL_SHA
        assert manifest["parent_checkpoint_sha256"] == parent_sha
        assert target.is_file() and sha256_file(target) == manifest["checkpoint_sha256"]
        print(f"IMMUTABLE STAGE VERIFIED: {age}M")
        return

    # An archived earlier age must remain verifiable even when the active
    # one-run training config has since been extended to a later target.
    cfg = TrainingConfig.load(root / "training_config.json")
    assert cfg.steps == age * 1000000, "Stage target must be completed first"
    assert cfg.physical_profile == "h40-spring-only" and cfg.seed == 735
    assert cfg.terrain_step_max_m == cfg.academy_height_cap_m == .055
    assert cfg.science_step_goal_weight == .12
    assert cfg.science_bilateral_support_bonus == 6.
    assert cfg.science_followthrough_bonus == 7.
    assert hashlib.sha256(ROBOT_XML.read_bytes()).hexdigest() == MODEL_SHA
    if not source.is_file() or target.exists():
        raise RuntimeError("Missing completed checkpoint or unexpected partial stage archive")
    payload = load_checkpoint(source, expected_sensor_contract=training_sensor_contract(cfg))
    step = int(payload["meta"]["step"])
    assert step == update_boundary_at_or_after(age * 1000000, cfg.envs), (age, step)
    assert (payload["meta"].get("source_provenance") or {}).get("git_commit") == source_git

    temp = stage / ".final.pkl.partial"
    if temp.exists():
        raise FileExistsError(f"Interrupted stage copy: {temp}")
    try:
        shutil.copy2(source, temp)
        if sha256_file(temp) != sha256_file(source):
            raise RuntimeError("Stage copy checksum mismatch")
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)
    data = dict(
        schema="emu_ks150_balanced_stairs_stage_v1", age_m=age,
        actual_step=step, source_git_commit=source_git,
        physical_sha256=MODEL_SHA, checkpoint_sha256=sha256_file(target),
        parent_checkpoint_sha256=parent_sha, parent_source_git=PARENT_SOURCE,
        config_sha256=sha256_file(root / "training_config.json"),
        actor_export="actor.emupolicy", pulse="pulse-s1701.zip",
        signed_stair_matrix="stairs/matrix_summary.json",
        note="Complete PPO optimizer, normalizer, governor, field and health state; not just actor weights",
    )
    write_new_json(marker, data)
    print(f"FROZEN FULL PPO CHECKPOINT: {age}M / {step:,} steps / {target}")


def read_pulse(path: Path):
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        score = json.loads(archive.read("summary.json"))
    assert manifest["schema"] == "emu_h40_compact_specialist_pulse_v1"
    assert manifest["robot_model_sha256"] == MODEL_SHA
    assert manifest["physical_profile"] == "h40-spring-only"
    assert manifest["seed"] == 1701 and manifest["worlds"] == 2048
    assert manifest["horizon_frames"] == 600
    assert manifest["training_updates"] == 0 and manifest["first_life_only"]
    groups = {r["specialty"]: r for r in score["headlines"]}
    assert set(groups) == set(PULSE_GROUPS)
    curves = {(r["specialty"], r["kind"], r["level"]): r
              for r in score["curves"]}
    return manifest, groups, curves


def read_stairs(path: Path):
    data = json.loads(path.read_text())
    m = data["manifest"]
    assert m["schema"] == "h40_true_prism_heterogeneous_matrix_v4"
    assert m["robot_model_sha256"] == MODEL_SHA
    assert m["seed"] == 1701 and m["worlds"] == 2048
    assert m["mode"] == "stairs" and m["horizon"] == 350
    assert m["zero_optimizer_updates"] and m["first_life_only"]
    assert m["assigned_worlds"] == 2000
    return m, data["curves"]


def group_stairs(curves: list[dict], kind: str, lo: float, hi: float) -> dict:
    rows = [r for r in curves if r["kind"] == kind
            and lo - 0.01 <= abs(float(r["requested_step_mm"])) <= hi + 0.01]
    n = sum(int(r["worlds"]) for r in rows)
    if n == 0:
        raise RuntimeError(f"No stair trial group: {kind} {lo}-{hi}mm")
    return dict(trials=n,
                crossing=sum(int(r["traversed_5frame"]) for r in rows),
                survived=sum(round(float(r["survival_rate"]) * int(r["worlds"])) for r in rows))


def score(root: Path):
    base = root / "reference"
    if not (base / "pulse-s1701.zip").is_file() or not (
            base / "stairs" / "matrix_summary.json").is_file():
        print("Missing matched baseline exams; no selection yet")
        return
    bm, bg, bc = read_pulse(base / "pulse-s1701.zip")
    assert bm["actor_sha256"] == BASE_ACTOR_SHA
    bsm, bsc = read_stairs(base / "stairs" / "matrix_summary.json")
    assert bsm["actor_sha256"] == BASE_ACTOR_SHA
    rows = []
    for age in AGENUM:
        stage = root / "stages" / f"{age}m"
        if not (stage / "pulse-s1701.zip").is_file() or not (
                stage / "stairs" / "matrix_summary.json").is_file():
            continue
        marker = json.loads((stage / "stage_manifest.json").read_text())
        assert marker["age_m"] == age and marker["physical_sha256"] == MODEL_SHA
        m, groups, curves = read_pulse(stage / "pulse-s1701.zip")
        sm, stair_curves = read_stairs(stage / "stairs" / "matrix_summary.json")
        assert m["actor_sha256"] == sm["actor_sha256"]
        assert m["actor_step"] == marker["actual_step"]
        assert m["source_sha256"] == bm["source_sha256"], "Changed compact examiner; invalid comparison"
        assert sm["exam_source_sha256"] == bsm["exam_source_sha256"], "Changed stair examiner; invalid comparison"
        assert m["actor_sha256"] != BASE_ACTOR_SHA
        good = True
        issues = []
        row = dict(age_m=age, actor_sha256=m["actor_sha256"])
        for name in PULSE_GROUPS:
            current = groups[name]["pass_fraction"]
            original = bg[name]["pass_fraction"]
            delta_pp = 100. * (current - original)
            row[f"{name}_passed"] = groups[name]["passed"]
            row[f"{name}_delta_pp"] = round(delta_pp, 2)
            if name in LOSS_LIMIT_PP and delta_pp < -LOSS_LIMIT_PP[name]:
                good = False
                issues.append(f"{name} {delta_pp:+.1f}pp")
        if row["step_delta_pp"] < MIN_STEP_GAIN_PP:
            good = False
            issues.append("insufficient compact step improvement")
        for tag, kind, lo, hi in (
            ("up_28_56", "step_up", 28, 56),
            ("up_44_56", "step_up", 44, 56),
            ("down_4_56", "step_down", 4, 56),
            ("down_44_80", "step_down", 44, 80),
        ):
            baseline = group_stairs(bsc, kind, lo, hi)
            candidate = group_stairs(stair_curves, kind, lo, hi)
            assert candidate["trials"] == baseline["trials"], "Different stair fixture allocation"
            delta = 100. * (candidate["crossing"] - baseline["crossing"]) / baseline["trials"]
            row[f"{tag}_baseline"] = baseline["crossing"]
            row[f"{tag}_candidate"] = candidate["crossing"]
            row[f"{tag}_delta_pp"] = round(delta, 2)
            if tag == "up_28_56" and delta < MIN_UP_GAIN_PP:
                good = False
                issues.append(f"uphill controlled crossing {delta:+.1f}pp")
            if tag.startswith("down_") and delta < MIN_DOWN_RETENTION_PP:
                good = False
                issues.append(f"{tag} crossing {delta:+.1f}pp")
        # Specific baseline regressions must not be averaged away.
        for kind, level, label in (
            ("reverse", 2, "reverse_025"),
            ("step_down", 6, "down_5cm"),
        ):
            specialty = "command" if kind == "reverse" else "step"
            before = bc[(specialty, kind, level)]
            after = curves[(specialty, kind, level)]
            row[f"{label}_survived"] = after["survived"]
            row[f"{label}_passed"] = after["passed"]
            if (kind == "reverse" and after["passed"] < before["passed"]-2):
                good = False
                issues.append("reverse 0.25m/s regression")
            if (kind == "step_down" and after["survived"] < before["survived"]):
                good = False
                issues.append("5cm downhill survival regression")
        row["screen_eligible"] = good
        row["issues"] = "; ".join(issues)
        rows.append(row)
    if not rows:
        print("No completed 130-160M stage exams yet")
        return
    fields = list(rows[0])
    out = root / "selection.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    eligible = [r for r in rows if r["screen_eligible"]]
    selected = max(eligible, key=lambda r: (
        r["up_28_56_delta_pp"], r["step_delta_pp"], -r["age_m"])) if eligible else None
    md = ["# H40-KS150 balanced stair school — qualification screen", "",
          "Both exams: same 1701 seed, 2048 worlds, first-life, one physics model.",
          "This is *not* an automatic promotion or physical hardware validation.",
          "A full deep flat-ground exam and independent seed are still required.", "",
          "| Stage | Compact step gain | Up 28–56mm crossing gain | Down 44–80mm crossing change | Eligible? |",
          "|---|---:|---:|---:|---|"]
    for r in rows:
        md.append(f"| {r['age_m']}M | {r['step_delta_pp']:+.1f}pp "
                  f"| {r['up_28_56_delta_pp']:+.1f}pp "
                  f"| {r['down_44_80_delta_pp']:+.1f}pp "
                  f"| {'screen pass' if r['screen_eligible'] else 'NO: '+r['issues']} |")
    md.extend(["", "Selection criterion: compact mission retention across all six",
               "specialties plus ≥3pp compact steps and ≥5pp controlled",
               "28–56mm uphill crossings; downhill 4–56 and 44–80mm",
               "controlled crossing each within −5pp; reverse and 5cm",
               "downhill checks. These are exploratory thresholds.",
               "No candidate may claim proven safe 8cm climbing just from",
               "first-foot top contact.", ""])
    md.append(
        f"**Provisional exam leader: {selected['age_m']}M** (NOT promoted)."
        if selected else "**No stage currently passes the preregistered screen.**")
    (root / "selection.md").write_text("\n".join(md) + "\n")
    print(f"Updated: {out}, {root/'selection.md'}")
    if selected:
        print(f"Provisional exam leader: {selected['age_m']}M; HUMAN REVIEW required")
    else:
        print("No eligible stage; preserve 120M generalist")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--archive-stage", type=int, choices=AGENUM)
    p.add_argument("--parent", type=Path)
    p.add_argument("--parent-sha256", default="")
    args = p.parse_args()
    root = args.root.expanduser().resolve()
    if args.archive_stage is not None:
        if args.parent is None or not args.parent_sha256:
            p.error("--archive-stage requires --parent and --parent-sha256")
        freeze_stage(root, args.archive_stage,
                     args.parent.expanduser().resolve(), args.parent_sha256)
        return
    score(root)


if __name__ == "__main__":
    main()
