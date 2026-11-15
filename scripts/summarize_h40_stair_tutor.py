#!/usr/bin/env python3
"""Read-only, decision-grade 50mm tutor A/B report; no trainer/winner claims."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

COUNTS = (
    "academy_anchor_attempts_up",
    "academy_near_attempts_up",
    "academy_near_first_lip_up",
    "academy_near_successes_up",
    "academy_near_attempts_down",
    "academy_near_first_lip_down",
    "academy_near_successes_down",
    "academy_up_crossings",
    "academy_down_crossings",
    "academy_two_lip_continuations",
    "academy_two_lip_crossed_episodes",
    "academy_short_climb_successes",
    "tutor_pre_root_plants",
    "tutor_escrow_releases",
    "nursery_reward_sum",
    "academy_episodes",
)
FINAL=("academy_active_fraction","academy_height_up_m","academy_height_down_m",
       "outcome_stair_alpha","fall_rate","academy_actual_vx_mps")
ANCHOR=80_000_000


def _float(value):
    try:
        v=float(value)
        if v==v and abs(v)!=float("inf"):
            return v
    except (TypeError,ValueError):
        pass
    return 0.


def physical_exams(root: Path):
    """Untrained two-lip trajectory metrics, per frozen seed and programme."""
    result={}
    names=("up","mixed","down")
    for arm in ("legacy","escrow"):
        for program in names:
            for seed in (1701,2701):
                path=root/"exams"/arm/f"{program}_s{seed}"/"physical_worlds.csv"
                if not path.is_file():
                    continue
                with path.open(newline="",encoding="utf-8-sig") as stream:
                    rows=list(csv.DictReader(stream))
                if len(rows)!=128:
                    raise RuntimeError(f"Incomplete physical exam {path}: {len(rows)} worlds")
                def yes(row,key):
                    return row.get(key,"").strip().lower() in ("true","1")
                count=lambda key:sum(yes(r,key) for r in rows)
                result[f"{arm}/{program}_s{seed}"]=dict(
                    worlds=len(rows),
                    first_pre_root_plant=count("first_upper_contact_before_root"),
                    first_both_individual=count("first_sequential_both"),
                    second_both_individual=count("second_sequential_both"),
                    second_50cm_recovered=count("half_m_after_second"),
                    examiner_postsecond_healthy=count("course_healthy_commanded"),
                    strict_both_supported_course=sum(
                        yes(r,"first_sequential_both") and
                        yes(r,"second_sequential_both") and
                        yes(r,"course_healthy_commanded") for r in rows),
                    falls=count("fell"),
                    mean_max_abs_torque_nm=sum(_float(r.get("max_abs_servo_nm"))
                                                for r in rows)/len(rows))
    return result


def paired_matrix(root: Path):
    pairs={}
    for arm in ("legacy","escrow"):
        path=root/"exams"/arm/"paired_comparison_s1701"/"matrix_paired_curves.csv"
        if not path.is_file():
            continue
        with path.open(newline="",encoding="utf-8-sig") as stream:
            rows=list(csv.DictReader(stream))
        for row in rows:
            if row["kind"] not in ("step_up","step_down"):
                raise RuntimeError(f"Unexpected comparison kind {row['kind']}")
        pairs[arm]=rows
    return pairs


def report(root: Path):
    cases={}
    for arm in ("legacy","escrow"):
        path=root/arm/"metrics.csv"
        if not path.is_file():
            cases[arm]={"status":"missing (not a zero-success run)"}
            continue
        with path.open(newline="",encoding="utf-8-sig") as stream:
            rows=[r for r in csv.DictReader(stream)
                  if _float(r.get("step",0))>=ANCHOR]
        if not rows:
            cases[arm]={"status":"no post-parent rollout logs (not a failure)"}
            continue
        counts={key:sum(_float(row.get(key,0)) for row in rows)
                for key in COUNTS}
        last=rows[-1]
        cases[arm]=dict(
            status="logged (selected training rollouts, NOT independent PPO seeds)",
            n_logged_rollouts=len(rows),first_step=_float(rows[0].get("step")),
            final_step=_float(last.get("step")),counts=counts,
            last={key:_float(last.get(key)) for key in FINAL})
    exams=physical_exams(root)
    paired=paired_matrix(root)
    payload=dict(
        schema="h40_stair_tutor_ab_v1",parent_steps=ANCHOR,
        source_note="same frozen 80M bridge learned from historical 60M D foundation; H40 MJX; 50mm UP kept",
        warning=("Selected rollout event counts are not complete episodes; "
                 "last alpha/height are teacher state. Do not sum cumulative "
                 "PPO audit metrics; do not interpret retired clearance/goal counts."),
        arms=cases,physical_exams=exams,paired_flat_matrix=paired)
    root.mkdir(parents=True,exist_ok=True)
    (root/"tutor_ab_summary.json").write_text(
        json.dumps(payload,indent=2,sort_keys=True)+"\n")
    lines=[
        "# H40 50mm stair tutor — initial matched A/B","",
        "All counts below are selected rollout diagnostics, NOT independent "
        "frozen-policy 50mm exam certifications. Two arms share an immutable "
        "80M bridge; main was not changed.","",
        "| Signal | Legacy | Escrow |","|---|---:|---:|"]
    metrics=(
        ("Logged rollouts","n_logged_rollouts"),
        ("Near UP attempts","academy_near_attempts_up"),
        ("First-lip UP supported","academy_near_first_lip_up"),
        ("Near UP complete","academy_near_successes_up"),
        ("Near DOWN attempts","academy_near_attempts_down"),
        ("Near DOWN complete","academy_near_successes_down"),
        ("Supported UP crossings","academy_up_crossings"),
        ("Full two-lip continuations","academy_two_lip_continuations"),
        ("Pre-root confirmed upper-foot contacts","tutor_pre_root_plants"),
        ("Deferred chassis reward released","tutor_escrow_releases"),
        ("Academy episodes","academy_episodes"),
    )
    for label,key in metrics:
        def value(arm):
            case=cases[arm]
            if key=="n_logged_rollouts":
                return f"{case[key]:,}" if key in case else "—"
            if "counts" not in case:
                return "—"
            return f"{case['counts'][key]:,.0f}"
        lines.append(f"| {label} | {value('legacy')} | {value('escrow')} |")
    lines += [
        "","## Pre-registered decision","",
        "1. STOP if CUDA smoke fails or physical/motor limits change.",
        "2. INVALID COMPARISON if either arm has negligible Academy UP exposure "
        "(roughly <100 scored near-UP attempts) or source/parent hashes differ.",
        "3. PROMISING only if escrow improves verified upper-foot support "
        "before chassis crossing, and independent 50mm UP recovery without "
        "sacrificing flat walking.",
        "4. NO LONG RUN based solely on nursery reward, first-lip chassis "
        "crossings, old goal_completions or teacher graduation.",
        "5. Recheck both actors on frozen 50mm UP/mixed/DOWN physical exams "
        "using two evaluation seeds; continue to 110M only if warranted.",""]
    if exams:
        lines += ["", "## Frozen two-lip physics (128 worlds/cell)", "",
                  "| Arm/program | First foot planted before root | First both feet | "
                  "Second both feet | Strict two-lip recovery | Falls |",
                  "|---|---:|---:|---:|---:|---:|"]
        for name in sorted(exams):
            e=exams[name]
            lines.append(f"| {name} | {e['first_pre_root_plant']}/128 | "
                         f"{e['first_both_individual']}/128 | "
                         f"{e['second_both_individual']}/128 | "
                         f"{e['strict_both_supported_course']}/128 | "
                         f"{e['falls']}/128 |")
        lines += ["", "The strict outcome additionally requires the original "
                  "examiner's post-second recovery gate. Double-foot "
                  "simultaneous standing is NOT mandatory.", ""]
    if paired:
        lines += ["", "## Exact 50mm versus matched flat control", "",
                  "| Actor | Direction | Flat traversals | 50mm traversals | Difference |",
                  "|---|---|---:|---:|---:|"]
        for arm in ("legacy","escrow"):
            for row in paired.get(arm,[]):
                lines.append(
                    f"| {arm} | {row['kind']} | "
                    f"{row['traversed_5frame_flat']}/{row['pairs']} | "
                    f"{row['traversed_5frame_stair']}/{row['pairs']} | "
                    f"{row['traversed_5frame_difference_pct']} pp |")
        lines += ["", "Same world index / seed / initial command; compare "
                  "direction-specific actual prism traversal rather than "
                  "raw root crossings. A zero-height success alone is "
                  "NOT a stair skill.", ""]
    (root/"tutor_ab_summary.md").write_text("\n".join(lines),encoding="utf-8")
    print("\n".join(lines))
    return payload


def gate_useful_legacy_exposure(payload):
    """Fail BEFORE spending any escrow-arm GPU time if no useful 50mm births."""
    trial=payload["arms"].get("legacy",{})
    counters=trial.get("counts",{})
    age=trial.get("final_step",0)
    anchored=counters.get("academy_anchor_attempts_up",0)
    approached=counters.get("academy_near_attempts_up",0)
    # Console metrics are logged every ten 65,536-step PPO updates; a
    # completed 100M run may legitimately have its last CSV row at 99.6M.
    if age < 99_000_000 or anchored < 20 or approached < 5:
        raise RuntimeError(
            "NO USEFUL TUTOR COMPARISON: legacy arm at "
            f"{age:,.0f} steps, {anchored:,.0f} sampled 50mm UP "
            f"birth outcomes, {approached:,.0f} near-lip attempts. "
            "Do not spend 20M more on escrow until Academy exposure "
            "and walking graduation are diagnosed.")
    print(f"STAIR EXPOSURE GATE PASS: {anchored:.0f} 50mm UP births, "
          f"{approached:.0f} near-lip cases in selected rollouts.")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,required=True)
    ap.add_argument("--require-up-exposure",action="store_true",
                    help="abort before escrow-arm training if legacy never sees 50mm stair encounters")
    args=ap.parse_args()
    payload=report(args.root.expanduser())
    if args.require_up_exposure:
        gate_useful_legacy_exposure(payload)


if __name__=="__main__":
    main()
