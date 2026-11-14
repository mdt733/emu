#!/usr/bin/env python3
"""Read-only, decision-grade 50mm tutor A/B report; no trainer/winner claims."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

COUNTS = (
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
ANCHOR=60_000_000


def _float(value):
    try:
        v=float(value)
        if v==v and abs(v)!=float("inf"):
            return v
    except (TypeError,ValueError):
        pass
    return 0.


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
    payload=dict(
        schema="h40_stair_tutor_ab_v1",parent_steps=ANCHOR,
        source_note="same 60M frozen learner; physical H40 MJX; 50mm UP kept",
        warning=("Selected rollout event counts are not complete episodes; "
                 "last alpha/height are teacher state. Do not sum cumulative "
                 "PPO audit metrics; do not interpret retired clearance/goal counts."),
        arms=cases)
    root.mkdir(parents=True,exist_ok=True)
    (root/"tutor_ab_summary.json").write_text(
        json.dumps(payload,indent=2,sort_keys=True)+"\n")
    lines=[
        "# H40 50mm stair tutor — initial matched A/B","",
        "All counts below are selected rollout diagnostics, NOT independent "
        "frozen-policy 50mm exam certifications. Two arms share an immutable "
        "60M parent; main was not changed.","",
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
    (root/"tutor_ab_summary.md").write_text("\n".join(lines),encoding="utf-8")
    print("\n".join(lines))
    return payload


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,required=True)
    args=ap.parse_args()
    report(args.root.expanduser())


if __name__=="__main__":
    main()
