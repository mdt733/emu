#!/usr/bin/env python3
"""Qualification audit for the frozen 150->170M V2 experiment.

Compare actual per-height first-life survival AND supported traversal.
No actor promotion, no PPO update, no GPS-like world-XY station-keeping gate.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import zipfile
from pathlib import Path

from scripts.score_h40_ks150_heldout import MODEL

AGES = (150, 160, 170)
BANDS = (("easy_4_40", 4, 40), ("frontier_44_56", 44, 56),
         ("stretch_44_80", 44, 80), ("all_4_80", 4, 80))
BASE_ACTOR = "9a6e5f4bd09ff6fc258d8e32055f7b258c04ac1206f872d517d5a22821280240"
CAPABILITY_TOLERANCE_PP = {
    "command": -5., "control": -2., "transition": -3.,
    "slope": -5., "resilience": -5.,
}
EXPECTED_STEPS = {150: 150011904, 160: 160038912, 170: 170000384}


def _zip_json(path: Path, name: str) -> dict:
    with zipfile.ZipFile(path) as archive:
        return json.loads(archive.read(name))


def _totals(curves: list[dict], direction: str, lo: int, hi: int) -> dict:
    subset = [row for row in curves
              if row["kind"] == direction and
              lo <= abs(float(row["requested_step_mm"])) <= hi]
    n = sum(row["worlds"] for row in subset)
    assert n == ((hi-lo)//4+1)*50, (direction, lo, hi, n)
    return dict(
        n=n,
        crossed=sum(int(row["traversed_5frame"]) for row in subset),
        survived=sum(round(row["survival_rate"]*row["worlds"]) for row in subset),
        completed=sum(int(row["completed_destinations"]) for row in subset),
    )


def _stage(root: Path, age: int, seed: int):
    where = root/"exams"/f"{age}m"
    m=json.loads((where/"stairs"/"matrix_summary.json").read_text())
    pm=_zip_json(where/f"pulse-s{seed}.zip","manifest.json")
    ps=_zip_json(where/f"pulse-s{seed}.zip","summary.json")
    dm=_zip_json(where/f"deep-s{seed}.zip","manifest.json")
    ds=_zip_json(where/f"deep-s{seed}.zip","summary.json")
    sm=m["manifest"]
    assert sm["schema"]=="h40_true_prism_heterogeneous_matrix_v5_signed80"
    assert sm["seed"]==pm["seed"]==dm["seed"]==seed
    assert sm["robot_model_sha256"]==pm["robot_model_sha256"]==dm["robot_model_sha256"]==MODEL
    assert sm["worlds"]==pm["worlds"]==2048 and dm["worlds"]==8
    assert sm["mode"]=="stairs" and sm["assigned_worlds"]==2000
    assert sm["levels"]==20 and sm["trials_per_level"]==50
    assert sm["first_life_only"] and sm["zero_optimizer_updates"]
    assert pm["first_life_only"] and pm["training_updates"]==0
    assert dm["first_life_only"] and dm["no_optimizer_updates"]
    assert pm["actor_step"]==dm["actor_training_source"]["step"]==EXPECTED_STEPS[age]
    actor=sm["actor_sha256"]
    assert actor==pm["actor_sha256"]==dm["actor_weights_sha256"]
    if age==150:
        assert actor==BASE_ACTOR
    else:
        assert actor!=BASE_ACTOR
    curves=m["curves"]
    assert len(curves)==40
    for direction,sign in (("step_up",1),("step_down",-1)):
        rows=[x for x in curves if x["kind"]==direction]
        assert len(rows)==20
        assert sorted(round(sign*x["requested_step_mm"],4) for x in rows)==[float(i) for i in range(4,81,4)]
        assert all(x["worlds"]==50 and
                   abs(x["measured_step_mm_mean"]-x["requested_step_mm"])<=.5
                   for x in rows)
    groups={r["specialty"]:r for r in ps["headlines"]}
    assert set(groups)==set(CAPABILITY_TOLERANCE_PP)|{"step"}
    deep={r["scenario"]:r for r in ds["scenarios"]}
    assert len(deep)==9
    return dict(actor=actor, stair=sm, pulse=pm, deepmeta=dm, groups=groups,
                deep=deep,
                scores={f"{direction}_{band}":_totals(curves,direction,lo,hi)
                        for direction in ("step_up","step_down")
                        for band,lo,hi in BANDS})


def _frac(v: dict, outcome: str) -> float:
    return v[outcome]/v["n"]


def audit(reference: dict, candidate: dict) -> list[str]:
    reasons=[]
    b,c=reference["scores"],candidate["scores"]
    def compare(key, metric, min_delta):
        before=_frac(b[key],metric)
        after=_frac(c[key],metric)
        if after < before+min_delta-1e-10:
            reasons.append(f"{key} {metric}: {after:.1%} < reference {before:.1%} plus {min_delta:+.0%}")
    compare("step_up_easy_4_40","survived",-.02)
    compare("step_up_easy_4_40","crossed",-.05)
    compare("step_down_all_4_80","survived",-.05)
    compare("step_down_all_4_80","crossed",-.05)
    hard=c["step_up_frontier_44_56"]
    if _frac(hard,"survived") < .90:
        reasons.append("44–56mm uphill survival below 90%")
    if _frac(hard,"crossed") < .15:
        reasons.append("44–56mm controlled crossing below 15%")
    for key,limit in CAPABILITY_TOLERANCE_PP.items():
        delta=100*(candidate["groups"][key]["pass_fraction"]-
                   reference["groups"][key]["pass_fraction"])
        if delta < limit:
            reasons.append(f"{key} 12s regression {delta:+.1f}pp")
    stand=candidate["deep"]["stand_zero_60s"]
    if stand["zero_command_balance_passes"]<7 or stand["full_survivals"]<7:
        reasons.append("60s dynamic-balance regression")
    if (stand["mean_horizontal_speed_mps"]>.08 or
        stand["mean_yaw_mae_rad_s"]>.12):
        reasons.append("60s unwanted velocity/yaw exceeded dynamic bounds")
    for name,orig in reference["deep"].items():
        now=candidate["deep"][name]
        if now["full_survivals"]<7:
            reasons.append(f"{name}: 30s first-life survival decline")
        for key,margin in (("mean_vx_mae_mps",.03),
                           ("mean_vy_mae_mps",.03),
                           ("mean_yaw_mae_rad_s",.06)):
            max_error=max(orig[key]*1.5,orig[key]+margin)
            if now[key] > max_error:
                reasons.append(f"{name}: {key} worsened")
        if now["mean_positive_motor_shaft_work_j"]> (
                orig["mean_positive_motor_shaft_work_j"]*1.25):
            reasons.append(f"{name}: servo shaft work increase >25%")
    return reasons


def report(root: Path, seed: int) -> str:
    stages={age:_stage(root,age,seed) for age in AGES}
    # An unchanged frozen parent is re-examined under the SAME v2
    # physical/scoring revision; no cross-source examiner aliasing.
    for key in ("stair","pulse","deepmeta"):
        source_key={
            "stair":"exam_source_sha256","pulse":"source_sha256",
            "deepmeta":"source_files_sha256"}[key]
        sources=[x[key][source_key] for x in stages.values()]
        assert sources[0]==sources[1]==sources[2],f"{key}: different examiner"
    root_sha=subprocess.check_output(
        ["git","rev-parse","HEAD"],
        cwd=Path(__file__).resolve().parents[1],text=True).strip()
    for age in (160,170):
        meta=stages[age]["pulse"]["actor_source"]
        assert meta["git_commit"]==root_sha, "Source commit changed after training"
    rows=[
        "# KS150 Safe Ascent V2 — independent frozen physical audit","",
        f"Evaluation seed **{seed}** (not used to select PPO rewards).",
        "One Spring-only physics model, 2,000 signed physical stair trials per actor.",
        "World-XY parked arrival is diagnostic only. No promotion implied.","",
        "| Learner | Up 4–40mm cross | Up 4–40mm survive | Up 44–56mm cross | Up 44–56mm survive | Down 4–80mm cross | Down 4–80mm survive |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for age,stage in stages.items():
        s=stage["scores"]
        def out(name,metric):
            t=s[name];return f"{t[metric]}/{t['n']} ({t[metric]/t['n']:.1%})"
        rows.append(
            f"| {age}M | {out('step_up_easy_4_40','crossed')} | "
            f"{out('step_up_easy_4_40','survived')} | "
            f"{out('step_up_frontier_44_56','crossed')} | "
            f"{out('step_up_frontier_44_56','survived')} | "
            f"{out('step_down_all_4_80','crossed')} | "
            f"{out('step_down_all_4_80','survived')} |")
    rows+=["","## Predeclared V2 safety qualification","",
           "Reference is frozen V1 150M on same seed, same examiner.",
           "Safe 4–40mm survival within −2pp and traversal within −5pp.",
           "Hard 44–56mm survival >=90% AND crossing >=15%.",
           "Downhill 4–80mm crossings and survival within −5pp.",
           "Ordinary compact missions, dynamic balance, 30s velocities and shaft work retain.",
           "No global-position gate or automatic promotion.",""]
    for age in (160,170):
        blockers=audit(stages[150],stages[age])
        rows.append(f"**{age}M**: " + ("screen pass; human validation still required"
                 if not blockers else "NOT QUALIFIED") + ".")
        for reason in blockers: rows.append(f"- {reason}")
        rows.append("")
    rows+=["Existing 150M, 160M and 170M actors remain immutable.",
           "Separate cold-seed and hardware validation remain unproven.",""]
    return "\n".join(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--seed",type=int,default=3701)
    a=p.parse_args()
    root=a.root.expanduser().resolve()
    out=report(root,a.seed)
    (root/"v2-exam-report.md").write_text(out,encoding="utf-8")
    print(out)


if __name__=="__main__":
    main()
