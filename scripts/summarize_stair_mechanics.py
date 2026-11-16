"""Evidence-only signed hip-roll and first-life stair-failure autopsy.

Uses paired independent exact physical stairs-vs-flat exams at identical
policy/seed/world assignment. Never claims torque correlation is causation.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path


SIDES=("right","left")


def read_exam(path: Path):
    with (path/"matrix_summary.json").open(encoding="utf-8") as handle:
        summary=json.load(handle)
    with (path/"matrix_worlds.csv").open(newline="",encoding="utf-8") as handle:
        rows=list(csv.DictReader(handle))
    return summary,rows


def weighted_mean(rows, field):
    total=0.
    denom=0.
    for row in rows:
        frames=float(row["first_life_frames"])
        if frames<=0:
            continue
        total+=float(row[field])*frames
        denom+=frames
    return total/denom if denom else 0.


def group_stats(rows):
    n=len(rows)
    if n==0:
        raise ValueError("Empty mechanical trial group")
    falls=[r for r in rows if int(float(r["fall_frame"]))>=0]
    stats=dict(
        worlds=n,falls=len(falls),fall_fraction=len(falls)/n,
        survived=sum(r["survived"].lower()=="true" for r in rows),
        lead_foot=sum(r["lead_foot_top"].lower()=="true" for r in rows),
        both_feet=sum(r["both_feet_top_five"].lower()=="true" for r in rows),
        upright_cross=sum(r["traversed_upright_5frame"].lower()=="true" for r in rows),
        aligned_cross=sum(r["traversed_5frame"].lower()=="true" for r in rows),
        fell_before_lead=sum(r["fell_before_lead_foot"].lower()=="true" for r in rows),
        fell_after_lead=sum(r["fell_after_lead_before_both"].lower()=="true" for r in rows),
        fell_after_both=sum(r["fell_after_both_feet"].lower()=="true" for r in rows),
        fell_height=sum(r["fell_height"].lower()=="true" for r in rows),
        fell_tilt=sum(r["fell_tilt"].lower()=="true" for r in rows),
        fall_near95_either_hip=sum(
            r["fall_frame"]!="-1" and (
                r["hip_right_near95_at_fall"].lower()=="true" or
                r["hip_left_near95_at_fall"].lower()=="true") for r in rows),
        fall_near95_fighting_spring=sum(
            r["fall_frame"]!="-1" and (
                (r["hip_right_near95_at_fall"].lower()=="true" and
                 r["hip_right_opposing_at_fall"].lower()=="true") or
                (r["hip_left_near95_at_fall"].lower()=="true" and
                 r["hip_left_opposing_at_fall"].lower()=="true"))
            for r in rows),
        fall_lateral_offset_mean_m=(
            sum(abs(float(r["first_fall_y_offset_m"])) for r in falls)/len(falls)
            if falls else 0.),
        max_lateral_offset_mean_m=sum(float(
            r["max_abs_lateral_offset_m"]) for r in rows)/n,
    )
    for side in SIDES:
        prefix=f"hip_{side}_"
        near=prefix+"near95_sum_fraction"
        opposed=prefix+"near95_opposing_sum_fraction"
        stats[f"{side}_motor_load_mean"]=weighted_mean(
            rows,prefix+"motor_abs_fraction_sum_mean")
        stats[f"{side}_spring_abs_nm_mean"]=weighted_mean(
            rows,prefix+"spring_abs_nm_sum_mean")
        stats[f"{side}_roll_angle_abs_deg_mean"]=weighted_mean(
            rows,prefix+"roll_angle_abs_deg_sum_mean")
        stats[f"{side}_near95_frame_frac"]=weighted_mean(rows,near)
        stats[f"{side}_opposing_frame_frac"]=weighted_mean(
            rows,prefix+"opposing_sum_fraction")
        stats[f"{side}_assisting_frame_frac"]=weighted_mean(
            rows,prefix+"assisting_sum_fraction")
        near_count=sum(float(r[near])*float(r["first_life_frames"]) for r in rows)
        opposing_near_count=sum(float(r[opposed])*float(
            r["first_life_frames"]) for r in rows)
        stats[f"{side}_near95_opposition_conditional"]=(
            opposing_near_count/near_count if near_count else None)
    return stats


def assemble(stairs:Path,flat:Path):
    s_summary,s_rows=read_exam(stairs)
    f_summary,f_rows=read_exam(flat)
    sm,fm=s_summary["manifest"],f_summary["manifest"]
    if (sm["policy_sha256"]!=fm["policy_sha256"]
            or sm["robot_model_sha256"]!=fm["robot_model_sha256"]
            or sm["seed"]!=fm["seed"] or sm["horizon"]!=fm["horizon"]
            or sm["worlds"]!=fm["worlds"]
            or sm["exam_source_sha256"]!=fm["exam_source_sha256"]
            or sm["mode"]!="stairs" or fm["mode"]!="flat"):
        raise RuntimeError("Cannot compare exams with different actors/physics/source/seed/commands")
    if (len(s_rows)!=2000 or len(f_rows)!=2000
            or set(r["kind"] for r in s_rows)!={"step_up","step_down"}
            or set(r["kind"] for r in f_rows)!={"flat_control"}):
        raise RuntimeError("Need matched 2000 exact signed stairs and 2000 flat controls")
    if ([(r["world"],r["vx"],r["vy"],r["yaw"]) for r in s_rows] !=
            [(r["world"],r["vx"],r["vy"],r["yaw"]) for r in f_rows]):
        raise RuntimeError("Stair and flat controls are not matched on indexed world/command")
    groups={
        "flat_25cms":f_rows,
        "up_all":[r for r in s_rows if r["kind"]=="step_up"],
        "down_all":[r for r in s_rows if r["kind"]=="step_down"],
        "up_4_to_24mm":[r for r in s_rows if r["kind"]=="step_up"
                        and float(r["exact_step_m"])<=.024001],
        "up_28_to_40mm":[r for r in s_rows if r["kind"]=="step_up"
                         and .02799<=float(r["exact_step_m"])<=.040001],
        "up_44_to_60mm":[r for r in s_rows if r["kind"]=="step_up"
                         and .04399<=float(r["exact_step_m"])<=.060001],
        "up_64_to_80mm":[r for r in s_rows if r["kind"]=="step_up"
                         and float(r["exact_step_m"])>=.06399],
        "down_40_to_80mm":[r for r in s_rows if r["kind"]=="step_down"
                           and abs(float(r["exact_step_m"]))>=.03999],
    }
    return dict(
        manifest=dict(
            policy_sha256=sm["policy_sha256"],
            robot_model_sha256=sm["robot_model_sha256"],
            exam_source_sha256=sm["exam_source_sha256"],
            seed=sm["seed"],horizon=sm["horizon"],
            stairs_worlds=len(s_rows),flat_worlds=len(f_rows),
            caveat="Correlation is not causation; source is frozen first-life MJX physical exam"),
        groups={name:group_stats(rows) for name,rows in groups.items()},
    )


def format_report(data):
    groups=data["groups"]
    lines=[
        "# H40 Spring V1 — signed hip roll & first-life stair failure autopsy",
        "",
        "Same frozen 120M policy, source, seed, 0.25m/s command and 350-frame "
        "first-life horizon in both true-prism and zero-height seam controls.",
        "No weights/optimiser updated. Spring is **2 N·m/rad**, centred at zero.",
        "Motor opposes spring only where actual signed motor torque × "
        "passive spring torque is **negative**.",
        "Torque saturation and fall categories are **observations**, "
        "not proof that spring causes the falls.", "",
        "## Physical crossing and failure stage",
        "",
        "| Task | Attempts | Falls | Before lead foot | After lead, before both | After both | Both feet | Upright crossing | Aligned crossing |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name,stats in groups.items():
        lines.append(
            f"| {name} | {stats['worlds']} | {stats['falls']} "
            f"({100*stats['fall_fraction']:.0f}%) | "
            f"{stats['fell_before_lead']} | {stats['fell_after_lead']} | "
            f"{stats['fell_after_both']} | {stats['both_feet']} | "
            f"{stats['upright_cross']} | {stats['aligned_cross']} |")
    lines.extend(["", "## Joint load *at the actual fall* (not average load)", "",
                  "| Task | Falls | Either hip ≥95% at death | Hip ≥95% AND fighting spring at death |",
                  "|---|---:|---:|---:|"])
    for name,stats in groups.items():
        lines.append(
            f"| {name} | {stats['falls']} | "
            f"{stats['fall_near95_either_hip']} | "
            f"{stats['fall_near95_fighting_spring']} |")
    lines.extend(["", "Height- and tilt-triggered death flags can overlap. "
                  "A fall before obtaining both feet on the top is not "
                  "necessarily a lack of torque; compare the signed motor evidence.",
                  "", "## Hip roll actual load and centring-spring opposition", "",
                  "| Task | Side | Mean motor rated load | Near 95% frames | Motor fights spring | Among near-95% frames, fights spring | Mean spring torque magnitude |",
                  "|---|---|---:|---:|---:|---:|---:|"])
    for name,stats in groups.items():
        for side in SIDES:
            conditional=stats[f"{side}_near95_opposition_conditional"]
            conditional_label=(
                f"{conditional*100:.1f}%" if conditional is not None else "n/a")
            lines.append(
                f"| {name} | {side} | "
                f"{stats[f'{side}_motor_load_mean']*100:.1f}% | "
                f"{stats[f'{side}_near95_frame_frac']*100:.1f}% | "
                f"{stats[f'{side}_opposing_frame_frac']*100:.1f}% | "
                f"{conditional_label} | "
                f"{stats[f'{side}_spring_abs_nm_mean']:.3f} N·m |")
    lines.extend([
        "", "## Interpretation boundaries",
        "",
        "- A large *near-limit opposition* fraction suggests the centring spring "
        "is consuming actuator headroom in that context; it does **not** "
        "prove removing it improves balance.",
        "- Low opposition with high saturation suggests gravity, dynamics, "
        "balance control or another load rather than the spring alone.",
        "- Compare failures before top contact, during trailing-foot transfer "
        "and after two-foot support to choose the next **single** tutor.",
        "- Do not change springs, torque caps, PPO, or reward shaping based "
        "on a mixed-episode fall percentage alone.",
        "- Only one trained seed and one fixed examiner seed are measured. "
        "Do not infer statistical certainty across independent training seeds.",
        "",
    ])
    return "\n".join(lines)


def main(argv=None):
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--stairs",type=Path,required=True)
    p.add_argument("--flat",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args(argv)
    data=assemble(args.stairs,args.flat)
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/"mechanics_summary.json").write_text(
        json.dumps(data,indent=2)+"\n",encoding="utf-8")
    (args.out/"mechanics_report.md").write_text(
        format_report(data),encoding="utf-8")
    print((args.out/"mechanics_report.md").resolve(),flush=True)


if __name__=="__main__":
    main()
