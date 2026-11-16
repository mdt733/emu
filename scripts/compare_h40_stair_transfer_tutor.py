"""Paired 140M explorer-versus-160M transfer tutor physical exam report.

One source-fingerprinted frozen physical examiner; same 2000 per-world
commands, exact stair fixtures, 1701 random seed and true-robot model.
No training-side Academy flags count as independent success.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import zipfile


PARENT_EXAM=Path.home()/"emu-data/runs/h40-stair-outcome-tutor-140m-s735/heldout"
PARENT_ACTOR_SHA="0324fa18d5e624c7a38a7085050cc2abc964f30e8eb453d00d5b280e6097e039"
PARENT_MODEL_SHA="7085b1732fe6b876b27ef9b485853d0127c70f6e1c1a9b52b57af82c34de6cfd"
EXAM_SOURCE_SHA="99b690b3c4aa0758157a9854e6442f5abc5572d1dd9f53c958bd2e4299ac4953"

def read_exam(root:Path,kind:str):
    directory=root/"heldout"/kind if kind in ("stairs","flat") and (root/"heldout").exists() else root/kind
    manifest=json.loads((directory/"matrix_summary.json").read_text(encoding="utf-8"))["manifest"]
    with (directory/"matrix_worlds.csv").open(newline="",encoding="utf-8") as stream:
        rows=list(csv.DictReader(stream))
    return directory,manifest,rows

def _rate(rows,key):
    return sum(str(r[key]).lower()=="true" for r in rows)/len(rows) if rows else 0.

def summarize(rows):
    return dict(attempts=len(rows),
                falls=sum(int(r["fall_frame"])>=0 for r in rows),
                lead_top=_rate(rows,"lead_foot_top"),
                both_top=_rate(rows,"both_feet_top_five"),
                upright_cross=_rate(rows,"traversed_upright_5frame"),
                aligned_cross=_rate(rows,"traversed_5frame"),
                survival=_rate(rows,"survived"),
                fell_before_lead=sum(r["fell_before_lead_foot"].lower()=="true" for r in rows),
                fell_during_transfer=sum(r["fell_after_lead_before_both"].lower()=="true" for r in rows),
                fell_after_both=sum(r["fell_after_both_feet"].lower()=="true" for r in rows),
                mean_forward_progress_m=sum(float(r["best_forward_progress_m"]) for r in rows)/len(rows),
                mean_max_lateral_offset_m=sum(float(r["max_abs_lateral_offset_m"]) for r in rows)/len(rows))

def compare(root:Path):
    data={}
    examined={}
    for label,directory in (("parent",PARENT_EXAM),("tutor",root)):
        for kind in ("stairs","flat"):
            src,manifest,rows=read_exam(directory,kind)
            examined[(label,kind)]=(src,manifest,rows)
            expected=dict(seed=1701,worlds=2048,assigned_worlds=2000,
                          horizon=350,mode=kind,
                          robot_model_sha256=PARENT_MODEL_SHA,
                          exam_source_sha256=EXAM_SOURCE_SHA)
            if label=="parent":
                expected["policy_sha256"]=PARENT_ACTOR_SHA
            for k,v in expected.items():
                if manifest.get(k)!=v:
                    raise RuntimeError(f"{label}/{kind}: incompatible {k}: {manifest.get(k)!r} != {v!r}")
            if kind=="stairs" and (manifest.get("max_step_m")!=.08 or manifest.get("levels")!=20):
                raise RuntimeError(f"{label}: stair examiner did not test exact 4–80mm")
            if len(rows)!=2000:
                raise RuntimeError(f"{label}/{kind}: expected 2000 first-life worlds")
            if not manifest.get("zero_optimizer_updates") or not manifest.get("first_life_only"):
                raise RuntimeError("Exam updated actor or recorded second lives")
    new_actor=examined[("tutor","stairs")][1]["policy_sha256"]
    if new_actor==PARENT_ACTOR_SHA:
        raise RuntimeError("Tutor exported actor is byte-identical to original; cannot claim improvement")
    for label in ("parent","tutor"):
        a=examined[(label,"stairs")][2]
        b=examined[(label,"flat")][2]
        if set(r["kind"] for r in a)!={"step_up","step_down"} or set(r["kind"] for r in b)!={"flat_control"}:
            raise RuntimeError("Exam categories not matching exact requested fixture")
        for x,y in zip(a,b):
            if (x["world"],x["vx"],x["vy"],x["yaw"])!=(y["world"],y["vx"],y["vy"],y["yaw"]):
                raise RuntimeError("Matched control lost identical world/command")
    for kind in ("stairs","flat"):
        a=examined[("parent",kind)][2]
        b=examined[("tutor",kind)][2]
        if [(x["world"],x["kind"],x["exact_step_m"]) for x in a]!=[
              (x["world"],x["kind"],x["exact_step_m"]) for x in b]:
            raise RuntimeError("Parent and tutor were not evaluated on exactly identical stair fixtures")

    def bins(rows):
        return dict(
            flat=[r for r in rows if r["kind"]=="flat_control"],
            up_4_40=[r for r in rows if r["kind"]=="step_up" and float(r["exact_step_m"])<=.040001],
            up_44_60=[r for r in rows if r["kind"]=="step_up" and .04399<=float(r["exact_step_m"])<=.060001],
            up_64_80=[r for r in rows if r["kind"]=="step_up" and float(r["exact_step_m"])>=.06399],
            down_44_80=[r for r in rows if r["kind"]=="step_down" and float(r["exact_step_m"])<=-.04399],
        )
    result={}
    for label in ("parent","tutor"):
        examined_rows=(examined[(label,"stairs")][2]+examined[(label,"flat")][2])
        result[label]={key:summarize(rows) for key,rows in bins(examined_rows).items()}
    target=["lead_top","both_top","upright_cross","aligned_cross","survival",
            "mean_forward_progress_m","mean_max_lateral_offset_m",
            "fell_during_transfer"]
    delta={grp:{k:result["tutor"][grp][k]-result["parent"][grp][k]
                for k in target}
           for grp in result["parent"]}
    return dict(
        manifest=dict(
            physical_model_sha256=PARENT_MODEL_SHA,
            original_frozen_actor_sha256=PARENT_ACTOR_SHA,
            new_frozen_actor_sha256=new_actor,
            fixed_exam_source_sha256=EXAM_SOURCE_SHA,
            matched_seed=1701,up_and_down_heights_mm="4-80 exact",
            experiment_intervention="from exact 140M explorer: reduce commitment payment, reward true bilateral upper support and verified forward follow-through, increase retention",
            not_claimed="No causal attribution to hip spring geometry; same motors, springs, physical model, actor schema, and heldout physics; one PPO seed"),
        parent=result["parent"],tutor=result["tutor"],delta=delta)

def report(data):
    rows=["# Spring V1 — 140M explorer vs 160M transfer tutor", "",
          "**Matched, independent real-prism exams**: 2,000 signed stairs + 2,000 flat controls per actor.",
          "Same 4–80mm heights, 350 first-life frames, seed 1701, physical model and examiner SHA.",
          "Both-foot means *five-frame genuine contact*. Aligned crossing also requires lane/heading.",
          "",
          "| Task | Actor | Trials | Lead foot | Both feet | Upright cross | Aligned cross | Survive | Transfer falls | Forward progress |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k in ("flat","up_4_40","up_44_60","up_64_80","down_44_80"):
        for label in ("parent","tutor"):
            x=data[label][k]
            rows.append(
                f"| {k} | {label} | {x['attempts']} | {x['lead_top']:.0%} | "
                f"{x['both_top']:.0%} | {x['upright_cross']:.0%} | "
                f"{x['aligned_cross']:.0%} | {x['survival']:.0%} | "
                f"{x['fell_during_transfer']} | {x['mean_forward_progress_m']:.3f} m |")
    rows.extend(["", "## Prespecified outcome / retention gates", ""])
    up=data["delta"]["up_44_60"]
    new=data["tutor"]
    old=data["parent"]
    gates={
        "transfer_gain_10pp":up["both_top"]>=.10,
        "controlled_cross_gain_4pp":up["aligned_cross"]>=.04,
        "difficult_up_survival_60pct":new["up_44_60"]["survival"]>=.60,
        "difficult_up_lead_retained_42pct":new["up_44_60"]["lead_top"]>=.42,
        "easy_up_alignment_recovered_38pct":new["up_4_40"]["aligned_cross"]>=.38,
        "easy_up_survival_retained_90pct":new["up_4_40"]["survival"]>=.90,
        "downhill_alignment_retained_45pct":new["down_44_80"]["aligned_cross"]>=.45,
        "downhill_survival_retained_80pct":new["down_44_80"]["survival"]>=.80,
        "flat_alignment_retained_50pct":new["flat"]["aligned_cross"]>=.50,
        "flat_survival_retained_95pct":new["flat"]["survival"]>=.95,
    }
    for name,ok in gates.items():
        rows.append(f"- **{'PASS' if ok else 'FAIL'}** {name}")
    rows.extend([
        "", "**Interpretation:** A failure can indicate curriculum trade-offs; this is a"
        " combined 20M continuation and teacher-reweighting, not a causal test of"
        " individual rewards. A successful single-seed exam does not guarantee"
        " repeatability on physical hardware.",
        "",
        "No automatic merge to main; keep the 120M generalist and 140M explorer as references."
    ])
    return "\n".join(rows)+"\n",{
        **gates,"all_gates_pass":all(gates.values()),
        "champion_candidate":all(gates.values())
    }

def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--run",type=Path,required=True)
    args=p.parse_args()
    data=compare(args.run)
    md,criteria=report(data)
    data["acceptance"]=criteria
    (args.run/"transfer_comparison.json").write_text(json.dumps(data,indent=2)+"\n",encoding="utf-8")
    (args.run/"transfer_comparison.md").write_text(md,encoding="utf-8")
    bundle=args.run/"transfer_results_bundle.zip"
    paths=[
        args.run/"transfer_comparison.md",args.run/"transfer_comparison.json",
        args.run/"transfer_source_manifest.json",args.run/"transfer_2m_gate.json",
        args.run/"training_config.json",args.run/"run_meta.json",
        args.run/"metrics.csv",args.run/"results.md",
        args.run/"h40-stair-transfer-tutor-160m.emupolicy",
    ]
    for label,root in (("reference140m",PARENT_EXAM),("tutor160m",args.run/"heldout")):
        for kind in ("stairs","flat"):
            for name in ("matrix_summary.json","matrix_curve.csv","matrix_worlds.csv"):
                paths.append(root/kind/name)
    with zipfile.ZipFile(bundle,"w",compression=zipfile.ZIP_DEFLATED) as z:
        for path in paths:
            if not path.exists():
                if path.name in ("results.md",):
                    continue
                raise FileNotFoundError(f"Required source/result missing: {path}")
            if path.is_relative_to(args.run):
                target=path.relative_to(args.run)
            else:
                target=Path("reference")/path.relative_to(PARENT_EXAM)
            z.write(path,arcname=target.as_posix())
    print(md)
    print("COMPLETE:",bundle.resolve(),flush=True)

if __name__=="__main__":
    main()
