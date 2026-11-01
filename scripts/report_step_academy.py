#!/usr/bin/env python3
"""Read-only v5037 Step Academy report + optional independent physical exams.

This tool is deliberately kept OUTSIDE the completed training source commit.
Run it from the source-pinned v5037 checkout via:
  git show origin/feature/v5038-step-academy-report-and-exams:scripts/report_step_academy.py | python - 
For physical exams append --exams (expensive, CPU); already complete exams are skipped.
It writes only to <run-root>/analysis; no training, promotion or checkpoint changes.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import zipfile
from pathlib import Path
from statistics import mean

DEFAULT = Path.home() / "robot_lab/emu_runs/v5037_step_academy_stage_a"
PARENT_STEP = 280_035_328
TARGET = 290_000_000
HEIGHTS = (1.0, 2.0, 2.5, 3.0, 3.5, 4.0)
READABLE = ("control", "dense step", "rapid repeats", "easy nursery",
            "100% specialist", "verified upper support",
            "approach rhythm", "square-edge control")

def records(path: Path) -> list[dict[str, str]]:
    if not path.is_file(): return []
    with path.open(newline="",encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))

def number(value):
    try:
        out=float(value)
        return out if math.isfinite(out) else None
    except (ValueError, TypeError):
        return None

def first_number(*values):
    return next((x for value in values if (x:=number(value)) is not None),None)

def last(rows,field):
    return next((value for row in reversed(rows)
                 if (value:=number(row.get(field))) is not None),None)

def total(rows,field):
    return sum(number(row.get(field)) or 0. for row in rows)

def fraction(n,d):
    return n/d if d else None

def fmt(value,digits=3):
    return "—" if value is None else f"{value:.{digits}f}"

def percent(value):
    return "—" if value is None else f"{value*100:.1f}%"

def wilson(k,n):
    if not n: return None
    z=1.959963984540054
    p=k/n
    denom=1+z*z/n
    centre=(p+z*z/(2*n))/denom
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denom
    return [max(0.,centre-half),min(1.,centre+half)]

def physical_matrix(analysis:Path,arm_name:str):
    per_height={}
    for h in HEIGHTS:
        per_height[f"{h:g}"]={"up_ok":0,"up_total":0,"down_ok":0,"down_total":0,
                              "up_mission":0,"down_mission":0}
    exams=[]
    for path in sorted((analysis/"physical"/arm_name).glob("seed_*/unified_exam.json")):
        doc=json.loads(path.read_text(encoding="utf-8"))
        protocol=doc.get("protocol",{})
        if (not protocol.get("terrain_matrix") or
            tuple(float(x) for x in protocol.get("matrix_heights_cm",[])) != HEIGHTS):
            continue
        exams.append(str(path))
        for block in doc.get("results",[]):
            case=block.get("case",{})
            if case.get("kind")!="step":continue
            key=f"{float(case.get('step_m',0.))*100:g}"
            if key not in per_height:continue
            for world in block.get("worlds",[]):
                direction=world.get("direction")
                if direction=="step-up":
                    per_height[key]["up_total"]+=1
                    per_height[key]["up_ok"]+=int(world.get("crossing_pass") is True)
                    per_height[key]["up_mission"]+=int(world.get("mission_pass") is True)
                elif direction=="step-down":
                    per_height[key]["down_total"]+=1
                    per_height[key]["down_ok"]+=int(world.get("crossing_pass") is True)
                    per_height[key]["down_mission"]+=int(world.get("mission_pass") is True)
    return exams,per_height

def gather(root:Path):
    manifest=root/"campaign_manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing study manifest: {manifest}")
    doc=json.loads(manifest.read_text(encoding="utf-8"))
    if (doc.get("study")!="v5037-stage-a-single-lip" or
        doc.get("parent_step")!=PARENT_STEP or len(doc.get("arms",[]))!=8):
        raise RuntimeError("Not the complete eight-arm v5037 stage A manifest")
    analysis=root/"analysis"
    runs=[]
    for a in doc["arms"]:
        idx=int(a["index"]); name=a["name"]
        folder=root/name
        rows=records(folder/"metrics.csv")
        evals=records(folder/"fixed_eval.csv")
        final=folder/"checkpoints/final.pkl"
        meta_path=folder/"run_meta.json"
        meta=json.loads(meta_path.read_text()) if meta_path.is_file() else {}
        source=(meta.get("source_provenance") or {}).get("git_commit")
        actual=last(rows,"step")
        exam=evals[-1] if evals else {}
        exam_step=number(exam.get("step"))
        valid=(final.is_file() and actual is not None and
               actual>=doc["steps_absolute"] and source==doc["source_commit"])
        physics,heights=physical_matrix(analysis,name)
        per_direction={}
        for direc in ("up","down"):
            at=total(rows,f"terrain_near_attempts_{direc}")
            ok=total(rows,f"terrain_near_successes_{direc}")
            easy_at=total(rows,f"terrain_easy_attempts_{direc}")
            easy_ok=total(rows,f"terrain_easy_successes_{direc}")
            per_direction[direc]={
                "near_attempts":round(at),"near_successes":round(ok),
                "near_rate":fraction(ok,at),
                "near_ci95":wilson(ok,at) if at else None,
                "easy_attempts":round(easy_at),"easy_successes":round(easy_ok),
                "easy_rate":fraction(easy_ok,easy_at)}
        runs.append(dict(
            id=idx,name=name,description=READABLE[idx],complete=valid,
            final_checkpoint_present=final.is_file(),
            latest_step=actual,eval_step=exam_step,source=source,
            frozen_score=number(exam.get("eval_broad_score")),
            frozen_core=number(exam.get("eval_core_survival")),
            frozen_resilience=number(exam.get("eval_resilience75_survival")),
            frozen_vx_error=number(exam.get("eval_core_vx_mae")),
            fall_rate=last(rows,"fall_rate"),
            vx_cmd=last(rows,"cmd_vx_mean"),
            vx_actual=last(rows,"vx_mean"),
            shaft_jm=last(rows,"control_shaft_j_per_m"),
            servo_over_rated=last(rows,"servo_over_rated_frac"),
            step_focus=last(rows,"focus_alloc_step"),
            frontier_step_m=last(rows,"terrain_height_m"),
            uphill=per_direction["up"],downhill=per_direction["down"],
            physical_exams=physics,physical_steps=heights,
            treatment=a.get("changes",{})))
    return doc,runs

def physical_exams(root:Path,manifest:dict,arm_ids:list[int],seeds:list[int],
                   worlds:int,horizon:int,speed:float,device:str):
    analysis=root/"analysis"
    for a in manifest["arms"]:
        idx=int(a["index"])
        if idx not in arm_ids:continue
        name=a["name"]
        checkpoint=root/name/"checkpoints/final.pkl"
        if not checkpoint.is_file():
            raise RuntimeError(f"Cannot physically examine unfinished arm: {name}")
        exported=analysis/"exports"/f"{name}.emupolicy"
        if not exported.is_file():
            exported.parent.mkdir(parents=True,exist_ok=True)
            print(f"EXPORT {name}",flush=True)
            subprocess.run([sys.executable,"-m","emu.cli","export-policy",
                            str(checkpoint),str(exported)],check=True)
        for seed in seeds:
            dest=analysis/"physical"/name/f"seed_{seed}"
            done=dest/"unified_exam.json"
            if done.is_file():
                payload=json.loads(done.read_text(encoding="utf-8"))
                protocol=payload.get("protocol",{})
                expected=(protocol.get("seed")==seed and
                          protocol.get("worlds")==worlds and
                          protocol.get("horizon")==horizon and
                          abs(float(protocol.get("speed",-1))-speed)<1e-9 and
                          tuple(float(x) for x in protocol.get("matrix_heights_cm",[]))==HEIGHTS and
                          tuple(float(x) for x in protocol.get("matrix_grades_deg",[]))==(3.,))
                if not expected:raise RuntimeError(
                    f"Existing physical exam has DIFFERENT protocol: {done}")
                print(f"EXAM SKIP verified protocol {name} seed {seed}",flush=True)
                continue
            if dest.exists() and any(dest.iterdir()):
                raise RuntimeError(f"Ambiguous interrupted exam {dest}; do not overwrite it")
            print(f"\nPHYSICAL EXAM {name} seed {seed}, heights {HEIGHTS}cm",
                  flush=True)
            subprocess.run([
                sys.executable,"-m","emu.cli","capability-exam",
                "--policy",str(exported),"--output",str(dest),
                "--seed",str(seed),"--device",device,"--worlds",str(worlds),
                "--horizon",str(horizon),"--speed",str(speed),
                "--terrain-matrix","--matrix-grades-deg","3",
                "--matrix-heights-cm",*(f"{x:g}" for x in HEIGHTS)],check=True)

def render(root:Path,manifest:dict,runs:list[dict]):
    analysis=root/"analysis"
    analysis.mkdir(parents=True,exist_ok=True)
    control=runs[0]
    rows=[]
    for r in runs:
        fixed=r["frozen_score"]
        delta=(fixed-control["frozen_score"] if fixed is not None and
               control["frozen_score"] is not None else None)
        at=r["uphill"]["near_attempts"]; up=r["uphill"]["near_successes"]
        dn=r["downhill"]["near_successes"]; dat=r["downhill"]["near_attempts"]
        physical3=r["physical_steps"]["3"]
        up3=fraction(physical3["up_ok"],physical3["up_total"])
        rows.append(dict(
            arm=r["id"],name=r["name"],status="complete" if r["complete"] else "INCOMPLETE",
            frozen_score=fmt(fixed,5),delta_vs_control=fmt(delta,5),
            frozen_core_survival=percent(r["frozen_core"]),
            frozen_resilience75=percent(r["frozen_resilience"]),
            near_up_successes=up,near_up_attempts=at,near_up_rate=percent(fraction(up,at)),
            near_down_successes=dn,near_down_attempts=dat,
            near_down_rate=percent(fraction(dn,dat)),
            measured_3cm_up_successes=physical3["up_ok"],
            measured_3cm_up_attempts=physical3["up_total"],
            measured_3cm_up_rate=percent(up3),
            step_focus_allocation=percent(r["step_focus"]),
            frontier_cm=fmt(r["frontier_step_m"]*100 if r["frontier_step_m"] is not None else None,2),
            fall_rate=percent(r["fall_rate"]),
            control_shaft_j_per_m=fmt(r["shaft_jm"],1)))
    fields=list(rows[0]) if rows else []
    with (analysis/"step_academy_report.csv").open("w",newline="",encoding="utf-8") as fh:
        w=csv.DictWriter(fh,fieldnames=fields);w.writeheader();w.writerows(rows)
    raw=dict(schema="v5037-stage-a-postrun-report-v1",
             study=manifest.get("study"),source_commit=manifest["source_commit"],
             parent_sha256=manifest["parent_sha256"],
             target=manifest["steps_absolute"],arms=runs,
             interpretation="On-policy near-frontier counts vary by curriculum; compare fixed frozen physical heights before selecting a climbing recipe.")
    (analysis/"step_academy_report.json").write_text(
        json.dumps(raw,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    md=["# EMU v5037 — Step Academy Stage A: completed-campaign report","",
        f"**Source SHA:** {manifest['source_commit']}",
        f"**Common full-state parent:** {manifest['parent_step']:,} steps; SHA256 {manifest['parent_sha256']}",
        f"**Target:** {manifest['steps_absolute']:,} world steps.",
        f"**Validated source-aligned final artifacts:** {sum(r['complete'] for r in runs)}/8 (presence + metadata/metrics; not checkpoint-byte revalidation).","",
        "## Matched final frozen evaluation and training observations","",
        "| Arm | Final broad score | Δ vs control | Core | Near-frontier uphill | Downhill | 3cm uphill held-out |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        md.append(f"| {r['arm']:02d} {r['name'][7:]} | {r['frozen_score']} "
                  f"| {r['delta_vs_control']} | {r['frozen_core_survival']} "
                  f"| {r['near_up_successes']}/{r['near_up_attempts']} ({r['near_up_rate']}) "
                  f"| {r['near_down_successes']}/{r['near_down_attempts']} ({r['near_down_rate']}) "
                  f"| {r['measured_3cm_up_successes']}/{r['measured_3cm_up_attempts']} "
                  f"({r['measured_3cm_up_rate']}) |")
    md+=["","## Physics: matched heights and direction","",
         "Held-out physical testing uses the same independent random seeds and exact two-slab step heights for every arm. "
         "A count of zero attempts means **not examined**, not zero success.",""]
    for h in HEIGHTS:
        key=f"{h:g}"
        md+=[f"### {h:g} cm", "",
             "| Arm | Supported uphill crossing | Downhill crossing |",
             "|---|---:|---:|"]
        for r in runs:
            s=r["physical_steps"][key]
            up=percent(fraction(s["up_ok"],s["up_total"]))
            down=percent(fraction(s["down_ok"],s["down_total"]))
            md.append(f"| {r['id']:02d} {r['description']} | "
                      f"{s['up_ok']}/{s['up_total']} ({up}) | "
                      f"{s['down_ok']}/{s['down_total']} ({down}) |")
        md.append("")
    md+=["## Interpretation constraints","",
         "- **Near-frontier training attempts are not height-matched.** A 2cm nursery and a 3cm frontier do not constitute the same test.",
         "- Fixed broad score is matched but not a dedicated 3cm climbing measure.",
         "- Multiple evaluation seeds help, but PPO training seed remains common; this screen does not establish statistical replication.",
         "- Actual dynamics/actuation, sensor ablations and consecutive multi-lip geometry are NOT evaluated by the Stage A training itself.",
         "- Do not select a policy based solely on the highest displayed step curriculum frontier.",
         "- The 100% STEP arm should be rejected as a generalist replacement if core speed, direction, recovery or resilience regress.",
         "- This report is analysis-only: no policy promotion, checkpoint mutation or training.",""]
    (analysis/"step_academy_report.md").write_text("\n".join(md)+"\n",encoding="utf-8")
    bundle=analysis/"step_academy_results.zip"
    with zipfile.ZipFile(bundle,"w",compression=zipfile.ZIP_DEFLATED,
                         compresslevel=7) as zipout:
        for p in sorted((analysis).rglob("*")):
            if p.is_file() and p!=bundle and p.suffix!=".emupolicy":
                zipout.write(p,p.relative_to(root))
        for p in (root/"campaign_manifest.json",):
            if p.is_file():zipout.write(p,p.relative_to(root))
        for r in runs:
            folder=root/r["name"]
            for name in ("metrics.csv","fixed_eval.csv","training_config.json",
                         "run_meta.json","results.md","envelope_summary.json"):
                p=folder/name
                if p.is_file():zipout.write(p,p.relative_to(root))
    print("\nREPORT:",analysis/"step_academy_report.md",flush=True)
    print("CSV:",analysis/"step_academy_report.csv",flush=True)
    print("SHAREABLE ZIP:",bundle,flush=True)
    return bundle

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=DEFAULT)
    p.add_argument("--exams",action="store_true",
                   help="run all requested frozen CPU physical matrices, then write report")
    p.add_argument("--exam-arms",type=int,nargs="+",default=list(range(8)))
    p.add_argument("--seeds",type=int,nargs="+",default=[1701,1702])
    p.add_argument("--worlds",type=int,default=8)
    p.add_argument("--horizon",type=int,default=350)
    p.add_argument("--speed",type=float,default=.15)
    p.add_argument("--device",choices=("cpu","cuda","gpu","auto"),default="cpu")
    args=p.parse_args(argv)
    root=args.root.expanduser().resolve()
    manifest,runs=gather(root)
    if args.exams:
        if len(set(args.exam_arms))!=len(args.exam_arms) or any(x not in range(8) for x in args.exam_arms):
            p.error("--exam-arms must be unique integers 0..7")
        if len(set(args.seeds))!=len(args.seeds) or any(seed<0 for seed in args.seeds):
            p.error("--seeds must be distinct nonnegative integers")
        if any(not r["complete"] for r in runs if r["id"] in args.exam_arms):
            raise RuntimeError("Selected arms are incomplete; refusing physical-exam promotion")
        physical_exams(root,manifest,args.exam_arms,args.seeds,
                       args.worlds,args.horizon,args.speed,args.device)
        manifest,runs=gather(root)
    render(root,manifest,runs)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
