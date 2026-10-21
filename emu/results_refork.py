"""Read-only v5020 winner-refork results.md + JSON/ZIP evidence pack.

Never imports JAX, reads policy pickle, runs evaluation, or rewrites
scoreboard/selection/checkpoint. Safe even if a probe has been interrupted.
"""
from __future__ import annotations

import csv
import json
import math
import zipfile
from pathlib import Path

STAGES=("walk40","walk60","walk80","terrain100")
ARMS=("general","turns","recovery","efficiency","steps")
RUN_FILES=("metrics.csv","fixed_eval.csv","training_config.json","run_meta.json",
           "stage_spec.json","performance.json","capability_field_summary.json")
STAGE_FILES=("scoreboard.json","scoreboard.md","selection.json")
SCALARS=("step","reward_mean","fall_rate","ep_len_mean","servo_shaft_abs_power_w",
         "servo_over_rated_frac","terrain_edge_crossing",
         "frontier_terrain_height_m","terrain_edge_done_count")


def _json(path:Path):
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path:Path):
    if not path.is_file():
        return []
    with path.open(newline="",encoding="utf-8-sig") as fd:
        return list(csv.DictReader(fd))


def _number(row,key):
    try:
        x=float((row or {}).get(key,""))
        return x if math.isfinite(x) else None
    except (ValueError,TypeError):
        return None


def collect(root:Path)->dict:
    root=Path(root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"refork root does not exist: {root}")
    context=_json(root/"refork_context.json")
    if context is None:
        raise ValueError("no refork_context.json; cannot attribute evidence to a pinned study")
    report={"schema":"v5020-refork-results-1","root":str(root),"context":context,
            "stages":{}}
    for stage in STAGES:
        group=root/stage
        if not group.is_dir():
            continue
        board=_json(group/"scoreboard.json")
        chosen=_json(group/"selection.json")
        arms={}
        for arm in ARMS:
            folder=group/arm
            if not folder.exists():
                continue
            probe=_json(folder/"probe_health.json")
            phases={}
            for segment in ("probe","finish"):
                phase=folder/segment
                if not phase.is_dir():
                    continue
                metrics=_rows(phase/"metrics.csv")
                exam=_rows(phase/"fixed_eval.csv")
                spec=_json(phase/"stage_spec.json") or {}
                training=_json(phase/"training_config.json") or {}
                provenance=(_json(phase/"run_meta.json") or {}).get("source_provenance") or {}
                late=metrics[-1] if metrics else {}
                early=metrics[0] if metrics else {}
                phases[segment]={"path":str(phase),
                                 "logged_rows":len(metrics),
                                 "metrics_start":{key:_number(early,key) for key in SCALARS},
                                 "metrics_end":{key:_number(late,key) for key in SCALARS},
                                 "fixed_eval_step":_number(exam[-1] if exam else {},"step"),
                                 "training_target":training.get("steps"),
                                 "source_git_commit":provenance.get("git_commit"),
                                 "parent_sha256":(spec.get("parent") or {}).get("sha256"),
                                 "has_final_checkpoint":(phase/"checkpoints"/"final.pkl").is_file()}
            arms[arm]={"probe_health":probe,"phases":phases}
        report["stages"][stage]={"scoreboard":board,"selection":chosen,"arms":arms}
    return report


def _fmt(v):
    if v is None: return "—"
    if isinstance(v,float): return f"{v:.3f}"
    return str(v)


def markdown(data:dict)->str:
    ctx=data["context"]
    foundation=ctx.get("foundation") or {}
    lines=["# EMU v5020 — five-way winner/refork research results","",
           "Offline evidence only. No policy weights, training state, or "
           "scorecards were modified by this report.","",
           "## Provenance","",
           f"- Pinned training source: `{ctx.get('source_commit','unknown')}`",
           f"- Parent foundation: `{foundation.get('path','unavailable')}`",
           f"- Foundation step: {_fmt(foundation.get('step'))}",
           f"- Parent SHA-256: `{foundation.get('sha256','unavailable')}`",
           f"- Worlds: {_fmt(ctx.get('envs'))}; seed: {_fmt(ctx.get('seed'))}; "
           f"screen worlds: {_fmt(ctx.get('screen_envs'))}; horizon: {_fmt(ctx.get('horizon'))}",""]
    for stage,info in data["stages"].items():
        board=info["scoreboard"]
        winner=(board or {}).get("winner")
        lines += [f"## {stage}","",f"Winner: **{winner or 'not selected'}**",
                  f"Decision: {(board or {}).get('reason','stage in progress')}","",
                  "| Arm | Probe eligible | Probe fall | Steps after finish | Exam eligible | Frozen score | Core | Recovery | Slopes | Graded crossings |",
                  "|---|---|---:|---:|---|---:|---:|---:|---:|---:|"]
        grades={x["arm"]:x for x in (board or {}).get("rows",[])}
        for arm in ARMS:
            info_arm=info["arms"].get(arm,{})
            probe=info_arm.get("probe_health") or {}
            end=(info_arm.get("phases") or {}).get("finish") or {}
            step=(end.get("metrics_end") or {}).get("step")
            row=grades.get(arm,{})
            lines.append(
                f"| {arm} | {_fmt(probe.get('eligible'))} | "
                f"{_fmt(probe.get('fall_rate'))} | {_fmt(step)} | "
                f"{_fmt(row.get('eligible'))} | {_fmt(row.get('score'))} | "
                f"{_fmt(row.get('core'))} | {_fmt(row.get('hard'))} | "
                f"{_fmt(row.get('slope'))} | {_fmt(row.get('graded_step_score'))} |")
        lines.append("")
        if board is None:
            lines += ["*Training is still in progress; no frozen-policy ranking yet.*",""]
        else:
            lines += ["The parent and every candidate were graded on matched frozen-policy",
                      "core/recovery/contact-slope tests and real collidable obstacles",
                      "at 5/10/20/40/80/150mm frontiers. Graded crossings are observed",
                      "physical events, not training rewards or proof of hardware success.",""]
        for arm,arm_data in info["arms"].items():
            for phase,phase_data in arm_data["phases"].items():
                late=phase_data["metrics_end"]
                lines.append(
                    f"- **{arm}/{phase}**: step {_fmt(late.get('step'))}, "
                    f"reward {_fmt(late.get('reward_mean'))}, "
                    f"fall {_fmt(late.get('fall_rate'))}, "
                    f"episode length {_fmt(late.get('ep_len_mean'))}, "
                    f"torque over rating {_fmt(late.get('servo_over_rated_frac'))}; "
                    f"checkpoint {'saved' if phase_data['has_final_checkpoint'] else 'not complete'}.")
        lines.append("")
    lines += ["## Limitations","",
              "- Training reward/fall rate comparisons across specialists are not matched exams.",
              "- Servo shaft mechanical watts are not battery electrical watts.",
              "- A probe passing its collapse gate is not proof of controlled walking.",
              "- Saved scoreboards and selected winners are authoritative; this report only reads them.",
              "- Archive excludes large neural checkpoints and console logs.",""]
    return "\n".join(lines)+"\n"


def write_refork_report(root:Path)->list[Path]:
    root=Path(root).expanduser().resolve()
    data=collect(root)
    outputs=[]
    for filename,body in (("results.md",markdown(data)),
                          ("results_data.json",json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+"\n")):
        temp=root/(filename+".tmp")
        temp.write_text(body,encoding="utf-8")
        dest=root/filename
        temp.replace(dest)
        outputs.append(dest)
    dest=root/"results_bundle.zip"
    temp=root/"results_bundle.zip.tmp"
    with zipfile.ZipFile(temp,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for output in outputs:z.write(output,arcname=output.name)
        z.write(root/"refork_context.json",arcname="refork_context.json")
        for stage in data["stages"]:
            for filename in STAGE_FILES:
                item=root/stage/filename
                if item.is_file() and not item.is_symlink():
                    z.write(item,arcname=f"{stage}/{filename}")
            for arm in data["stages"][stage]["arms"]:
                ph=root/stage/arm/"probe_health.json"
                if ph.is_file() and not ph.is_symlink():
                    z.write(ph,arcname=f"{stage}/{arm}/probe_health.json")
                for phase in ("probe","finish"):
                    for filename in RUN_FILES:
                        item=root/stage/arm/phase/filename
                        if item.is_file() and not item.is_symlink():
                            z.write(item,arcname=f"{stage}/{arm}/{phase}/{filename}")
    temp.replace(dest)
    outputs.append(dest)
    return outputs
