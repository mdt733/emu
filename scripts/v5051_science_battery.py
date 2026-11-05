"""v5051 frozen-actor science battery; examinations ONLY, no PPO updates.

Run:
 python -m scripts.v5051_science_battery --policy showcase/champion.emupolicy \
   --device cuda --profile pilot --output ~/emu-data/runs/v5051-audit-s735

The nominal step difficulty controls sampling, NOT the actual wall height.
See independent mission_autopsy in each prism_exam.json. This battery does
not prove fixed-height stair mastery or physical standing feasibility.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from statistics import mean


def wilson(k, n, z=1.959963984540054):
    """95% Wilson score interval for independent Bernoulli *trial* counts.

    Caution: shared training policy and related physical worlds may cluster.
    """
    if n <= 0:
        return None
    p = k/n
    den = 1+z*z/n
    mid = (p+z*z/(2*n))/den
    hw = z*math.sqrt((p*(1-p)+z*z/(4*n))/n)/den
    return [round(max(0.,mid-hw),4),round(min(1.,mid+hw),4)]


def sha256(path):
    digest = hashlib.sha256()
    with open(path,"rb") as handle:
        for data in iter(lambda: handle.read(1024*1024),b""):
            digest.update(data)
    return digest.hexdigest()


def git_sha():
    try:
        return subprocess.check_output(
            ["git","rev-parse","HEAD"],text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError):
        return "unavailable"


def panels(profile):
    if profile == "pilot":
        # Minimum meaningful smoke: low and high commands, actual contacts
        # still sampled procedurally. 2 held-out simulation seeds, not PPO seeds.
        speeds=(0.,.06,.30,.70)
        lips=(1.,2.5,3.)
        seeds=(1701,1702)
    else:
        speeds=(0.,.03,.06,.10,.30,.50,.70,1.0)
        lips=(.5,1.,2.,2.5,3.,3.5,4.)
        seeds=(1701,1702,1703)
    # Flat panels test stillness, drift, speed, stride and survival.
    for seed in seeds:
        for speed in speeds:
            yield dict(seed=seed,mode="flat",speed=speed,lip_cm=0.,phase_probe="normal",tof_probe="normal")
        # Stair panels use two approach velocities and signed physical lips.
        for lip in lips:
            for speed in (.10,.20):
                yield dict(seed=seed,mode="step",speed=speed,lip_cm=lip,phase_probe="normal",tof_probe="normal")

        # Same frozen checkpoint, initial-state seed and operator command.
        for mode,speed,lip in (("flat",0.,0.),("flat",.70,0.),("step",.10,3.)):
            for phase in ("hold","invert","random"):
                yield dict(seed=seed,mode=mode,speed=speed,lip_cm=lip,
                           phase_probe=phase,tof_probe="normal")
            yield dict(seed=seed,mode=mode,speed=speed,lip_cm=lip,
                       phase_probe="normal",tof_probe="mask")


def panel_name(spec):
    return (f"s{spec['seed']}-{spec['mode']}-"
            f"v{round(spec['speed']*1000):04d}mmps-"
            f"lip{round(spec['lip_cm']*10):03d}mm-"
            f"ph{spec['phase_probe']}-tof{spec['tof_probe']}")


def aggregate(rows):
    out=[]
    for key in sorted(set((x["mode"],x["speed"],x["lip_cm"],x["phase_probe"],x["tof_probe"]) for x in rows)):
        selected=[x for x in rows if (x["mode"],x["speed"],x["lip_cm"],x["phase_probe"],x["tof_probe"])==key]
        worlds=[w for item in selected for w in item["worlds"]]
        missions=[item["mission_autopsy"] for item in selected]
        n=len(worlds)
        survived=sum(bool(w["survived"]) for w in worlds)
        completed=sum(x["completed_missions"] for x in missions)
        contacts=sum(x["eligible_supported_contacts"] for x in missions)
        def avg(name):
            values=[float(w[name]) for w in worlds if w.get(name) is not None
                    and math.isfinite(float(w[name]))]
            return round(mean(values),5) if values else None
        out.append(dict(mode=key[0],phase_probe=key[3],tof_probe=key[4],requested_vx_mps=key[1],
                        requested_lip_frontier_cm=key[2],
                        panels=len(selected),worlds=n,
                        survived=survived,survival_fraction=round(survived/n,4),
                        survival_wilson_95=wilson(survived,n),
                        supported_contact_events=contacts,
                        completed_post_contact_missions=completed,
                        completion_fraction=(round(completed/contacts,4) if contacts else None),
                        completion_wilson_95=wilson(completed,contacts),
                        mean_achieved_vx_mps=avg("achieved_vx_mps"),
                        mean_footfall_hz=avg("footfall_rate_hz"),
                        mean_phase_locking=avg("phase_locking_strength"),
                        mean_stride_m=avg("same_foot_stride_mean_m"),
                        mean_xy_drift_m=avg("net_xy_drift_m"),
                        mean_shaft_j_per_progress_m=avg("shaft_drive_j_per_progress_m")))
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    parser.add_argument("--profile",choices=("pilot","full"),default="pilot")
    parser.add_argument("--worlds",type=int,default=16)
    parser.add_argument("--horizon",type=int,default=650)
    parser.add_argument("--max-panels",type=int,default=0,
                        help="Limit initial integration test without changing panel order")
    args=parser.parse_args()
    if not 2 <= args.worlds <= 32 or not 100 <= args.horizon <= 650:
        parser.error("worlds 2..32; horizon 100..650 (existing exam contract)")
    if args.max_panels < 0:
        parser.error("max-panels must be nonnegative")
    policy=args.policy.expanduser().resolve()
    out=args.output.expanduser().resolve()
    if not policy.is_file() or policy.suffix!=".emupolicy":
        parser.error("policy must be an existing .emupolicy")
    if out==policy or out==policy.parent or policy in out.parents:
        parser.error("do not write examinations over policy directory")
    out.mkdir(parents=True,exist_ok=True)
    identity=dict(schema="emu_v5051_science_battery_v1",
                  checkout_sha=git_sha(),actor_sha256=sha256(policy),
                  policy_path=str(policy),device=args.device,
                  profile=args.profile,worlds=args.worlds,horizon=args.horizon)
    lock=out/"identity.json"
    if lock.exists():
        if json.loads(lock.read_text())!=identity:
            raise RuntimeError("Output directory belongs to another actor/config/source. "
                               "Choose a new directory.")
    else:
        # Do not silently adopt a pre-existing directory.
        if any(out.iterdir()):
            raise FileExistsError("New battery output must start empty")
        lock.write_text(json.dumps(identity,indent=2)+"\n")
    from emu.prism_exam import evaluate_prism_policy
    planned=list(panels(args.profile))
    if args.max_panels:
        planned=planned[:args.max_panels]
    for i,spec in enumerate(planned,1):
        target=out/panel_name(spec)
        report=target/"prism_exam.json"
        if report.exists():
            print(f"[{i}/{len(planned)}] existing {target.name}",flush=True)
            continue
        if target.exists():
            raise RuntimeError(f"Incomplete panel {target}; do not overwrite "
                               "possible evidence. Move it to scratch and retry.")
        print(f"[{i}/{len(planned)}] running {target.name}",flush=True)
        evaluate_prism_policy(
            policy=policy, output=target,worlds=args.worlds,
            horizon=args.horizon,speed=spec["speed"],seed=spec["seed"],
            device=args.device,mode=spec["mode"],grade_deg=0.,
            lip_cm=spec["lip_cm"],phase_probe=spec["phase_probe"],tof_probe=spec["tof_probe"])
        if not report.is_file():
            raise RuntimeError(f"Exam did not produce {report}")
    rows=[]
    for spec in planned:
        path=out/panel_name(spec)/"prism_exam.json"
        if not path.exists():
            continue
        data=json.loads(path.read_text())
        if (data.get("protocol",{}).get("phase_probe")!=spec["phase_probe"] or
                data.get("protocol",{}).get("tof_probe")!=spec["tof_probe"]):
            raise RuntimeError(f"Observation probe mismatch: {path}")
        if data.get("source_actor_sha256")!=identity["actor_sha256"]:
            raise RuntimeError(f"Actor identity mismatch: {path}")
        autopsy=data.get("mission_autopsy")
        if autopsy is None:
            raise RuntimeError(f"v5051 mission autopsy missing: {path}")
        rows.append(dict(**spec,worlds=data["worlds"],mission_autopsy=autopsy))
    agg=aggregate(rows)
    data=dict(**identity,panels_planned=len(planned),
              panels_complete=len(rows),aggregates=agg)
    (out/"science_summary.json").write_text(
        json.dumps(data,indent=2,allow_nan=False)+"\n")
    lines=["# v5051 independent frozen-actor science battery","",
           f"Source: `{identity['checkout_sha']}`",
           f"Actor SHA256: `{identity['actor_sha256']}`",
           f"Completed: {len(rows)}/{len(planned)} panels; "
           f"{args.worlds} physical worlds/panel.","",
           "IMPORTANT: requested lip values are **sampling frontiers**, "
           "not imposed measured wall heights. The post-contact mission "
           "denominator is verified supported crossings, not eligible "
           "approaches; no fixed-height or eight-stage certification.",
           "Frozen phase and ToF probes alter policy observations only, not simulated source sensors, clock or reward. Dependence does not prove necessity.",
           "These intervals assume independent events and **do not** "
           "represent uncertainty across independently trained PPO seeds.",
           "Zero counted contacts means no mission rate can be estimated.",
           "10-second follow-up certification may be truncated by the "
           "650-frame original examiner horizon.","",
           "| Mode | Probe | Requested speed | Lip frontier | Survival | Footfalls/s | Phase lock | "
           "Stride m | Drift m | Supported contacts | Completed follow-ups |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in agg:
        def fmt(v):
            return "—" if v is None else str(v)
        lines.append(
            f"| {row['mode']} | phase={row['phase_probe']},ToF={row['tof_probe']} | "
            f"{row['requested_vx_mps']:.2f} | "
            f"{row['requested_lip_frontier_cm']:.2f} | "
            f"{row['survived']}/{row['worlds']} | "
            f"{fmt(row['mean_footfall_hz'])} | {fmt(row['mean_phase_locking'])} | "
            f"{fmt(row['mean_stride_m'])} | "
            f"{fmt(row['mean_xy_drift_m'])} | "
            f"{row['supported_contact_events']} | "
            f"{row['completed_post_contact_missions']} |")
    lines += ["","## Stop/go research gate","",
              "**STOP: this battery is a measurement stage, not permission "
              "to start new PPO training.** Separate fixed-actual-height "
              "obstacles, supported standing feasibility, sensor/phase "
              "ablation, contact parity and source-pinned optimizer "
              "checkpoints remain required.",
              "No result from these procedural exams alone promotes a champion."]
    (out/"SCIENCE_REVIEW.md").write_text("\n".join(lines)+"\n")
    print(f"Completed {len(rows)}/{len(planned)} panels; review {out/'SCIENCE_REVIEW.md'}")


if __name__ == "__main__":
    main()
