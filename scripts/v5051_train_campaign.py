"""v5051: ONE 0-60M PPO foundation -> independent 60-120M PPO treatment arms.

Never trains when called with --stage plan. In all other stages, each 'train'
call is the real, source-pinned current emu.cli trainer (six motors, 360 actor,
PPO optimiser and original real convex prism carpet); no fake trainer.

All arms resume the SAME completed parent final.pkl, including curriculum,
observation normaliser, value network, optimiser and saved policy. Environment
rollout state is fresh, as in EMU's existing rewind contract.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from emu.checkpoint_io import load_checkpoint, update_boundary_at_or_after
from emu.training_config import TrainingConfig, training_sensor_contract

SOURCE = Path(__file__).resolve().parents[1]
PARENT_STEPS = 60_000_000
FINAL_STEPS = 120_000_000
DEFAULT_ENVS = 2048
COMMON = (
    "--profile","sim2real", "--terrain-actor", "--unified-capabilities",
    "--terrain-carpet","--carpet-prisms","--academy-stream",
    "--terrain-step-max-m",".08","--terrain-lesson","coherent-careful",
    "--growth-recipe","main-mature","--foundation-steps","20000000",
    "--fixed-eval-every-steps","60000000",
    "--solver-iterations","2","--solver-ls-iterations","4",
    # Keep an identical post-60M Academy schedule across controls and
    # all single-factor rewards, rather than silently changing exposure.
    "--academy-intro-steps","60000000",
    "--academy-start-steps","75000000",
    "--academy-ramp-steps","15000000",
    "--academy-intro-share",".10", "--academy-max-share",".50",
    "--athletic-rehearsal-share",".15",
    "--retention-share",".15",
    "--temporal-vx-reward-share",".25",
)


@dataclass(frozen=True)
class Arm:
    name: str
    hypothesis: str
    overrides: tuple[str,...] = ()


ARMS: tuple[Arm,...] = (
    Arm("00_control","Unchanged reward, clock and Academy continuation"),
    Arm("01_gate_zero","Remove stepping-activity penalty only at near-zero requests",
        ("--science-gate-mode","zero-only")),
    Arm("02_gate_off","Remove episode-wide stepping gate throughout training",
        ("--science-gate-mode","off")),
    Arm("03_phase_slow","Clock runs at 25% around zero command",
        ("--science-phase-lowcmd-scale",".25")),
    Arm("04_phase_pause","Clock pauses at zero speed, recovers at commanded speed",
        ("--science-phase-lowcmd-scale","0")),
    Arm("05_phase_reward_off","Remove phase synchronisation shaping, keep physical lift",
        ("--science-phase-sync-strength","0")),
    Arm("06_stillness","15% true stillness birth exposure and zero-command gate exemption",
        ("--stillness-birth-frac",".15")),
    Arm("07_crouch","Allow 3cm additional posture-height margin only for STEP role",
        ("--science-crouch-allowance-m",".03")),
    Arm("08_supported_transfer","Extra real upper foothold, verified crossing and bilateral support",
        ("--science-foothold-bonus","3",
         "--science-supported-cross-bonus","6",
         "--science-bilateral-support-bonus","4")),
    Arm("09_sensor_teacher","Pay first real foothold when a valid close-range ToF ray was present",
        ("--science-sensor-teacher-bonus","5")),
    Arm("10_frontier_1mm","Fixed 1mm ordinary-step growth and .5mm retreat; 12 attempts",
        ("--science-step-promotion-mm","1",
         "--science-step-retreat-mm",".5",
         "--science-step-min-attempts","12")),
    Arm("11_academy_heavy","Increase Academy birth share to 65% sooner",
        ("--academy-start-steps","60000000",
         "--academy-max-share",".65")),
    Arm("12_long_stride","Small bounded useful-forward-progress-per-lift lesson",
        ("--science-stride-efficiency-weight",".08")),
    Arm("13_freedom_combo","Gate+phase+stillness+crouch: interaction screen, not attribution",
        ("--science-gate-mode","zero-only",
         "--science-phase-lowcmd-scale",".25",
         "--science-phase-sync-strength","0",
         "--stillness-birth-frac",".15",
         "--science-crouch-allowance-m",".025")),
    Arm("14_step_combo","Real foothold/bilateral payments, sensor cue, Academy density and frontier",
        ("--science-foothold-bonus","3",
         "--science-supported-cross-bonus","6",
         "--science-bilateral-support-bonus","4",
         "--science-sensor-teacher-bonus","3",
         "--academy-start-steps","60000000",
         "--academy-max-share",".60",
         "--science-step-promotion-mm","1",
         "--science-step-retreat-mm",".5",
         "--science-step-min-attempts","12")),
)


def source_sha() -> str:
    return subprocess.check_output(
        ["git","rev-parse","HEAD"],cwd=SOURCE,text=True).strip()


def validate_checkout():
    if subprocess.check_output(
            ["git","status","--porcelain"],cwd=SOURCE,text=True).strip():
        raise RuntimeError("v5051 refuses dirty source; commit fixes before comparing arms")
    if (SOURCE / ".git").exists() is False:
        raise RuntimeError("Must run in exact Git checkout")


def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as fd:
        for block in iter(lambda:fd.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def select_arms(names):
    if names in ("all",""):
        return ARMS
    selected=tuple(names.split(","))
    known={x.name:x for x in ARMS}
    if len(set(selected))!=len(selected) or any(x not in known for x in selected):
        raise ValueError("Unknown or duplicated --only arm; choose from: "+", ".join(known))
    return tuple(known[name] for name in selected)


def cli_command(root, *, seed, envs, arm=None, resume=None, check=False):
    target=PARENT_STEPS if arm is None else FINAL_STEPS
    directory=root/"foundation" if arm is None else root/"arms"/arm.name
    cmd=[sys.executable,"-m","emu.cli","train","--run-dir",str(directory),
         "--seed",str(seed),"--envs",str(envs),
         "--device","cuda","--steps",str(target),*COMMON]
    if arm is not None:
        cmd.extend(arm.overrides)
    if resume is not None:
        cmd.extend(("--resume",str(resume)))
    if check:
        cmd.append("--check-resume-only")
    return cmd


def run(cmd):
    print("+ "+" ".join(cmd),flush=True)
    env=dict(os.environ,EMU_UNIFIED_CONTINUOUS="cold-growth-approved",
             PYTHONPATH=str(SOURCE),
             XLA_PYTHON_CLIENT_PREALLOCATE="false")
    subprocess.run(cmd,cwd=SOURCE,env=env,check=True)


def parent_checkpoint(root,seed,envs):
    directory=root/"foundation"
    checkpoint=directory/"checkpoints"/"final.pkl"
    if not checkpoint.is_file():
        raise FileNotFoundError("Train complete 60M foundation FIRST: "+str(checkpoint))
    cfg=TrainingConfig.load(directory/"training_config.json")
    if cfg.seed!=seed or cfg.envs!=envs or cfg.steps!=PARENT_STEPS:
        raise RuntimeError("Parent was trained with different seed/worlds/steps")
    payload=load_checkpoint(
        checkpoint,validate=True,
        expected_observation_schema=cfg.observation_schema,
        expected_sensor_contract=training_sensor_contract(cfg))
    wanted=update_boundary_at_or_after(PARENT_STEPS,envs)
    if int(payload["meta"]["step"])!=wanted:
        raise RuntimeError("Parent final checkpoint age is not the 60M PPO boundary")
    commit=(payload["meta"].get("source_provenance") or {}).get("git_commit")
    if commit!=source_sha():
        raise RuntimeError("Research parent must be trained at THIS EXACT source SHA; "
                           "do not fork after changing source")
    return checkpoint,digest(checkpoint),int(payload["meta"]["step"])


def save_contract(root,seed,envs,sha,parent=None):
    # Study-level immutable identity, independent of each arm's resume.
    # Never invent a parent SHA before the 60M final.pkl exists.
    manifest=dict(schema="v5051_shared_60m_to_120m_v1",
                  source_sha=sha,seed=seed,envs=envs,
                  common_cli=list(COMMON),foundation_steps=PARENT_STEPS,
                  arm_target_steps=FINAL_STEPS,
                  arms=[dict(name=x.name,hypothesis=x.hypothesis,
                             overrides=list(x.overrides)) for x in ARMS])
    path=root/"science_contract.json"
    if path.is_file():
        if json.loads(path.read_text())!=manifest:
            raise RuntimeError("Study contract changed; refuse mismatched resumed arms")
    else:
        if root.exists() and any(root.iterdir()):
            raise RuntimeError("Study directory nonempty without science contract")
        root.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(manifest,indent=2)+"\n")
    if parent is not None:
        lock=root/"parent_identity.json"
        expected=dict(source_sha=sha,sha256=parent[1],ppo_boundary=parent[2])
        if lock.exists():
            if json.loads(lock.read_text())!=expected:
                raise RuntimeError("Shared 60M parent final.pkl changed")
        else:
            lock.write_text(json.dumps(expected,indent=2)+"\n")


def export_and_examine(root,arm,seed,device,worlds):
    run_dir=root/"arms"/arm.name
    final=run_dir/"checkpoints"/"final.pkl"
    if not final.is_file():
        raise FileNotFoundError(f"Cannot exam incomplete training arm: {arm.name}")
    export_dir=root/"exports"
    export_dir.mkdir(parents=True,exist_ok=True)
    exported=export_dir/(arm.name+".emupolicy")
    if not exported.exists():
        run([sys.executable,"-m","emu.cli","export-policy",
             str(final),str(exported)])
    from emu.prism_exam import evaluate_prism_policy
    # Identical, predeclared held-out conditions for EVERY arm.
    # Each panel is interpreted by measured height, NEVER nominal label.
    panel_protocol=(("flat",0.,0.),("flat",.70,0.),
                    ("step",.10,3.),("step",.10,4.))
    for key in (1701,1702):
        for mode,speed,lip in panel_protocol:
            name=f"s{key}-{mode}-vx{int(round(speed*100)):03d}-lip{int(round(lip*10)):03d}"
            directory=root/"exams"/arm.name/name
            report=directory/"prism_exam.json"
            if report.is_file():
                found=json.loads(report.read_text())
                if found.get("source_actor_sha256")!=json.loads(
                        exported.read_text())["source"]["actor_sha256"]:
                    raise RuntimeError(f"Frozen exam actor changed: {report}")
                continue
            if directory.exists():
                raise RuntimeError(f"Incomplete frozen examination; move aside {directory}")
            print(f"v5051 independent exam: {arm.name} {name}",flush=True)
            evaluate_prism_policy(
                policy=exported,output=directory,worlds=worlds,
                horizon=650,speed=speed,seed=key,device=device,
                mode=mode,lip_cm=lip,grade_deg=0.)


def compile_report(root,only):
    rows=[]
    for arm in only:
        directory=root/"arms"/arm.name
        config=directory/"training_config.json"
        final=directory/"checkpoints"/"final.pkl"
        if not config.is_file() or not final.is_file():
            continue
        reports=list((root/"exams"/arm.name).glob("*/prism_exam.json"))
        world=[]
        up20=up25=down20=missions=contacts=0
        for f in sorted(reports):
            d=json.loads(f.read_text())
            protocol=d["protocol"]
            mode="step" if protocol["requested_lip_frontier_cm"]>0 else "flat"
            for w in d["worlds"]:
                world.append((mode,protocol["requested_vx_mps"],w))
                if mode=="step":
                    for e in w["supported_lips"]:
                        h=float(e["signed_height_cm"])
                        up20+=h>=2.
                        up25+=h>=2.5
                        down20+=h<=-2.
            a=d.get("mission_autopsy") or {}
            missions+=a.get("completed_missions",0)
            contacts+=a.get("eligible_supported_contacts",0)
        def avg(mode,speed,field):
            x=[float(w[field]) for m,v,w in world if m==mode and v==speed
               and w.get(field) is not None]
            return round(sum(x)/len(x),4) if x else None
        rows.append(dict(arm=arm.name,hypothesis=arm.hypothesis,
                         source_training=source_sha(),
                         parent_sha256=(json.loads((root/"parent_identity.json").read_text())
                                        ["sha256"] if (root/"parent_identity.json").exists()
                                        else None),
                         supported_up_ge_2cm=up20,supported_up_ge_2_5cm=up25,
                         supported_down_ge_2cm=down20,
                         supported_contact_events=contacts,
                         completed_post_contact_missions=missions,
                         fast_vx=avg("flat",.70,"achieved_vx_mps"),
                         fast_survival_fraction=avg("flat",.70,"survived"),
                         still_footfalls_hz=avg("flat",0.,"footfall_rate_hz"),
                         still_drift_m=avg("flat",0.,"net_xy_drift_m"),
                         fast_stride_m=avg("flat",.70,"same_foot_stride_mean_m"),
                         exams=len(reports)))
    out=root/"SCIENCE_RESULTS.md"
    lines=["# v5051 60M shared foundation → 120M PPO arms","",
           "Counts are observed *supported contact events*, not exposure-"
           "normalised stair success. Nominal lip is a teacher frontier, "
           "not the measured step. Mission follows five-frame bilateral "
           "support and >50cm controlled progress; load transfer/clearance "
           "are still not measured. Each arm is one PPO seed and 2 held-out "
           "simulation seeds, NOT 2 independent PPO seeds.",
           "","| Arm | ≥2cm up | ≥2.5cm up | ≥2cm down | Completed post-lip | "
           "Fast vx | Fast survival | Still footfalls/s | Still drift m | Exams |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        def f(v):
            return "—" if v is None else str(v)
        lines.append(
            f"| {row['arm']} | {row['supported_up_ge_2cm']} | "
            f"{row['supported_up_ge_2_5cm']} | {row['supported_down_ge_2cm']} | "
            f"{row['completed_post_contact_missions']}/"
            f"{row['supported_contact_events']} | "
            f"{f(row['fast_vx'])} | {f(row['fast_survival_fraction'])} | "
            f"{f(row['still_footfalls_hz'])} | {f(row['still_drift_m'])} | "
            f"{row['exams']} |")
    lines+=["","## Review rule","",
            "Do NOT choose the largest training reward or a composite with "
            "saturated step credit. Prefer Pareto non-regression: genuine "
            "completed post-lip missions, hard signed-height contact evidence, "
            "speed, survival, stillness, commanded heading and mechanical "
            "work. If all arms have zero completed post-lip missions, no arm "
            "has demonstrated stair mastery; request fixed-height actor exams "
            "and more seeds before picking a successor."]
    out.write_text("\n".join(lines)+"\n")
    (root/"SCIENCE_RESULTS.json").write_text(
        json.dumps(dict(arms=rows),indent=2,allow_nan=False)+"\n")
    print(f"Report: {out}",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",choices=("plan","foundation","arms","exams","report","all"),
                   required=True)
    p.add_argument("--root",type=Path,default=Path.home()/"emu-data/runs/v5051-science-s735")
    p.add_argument("--seed",type=int,default=735)
    p.add_argument("--envs",type=int,default=DEFAULT_ENVS)
    p.add_argument("--only",default="all",help="comma-separated arm IDs, default all")
    p.add_argument("--exam-worlds",type=int,default=16)
    p.add_argument("--exam-device",choices=("cpu","cuda"),default="cuda")
    a=p.parse_args()
    if not 0<=a.seed<2**31 or not 8<=a.envs<=4096:
        p.error("seed or env count out of supported range")
    if not 2<=a.exam_worlds<=32:
        p.error("exam worlds must be 2..32")
    selected=select_arms(a.only)
    root=a.root.expanduser().resolve()
    if root==SOURCE or SOURCE in root.parents or root==Path.home():
        p.error("Study results must be outside checkout and home root")
    if a.stage=="plan":
        print(f"v5051: exact 60M PPO parent + {len(selected)} x 60M PPO forks "
              f"= {60+60*len(selected)}M newly trained world-steps")
        for arm in selected:
            print(arm.name,"—",arm.hypothesis)
        print("Parent command:"," ".join(cli_command(root,seed=a.seed,envs=a.envs)))
        for arm in selected:
            print("Child command:",arm.name,
                  " ".join(cli_command(root,seed=a.seed,envs=a.envs,
                                       arm=arm,resume=root/"foundation/checkpoints/final.pkl")))
        print("No GPU or filesystem work performed.")
        return
    validate_checkout()
    sha=source_sha()
    save_contract(root,a.seed,a.envs,sha)
    parent=None
    if a.stage in ("foundation","all"):
        parent_dir=root/"foundation"
        final=parent_dir/"checkpoints/final.pkl"
        command=cli_command(root,seed=a.seed,envs=a.envs,
                            resume=("auto" if parent_dir.joinpath("checkpoints/latest.pkl").exists()
                                    or final.exists() else None))
        run(cli_command(root,seed=a.seed,envs=a.envs,
                        resume=("auto" if "--resume" in command else None),
                        check=True))
        run(command)
    if a.stage in ("arms","exams","report","all"):
        parent=parent_checkpoint(root,a.seed,a.envs)
        save_contract(root,a.seed,a.envs,sha,parent)
    if a.stage in ("arms","all"):
        for arm in selected:
            directory=root/"arms"/arm.name
            # An incomplete existing run ALWAYS uses its own latest
            # checkpoint; no silent restart from 60M parent.
            already=directory.joinpath("checkpoints/final.pkl").exists()
            partial=directory.joinpath("checkpoints/latest.pkl").exists()
            if directory.exists() and (directory/"training_config.json").exists():
                checkpoint="auto"
            else:
                checkpoint=parent[0]
            command=cli_command(root,seed=a.seed,envs=a.envs,arm=arm,
                                resume=checkpoint)
            print(f"\n=== {arm.name}: 60M -> 120M actual PPO updates ===",flush=True)
            run(cli_command(root,seed=a.seed,envs=a.envs,arm=arm,
                            resume=checkpoint,check=True))
            run(command)
    if a.stage in ("exams","all"):
        for arm in selected:
            export_and_examine(root,arm,a.seed,a.exam_device,a.exam_worlds)
    if a.stage in ("report","exams","all"):
        compile_report(root,selected)


if __name__=="__main__":
    main()
