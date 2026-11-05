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
# One compact physical exam after training: six meaningful motion/terrain
# scenarios, not tens of overlapping diagnostic intervention panels.
# Steering and push recovery remain the fixed training retention guards.
# Grade/lip settings are SAMPLING frontiers, never measured wall heights.
PHYSICAL_PANELS = (
    ("standing",  "flat",  0.00, 0.0, 0.0),
    ("slow_walk", "flat",  0.10, 0.0, 0.0),
    ("fast_walk", "flat",  0.70, 0.0, 0.0),
    ("slope",     "slope", 0.12, 0.0, 8.0),
    ("step_3cm",  "step",  0.10, 3.0, 0.0),
    ("step_4cm",  "step",  0.10, 4.0, 0.0),
)

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
    Arm("08_supported_transfer","Real foothold, bilateral support and additional 50cm heading-controlled travel",
        ("--science-foothold-bonus","3",
         "--science-supported-cross-bonus","6",
         "--science-bilateral-support-bonus","4",
         "--science-followthrough-bonus","10")),
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
    Arm("14_step_combo","Real step/50cm followthrough rewards, ToF cue, Academy density and frontier",
        ("--science-foothold-bonus","3",
         "--science-supported-cross-bonus","6",
         "--science-bilateral-support-bonus","4",
         "--science-followthrough-bonus","10",
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


def exam_protocol(root, worlds, horizon=650):
    """Immutable per-study physical exam geometry/sampling protocol."""
    protocol=dict(version="v5051_capability_screen_v2",
                  worlds=worlds,horizon=horizon,
                  panels=[dict(label=label,mode=mode,vx_mps=speed,
                               lip_frontier_cm=lip,grade_frontier_deg=grade)
                          for label,mode,speed,lip,grade in PHYSICAL_PANELS])
    path=root/"physical_exam_protocol.json"
    if path.exists():
        if json.loads(path.read_text())!=protocol:
            raise RuntimeError("Physical exam world count or panel definition changed. "
                               "Use a separate study directory to compare like with like.")
    else:
        path.write_text(json.dumps(protocol,indent=2)+"\n")
    return protocol


def export_and_examine(root,arm,seeds,device,worlds):
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
    protocol=exam_protocol(root,worlds)
    for key in seeds:
        for label,mode,speed,lip,grade in PHYSICAL_PANELS:
            directory=root/"exams"/arm.name/f"s{key}-{label}"
            report=directory/"prism_exam.json"
            if report.is_file():
                found=json.loads(report.read_text())
                p=found.get("protocol") or {}
                if (found.get("source_actor_sha256")!=digest(exported) or
                        p.get("worlds")!=worlds or
                        p.get("horizon")!=protocol["horizon"] or
                        p.get("seed")!=key or
                        found.get("mode")!=mode or
                        abs(float(p.get("requested_vx_mps",-1))-speed)>1e-7 or
                        abs(float(p.get("requested_lip_frontier_cm",-1))-
                            (lip if mode=="step" else 0))>1e-7 or
                        abs(float(p.get("requested_grade_frontier_deg",-1))-
                            (grade if mode=="slope" else 0))>1e-7):
                    raise RuntimeError(f"Frozen exam identity changed: {report}")
                continue
            if directory.exists():
                raise RuntimeError(f"Incomplete frozen examination; move aside {directory}")
            print(f"v5051 physical screen: {arm.name} {label} seed={key} "
                  f"({worlds} worlds, {protocol['horizon']} frames)",flush=True)
            evaluate_prism_policy(
                policy=exported,output=directory,worlds=worlds,
                horizon=protocol["horizon"],speed=speed,seed=key,
                device=device,mode=mode,lip_cm=lip,grade_deg=grade)


def compile_report(root,only):
    """One honest 6-capability report, not an opaque test/score tournament.

    Four fixed retention guards provide steering/transition/recovery;
    six separate actor-only terrain scenarios provide stillness, speed,
    physical slopes and measured step missions. A *touch* is not a climb.
    """
    rows=[]
    known={spec[0] for spec in PHYSICAL_PANELS}
    protocol_file=root/"physical_exam_protocol.json"
    standard=json.loads(protocol_file.read_text()) if protocol_file.is_file() else {}
    for arm in only:
        directory=root/"arms"/arm.name
        final=directory/"checkpoints"/"final.pkl"
        if not (directory/"training_config.json").is_file() or not final.is_file():
            continue
        physical={}
        for f in sorted((root/"exams"/arm.name).glob("*/prism_exam.json")):
            label=f.parent.name.split("-",1)[1]
            if label not in known:
                raise RuntimeError(f"Unknown physical panel {label} in {f}")
            d=json.loads(f.read_text())
            if (d["protocol"]["worlds"]!=standard.get("worlds") or
                    d["protocol"]["horizon"]!=standard.get("horizon")):
                raise RuntimeError(f"Physical panel changed worlds/horizon: {f}")
            physical.setdefault(label,[]).append(d)
        def measured(label,key):
            values=[float(w[key]) for d in physical.get(label,[])
                    for w in d["worlds"] if w.get(key) is not None]
            return round(sum(values)/len(values),4) if values else None
        def survived(label):
            worlds=[w for d in physical.get(label,[]) for w in d["worlds"]]
            return f"{sum(bool(w['survived']) for w in worlds)}/{len(worlds)}" if worlds else None
        # Count signed *measured* heights from real top contacts, never
        # requested teacher frontiers. World-incidence is separate from
        # raw transition event count and from completed stair missions.
        steps=[d for label in ("step_3cm","step_4cm")
               for d in physical.get(label,[])]
        lips=[float(e["signed_height_cm"])
              for d in steps for w in d["worlds"]
              for e in w["supported_lips"]]
        contacts=sum(d["mission_autopsy"]["eligible_supported_contacts"]
                     for d in steps)
        missions=sum(d["mission_autopsy"]["completed_missions"] for d in steps)
        fixed={}
        fixed_file=directory/"fixed_eval.csv"
        if fixed_file.is_file():
            with fixed_file.open(newline="",encoding="utf-8") as fh:
                records=list(csv.DictReader(fh))
            if records:
                fixed=records[-1]
        def value(key):
            try:
                v=float(fixed.get(key,"nan"))
                return round(v,4) if v==v and abs(v)<1e10 else None
            except (ValueError,TypeError):
                return None
        rows.append(dict(
            arm=arm.name,hypothesis=arm.hypothesis,
            parent_sha256=(json.loads((root/"parent_identity.json").read_text())
                           ["sha256"] if (root/"parent_identity.json").exists() else None),
            physical_panels=sum(len(v) for v in physical.values()),
            physical_seed_count=len({f.parent.name.split("-")[0]
                                     for f in (root/"exams"/arm.name).glob("*/prism_exam.json")}),
            still_footfalls_hz=measured("standing","footfall_rate_hz"),
            still_drift_m=measured("standing","net_xy_drift_m"),
            still_survival=survived("standing"),
            slow_vx=measured("slow_walk","achieved_vx_mps"),
            fast_vx=measured("fast_walk","achieved_vx_mps"),
            fast_survival=survived("fast_walk"),
            fast_stride_m=measured("fast_walk","same_foot_stride_mean_m"),
            steering_yaw_mae=value("eval_combo70_yaw_mae"),
            steering_survival=value("eval_combo70_survival"),
            recovery_survival=value("eval_resilience75_survival"),
            transition_survival=value("eval_dynamic80_survival"),
            slope_actual_peak_deg=measured("slope","actual_grade_max_deg"),
            slope_vx=measured("slope","achieved_vx_mps"),
            slope_survival=survived("slope"),
            supported_up_ge_2cm=sum(h>=2.0 for h in lips),
            supported_up_ge_2_5cm=sum(h>=2.5 for h in lips),
            supported_down_ge_2cm=sum(h<=-2.0 for h in lips),
            completed_post_contact_missions=missions,
            eligible_supported_contacts=contacts,
            post_contact_completion_pct=(round(100*missions/contacts,1)
                                         if contacts else None),
        ))
    def fmt(x):
        return "—" if x is None else str(x)
    out=root/"SCIENCE_RESULTS.md"
    lines=[
        "# EMU capability screen — shared 60M → 120M PPO arms","",
        "All rows are *measured* physical evidence from independent frozen actors.",
        "The four in-training fixed panels are locomotion retention guards; "
        "the six post-training scenarios measure stillness, slow/fast speed, "
        "slopes and signed lip encounters. One held-out environment seed is "
        "a **screen, not a replicated learning result**. Each arm uses one "
        "PPO training seed. Do not rank by training reward.",
        "",
        "The step count is distinct signed-height **contact events**; "
        "complete missions require verified bilateral support followed by "
        "50cm of controlled progress. Completion/contact is conditional on "
        "first reaching real upper support and is NOT opportunity-normalised "
        "stair success. Requested 3/4cm refers to generator sampling, "
        "never an actual measured obstacle of that size.","",
        "| Arm | Stand feet/s | Stand drift m | Slow vx | Fast vx | Fast survival | "
        "Steering yaw error | Recovery survival | Slope max measured ° | "
        "Slope survival | ≥2cm UP | ≥2.5cm UP | ≥2cm DOWN | "
        "Completed/contact | Physical panels |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['arm']} | {fmt(row['still_footfalls_hz'])} | "
            f"{fmt(row['still_drift_m'])} | {fmt(row['slow_vx'])} | "
            f"{fmt(row['fast_vx'])} | {fmt(row['fast_survival'])} | "
            f"{fmt(row['steering_yaw_mae'])} | "
            f"{fmt(row['recovery_survival'])} | "
            f"{fmt(row['slope_actual_peak_deg'])} | "
            f"{fmt(row['slope_survival'])} | "
            f"{row['supported_up_ge_2cm']} | "
            f"{row['supported_up_ge_2_5cm']} | "
            f"{row['supported_down_ge_2cm']} | "
            f"{row['completed_post_contact_missions']}/"
            f"{row['eligible_supported_contacts']} | "
            f"{row['physical_panels']} |")
    lines+=["","## What can justify promotion?","",
            "Demand non-regression in basic dynamic walking, steering, recovery, "
            "quiet balance and *actual measured* grades. For stairs, demand "
            "greater signed height plus authentic complete post-contact missions. "
            "If nothing completes the post-contact mission, no arm is a "
            "demonstrated stair-climbing solution. Re-examine promising "
            "arms and the unchanged control on additional held-out environment "
            "seeds AND independent PPO training seeds. A separate exact-height "
            "fixed-wall actor exam remains necessary to certify a specified "
            "2/3/4cm obstacle.",""]
    out.write_text("\n".join(lines),encoding="utf-8")
    (root/"SCIENCE_RESULTS.json").write_text(
        json.dumps(dict(protocol=standard,arms=rows),indent=2,allow_nan=False)+"\n",
        encoding="utf-8")
    print(f"Report: {out}",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",choices=("plan","foundation","arms","exams","report","all"),
                   required=True)
    p.add_argument("--root",type=Path,default=Path.home()/"emu-data/runs/v5051-capability-v2-s735")
    p.add_argument("--seed",type=int,default=735)
    p.add_argument("--envs",type=int,default=DEFAULT_ENVS)
    p.add_argument("--only",default="all",help="comma-separated arm IDs, default all")
    p.add_argument("--exam-worlds",type=int,default=8,
                   help="physical screening worlds per panel (default 8)")
    p.add_argument("--exam-seeds",default="1701",
                   help="comma-separated held-out seeds; use 1701,1702 for finalist verification")
    p.add_argument("--exam-device",choices=("cpu","cuda"),default="cuda")
    a=p.parse_args()
    if not 0<=a.seed<2**31 or not 8<=a.envs<=4096:
        p.error("seed or env count out of supported range")
    if not 2<=a.exam_worlds<=32:
        p.error("exam worlds must be 2..32")
    selected=select_arms(a.only)
    try:
        exam_seeds=tuple(int(x) for x in a.exam_seeds.split(","))
    except ValueError as exc:
        p.error(f"invalid --exam-seeds: {exc}")
    if (not exam_seeds or len(exam_seeds)>5 or
            len(set(exam_seeds))!=len(exam_seeds) or
            any(x<0 or x>=2**31 for x in exam_seeds)):
        p.error("--exam-seeds must contain 1-5 distinct nonnegative integers")
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
        print(f"Independent physical screen: {len(PHYSICAL_PANELS)} panels "
              f"x {len(exam_seeds)} seed(s) x {len(selected)} arm(s), "
              f"{a.exam_worlds} worlds per panel")
        print("Extra seeds: re-run --stage exams --only 00_control,... "
              "--exam-seeds 1701,1702 after screening.")
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
            export_and_examine(root,arm,exam_seeds,a.exam_device,a.exam_worlds)
    if a.stage in ("report","exams","all"):
        compile_report(root,selected)


if __name__=="__main__":
    main()
