#!/usr/bin/env python3
"""v5048: disposable 50M shared foundation -> eight 50M research forks ->
100M continuation of the physically best branch (200M cumulative age).

Every arm inherits exactly the SAME 50M network, optimizer, observation
normalizer, curriculum, health state and field from an unchanged final.pkl.
Uses the existing GPU trainer and independent real prism examiner.
One 9-hour soft budget, never kills a running process. No main merge.
DELETE this tournament launcher after writing permanent experiment findings.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path.home() / "emu-data" / "runs"
SCRATCH = Path.home() / "emu-data" / "scratch" / "v5048-shared-parent"
EXPECTED_BRANCH = "experiment/v5048-shared-parent-pinnacle"
FOUNDATION_STEPS = 50_000_000
FORK_TARGET = 100_000_000
FINAL_TARGET = 200_000_000
PARENT_SPEC = {"intro": 600_000_000, "start": 800_000_000}
SEEDS = (1701, 1702)
WORLDS = 16
HORIZON = 500
PHYSICS = {"solver": 2, "ls": 4}
# One 50M physical, network and optimizer ancestor. Each intervention
# below changes only one nominated teaching/physics parameter from early.
EARLY = {"intro": 50_000_000, "start": 50_000_000}
ARMS = (
    ("no_academy", PARENT_SPEC),
    ("early_lips", EARLY),
    ("precise_frontier", {**EARLY, "step_mm": 1, "retreat_mm": 1,
                          "attempts": 12}),
    ("zero_balance", {**EARLY, "stillness": .10}),
    ("temporal_speed", {**EARLY, "temporal": .50}),
    ("fast_rehearsal", {**EARLY, "athletic": .30}),
    ("heavy_academy", {**EARLY, "academy_share": .65}),
    ("solver4_control", {**EARLY, "solver": 4}),
)
# Frozen, raw physical contacts; all exam scenarios run on both held-out
# seeds, and use the EXACT exported actor's physics (not main defaults).
EXAMS = (
    ("still", "flat", .00, 0., 0.),
    ("slow", "flat", .06, 0., 0.),
    ("cruise", "flat", .20, 0., 0.),
    ("fast", "flat", .70, 0., 0.),
    ("step_5mm", "step", .15, 0., .5),
    ("step_2cm", "step", .15, 0., 2.),
    ("step_3cm", "step", .20, 0., 3.),
    ("slope_8deg", "slope", .15, 8., 0.),
)


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT,
                                   stderr=subprocess.DEVNULL, text=True).strip()


def atomic(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, sort_keys=True,
                               allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def policy_args(spec, *, steps, seed, run, envs=2048):
    # Same exact source-pinned PPO/world/teacher/reward recipe for each arm
    # except the named study switch. 2/4 is research physics, not a
    # replacement for robot.xml. CLI resolves and validates all flags.
    solver = spec.get("solver", PHYSICS["solver"])
    opts = [
        "--run-dir", str(run), "--steps", str(steps),
        "--envs", str(envs), "--seed", str(seed), "--device", "cuda",
        "--profile", "sim2real", "--pitch-deg", "8",
        "--terrain-actor", "--terrain-step-max-m", ".08",
        "--unified-capabilities", "--terrain-carpet", "--carpet-prisms",
        "--carpet-transition-cap-deg", "3",
        "--foundation-steps", "10000000", "--foundation-mix", "main",
        "--growth-recipe", "main-mature",
        "--teaching-schedule", "terrain-early",
        "--terrain-lesson", "coherent-careful", "--force-program", "gentle",
        "--athletic-rehearsal-share", str(spec.get("athletic", .10)),
        "--retention-share", ".15",
        "--temporal-vx-reward-share", str(spec.get("temporal", 0.)),
        "--stillness-birth-frac", str(spec.get("stillness", 0.)),
        "--academy-stream", "--academy-intro-steps", str(spec["intro"]),
        "--academy-intro-share", ".10",
        "--academy-start-steps", str(spec["start"]),
        "--academy-ramp-steps", "10000000",
        "--academy-max-share", str(spec.get("academy_share", .40)),
        "--academy-height-cap-m", ".15",
        "--academy-course-episode-frames", "1200",
        "--science-step-promotion-mm", str(spec.get("step_mm", 0)),
        "--science-step-retreat-mm", str(spec.get("retreat_mm", 0)),
        "--science-step-min-attempts", str(spec.get("attempts", 4)),
        "--fixed-eval-every-steps", "50000000",
        "--solver-iterations", str(solver),
        "--solver-ls-iterations", str(PHYSICS["ls"]),
    ]
    return [sys.executable, "-u", "-m", "emu.cli", "train", *opts]


def run_dir(label, seed):
    return RUNS / f"v5048-{label}-s{seed}"


def exam_cmd(actor, out, name, mode, speed, grade, lip, seed):
    return [sys.executable, "-u", "-m", "emu.cli", "prism-exam",
            "--policy", str(actor), "--output", str(out),
            "--worlds", str(WORLDS), "--horizon", str(HORIZON),
            "--seed", str(seed), "--device", "cuda",
            "--mode", mode, "--speed", str(speed),
            "--grade-deg", str(grade), "--lip-cm", str(lip)]


def process(command, log, env, *, echo=True):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", buffering=1, encoding="utf-8") as f:
        f.write("$ " + " ".join(command) + "\n")
        with subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True,
                              bufsize=1, errors="replace") as p:
            assert p.stdout is not None
            from collections import deque
            tail = deque(maxlen=48)
            for line in p.stdout:
                f.write(line)
                tail.append(line.rstrip("\n"))
                if echo and (line.startswith(("[emu_", "EMU ", "PPO ", "steady state",
                                              "Traceback", "RuntimeError", "ValueError",
                                              "GPU memory", "CONTACT SOLVER"))
                             or "fixed-eval" in line or "PPO TARGET" in line):
                    print(line.rstrip(), flush=True)
            code = p.wait()
    if code:
        print(f"FAILED with code {code}: {log}", flush=True)
        for line in tail:
            print(line, flush=True)
    return code


def read_last_csv(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows in {path}")
    return rows[-1]


def checkpoint_boundary(step, envs=2048):
    batch=envs*32
    return (int(step)+batch-1)//batch*batch


def checked_actor(actor, sha, spec, step, *, envs=2048):
    data = json.loads(actor.read_text(encoding="utf-8"))
    source = data.get("source", {})
    physics = data.get("training", {})
    if (source.get("git_commit") != sha or
            source.get("step") != checkpoint_boundary(step,envs) or
            physics.get("terrain_carpet_prisms") is not True or
            physics.get("solver_iterations") != spec.get("solver", 2) or
            physics.get("solver_ls_iterations") != 4 or
            abs(physics.get("stillness_birth_frac",0.)
                -spec.get("stillness",0.)) > 1e-7):
        raise RuntimeError(
            f"Export does not match source, target step or solver: {actor}")
    return data


def lifecycle_acceptance(sha, state, env):
    """Small real checkpoint/export/exam contract BEFORE expensive forks.

    Validates both Newton 2/4 and 4/4 across actual checkpoint,
    training_config, exported policy metadata and a CUDA frozen physics
    trace. A tiny policy need not walk. Deliberately no extra full suite.
    """
    for key, spec in (
            ("solver2_ls4", {**EARLY, "solver":2}),
            ("solver4_ls4", {**EARLY, "solver":4}),
            ("stillness_10pct", {**EARLY, "solver":2, "stillness":.10})):
        solver=spec["solver"]
        folder=SCRATCH/"acceptance"/key
        checkpoint=folder/"checkpoints"/"final.pkl"
        acceptance_env=dict(env)
        acceptance_env["EMU_UNIFIED_ACCEPTANCE"]="bounded-ppo-lifecycle"
        if not checkpoint.is_file():
            if folder.exists() and any(folder.iterdir()):
                if not (folder/"training_config.json").is_file():
                    raise RuntimeError(f"Acceptance dir nonempty: {folder}")
                resume=["--resume","auto"]
            else:
                resume=[]
            cmd=policy_args(spec,steps=512,seed=735,run=folder,envs=8)+resume
            print(f"ACTUAL 2-UPDATE CUDA CHECKPOINT GATE: {key}",flush=True)
            if process(cmd,SCRATCH/"logs"/(key+"_lifecycle.log"),
                       acceptance_env):
                raise RuntimeError(f"CUDA checkpoint lifecycle failed: {key}")
        actor=folder/"tiny.emupolicy"
        if not actor.exists():
            cmd=[sys.executable,"-u","-m","emu.cli","export-policy",
                 str(checkpoint),str(actor)]
            if process(cmd,SCRATCH/"logs"/(key+"_export.log"),env,echo=False):
                raise RuntimeError(f"Acceptance actor export failed: {key}")
        checked_actor(actor,sha,spec,512,envs=8)
        out=folder/"frozen_exam"
        physical=out/"prism_exam.json"
        if not physical.exists():
            cmd=exam_cmd(actor,out,"flat","flat",.10,0.,0.,1701)
            # Keep the acceptance bounded: the actual tournament exams
            # later use 16 worlds x 500 frames on both held-out seeds.
            cmd[cmd.index("--worlds")+1]="2"
            cmd[cmd.index("--horizon")+1]="100"
            if process(cmd,SCRATCH/"logs"/(key+"_frozen.log"),
                       env,echo=False):
                raise RuntimeError(f"CUDA actor physics exam failed: {key}")
        exam=json.loads(physical.read_text(encoding="utf-8"))
        protocol=exam.get("protocol") or {}
        if (exam.get("source_training_sha")!=sha or
            protocol.get("compiled_solver_iterations")!=solver or
            protocol.get("compiled_solver_ls_iterations")!=4 or
            protocol.get("worlds")!=2 or protocol.get("horizon")!=100):
            raise RuntimeError(f"Frozen solver physics mismatch in {key}")
        print(f"CUDA source/solver/checkpoint/export/exam gate PASS: {key}",
              flush=True)
    state["acceptance"]="PASS"
    atomic(SCRATCH/"state.json",state)


def run_train(label, spec, steps, seed, sha, state, env, *,
              parent=None, parent_hash=None):
    """Cold foundation, SAME-SHA parent fork, or validated same-run extension.

    Parent is immutable. An interrupted child resumes its own checkpoint,
    never restarts the original 50M checkpoint under a nonempty run path.
    """
    key=f"{label}_s{seed}_{steps}"
    folder=run_dir(label,seed)
    checkpoint=folder/"checkpoints"/"final.pkl"
    if parent is not None:
        parent=Path(parent).resolve()
        if not parent.is_file() or parent.name!="final.pkl":
            raise RuntimeError("Research fork parent must be saved final.pkl")
        if not parent_hash:
            raise RuntimeError("Fork requires immutable 50M parent SHA256")
        digest=hashlib.sha256(parent.read_bytes()).hexdigest()
        if digest!=parent_hash:
            raise RuntimeError("The shared 50M ancestor changed: refuse fork")
    if key in state["trained"]:
        if not checkpoint.is_file():
            raise RuntimeError(f"Recorded stage but final checkpoint missing: {checkpoint}")
        prior=state["trained"][key]
        if parent_hash is not None and prior.get("parent_sha256")!=parent_hash:
            raise RuntimeError("Fork manifest differs from the declared ancestor")
        return folder
    folder.parent.mkdir(parents=True,exist_ok=True)
    base=policy_args(spec,steps=steps,seed=seed,run=folder)
    if folder.exists() and any(folder.iterdir()):
        if not (folder/"training_config.json").is_file():
            raise RuntimeError(f"Nonempty foreign run directory: {folder}")
        # If first-use compilation failed BEFORE any checkpoint existed,
        # there is nothing to auto-resume. Retry the exact original
        # parent (or cold foundation) only when no training row was saved.
        own_ckpt=folder/"checkpoints"
        any_ckpt=own_ckpt.is_dir() and any(own_ckpt.glob("*.pkl"))
        if any_ckpt:
            cmd=base+["--resume","auto"]
            print(f"RESUME/EXTEND {key} using own checkpoint",flush=True)
        else:
            metrics_path=folder/"metrics.csv"
            if metrics_path.exists() and metrics_path.stat().st_size>0:
                raise RuntimeError(f"Nonempty metrics without checkpoint: {folder}")
            cmd=base+(["--resume",str(parent)] if parent is not None else [])
            print(f"RETRY pre-checkpoint failure for {key} "
                  "(same source/recipe, no learned updates)",flush=True)
    elif parent is not None:
        cmd=base+["--resume",str(parent)]
        print(f"NEW MATCHED FORK {key} from immutable {parent_hash[:12]}",flush=True)
        # Before GPU investment, validate parent shape/schema and explicit
        # fork compatibility with the actual CLI. --plan alone isn't enough.
        check=cmd+["--check-resume-only"]
        if process(check,SCRATCH/"logs"/(key+"_fork_gate.log"),env,echo=False):
            raise RuntimeError(f"Physical/actor checkpoint parent gate failed: {key}")
    else:
        cmd=base
        print(f"NEW 50M FOUNDATION {key}",flush=True)
    begun=time.monotonic()
    code=process(cmd,SCRATCH/"logs"/(key+".log"),env)
    if code:
        raise RuntimeError(f"Training failed for {key}: exit={code}. "
                           "Preserved all checkpoints for same-SHA resume.")
    if not checkpoint.is_file():
        raise RuntimeError(f"No final.pkl for completed {key}")
    config=json.loads((folder/"training_config.json").read_text())
    meta=json.loads((folder/"run_meta.json").read_text())
    source=(meta.get("source_provenance") or {}).get("git_commit")
    last=read_last_csv(folder/"metrics.csv")
    achieved=int(float(last["step"]))
    if (source!=sha or config.get("steps")!=steps or
            config.get("seed")!=seed or achieved<steps):
        raise RuntimeError(f"Run/source/step mismatched at {key}: "
                           f"source={source}, step={achieved}, target={steps}")
    if parent is not None:
        # A 100M fork must actually inherit that 50M checkpoint, rather
        # than accidentally cold-start with the desired research flags.
        ancestor_step=meta.get("ancestor_checkpoint_step")
        ancestor_source=meta.get("ancestor_checkpoint_source")
        if ancestor_source!=sha or not (FOUNDATION_STEPS<=int(ancestor_step or 0)<steps):
            raise RuntimeError(f"Fork source/step provenance mismatch: {key}")
        if hashlib.sha256(parent.read_bytes()).hexdigest()!=parent_hash:
            raise RuntimeError("Parent modified while child was training")
    state["trained"][key]={
        "folder":str(folder),"seconds":round(time.monotonic()-begun,2),
        "step":achieved,"fall_rate":float(last["fall_rate"]),
        "vx":float(last["vx_mean"]),
        "academy_up":float(last.get("academy_up_crossings") or 0.),
        "academy_down":float(last.get("academy_down_crossings") or 0.),
        "academy_attempts_up":float(last.get("academy_near_attempts_up") or 0.),
        "academy_attempts_down":float(last.get("academy_near_attempts_down") or 0.),
        "training_fps":float(last.get("train_steps_per_sec") or 0.),
        "parent_sha256":parent_hash,
        "config_sha256":hashlib.sha256(
            (folder/"training_config.json").read_bytes()).hexdigest(),
    }
    atomic(SCRATCH/"state.json",state)
    return folder


def export_actor(folder, sha, spec, steps, state, env):
    actor = folder / "exports" / f"step_{steps}.emupolicy"
    key = f"{folder.name}_{steps}"
    if not actor.exists():
        actor.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-u", "-m", "emu.cli", "export-policy",
               str(folder / "checkpoints" / "final.pkl"), str(actor)]
        if process(cmd, SCRATCH / "logs" / (key+"_export.log"), env,echo=False):
            raise RuntimeError(f"Export failed: {folder}")
    checked_actor(actor, sha, spec, steps)
    return actor


def get_exams(label, seed, spec, folder, actor, sha, steps, state, env,
              *, cases=EXAMS, exam_seeds=SEEDS):
    key = f"{label}_s{seed}_{steps}"
    outputs = []
    for exam_seed in exam_seeds:
        for name, mode, speed, grade, lip in cases:
            exam_id = f"{key}__{name}__s{exam_seed}"
            out = folder / "exams" / f"step_{steps}" / f"{name}_s{exam_seed}"
            result = out / "prism_exam.json"
            if result.exists():
                data = json.loads(result.read_text())
            else:
                cmd = exam_cmd(actor, out, name, mode, speed, grade, lip, exam_seed)
                print(f"  EXAM {exam_id}", flush=True)
                if process(cmd, SCRATCH/"logs"/(exam_id+".log"),env,echo=False):
                    raise RuntimeError(f"Physical exam failed: {exam_id}")
                data = json.loads(result.read_text())
            actor_hash=hashlib.sha256(actor.read_bytes()).hexdigest()
            protocol = data.get("protocol") or {}
            source = json.loads(actor.read_text())["source"]
            if (data.get("schema") != "emu_prism_frozen_exam_v1" or
                    data.get("source_training_sha") != sha or
                    data.get("source_step") != checkpoint_boundary(steps) or
                    data.get("source_actor_sha256") != actor_hash or
                    protocol.get("seed") != exam_seed or
                    protocol.get("worlds") != WORLDS or
                    protocol.get("horizon") != HORIZON or
                    protocol.get("compiled_solver_iterations") != spec.get("solver",2) or
                    protocol.get("compiled_solver_ls_iterations") != 4):
                raise RuntimeError(f"Exam does not match frozen policy/physical solver: {exam_id}")
            if source.get("git_commit") != sha:
                raise RuntimeError(f"Unexpected actor source: {actor}")
            outputs.append((name, exam_seed, data))
            state["exams"][exam_id] = {
                "path": str(result), "up": data["measured"]["signed_supported_up"],
                "down": data["measured"]["signed_supported_down"],
                "survived": data["measured"]["survived"],
            }
            atomic(SCRATCH/"state.json",state)
    return outputs


def score(exams):
    vals = {"supported_up":0, "supported_down":0,
            "up_seeds":set(), "down_seeds":set(),
            "step_worlds":0,"flat_survival":[], "step_survival":[],
            "slope_survival":[], "zero_drift":[], "slow_error":[],
            "fast_vx":[], "fast_stride":[], "fast_footfall":[],
            "fast_yaw_bias":[], "cruise_error":[]}
    for name, seed, item in exams:
        rows = item["worlds"]
        if name.startswith("step_"):
            ups=item["measured"]["signed_supported_up"]
            downs=item["measured"]["signed_supported_down"]
            vals["supported_up"]+=ups
            vals["supported_down"]+=downs
            if ups: vals["up_seeds"].add(seed)
            if downs: vals["down_seeds"].add(seed)
            vals["step_worlds"]+=len(rows)
        for row in rows:
            survived=float(bool(row["survived"]))
            if name.startswith("step_"):
                vals["step_survival"].append(survived)
            elif name=="slope_8deg":
                vals["slope_survival"].append(survived)
            else:
                vals["flat_survival"].append(survived)
            if name=="still":
                vals["zero_drift"].append(row["net_xy_drift_m"])
            if name=="slow" and row["vx_mae_mps"] is not None:
                vals["slow_error"].append(row["vx_mae_mps"])
            if name=="cruise" and row["vx_mae_mps"] is not None:
                vals["cruise_error"].append(row["vx_mae_mps"])
            if name=="fast":
                if row["achieved_vx_mps"] is not None:
                    vals["fast_vx"].append(row["achieved_vx_mps"])
                if row["same_foot_stride_mean_m"] is not None:
                    vals["fast_stride"].append(row["same_foot_stride_mean_m"])
                vals["fast_footfall"].append(row["footfall_rate_hz"])
                if row["yaw_bias_rad_s"] is not None:
                    vals["fast_yaw_bias"].append(abs(row["yaw_bias_rad_s"]))
    avg=lambda name:(statistics.mean(vals[name]) if vals[name] else None)
    out={k:vals[k] for k in ("supported_up","supported_down","step_worlds")}
    out["up_exam_seeds"]=len(vals["up_seeds"])
    out["down_exam_seeds"]=len(vals["down_seeds"])
    out["both_signed_replicated"]=(
        out["up_exam_seeds"]==len(SEEDS) and
        out["down_exam_seeds"]==len(SEEDS))
    out.update({k:avg(k) for k in ("flat_survival","step_survival",
                                  "slope_survival","zero_drift","slow_error",
                                  "cruise_error","fast_vx","fast_stride",
                                  "fast_footfall","fast_yaw_bias")})
    if any(out[k] is None for k in ("flat_survival","step_survival",
                                   "zero_drift","slow_error","fast_vx")):
        raise RuntimeError("Missing mandatory physical scoreboard channels")
    out["qualified"]=(out["flat_survival"]>=.65 and
                      out["step_survival"]>=.50 and
                      out["fast_vx"]>=.20)
    # Predetermined ranks; *true* supported signed lips outrank
    # potentially reward-hacked PPO scores. Secondary axes must be safe.
    # Step evidence from only one test seed is interesting, not a
    # mastered stair claim; half-credit such events in selection.
    step_evidence=(out["supported_up"]+out["supported_down"])
    if not out["both_signed_replicated"]:
        step_evidence=min(3.,.5*step_evidence)
    out["research_score"]=round(
        8*min(12,step_evidence)
        +20*out["flat_survival"]+12*out["step_survival"]
        +8*(out["slope_survival"] or 0)
        +5*min(.8,max(0.,out["fast_vx"]))
        -15*min(1.,out["zero_drift"])
        -6*min(1.,out["slow_error"]),3)
    return out


def choose(scored):
    eligible = [(name,record) for name,record in scored.items()
                if record["qualified"]]
    if not eligible:
        raise RuntimeError("No arm passes predeclared flat/step/fast physical gate. "
                           "No high-budget finalist is authorised; investigate results.")
    # Never interpret a bare training reward or surviving in place as a
    # solved stair. Best held-out physical score wins, and the report
    # explicitly says whether up and down were shown on independent seeds.
    return max(eligible,key=lambda item:(item[1]["research_score"],
                                         item[0]=="early_lips"))[0]


def report(state):
    scored = state.get("scored") or {}
    lines = [
        "# v5048 shared-parent 50M + 8x50M + 100M research", "",
        f"Source SHA: `{state['source_sha']}`",
        f"Started: {state['started']}; last update: {utc_now()}",
        f"Training jobs: {len(state['trained'])}; frozen exam panels: {len(state['exams'])}",
        f"Total unique training steps planned: {(FOUNDATION_STEPS+len(ARMS)*(FORK_TARGET-FOUNDATION_STEPS)+(FINAL_TARGET-FORK_TARGET)):,}",
        "",
        "All eight 100M policies fork the IDENTICAL immutable 50M", 
        "checkpoint (actor, optimizer, normalizer, curriculum and health).",
        "Each arm adds 50M steps; the winner adds another 100M in its SAME run.",
        "All training is real 2048-world PPO; no policy is",
        "promoted to `main` by this script. Scores only use independent",
        "raw-physical frozen exams with matched held-out seeds, never PPO",
        "training reward or Academy's internal success flags.",
        "",
        "| Arm | True up/down | Exam seeds up/down | Flat survival | Step survival | 0cmd drift m | Fast vx m/s | Fast stride m | Footfall Hz | Score | General gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    fmt=lambda x:"—" if x is None else f"{x:.3f}"
    for label, s in scored.items():
        lines.append(
            f"| {label} | {s['supported_up']}/{s['supported_down']} "
            f"| {s['up_exam_seeds']}/{s['down_exam_seeds']} "
            f"| {fmt(s['flat_survival'])} | {fmt(s['step_survival'])} "
            f"| {fmt(s['zero_drift'])} | {fmt(s['fast_vx'])} "
            f"| {fmt(s['fast_stride'])} | {fmt(s['fast_footfall'])} "
            f"| {s['research_score']:.2f} | {'PASS' if s['qualified'] else 'FAIL'} |")
    lines += [
        "", f"Selected next research recipe: **{state.get('choice') or 'not selected'}**",
        "",
        "## Interpretation gates", "",
        "- Repeatable stair evidence requires independent UP AND DOWN support",
        "  on both exam seeds (up/down exam seed counts are listed above).",
        "- Zero observed signed lips is zero demonstrated true steps, NOT proof",
        "  that the policy ever approached a sufficiently high edge.",
        "- Mean step/slope survival alone is not evidence of locomotion.",
        "- Footfall/stride diagnostics are measured from raw foot top contacts",
        "  and heading; high cadence alone is not dynamic fast running.",
        "- Claim main parity ONLY after source-matched incumbent physical",
        "  missions, reproducible steering/speed and contact-fidelity review.",
        "- 2/4 versus 4/4 is a real trained-policy control; CUDA reset .qacc",
        "  compact/full parity remains a separate unresolved physical caveat.",
        "- If no arm passes physical admission gates, the 100M extension",
        "  is skipped; no fake Pinnacle is declared from raw training reward.",
        "- The tournament and its code are temporary; write findings into",
        "  GitHub issue #17 and the permanent report, then delete the runner.",
        "",
    ]
    path=SCRATCH/"SUMMARY.md"
    tmp=path.with_suffix(".md.tmp")
    tmp.write_text("\n".join(lines)+"\n",encoding="utf-8")
    os.replace(tmp,path)


def archive(state):
    # Compact upload: NEVER include large saved .pkl optimizer checkpoints.
    # The user may deliberately discard them when results are reviewed.
    zpath=SCRATCH/"results_bundle.zip"
    temp=zpath.with_suffix(".zip.tmp")
    included={"metrics.csv","fixed_eval.csv","training_config.json",
              "run_meta.json","results.md","performance.json",
              "envelope_summary.json","capability_evidence.json",
              "capability_field_summary.json"}
    with zipfile.ZipFile(temp,"w",compression=zipfile.ZIP_DEFLATED) as z:
        for p in (SCRATCH/"state.json",SCRATCH/"SUMMARY.md"):
            if p.is_file():
                z.write(p,p.relative_to(SCRATCH))
        seen=set()
        for r in state["trained"].values():
            folder=Path(r["folder"])
            if folder in seen:
                continue
            seen.add(folder)
            for p in sorted(folder.rglob("*")):
                if not p.is_file():
                    continue
                if p.name in included or p.name=="prism_exam.json":
                    z.write(p,p.relative_to(RUNS))
    os.replace(temp,zpath)
    print(f"UPLOAD-READY (NO checkpoint): {zpath}",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("plan","run","status"),nargs="?",default="plan")
    p.add_argument("--hours",type=float,default=9.0,
                   help="soft wall-clock budget; never kills an active arm")
    p.add_argument("--exam-worlds",type=int,default=WORLDS,
                   help="exact predefined examiner contract: must equal 16")
    args=p.parse_args()
    if not math.isfinite(args.hours) or not 1 <= args.hours <= 24:
        p.error("--hours requires 1..24")
    if args.exam_worlds != WORLDS:
        p.error("Worlds=16 is source-pinned: change requires a fresh source/plan")
    source_sha=git("rev-parse","HEAD")
    branch=git("branch","--show-current")
    # The source-linked CLI plan is only meaningful on the actual v5047
    # branch. A single-branch WSL clone can contain a manually fetched
    # script yet remain checked out at v5046; fail BEFORE printing PASS.
    if args.mode in ("plan","run") and branch != EXPECTED_BRANCH:
        p.error(f"{args.mode} requires {EXPECTED_BRANCH}; currently "
                f"{branch or 'detached'} at {source_sha[:12]}. "
                "A --single-branch clone first needs: git remote "
                "set-branches --add origin "+EXPECTED_BRANCH+
                " ; git fetch origin ; git switch --track origin/"+EXPECTED_BRANCH)
    plan_identity=hashlib.sha256(json.dumps(
        {"source":source_sha,"parent":PARENT_SPEC,"arms":ARMS,"exams":EXAMS,
         "steps":[FOUNDATION_STEPS,FORK_TARGET,FINAL_TARGET],
         "worlds":WORLDS,"horizon":HORIZON,"seeds":SEEDS},
        sort_keys=True).encode()).hexdigest()
    planned=FOUNDATION_STEPS+len(ARMS)*(FORK_TARGET-FOUNDATION_STEPS)+(
        FINAL_TARGET-FORK_TARGET)
    print(f"EMU v5048 | source={source_sha} branch={branch}",flush=True)
    print(f"Budget: {args.hours:g}h soft, 50M shared foundation + "
          f"8 x 50M forks + winner +100M = {planned:,} unique world-steps",
          flush=True)
    print(f"Parent: 50M on seed 735 with no special Academy introduction")
    for label,spec in ARMS:
        print(f"  {label}: 50M->100M from SAME final.pkl, changes {spec}")
    print(f"Finalist: winning 100M run extends IN PLACE to 200M")
    print(f"Exam schedule: 8 protocols x 2 seeds x 8 arms, "
          "then 8 x 2 for finalist; 4 foundation baseline protocols")
    print(f"All run-owned data: {RUNS}; only experiment cache: {SCRATCH}")
    if args.mode=="plan":
        # A real CLI --plan is run for EVERY child and the original
        # parent, before even creating run directories or touching GPU.
        cases=[("foundation",PARENT_SPEC,FOUNDATION_STEPS)]+[
            (name,spec,FORK_TARGET) for name,spec in ARMS]
        for label,spec,target in cases:
            cmd=policy_args(spec,steps=target,seed=735,run=run_dir(label,735))
            check=subprocess.run(cmd+["--plan"],cwd=ROOT,text=True,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT,timeout=90)
            if check.returncode:
                print(check.stdout,flush=True)
                raise RuntimeError(f"Invalid v5048 resolved CLI config for {label}")
            resolved=json.loads(check.stdout[
                check.stdout.index("{"):check.stdout.rindex("}")+1])
            expected={
                "steps":target,"envs":2048,"seed":735,
                "solver_iterations":spec.get("solver",2),
                "solver_ls_iterations":4,
                "academy_intro_steps":spec["intro"],
                "academy_start_steps":spec["start"],
                "stillness_birth_frac":spec.get("stillness",0.),
                "science_step_promotion_mm":spec.get("step_mm",0.),
                "science_step_retreat_mm":spec.get("retreat_mm",0.),
                "science_step_min_attempts":spec.get("attempts",4),
                "temporal_vx_reward_share":spec.get("temporal",0.),
                "athletic_rehearsal_frac":spec.get("athletic",.10),
                "academy_max_share":spec.get("academy_share",.40)}
            for key,wanted in expected.items():
                got=resolved.get(key)
                if isinstance(wanted,float):
                    valid=isinstance(got,(float,int)) and abs(got-wanted)<1e-6
                else:
                    valid=(got==wanted)
                if not valid:
                    raise RuntimeError(f"{label} real CLI config {key} "
                                       f"got {got!r}, expected {wanted!r}")
            print(f"  ACTUAL --plan PASS: {label}, parent/fork "
                  f"target {target:,}, exact solver/teacher/sensor contract",
                  flush=True)
        print("STAGE ORDER: 50M foundation; eight matched 50M research forks;"
              " frozen exams; same-run winner extension +100M",flush=True)
        return 0
    if args.mode=="run":
        if git("status","--porcelain","--untracked-files=no"):
            p.error("Tracked Git source dirty: preserve immutable training source")
        if ROOT != Path.home()/"emu":
            p.error("Canonical checkout must be ~/emu, not a random folder")
    manifest=SCRATCH/"state.json"
    if args.mode=="status":
        if not manifest.is_file():
            print("No campaign has been started.");return 0
        st=json.loads(manifest.read_text())
        if st.get("source_sha")!=source_sha:
            print("WARNING: status belongs to another source SHA")
        report(st)
        print((SCRATCH/"SUMMARY.md").read_text());return 0
    SCRATCH.mkdir(parents=True,exist_ok=True)
    if manifest.is_file():
        st=json.loads(manifest.read_text())
        if st.get("fingerprint") != plan_identity:
            p.error("This campaign folder belongs to another source/plan. "
                    "Do not mix evidence: archive it and choose a fresh root.")
    else:
        st={"source_sha":source_sha,"fingerprint":plan_identity,
            "started":utc_now(),"trained":{},"exams":{},"scored":{},
            "choice":None,"smoke":"NOT_RUN","acceptance":"NOT_RUN"}
        atomic(manifest,st)
    env=dict(os.environ)
    env["PYTHONPATH"]=str(ROOT)+(
        os.pathsep+env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["EMU_UNIFIED_CONTINUOUS"]="cold-growth-approved"
    env["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
    begun=time.monotonic()
    limit=3600*args.hours

    def has_budget(min_seconds):
        return time.monotonic()-begun+min_seconds < limit

    try:
        if st["smoke"]!="PASS":
            smoke=[sys.executable,"-u",str(ROOT/"scripts/smoke_gpu_carpet.py"),
                   "--device","cuda","--worlds","2048","--frames","2",
                   "--seed","735","--ppo","--prisms","--full-update",
                   "--benchmark-updates","2","--solver-iterations","2",
                   "--solver-ls-iterations","4"]
            if process(smoke,SCRATCH/"gpu_gate.log",env):
                raise RuntimeError("CUDA 2048-world real PPO preflight failed")
            st["smoke"]="PASS";atomic(manifest,st)
        if st.get("acceptance")!="PASS":
            lifecycle_acceptance(source_sha,st,env)
        parent_key=f"foundation_s735_{FOUNDATION_STEPS}"
        if parent_key not in st["trained"] and not has_budget(900):
            print("SOFT DEADLINE before 50M foundation; nothing destroyed",
                  flush=True)
            return 0
        parent_folder=run_train("foundation",PARENT_SPEC,FOUNDATION_STEPS,
                                735,source_sha,st,env)
        parent_path=parent_folder/"checkpoints"/"final.pkl"
        parent_hash=hashlib.sha256(parent_path.read_bytes()).hexdigest()
        old_hash=st.get("parent_sha256")
        if old_hash and old_hash!=parent_hash:
            raise RuntimeError("Shared 50M parent final.pkl changed on disk")
        st["parent_sha256"]=parent_hash
        st["parent_path"]=str(parent_path)
        atomic(manifest,st)
        if not st.get("foundation_exam"):
            actor=export_actor(parent_folder,source_sha,PARENT_SPEC,
                               FOUNDATION_STEPS,st,env)
            # Four quick baselines, one seed; the main eight arms are
            # independently checked on both held-out seeds.
            basic=tuple(e for e in EXAMS if e[0] in (
                "still","cruise","step_2cm","slope_8deg"))
            get_exams("foundation",735,PARENT_SPEC,parent_folder,actor,
                      source_sha,FOUNDATION_STEPS,st,env,
                      cases=basic,exam_seeds=(SEEDS[0],))
            st["foundation_exam"]=True
            atomic(manifest,st)
            report(st);archive(st)
        for label,spec in ARMS:
            key=f"{label}_s735_{FORK_TARGET}"
            if key not in st["trained"] and not has_budget(900):
                print("SOFT DEADLINE before next 50M fork; preserved "
                      "every finished branch and 50M parent",flush=True)
                break
            folder=run_train(label,spec,FORK_TARGET,735,source_sha,st,env,
                             parent=parent_path,parent_hash=parent_hash)
            actor=export_actor(folder,source_sha,spec,FORK_TARGET,st,env)
            exams=get_exams(label,735,spec,folder,actor,source_sha,
                            FORK_TARGET,st,env)
            st["scored"][label]=score(exams)
            atomic(manifest,st);report(st);archive(st)
        if any(label not in st["scored"] for label,_ in ARMS):
            print("Incomplete paired fork bracket; NOT choosing winner.",
                  flush=True)
            report(st);archive(st)
            return 0
        if st["choice"] is None:
            st["choice"]=choose({
                label:st["scored"][label] for label,_ in ARMS})
            atomic(manifest,st)
            print(f"Best predeclared held-out physical fork: "
                  f"{st['choice']}",flush=True)
        winning_spec=dict(ARMS)[st["choice"]]
        key=f"{st['choice']}_s735_{FINAL_TARGET}"
        if key not in st["trained"] and not has_budget(1500):
            print("SOFT DEADLINE: all 8 matched 100M checkpoints retained; "
                  "winner can resume/extend with same source",flush=True)
            report(st);archive(st)
            return 0
        # The user requested +100M ON TOP of the winning 100M policy,
        # not a freshly initialised, unrelated actor. The trainer only
        # allows same-source target extensions of fully completed runs,
        # requiring identical configuration except target step count.
        folder=run_train(st["choice"],winning_spec,FINAL_TARGET,735,
                         source_sha,st,env)
        actor=export_actor(folder,source_sha,winning_spec,
                           FINAL_TARGET,st,env)
        full_exam=get_exams(st["choice"],735,winning_spec,folder,actor,
                            source_sha,FINAL_TARGET,st,env)
        st["scored"][st["choice"]+"_200M"]=score(full_exam)
        atomic(manifest,st);report(st);archive(st)
        print(f"COMPLETED TRAINING STAGES: {len(st['trained'])}; "
              f"elapsed {time.monotonic()-begun:.1f}s; "
              f"summary {SCRATCH/'SUMMARY.md'}",flush=True)
        return 0
    except BaseException as exc:
        report(st);archive(st)
        print(f"STOPPED SAFELY: {type(exc).__name__}: {exc}",flush=True)
        raise


if __name__=="__main__":
    raise SystemExit(main())
