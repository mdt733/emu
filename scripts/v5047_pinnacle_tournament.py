#!/usr/bin/env python3
"""v5047 disposable tournament runner: 7 controlled 100M arms, then a
300M finalist and optional 200M independent-seed replication.

One real trainer, no second PPO, no separate Academy implementation.
One source SHA and 9-hour budget, but NO killing of running training.
All trained data in ~/emu-data/runs; only ephemeral runner state in scratch.
Delete this script after results are recorded in HANDOVER / issue #17.
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
SCRATCH = Path.home() / "emu-data" / "scratch" / "v5047-pinnacle-tournament"
EXPECTED_BRANCH = "experiment/v5047-pinnacle-stairs-gaits"
EXPLORER_STEPS = 100_000_000
FINAL_STEPS = 300_000_000
REPLICA_STEPS = 200_000_000
SEEDS = (1701, 1702)
WORLDS = 16
HORIZON = 500
PHYSICS = {"solver": 2, "ls": 4}
# Only one hypothesis changes relative to early_lips in C/D/E/F/G.
ARMS = (
    ("no_academy", {"intro": 600_000_000, "start": 800_000_000}),
    ("early_lips", {"intro": 11_000_000, "start": 16_000_000}),
    ("precise_frontier", {"intro": 11_000_000, "start": 16_000_000,
                          "step_mm": 1, "retreat_mm": 1, "attempts": 12}),
    ("zero_balance", {"intro": 11_000_000, "start": 16_000_000,
                       "stillness": .10}),
    ("temporal_speed", {"intro": 11_000_000, "start": 16_000_000,
                         "temporal": .50}),
    ("fast_rehearsal", {"intro": 11_000_000, "start": 16_000_000,
                         "athletic": .30}),
    ("solver4_control", {"intro": 11_000_000, "start": 16_000_000,
                         "solver": 4}),
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


def policy_args(spec, *, steps, seed, run):
    # Same exact source-pinned PPO/world/teacher/reward recipe for each arm
    # except the named study switch. 2/4 is research physics, not a
    # replacement for robot.xml. CLI resolves and validates all flags.
    solver = spec.get("solver", PHYSICS["solver"])
    opts = [
        "--run-dir", str(run), "--steps", str(steps),
        "--envs", "2048", "--seed", str(seed), "--device", "cuda",
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
        "--academy-ramp-steps", "12000000",
        "--academy-max-share", ".40", "--academy-height-cap-m", ".15",
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
    return RUNS / f"v5047-{label}-s{seed}"


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


def checked_actor(actor, sha, spec, step):
    data = json.loads(actor.read_text(encoding="utf-8"))
    source = data.get("source", {})
    physics = data.get("training", {})
    if (source.get("git_commit") != sha or source.get("step") < step or
            physics.get("terrain_carpet_prisms") is not True or
            physics.get("solver_iterations") != spec.get("solver", 2) or
            physics.get("solver_ls_iterations") != 4):
        raise RuntimeError(
            f"Export does not match source, target step or solver: {actor}")
    return data


def run_train(label, spec, steps, seed, sha, state, env):
    key = f"{label}_s{seed}_{steps}"
    folder = run_dir(label, seed)
    checkpoint = folder / "checkpoints" / "final.pkl"
    if key in state["trained"]:
        if not checkpoint.is_file():
            raise RuntimeError(f"manifest said trained but checkpoint missing: {checkpoint}")
        return folder
    folder.parent.mkdir(parents=True, exist_ok=True)
    base = policy_args(spec, steps=steps, seed=seed, run=folder)
    if folder.exists() and any(folder.iterdir()):
        if not (folder / "training_config.json").is_file():
            raise RuntimeError(
                f"Nonempty run dir without validated config: {folder}")
        print(f"RESUME {key} from existing same-SHA checkpoints", flush=True)
        cmd = base + ["--resume", "auto"]
    else:
        print(f"NEW COLD-START {key}", flush=True)
        cmd = base
    begun = time.monotonic()
    result = process(cmd, SCRATCH / "logs" / (key+".log"), env)
    if result:
        raise RuntimeError(f"Train failed {key} (code {result}); partial run kept")
    if not checkpoint.is_file():
        raise RuntimeError(f"No final checkpoint after training: {checkpoint}")
    config = json.loads((folder / "training_config.json").read_text())
    provenance = json.loads((folder / "run_meta.json").read_text())
    got = (provenance.get("source_provenance") or {}).get("git_commit")
    if got != sha or config.get("steps") != steps or config.get("seed") != seed:
        raise RuntimeError(f"Wrong physical lineage in {folder}: {got}")
    metrics = read_last_csv(folder / "metrics.csv")
    state["trained"][key] = {
        "folder": str(folder), "seconds": round(time.monotonic()-begun,2),
        "step": int(float(metrics["step"])),
        "fall_rate": float(metrics["fall_rate"]),
        "vx": float(metrics["vx_mean"]),
        "academy_up": float(metrics.get("academy_up_crossings") or 0.),
        "academy_down": float(metrics.get("academy_down_crossings") or 0.),
        "academy_attempts_up": float(metrics.get("academy_near_attempts_up") or 0.),
        "academy_attempts_down": float(metrics.get("academy_near_attempts_down") or 0.),
        "training_fps": float(metrics.get("train_steps_per_sec") or 0.),
        "config_sha": hashlib.sha256((folder / "training_config.json").read_bytes()).hexdigest(),
    }
    atomic(SCRATCH / "state.json", state)
    return folder


def export_actor(folder, sha, spec, steps, state, env):
    actor = folder / "exports" / "final.emupolicy"
    key = folder.name
    if not actor.exists():
        actor.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-u", "-m", "emu.cli", "export-policy",
               str(folder / "checkpoints" / "final.pkl"), str(actor)]
        if process(cmd, SCRATCH / "logs" / (key+"_export.log"), env,echo=False):
            raise RuntimeError(f"Export failed: {folder}")
    checked_actor(actor, sha, spec, steps)
    return actor


def get_exams(label, seed, spec, folder, actor, sha, steps, state, env):
    key = f"{label}_s{seed}_{steps}"
    outputs = []
    for exam_seed in SEEDS:
        for name, mode, speed, grade, lip in EXAMS:
            exam_id = f"{key}__{name}__s{exam_seed}"
            out = folder / "exams" / f"{name}_s{exam_seed}"
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
                    data.get("source_step") < steps or
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
            "step_worlds":0,"flat_survival":[], "step_survival":[],
            "slope_survival":[], "zero_drift":[], "slow_error":[],
            "fast_vx":[], "fast_stride":[], "fast_footfall":[],
            "fast_yaw_bias":[], "cruise_error":[]}
    for name, seed, item in exams:
        rows = item["worlds"]
        if name.startswith("step_"):
            vals["supported_up"]+=item["measured"]["signed_supported_up"]
            vals["supported_down"]+=item["measured"]["signed_supported_down"]
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
    out["research_score"]=round(
        8*min(12,out["supported_up"]+out["supported_down"])
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
        "# v5047 nine-hour Pinnacle research tournament", "",
        f"Source SHA: `{state['source_sha']}`",
        f"Started: {state['started']}; last update: {utc_now()}",
        f"Training jobs: {len(state['trained'])}; frozen exam panels: {len(state['exams'])}",
        f"Total nominal full PPO steps planned: {(len(ARMS)*EXPLORER_STEPS+FINAL_STEPS+REPLICA_STEPS):,}",
        "",
        "One actor per arm. All training is real 2048-world PPO; no policy is",
        "promoted to `main` by this script. Scores only use independent",
        "raw-physical frozen exams with matched held-out seeds, never PPO",
        "training reward or Academy's internal success flags.",
        "",
        "| Arm | True supported up/down | Flat survival | Step survival | 0cmd drift m | Fast vx m/s | Fast stride m | Footfall Hz | Score | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    fmt=lambda x:"—" if x is None else f"{x:.3f}"
    for label, s in scored.items():
        lines.append(
            f"| {label} | {s['supported_up']}/{s['supported_down']} "
            f"| {fmt(s['flat_survival'])} | {fmt(s['step_survival'])} "
            f"| {fmt(s['zero_drift'])} | {fmt(s['fast_vx'])} "
            f"| {fmt(s['fast_stride'])} | {fmt(s['fast_footfall'])} "
            f"| {s['research_score']:.2f} | {'PASS' if s['qualified'] else 'FAIL'} |")
    lines += [
        "", f"Selected next research recipe: **{state.get('choice') or 'not selected'}**",
        "",
        "## Interpretation gates", "",
        "- Zero observed signed lips is zero demonstrated true steps, NOT proof",
        "  that the policy ever approached a sufficiently high edge.",
        "- Mean step/slope survival alone is not evidence of locomotion.",
        "- Footfall/stride diagnostics are measured from raw foot top contacts",
        "  and heading; high cadence alone is not dynamic fast running.",
        "- Claim main parity ONLY after source-matched incumbent physical",
        "  missions, reproducible steering/speed and contact-fidelity review.",
        "- 2/4 versus 4/4 is a real trained-policy control; CUDA reset .qacc",
        "  compact/full parity remains a separate unresolved physical caveat.",
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
        for r in state["trained"].values():
            folder=Path(r["folder"])
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
    plan_identity=hashlib.sha256(json.dumps(
        {"source":source_sha,"arms":ARMS,"exams":EXAMS,
         "steps":[EXPLORER_STEPS,FINAL_STEPS,REPLICA_STEPS],
         "worlds":WORLDS,"horizon":HORIZON,"seeds":SEEDS},
        sort_keys=True).encode()).hexdigest()
    print(f"EMU v5047 | source={source_sha} branch={branch}",flush=True)
    print(f"Budget: {args.hours:g}h (soft), seven 100M arms + 300M + 200M "
          f"= {(len(ARMS)*EXPLORER_STEPS+FINAL_STEPS+REPLICA_STEPS):,} world-steps",
          flush=True)
    for label,spec in ARMS:
        print(f"  {label}: {spec}",flush=True)
        if args.mode=="plan":
            cmd=policy_args(spec,steps=EXPLORER_STEPS,
                            seed=735,run=run_dir(label,735))+["--plan"]
            # Actual CLI resolves the real TrainerConfig contract before
            # CUDA work. Catch unsupported flags / wrong active teachers
            # NOW, not 12 minutes into the overnight campaign.
            probe=subprocess.run(cmd,cwd=ROOT,text=True,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT,timeout=60)
            if probe.returncode:
                print(probe.stdout,flush=True)
                raise RuntimeError(f"Preflight CLI recipe invalid for {label}")
            config=json.loads(probe.stdout[
                probe.stdout.index("{"):probe.stdout.rindex("}")+1])
            if (config.get("steps")!=EXPLORER_STEPS or config.get("envs")!=2048
                or config.get("solver_iterations")!=spec.get("solver",2)
                or config.get("solver_ls_iterations")!=4
                or abs(config.get("stillness_birth_frac",0.)
                       -spec.get("stillness",0.))>1e-7
                or config.get("academy_intro_steps")!=spec["intro"]):
                raise RuntimeError(f"Resolved CLI config differs from {label} plan")
            print(f"    real CLI config PASS; run-dir {run_dir(label,735)}")
    print(f"Exam schedule: {len(EXAMS)} modes x {len(SEEDS)} seeds x up to 9 policies")
    print(f"All run-owned data: {RUNS}; only experiment cache: {SCRATCH}")
    if args.mode=="plan":
        return 0
    if args.mode=="run":
        if branch != EXPECTED_BRANCH:
            p.error(f"Must run from {EXPECTED_BRANCH}, not {branch or 'detached'}")
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
            "choice":None,"smoke":"NOT_RUN"}
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
        for label,spec in ARMS:
            key=f"{label}_s735_{EXPLORER_STEPS}"
            if key not in st["trained"] and not has_budget(1800):
                print("SOFT DEADLINE before next training arm; resume later.",flush=True)
                break
            folder=run_train(label,spec,EXPLORER_STEPS,735,source_sha,st,env)
            actor=export_actor(folder,source_sha,spec,EXPLORER_STEPS,st,env)
            exams=get_exams(label,735,spec,folder,actor,source_sha,
                            EXPLORER_STEPS,st,env)
            st["scored"][label]=score(exams)
            atomic(manifest,st);report(st);archive(st)
        if len(st["scored"]) != len(ARMS):
            print("Incomplete matched bracket; NOT selecting a winner.",flush=True)
            return 0
        if st["choice"] is None:
            st["choice"]=choose(st["scored"])
            atomic(manifest,st)
            print(f"Predeclared held-out physical choice: {st['choice']}",flush=True)
        winning_spec=dict(dict(ARMS)[st["choice"]])
        for label,seed,steps in (("candidate",735,FINAL_STEPS),
                                 ("replica",761,REPLICA_STEPS)):
            name=f"{label}-{st['choice']}"
            key=f"{name}_s{seed}_{steps}"
            if key not in st["trained"] and not has_budget(3600):
                print("SOFT DEADLINE before next optional finalist run; "
                      "preserving all results.",flush=True)
                break
            folder=run_train(name,winning_spec,steps,seed,source_sha,st,env)
            actor=export_actor(folder,source_sha,winning_spec,steps,st,env)
            result=get_exams(name,seed,winning_spec,folder,actor,source_sha,
                             steps,st,env)
            st["scored"][name]=score(result)
            atomic(manifest,st);report(st);archive(st)
        print(f"COMPLETED AVAILABLE ARMS: {len(st['trained'])}; "
              f"elapsed {time.monotonic()-begun:.1f}s; "
              f"summary {SCRATCH/'SUMMARY.md'}",flush=True)
        return 0
    except BaseException as exc:
        report(st);archive(st)
        print(f"STOPPED SAFELY: {type(exc).__name__}: {exc}",flush=True)
        raise


if __name__=="__main__":
    raise SystemExit(main())
