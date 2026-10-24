#!/usr/bin/env python3
"""Ten-hour *sequential* EMU campaign; no source mutation, no champion promotion.

Run from a CLEAN, frozen NEW v5023 feature checkout under WSL. Requires the
separate OLD source-pinned 411b841 checkout and its completed 34M checkpoint.

All stages are separate run directories because config.steps is an exact
checkpoint contract. No concurrent 2048-world CUDA trainers are ever launched.
An entire stage may overrun the nominal deadline; never terminate a stage
mid-PPO update just to satisfy the wall-clock estimate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

OLD_SHA = "411b841de08ebfbb65c29b357c8ee0d39d577dd1"
SEED = 733
ENVS = 2048
SPEEDS = (0.15, 0.25, 0.30)
EXAM_SEEDS = (1701, 1702, 2417)


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args],
                                   text=True).strip()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_root(root: Path, old: Path, new: Path, *, dry: bool) -> dict:
    for name, src in (("control", old), ("experiment", new)):
        if not (src / ".venv/bin/emu").is_file():
            raise SystemExit(f"{name} checkout missing .venv/bin/emu: {src}")
        if git(src, "status", "--porcelain"):
            raise SystemExit(f"{name} checkout has uncommitted changes: {src}")
    if git(old, "rev-parse", "HEAD") != OLD_SHA:
        raise SystemExit(f"Control source must be pinned to {OLD_SHA}")
    new_sha = git(new, "rev-parse", "HEAD")
    if new_sha == OLD_SHA:
        raise SystemExit("Experiment checkout must contain committed G6 teaching changes")
    old_ck = Path.home() / "robot_lab/emu_runs/v5023_window_fresh733/growth34/checkpoints/final.pkl"
    if not old_ck.is_file():
        raise SystemExit(f"Missing original 34M parent: {old_ck}")
    identity = dict(control_source=OLD_SHA, experiment_source=new_sha,
                    control_parent=str(old_ck), control_parent_sha256=sha_file(old_ck),
                    seed=SEED, worlds=ENVS)
    if dry:
        return identity
    root.mkdir(parents=True, exist_ok=True)
    path = root / "campaign_identity.json"
    if path.exists():
        if json.loads(path.read_text()) != identity:
            raise SystemExit("Campaign provenance changed: refuse cross-source/cross-parent resume")
    else:
        path.write_text(json.dumps(identity, indent=2) + "\n")
    return identity


def run_cmd(args: list[str], *, dry: bool):
    print("$ " + " ".join(args), flush=True)
    if not dry:
        subprocess.run(args, check=True)


def stage(root: Path, src: Path, name: str, *, parent: Path | None,
          target: int, old: bool, arm: str = "off",
          remaining: float, dry: bool) -> bool:
    out = root / name
    final = out / "checkpoints/final.pkl"
    if final.is_file():
        print(f"[done] {name}: {final}", flush=True)
        return True
    if parent is not None and not parent.is_file() and not dry:
        raise SystemExit(f"{name}: missing parent {parent}")
    prev_step = 0 if parent is None else (34_013_184 if name == "control60" else
        60_000_000 if name == "control110" else
        20_000_000 if name in ("A60", "B60") else
        60_000_000 if name in ("A100", "B100") else
        100_000_000 if name.endswith("160") or name.endswith("180") else
        160_000_000 if name.endswith("260") else 260_000_000)
    estimate = max(0, target - prev_step) / 24_000 + 600
    if not dry and remaining < estimate + 2700:
        print(f"[budget] skip {name}: {remaining/3600:.2f}h left, estimated "
              f"{estimate/3600:.2f}h + 45min reserve", flush=True)
        return False
    exe = str(src / ".venv/bin/emu")
    args = [exe, "train", "--run-dir", str(out), "--profile",
            "sim2real" if parent else "foundation", "--steps", str(target),
            "--envs", str(ENVS), "--seed", str(SEED), "--device", "cuda",
            "--pitch-deg", "8", "--terrain-actor", "--terrain-step-max-m", ".04"]
    if not old:
        args.extend(["--terrain-training-arm", arm])
    if (out / "checkpoints/latest.pkl").exists():
        args.extend(["--resume", "auto"])
    elif parent:
        args.extend(["--resume", str(parent)])
    run_cmd(args, dry=dry)
    if not dry and not final.is_file():
        raise RuntimeError(f"{name}: command returned but no validated final checkpoint")
    return True


def read_exam(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    record = dict(up=0, down=0, up_trials=0, down_trials=0,
                  falls=0, flat_metres=0.0, flat_joules=0.0, flat_worlds=0)
    for case in data["cases"]:
        if case["case"] == "original_flat_physics":
            for w in case["worlds"]:
                record["flat_metres"] += w["progress_m"]
                record["flat_joules"] += w["drive_shaft_work_j"]
                record["flat_worlds"] += 1
        elif case["case"].startswith("terrain_topology_") and case["max_height_cm"] > 0:
            for w in case["worlds"]:
                direction = w["lesson"]
                if direction not in ("step-up", "step-down"):
                    continue
                key = "up" if direction == "step-up" else "down"
                record[key + "_trials"] += 1
                record[key] += bool(w["crossing_pass"])
                record["falls"] += bool(w["fell"])
    return record


def exams(root: Path, src: Path, run_name: str, *, dry: bool) -> dict | None:
    ck = root / run_name / "checkpoints/final.pkl"
    policy = root / "exam_policies" / f"{run_name}.emupolicy"
    if not dry:
        policy.parent.mkdir(exist_ok=True)
    if not policy.exists():
        run_cmd([str(src / ".venv/bin/emu"), "export-policy", str(ck), str(policy)], dry=dry)
    results = []
    for seed in EXAM_SEEDS:
        for speed in SPEEDS:
            label = f"{int(speed*100):02d}"
            path = root / "exams" / run_name / f"seed{seed}_vx{label}"
            report = path / "terrain_exam.json"
            if not report.exists():
                run_cmd([str(src / ".venv/bin/emu"), "terrain-exam",
                         "--policy", str(policy), "--output", str(path),
                         "--device", "cpu", "--worlds", "8", "--horizon", "350",
                         "--speed", str(speed), "--heights-cm", "0", "1", "2", "4",
                         "--seed", str(seed)], dry=dry)
            if not dry:
                results.append(read_exam(report))
    if dry:
        return None
    return {k: sum(r[k] for r in results) for k in results[0]}


def ranking(root: Path, candidate_results: dict[str, dict]) -> list[str]:
    def score(r):
        # Crossing quality matters more than slight raw forward progress.
        # Penalize falls and shaft-work/m; NEVER promote a winner automatically.
        metres = max(0.1, r["flat_metres"])
        jpm = r["flat_joules"] / metres
        return (5.0*r["up"] + r["down"] - 2.0*r["falls"]
                - max(0.0, jpm - 70.0)*0.15)
    ranks = sorted(("A100", "B100"), key=lambda n: score(candidate_results[n]),
                   reverse=True)
    control = candidate_results["control110"]
    good = []
    for name in ranks:
        r = candidate_results[name]
        flat_progress = r["flat_metres"] / max(.001, control["flat_metres"])
        effort_ratio = ((r["flat_joules"] / max(.001,r["flat_metres"])) /
                        (control["flat_joules"] / max(.001,control["flat_metres"])))
        qualified = (r["up"] > control["up"] and r["falls"] <= control["falls"]+6
                     and flat_progress >= .85 and effort_ratio <= 1.35)
        print(f"[A/B] {name}: up={r['up']} (control {control['up']}), "
              f"falls={r['falls']} (control {control['falls']}), "
              f"flat progress ratio={flat_progress:.3f}, "
              f"shaft J/m ratio={effort_ratio:.3f}, qualified={qualified}",flush=True)
        if qualified:
            good.append(name)
    print("[A/B] paired frozen exams", json.dumps(candidate_results, indent=2), flush=True)
    (root / "ab_exam_scores.json").write_text(json.dumps(
        {"summaries": candidate_results, "rank": ranks, "eligible": good,
         "rule": "5*up + down - 2*falls - 0.15*max(0,shaftJ/m - 70)",
         "qualification": "up > control, falls <= control+6, flat progress >=85%, flat shaft J/m <=135%"}, indent=2)+"\n")
    return good


def main():
    raise SystemExit(
        "STOP: v5023 control/A/B-only campaign has been retired by owner. "
        "It deliberately excludes concurrent real slope+step learning. "
        "Do not run this launcher; see docs/UNIFIED_CAPABILITY_GROWTH.md. "
        "No checkpoints were changed."
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--control-repo",type=Path,default=Path.home()/"emu-v5023-window")
    p.add_argument("--experiment-repo",type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument("--root",type=Path,default=Path.home()/"robot_lab/emu_runs/v5023_overnight733")
    p.add_argument("--hours",type=float,default=10.0)
    p.add_argument("--dry-run",action="store_true")
    args=p.parse_args()
    root=args.root.expanduser().resolve()
    old=args.control_repo.expanduser().resolve()
    new=args.experiment_repo.expanduser().resolve()
    identity=audit_root(root,old,new,dry=args.dry_run)
    print("[campaign] identity",json.dumps(identity,indent=2),flush=True)
    begin=time.monotonic()
    def remaining(): return args.hours*3600 - (time.monotonic()-begin)
    old_parent=Path(identity["control_parent"])
    path=lambda name:root/name/"checkpoints/final.pkl"
    seq=[
      ("control60",old,old_parent,60_000_000,True,"off"),
      ("control110",old,path("control60"),110_000_000,True,"off"),
      ("foundation20",new,None,20_000_000,False,"off"),
      ("A60",new,path("foundation20"),60_000_000,False,"speed"),
      ("A100",new,path("A60"),100_000_000,False,"speed"),
      ("B60",new,path("foundation20"),60_000_000,False,"foothold"),
      ("B100",new,path("B60"),100_000_000,False,"foothold")
    ]
    for name,src,parent,target,is_old,arm in seq:
        if not stage(root,src,name,parent=parent,target=target,old=is_old,arm=arm,
                     remaining=remaining(),dry=args.dry_run):
            print("[budget] stopping after last completed stage",flush=True)
            return
    if args.dry_run:
        print("[dry run] after A/B: frozen CPU exams, selection, winner/runner-up extensions")
        return
    results = {name:exams(root,src,name,dry=False) for name,src in
               (("control110",old),("A100",new),("B100",new))}
    ranks=ranking(root,results)
    if not ranks:
        print("[A/B] neither teaching arm outperformed the control safely. "
              "Preserving budget rather than blindly running another 280M.",flush=True)
        return
    winner=ranks[0]
    runner=ranks[1] if len(ranks)>1 else None
    # The next blocks are optional, and cannot run unless the A/B exams exist.
    # Selection is for experiment allocation only, NOT champion promotion.
    steps=[
        (winner+"160",winner,160_000_000, "foothold" if winner.startswith("B") else "speed"),
        (winner+"260",winner+"160",260_000_000,"foothold" if winner.startswith("B") else "speed"),
        (winner+"380",winner+"260",380_000_000,"foothold" if winner.startswith("B") else "speed")
    ]
    if runner:
        steps.insert(2,(runner+"180",runner,180_000_000,
                        "foothold" if runner.startswith("B") else "speed"))
    for name,source,target,arm in steps:
        if not stage(root,new,name,parent=path(source),target=target,old=False,arm=arm,
                     remaining=remaining(),dry=False):
            break
    print(f"[campaign] finished with {remaining()/3600:.2f}h remaining")
    print(f"[campaign] source hashes and controls: {root/'campaign_identity.json'}")
    print(f"[campaign] all checkpoints remain under {root}; no promotion performed")

if __name__=="__main__":
    main()
