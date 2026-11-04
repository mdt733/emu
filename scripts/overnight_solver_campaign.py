"""Sequential v5046 solver / reset / contact A/B campaign.

One GPU, one subprocess per arm, pinned source SHA, no checkpoint writes.
Run from your checkout: python scripts/overnight_solver_campaign.py
Results are checkpointed after EVERY arm. A resumed run must use the
same checkout SHA. This does not certify learned-policy locomotion.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
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
SMOKE = ROOT / "scripts" / "smoke_gpu_carpet.py"
PROBE = ROOT / "scripts" / "probe_carpet_prism_mjx.py"
SNAPSHOT_FIELDS = (
    "mean_episode_fell", "mean_support", "mean_bad_contact",
    "mean_exam_stance_slip_proxy", "mean_virtual_nonfoot_strike",
    "mean_fall_virtual_contact", "mean_vx", "mean_cmd_vx",
    "mean_servo_shaft_abs_power_w", "mean_academy_cross_new",
    "mean_terrain_cross_new",
)


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True).strip()


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def atomic_json(path: Path, value: object) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def cases(preset: str, seeds: list[int]) -> list[dict]:
    # First two controls are interleaved with the sweep. Repeat them at
    # the end to expose drifting clocks, GPU interference or heat effects.
    standard = [
        ("default", None, None, 128, False),
        ("solver4_ls4", 4, 4, 128, False),
        ("solver8_ls8", 8, 8, 128, False),
    ]
    if preset == "overnight":
        standard += [
            ("solver2_ls2", 2, 2, 128, False),
            ("solver4_ls8", 4, 8, 128, False),
            ("solver8_ls4", 8, 4, 128, False),
            ("solver4_lsdefault", 4, None, 128, False),
            ("solverdefault_ls4", None, 4, 128, False),
            ("solver2_ls4", 2, 4, 128, False),
            ("solver4_ls2", 4, 2, 128, False),
            ("solver1_ls1", 1, 1, 128, False),
            ("solver4_ls4_reset192", 4, 4, 192, False),
            ("solver4_ls4_reset256", 4, 4, 256, False),
            ("solver4_ls4_reset0", 4, 4, 0, False),
            ("solver4_ls4_torso", 4, 4, 128, True),
            ("default_repeat", None, None, 128, False),
            ("solver4_ls4_repeat", 4, 4, 128, False),
        ]
    return [
        dict(id=f"s{seed}_{label}", seed=seed, label=label,
             solver_iterations=solver, solver_ls_iterations=ls,
             reset_capacity=capacity, torso=torso)
        for seed in seeds
        for label, solver, ls, capacity, torso in standard
    ]


def command(arm: dict, updates: int) -> list[str]:
    cmd = [
        sys.executable, "-u", str(SMOKE),
        "--device", "cuda", "--worlds", "2048", "--frames", "2",
        "--seed", str(arm["seed"]), "--ppo", "--full-update",
        "--benchmark-updates", str(updates), "--prisms",
        "--reset-forward-capacity", str(arm["reset_capacity"]),
    ]
    if arm["solver_iterations"] is not None:
        cmd += ["--solver-iterations", str(arm["solver_iterations"])]
    if arm["solver_ls_iterations"] is not None:
        cmd += ["--solver-ls-iterations",
                str(arm["solver_ls_iterations"])]
    if arm["torso"]:
        cmd += ["--torso-contacts"]
    return cmd


def run_to_log(cmd: list[str], log_file: Path, env: dict) -> tuple[int, dict | None]:
    last_json = None
    with log_file.open("w", encoding="utf-8", buffering=1) as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        with subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True,
                              bufsize=1, errors="replace") as process:
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                stripped = line.strip()
                if stripped.startswith("{"):
                    try:
                        item = json.loads(stripped)
                        if item.get("status") == "PASS":
                            last_json = item
                    except (ValueError, AttributeError):
                        pass
                # Keep the interactive overnight console readable; the
                # complete raw output is always in per-arm logs.
                if (stripped.startswith((
                        "EMU SOURCE:", "CONTACT SOLVER:",
                        "SOLVER A/B:", "steady state PPO PASS:",
                        "RESET BATCH STATS:", "PHYSICAL SNAPSHOT",
                        "Traceback", "RuntimeError", "ValueError",
                        "AssertionError", "GPU memory:", "physics PASS:",
                        "surface PASS:", "reset PASS:", "recycler PASS:"))
                        or "error:" in stripped.lower()
                        or "out of memory" in stripped.lower()):
                    print("  " + stripped[:500], flush=True)
            code = process.wait()
    return code, last_json


def good(v) -> str:
    if not isinstance(v, (float, int)) or not math.isfinite(float(v)):
        return ""
    return f"{v:.5g}"


def render(state: dict, output: Path) -> None:
    completed = list(state["results"].values())
    all_rows = []
    for row in completed:
        record = {key: value for key, value in row.items()
                  if key not in ("physics_summary",)}
        record.update(row.get("physics_summary") or {})
        all_rows.append(record)
    fields = ["id", "seed", "label", "status", "seconds_wall",
              "steady_update_seconds_mean",
              "steady_world_steps_per_second",
              "contact_slots_per_world", "constraint_rows_per_world",
              "solver_iterations", "solver_ls_iterations",
              "reset_forward_capacity", "log"]
    fields += list(SNAPSHOT_FIELDS)
    with (output / "results.csv").open("w", newline="",
                                        encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    speed = {}
    for row in completed:
        fps = row.get("steady_world_steps_per_second")
        if row["status"] == "PASS" and isinstance(fps, (int, float)):
            speed.setdefault(row["label"], []).append(float(fps))
    baselines = {
        int(row["seed"]): row["steady_world_steps_per_second"]
        for row in completed if row["label"] == "default"
        and row["status"] == "PASS"
    }
    ranked = []
    for label, speeds in speed.items():
        reference = []
        for row in completed:
            if row["label"] == label and row["status"] == "PASS":
                b = baselines.get(int(row["seed"]))
                if b:
                    reference.append(row["steady_world_steps_per_second"] / b)
        ranked.append((label, statistics.median(speeds),
                       statistics.median(reference) if reference else None,
                       len(speeds)))
    ranked.sort(key=lambda x: x[1], reverse=True)
    lines = [
        "# v5046 Overnight GPU Solver Campaign",
        "",
        f"- Source SHA: `{state['source_sha']}`",
        f"- Started UTC: {state['started_utc']}",
        f"- Updated UTC: {now()}",
        f"- Status: {sum(r['status']=='PASS' for r in completed)} passed / "
        f"{len(completed)} attempted / {len(state['plan'])} planned",
        f"- Full PPO updates per arm: {state['updates']}",
        f"- Python: `{state['python']}`",
        "",
        "Each arm uses its own process on one GPU. The first update "
        "includes compilation and is **excluded** from steady timings.",
        "These are short *untrained-policy* rollouts. They do NOT establish "
        "physical parity for a competent walking policy.",
        "",
        "## Ranked steady throughput",
        "",
        "| Arm | Median world-steps/s | Median x same-seed default | Passed seeds |",
        "|---|---:|---:|---:|",
    ]
    for label, fps, ratio, count in ranked:
        lines.append(
            f"| {label} | {fps:,.0f} | "
            f"{ratio:.2f}x | {count} |" if ratio is not None else
            f"| {label} | {fps:,.0f} | — | {count} |")
    lines += [
        "", "## Physical and acceptance notes", "",
        "- See results.csv: final 32-frame mean support, falls, strikes, "
        "slip, shaft power and command tracking for each arm.",
        "- Compare fields at **matching seed**; PPO learning and reset "
        "paths may diverge between solver options.",
        "- Passing a 2-frame sphere/sidewall smoke and finite PPO does "
        "NOT validate stable learned walking or correct contact impulses.",
        "- Do not promote a fast solver cap to production until matched "
        "frozen-policy MJX physical exams and longer episodes pass.",
        "- Solver iteration and linesearch changes were benchmark-only; "
        "robot.xml and production training remain unchanged.",
        "- Retry any failed arm after investigating its per-arm .log; "
        "a failed arm is never silently counted as zero throughput.",
        "",
    ]
    atomic = output / "SUMMARY.md.tmp"
    atomic.write_text("\n".join(lines), encoding="utf-8")
    os.replace(atomic, output / "SUMMARY.md")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("overnight", "quick"),
                        default="overnight")
    parser.add_argument("--seeds", default=None,
                        help="comma-separated paired seeds; default 735,761 "
                             "for overnight or 735 for quick")
    parser.add_argument("--updates", type=int, default=None,
                        help="3..8 PPO updates per arm; first is JIT warmup")
    parser.add_argument("--output", default="~/emu_gpu_audit/v5046_solver_overnight",
                        help="persistent directory, safe to resume")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="print exact schedule without GPU work")
    args = parser.parse_args()
    seeds = ([735, 761] if args.preset == "overnight" else [735])         if args.seeds is None else [
            int(s.strip()) for s in args.seeds.split(",")]
    if not seeds or len(set(seeds)) != len(seeds) or any(
            s < 0 or s >= 2**31 for s in seeds):
        parser.error("seeds must be distinct integers 0..2147483647")
    updates = args.updates if args.updates is not None else (
        8 if args.preset == "overnight" else 3)
    if not 3 <= updates <= 8:
        parser.error("updates must be 3..8")
    if not SMOKE.exists() or not PROBE.exists():
        parser.error("Run from a complete EMU checkout")
    head = git("rev-parse", "HEAD")
    branch = git("branch", "--show-current")
    dirty = git("status", "--porcelain", "--untracked-files=no")
    if dirty:
        parser.error("tracked source files are modified; commit/stash "
                     "before benchmarking to preserve source identity")
    if not args.dry_run and not branch.startswith(("experiment/", "feature/")):
        parser.error("refusing to run the experiment on main")
    plan = cases(args.preset, seeds)
    output = Path(args.output).expanduser().resolve()
    print(f"SOURCE: {head} ({branch})", flush=True)
    print(f"PLAN: {len(plan)} sequential GPU arms, "
          f"{updates} full PPO updates/arm, seeds={seeds}", flush=True)
    print(f"OUTPUT: {output}", flush=True)
    for i, arm in enumerate(plan, 1):
        if args.dry_run:
            print(f"{i:02d}/{len(plan):02d} {arm['id']}: "
                  + " ".join(command(arm, updates)))
    if args.dry_run:
        return 0

    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    manifest = output / "state.json"
    if manifest.exists():
        state = json.loads(manifest.read_text(encoding="utf-8"))
        if (state.get("source_sha") != head or
                state.get("plan") != plan or
                state.get("updates") != updates or
                state.get("python") != sys.executable):
            parser.error("existing campaign identity differs. Choose a "
                         "new --output directory; do not mix runs")
    else:
        state = dict(source_sha=head, branch=branch, started_utc=now(),
                     python=sys.executable, updates=updates,
                     plan=plan, results={}, preflight_status="NOT_RUN")
        atomic_json(manifest, state)

    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if not args.skip_preflight and state["preflight_status"] != "PASS":
        print("PREFLIGHT: real 4-world prism/contact/recycle/reset gate",
              flush=True)
        code, _ = run_to_log(
            [sys.executable, "-u", str(PROBE), "--worlds", "4",
             "--frames", "2"], output / "preflight.log", env)
        state["preflight_status"] = "PASS" if code == 0 else "FAIL"
        atomic_json(manifest, state)
        if code != 0:
            print(f"Preflight failed; inspect {output/'preflight.log'}; "
                  "GPU sweep NOT started.", flush=True)
            return 2
    if args.skip_preflight:
        print("WARNING: preflight skipped by operator", flush=True)

    failed_in_a_row = 0
    for i, arm in enumerate(plan, 1):
        key = arm["id"]
        if (state["results"].get(key) or {}).get("status") == "PASS":
            print(f"[{i}/{len(plan)}] resume skip {key}", flush=True)
            continue
        print(f"[{i}/{len(plan)}] RUN {key}", flush=True)
        start = time.monotonic()
        filename = "logs/" + key + ".log"
        code, result = run_to_log(command(arm, updates),
                                  output / filename, env)
        record = dict(id=key, label=arm["label"], seed=arm["seed"],
                      status="PASS" if code == 0 and
                      result is not None else "FAIL",
                      seconds_wall=round(time.monotonic() - start, 2),
                      log=filename, returncode=code)
        if result is not None:
            record.update(result)
        state["results"][key] = record
        atomic_json(manifest, state)
        render(state, output)
        print(f"[{i}/{len(plan)}] {record['status']}: "
              f"{good(record.get('steady_world_steps_per_second'))} "
              f"steps/s, elapsed {record['seconds_wall']:.1f}s; "
              f"summary={output/'SUMMARY.md'}", flush=True)
        if record["status"] == "PASS":
            failed_in_a_row = 0
        else:
            failed_in_a_row += 1
            if failed_in_a_row >= 3:
                print("STOP: 3 consecutive arm failures; investigate logs. "
                      "Completed arms are saved; rerun to resume.", flush=True)
                return 2
    passes = sum(r["status"] == "PASS"
                 for r in state["results"].values())
    # Leave one easy-to-upload archive as well as the resumable raw data.
    archive = output / "overnight_results.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6) as bundle:
        for file in sorted(output.rglob("*")):
            if file.is_file() and file != archive and not file.name.endswith(".tmp"):
                bundle.write(file, file.relative_to(output))
    print(f"COMPLETE: {passes}/{len(plan)} passed; "
          f"results: {output/'SUMMARY.md'}, {output/'results.csv'} "
          f"and upload-ready {archive}", flush=True)
    return 0 if passes == len(plan) else 1


if __name__ == "__main__":
    raise SystemExit(main())
