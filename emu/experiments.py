#!/usr/bin/env python3
"""Small experiment runner for modern EMU.

EMU deliberately keeps experiment orchestration boring:

  random weights -> 20M foundation target -> one exact update-boundary fork -> shadow / graph

Historical ladders, imported phase continuations and automated winner tournaments live
in Git history, not in the current runtime.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .paths import VERSION, run_base, results_base, source_repo_root
from .training_config import FOUNDATION_STEPS, ROLLOUT_STEPS, TrainingConfig, foundation_config, mature_config
from .checkpoints import find_resume, find_valid_resume
from .artifacts import read_csv_rows
from .checkpoint_io import load_checkpoint

LOG = f"[emu_{VERSION}]"


def update_boundary_at_or_after(target_steps: int, envs: int) -> int:
    """First complete PPO-update boundary at or beyond an environment-step target."""
    batch = int(envs) * int(ROLLOUT_STEPS)
    if batch <= 0:
        raise ValueError(f"invalid PPO batch size: envs={envs} rollout={ROLLOUT_STEPS}")
    target = int(target_steps)
    return ((target + batch - 1) // batch) * batch


def foundation_handoff_step(envs: int) -> int:
    return update_boundary_at_or_after(FOUNDATION_STEPS, envs)


def checkpoint_step(path: Path) -> int:
    return int(load_checkpoint(path, validate=True)['meta']['step'])


def promote_checkpoint(src: Path, dst: Path) -> Path:
    """Give an exact validated handoff checkpoint its canonical final.pkl name."""
    if src.resolve() == dst.resolve():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    return dst


def resolve_target_checkpoint(run_dir: Path, *, target_steps: int, envs: int, label: str) -> tuple[Path | None, Path | None, int]:
    """Return (complete_final, resumable_partial, expected_update_boundary).

    Completion and resumability are deliberately separate concepts. A partial
    latest/snapshot checkpoint may continue a run, but only a current-schema
    checkpoint on the exact target update boundary counts as complete.
    """
    expected = update_boundary_at_or_after(target_steps, envs)
    raw_resume = find_resume(run_dir)
    resume = find_valid_resume(run_dir)
    if raw_resume is not None and resume is None:
        raise RuntimeError(
            f"{label} has checkpoints but none are valid current-schema rewind points under {run_dir/'checkpoints'}"
        )
    if resume is None:
        return None, None, expected
    step = checkpoint_step(resume)
    if step > expected:
        raise RuntimeError(
            f"{label} checkpoint is beyond the expected target boundary: step={step:,} expected={expected:,}; "
            "use the original env-count/config or start a clean run"
        )
    if step == expected:
        final = promote_checkpoint(resume, run_dir / 'checkpoints' / 'final.pkl')
        return final, None, expected
    return None, resume, expected


def resolve_foundation_handoff(foundation: Path, *, envs: int) -> tuple[Path | None, Path | None, int]:
    """Foundation-specific name for the shared-ancestor completion contract."""
    return resolve_target_checkpoint(
        foundation, target_steps=FOUNDATION_STEPS, envs=envs, label='foundation'
    )


def git_provenance() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=source_repo_root(), text=True, stderr=subprocess.DEVNULL).strip()
    try:
        status = git("status", "--porcelain")
        return {
            "available": True,
            "commit": git("rev-parse", "HEAD"),
            "branch": git("branch", "--show-current"),
            "describe": git("describe", "--always", "--dirty"),
            "dirty": bool(status),
        }
    except Exception:
        return {"available": False, "commit": "", "branch": "", "describe": "", "dirty": False}


def number(row: dict[str, Any], key: str, default: float = -1.0) -> float:
    try:
        value = float(row.get(key, default))
        return value if value == value else default
    except (TypeError, ValueError):
        return default


def summarize(run_dir: Path) -> dict[str, Any]:
    rows = read_csv_rows(run_dir / "metrics.csv")
    evals = read_csv_rows(run_dir / "fixed_eval.csv")
    cloud = read_csv_rows(run_dir / "heldout_capability_cloud.csv")
    if not rows:
        return {"final_step": -1, "ok": False}
    last = rows[-1]
    ev = evals[-1] if evals else {}
    final_cloud = [r for r in cloud if str(r.get("forced_final", "0")).lower() in {"1", "true", "yes"}]
    controls = [r for r in final_cloud if r.get("family") == "none"]
    disturbed = [r for r in final_cloud if r.get("family") != "none" and str(r.get("resilience_valid", "0")).lower() in {"1", "true", "yes"}]

    def mean(key: str, seq: list[dict[str, str]], default: float = -1.0) -> float:
        vals = [number(r, key, float("nan")) for r in seq]
        vals = [x for x in vals if x == x]
        return sum(vals) / len(vals) if vals else default

    pairs = [(number(r, "field_predicted_competence", float("nan")), number(r, "node_window_rate", float("nan"))) for r in controls]
    pairs = [(a, b) for a, b in pairs if a == a and b == b]
    try:
        field = json.loads((run_dir / "capability_field_summary.json").read_text(encoding="utf-8"))
    except Exception:
        field = {}
    return {
        "final_step": int(number(last, "step", -1)),
        "fall_rate": number(last, "fall_rate"),
        "ep_len": number(last, "ep_len_mean"),
        "broad": number(ev, "eval_broad_score"),
        "dynamic80_survival": number(ev, "eval_dynamic80_survival"),
        "dynamic80_vx_mae": number(ev, "eval_dynamic80_vx_mae"),
        "combo70_vy_mae": number(ev, "eval_combo70_vy_mae"),
        "recovery": mean("recovery_rate", disturbed),
        "worst_direction": mean("recovery_worst_direction", disturbed),
        "calibration_mae": sum(abs(a-b) for a,b in pairs)/len(pairs) if pairs else -1.0,
        "known_nodes": field.get("field_known_nodes", -1),
        "competent_nodes": field.get("field_competent_nodes", -1),
        "known_edges": field.get("field_known_edges", -1),
        "competent_edges": field.get("field_competent_edges", -1),
        "frontier_vx_hi": number(last, "frontier_vx_hi"),
        "frontier_vx_lo": number(last, "frontier_vx_lo"),
    }


def run_training(config: TrainingConfig, *, provenance: dict[str, Any]) -> None:
    run_dir = Path(config.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "training_config.json"
    config.save(config_path)
    cmd = [sys.executable, "-c", "from emu.training import main; raise SystemExit(main())", "--config", str(config_path)]
    env = dict(os.environ)
    env.update({
        "EMU_SOURCE_VERSION": VERSION,
        "EMU_GIT_COMMIT": str(provenance.get("commit", "")),
        "EMU_GIT_BRANCH": str(provenance.get("branch", "")),
        "EMU_GIT_DESCRIBE": str(provenance.get("describe", "")),
        "EMU_GIT_DIRTY": "true" if provenance.get("dirty") else "false",
    })
    with (run_dir / "console.log").open("a", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line); log.flush()
            t = line.strip()
            if t.startswith(LOG + " step=") or "fixed-eval step=" in t or "AUTO-RECOVERY" in t or "training complete" in t or "Traceback" in t:
                print("    " + t[:260], flush=True)
        rc = proc.wait()
    if rc:
        raise RuntimeError(f"training failed for {run_dir.name}; see {run_dir/'console.log'}")


def write_report(root: Path, results_dir: Path, meta: dict[str, Any]) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    rows = meta["runs"]
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["name", "seed", "mode", "seconds", *next(iter(rows.values()))["summary"].keys()]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for name, item in rows.items():
            w.writerow({"name": name, "seed": item["seed"], "mode": item["mode"], "seconds": item["seconds"], **item["summary"]})
    (root / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    lines = [f"# EMU {VERSION} experiment", "", f"Git: `{meta['git'].get('commit') or 'unavailable'}`", "",
             "| run | seed | mode | broad | recovery | worst-dir | cal MAE | nodes | edges | vx+ | vx- | seconds |",
             "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, item in rows.items():
        s=item["summary"]
        lines.append(f"| {name} | {item['seed']} | {item['mode']} | {s.get('broad',-1):.3f} | {s.get('recovery',-1):.3f} | {s.get('worst_direction',-1):.3f} | {s.get('calibration_mae',-1):.3f} | {s.get('competent_nodes',-1)}/{s.get('known_nodes',-1)} | {s.get('competent_edges',-1)}/{s.get('known_edges',-1)} | {s.get('frontier_vx_hi',-1):.3f} | {s.get('frontier_vx_lo',-1):.3f} | {item['seconds']:.1f} |")
    lines += ["", "No automated winner is declared. Compare paired shadow-vs-graph deltas directly.", ""]
    report = "\n".join(lines)
    (results_dir / "results.md").write_text(report, encoding="utf-8")
    shutil.copy2(root / "summary.csv", results_dir / "summary.csv")
    shutil.copy2(root / "run_meta.json", results_dir / "run_meta.json")


def run_seed(seed: int, *, target_steps: int, modes: tuple[str, ...], device: str,
             run_root: Path, results_dir: Path, envs: int) -> None:
    provenance = git_provenance()
    root = run_root.expanduser().resolve(); root.mkdir(parents=True, exist_ok=True)
    repo = source_repo_root()
    if repo is not None and (repo == root or repo in root.parents):
        raise SystemExit(f"raw run root must be outside the source repository: {root}")
    foundation = root / "foundation20"
    foundation_ck, foundation_resume, handoff_step = resolve_foundation_handoff(foundation, envs=envs)
    if foundation_ck is None:
        source = f"resume {foundation_resume}" if foundation_resume is not None else "random weights"
        print(f"{LOG} foundation seed={seed}: {source} -> target {FOUNDATION_STEPS:,} "
              f"(handoff boundary {handoff_step:,})", flush=True)
        config = foundation_config(
            run_dir=foundation, seed=seed, device=device, envs=envs, resume=foundation_resume
        )
        t0=time.time(); run_training(config, provenance=provenance)
        print(f"{LOG} foundation training process complete in {time.time()-t0:.1f}s", flush=True)
        foundation_ck, leftover_resume, observed_handoff = resolve_foundation_handoff(foundation, envs=envs)
        if foundation_ck is None:
            partial = f"; newest valid partial={leftover_resume}" if leftover_resume is not None else ""
            raise RuntimeError(
                f"foundation did not produce the required handoff checkpoint at step={observed_handoff:,}{partial}"
            )

    actual_handoff_step = checkpoint_step(foundation_ck)
    meta: dict[str, Any] = {"version": VERSION, "seed": seed, "target_steps": target_steps,
                            "foundation_target_steps": FOUNDATION_STEPS,
                            "foundation_handoff_step": actual_handoff_step,
                            "foundation_checkpoint": str(foundation_ck), "git": provenance,
                            "envs": int(envs), "runs": {}}
    for mode in modes:
        run = root / mode
        complete_ck, partial_ck, expected_step = resolve_target_checkpoint(
            run, target_steps=target_steps, envs=envs, label=mode
        )
        elapsed_s = 0.0
        if complete_ck is None:
            resume = partial_ck or foundation_ck
            print(f"{LOG} {mode} seed={seed}: -> target {target_steps:,} "
                  f"(boundary {expected_step:,}) from {resume}", flush=True)
            config = mature_config(mode, run_dir=run, steps=target_steps, seed=seed, device=device, envs=envs, resume=resume)
            t0=time.time(); run_training(config, provenance=provenance); elapsed_s=time.time()-t0
            complete_ck, leftover_partial, observed_target = resolve_target_checkpoint(
                run, target_steps=target_steps, envs=envs, label=mode
            )
            if complete_ck is None:
                partial = f"; newest valid partial={leftover_partial}" if leftover_partial is not None else ""
                raise RuntimeError(
                    f"{mode} did not produce the required final checkpoint at step={observed_target:,}{partial}"
                )
        else:
            print(f"{LOG} {mode} seed={seed}: already complete at {checkpoint_step(complete_ck):,}; not advancing it", flush=True)
        meta["runs"][mode] = {"seed": seed, "mode": mode, "seconds": round(elapsed_s, 2), "summary": summarize(run)}
        write_report(root, results_dir, meta)
    write_report(root, results_dir, meta)


def main(argv=None) -> int:
    p=argparse.ArgumentParser(description=f"EMU {VERSION} lean experiment runner")
    p.add_argument("command", choices=["train", "compare"])
    p.add_argument("--seed", type=int, default=733)
    p.add_argument("--steps", type=int, default=120_000_000)
    p.add_argument("--mode", choices=["shadow", "graph"], default="graph")
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "gpu"], default="cuda")
    p.add_argument("--run-root", default="")
    p.add_argument("--results-dir", default="")
    p.add_argument("--envs", type=int, default=2048)
    args=p.parse_args(argv)
    if args.steps < FOUNDATION_STEPS:
        raise SystemExit("--steps must be at least 20,000,000")
    suffix=f"{VERSION}_seed{args.seed}"
    root=Path(args.run_root).expanduser() if args.run_root else run_base()/suffix
    results=Path(args.results_dir).expanduser() if args.results_dir else results_base()/suffix
    modes=(args.mode,) if args.command=="train" else ("shadow", "graph")
    run_seed(args.seed,target_steps=args.steps,modes=modes,device=args.device,run_root=root,results_dir=results,envs=args.envs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
