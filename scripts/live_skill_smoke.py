"""Bounded real-PPO acceptance for the passive 2048/4096 live observer."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from scripts.ks150_abc_school import RECIPE, call, root_checked, source, verify_model, write_new


def run(root):
    # Must precede emu imports, including training_config.
    os.environ["EMU_PHYSICAL_PROFILE"] = "h40-spring-only"
    commit = source()
    verify_model()
    from emu.training_config import ROLLOUT_STEPS, ppo_minibatches_for_envs
    root = root_checked(root)
    root.mkdir(parents=True, exist_ok=True)
    registration = root / "smoke-registration.json"
    expected = dict(source_commit=commit, worlds=[2048, 4096], updates=2,
                    observer=True, seed=9611)
    if registration.exists():
        if json.loads(registration.read_text()) != expected:
            raise RuntimeError("Smoke registration differs; use a fresh root")
    else:
        write_new(registration, expected)
    for worlds in (2048, 4096):
        run_dir = root / f"worlds-{worlds}"
        receipt = root / f"worlds-{worlds}.json"
        if receipt.exists():
            continue
        target = 2 * worlds * ROLLOUT_STEPS
        args = [*RECIPE, "--envs", str(worlds), "--seed", "9611",
                "--steps", str(target), "--run-dir", str(run_dir),
                "--live-skill-observer"]
        checkpoints = run_dir / "checkpoints"
        resume = ["--resume", "auto"] if checkpoints.exists() and any(checkpoints.glob("*.pkl")) else []
        t = time.perf_counter()
        call("-m", "emu.cli", "train", *args, *resume, bounded=True)
        elapsed = time.perf_counter() - t
        meta = json.loads((run_dir / "run_meta.json").read_text())
        if meta.get("effective_ppo_minibatches") != ppo_minibatches_for_envs(worlds):
            raise RuntimeError("Unexpected effective minibatch count")
        rows = []
        for path in (run_dir / "live_skills").glob("segment-*.jsonl"):
            rows.extend(json.loads(line) for line in path.read_text().splitlines())
        if not rows or not any(r["end_step"] == target for r in rows):
            raise RuntimeError("Missing final observer window")
        for r in rows:
            if r["worlds"] != worlds or r["source_commit"] != commit:
                raise RuntimeError("Wrong observer lineage")
            for c in r["cells"].values():
                n, k = c["attempts"], c["passes"]
                if type(n) is not int or type(k) is not int or not 0 <= k <= n:
                    raise RuntimeError("Invalid observer count")
        if not (checkpoints / "final.pkl").is_file():
            raise RuntimeError("Missing validated final checkpoint")
        write_new(receipt, dict(source_commit=commit, worlds=worlds,
                  target=target, minibatches=ppo_minibatches_for_envs(worlds),
                  elapsed_including_compile_seconds=elapsed,
                  windows=len(rows), observer_host_seconds=sum(
                      r["host_aggregation_seconds"] for r in rows)))
        print(f"PASS {worlds} worlds, {len(rows)} observer windows", flush=True)
    print("Bounded live observer acceptance complete:", root, flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    run(p.parse_args(argv).root)


if __name__ == "__main__":
    main()
