"""v5057: clean, sequential, paired-seed 20M cold-start physical tournament.

ONE existing EMU trainer and source commit; no forked PPO or obsolete worlds.
Default is a read-only plan. --execute runs in the foreground, exactly one
CUDA learner at a time (safe for an 8GB GPU). --report collates CSV evidence.
Each body and seed gets an independent source-pinned cold start.
Retire this campaign launcher once the physical selection is documented.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

# Current source, not some other installed editable worktree.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from emu.physical_variants import PROFILES, profile_numbers


def selections(value: str, *, profiles: bool) -> list[str] | list[int]:
    result = [x.strip() for x in value.split(",") if x.strip()]
    if not result or len(result) != len(set(result)):
        raise ValueError("Selections must be nonempty and without duplicates")
    if profiles:
        missing = set(result) - set(PROFILES)
        if missing:
            raise ValueError(f"Unknown physical profiles: {sorted(missing)}")
        return result
    seeds = [int(x) for x in result]
    if any(s <= 0 or s >= 2**31 for s in seeds):
        raise ValueError("Invalid seed")
    return seeds


def git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def recipe(run_dir: Path, seed: int, *, resume: bool) -> list[str]:
    # Identical learning economics across ALL independent cold starts.
    # Academy ramp happens AFTER the 20M stability screening, so this
    # tournament tests locomotion without crediting unlearned 3cm stairs.
    return [
        sys.executable, "-m", "emu.cli", "train",
        "--run-dir", str(run_dir), "--steps", "20000000",
        "--envs", "2048", "--seed", str(seed),
        "--device", "cuda", "--profile", "sim2real",
        "--terrain-actor", "--unified-capabilities",
        "--terrain-carpet", "--carpet-prisms", "--academy-stream",
        "--terrain-step-max-m", ".08", "--terrain-lesson",
        "coherent-careful",
        "--growth-recipe", "main-mature",
        "--foundation-steps", "20000000",
        "--fixed-eval-every-steps", "20000000",
        "--solver-iterations", "2", "--solver-ls-iterations", "4",
        "--academy-intro-steps", "20000000",
        "--academy-start-steps", "40000000",
        "--academy-ramp-steps", "15000000",
        "--academy-intro-share", ".20", "--academy-max-share", ".60",
        "--athletic-rehearsal-share", ".15",
        "--retention-share", ".15",
        "--temporal-vx-reward-share", ".25",
        "--science-step-approach-m", ".22",
        "--science-step-goal-weight", ".30",
        "--science-step-finish-on-transfer",
        *(["--resume", "auto"] if resume else []),
    ]


def _float(row: dict, key: str) -> float | None:
    try:
        n = float(row.get(key, "nan"))
        return n if math.isfinite(n) else None
    except (ValueError, TypeError):
        return None


def _nearest(rows: list[dict], step: int) -> dict:
    return min(rows, key=lambda r: abs((_float(r,"step") or 0) - step))


def _last_csv(path: Path) -> dict:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    return rows[-1] if rows else {}


def row_for(run_dir: Path, profile: str, seed: int) -> dict:
    result=dict(profile=profile, seed=seed,
                nominal_hip_cm=round(100 * profile_numbers(profile)["hip_height_m"],2),
                mass_g=round(1000 * profile_numbers(profile)["mass_kg"],1),
                knee_k=round(profile_numbers(profile)["knee_strut_k_n_m"]),
                roll_k=profile_numbers(profile)["hip_roll_stiffness_nm_rad"],
                checkpoint="pending")
    file=run_dir/"metrics.csv"
    if not file.exists():
        return result
    with file.open(newline="",encoding="utf-8-sig") as source:
        rows=list(csv.DictReader(source))
    if not rows:
        return result
    final=max(rows,key=lambda r: _float(r,"step") or 0)
    result["world_steps"]=round(_float(final,"step") or 0)
    result["checkpoint"]="complete" if (run_dir/"checkpoints"/"final.pkl").is_file() and result["world_steps"]>=20_000_000 else "partial"
    for key in ("fall_rate","core_fall_rate","ep_len_mean",
                "vx_mean","cmd_vx_mean","servo_over_rated_frac",
                "servo_peak_torque_limit_frac","servo_speed_over_noload_frac",
                "knee_strut_energy_j","ppo_hip_roll_clip_left_other",
                "shaft_j_per_m_forward"):
        result[key]=_float(final,key)
    for million in (5,10,20):
        sample=_nearest(rows,million*1_000_000)
        result[f"fall_{million}m"]=_float(sample,"fall_rate")
        result[f"len_{million}m"]=_float(sample,"ep_len_mean")
    fixed=_last_csv(run_dir/"fixed_eval.csv")
    # The owned exam CSV may expose multiple physical grades and scores.
    # Don't pretend an unknown numeric column is a held-out success rate.
    for key in ("score","overall_score","total_score"):
        if _float(fixed,key) is not None:
            result["heldout_score"]=_float(fixed,key)
            break
    return result


def _fmt(value, fmt=".2f"):
    return "—" if value is None else format(value,fmt)


def report(root: Path, profiles: list[str], seeds: list[int]) -> None:
    root.mkdir(parents=True,exist_ok=True)
    rows=[row_for(root/f"{p}-s{s}",p,s) for p in profiles for s in seeds]
    csv_path=root/"physical_matrix.csv"
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with csv_path.open("w",newline="",encoding="utf-8") as file:
        writer=csv.DictWriter(file,fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    lines=[
        "# EMU v5057 physical cold-start tournament", "",
        "Every row is one genuinely independent cold start, not a resumed",
        "branch or duplicate policy. Training age is **20M world steps**; ",
        "uncompleted runs are labelled partial. Each variant uses identical",
        "curriculum and seed pairing. Training rewards are NOT exam evidence.",
        "",
        "| Profile | Seed | Hip | Mass | 20M? | Fall 5M → 20M | Ep length 20M | vx / cmd | Over rated | Speed > no-load | Held-out |",
        "|:---|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['profile']} | {row['seed']} | {row['nominal_hip_cm']:.1f}cm "
            f"| {row['mass_g']:.0f}g | {row['checkpoint']} "
            f"| {_fmt(row.get('fall_5m'),'.1%')} → {_fmt(row.get('fall_20m'),'.1%')} "
            f"| {_fmt(row.get('len_20m'),'.0f')} "
            f"| {_fmt(row.get('vx_mean'))}/{_fmt(row.get('cmd_vx_mean'))} "
            f"| {_fmt(row.get('servo_over_rated_frac'),'.1%')} "
            f"| {_fmt(row.get('servo_speed_over_noload_frac'),'.1%')} "
            f"| {_fmt(row.get('heldout_score'),'.3f')} |")
    lines += ["", "## Selection rules", "",
        "- Evaluate paired seeds, not one lucky walk.",
        "- Require sustained walking and actual velocity tracking;",
        "  low falls caused by standing still are not success.",
        "- Compare measured episode length and fall fraction at 5M, 10M, 20M;",
        "  don't select on the training reward alone.",
        "- High servo speed beyond C018's nominal no-load limit is a",
        "  **sim-to-real warning**, not motor-realistic success;",
        "  simulator torque-speed enforcement remains uncalibrated.",
        "- Hip-roll position clipping is distinct from true motor torque.",
        "- Do exact signed independent physical stair exams only after",
        "  selecting stable finalist profiles; never treat a 20M foundation",
        "  policy as a proven stair climber.",
        "", "Source: per-run training_config.json + robot-model SHA256 in",
        "checkpoint and actor export. No run is overwritten by this report.", ""]
    dest=root/"physical_matrix.md"
    dest.write_text("\n".join(lines),encoding="utf-8")
    print(f"{dest}\n{csv_path}")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path.home()/"emu-data"/"runs"/"v5057-physical-matrix")
    parser.add_argument("--profiles",default=",".join(PROFILES))
    parser.add_argument("--seeds",default="735,761")
    group=parser.add_mutually_exclusive_group()
    group.add_argument("--execute",action="store_true",help="run sequential real GPU cold starts")
    group.add_argument("--report",action="store_true",help="aggregate existing CSVs; no GPU")
    a=parser.parse_args(argv)
    profiles=selections(a.profiles,profiles=True)
    seeds=selections(a.seeds,profiles=False)
    root=a.root.expanduser().resolve()
    if ROOT==root or ROOT in root.parents:
        parser.error("Output must be outside the Git checkout")
    if a.report:
        return report(root,profiles,seeds)
    if git(["status","--porcelain"]):
        raise RuntimeError("Source checkout must be clean before training")
    commit=git(["rev-parse","HEAD"])
    if len(commit)!=40:
        raise RuntimeError("No pinned Git source revision")
    manifest=root/"experiment.json"
    identity=dict(schema="v5057_sequential_cold_matrix_1",commit=commit,
        total_steps=20_000_000,envs=2048,recipe="shared_v5057_foundation_1")
    if manifest.exists():
        if json.loads(manifest.read_text()) != identity:
            raise RuntimeError("Different source/recipe at the same experiment root")
    elif a.execute:
        root.mkdir(parents=True,exist_ok=True)
        manifest.write_text(json.dumps(identity,indent=2)+"\n")
    print(f"v5057 source={commit}, profiles={profiles}, seeds={seeds}")
    print("One GPU learner at a time; original model unchanged; no old actor reused.")
    for seed in seeds:
        for name in profiles:
            run=root/f"{name}-s{seed}"
            resume=(run/"checkpoints"/"latest.pkl").exists()
            cmd=recipe(run,seed,resume=resume)
            env=dict(os.environ,EMU_PHYSICAL_PROFILE=name,
                     EMU_UNIFIED_CONTINUOUS="cold-growth-approved",
                     XLA_PYTHON_CLIENT_PREALLOCATE="false",
                     PYTHONPATH=str(ROOT)+os.pathsep+os.environ.get("PYTHONPATH",""))
            print(f"\n{seed=} {name=} hip={profile_numbers(name)['hip_height_m']*100:.1f}cm "
                  f"knee={profile_numbers(name)['knee_strut_k_n_m']:.0f}N/m")
            print(shlex.join(cmd),flush=True)
            if not a.execute:
                continue
            # Same identity/checkpoint/robot SHA checks performed by EMU.
            subprocess.run([*cmd,"--check-resume-only"],cwd=ROOT,env=env,check=True)
            subprocess.run(cmd,cwd=ROOT,env=env,check=True)
            report(root,profiles,seeds)
    if not a.execute:
        print("\nDRY PLAN ONLY. Pass --execute to run in the foreground.")
    else:
        report(root,profiles,seeds)


if __name__=="__main__":
    main()
