"""Safely finish the 19.xM spring-v1 cold start after a final CSV schema bug.

This is a deliberate, audited *cross-source recovery fork*, not a silent
in-place source change. The original run directory, checkpoints, CSV, and
source SHA remain unmodified. A current-source child inherits the ORIGINAL
checkpoint's PPO params + optimiser + normaliser + curriculum + field.
"""
from __future__ import annotations

import csv
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from emu.contracts import ROBOT_XML
from emu.training_config import (
    TrainingConfig, training_sensor_contract,
)
from emu.checkpoint_io import (
    load_checkpoint, find_valid_resume, update_boundary_at_or_after,
)

ROOT = Path(__file__).resolve().parent.parent
LAUNCHER = ROOT / "scripts" / "run_h40_spring_v1_wsl.sh"
ORIGIN = Path.home() / "emu-data/runs/h40-spring-v1-outcomes-s735"
CHILD = Path.home() / "emu-data/runs/h40-spring-v1-outcomes-s735-repaired"

# These are the physics, observations and PPO training semantics. The
# revised source must NOT change any of them relative to the trained actor.
UNCHANGED_SOURCE = (
    "emu/rewards.py", "emu/environment.py", "emu/physical_variants.py",
    "emu/contracts.py", "emu/robot.xml", "emu/ppo.py",
    "emu/training_config.py", "emu/capability_focus.py",
    "emu/curriculum.py", "emu/training_health.py",
    "emu/step_goal.py", "emu/domain_randomization.py",
    "emu/terrain_carpet_mjx.py", "emu/carpet_prism_mjx.py",
    "emu/tof_geometry.py", "emu/sensors.py",
)


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.PIPE).strip()


def check_ancestor_source(source: str) -> None:
    now = git("rev-parse", "HEAD")
    if (len(source) != 40 or source == now or
            any(ch not in "0123456789abcdef" for ch in source)):
        raise RuntimeError("Original source SHA invalid or same as repaired source")
    subprocess.run(["git","merge-base","--is-ancestor",source,now],
                   cwd=ROOT,check=True)
    changed = git("diff", "--name-only", source, now, "--", *UNCHANGED_SOURCE)
    if changed:
        raise RuntimeError("Cannot claim reporting-only repair: "
                           "physics/learning files differ from original source:\\n" + changed)
    if git("status", "--porcelain"):
        raise RuntimeError("Commit/stash local changes before source-pinned recovery")
    if git("branch", "--show-current") != "work/emu-mechanical-feasibility":
        raise RuntimeError("Recovery must run on the one research branch")
    print(f"SAFE SOURCE FORK: {source[:12]} → {now[:12]}; physics/PPO/rewards unchanged", flush=True)


def read_original() -> tuple[dict, Path]:
    cfg_file = ORIGIN / "training_config.json"
    meta_file = ORIGIN / "run_meta.json"
    if not cfg_file.is_file() or not meta_file.is_file():
        raise RuntimeError("Original 20M run directory/config/provenance missing; not cold-starting instead")
    cfg = TrainingConfig.load(cfg_file)
    if (cfg.steps != 20_000_000 or cfg.envs != 2048 or cfg.seed != 735 or
            cfg.physical_profile != "h40-spring-v1" or
            cfg.gait_guidance_fade_steps != 20_000_000 or
            not cfg.unified_capability_focus or not cfg.terrain_carpet_prisms):
        raise RuntimeError("Origin is not the expected spring-v1 gait nursery recipe")
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    source = (meta.get("source_provenance") or {}).get("git_commit", "")
    check_ancestor_source(source)
    if (ORIGIN/"checkpoints/final.pkl").exists():
        raise RuntimeError("Origin already has final.pkl; use its completed-run continuity, not a partial repair")
    ckpt = find_valid_resume(
        ORIGIN, expected_observation_schema=cfg.observation_schema,
        expected_sensor_contract=training_sensor_contract(cfg))
    if ckpt is None:
        raise RuntimeError("No validated latest/snapshot PPO checkpoint in origin; original data preserved")
    if not (ckpt.name == "latest.pkl" or
            (ckpt.name.startswith("snapshot_step_") and ckpt.name.endswith(".pkl"))):
        raise RuntimeError("Refusing nonstandard partial checkpoint")
    payload = load_checkpoint(
        ckpt,validate=True,
        expected_observation_schema=cfg.observation_schema,
        expected_sensor_contract=training_sensor_contract(cfg))
    cmeta = payload["meta"]
    if (cmeta.get("source_provenance") or {}).get("git_commit") != source:
        raise RuntimeError("Original checkpoint source is not original run source")
    physical = hashlib.sha256(ROBOT_XML.read_bytes()).hexdigest()
    if cmeta.get("robot_model_sha256") != physical:
        raise RuntimeError("Current physical model hash differs from original learned actor")
    step = int(cmeta["step"])
    upper = update_boundary_at_or_after(20_000_000,2048)
    if not 18_000_000 <= step < upper:
        raise RuntimeError(f"Expected validated late-20M checkpoint, found step={step:,}; refuse")
    old_metrics = ORIGIN/"metrics.csv"
    if not old_metrics.is_file():
        raise RuntimeError("Origin metrics.csv missing")
    with old_metrics.open(newline="",encoding="utf-8") as stream:
        rows=list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError("Original metrics empty; cannot establish gait survival")
    latest=rows[-1]
    values={key:float(latest[key]) for key in
            ("step","fall_rate","ep_len_mean","support","nan_grad_skips_total")}
    import math
    if (not all(math.isfinite(v) for v in values.values()) or
            values["step"] < 19_000_000 or values["fall_rate"] >= .85 or
            values["ep_len_mean"] < 100 or values["support"] < .15 or
            values["nan_grad_skips_total"] > 10):
        raise RuntimeError("Original foundation failed basic recorded-survival gate; refusing unattended 120M")
    sha=hashlib.sha256(ckpt.read_bytes()).hexdigest()
    manifest={
        "repair_type":"explicit_reporting_only_partial_checkpoint_source_fork",
        "original_run":str(ORIGIN),"repaired_run":str(CHILD),
        "original_git_commit":source,"repair_git_commit":git("rev-parse","HEAD"),
        "parent_checkpoint":str(ckpt),"parent_checkpoint_sha256":sha,
        "parent_checkpoint_world_steps":step,
        "original_20m_target_boundary":upper,
        "world_steps_replayed_at_40m_target":upper-step,
        "original_latest_metrics":values,
        "original_trainer_completed_20m_but_terminal_CSV_missed_gait_guidance_strength":True,
        "changes_to_policy_physics_ppo_reward_teacher":"none; current commit only logging/repair scripts",
        "note":"The final 20M weights were not checkpointed. Continuing from the last validated 19.xM weights; original source/run unchanged.",
    }
    return manifest,ckpt


def main() -> int:
    os.environ["EMU_PHYSICAL_PROFILE"]="h40-spring-v1"
    os.environ["PYTHONPATH"]=str(ROOT) + (
        os.pathsep+os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")
    os.environ["EMU_UNIFIED_CONTINUOUS"]="cold-growth-approved"
    os.environ["EMU_SPRING_RUN_DIR"]=str(CHILD)
    manifest,parent=read_original()
    print("RECOVERED CHECKPOINT:",parent,flush=True)
    print("RECOVERED STEP:",f"{manifest['parent_checkpoint_world_steps']:,}",flush=True)
    print("WILL CONTINUE at 40M, THEN 120M; no new neural-network birth",flush=True)
    if CHILD.exists() and any(CHILD.iterdir()):
        raise RuntimeError("Recovery destination already contains data; refusing overwrite. "
                           "Inspect existing repaired lineage before attempting restart.")
    # The original 2048-world PPO already compiled and trained successfully.
    # These smoke runs exercise *new* source, not a new learning lineage.
    for mode in ("smoke","gpu-gate"):
        print(f"===== RECOVERY: {mode} =====",flush=True)
        subprocess.run(["bash",str(LAUNCHER),mode],check=True,env=os.environ)
    env=dict(os.environ)
    env.update({
        "EMU_REPAIR_PARENT_CKPT":str(parent),
        "EMU_REPAIR_PARENT_SOURCE":manifest["original_git_commit"],
        "EMU_REPAIR_PARENT_SHA256":manifest["parent_checkpoint_sha256"],
    })
    print("===== RECOVERY: verify exact ancestor before GPU training =====",flush=True)
    # Source change only fixes terminal metrics schema and explicitly
    # acknowledges the incomplete parent by two hashes.
    # All model/actor/PPO/reward files were confirmed byte-equivalent above.
    print("===== RECOVERY: resume PPO to 40M =====",flush=True)
    subprocess.run(["bash",str(LAUNCHER),"repair-40m"],check=True,env=env)
    (CHILD/"recovery_manifest.json").write_text(
        json.dumps(manifest,indent=2,sort_keys=True)+"\\n",encoding="utf-8")
    shutil.copy2(ORIGIN/"metrics.csv",CHILD/"foundation_before_repair.csv")
    print("===== RECOVERY: 40M→120M =====",flush=True)
    subprocess.run(["bash",str(LAUNCHER),"extend-120m"],check=True,env=os.environ)
    print("===== RECOVERY: independent 80mm frozen stair examination =====",flush=True)
    subprocess.run(["bash",str(LAUNCHER),"exam"],check=True,env=os.environ)
    print("COMPLETE:",CHILD/"results_bundle.zip",flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
