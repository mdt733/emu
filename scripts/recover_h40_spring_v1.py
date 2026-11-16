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

# The physical profile is resolved during emu.contracts IMPORT, so it must
# be set BEFORE importing anything from emu (including the saved model hash).
if os.environ.get("EMU_PHYSICAL_PROFILE") not in (None, "h40-spring-v1"):
    raise RuntimeError("Do not run partial recovery under another robot profile")
os.environ["EMU_PHYSICAL_PROFILE"] = "h40-spring-v1"

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
                           "physics/learning files differ from original source:\n" + changed)
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
    contract=training_sensor_contract(cfg)
    expected_boundary=update_boundary_at_or_after(20_000_000,2048)
    best=ORIGIN/"checkpoints/best.pkl"
    ckpt=None
    payload=None
    if best.is_file():
        try:
            tested=load_checkpoint(
                best,validate=True,expected_observation_schema=cfg.observation_schema,
                expected_sensor_contract=contract)
            bmeta=tested["meta"]
            import math
            if (int(bmeta["step"])==expected_boundary and
                    bmeta.get("evaluated_policy") is True and
                    math.isfinite(float(bmeta.get("best_eval_score",float("nan"))))):
                ckpt,payload=best,tested
                print("FOUND EXACT FULL-20M EVALUATED BEST CHECKPOINT",flush=True)
        except (OSError,ValueError,RuntimeError,KeyError,TypeError) as exc:
            print(f"best.pkl not valid for repair ({exc}); checking normal snapshots",flush=True)
    if ckpt is None:
        ckpt=find_valid_resume(
            ORIGIN, expected_observation_schema=cfg.observation_schema,
            expected_sensor_contract=contract)
        if ckpt is None:
            raise RuntimeError("No validated best/latest/snapshot PPO checkpoint in origin; preserved")
        if not (ckpt.name == "latest.pkl" or
                (ckpt.name.startswith("snapshot_step_") and ckpt.name.endswith(".pkl"))):
            raise RuntimeError("Refusing nonstandard partial checkpoint")
        payload=load_checkpoint(
            ckpt,validate=True,expected_observation_schema=cfg.observation_schema,
            expected_sensor_contract=contract)
    cmeta = payload["meta"]
    if (cmeta.get("source_provenance") or {}).get("git_commit") != source:
        raise RuntimeError("Original checkpoint source is not original run source")
    physical = hashlib.sha256(ROBOT_XML.read_bytes()).hexdigest()
    if cmeta.get("robot_model_sha256") != physical:
        raise RuntimeError("Current physical model hash differs from original learned actor")
    step = int(cmeta["step"])
    upper = update_boundary_at_or_after(20_000_000,2048)
    if not 18_000_000 <= step <= upper:
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
        "world_steps_missing_from_original_terminal_checkpoint":upper-step,
        "exact_20m_evaluated_checkpoint_recovered":ckpt.name=="best.pkl",
        "original_latest_metrics":values,
        "original_trainer_completed_20m_but_terminal_CSV_missed_gait_guidance_strength":True,
        "changes_to_policy_physics_ppo_reward_teacher":"none; current commit only logging/repair scripts",
        "note":(
            "The complete 20M evaluated weights and optimiser were captured in best.pkl before "
            "the terminal CSV failure." if ckpt.name=="best.pkl" else
            "Final 20M weights not preserved; resume from last validated 19.xM rotating checkpoint."
        ) + " Original run and source remain unchanged.",
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
    prior_cfg=CHILD/"training_config.json"
    already=CHILD.exists() and any(CHILD.iterdir())
    already_120=False
    if already:
        if not prior_cfg.is_file() or not (CHILD/"run_meta.json").is_file():
            raise RuntimeError("Recovery destination is incomplete/ambiguous. "
                               "Original remains safe; inspect its child before retrying.")
        prior=TrainingConfig.load(prior_cfg)
        current_meta=json.loads((CHILD/"run_meta.json").read_text(encoding="utf-8"))
        if ((current_meta.get("source_provenance") or {}).get("git_commit") !=
                manifest["repair_git_commit"] or
                prior.steps not in (40_000_000,120_000_000) or
                prior.seed != 735 or prior.envs != 2048 or
                prior.physical_profile != "h40-spring-v1"):
            raise RuntimeError("Refusing to mix an existing repaired run with "
                               "different source, recipe or actor")
        already_120=prior.steps == 120_000_000
        print("SAFE RESTART of authenticated repaired run:",prior.steps,flush=True)
    else:
        # The original 20M run already proved sustained 2048-world CUDA PPO
        # on this exact robot+reward model. Repeating CUDA compilation/smoke
        # would spend GPU time but not test the reporting-only change.
        # The next stage still performs compiled MuJoCo motor/spring preflight
        # and authentic checkpoint/source verification before PPO updates.
        print("SKIPPING REDUNDANT GPU SMOKE: actual original 20M PPO already succeeded",flush=True)
    env=dict(os.environ)
    env.update({
        "EMU_REPAIR_PARENT_CKPT":str(parent),
        "EMU_REPAIR_PARENT_SOURCE":manifest["original_git_commit"],
        "EMU_REPAIR_PARENT_SHA256":manifest["parent_checkpoint_sha256"],
    })
    if not already_120:
        print("===== RECOVERY: resume PPO to 40M =====",flush=True)
        # First invocation verifies the full ancestor source+bytes; a retry
        # auto-resumes the child with its *own* newly pinned source.
        subprocess.run(["bash",str(LAUNCHER),"repair-40m"],check=True,env=env)
    manifest_path=CHILD/"recovery_manifest.json"
    if manifest_path.exists():
        saved=json.loads(manifest_path.read_text(encoding="utf-8"))
        for name in ("original_git_commit","parent_checkpoint_sha256",
                     "repair_git_commit"):
            if saved.get(name)!=manifest[name]:
                raise RuntimeError("Existing repair manifest differs; no lineage mutation")
    else:
        manifest_path.write_text(
            json.dumps(manifest,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    if not (CHILD/"foundation_before_repair.csv").is_file():
        shutil.copy2(ORIGIN/"metrics.csv",CHILD/"foundation_before_repair.csv")
    print("===== RECOVERY: 40M→120M =====",flush=True)
    subprocess.run(["bash",str(LAUNCHER),"extend-120m"],check=True,env=os.environ)
    print("===== RECOVERY: independent 80mm frozen stair examination =====",flush=True)
    subprocess.run(["bash",str(LAUNCHER),"exam"],check=True,env=os.environ)
    print("COMPLETE:",CHILD/"results_bundle.zip",flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
