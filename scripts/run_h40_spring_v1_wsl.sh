#!/usr/bin/env bash
# H40 spring-v1 with 20M gait nursery -> 40M outcome-only reward, cold start.
# Usage: bash scripts/run_h40_spring_v1_wsl.sh all|smoke|gpu-gate|plan|train-20m|extend-40m|extend-120m|exam
# One run directory and one trainer, founded on the existing main-style recipe.
set -euo pipefail

MODE="${1:-}"
case "$MODE" in
  all|smoke|gpu-gate|plan|train-20m|extend-40m|extend-120m|exam) ;;
  *) echo "Usage: bash scripts/run_h40_spring_v1_wsl.sh all|smoke|gpu-gate|plan|train-20m|extend-40m|extend-120m|exam" >&2; exit 2 ;;
esac

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

if [[ "$MODE" == all ]]; then
  # Unattended one-paste campaign. Each stage independently verifies the
  # actual physical model and latest source-pinned run contract. Never
  # combine the old champion checkpoint with this new reward schedule.
  for stage in smoke gpu-gate train-20m extend-40m extend-120m exam; do
    echo "===== H40 SPRING V1 OUTCOME TRAINING: $stage ====="
    bash "$REPO/scripts/run_h40_spring_v1_wsl.sh" "$stage"
  done
  echo "===== COMPLETE: $HOME/emu-data/runs/h40-spring-v1-outcomes-s735 ====="
  echo "Upload: $HOME/emu-data/runs/h40-spring-v1-outcomes-s735/results_bundle.zip"
  exit 0
fi

# The env var must precede *every* EMU Python import, including policy export
# and viewer/exam. Never silently reuse main's frozen actor.
export EMU_PHYSICAL_PROFILE=h40-spring-v1
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export EMU_UNIFIED_CONTINUOUS=cold-growth-approved
RUN_DIR="${EMU_SPRING_RUN_DIR:-$HOME/emu-data/runs/h40-spring-v1-outcomes-s735}"
mkdir -p "$(dirname "$RUN_DIR")"

# Prevent non-source-pinned training or a dirty experimental model.
if [[ "$MODE" == train-20m || "$MODE" == extend-40m || "$MODE" == extend-120m ]]; then
  if [[ -n "$(git status --porcelain)" ]]; then
    echo "Refusing training: uncommitted/untracked Git changes." >&2
    exit 1
  fi
fi

# Local, hardware-relevant invariant before paying for GPU JIT or PPO.
python - <<'PY'
import hashlib
import numpy as np
import mujoco
from emu.contracts import ROBOT_XML
from emu.environment import TARGET_SPAN
from emu.physical_variants import selected_profile, RESEARCH_PROFILE, SERVO_RATED_CONTINUOUS_TORQUE_NM
assert selected_profile() == RESEARCH_PROFILE
m = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
assert m.nu == 6 and m.ntendon == 2
assert (m.actuator_forcelimited == 1).all()
assert np.allclose(m.actuator_forcerange,
                   [[-SERVO_RATED_CONTINUOUS_TORQUE_NM,
                     SERVO_RATED_CONTINUOUS_TORQUE_NM]] * 6)
assert np.isclose(TARGET_SPAN[2], np.pi/2, atol=1e-6)
assert all(np.isclose(m.jnt_range[
    mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_knee")][0],
    -np.pi/2, atol=1e-5) for side in "rl")
# MUJOCO ACTUATOR CTRL LIMITS ARE RAW RADIANS, NOT THE DEGREE-BASED
# JOINT RANGE: validate the actual compiled model in the user's WSL.
assert (m.actuator_ctrllimited == 1).all()
for side in "rl":
    for axis, lo, hi in (("knee",-90.,95.),
                         ("hip_pitch",-50.,50.),
                         ("hip_roll",-25.,25.)):
        aid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_ACTUATOR,
                              f"{side}_{axis}")
        assert aid >= 0
        actual=m.actuator_ctrlrange[aid]
        assert np.allclose(actual,np.deg2rad([lo,hi]),atol=1e-6), (
            f"Actuator {side}_{axis}: expected radian targets "
            f"{np.deg2rad([lo,hi]).tolist()}, compiled {actual.tolist()}")
assert np.allclose(m.tendon_stiffness[:2],[150.,150.],atol=1e-4)
# The replacement spring should help support neutral standing, but should
# NOT itself overrun a rated knee servo during a substantial swing bend.
data=mujoco.MjData(m)
for side in "rl":
    jid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f"{side}_knee")
    qid,vid=m.jnt_qposadr[jid],m.jnt_dofadr[jid]
    torques={}
    for degrees in (0,-55,-85,-90):
        mujoco.mj_resetData(m,data)
        data.qpos[qid]=np.deg2rad(degrees)
        mujoco.mj_forward(m,data)
        torque=float(data.qfrc_passive[vid])
        assert np.isfinite(torque) and np.isfinite(data.ten_length).all()
        torques[degrees]=torque
    assert np.isclose(torques[0],.520606,atol=.008),torques
    assert 0. < torques[-55] < SERVO_RATED_CONTINUOUS_TORQUE_NM,torques
    print(side,"knee spring passive torques (0/-55/-85/-90 deg):",
          [round(torques[d],3) for d in (0,-55,-85,-90)],"Nm")
print("Research source model:", hashlib.sha256(ROBOT_XML.read_bytes()).hexdigest())
print("Research actor targets:", np.round(np.rad2deg(TARGET_SPAN), 2).tolist(), "degrees")
print("Rated actuator limits:", m.actuator_forcerange[0].tolist(), "Nm")
print("MuJoCo physics solver:", m.opt.iterations, "iterations,",
      m.opt.ls_iterations, "line searches")
assert m.opt.iterations >= 50 and m.opt.ls_iterations >= 10, (
    "Do not train with a low-contact-fidelity solver without paired smoke")
PY

if [[ "$MODE" == smoke || "$MODE" == gpu-gate ]]; then
  python -m emu.cli doctor --require-gpu
  # smoke_gpu_carpet.py explicitly reserves --full-update for >=1024
  # worlds. Test the lightweight MJX+PPO path first, then the real 2048
  # world/32-frame full PPO update with the unchanged production solver.
  if [[ "$MODE" == smoke ]]; then
    python scripts/smoke_gpu_carpet.py \
      --device cuda --worlds 128 --frames 2 --prisms --ppo
  else
    python scripts/smoke_gpu_carpet.py \
      --device cuda --worlds 2048 --frames 2 --prisms --ppo --full-update
  fi
  exit 0
fi


if [[ "$MODE" == exam ]]; then
  # Read-only after training. Two separate executable processes load the
  # *correct* physical model and target span for each frozen actor.
  # A 2,000-world exact signed lip matrix is 0.7M physics frames per
  # actor, below 1% of the training budget. No PPO updates occur.
  FINAL="$RUN_DIR/checkpoints/final.pkl"
  POLICY="$RUN_DIR/h40-spring-v1-outcomes-120m.emupolicy"
  EXAM_ROOT="$RUN_DIR/heldout"
  NEW_EXAM="$EXAM_ROOT/spring-v1"
  MAIN_EXAM="$EXAM_ROOT/main-champion"
  mkdir -p "$EXAM_ROOT"
  [[ -f "$FINAL" ]] || { echo "No completed final checkpoint for examination" >&2; exit 1; }
  if [[ ! -f "$POLICY" ]]; then
    python -m emu.cli export-policy "$FINAL" "$POLICY"
  fi
  if [[ ! -f "$NEW_EXAM/matrix_summary.json" ]]; then
    python scripts/v5053_matrix_exam.py \
      --policy "$POLICY" --output "$NEW_EXAM" \
      --mode stairs --worlds 2048 --horizon 350 \
      --levels 20 --trials-per-level 50 --max-step-m .08 \
      --seed 1701 --device cuda
  fi
  # The champion is evaluated with its OWN unchanged H40 physical profile,
  # not the new 150N/m spring or the widened neural action targets.
  BASELINE="$REPO/showcase/champion.emupolicy"
  [[ -f "$BASELINE" ]] || {
    echo "Cannot complete matched exam: original champion actor missing" >&2
    exit 1
  }
  if [[ ! -f "$MAIN_EXAM/matrix_summary.json" ]]; then
    EMU_PHYSICAL_PROFILE=h40 python scripts/v5053_matrix_exam.py \
      --policy "$BASELINE" --output "$MAIN_EXAM" \
      --mode stairs --worlds 2048 --horizon 350 \
      --levels 20 --trials-per-level 50 --max-step-m .08 \
      --seed 1701 --device cuda
  fi
  export EMU_SPRING_REPORT_ROOT="$RUN_DIR"
  python - <<'PY'
import json
import os
import zipfile
from pathlib import Path
root = Path(os.environ["EMU_SPRING_REPORT_ROOT"])
exam = root / "heldout"
sources = {
    "Spring-v1": exam / "spring-v1" / "matrix_summary.json",
    "Main H40": exam / "main-champion" / "matrix_summary.json",
}
data = {label: json.loads(p.read_text(encoding="utf-8"))
        for label,p in sources.items() if p.is_file()}
if "Spring-v1" not in data:
    raise RuntimeError("New frozen actor has no physically verified matrix results")
if set(data) != set(sources):
    raise RuntimeError("Matched control missing: need BOTH independent actor matrices")
for label, summary in data.items():
    manifest=summary["manifest"]
    expected=dict(seed=1701, worlds=2048, assigned_worlds=2000,
                  horizon=350, mode="stairs", levels=20, max_step_m=.08)
    for key,wanted in expected.items():
        if manifest.get(key) != wanted:
            raise RuntimeError(f"{label} matrix contract {key}: "
                               f"expected {wanted}, got {manifest.get(key)}")
    if not manifest.get("zero_optimizer_updates") or not manifest.get("first_life_only"):
        raise RuntimeError(f"{label} exam did not verify frozen first-life evaluation")
old_hash=data["Main H40"]["manifest"]["robot_model_sha256"]
new_hash=data["Spring-v1"]["manifest"]["robot_model_sha256"]
if old_hash != "b513bb3326a73d9ca661f761f1352b22c36d0f1b6d51be72c2eca15d8712ef5b":
    raise RuntimeError(f"Wrong main H40 control model SHA: {old_hash}")
if old_hash == new_hash:
    raise RuntimeError("Both exams used same robot physics! Wrong EMU_PHYSICAL_PROFILE")
lines = ["# H40 vs Spring-v1: independent frozen physical stair examinations", "",
         "Both use 2,000 real-prism attempts, 50 each at signed 4mm increments",
         "up to +/-80mm, same initial command and random seed. This compares",
         "two independently trained actors and *different physics*, so it",
         "does not isolate the causal contribution of spring, range or reward.",
         "Leading-foot upper contact is NOT equivalent to two-foot support",
         "or to traversing the 60cm destination window.", "",
         "| Actor | Signed riser | First upper foot | Both feet held | Safe traversal |",
         "|---|---:|---:|---:|---:|"]
for label, summary in data.items():
    lookup = {(r["kind"], int(r["level"])):r
              for r in summary["curves"]}
    manifest = summary["manifest"]
    lines.append(f"")
    lines.append(f"**{label}** physical SHA256: " +
                 f"`{manifest['robot_model_sha256']}`")
    for level in (10,12,14,16,20):
        for kind in ("step_up", "step_down"):
            row = lookup[(kind,level)]
            lines.append(
                f"| {label} | {row['requested_step_mm']:+.1f}mm | "
                f"{row['leading_top']}/{row['worlds']} | "
                f"{row['each_foot_top_five']}/{row['worlds']} | "
                f"{row['traversed_5frame']}/{row['worlds']} |")
lines += ["", "Full 20-height curves and first-life world-level"
          " evidence are in heldout/*/matrix_*.csv.",
          "Training metrics and this exam are distinct sources of truth.", ""]
comparison=exam/"comparison.md"
comparison.write_text("\n".join(lines),encoding="utf-8")
# The normal trainer already produced a compact results bundle without
# huge optimizer/PKL files. Add the heldout physical comparison and actor.
bundle=root/"results_bundle.zip"
if not bundle.is_file():
    raise RuntimeError("Missing trainer results bundle after final checkpoint")
with zipfile.ZipFile(bundle,mode="a",compression=zipfile.ZIP_DEFLATED) as z:
    present=set(z.namelist())
    sources=[comparison,root/"h40-spring-v1-outcomes-120m.emupolicy"]
    sources.extend(p for p in exam.rglob("*")
                   if p.is_file() and p != comparison and p.suffix in (
                       ".csv",".json",".md"))
    for p in sources:
        if p.is_file():
            name=p.relative_to(root).as_posix()
            if name not in present:
                z.write(p,arcname=name)
                present.add(name)
print("HELD-OUT EXAM:",comparison)
print("RESULTS BUNDLE:",bundle)
PY
  exit 0
fi

# Mainline PPO, one actor, original main foundation 0–20M, then earned
# specialists. NEW v2 teaching contract: the special gait reward and
# stepping-activity payment gate fade over 20M→40M, without rewarding a
# particular step timing or stride thereafter. Original physical safety,
# command and outcome payments remain. The different run directory is
# essential: never resume a historical v1 spring run into this source.
STEPS=20000000
RESUME=()
if [[ "$MODE" == extend-40m ]]; then
  STEPS=40000000
  RESUME=(--resume auto)
fi
if [[ "$MODE" == extend-120m ]]; then
  STEPS=120000000
  RESUME=(--resume auto)
fi

# Two automatic checkpoints, with no human interaction:
# 20M checks basic cold-start viability before 20M→40M.
# 40M checks whether withdrawing preferred-gait rewards collapsed the walker
# before committing the final 80M. Both are deliberately permissive.
# No comparisons of training reward means across changing reward economics.
if [[ "$MODE" == extend-40m || "$MODE" == extend-120m ]]; then
  export EMU_SPRING_GATE_RUN_DIR="$RUN_DIR"
  if [[ "$MODE" == extend-40m ]]; then
    export EMU_SPRING_GATE_AGE=20
  else
    export EMU_SPRING_GATE_AGE=40
  fi
  python - <<'PY'
import csv
import json
import math
import os
from pathlib import Path

root = Path(os.environ["EMU_SPRING_GATE_RUN_DIR"])
age = int(os.environ["EMU_SPRING_GATE_AGE"])
assert age in (20,40)
csv_path = root / "metrics.csv"
if not csv_path.is_file():
    raise RuntimeError(f"Cannot extend: no {age}M training metrics")
with csv_path.open(newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
if not rows:
    raise RuntimeError("Cannot extend: empty training metrics")
last = rows[-1]
def number(name):
    value = float(last.get(name, float("nan")))
    if not math.isfinite(value):
        raise RuntimeError(f"Cannot extend: nonfinite {name} at {age}M")
    return value
step = number("step")
fall = number("fall_rate")
length = number("ep_len_mean")
support = number("support")
skips = number("nan_grad_skips_total")
reasons = []
if step < age * 1_000_000: reasons.append(f"{age}M checkpoint incomplete")
if not 0 <= fall < .85: reasons.append("more than 85% of completed episodes fall")
if length < 100: reasons.append("mean completed episode shorter than 100 frames")
if support < .15: reasons.append("too little foot support")
if skips > 10: reasons.append("PPO had more than 10 nonfinite gradient skips")
facts = dict(step=step,fall_rate=fall,ep_len_mean=length,
             episode_support=support,nan_grad_skips_total=skips,
             interpretation="permissive gross-collapse gate, NOT proven locomotion",
             approved=(not reasons),blockers=reasons)
(root/f"{age}m_unattended_gate.json").write_text(
    json.dumps(facts,indent=2)+"\n",encoding="utf-8")
print(f"{age}M AUTOMATIC GATE:",json.dumps(facts),flush=True)
if reasons:
    raise SystemExit(f"Stopping at {age}M rather than continuing a collapsed learner")
PY
fi

python -m emu.cli train \
  --profile sim2real \
  --run-dir "$RUN_DIR" \
  --steps "$STEPS" --envs 2048 --seed 735 --device cuda \
  --terrain-actor --terrain-step-max-m .08 \
  --unified-capabilities --terrain-carpet --carpet-prisms \
  --carpet-transition-cap-deg 3 \
  --academy-stream --terrain-lesson coherent-careful \
  --foundation-steps 20000000 --foundation-mix main \
  --growth-recipe main-mature --teaching-schedule terrain-early \
  --force-program gentle --athletic-rehearsal-share .10 \
  --retention-share .15 --academy-intro-steps 60000000 \
  --academy-start-steps 80000000 --academy-ramp-steps 20000000 \
  --academy-intro-share .10 --academy-max-share .40 \
  --academy-height-cap-m .15 \
  --fixed-eval-every-steps 60000000 \
  "${RESUME[@]}" \
  $(if [[ "$MODE" == plan ]]; then printf '%s' '--plan'; fi)

# The same v2 run directory is preserved across 20M -> 40M -> 120M. Never change
# source SHA, profile or run directory between its stages.
# WARNING: do not launch alongside the earlier spring-v1 GPU experiment.
