#!/usr/bin/env bash
# H40 spring-v1 cold-start training; source-pinned, no champion resume.
# Usage: bash scripts/run_h40_spring_v1_wsl.sh smoke|gpu-gate|plan|train-20m|extend-120m
# One run directory and one trainer, founded on the existing main-style recipe.
set -euo pipefail

MODE="${1:-}"
case "$MODE" in
  smoke|gpu-gate|plan|train-20m|extend-120m) ;;
  *) echo "Usage: bash scripts/run_h40_spring_v1_wsl.sh smoke|gpu-gate|plan|train-20m|extend-120m" >&2; exit 2 ;;
esac

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# The env var must precede *every* EMU Python import, including policy export
# and viewer/exam. Never silently reuse main's frozen actor.
export EMU_PHYSICAL_PROFILE=h40-spring-v1
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export EMU_UNIFIED_CONTINUOUS=cold-growth-approved
RUN_DIR="${EMU_SPRING_RUN_DIR:-$HOME/emu-data/runs/h40-spring-v1-s735}"
mkdir -p "$(dirname "$RUN_DIR")"

# Prevent non-source-pinned training or a dirty experimental model.
if [[ "$MODE" == train-20m || "$MODE" == extend-120m ]]; then
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
print("Research source model:", hashlib.sha256(ROBOT_XML.read_bytes()).hexdigest())
print("Research actor targets:", np.round(np.rad2deg(TARGET_SPAN), 2).tolist(), "degrees")
print("Rated actuator limits:", m.actuator_forcerange[0].tolist(), "Nm")
PY

if [[ "$MODE" == smoke || "$MODE" == gpu-gate ]]; then
  python -m emu.cli doctor --require-gpu
  WORLDS=128
  if [[ "$MODE" == gpu-gate ]]; then WORLDS=2048; fi
  python scripts/smoke_gpu_carpet.py --device cuda --worlds "$WORLDS" --frames 2 --prisms --ppo --full-update
  exit 0
fi

# Mainline philosophy: one actor, foundation 0–20M then earned specialists,
# main-mature commands, gentle pushes, original reward/PPO, same prisms.
# No reward, teacher, optimizer or phase alterations to favor the redesign.
STEPS=20000000
RESUME=()
if [[ "$MODE" == extend-120m ]]; then
  STEPS=120000000
  RESUME=(--resume auto)
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
  --solver-iterations 2 --solver-ls-iterations 4 \
  "${RESUME[@]}" \
  $(if [[ "$MODE" == plan ]]; then printf '%s' '--plan'; fi)

# The same run directory is preserved across 20M -> 120M. Never change
# code/branch/environment profile in between: source SHA is the resume key.
