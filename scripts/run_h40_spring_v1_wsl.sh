#!/usr/bin/env bash
# H40 spring-v1 with 20M gait nursery -> 40M outcome-only reward, cold start.
# Usage: bash scripts/run_h40_spring_v1_wsl.sh all|smoke|gpu-gate|plan|train-20m|extend-120m
# One run directory and one trainer, founded on the existing main-style recipe.
set -euo pipefail

MODE="${1:-}"
case "$MODE" in
  all|smoke|gpu-gate|plan|train-20m|extend-120m) ;;
  *) echo "Usage: bash scripts/run_h40_spring_v1_wsl.sh all|smoke|gpu-gate|plan|train-20m|extend-120m" >&2; exit 2 ;;
esac

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

if [[ "$MODE" == all ]]; then
  # Unattended one-paste campaign. Each stage independently verifies the
  # actual physical model and latest source-pinned run contract. Never
  # combine the old champion checkpoint with this new reward schedule.
  for stage in smoke gpu-gate train-20m extend-120m; do
    echo "===== H40 SPRING V1 OUTCOME TRAINING: $stage ====="
    bash "$REPO/scripts/run_h40_spring_v1_wsl.sh" "$stage"
  done
  echo "===== COMPLETE: $HOME/emu-data/runs/h40-spring-v1-outcomes-s735 ====="
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

# Mainline PPO, one actor, original main foundation 0–20M, then earned
# specialists. NEW v2 teaching contract: the special gait reward and
# stepping-activity payment gate fade over 20M→40M, without rewarding a
# particular step timing or stride thereafter. Original physical safety,
# command and outcome payments remain. The different run directory is
# essential: never resume a historical v1 spring run into this source.
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

# The same v2 run directory is preserved across 20M -> 120M. Never change
# source SHA, profile or run directory between its stages.
# WARNING: do not launch alongside the earlier spring-v1 GPU experiment.
