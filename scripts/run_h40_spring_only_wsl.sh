#!/usr/bin/env bash
# Protected H40 spring-only A/B cold training. NO Spring V1 reward/curriculum
# or wider joint targets. The original 120M actor and main are untouched.
# Usage: bash scripts/run_h40_spring_only_wsl.sh [plan|preflight|gpu-smoke|
#        start-20m|resume-20m|pulse-20m|extend-40m|pulse-40m|
#        extend-120m|pulse-120m]
# To train the paired unchanged-physics control use EMU_SPRING_AB_ARM=control.
set -euo pipefail
MODE="plan"
if (( $# )); then MODE="$1"; fi
case "$MODE" in
  plan|preflight|gpu-smoke|start-20m|resume-20m|pulse-20m|extend-40m|pulse-40m|extend-120m|pulse-120m) ;;
  *) echo "Unknown mode: $MODE" >&2; exit 2 ;;
esac
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
PY="$REPO/.venv/bin/python"
[[ -x "$PY" ]] || { echo "Missing .venv/bin/python" >&2; exit 1; }
ARM="$(printenv EMU_SPRING_AB_ARM || printf spring)"
case "$ARM" in
  spring) export EMU_PHYSICAL_PROFILE=h40-spring-only ;;
  control) export EMU_PHYSICAL_PROFILE=h40 ;;
  *) echo "EMU_SPRING_AB_ARM must be spring or control" >&2; exit 2 ;;
esac
export PYTHONPATH="$REPO"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export EMU_UNIFIED_CONTINUOUS=cold-growth-approved
RUN_DIR="$HOME/emu-data/runs/h40-$ARM-ab-s735"
if [[ -n "$(printenv EMU_SPRING_AB_RUN_DIR || true)" ]]; then
  RUN_DIR="$(printenv EMU_SPRING_AB_RUN_DIR)"
fi
EXAM_DIR="$HOME/emu-data/exams"
echo "Study arm: $ARM ($EMU_PHYSICAL_PROFILE), run: $RUN_DIR"

if [[ "$MODE" == start-20m || "$MODE" == resume-20m ||
      "$MODE" == extend-40m || "$MODE" == extend-120m ]]; then
  [[ "$(git branch --show-current)" == research/h40-spring-only-cold-ab ]] || {
    echo "Wrong experiment branch; refusing to train" >&2; exit 1;
  }
  [[ -z "$(git status --porcelain)" ]] || {
    echo "Dirty Git checkout; refusing unauthenticated source" >&2; exit 1;
  }
fi

# Before *every* run, prove original robot bytes and isolation of spring
# geometry; then verify actual compiled MuJoCo servo/ROM limits.
"$PY" - <<'PY'
import hashlib
from xml.etree import ElementTree as ET
import numpy as np
import mujoco
from emu.contracts import PACKAGE_ROOT, ROBOT_XML
from emu.environment import TARGET_SPAN
from emu.physical_variants import generate_xml, selected_profile, SPRING_ONLY_PROFILE
src = (PACKAGE_ROOT/"robot.xml").read_text()
base = generate_xml(src,"h40")
spring = generate_xml(src,SPRING_ONLY_PROFILE)
assert hashlib.sha256(base.encode()).hexdigest() == "b513bb3326a73d9ca661f761f1352b22c36d0f1b6d51be72c2eca15d8712ef5b"
a,b = ET.fromstring(base), ET.fromstring(spring)
a.set("model",b.get("model"))
for side in "rl":
    for tag,name,key in (
        ("site", f"{side}_knee_strut_thigh", "pos"),
        ("site", f"{side}_knee_strut_shin", "pos"),
        ("spatial", f"{side}_knee_support", "stiffness"),
        ("spatial", f"{side}_knee_support", "springlength"),
    ):
        x,y=a.find(f".//{tag}[@name='{name}']"),b.find(f".//{tag}[@name='{name}']")
        assert x is not None and y is not None
        x.set(key,y.get(key))
assert ET.tostring(a)==ET.tostring(b), "Non-spring XML changed"
old,new=(mujoco.MjModel.from_xml_string(x) for x in (base,spring))
np.testing.assert_array_equal(old.actuator_ctrlrange,new.actuator_ctrlrange)
np.testing.assert_array_equal(old.actuator_forcerange,new.actuator_forcerange)
np.testing.assert_array_equal(old.jnt_range,new.jnt_range)
np.testing.assert_allclose(new.tendon_stiffness[:2],[150.,150.])
np.testing.assert_allclose(TARGET_SPAN,[.38,.48,.58,.38,.48,.58],atol=1e-7)
assert np.allclose(old.actuator_forcerange,[[-.980665,.980665]]*6)
wanted=base if selected_profile()=="h40" else spring
assert ROBOT_XML.read_bytes()==wanted.encode(), "Selected profile differs"
print("PASS: original H40 preserved; spring-only changes no motor/ROM/action limits")
print("Current physical SHA256:",hashlib.sha256(ROBOT_XML.read_bytes()).hexdigest())
PY
if [[ "$MODE" == preflight ]]; then exit 0; fi
if [[ "$MODE" == gpu-smoke ]]; then
  "$PY" scripts/smoke_gpu_carpet.py --worlds 2048 --frames 2 \
    --device cuda --prisms --ppo --full-update
  exit 0
fi

# Exactly identical source/seed/lesson/PPO flags for spring and control arms.
# This is current mainline's mature recipe, not an imported Spring V1 tutor.
set -- --profile sim2real --run-dir "$RUN_DIR" \
  --envs 2048 --seed 735 --device cuda --pitch-deg 8 \
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
  --academy-height-cap-m .15 --fixed-eval-every-steps 60000000

case "$MODE" in
  plan) "$PY" -m emu.cli train --steps 20000000 --plan "$@" ;;
  start-20m) "$PY" -m emu.cli train --steps 20000000 "$@" ;;
  resume-20m) "$PY" -m emu.cli train --steps 20000000 --resume auto "$@" ;;
  extend-40m)
    echo "MANUAL GATE: inspect 20M pulse and completed checkpoint first."
    "$PY" -m emu.cli train --steps 40000000 --resume auto "$@"
    ;;
  extend-120m)
    echo "MANUAL GATE: inspect 40M pulse and completed checkpoint first."
    "$PY" -m emu.cli train --steps 120000000 --resume auto "$@"
    ;;
  pulse-20m|pulse-40m|pulse-120m)
    AGE=20
    if [[ "$MODE" == pulse-40m ]]; then AGE=40; fi
    if [[ "$MODE" == pulse-120m ]]; then AGE=120; fi
    "$PY" - "$RUN_DIR" "$AGE" <<'PY'
from pathlib import Path
import sys
from emu.training_config import TrainingConfig, training_sensor_contract
from emu.checkpoint_io import load_checkpoint
root=Path(sys.argv[1]).expanduser()
age=int(sys.argv[2])*1_000_000
cfg=TrainingConfig.load(root/"training_config.json")
assert cfg.steps==age,"Exam frozen stage BEFORE extending run"
ck=root/"checkpoints"/"final.pkl"
payload=load_checkpoint(ck,expected_sensor_contract=training_sensor_contract(cfg))
assert payload["meta"]["step"]>=age,"Stage checkpoint is incomplete"
print("Valid exam snapshot",ck,"steps",payload["meta"]["step"])
PY
    mkdir -p "$EXAM_DIR"
    OUT="$EXAM_DIR/h40-$ARM-ab-$AGE""m-s1701.zip"
    [[ ! -e "$OUT" ]] || { echo "Refusing to overwrite exam: $OUT" >&2; exit 1; }
    "$PY" -m scripts.examine_h40_specialties \
      --policy "$RUN_DIR/checkpoints/final.pkl" --seed 1701 \
      --device cuda --worlds 2048 --horizon 600 --output "$OUT"
    echo "Inspect pulse before continuing: $OUT"
    ;;
esac
