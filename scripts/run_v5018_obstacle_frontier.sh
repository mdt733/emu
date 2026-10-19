#!/usr/bin/env bash
# v5018: frozen-champion obstacle frontier, not training.
# Paired flat/5/10/15/20mm physical boxes; ±25-degree approach edges;
# speed variation; heading OFF/ON. Logs and JSON outside Git.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
[[ "$(git -C "$ROOT" branch --show-current)" == "feature/v5018-terrain-awareness" ]] ||
  { echo "ERROR: work on the dedicated v5018 feature branch only" >&2; exit 2; }
[[ -x "$ROOT/.venv/bin/python" ]] ||
  { echo "Missing Python virtual environment" >&2; exit 2; }
source "$ROOT/.venv/bin/activate"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
BASE="$HOME/robot_lab/emu_exams"
mkdir -p "$BASE"
OUT="$BASE/v5018_frontier_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
printf 'stage\tstatus\texit_code\n' > "$OUT/stages.tsv"

run() {
 local stage="$1"; shift
 echo "========== $stage ==========" | tee -a "$OUT/campaign.log"
 (cd "$ROOT" && "$@") 2>&1 | tee "$OUT/$stage.log"
 local code="${PIPESTATUS[0]}"
 if [[ "$code" == 0 ]]; then
   printf '%s\tPASS\t0\n' "$stage" >> "$OUT/stages.tsv"
 else
   printf '%s\tFAIL\t%s\n' "$stage" "$code" >> "$OUT/stages.tsv"
 fi
 echo "$stage exit=$code" | tee -a "$OUT/campaign.log"
}
{
 echo "Study: frozen 232-input champion real-step orientation/speed/height frontier"
 date -Is
 git -C "$ROOT" rev-parse HEAD
 git -C "$ROOT" status --short
 sha256sum "$ROOT/showcase/champion.emupolicy"
 python -c 'import jax,mujoco;print("JAX",jax.__version__,"MuJoCo",mujoco.__version__,"devices",jax.devices())'
 nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader 2>&1 || true
} | tee "$OUT/environment.txt"

run targeted_python_tests python -m pytest -q \
 tests/test_direction.py::test_v5018_stride_diagnostics_identifies_strict_gait_rejection_without_changing_pass \
 tests/test_direction.py::test_v5018_step_pilot_xml_yaw_and_declared_height \
 tests/test_direction.py::test_v5018_crossing_gate_accepts_only_supported_world_fixed_crossings \
 tests/test_direction.py::test_v5018_crossing_gate_rejects_incomplete_or_fabricated_support

# Re-examine contact-derived swing events on existing flat and actual 10-degree
# grades. Controller ON/OFF is held to the same 0.75 setting across pairs.
for course in flat uphill10 downhill10 cross_left10; do
  case "$course" in
    flat) angle=0; direction=uphill ;;
    uphill10) angle=10; direction=uphill ;;
    downhill10) angle=10; direction=downhill ;;
    cross_left10) angle=10; direction=cross_left ;;
  esac
  run "stride_${course}" emu directional-exam \
    --policy "$ROOT/showcase/champion.emupolicy" \
    --output "$OUT/stride_${course}" --contact-slope \
    --angle "$angle" --direction "$direction" \
    --worlds 4 --horizon 350 --settle-frames 25 \
    --speed .10 --seed 1701 --device cuda \
    --compare-heading --heading-kp .75
done

step() {
  local h="$1" yaw="$2" speed="$3" seed="$4" kp="$5"
  local mode=on
  if [[ "$kp" == 0 ]]; then mode=off; fi
  local code_speed="${speed/./p}"
  local code_yaw="${yaw/-/m}"
  code_yaw="${code_y/+/p}"
  local stage="step_${h}mm_yaw${code_y}_speed${code_speed}_${mode}_seed${seed}"
  run "$stage" python "$ROOT/scripts/v5018_champion_step_pilot.py" \
    --height-mm "$h" --step-yaw-deg "$yaw" --speed "$speed" \
    --heading-kp "$kp" --seed "$seed" --worlds 4 \
    --horizon 350 --output "$OUT/$stage.json"
}

# Controlled height frontier with matched 2-seed physical course samples
# at the original .10m/s speed and heading hold ON.
for seed in 1701 1702; do
  for h in 0 5 10 15 20; do step "$h" 0 .10 "$seed" .75; done
done

# Vary relative face orientation, holding forward course/headings fixed.
# No special directions or hidden knowledge are given to the policy.
for h in 5 10 15; do
  for yaw in -25 25; do step "$h" "$yaw" .10 1701 .75; done
done
for yaw in -25 25; do step 10 "$yaw" .10 1702 .75; done

# A faster/slower walk changes available time-to-contact. Flat controls
# prevent interpreting general speed effects as new obstacle ability.
for speed in .07 .14; do
  for h in 0 5 10; do step "$h" 0 "$speed" 1701 .75; done
done

# A direct matched heading ablation for the original course and 15 mm.
for h in 0 5 10 15; do step "$h" 0 .10 1701 0; done

run aggregate_reports python "$ROOT/scripts/v5018_summarize_gait_and_step.py" "$OUT"

echo "================== SUMMARY =================="
cat "$OUT/stages.tsv"
tar -czf "$OUT.tar.gz" -C "$BASE" "$(basename "$OUT")"
echo "Archive: $OUT.tar.gz"
echo "Records: $OUT"
if grep -q $'\tFAIL\t' "$OUT/stages.tsv"; then
  echo "Some research stages failed; preserve the archive and diagnose separately."
  exit 1
fi
echo "All scripts exited successfully. Examine crossing counts; this is NOT a terrain-training pass."
