#!/usr/bin/env bash
# v5018 follow-up: frozen champion strict walking diagnostics and first
# true 0/5/10mm step-collision crossings. No terrain learning and no
# changes to protected policy, main, or existing experiment directories.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
[[ "$(git -C "$ROOT" branch --show-current)" == "feature/v5018-terrain-awareness" ]] ||
  { echo 'Must run on approved v5018 branch' >&2; exit 2; }
[[ -x "$ROOT/.venv/bin/python" ]] ||
  { echo 'Python venv missing' >&2; exit 2; }
source "$ROOT/.venv/bin/activate"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
BASE="$HOME/robot_lab/emu_exams"
mkdir -p "$BASE"
OUT="$BASE/v5018_gait_step_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
printf 'stage\tstatus\texit_code\n' > "$OUT/stages.tsv"

run() {
  local title="$1";shift
  echo "========== $title ==========" | tee -a "$OUT/campaign.log"
  (cd "$ROOT" && "$@") 2>&1 | tee "$OUT/$title.log"
  local code="${PIPESTATUS[0]}"
  if [[ "$code" -eq 0 ]]; then
    printf '%s\tPASS\t0\n' "$title" >> "$OUT/stages.tsv"
  else
    printf '%s\tFAIL\t%s\n' "$title" "$code" >> "$OUT/stages.tsv"
  fi
  echo "$title exit=$code" | tee -a "$OUT/campaign.log"
}

{
  date -Is
  echo "Branch: $(git -C "$ROOT" branch --show-current)"
  echo "Source commit: $(git -C "$ROOT" rev-parse HEAD)"
  sha256sum "$ROOT/showcase/champion.emupolicy"
  git -C "$ROOT" status --short
  python -c 'import jax,mujoco;print("JAX",jax.__version__,"MuJoCo",mujoco.__version__,"backend",jax.devices())'
  nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader 2>&1 || true
} | tee "$OUT/environment.txt"

# Only rerun corrected/extended tests; the big P0 suite is archived already.
run targeted_regressions python -m pytest -q \
    tests/test_learning.py::test_training_config_roundtrip \
    tests/test_learning.py::test_v5016_schema8_viewer_bundle_roundtrip \
    tests/test_direction.py::test_v5018_read_only_exam_allows_10deg_real_plane_but_not_training_or_tilted_gravity \
    tests/test_direction.py::test_v5018_crossing_gate_accepts_only_supported_world_fixed_crossings \
    tests/test_direction.py::test_v5018_crossing_gate_rejects_incomplete_or_fabricated_support \
    tests/test_direction.py::test_v5018_stride_diagnostics_identifies_strict_gait_rejection_without_changing_pass

# First isolate exactly WHY a confirmed contact-stride count remains zero.
# Matched original 260M policy, real 10-degree surfaces, world-fixed 0.1m/s.
for course in flat uphill10 downhill10 cross_left10; do
  case "$course" in
    flat) angle=0; direction=uphill ;;
    uphill10) angle=10; direction=uphill ;;
    downhill10) angle=10; direction=downhill ;;
    cross_left10) angle=10; direction=cross_left ;;
  esac
  run "stride_${course}" emu directional-exam \
    --policy "$ROOT/showcase/champion.emupolicy" \
    --output "$OUT/stride_${course}" \
    --contact-slope --angle "$angle" --direction "$direction" \
    --worlds 4 --horizon 350 --settle-frames 25 \
    --speed .10 --seed 1701 --device cuda \
    --compare-heading --heading-kp .75
done

# Real collidable world-level 5mm/10mm step and a flat control. All are
# READ-ONLY actor runs. Each case runs in its own process/GPU allocation.
# A valid run can report zero successes; do not mislabel it PASS on crossing.
for seed in 1701 1702; do
  for height in 0 5 10; do
    for mode in off on; do
      # Use the second seed for obstacle and flat control with heading ON:
      # sufficient extra checks without repeating all OFF conditions.
      if [[ "$seed" == 1702 && "$mode" == off ]]; then continue; fi
      kp=0.
      if [[ "$mode" == on ]]; then kp=.75; fi
      title="step_${height}mm_${mode}_seed${seed}"
      run "$title" python "$ROOT/scripts/v5018_champion_step_pilot.py" \
        --height-mm "$height" --heading-kp "$kp" \
        --seed "$seed" --worlds 4 --horizon 350 \
        --output "$OUT/$title.json"
    done
  done
done

run aggregate_reports python "$ROOT/scripts/v5018_summarize_gait_and_step.py" "$OUT"

echo '============== SUMMARY =============='
cat "$OUT/stages.tsv"
tar -czf "$OUT.tar.gz" -C "$BASE" "$(basename "$OUT")"
echo "Archive: $OUT.tar.gz"
echo "Reports: $OUT"
if grep -q $'\tFAIL\t' "$OUT/stages.tsv"; then
  echo "Some experiments failed; preserve archive and inspect their logs."
  exit 1
fi
echo "All executables finished. Inspect actual crossing_success_worlds: zero is a valid research result."
