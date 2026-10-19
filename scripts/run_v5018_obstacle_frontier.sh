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
# --resume preserves every completed stage AND its original source revision.
# Never silently re-run the earlier good 10-degree GPU examinations.
if [[ "${1:-}" == "--resume" ]]; then
  [[ $# == 2 ]] || { echo "Usage: $0 [--resume EXISTING_OUTPUT_DIR]" >&2; exit 2; }
  OUT="$(realpath "$2")"
  case "$OUT" in
    "$BASE"/v5018_frontier_*) ;;
    *) echo "Refusing resume outside $BASE/v5018_frontier_*" >&2; exit 2 ;;
  esac
  [[ -d "$OUT" && -s "$OUT/stages.tsv" && -s "$OUT/environment.txt" ]] || {
    echo "Existing campaign directory/stages/environment missing: $OUT" >&2
    exit 2
  }
  current_policy="$(sha256sum "$ROOT/showcase/champion.emupolicy" | cut -d' ' -f1)"
  grep -Fq "$current_policy" "$OUT/environment.txt" || {
    echo "Protected champion policy hash differs from original campaign; refusing resume" >&2
    exit 2
  }
  printf '%s\t%s\t%s\n' "$(date -Is)" "$(git -C "$ROOT" rev-parse HEAD)" "resume_after_runner_bugfix" >> "$OUT/revision_history.tsv"
  echo "Resuming saved campaign: $OUT"
  echo "Original execution commit remains recorded in environment.txt; resumed tests use the CURRENT commit." | tee -a "$OUT/campaign.log"
else
  [[ $# == 0 ]] || { echo "Usage: $0 [--resume EXISTING_OUTPUT_DIR]" >&2; exit 2; }
  OUT="$BASE/v5018_frontier_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$OUT"
  printf 'stage\tstatus\texit_code\n' > "$OUT/stages.tsv"
  {
    echo "Study: frozen 232-input champion real-step orientation/speed/height frontier"
    date -Is
    git -C "$ROOT" rev-parse HEAD
    git -C "$ROOT" status --short
    sha256sum "$ROOT/showcase/champion.emupolicy"
    python -c 'import jax,mujoco;print("JAX",jax.__version__,"MuJoCo",mujoco.__version__,"devices",jax.devices())'
    nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader 2>&1 || true
  } | tee "$OUT/environment.txt"
  printf '%s\t%s\t%s\n' "$(date -Is)" "$(git -C "$ROOT" rev-parse HEAD)" "original_campaign_start" >> "$OUT/revision_history.tsv"
fi

# Archive even if an unexpected shell error aborts the run. The original
# unarchived directory remains untouched, including old successful stages.
archive_on_exit() {
  local status="$?"
  trap - EXIT
  if tar -czf "$OUT.tar.gz" -C "$BASE" "$(basename "$OUT")"; then
    echo "Saved results archive: $OUT.tar.gz (script exit $status)"
  else
    echo "WARNING: could not archive $OUT; preserve raw results directory" >&2
  fi
}
trap archive_on_exit EXIT
# Capture unexpected shell errors as evidence (including set -u failures
# outside individually logged stages). Preserve live terminal visibility.
exec 2> >(tee -a "$OUT/runner_stderr.log" >&2)

already_completed() {
  local stage="$1"
  [[ -f "$OUT/$stage.log" ]] || return 1
  awk -F '\t' -v name="$stage" '$1 == name && $2 == "PASS" { ok=1 } END { exit !ok }' "$OUT/stages.tsv" || return 1
  # Only skip if the corresponding artifact exists. A prior log and
  # incorrectly marked exit code must not hide missing outputs.
  case "$stage" in
    stride_*) [[ -f "$OUT/$stage/heading_comparison.json" ]] ;;
    step_*) [[ -f "$OUT/$stage.json" ]] ;;
    aggregate_reports) [[ -f "$OUT/v5018_gait_and_step_summary.json" ]] ;;
    *) return 0 ;;
  esac
}

run() {
  local stage="$1"; shift
  if already_completed "$stage"; then
    echo "SKIP completed stage: $stage (saved evidence retained)" | tee -a "$OUT/campaign.log"
    return 0
  fi
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
  code_yaw="${code_yaw/+/p}"
  local stage="step_${h}mm_yaw${code_yaw}_speed${code_speed}_${mode}_seed${seed}"
  run "$stage" python "$ROOT/scripts/v5018_champion_step_pilot.py" \
    --height-mm "$h" --step-yaw-deg "$yaw" --speed "$speed" \
    --heading-kp "$kp" --seed "$seed" --worlds 4 \
    --horizon 350 --output "$OUT/$stage.json"
}

# Tiny no-GPU runner preflight: execute filename expansion for every
# parameter shape, and refuse duplicates before launching the frontier.
# This specifically guards against unbound/obsolete shell variable names.
check_step_naming() {
  local h yaw speed seed kp mode code_speed code_yaw stage
  local -A names=()
  for h in 0 5 10 15 20; do
    for yaw in 0 -25 25; do
      for speed in .07 .10 .14; do
        for seed in 1701 1702; do
          for kp in 0 .75; do
            mode=on
            if [[ "$kp" == 0 ]]; then mode=off; fi
            code_speed="${speed/./p}"
            code_yaw="${yaw/-/m}"
            code_yaw="${code_yaw/+/p}"
            stage="step_${h}mm_yaw${code_yaw}_speed${code_speed}_${mode}_seed${seed}"
            if [[ -n "${names[$stage]+seen}" ]]; then
              echo "Duplicate frontier experiment name: $stage" >&2
              return 2
            fi
            names[$stage]=1
          done
        done
      done
    done
  done
  echo "Frontier naming preflight PASS: ${#names[@]} unique combinations"
}
check_step_naming || exit 2

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
echo "Archive will be finalized on exit: $OUT.tar.gz"
echo "Records: $OUT"
if grep -q $'\tFAIL\t' "$OUT/stages.tsv"; then
  echo "Some research stages failed; preserve the archive and diagnose separately."
  exit 1
fi
echo "All scripts exited successfully. Examine crossing counts; this is NOT a terrain-training pass."
