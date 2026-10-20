#!/usr/bin/env bash
# v5018 P1: bounded reward-only terrain geometry acceptance.
# Tests complete independently of the 260M champion, and never touches
# main, showcase, existing checkpoints or training output roots.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"
if [[ "$(git branch --show-current)" != "feature/v5018-terrain-awareness" ]]; then
  echo "STOP: must be on the approved feature/v5018-terrain-awareness branch" >&2
  exit 2
fi
if [[ ! -x .venv/bin/python ]]; then
  echo "STOP: missing $ROOT/.venv/bin/python" >&2
  exit 2
fi
source .venv/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

BASE="$HOME/robot_lab/emu_exams"
mkdir -p "$BASE"
OUT="$BASE/v5018_p1_terrain_$(date +%Y%m%d_%H%M%S)"
if [[ -e "$OUT" ]]; then
  echo "STOP: experiment output already exists: $OUT" >&2
  exit 2
fi
mkdir "$OUT"
printf 'stage\tstatus\texit_code\n' > "$OUT/stages.tsv"
archive_on_exit() {
  local code="$?"
  trap - EXIT
  if tar -czf "$OUT.tar.gz" -C "$BASE" "$(basename "$OUT")"; then
    echo "Evidence: $OUT.tar.gz (shell exit $code)"
  else
    echo "WARNING: failed to archive $OUT; preserve that directory" >&2
  fi
}
trap archive_on_exit EXIT
exec 2> >(tee -a "$OUT/runner_stderr.log" >&2)

run() {
  local stage="$1"; shift
  echo "========== $stage ==========" | tee -a "$OUT/campaign.log"
  set +e
  (cd "$ROOT" && "$@") 2>&1 | tee "$OUT/$stage.log"
  local code="${PIPESTATUS[0]}"
  set -e
  if [[ "$code" -eq 0 ]]; then
    printf '%s\tPASS\t0\n' "$stage" >> "$OUT/stages.tsv"
  else
    printf '%s\tFAIL\t%s\n' "$stage" "$code" >> "$OUT/stages.tsv"
    echo "STOP: stage $stage failed ($code); all earlier evidence preserved" >&2
  fi
  return "$code"
}

{
  echo "Study: physically grounded P1 box-surface reward truth; champion-independent"
  date -Is
  echo "Source commit: $(git rev-parse HEAD)"
  echo "Current branch: $(git branch --show-current)"
  git status --short
  python --version
  python -c 'import jax,mujoco; print("JAX",jax.__version__,"MuJoCo",mujoco.__version__,"devices",jax.devices())'
  nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader || true
  sha256sum emu/robot.xml
} | tee "$OUT/environment.txt"

# These fast checks fail before GPU compilation, including the new
# explicit from-zero/no-champion regression for both observation schemas.
run syntax python -m py_compile emu/terrain.py emu/environment.py \
  scripts/v5018_p1_surface_smoke.py
run physical_terrain_truth python -m pytest -q \
  tests/test_v5018_terrain_truth.py
run legacy_and_cold_start_contracts python -m pytest -q \
  tests/test_v5018_cold_start_contract.py \
  tests/test_learning.py::test_weighted_reward_is_bitwise_equivalent_to_original_float32_expression \
  tests/test_learning.py::test_training_config_roundtrip

# Actual GPU/MJX stepping on the copied same-XML 0mm off-course control,
# verifying bitwise legacy scalar reward and identical 232 actor obs.
# Then probe a physically collidable 10mm challenge and report active
# contact/clearance metrics. No pretrained actor is loaded here.
run actual_mjx_reward_parity python scripts/v5018_p1_surface_smoke.py \
  --worlds 4 --device cuda --output "$OUT/mjx_surface_truth.json"

echo '============ Stage summary ============'
cat "$OUT/stages.tsv"
echo "PASS: bounded terrain measurement/reward acceptance, NOT hardware safe stepping"
echo "Archive will be written on exit: $OUT.tar.gz"
