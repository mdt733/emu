#!/usr/bin/env bash
# v5018 branch-only, independent blind-box 0mm/5mm PPO plumbing A/B.
# By default PRINTS the plan. Use --run ONLY after the P1 deep-audit PASS.
# This is a reproducibility gate, not a claim of learned obstacle negotiation.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"
RUN=0; ENVS=256; STEPS=8192; SEED=733
while (($#)); do
  case "$1" in
    --run) RUN=1; shift ;;
    --envs) ENVS="${2:?missing --envs value}"; shift 2 ;;
    --steps) STEPS="${2:?missing --steps value}"; shift 2 ;;
    --seed) SEED="${2:?missing --seed value}"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ "$(git branch --show-current)" != 'feature/v5018-terrain-awareness' ]]; then
  echo 'Requires the approved feature branch; no main edits.' >&2; exit 2
fi
if [[ ! -x .venv/bin/python ]]; then
  echo 'Expected source .venv; see docs/INSTALL.md' >&2; exit 2
fi
if ! [[ "$ENVS" =~ ^[0-9]+$ && "$STEPS" =~ ^[0-9]+$ && "$SEED" =~ ^[0-9]+$ ]] ||
   (( ENVS < 2 || ENVS > 2048 || STEPS < 8192 || STEPS > 2100000 )); then
  echo 'Use 2..2048 envs, 8192..2100000 total steps, nonnegative seed.' >&2; exit 2
fi
source .venv/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
BASE="$HOME/robot_lab/emu_experiments"
echo "Branch: $(git branch --show-current)"
echo "Commit: $(git rev-parse HEAD)"
echo "One-run total env steps: $STEPS; worlds: $ENVS; seed: $SEED"
if (( ! RUN )); then
  echo 'PLAN ONLY (no CUDA update or files written):'
  for H in 0 5; do
    emu train --run-dir "$BASE/PLAN_height_${H}mm" --steps "$STEPS" --envs "$ENVS" --seed "$SEED" --device cuda --profile sim2real --sensor-actor --terrain-box-height-mm "$H" --plan
  done
  echo 'For a new archive and actual experiments: bash scripts/run_v5018_step_ppo_probe.sh --run'
  exit 0
fi
if [[ -n "$(git status --porcelain)" ]]; then
  git status --short >&2
  echo 'Dirty source checkout: will not train or create results.' >&2; exit 2
fi
mkdir -p "$BASE"
OUT="$BASE/v5018_step_ppo_$(date +%Y%m%d_%H%M%S)"
if [[ -e "$OUT" || -e "$OUT.tar.gz" ]]; then
  echo "Experiment root already exists: $OUT" >&2; exit 2
fi
mkdir "$OUT"
printf 'stage\tstatus\texit_code\n' > "$OUT/stages.tsv"
archive_on_exit() {
  local ec="$?"
  trap - EXIT
  # Keep huge checkpoints on user's disk; zip only reproducible logs/metadata.
  if tar -czf "$OUT.tar.gz" --exclude='*.pkl' -C "$BASE" "$(basename "$OUT")"; then
    echo "Evidence archive: $OUT.tar.gz (exit $ec)"
  else
    echo "Tar failed: preserve raw results at $OUT" >&2
  fi
}
trap archive_on_exit EXIT
run_stage() {
  local label="$1"; shift
  echo "========== $label ==========" | tee -a "$OUT/campaign.log"
  set +e
  (cd "$ROOT" && "$@") 2>&1 | tee "$OUT/${label}.log"
  local rc="${PIPESTATUS[0]}"
  set -e
  printf '%s\t%s\t%d\n' "$label" "$([[ "$rc" -eq 0 ]] && echo PASS || echo FAIL)" "$rc" >> "$OUT/stages.tsv"
  if (( rc != 0 )); then
    echo "Failed $label (exit=$rc); do not interpret incomplete metrics." >&2
    return "$rc"
  fi
}
{
  date -Is
  git rev-parse HEAD
  git status --short
  sha256sum showcase/champion.emupolicy
  nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader || true
  python -c 'import jax,mujoco; print("JAX",jax.__version__,"MJ",mujoco.__version__,"devices",jax.devices())'
} > "$OUT/environment.txt" 2>&1
run_stage geometry_contracts python -m pytest -q tests/test_v5018_terrain_training.py tests/test_direction.py::test_v5018_step_pilot_xml_yaw_and_declared_height
for H in 0 5; do
  run_stage "train_${H}mm" emu train --run-dir "$OUT/train_${H}mm" --steps "$STEPS" --envs "$ENVS" --seed "$SEED" --device cuda --profile sim2real --sensor-actor --terrain-box-height-mm "$H"
  CK="$OUT/train_${H}mm/checkpoints/final.pkl"
  [[ -s "$CK" ]] || { echo "Missing final checkpoint: $CK" >&2; exit 1; }
  run_stage "export_${H}mm" emu export-policy "$CK" "$OUT/policy_${H}mm.emupolicy"
  for TEST in 0 5 10; do
    for S in 1701 1702; do
      run_stage "exam_${H}mm_vs_${TEST}mm_seed${S}" python scripts/v5018_champion_step_pilot.py --policy "$OUT/policy_${H}mm.emupolicy" --height-mm "$TEST" --heading-kp .75 --seed "$S" --speed .10 --worlds 4 --horizon 350 --output "$OUT/exam_${H}mm_vs_${TEST}mm_seed${S}.json"
    done
  done
done
sha256sum showcase/champion.emupolicy > "$OUT/protected_policy_after.txt"
echo 'All processes exited zero; examine physical crossing_pass, collisions and servo load before claiming improvement.'
cat "$OUT/stages.tsv"
