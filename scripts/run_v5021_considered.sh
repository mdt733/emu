#!/usr/bin/env bash
# V5021: true cold-start, torque-aware, 312-input considered walker.
# Branch only. Each stage resumes exactly the previous source-pinned learning state.
set -Eeuo pipefail

usage() {
cat <<'HELP'
EMU v5021 Considered Walker, true cold start (2048 MJX worlds, seed 733)
Usage: bash scripts/run_v5021_considered.sh --plan|--run [--stop-after s10|s25|s50|s100]
  --plan: print stage command and verify CLI shape; do not start GPU.
  --run: automatically train 0->10M->25M->50M->100M cumulative steps.
  --stop-after: stop cleanly after a chosen stage.
Stages are continued full PPO/checkpoint/curriculum/normalizer state, not forks.
Full 312-input actor from step zero, V memory unmasked, randomized physical
step-up platforms (up to 150mm technical MJX geometry maximum), initially 0mm
rated, evidence-based growing speed/yaw/terrain frontiers, motor-aware pay,
wide numeric safety bounds rather than 0.4m/s or 6mm lesson ceilings.
No pretrained policy and no changes to main or any v5020 checkpoint.
CURRENT LIMITATIONS: step-up geometry only, real varied slopes still separate;
not yet a verified sensor-dependence result; actuator/physics validity safety
limits and 2.6Hz phase cap remain. No claim of unlimited physical speed.
Safe interruption: CTRL+C, later rerun --run. It resumes partial stages only
from validated same-source checkpoints. Never switch this worktree mid-run.
For another terminal's viewer, use:
  python -m emu.cli viewer --live --root "<RUN_ROOT>" --port 8766
HELP
}

MODE="${1:---plan}"
if [[ "$MODE" == --help ]]; then usage; exit 0; fi
[[ "$MODE" == --plan || "$MODE" == --run ]] || { usage >&2; exit 2; }
STOP_AFTER=s100
if (( $# > 1 )); then
  [[ $# == 3 && "$2" == --stop-after ]] || { usage >&2; exit 2; }
  STOP_AFTER="$3"
fi
case "$STOP_AFTER" in s10|s25|s50|s100) ;; *) echo "Invalid stop stage" >&2; exit 2;; esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"
BRANCH="$(git branch --show-current)"
[[ "$BRANCH" == feature/v5021-considered-walker ]] ||
  { echo "STOP: must run from feature/v5021-considered-walker, got $BRANCH" >&2; exit 1; }
SHA="$(git rev-parse HEAD)"
RUN_ROOT="${RUN_ROOT:-$HOME/robot_lab/emu_runs/v5021_considered_${SHA:0:12}}"
[[ "$RUN_ROOT" == /* && "$RUN_ROOT" != "$HOME" ]] ||
  { echo "STOP: RUN_ROOT must be a separate absolute output directory" >&2; exit 1; }
case "$RUN_ROOT/" in "$ROOT/"*) echo "STOP: training under Git checkout" >&2; exit 1;; esac
echo "Source: $SHA"
echo "Output: $RUN_ROOT"
echo "Schedule: s10=10M, s25=25M, s50=50M, s100=100M absolute steps."
echo "Fresh initialization, no source policy or external parent."
echo "Live viewer: python -m emu.cli viewer --live --root '$RUN_ROOT' --port 8766"

PROFILE=(
  --profile considered --forward-preview --pitch-deg 8
  --terrain-varied-placement --terrain-elevation-mode up
  --terrain-height-ceiling-mm 150 --terrain-height-initial-mm 0
  --terrain-height-exposure 0.40 --terrain-height-edge-fraction 0.30
  --terrain-height-growth-mm 1
  --terrain-progress-weight 0.04
)

if [[ "$MODE" == --plan ]]; then
  for pair in "s10:10000000" "s25:25000000" "s50:50000000" "s100:100000000"; do
    local_name="${pair%%:*}"; local_target="${pair##*:}"
    printf 'stage=%s target=%s\n' "$local_name" "$local_target"
  done
  if [[ -f "$HOME/emu-v5020/.venv/bin/activate" ]]; then
    source "$HOME/emu-v5020/.venv/bin/activate"
    PYTHONPATH="$ROOT" python -m emu.cli train \
      --run-dir "$RUN_ROOT/s10" --steps 10000000 \
      --envs 2048 --seed 733 --device cuda "${PROFILE[@]}" --plan
  fi
  exit 0
fi

[[ -z "$(git status --porcelain)" ]] ||
  { echo "STOP: dirty source tree. Commit first for reproducibility." >&2; exit 1; }
if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  source "$ROOT/.venv/bin/activate"
elif [[ -f "$HOME/emu-v5020/.venv/bin/activate" ]]; then
  source "$HOME/emu-v5020/.venv/bin/activate"
else
  echo "STOP: cannot find existing WSL .venv; run setup first." >&2
  exit 1
fi
python -c 'import pytest, numpy, jax, mujoco; print("Python/JAX/MuJoCo imports OK")'
bash -n "$ROOT/scripts/run_v5021_considered.sh"
python -m compileall -q emu
PYTHONPATH="$ROOT" python -m pytest -q \
  tests/test_v5021_considered.py tests/test_v5020_rated_elevation.py \
  tests/test_v5020_console_feedback.py ||
  { echo "STOP: V5021 preflight failed before CUDA PPO"; exit 1; }

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/lineage"
PIN="$RUN_ROOT/lineage/source_commit.txt"
[[ ! -f "$PIN" || "$(cat "$PIN")" == "$SHA" ]] ||
  { echo "STOP: existing campaign pinned to a different source commit"; exit 1; }
printf '%s\n' "$SHA" > "$PIN"

run_stage() {
  local name="$1" target="$2" parent="$3"
  local dir="$RUN_ROOT/$name" final="$RUN_ROOT/$name/checkpoints/final.pkl"
  local -a resume=()
  if [[ -d "$dir/checkpoints" ]] &&
     find "$dir/checkpoints" -maxdepth 1 -type f -name '*.pkl' -print -quit | grep -q .; then
    resume=(--resume auto)
  elif [[ -n "$parent" ]]; then
    [[ -s "$parent" ]] || { echo "STOP: missing prior final $parent" >&2; exit 1; }
    resume=(--resume "$parent")
  fi
  [[ ! -f "$dir/UNSAFE_RESUME.json" ]] ||
    { echo "STOP: quarantined stage: $name" >&2; exit 1; }

  echo "=== V5021 $name => $target cumulative steps; resume=${resume[*]:-cold start} ==="
  local log="$RUN_ROOT/logs/${name}_$(date +%Y%m%d_%H%M%S).log"
  XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH="$ROOT" \
    python -m emu.cli train --run-dir "$dir" \
      --steps "$target" --envs 2048 --seed 733 --device cuda \
      "${PROFILE[@]}" "${resume[@]}" 2>&1 | tee -a "$log"
  [[ -s "$final" ]] || { echo "STOP: missing final checkpoint $final"; exit 1; }
  echo "Completed $name; checkpoint SHA256 $(sha256sum "$final" | cut -d' ' -f1)"
  printf '%s\n' "$(sha256sum "$final" | cut -d' ' -f1)" > "$RUN_ROOT/lineage/$name.final.sha256"
  echo "=== Regenerating read-only V5021 shareable results ==="
  PYTHONPATH="$ROOT" python -m emu.cli report --root "$RUN_ROOT"
}

run_stage s10 10000000 ""
[[ "$STOP_AFTER" == s10 ]] && exit 0
run_stage s25 25000000 "$RUN_ROOT/s10/checkpoints/final.pkl"
[[ "$STOP_AFTER" == s25 ]] && exit 0
run_stage s50 50000000 "$RUN_ROOT/s25/checkpoints/final.pkl"
[[ "$STOP_AFTER" == s50 ]] && exit 0
run_stage s100 100000000 "$RUN_ROOT/s50/checkpoints/final.pkl"
echo "V5021 campaign completed; results in $RUN_ROOT"
