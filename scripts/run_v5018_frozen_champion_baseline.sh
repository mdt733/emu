#!/usr/bin/env bash
# v5018: frozen-champion 232-input heading ON/OFF baseline on real slopes.
# Read-only exams. No PPO training and no showcase modifications.
# Intentionally separates source checkout from all experiment outputs.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
if [[ "$(git -C "$ROOT" branch --show-current)" != "feature/v5018-terrain-awareness" ]]; then
  echo "STOP: requires the approved v5018 feature branch" >&2
  exit 2
fi
if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  echo "STOP: missing $ROOT/.venv/bin/python" >&2
  exit 2
fi
source "$ROOT/.venv/bin/activate"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

OUTPUT_ROOT="$HOME/robot_lab/emu_exams"
mkdir -p "$OUTPUT_ROOT"
OUT="$OUTPUT_ROOT/v5018_heading_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
printf 'stage\tstatus\texit_code\n' > "$OUT/stages.tsv"

run() {
  local name="$1"; shift
  echo "========== $name ==========" | tee -a "$OUT/campaign.log"
  (cd "$ROOT" && "$@") 2>&1 | tee "$OUT/$name.log"
  local code="${PIPESTATUS[0]}"
  if [[ "$code" -eq 0 ]]; then
    printf '%s\tPASS\t0\n' "$name" >> "$OUT/stages.tsv"
  else
    printf '%s\tFAIL\t%s\n' "$name" "$code" >> "$OUT/stages.tsv"
  fi
  echo "$name exit=$code" | tee -a "$OUT/campaign.log"
}

{
 echo "EMU v5018 frozen-policy slope/heading baseline"
 date -Is
 git -C "$ROOT" rev-parse HEAD
 git -C "$ROOT" status --short
 python --version
 python -c 'import jax,mujoco; print("JAX",jax.__version__,"MuJoCo",mujoco.__version__,"device",jax.devices())'
 nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader 2>&1 || true
 sha256sum "$ROOT/showcase/champion.emupolicy"
} | tee "$OUT/environment.txt"

POLICY="$ROOT/showcase/champion.emupolicy"
if [[ ! -f "$POLICY" ]]; then
 echo "Missing protected champion; no baseline exams executed" >&2
 exit 2
fi

# Focused tests that failed in the large archive and the newly widened,
# explicitly read-only, real-contact slope exam. No seven-minute full
# suite repetition just to fix stale assertions.
run corrected_contracts python -m pytest -q \
 tests/test_learning.py::test_training_config_roundtrip \
 tests/test_learning.py::test_v5016_schema8_viewer_bundle_roundtrip \
 tests/test_direction.py::test_protected_export_reader_refuses_contract_and_actor_tampering \
 tests/test_direction.py::test_v5018_read_only_exam_allows_10deg_real_plane_but_not_training_or_tilted_gravity

# The large P0 batch only ran the 220-input PPO smoke. Explicitly compile
# and run the *232-input* opt-in PPO/update smoke before frozen exams.
run sensor232_ppo_smoke emu doctor --v5016-smoke --no-mjx-smoke

# Four-world matched physical exams. Fixed original commands and source
# identity across heading controller OFF/ON, never direct motor control.
# Each case is an independent read-only exam with its own output folder.
# A failure is preserved and does not invalidate other completed cases.
for seed in 1701 1702; do
  for fixture in flat uphill10 downhill10 cross_left10; do
    case "$fixture" in
      flat) angle=0; direction=uphill ;;
      uphill10) angle=10; direction=uphill ;;
      downhill10) angle=10; direction=downhill ;;
      cross_left10) angle=10; direction=cross_left ;;
    esac
    run "champion_${fixture}_seed${seed}" emu directional-exam \
      --policy "$POLICY" --output "$OUT/${fixture}_seed${seed}" \
      --contact-slope --angle "$angle" --direction "$direction" \
      --worlds 4 --horizon 350 --settle-frames 25 \
      --speed .10 --scenario steady --seed "$seed" \
      --device cuda --compare-heading --heading-kp .75
  done
done

# Directional obedience can fail despite surviving; examine actual requested
# turns and reversals under the same controller, with no terrain handicap.
for scenario in reverse turn_stop_left; do
  run "champion_${scenario}_seed1701" emu directional-exam \
    --policy "$POLICY" --output "$OUT/${scenario}_seed1701" \
    --contact-slope --angle 0 --direction uphill \
    --worlds 4 --horizon 350 --settle-frames 25 \
    --speed .10 --scenario "$scenario" --seed 1701 \
    --device cuda --compare-heading --heading-kp .75
done

echo '============== SUMMARY =============='
cat "$OUT/stages.tsv"
tar -czf "$OUT.tar.gz" -C "$OUTPUT_ROOT" "$(basename "$OUT")"
echo "Archive: $OUT.tar.gz"
echo "Reports: $OUT"
if grep -q $'\tFAIL\t' "$OUT/stages.tsv"; then
 echo "One or more stages failed; inspect logs; do not claim baseline passed."
 exit 1
fi
echo "All commands completed. Interpret reported course outcomes before claiming walking capability."
