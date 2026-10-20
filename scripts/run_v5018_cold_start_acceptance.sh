#!/usr/bin/env bash
# Optional v5018 "erase every brain" acceptance: produce two completely
# NEW PPO checkpoints and browser exports using source, RNG seed and
# fresh curriculum only. This is a one-update lifecycle smoke, NOT proof
# of a trained walking policy. NO SHOWCASE OR PRETRAINED FILE IS OPENED.
#
# Run only when you explicitly want to verify from-scratch GPU lifecycle:
#   bash scripts/run_v5018_cold_start_acceptance.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"

if [[ "$(git branch --show-current)" != "feature/v5018-terrain-awareness" ]]; then
  echo 'Use the approved v5018 feature branch, never main' >&2
  exit 2
fi
if [[ ! -x .venv/bin/python ]]; then
  echo 'Missing virtual environment; read docs/INSTALL.md' >&2
  exit 2
fi
source .venv/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Git provenance is a training contract: the real emu train launcher
# rejects a dirty checkout. Leave user results alone, fix ignore rules
# or archive local files deliberately rather than deleting them.
if [[ -n "$(git status --porcelain)" ]]; then
  echo 'Source checkout is dirty; inspect git status --short before a reproducible run' >&2
  git status --short >&2
  exit 2
fi

BASE="$HOME/robot_lab/emu_experiments"
mkdir -p "$BASE"
OUT="$BASE/v5018_cold_start_$(date +%Y%m%d_%H%M%S)"
if [[ -e "$OUT" ]]; then
  echo "Experiment folder exists; refusing overwrite: $OUT" >&2
  exit 2
fi
mkdir "$OUT"
printf 'stage\tstatus\texit_code\n' > "$OUT/stages.tsv"

archive_on_exit() {
  local exit_code="$?"
  trap - EXIT
  if tar -czf "$OUT.tar.gz" -C "$BASE" "$(basename "$OUT")"; then
    echo "Result archive: $OUT.tar.gz (shell exit $exit_code)"
  else
    echo "Could not create archive; preserve raw folder $OUT" >&2
  fi
}
trap archive_on_exit EXIT

{
  echo 'v5018 cold-start invariant: absolutely no parent network'
  date -Is
  echo "Git source: $(git rev-parse HEAD)"
  echo "Branch: $(git branch --show-current)"
  python -c 'import jax,mujoco; print("JAX",jax.__version__,"MuJoCo",mujoco.__version__,"devices",jax.devices())'
  nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader || true
} | tee "$OUT/environment.txt"

run() {
  local name="$1"; shift
  echo "========== $name ==========" | tee -a "$OUT/campaign.log"
  set +e
  (cd "$ROOT" && "$@") 2>&1 | tee "$OUT/$name.log"
  local status="${PIPESTATUS[0]}"
  set -e
  printf '%s\t%s\t%s\n' "$name" "$([[ "$status" -eq 0 ]] && echo PASS || echo FAIL)" "$status" >> "$OUT/stages.tsv"
  if [[ "$status" -ne 0 ]]; then
    echo "Failed $name; archiving existing evidence and STOPPING" >&2
    return "$status"
  fi
}

run fresh_contracts python -m pytest -q tests/test_v5018_cold_start_contract.py

for schema in legacy220 sensor232; do
  extra=()
  dim=220
  if [[ "$schema" == sensor232 ]]; then
    extra+=(--sensor-actor)
    dim=232
  fi
  # No --resume of any kind. Each run_dir did not exist before execution.
  run "train_from_zero_${schema}" emu train \
    --run-dir "$OUT/$schema" \
    --steps 256 --envs 8 --seed 733 --device cuda \
    --profile foundation "${extra[@]}"
  run "check_zero_lineage_${schema}" python - "$OUT/$schema" "$dim" <<'PY'
import json, pathlib, sys
from emu.checkpoint_io import load_checkpoint
run = pathlib.Path(sys.argv[1])
dim = int(sys.argv[2])
meta = json.loads((run/'run_meta.json').read_text())
assert meta.get('resumed_from_step') is None, 'TRAINER IMPORTED A PARENT'
checkpoint = load_checkpoint(run/'checkpoints/final.pkl',validate=True)
assert int(checkpoint['meta']['step']) == 256
weights = checkpoint['state']['params']['pi'][0]['w']
assert int(weights.shape[0]) == dim, (weights.shape, dim)
print(f'PASS no-parent checkpoint {dim}-input @ 256 steps, saved under {run}')
PY
  run "export_new_brain_${schema}" emu export-policy \
    "$OUT/$schema/checkpoints/final.pkl" "$OUT/${schema}_fresh.emupolicy"
done
echo '==========================='
cat "$OUT/stages.tsv"
echo 'COLD-START LIFECYCLE PASS: random network -> actual PPO -> checkpoint -> export for 220 AND 232'
echo 'One update does not teach locomotion. It proves the path needs no stored champion.'
