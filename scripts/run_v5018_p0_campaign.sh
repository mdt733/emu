#!/usr/bin/env bash
# EMU v5018 complete P0 feasibility campaign.
# Run ONLY from the dedicated feature-branch checkout, outside any active
# trainer process. All reports are saved outside Git and compressed at end.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
if [[ ! -f "$ROOT/scripts/v5018_batch_benchmark.py" ]]; then
  echo "Wrong checkout: v5018 feasibility scripts missing" >&2
  exit 2
fi
if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  echo "Missing virtualenv: $ROOT/.venv/bin/python" >&2
  exit 2
fi
# shellcheck source=/dev/null
source "$ROOT/.venv/bin/activate"
if [[ -s "$HOME/.nvm/nvm.sh" ]]; then
  export NVM_DIR="$HOME/.nvm"
  # shellcheck source=/dev/null
  source "$NVM_DIR/nvm.sh" --no-use
  if [[ -f "$ROOT/.nvmrc" ]]; then
    nvm use "$(cat "$ROOT/.nvmrc")" >/dev/null || true
  fi
fi

RUN_ROOT="$HOME/robot_lab/emu_experiments"
mkdir -p "$RUN_ROOT"
OUT="$RUN_ROOT/v5018_p0_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
printf 'stage\tstatus\texit_code\n' > "$OUT/stages.tsv"

run() {
  local stage="$1"
  shift
  echo
  echo "========== $stage ==========" | tee -a "$OUT/campaign.log"
  # Continue independent probes even after failures. GPU escalation within
  # the benchmark stops at its first failing batch size.
  (cd "$OUT" && "$@") 2>&1 | tee "$OUT/$stage.log"
  local code="${PIPESTATUS[0]}"
  if [[ "$code" -eq 0 ]]; then
    printf '%s\tPASS\t0\n' "$stage" >> "$OUT/stages.tsv"
  else
    printf '%s\tFAIL\t%s\n' "$stage" "$code" >> "$OUT/stages.tsv"
  fi
  echo "$stage: exit=$code" | tee -a "$OUT/campaign.log"
}

{
  echo "EMU v5018 P0 campaign"
  date -Is
  echo "Branch: $(git -C "$ROOT" branch --show-current)"
  echo "Commit: $(git -C "$ROOT" rev-parse HEAD)"
  echo "Python: $(python --version 2>&1)"
  echo "Node: $(node --version 2>&1 || true)"
  echo "npm: $(npm --version 2>&1 || true)"
  echo "Git checkout status:"
  git -C "$ROOT" status --short
  echo "GPU:"
  nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader 2>&1 || true
} | tee "$OUT/environment.txt"

if [[ "$(git -C "$ROOT" branch --show-current)" != "feature/v5018-terrain-awareness" ]]; then
  echo "STOP: This script only runs from the approved feature branch." >&2
  exit 2
fi

echo "Results folder: $OUT"
# Full tests are intentional: do not silently substitute a subset of the
# 78 Python test cases that previously disclosed the stale 220-input examiner.
run py_compile python -m compileall -q "$ROOT/scripts"
run python_tests python -m pytest -q "$ROOT/tests"
run npm_ci npm --prefix "$ROOT/emu/viewer_src" ci
run npm_tests npm --prefix "$ROOT/emu/viewer_src" test
run npm_build npm --prefix "$ROOT/emu/viewer_src" run build
run gpu_doctor_and_ppo_smoke emu doctor --v5015-smoke --domain-smoke
run contact_ray_v3 python "$ROOT/scripts/v5018_terrain_spike.py" --batch
run wasm_cpu_gap node "$ROOT/emu/viewer_src/scripts/v5018_wasm_contact.mjs" "$OUT/v5018_p0a_spike_v3_results.json"
run moving_contacts python "$ROOT/scripts/v5018_dynamic_contact.py"
run per_world_geometry python "$ROOT/scripts/v5018_per_world_terrain.py"
run scanner_coverage python "$ROOT/scripts/v5018_scanner_coverage.py"
run clearance_kinematics python "$ROOT/scripts/v5018_clearance_envelope.py"
run full_emu_gpu_batches python "$ROOT/scripts/v5018_batch_benchmark.py" \
  --envs 256 512 1024 2048 --frames 8 --repeats 5 \
  --cases flat step contact --output "$OUT/emu_v5018_batch_campaign.json"

echo
echo "============== SUMMARY =============="
cat "$OUT/stages.tsv"
tar -czf "$OUT.tar.gz" -C "$RUN_ROOT" "$(basename "$OUT")"
echo "Report archive: $OUT.tar.gz"
echo "Individual files: $OUT"
if grep -q $'\tFAIL\t' "$OUT/stages.tsv"; then
  echo "One or more gates failed. Archive includes failure diagnostics."
  exit 1
fi
echo "All campaign commands exited zero. Physical P0 exit STILL requires evidence review."
