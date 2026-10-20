#!/usr/bin/env bash
# v5018 viewer challenge-ring acceptance (NO PPO, NO champion edits).
# Runs all existing browser tests plus the new geometry/waypoint tests,
# checks pinned MuJoCo/WASM collision on real box/log fixtures, builds and
# installs the local viewer package. Captures all results outside Git.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"
if [[ "$(git branch --show-current)" != "feature/v5018-terrain-awareness" ]]; then
  echo 'Use the protected v5018 feature branch, never main' >&2;exit 2
fi
if [[ ! -x .venv/bin/python ]]; then
  echo 'Activate/setup ~/emu-v5018/.venv before running this campaign' >&2;exit 2
fi
source .venv/bin/activate
export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
if [[ -s "$NVM_DIR/nvm.sh" ]]; then
  # shellcheck disable=SC1091
  source "$NVM_DIR/nvm.sh"
fi
command -v npm >/dev/null || { echo 'Missing npm; read docs/INSTALL.md' >&2; exit 2; }
command -v node >/dev/null || { echo 'Missing Node; read docs/INSTALL.md' >&2; exit 2; }
BASE="$HOME/robot_lab/emu_exams"
mkdir -p "$BASE"
OUT="$BASE/v5018_viewer_course_$(date +%Y%m%d_%H%M%S)"
[[ ! -e "$OUT" ]] || { echo "Refusing overwrite: $OUT" >&2; exit 2; }
mkdir "$OUT"
printf 'stage\tstatus\texit_code\n' > "$OUT/stages.tsv"

archive_exit() {
  local status="$?"
  trap - EXIT
  if tar -czf "$OUT.tar.gz" -C "$BASE" "$(basename "$OUT")"; then
    echo "Archive: $OUT.tar.gz (exit $status)"
  else
    echo "WARNING: couldn't archive $OUT; preserve raw results" >&2
  fi
}
trap archive_exit EXIT
exec 2> >(tee -a "$OUT/runner_stderr.log" >&2)

run() {
  local stage="$1";shift
  echo "========== $stage ==========" | tee -a "$OUT/campaign.log"
  set +e
  (cd "$ROOT" && "$@") 2>&1 | tee "$OUT/$stage.log"
  local status="${PIPESTATUS[0]}"
  set -e
  printf '%s\t%s\t%s\n' "$stage" "$([[ "$status" -eq 0 ]] && echo PASS || echo FAIL)" "$status" >> "$OUT/stages.tsv"
  if [[ "$status" -ne 0 ]]; then
    echo "Stage failed: $stage; preserved evidence and stopped before browser publication" >&2
  fi
  return "$status"
}
{
  echo "Study: physical ring obstacles, sequential waypoint demo, WASM collision and browser packaging"
  date -Is
  echo "Branch: $(git branch --show-current)"
  echo "Source SHA: $(git rev-parse HEAD)"
  git status --short
  node --version
  npm --version
  python --version
  sha256sum showcase/champion.emupolicy
} | tee "$OUT/environment.txt"

run source_syntax node --check emu/viewer_src/src/main.js
run nav_worker_syntax node --check emu/viewer_src/src/goal-worker.js
run ring_module_syntax node --check emu/viewer_src/src/challenge-course.js
run browser_deps npm --prefix emu/viewer_src ci --no-audit --no-fund
run browser_tests npm --prefix emu/viewer_src test
run actual_wasm_ring_collisions node \
  emu/viewer_src/scripts/v5018_wasm_challenge_smoke.mjs \
  "$OUT/wasm_collision_results.json"
run browser_build npm --prefix emu/viewer_src run build
run local_viewer_install emu viewer-build

echo "============== COMPLETE =============="
cat "$OUT/stages.tsv"
echo 'The viewer supports public-policy and local --live mode. This test'
echo 'proves course collision fixtures and web build, NOT full walking/browser parity.'
echo "Use: emu viewer showcase/champion.emupolicy --no-open"
echo "Or:  emu viewer --live --root /path/to/CURRENT/validated/training/run --no-open"
