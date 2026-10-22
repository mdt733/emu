#!/usr/bin/env bash
# EMU v5020: explicit experimental cold-start 2048-world training ladder.
# Branch-only. No main/showcase promotion. No tiny 64-world rehearsal.
set -Eeuo pipefail

usage() {
  cat <<'HELP'
Usage: bash scripts/run_v5020_2048.sh --plan|--run [--stop-after foundation|up|mixed12|mixed20]

2048 physical MJX environments per training phase, from random weights.
  foundation   20M flat-first 312-input foundation (frozen gait qualification)
  up           32M cumulative steps, opt-in real 2m-long step-up, <=6mm
  mixed12      48M cumulative steps, real up/down, <=12mm
  mixed20      70M cumulative steps, real up/down, <=20mm
These heights are SAFETY CEILINGS. The curriculum earns actual heights from
post-landing gait/contact evidence. Flat walking is rehearsed in every phase.

--plan reads only; --run launches actual 2048-world PPO and writes checkpoints.
By default all phases run sequentially. Each completed phase has its own
source-pinned directory and survives restarts. Rerun the same command after
a failure: it uses --resume auto when validated same-phase snapshots exist.
The foundation must pass its saved exact-step walking exam before steps start.

Optional: RUN_ROOT=/absolute/path (must be outside Git), STOP_AFTER=...
The previously gated scripts/overnight_v5020.sh --run stays BLOCKED; this
is a separate explicitly requested experimental 2048 campaign.
HELP
}
MODE="${1:---plan}"
[[ "$MODE" == --plan || "$MODE" == --run || "$MODE" == --help ]] || { usage >&2; exit 2; }
if [[ "$MODE" == --help ]]; then usage; exit 0; fi
STOP_AFTER="${STOP_AFTER:-mixed20}"
if [[ ${2+x} ]]; then
  if [[ "$2" != --stop-after || -z "${3:-}" || ${4+x} ]]; then
    usage >&2; exit 2
  fi
  STOP_AFTER="$3"
fi
case "$STOP_AFTER" in foundation|up|mixed12|mixed20) ;; *) echo "Invalid STOP_AFTER=$STOP_AFTER" >&2; exit 2;; esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
BRANCH="$(git branch --show-current)"
[[ "$BRANCH" == "feature/v5020-coldstart-terrain" ]] ||
  { echo "STOP: wrong branch $BRANCH; main is protected." >&2; exit 1; }
SHA="$(git rev-parse HEAD)"
SHORT_SHA="${SHA:0:12}"
ENVS=2048
SEED=733
RUN_ROOT="${RUN_ROOT:-$HOME/robot_lab/emu_runs/v5020_2048_$SHORT_SHA}"
[[ "$RUN_ROOT" == /* ]] || { echo "RUN_ROOT must be absolute" >&2; exit 2; }
case "$RUN_ROOT/" in "$ROOT/"*) echo "Run output must be outside source checkout" >&2; exit 1 ;; esac
printf '\nEMU v5020 2048-world staged cold-start\n'
printf 'Branch: %s\nSource: %s\nOutput: %s\nWorlds: %s (ALL phases)\n' "$BRANCH" "$SHA" "$RUN_ROOT" "$ENVS"
printf 'Sequence: 20M flat, 32M up <=6mm, 48M mixed <=12mm, 70M mixed <=20mm\n'
printf 'Requested last stage: %s\n' "$STOP_AFTER"
if [[ "$MODE" == --plan ]]; then
  printf '\nPlanning only. NO training, checkpoints, or run directories created.\n'
  exit 0
fi
[[ -z "$(git status --porcelain)" ]] ||
  { echo "STOP: commit/stash source changes before source-pinned training" >&2; git status --short; exit 1; }
[[ -f .venv/bin/activate ]] ||
  { echo "STOP: virtualenv missing. Run setup_wsl.sh in this checkout." >&2; exit 1; }
# shellcheck disable=SC1091
source .venv/bin/activate
command -v emu >/dev/null ||
  { echo "STOP: emu CLI unavailable after venv activation" >&2; exit 1; }
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/lineage"
if [[ -f "$RUN_ROOT/lineage/source_commit.txt" ]] &&
   [[ "$(cat "$RUN_ROOT/lineage/source_commit.txt")" != "$SHA" ]]; then
  echo "STOP: source SHA differs from this run's original source" >&2; exit 1
fi
printf '%s\n' "$SHA" > "$RUN_ROOT/lineage/source_commit.txt"

step() {
  local name="$1" target="$2" parent="$3"
  shift 3
  local dir="$RUN_ROOT/$name" ck="$RUN_ROOT/$name/checkpoints/final.pkl"
  local log="$RUN_ROOT/logs/${name}_$(date +%Y%m%d_%H%M%S).log"
  local lineage="$RUN_ROOT/lineage/$name.parent.sha256"
  local parent_sha="COLDSTART"
  if [[ "$parent" != coldstart ]]; then
    [[ -f "$parent" ]] || { echo "STOP: missing parent $parent" >&2; exit 1; }
    parent_sha="$(sha256sum "$parent" | cut -d' ' -f1)"
  fi
  if [[ -f "$lineage" ]]; then
    [[ "$(cat "$lineage")" == "$parent_sha" ]] || {
      echo "STOP: $name parent checkpoint SHA changed, refusing different lineage" >&2
      exit 1
    }
  else
    printf '%s\n' "$parent_sha" > "$lineage"
  fi
  local -a resume=()
  if [[ -d "$dir/checkpoints" ]] &&
     find "$dir/checkpoints" -maxdepth 1 -type f -name '*.pkl' -print -quit | grep -q .; then
    resume=(--resume auto)
  elif [[ "$parent" != coldstart ]]; then
    resume=(--resume "$parent")
  fi
  echo
  echo "=== $name: 2048 envs, target $target total transitions ==="
  echo "Parent: $parent ($parent_sha)"
  echo "Log: $log"
  # "emu train" is intentionally foreground/synchronous and has pipefail.
  # It writes run-local checkpoints, metrics, fixed evals and console.log.
  XLA_PYTHON_CLIENT_PREALLOCATE=false emu train --run-dir "$dir" --steps "$target" \
    --envs "$ENVS" --seed "$SEED" --device cuda \
    "${resume[@]}" "$@" 2>&1 | tee -a "$log"
  [[ -s "$ck" ]] || { echo "STOP: $name returned without final checkpoint" >&2; exit 1; }
  echo "=== Completed $name: $ck ==="
}
echo "Source/policy/checkpoint contracts are strict. Any failure stops this"
echo "campaign with its console logs, stage output, and prior snapshots intact."

step foundation 20000000 coldstart --profile foundation --forward-preview
if [[ "$STOP_AFTER" == foundation ]]; then echo "Requested stop after foundation"; exit 0; fi

# Fail CLOSED if a structurally correct foundation only learned to stand.
# The evaluator checks the EXACT saved checkpoint's simultaneous commanded
# travel + gait + survival; it is not a generic evaluation score.
PARENT="$(python - "$RUN_ROOT/foundation" <<'PY'
import sys
from pathlib import Path
from emu.foundation_readiness import choose_fresh_foundation
selection=choose_fresh_foundation(Path(sys.argv[1]))
print(selection['path'])
PY
)"
echo "Qualified foundation walker: $PARENT"

step up 32000000 "$PARENT"   --profile sim2real --forward-preview --pitch-deg 8   --terrain-varied-placement --terrain-elevation-mode up   --terrain-height-ceiling-mm 6 --terrain-height-initial-mm 3   --terrain-height-exposure .40 --terrain-height-edge-fraction .30   --terrain-progress-weight .015
if [[ "$STOP_AFTER" == up ]]; then echo "Requested stop after up"; exit 0; fi

step mixed12 48000000 "$RUN_ROOT/up/checkpoints/final.pkl"   --profile sim2real --forward-preview --pitch-deg 8   --terrain-varied-placement --terrain-elevation-mode mixed   --terrain-height-ceiling-mm 12 --terrain-height-initial-mm 3   --terrain-height-exposure .55 --terrain-height-edge-fraction .35   --terrain-progress-weight .025
if [[ "$STOP_AFTER" == mixed12 ]]; then echo "Requested stop after mixed12"; exit 0; fi

step mixed20 70000000 "$RUN_ROOT/mixed12/checkpoints/final.pkl"   --profile sim2real --forward-preview --pitch-deg 8   --terrain-varied-placement --terrain-elevation-mode mixed   --terrain-height-ceiling-mm 20 --terrain-height-initial-mm 3   --terrain-height-exposure .65 --terrain-height-edge-fraction .40   --terrain-progress-weight .035

echo
echo "=== v5020 experimental 2048-world ladder complete ==="
echo "Results: $RUN_ROOT"
echo "Latest: $RUN_ROOT/mixed20/checkpoints/final.pkl"
echo "IMPORTANT: completing 70M transitions does NOT certify 20mm stairs."
echo "Run independent frozen up/down exams before declaring a new policy champion."
