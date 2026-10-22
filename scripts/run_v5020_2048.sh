#!/usr/bin/env bash
# EMU v5020: explicit experimental cold-start 2048-world training ladder.
# Branch-only. No main/showcase promotion. No tiny 64-world rehearsal.
set -Eeuo pipefail

usage() {
  cat <<'HELP'
Usage: bash scripts/run_v5020_2048.sh --plan|--run [--stop-after foundation|up]

2048 physical MJX environments per training phase; cold-start or approved saved foundation.
  foundation   20M flat-first 312-input foundation (frozen gait qualification)
  up           32M cumulative steps, opt-in real 2m-long step-up, <=6mm
  downhill/mixed and slopes are intentionally NOT available yet
These heights are SAFETY CEILINGS. The curriculum earns actual heights from
post-landing gait/contact evidence. Flat walking is rehearsed in every phase.

--plan reads only; --run launches actual 2048-world PPO and writes checkpoints.
Default stop is the first raised step-up stage (32M), to examine survival.
Mixed stages are currently LOCKED: downhill must begin with its own
0mm skill rating and cannot inherit the trained uphill height.
The prior known-good 20M foundation is auto-discovered ONLY if the saved
SHA256 is exactly the previously qualified value. No damaged 24M child resume.
To select another qualified foundation explicitly:
  FOUNDATION_FROM=/absolute/path/to/foundation/checkpoints/best.pkl bash scripts/run_v5020_2048.sh --run
On a fresh machine, opt into cold-start explicitly with COLD_START=1.
Each completed phase has its own
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
STOP_AFTER="${STOP_AFTER:-up}"
if [[ ${2+x} ]]; then
  if [[ "$2" != --stop-after || -z "${3:-}" || ${4+x} ]]; then
    usage >&2; exit 2
  fi
  STOP_AFTER="$3"
fi
case "$STOP_AFTER" in foundation|up) ;; *) echo "Invalid STOP_AFTER=$STOP_AFTER" >&2; exit 2;; esac

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
printf 'Sequence: qualified 20M flat -> 32M rated up-step (6mm absolute safety cap)\n'
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
echo "=== Fast source gate: rating/evidence/lineage contracts (NO GPU training) ==="
python -m pytest -q tests/test_v5020_rated_elevation.py tests/test_v5020_2048_launcher.py ||
  { echo "STOP: new rated-terrain regression tests failed; no GPU run started" >&2; exit 1; }
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/lineage"
if [[ -f "$RUN_ROOT/lineage/source_commit.txt" ]] &&
   [[ "$(cat "$RUN_ROOT/lineage/source_commit.txt")" != "$SHA" ]]; then
  echo "STOP: source SHA differs from this run's original source" >&2; exit 1
fi
printf '%s\n' "$SHA" > "$RUN_ROOT/lineage/source_commit.txt"
trap 'code=$?; echo "2048 campaign stopped (exit $code). Check $RUN_ROOT/logs; run the SAME command again on this same source commit to resume." >&2' ERR

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
  # A 500-column raw CSV row is not a human-readable result. The offline
  # campaign report below owns the detailed shared evidence and warnings.
  echo "Completed stage $name; metrics and frozen exam preserved in $dir"
}
echo "Source/policy/checkpoint contracts are strict. Any failure stops this"
echo "campaign with its console logs, stage output, and prior snapshots intact."


# This operator previously qualified a 20M frozen 312-input walker on
# v5020 source 624bcb92.... Reuse ONLY those exact immutable bytes, never
# accidentally rerun 20M or inherit its catastrophically failed 'up' child.
# On a fresh computer with no known candidate, genuine cold-start remains
# possible. An existing path with an unexpected hash requires investigation.
KNOWN_PARENT="$HOME/robot_lab/emu_runs/v5020_2048_624bcb92db8c/foundation/checkpoints/best.pkl"
KNOWN_PARENT_SHA="23ca16ecd7de3b9980108d3833bb47dd8276f02aab6a2db35a3f30208dd4fc23"
if [[ -z "${FOUNDATION_FROM:-}" && -f "$KNOWN_PARENT" ]]; then
  actual="$(sha256sum "$KNOWN_PARENT" | cut -d' ' -f1)"
  if [[ "$actual" != "$KNOWN_PARENT_SHA" ]]; then
    echo "STOP: known previously-good foundation path was changed!" >&2
    echo "expected=$KNOWN_PARENT_SHA actual=$actual path=$KNOWN_PARENT" >&2
    exit 1
  fi
  FOUNDATION_FROM="$KNOWN_PARENT"
  echo "Reusing original 20M champion at exact verified SHA256: $FOUNDATION_FROM"
fi
if [[ -z "${FOUNDATION_FROM:-}" && "${COLD_START:-0}" != 1 ]]; then
  echo "STOP: no verified existing 20M foundation found." >&2
  echo "Set FOUNDATION_FROM=/full/path/to/qualified/best.pkl to select one," >&2
  echo "or COLD_START=1 to deliberately train a fresh 2048-world foundation." >&2
  exit 1
fi
if [[ -n "${FOUNDATION_FROM:-}" ]]; then
  [[ "$FOUNDATION_FROM" == /* && -f "$FOUNDATION_FROM" ]] ||
    { echo "STOP: FOUNDATION_FROM must be an absolute, existing .pkl checkpoint" >&2; exit 1; }
  echo "=== Checking existing flat 312-input foundation; NO foundation retraining ==="
  PARENT="$(python - "$FOUNDATION_FROM" "$ENVS" <<'PY'
from pathlib import Path
import sys
from emu.foundation_readiness import source_foundation, basic_foundation_issue
p=Path(sys.argv[1]).resolve()
source=source_foundation(p)
if source['envs'] != int(sys.argv[2]):
    raise SystemExit(f"bad foundation env count: {source['envs']}")
issue=basic_foundation_issue(source.get('source_evaluation'))
if issue:
    raise SystemExit(f"existing parent failed walking gate: {issue}")
print(str(p))
PY
)"
else
  step foundation 20000000 coldstart --profile foundation --forward-preview
  # Exact-checkpoint joint walking grade, not an optimistic reward proxy.
  PARENT="$(python - "$RUN_ROOT/foundation" <<'PY'
from pathlib import Path
import sys
from emu.foundation_readiness import choose_fresh_foundation
print(choose_fresh_foundation(Path(sys.argv[1]))['path'])
PY
)"
fi
echo "Qualified foundation walker: $PARENT"
if [[ "$STOP_AFTER" == foundation ]]; then
  echo "Requested stop after foundation"
  if [[ -f "$RUN_ROOT/foundation/metrics.csv" ]]; then
    emu report --root "$RUN_ROOT"
    echo "Share: $RUN_ROOT/results_bundle.zip"
  fi
  exit 0
fi
echo "=== HARD GATE: same frozen 312-input policy on original flat, zero plateau, and microprobes ==="
echo "This is a CPU/4-world policy examination only. No PPO updates."
python scripts/verify_v5020_bridge.py --parent "$PARENT" --worlds 4 --horizon 160 \
  2>&1 | tee "$RUN_ROOT/logs/bridge_parity_$(date +%Y%m%d_%H%M%S).log"
echo "Frozen transfer preflight passed. 2048-world policy learning is now permitted."
# Explicit one-time source transfer only: pin BOTH original code commit
# and checkpoint bytes. Trainer validates full 312-input sensor contract,
# optimizer, observation normaliser and 2048 environment lineage too.
PARENT_COMMIT="$(python - "$PARENT" <<'PY'
from pathlib import Path
from emu.checkpoint_io import load_checkpoint
from emu.training_config import TrainingConfig, training_sensor_contract
import sys
p=Path(sys.argv[1])
cfg=TrainingConfig.load(p.parent.parent/'training_config.json')
meta=load_checkpoint(p,validate=True,expected_observation_schema=cfg.observation_schema,
                     expected_sensor_contract=training_sensor_contract(cfg))['meta']
print(meta['source_provenance']['git_commit'])
PY
)"
APPROVAL=()
if [[ "$PARENT_COMMIT" != "$SHA" ]]; then
  [[ "$PARENT_COMMIT" =~ ^[0-9a-f]{40}$ ]] || {
    echo "STOP: imported foundation has no authenticated source commit" >&2; exit 1; }
  APPROVAL=(--approve-parent-commit "$PARENT_COMMIT"
            --approve-parent-sha256 "$(sha256sum "$PARENT" | cut -d' ' -f1)")
  echo "Explicit historical parent transfer from $PARENT_COMMIT:"
  echo "  checkpoint: $PARENT"
  echo "  sha256: ${APPROVAL[3]}"
  echo "  new source: $SHA"
fi

# Preserve the core-foundation command/phase/push/field economics through
# the first physical step; never switch to aggressive mature sim2real here.
# No new reward shaping at the boundary; the only sampled difference from
# the original flat episodes is a small, physically collidable probe cohort.
step up 32000000 "$PARENT" \
  --profile terrain-bridge --forward-preview --pitch-deg 8 \
  --terrain-varied-placement --terrain-elevation-mode up \
  --terrain-height-ceiling-mm 6 --terrain-height-initial-mm 0 \
  --terrain-height-exposure .40 --terrain-height-edge-fraction .30 \
  --terrain-progress-weight 0 "${APPROVAL[@]}"
echo
echo "=== v5020 first continuous terrain bridge complete ==="
echo "Output: $RUN_ROOT/up"
echo "Checkpoint: $RUN_ROOT/up/checkpoints/final.pkl"
echo
echo "=== Building CLEAN results.md + results_data.json + evidence ZIP ==="
# No JAX, no model deserialisation, no PPO, no mutation of checkpoints.
emu report --root "$RUN_ROOT"
echo
echo "=== Share these results (not checkpoint pkl files) ==="
echo "Markdown: $RUN_ROOT/results.md"
echo "Data:     $RUN_ROOT/results_data.json"
echo "Bundle:   $RUN_ROOT/results_bundle.zip"
echo "This is NOT proof of a successful 6mm transition. Independently"
echo "check retained flat walking and physically supported step-up exams."
