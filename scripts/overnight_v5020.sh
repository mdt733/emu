#!/usr/bin/env bash
# One guarded entry point for the first 312-input five-way evening experiment.
# Does not alter main, a source checkout, old run roots, or the public demo.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"

case "${1:---check}" in
  --check|--run) MODE="${1:---check}" ;;
  --help|-h)
    cat <<'HELP'
Usage: bash scripts/overnight_v5020.sh [--check|--run]
  --check   Validate the current WSL checkout, 312-input foundation,
            regression tests and GPU. No PPO training, no new run root.
  --run     If no suitable saved foundation exists, train a fresh flat-first
            312-input 20M foundation and re-examine it. Only after it passes
            source screening run the five-arm 20->40M Champion Ladder round.
            Stop after its independent exam; do not launch 60/80M.

Optional environment overrides:
  FOUNDATION_PATH=...   Explicit saved 312-input 12-22M .pkl.
                        If unset: try a validated mature best.pkl,
                        falling back to final.pkl, under
                        ~/robot_lab/emu_runs/v5020_lab_4096_fixed/foundation20
                        or ~/robot_lab/emu_runs/v5020_lab_4096/foundation20
  RUN_ROOT=...          New output dir (or resume an identical, pinned study).
                        Defaults to ~/robot_lab/emu_runs/v5020_refork_<git-sha>
  EXAM_ENVS=32          Number of frozen-policy exam worlds (>=8).
  EXAM_HORIZON=300      Frozen-policy exam frames (>=100).
No silent reduction from 4096 to 2048 worlds: the foundation config
determines the required world count.
HELP
    exit 0 ;;
  *) echo "Usage: $0 [--check|--run]. See --help" >&2; exit 2 ;;
esac
if [[ ${2+x} ]]; then echo "Only one --check/--run argument is allowed" >&2; exit 2; fi

[[ -f .venv/bin/activate ]] || {
  echo "STOP: no venv in $ROOT. Follow docs/INSTALL.md (setup_wsl.sh) first." >&2
  exit 1
}
# shellcheck disable=SC1091
source .venv/bin/activate

BRANCH="$(git branch --show-current)"
[[ "$BRANCH" == "feature/v5020-coldstart-terrain" ]] || {
  echo "STOP: wrong Git branch: $BRANCH; main must remain untouched." >&2
  exit 1
}
[[ -z "$(git status --porcelain)" ]] || {
  echo "STOP: source checkout has local changes. Preserve them first." >&2
  git status --short
  exit 1
}
REVISION="$(git rev-parse HEAD)"
SHORT_SHA="${REVISION:0:12}"
# If no checkpoint was explicitly specified, prefer the saved policy
# judged best in the original foundation run (when it is mature and matches
# schema-9), rather than blindly using a possibly deteriorated final.pkl.
EXPLICIT_FOUNDATION="${FOUNDATION_PATH:-}"
RUN_ROOT="${RUN_ROOT:-$HOME/robot_lab/emu_runs/v5020_refork_$SHORT_SHA}"
EXAM_ENVS="${EXAM_ENVS:-32}"
EXAM_HORIZON="${EXAM_HORIZON:-300}"
[[ "$EXAM_ENVS" =~ ^[0-9]+$ && "$EXAM_HORIZON" =~ ^[0-9]+$ ]] || {
  echo "STOP: EXAM_ENVS and EXAM_HORIZON must be positive integers" >&2; exit 1
}
((EXAM_ENVS>=8 && EXAM_HORIZON>=100)) || {
  echo "STOP: not enough exam worlds or frames" >&2; exit 1
}

# Read-only source investigation; every candidate passes the exact saved
# observation/sensor contract and 12–22M physical-step maturity gate.
# A score from another checkpoint step never validates this policy.
echo "=== Checkout: $BRANCH @ $REVISION ==="
FOUNDATION_NEEDS_BUILD=0
FRESH_ROOT="$HOME/robot_lab/emu_runs/v5020_flatfirst_foundation_$SHORT_SHA"
if FOUNDATION_META="$(python - "$EXPLICIT_FOUNDATION" "$HOME" <<'PY'
import sys
from pathlib import Path
from emu.foundation_readiness import choose_foundation
specified,home=sys.argv[1],Path(sys.argv[2])
candidates=[
    home/"robot_lab/emu_runs/v5020_lab_4096_fixed/foundation20",
    home/"robot_lab/emu_runs/v5020_lab_4096/foundation20",
]
choice=choose_foundation(candidates,explicit=Path(specified) if specified else None)
print("FOUNDATION_PATH="+choice["path"])
print("FOUNDATION_WORLDS="+str(choice["envs"]))
print("FOUNDATION_STEP="+str(choice["step"]))
print("FOUNDATION_SHA256="+choice["sha256"])
print("FOUNDATION_SELECTION="+choice["selection_reason"])
exam=choice["source_evaluation"]
if exam is None:
    print("FOUNDATION_FIXED_EXAM=UNKNOWN: no exact checkpoint-step fixed evaluation")
else:
    print("FOUNDATION_FIXED_EXAM="+", ".join(f"{k}={v:.4f}" for k,v in sorted(exam.items())))
PY
)"; then
  printf '%s\n' "$FOUNDATION_META"
  FOUNDATION_PATH="$(printf '%s\n' "$FOUNDATION_META" | sed -n 's/^FOUNDATION_PATH=//p')"
  ENVS="$(printf '%s\n' "$FOUNDATION_META" | sed -n 's/^FOUNDATION_WORLDS=//p')"
  [[ -f "$FOUNDATION_PATH" && "$ENVS" =~ ^[0-9]+$ ]] || {
    echo "STOP: selected foundation has no verified path/world count" >&2; exit 1
  }
else
  if [[ -n "$EXPLICIT_FOUNDATION" ]]; then
    echo "STOP: explicitly requested checkpoint is invalid/unhealthy; no silent substitution." >&2
    exit 1
  fi
  FOUNDATION_NEEDS_BUILD=1
  ENVS="${FOUNDATION_ENVS:-4096}"
  FOUNDATION_PATH="$FRESH_ROOT/checkpoints/final.pkl"
  echo "=== No qualified source: will create/validate a fresh flat-first 312-input foundation ==="
  echo "New foundation run: $FRESH_ROOT"
  echo "This is an explicit opt-in only when --run is used; --check launches no training."
fi
[[ "$ENVS" =~ ^[0-9]+$ ]] || { echo "STOP: invalid foundation world count: $ENVS" >&2; exit 1; }
if (( ENVS != 2048 && ENVS != 4096 )); then
  echo "STOP: unsupported overnight GPU world count $ENVS; use a reviewed study." >&2
  exit 1
fi

# Running a different training lineage with different source/config in the
# same root is forbidden by the underlying campaign; do not bypass its guard.
if [[ -e "$RUN_ROOT" && ! -f "$RUN_ROOT/refork_context.json" ]]; then
  echo "STOP: chosen run root already exists without a refork identity: $RUN_ROOT" >&2
  exit 1
fi
echo "=== Run destination: $RUN_ROOT ==="
echo "=== STOP target: walk40 (five 2M health probes + surviving continuations) ==="
# Show what the source foundation actually did in its last saved fixed
# evaluations. A valid checkpoint is not proof that it learned healthy gait.
PARENT_DIR="$(dirname "$(dirname "$FOUNDATION_PATH")")"
if [[ -f "$PARENT_DIR/fixed_eval.csv" ]]; then
  python - "$PARENT_DIR/fixed_eval.csv" <<'PY'
import csv, sys
from pathlib import Path
with Path(sys.argv[1]).open(newline="",encoding="utf-8") as fd:
    rows=list(csv.DictReader(fd))
print("=== Foundation fixed-exam reference (last 3 saved rows) ===")
for row in rows[-3:]:
    keys=("step","eval_policy_step","eval_broad_score",
          "eval_core_survival","eval_dynamic80_survival",
          "eval_combo70_survival","eval_resilience75_survival")
    print("  "+" ".join(f"{key}={row[key]}" for key in keys if row.get(key) not in (None,"")))
if not rows:
    print("  No saved fixed-evaluation rows; checkpoint quality remains unproven.")
PY
else
  echo "WARNING: no fixed_eval.csv next to foundation; snapshot was validated structurally, NOT for gait health."
fi
echo "Do not equate checkpoint validity or source CI with a strong foundation."

echo "=== Focused source regressions (no training) ==="
python -m pytest -q \
  tests/test_v5020_refork.py tests/test_v5020_coldstart.py \
  tests/test_scoring.py tests/test_competence_shadow.py \
  tests/test_experiment_io.py tests/test_foundation_readiness.py \
  tests/test_overnight_launcher.py
echo "=== Live MJX/CUDA doctor (small smoke, no training) ==="
XLA_PYTHON_CLIENT_PREALLOCATE=false emu doctor --v5015-smoke --domain-smoke
echo "=== Research plan (read-only) ==="
emu campaign-refork

if [[ "$MODE" == "--check" ]]; then
  if ((FOUNDATION_NEEDS_BUILD)); then
    echo "CHECK PASSED: no qualified existing foundation; --run will train a fresh flat-first 312-input foundation FIRST."
  else
    echo "CHECK PASSED: existing source checkpoint validated."
  fi
  echo "No campaign was started. Use --run to launch."
  exit 0
fi

# Logs live OUTSIDE the run root because the campaign insists its new root
# contain no unknown files. nohup/tmux is a terminal/session choice,
# not silently implemented by this safety launcher.
LOG_DIR="$HOME/robot_lab/emu_launch_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/v5020_refork_${SHORT_SHA}_$(date +%Y%m%d_%H%M%S).log"
if ((FOUNDATION_NEEDS_BUILD)); then
  [[ "$ENVS" =~ ^[0-9]+$ ]] || { echo "STOP: invalid FOUNDATION_ENVS" >&2; exit 1; }
  if [[ -e "$FRESH_ROOT" && ! -f "$FRESH_ROOT/training_config.json" ]]; then
    echo "STOP: new foundation directory is nonempty/unidentified: $FRESH_ROOT" >&2; exit 1
  fi
  echo "=== Training a new 312-input, flat-first 20M foundation: $FRESH_ROOT ==="
  echo "It starts with attainable level walking: no contact-slope randomisation, no early steps."
  # An interruption may resume ONLY within the same pinned revision.
  RESUME=()
  if [[ -d "$FRESH_ROOT/checkpoints" ]] &&
       find "$FRESH_ROOT/checkpoints" -maxdepth 1 -type f -name '*.pkl' | grep -q .; then
    RESUME=(--resume auto)
  fi
  XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 \
    emu train --profile foundation --forward-preview \
      --envs "$ENVS" --seed 733 --device cuda \
      --run-dir "$FRESH_ROOT" --steps 20000000 "${RESUME[@]}" 2>&1 | tee -a "$LOG_FILE"

  # Reassess the exact completed policy. A saved 20M checkpoint alone does
  # NOT authorise five more expensive challengers.
  FOUNDATION_META="$(python - "$FRESH_ROOT" <<'PY'
import sys
from pathlib import Path
from emu.foundation_readiness import choose_foundation
p=Path(sys.argv[1]).resolve()
best=choose_foundation([p])
print("FOUNDATION_PATH="+best["path"])
print("FOUNDATION_WORLDS="+str(best["envs"]))
print("FOUNDATION_STEP="+str(best["step"]))
print("FOUNDATION_SELECTION="+best["selection_reason"])
print("FOUNDATION_SHA256="+best["sha256"])
print("FOUNDATION_FIXED_EXAM="+str(best["source_evaluation"]))
PY
)"
  printf '%s\n' "$FOUNDATION_META"
  FOUNDATION_PATH="$(printf '%s\n' "$FOUNDATION_META" | sed -n 's/^FOUNDATION_PATH=//p')"
  echo "=== Fresh foundation passed screening; proceeding to first 40M generation ==="
fi
echo "=== Starting five-way first round; live log: $LOG_FILE ==="
echo "If every challenger fails, campaign exits without promoting anything."
echo "A validated winner, if any, will be in $RUN_ROOT/walk40/selection.json"
XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 \
  emu campaign-refork --run \
    --foundation "$FOUNDATION_PATH" \
    --envs "$ENVS" --seed 733 \
    --screen-envs "$EXAM_ENVS" --screen-horizon "$EXAM_HORIZON" \
    --stop-after walk40 --root "$RUN_ROOT" 2>&1 | tee -a "$LOG_FILE"
