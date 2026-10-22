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
  --run     Perform the same checks, then run ONLY the five-arm 20->40M
            Champion Ladder round; stop after independent examination.

Optional environment overrides:
  FOUNDATION_PATH=...   Source 312-input 12-22M .pkl. Defaults to
                        ~/robot_lab/emu_runs/v5020_lab_4096_fixed/foundation20/checkpoints/final.pkl
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
FOUNDATION_PATH="${FOUNDATION_PATH:-$HOME/robot_lab/emu_runs/v5020_lab_4096_fixed/foundation20/checkpoints/final.pkl}"
RUN_ROOT="${RUN_ROOT:-$HOME/robot_lab/emu_runs/v5020_refork_$SHORT_SHA}"
EXAM_ENVS="${EXAM_ENVS:-32}"
EXAM_HORIZON="${EXAM_HORIZON:-300}"

[[ -f "$FOUNDATION_PATH" ]] || {
  echo "STOP: foundation not found: $FOUNDATION_PATH" >&2
  echo "Choose a real, healthy 312-input 12-22M snapshot:" >&2
  find "$HOME/robot_lab/emu_runs" -type f -path '*/checkpoints/*.pkl' \
    \( -name 'best.pkl' -o -name 'final.pkl' -o -name 'snapshot*.pkl' \) \
    2>/dev/null | grep -E '(foundation|v5020)' | head -40 >&2 || true
  echo "Then retry: FOUNDATION_PATH=/real/validated/checkpoint.pkl bash scripts/overnight_v5020.sh --run" >&2
  exit 1
}
[[ "$EXAM_ENVS" =~ ^[0-9]+$ && "$EXAM_HORIZON" =~ ^[0-9]+$ ]] || {
  echo "STOP: EXAM_ENVS and EXAM_HORIZON must be positive integers" >&2; exit 1
}
((EXAM_ENVS>=8 && EXAM_HORIZON>=100)) || {
  echo "STOP: not enough exam worlds or frames" >&2; exit 1
}

# Validate checkpoint and complete sensor contract BEFORE creating any run root.
# _foundation loads and verifies the original saved policy using its own
# matching training_config.json, rather than guessing what 4096 means.
echo "=== Checkout: $BRANCH @ $REVISION ==="
FOUNDATION_META="$(python - "$FOUNDATION_PATH" <<'PY'
import sys
from pathlib import Path
from emu.campaign_refork import _foundation
from emu.training_config import TrainingConfig
ck=Path(sys.argv[1]).expanduser().resolve()
cfg=TrainingConfig.load(ck.parent.parent/'training_config.json')
item=_foundation(ck,envs=cfg.envs)
print(f"FOUNDATION_WORLDS={cfg.envs}")
print(f"FOUNDATION_STEP={item['step']}")
print(f"FOUNDATION_SHA256={item['sha256']}")
PY
)"
printf '%s\n' "$FOUNDATION_META"
ENVS="$(printf '%s\n' "$FOUNDATION_META" | sed -n 's/^FOUNDATION_WORLDS=//p')"
[[ "$ENVS" =~ ^[0-9]+$ ]] || {
  echo "STOP: foundation world count could not be verified" >&2; exit 1
}
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
  tests/test_experiment_io.py
echo "=== Live MJX/CUDA doctor (small smoke, no training) ==="
XLA_PYTHON_CLIENT_PREALLOCATE=false emu doctor --v5015-smoke --domain-smoke
echo "=== Research plan (read-only) ==="
emu campaign-refork

if [[ "$MODE" == "--check" ]]; then
  echo "CHECK PASSED. No campaign was started. Use --run to launch."
  exit 0
fi

# Logs live OUTSIDE the run root because the campaign insists its new root
# contain no unknown files. nohup/tmux is a terminal/session choice,
# not silently implemented by this safety launcher.
LOG_DIR="$HOME/robot_lab/emu_launch_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/v5020_refork_${SHORT_SHA}_$(date +%Y%m%d_%H%M%S).log"
echo "=== Starting five-way first round; live log: $LOG_FILE ==="
echo "If every challenger fails, campaign exits without promoting anything."
echo "A validated winner, if any, will be in $RUN_ROOT/walk40/selection.json"
XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 \
  emu campaign-refork --run \
    --foundation "$FOUNDATION_PATH" \
    --envs "$ENVS" --seed 733 \
    --screen-envs "$EXAM_ENVS" --screen-horizon "$EXAM_HORIZON" \
    --stop-after walk40 --root "$RUN_ROOT" 2>&1 | tee -a "$LOG_FILE"
