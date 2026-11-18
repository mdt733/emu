#!/usr/bin/env bash
# Resume the failed 20M H40 spring experiment from its LAST VALID checkpoint.
# NEVER delete/overwrite the original ~/emu-data/runs/h40-spring-v1-outcomes-s735.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
[[ "$(git branch --show-current)" == "work/emu-mechanical-feasibility" ]] || {
  echo "Wrong Git branch: use work/emu-mechanical-feasibility" >&2; exit 1;
}
[[ -z "$(git status --porcelain)" ]] || {
  echo "Git is dirty; refusing source-ambiguous recovery" >&2; exit 1;
}
export EMU_PHYSICAL_PROFILE=h40-spring-v1
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export EMU_UNIFIED_CONTINUOUS=cold-growth-approved
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python -m scripts.recover_h40_spring_v1
