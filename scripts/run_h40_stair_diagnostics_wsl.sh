#!/usr/bin/env bash
# EMU Spring V1: freeze trained 120M weights + optimiser, then diagnose ONLY.
# 2,000 signed real-prism stair attempts + 2,000 matched flat controls.
# No neural-network updates, spring modifications or source changes.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
if [[ "$(git branch --show-current)" != "work/emu-stair-mechanics-diagnostics" ]]; then
  echo "Wrong Git branch; use work/emu-stair-mechanics-diagnostics" >&2
  exit 1
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Git tree is dirty; refusing ambiguous physics/actor evidence" >&2
  exit 1
fi
# Previous recover scripts failed on WSL without a 'python' alias. Use the
# exact Python virtual environment that already ran 120M JAX PPO.
if [[ ! -x "$REPO/.venv/bin/python" ]]; then
  echo "EMU .venv missing; activate/provision the training environment first" >&2
  exit 1
fi
export EMU_PHYSICAL_PROFILE=h40-spring-v1
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
exec "$REPO/.venv/bin/python" -m scripts.examine_h40_spring_v1_mechanics
