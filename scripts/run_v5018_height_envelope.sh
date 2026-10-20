#!/usr/bin/env bash
# v5018 continuous obstacle-height study. Plan by default; no tmux.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ ! -x .venv/bin/python ]]; then
  echo 'Missing WSL .venv/bin/python; follow docs/INSTALL.md' >&2
  exit 2
fi
source .venv/bin/activate
exec python -m scripts.v5018_height_envelope_campaign "$@"
