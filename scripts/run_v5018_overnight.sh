#!/usr/bin/env bash
# Branch-only v5018 cold-start + matched real-step overnight study.
# Plan is DEFAULT (no CUDA, no run directories). Pass --run explicitly.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"
if [[ ! -x .venv/bin/python ]]; then
  echo 'Missing .venv Python; see docs/INSTALL.md' >&2
  exit 2
fi
source .venv/bin/activate
exec python scripts/v5018_overnight_campaign.py "$@"
