#!/usr/bin/env bash
# Friendly entry point; actual source verification and GPU smoke are shared.
# This keeps the old validated wrapper available for existing instructions.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
exec bash "$HERE/run_lab5015.sh" "$@"
