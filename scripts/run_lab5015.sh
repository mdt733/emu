#!/usr/bin/env bash
# Compatibility entry point from v5015 before the human-first ladder rename.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
exec bash "$HERE/run_ladder.sh" "$@"
