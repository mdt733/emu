#!/usr/bin/env bash
# Quick source sanity by default; full test suite only when explicitly asked.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
case "${1:---quick}" in
  --quick) FULL=0 ;;
  --full) FULL=1 ;;
  *) echo "Usage: bash scripts/check.sh [--quick|--full]" >&2; exit 2 ;;
esac
python -m compileall -q emu scripts tests
python scripts/verify_showcase_release.py
python scripts/audit_core_dependencies.py --json >/dev/null
python -m emu.cli doctor --no-require-gpu --no-mjx-smoke
python -m emu.cli train --help >/dev/null
if ((FULL)); then
  python -m pytest -q tests
  npm test --prefix emu/viewer_src
fi
echo "EMU source checks complete (${1:---quick})."
