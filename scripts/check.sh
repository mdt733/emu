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
# Always validate the deployed actor's physical/sensor contract, even while
# publication approval is pending. Source checks must not depend on Pages.
python - <<'PY'
import json
from pathlib import Path
from scripts.verify_showcase_release import validate_browser_bundle
policy = Path("showcase/champion.emupolicy")
identity = validate_browser_bundle(json.loads(policy.read_text(encoding="utf-8")))
print(f"EMU actor contract PASS: {identity['step']:,} steps; 360 inputs")
PY
# The publication gate remains strict: once an approval exists, verify it.
if [[ -f showcase/release_approval.json ]]; then
  python scripts/verify_showcase_release.py
else
  echo "Public deployment unapproved (no showcase/release_approval.json); Pages remains gated."
fi
python scripts/audit_core_dependencies.py --json >/dev/null
python -m emu.cli doctor --no-require-gpu --no-mjx-smoke
python -m emu.cli train --help >/dev/null
if ((FULL)); then
  python -m pytest -q tests
  npm test --prefix emu/viewer_src
fi
echo "EMU source checks complete (${1:---quick})."
