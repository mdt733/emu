#!/usr/bin/env bash
# Launch v5015 from the correct checkout even if the shell has a stale PYTHONPATH.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"
unset PYTHONPATH
if [[ ! -x .venv/bin/emu || ! -x .venv/bin/python ]]; then
  echo "Missing local .venv; finish the v5015 installation before training." >&2
  exit 2
fi
./.venv/bin/python - "$ROOT" <<'PY'
import pathlib, sys
import emu.cli
from emu.paths import VERSION
expected = pathlib.Path(sys.argv[1], 'emu', 'cli.py').resolve()
actual = pathlib.Path(emu.cli.__file__).resolve()
if VERSION != 'v5015' or actual != expected:
    raise SystemExit(f'Wrong EMU import: {actual} version={VERSION}; expected {expected}')
print(f'Using EMU {VERSION} from {actual}', flush=True)
PY
echo "[v5015] Running real PPO-update and MJX smoke before GPU training."
./.venv/bin/emu doctor --v5015-smoke --domain-smoke
echo "[v5015] Smoke passed; running the eligibility-gated short experiment funnel."
exec ./.venv/bin/emu lab5015 --stage auto "$@"
