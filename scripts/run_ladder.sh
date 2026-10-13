#!/usr/bin/env bash
# Single current entry point. Validate source and CUDA/MJX/PPO before training.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"
unset PYTHONPATH
if [[ ! -x .venv/bin/emu || ! -x .venv/bin/python ]]; then
  echo "Missing local .venv; complete the installation first." >&2
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
echo "[EMU] Checking JAX/MJX and one real PPO update before training."
./.venv/bin/emu doctor --v5015-smoke --domain-smoke
echo "[EMU] GPU smoke passed. Starting the checkpoint-safe training ladder."
exec ./.venv/bin/emu ladder --stage auto "$@"
