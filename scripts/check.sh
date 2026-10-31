#!/usr/bin/env bash
# One small source/CLI check; extended CPU MJX and CUDA acceptance are separate.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m compileall -q emu tests scripts
python -m pytest -q tests
python -m emu.cli doctor --no-require-gpu --no-mjx-smoke
python -m emu.cli train --help >/dev/null
python -m emu.cli terrain-exam --help >/dev/null
python -m emu.cli export-policy --help >/dev/null
npm test --prefix emu/viewer_src
