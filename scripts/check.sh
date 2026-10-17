#!/usr/bin/env bash
# One small, complete software safety check. GPU walking exams are separate.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pytest -q tests
python -m emu.cli doctor --no-require-gpu --no-mjx-smoke
python -m emu.cli ladder --stage plan >/dev/null
python -m emu.cli directional-exam --help >/dev/null
python -m emu.cli directional-batch --help >/dev/null
python -m emu.cli export-policy --help >/dev/null
npm test --prefix emu/viewer_src
