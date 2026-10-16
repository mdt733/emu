#!/usr/bin/env bash
# One small, complete software safety check. GPU walking exams are separate.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pytest -q tests
python -m emu.cli doctor --no-require-gpu --no-mjx-smoke
npm test --prefix emu/viewer_src
