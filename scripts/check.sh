#!/usr/bin/env bash
# One obvious place to choose the depth of verification.
set -euo pipefail
cd "$(dirname "$0")/.."
mode="${1:-quick}"
case "$mode" in
  quick)
    # A carefully chosen set of behavioural contracts; no actual GPU.
    mapfile -t cases < <(grep -vE '^[[:space:]]*(#|$)' tests/quick_cases.txt)
    python -m pytest -q "${cases[@]}"
    ;;
  full) python -m pytest -q ;;
  physics) python -m pytest -q tests/test_v5016_exam.py::test_real_mjx_evaluator_trace_shapes_cpu ;;
  browser) npm test --prefix emu/viewer_src ;;
  all)
    python -m pytest -q
    npm test --prefix emu/viewer_src
    ;;
  *)
    echo "Usage: bash scripts/check.sh [quick|full|physics|browser|all]" >&2
    exit 2 ;;
esac
