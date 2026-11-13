#!/usr/bin/env bash
# Thin CPU-only browser launcher. Installation has one owner: setup_wsl.sh.
# This wrapper never installs apt packages, changes the checkout or trains.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

usage(){
  echo "Usage: bash scripts/start_sensor_lab_wsl.sh [--build-only|--policy PATH]" >&2
}
if ! command -v python3 >/dev/null || ! command -v node >/dev/null || ! command -v npm >/dev/null; then
  echo "Missing viewer prerequisites; run bash scripts/setup_wsl.sh --viewer-only --install-node" >&2
  exit 1
fi
python3 - <<'PY'
import sys
if sys.version_info<(3,10):
    raise SystemExit("EMU browser viewer requires Python >=3.10")
PY
node -e 'const n=Number(process.versions.node.split(".")[0]); process.exit(n>=24?0:1)' || {
  echo "Viewer needs Node 24; run bash scripts/setup_wsl.sh --viewer-only --install-node" >&2
  exit 1
}

if [[ "${1:-}" == "--build-only" && $# == 1 ]]; then
  exec python3 -m emu.cli viewer-build
fi
if [[ "${1:-}" == "--policy" && $# == 2 ]]; then
  [[ -f "$2" ]] || { echo "No such exported actor: $2" >&2; exit 1; }
  policy="$(realpath "$2")"
elif [[ $# == 0 ]]; then
  policy=""
else
  usage
  exit 2
fi

python3 -m emu.cli viewer-build
echo "Open http://127.0.0.1:8765/ in Windows; Ctrl+C stops this CPU-only viewer."
if [[ -n "$policy" ]]; then
  exec python3 -m emu.cli viewer "$policy" --no-open --port 8765
fi
exec python3 -m emu.cli viewer --sensor-lab --no-open --port 8765
