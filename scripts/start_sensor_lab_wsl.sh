#!/usr/bin/env bash
# One clean CPU-only Ubuntu/WSL viewer launcher.
# No Python ML venv or CUDA. With --policy PATH it previews a frozen actor;
# without arguments it starts the untrained sensor-mount laboratory.
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 is not installed. Run: sudo apt-get install -y python3" >&2
  exit 1
fi
python3 - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("EMU browser viewer requires Python 3.10+")
print("Python:", sys.version.split()[0])
PY

node_ok=false
if command -v node >/dev/null 2>&1; then
  if node -e "const [major,minor] = process.versions.node.split('.').map(Number);
             process.exit((major >= 22 || (major === 20 && minor >= 19)) ? 0 : 1);" >/dev/null 2>&1; then
    node_ok=true
  fi
fi
if [[ "$node_ok" != true ]]; then
  if ! command -v apt-get >/dev/null 2>&1; then
    echo "This script targets Ubuntu/WSL apt; install Linux Node.js 22 and retry." >&2
    exit 1
  fi
  echo "Installing Linux Node.js 22 (npm alone is not a Node runtime)..."
  sudo apt-get update
  sudo apt-get install -y curl ca-certificates
  curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
  sudo apt-get install -y nodejs
  hash -r
fi
printf 'Node: '
node --version
printf 'npm:  '
npm --version
echo "Building the browser viewer from the committed lockfile..."
python3 -m emu.cli viewer-build
if [[ "${1:-}" == "--build-only" ]]; then
  echo "Build-only complete."
  exit 0
fi
echo "Open in your Windows browser: http://127.0.0.1:8765/"
if [[ "${1:-}" == "--policy" ]]; then
  if [[ "$#" -ne 2 || ! -f "$2" ]]; then
    echo "Usage: bash scripts/start_sensor_lab_wsl.sh --policy /path/to/robot.emupolicy" >&2
    exit 2
  fi
  echo "FROZEN POLICY PREVIEW: $2 (CPU browser inference, no CUDA)"
  exec python3 -m emu.cli viewer "$2" --port 8765 --no-open
fi
if [[ "$#" -ne 0 ]]; then
  echo "Usage: bash scripts/start_sensor_lab_wsl.sh [--build-only|--policy PATH]" >&2
  exit 2
fi
exec python3 -m emu.cli viewer --sensor-lab --port 8765 --no-open
