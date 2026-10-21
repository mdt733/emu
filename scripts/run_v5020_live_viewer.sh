#!/usr/bin/env bash
# Viewer-only sidecar for the v5020 4096-world training tournament.
# Runs a local browser WASM robot and reads validated checkpoints.
# Never imports a PPO trainer or writes to the training repository/run.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
viewer_dir="$(pwd -P)"
python_bin="${EMU_V5020_PYTHON:-$HOME/emu-v5020/.venv/bin/python}"
root="${1:-$HOME/robot_lab/emu_runs/v5020_lab_4096}"
port="${EMU_V5020_VIEWER_PORT:-8765}"
if [[ ! -x "$python_bin" ]];then
  echo "Expected the existing training Python virtualenv at: $python_bin" >&2
  echo "Set EMU_V5020_PYTHON to the absolute Python path if it moved." >&2
  exit 1
fi
if [[ ! -d "$root" ]];then
  echo "No training root at $root. Pass your existing campaign root as argument." >&2
  exit 1
fi
if ! [[ "$port" =~ ^[0-9]+$ ]] || ((port < 1024 || port > 65535));then
  echo "EMU_V5020_VIEWER_PORT must be 1024..65535" >&2
  exit 2
fi
# Source imports ONLY from the spectator checkout; dependencies and ABI are
# reused from the already functioning training .venv, without editing it.
export PYTHONPATH="$viewer_dir${PYTHONPATH:+:$PYTHONPATH}"
if [[ ! -f "$viewer_dir/emu/webviewer_dist/index.html" ]];then
  command -v npm >/dev/null || { echo 'Node/npm required to build the viewer once' >&2;exit 1; }
  echo '[v5020 spectator] building independent 312-input browser assets...'
  "$python_bin" -m emu.cli viewer-build
fi
echo "[v5020 spectator] READ ONLY $root"
echo "[v5020 spectator] Open http://localhost:$port/ from Windows."
echo '[v5020 spectator] Watcher uses the newest complete validated checkpoint, not in-flight GPU memory.'
exec "$python_bin" -m emu.cli viewer --live --root "$root" --port "$port" --no-open
