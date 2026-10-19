#!/usr/bin/env bash
# Bootstrap a FRESH Ubuntu WSL2 checkout of feature/v5018-terrain-awareness.
# This script intentionally installs Ubuntu packages with sudo and user-local
# uv/Python/Node; it does not replace /usr/bin/python3 or install Linux GPU drivers.
# Run from a clone: bash scripts/bootstrap_wsl_v5018.sh [--cpu] [--skip-tests]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"
if [[ "$(id -u)" -eq 0 ]]; then
  echo 'Run as your normal WSL user, NOT sudo/root.' >&2
  exit 2
fi
if [[ "$(git branch --show-current)" != "feature/v5018-terrain-awareness" ]]; then
  echo 'Expected checkout of feature/v5018-terrain-awareness; refusing to install for another branch.' >&2
  exit 2
fi

CUDA13=1
SKIP_TESTS=0
for arg in "$@"; do
  case "$arg" in
    --cpu) CUDA13=0 ;;
    --skip-tests) SKIP_TESTS=1 ;;
    --help|-h)
      echo 'Usage: bash scripts/bootstrap_wsl_v5018.sh [--cpu] [--skip-tests]'
      echo 'Default is GPU/CUDA 13. --cpu avoids GPU setup. --skip-tests is NOT a validated install.'
      exit 0 ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

echo '=== Ubuntu system prerequisites ==='
sudo apt-get update
sudo apt-get install -y git curl ca-certificates build-essential \
  libgl1 libglib2.0-0 python3 python3-venv python3-pip

if [[ "$CUDA13" -eq 1 ]]; then
  echo '=== WSL NVIDIA device ==='
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo 'nvidia-smi unavailable. Check Windows NVIDIA driver and WSL GPU exposure.' >&2
    echo 'Do NOT install an NVIDIA Linux display driver inside WSL to fix this.' >&2
    exit 1
  fi
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
fi

echo '=== User-local Python 3.13 via uv ==='
if ! command -v uv >/dev/null 2>&1 && ! [[ -x "$HOME/.local/bin/uv" ]]; then
  # uv's official installer is user-local. Review its source/pinning policy
  # before use in managed or security-sensitive environments.
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || { echo 'uv installation failed' >&2; exit 1; }
uv python install 3.13

if [[ -e .venv && ! -x .venv/bin/python ]]; then
  echo 'Existing .venv is incomplete or inaccessible; inspect it instead of deleting it.' >&2
  exit 1
fi
if [[ ! -x .venv/bin/python ]]; then
  uv venv --python 3.13 .venv
fi
if ! .venv/bin/python -c 'import sys; sys.exit(sys.version_info < (3,13))'; then
  echo 'Existing .venv Python is below 3.13; preserve/recreate it explicitly.' >&2
  exit 1
fi
if ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
  uv pip install --python .venv/bin/python pip
fi
source .venv/bin/activate
python --version
python -m pip --version

echo '=== EMU Python/CUDA, nvm/Node/npm and locked browser installation ==='
setup=(--install-node)
if [[ "$CUDA13" -eq 1 ]]; then setup+=(--cuda13); fi
if [[ "$SKIP_TESTS" -eq 1 ]]; then setup+=(--skip-tests); fi
bash scripts/setup_wsl.sh "${setup[@]}"

echo '=== Installation identities ==='
git rev-parse HEAD
python -c 'import jax,mujoco; from mujoco import mjx; print("JAX",jax.__version__,"MuJoCo",mujoco.__version__,"devices",jax.devices())'
echo 'Full GPU terrain acceptance remains separate: bash scripts/run_v5018_p0_campaign.sh'
echo 'Fresh WSL setup complete; NO long training started.'
