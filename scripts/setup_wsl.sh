#!/usr/bin/env bash
# Reproducible source-checkout setup. No sudo, no Git changes, no training.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"
CUDA13=0
INSTALL_NODE=0
VIEWER_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --cuda13) CUDA13=1 ;;
    --install-node) INSTALL_NODE=1 ;;
    --viewer-only) VIEWER_ONLY=1 ;;
    --help|-h)
      printf 'Usage: bash scripts/setup_wsl.sh [--cuda13] [--install-node] [--viewer-only]\n'
      printf '  --cuda13         Install JAX CUDA 13 wheels for a compatible NVIDIA GPU\n'
      printf '  --install-node   Install Node 24 via official pinned nvm installer if missing\n'
      printf '  --viewer-only   Verify physical terrain + browser viewer, not the legacy schema-1 examiner suite\n'
      exit 0 ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done
command -v git >/dev/null || { echo 'Git required. On Ubuntu: sudo apt install git' >&2; exit 1; }
command -v curl >/dev/null || { echo 'curl required. On Ubuntu: sudo apt install curl' >&2; exit 1; }

if command -v python3.13 >/dev/null 2>&1; then
  PYTHON=python3.13
elif command -v python3 >/dev/null 2>&1 && python3 -c 'import sys;sys.exit(sys.version_info < (3,13))'; then
  PYTHON=python3
else
  echo 'EMU requires Python >= 3.13. Install Python 3.13+ with venv for your distribution before retrying.' >&2
  exit 1
fi
if [[ ! -f emu/viewer_src/package-lock.json ]]; then
  echo 'STOP: emu/viewer_src/package-lock.json is missing from this checkout.' >&2
  echo 'The lockfile from the tested npm build MUST be added and committed before a portable setup.' >&2
  echo 'On the already-working development PC: cd emu/viewer_src && npm install; git add emu/viewer_src/package-lock.json' >&2
  exit 1
fi

if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
  if [[ $INSTALL_NODE -ne 1 ]]; then
    echo 'Node/npm missing. Run setup with --install-node to install Node 24 via nvm.' >&2
    exit 1
  fi
fi
# --install-node also repairs an unsupported system Node, not just missing Node.
if [[ $INSTALL_NODE -eq 1 ]]; then
  [[ -s "$HOME/.nvm/nvm.sh" ]] || {
    # Pin nvm installation script for predictability; user opted into this.
    temp_script="$(mktemp)"
    trap 'rm -f "$temp_script"' EXIT
    curl -fsSL 'https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.8/install.sh' -o "$temp_script"
    bash "$temp_script"
    rm -f "$temp_script"
    trap - EXIT
  }
fi
# Use the repo's Node version when nvm exists, even if a system Node is present.
if [[ -s "$HOME/.nvm/nvm.sh" ]]; then
  export NVM_DIR="$HOME/.nvm"
  # shellcheck disable=SC1091
  # Prevent nvm from auto-using Node 24 from .nvmrc before it is installed.
  source "$NVM_DIR/nvm.sh" --no-use
  if [[ $INSTALL_NODE -eq 1 ]]; then
    nvm install "$(cat .nvmrc)"
  elif nvm ls "$(cat .nvmrc)" 2>/dev/null | grep -q 'N/A'; then
    echo 'Node 24 not installed in nvm. Run with --install-node.' >&2
    exit 1
  else
    nvm use "$(cat .nvmrc)" >/dev/null
  fi
fi
if ! node -e 'const [major,minor] = process.versions.node.split(".").map(Number); process.exit(major>=24 || (major===22 && minor>=12) || (major===20 && minor>=19) ? 0 : 1)' ; then
  echo "Node $(node --version) is too old for Vite 7. Use Node 24 (nvm install 24)." >&2
  exit 1
fi

if [[ ! -d .venv ]]; then "$PYTHON" -m venv .venv; fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -c 'import sys; assert sys.version_info >= (3,13), "Existing .venv uses the wrong Python; preserve it and recreate after review"'
python -m pip install --upgrade pip
if [[ $CUDA13 -eq 1 ]]; then python -m pip install 'jax[cuda13]'; fi
python -m pip install -e '.[dev]'
if [[ $VIEWER_ONLY -eq 1 ]]; then
  printf '\nViewer-only acceptance: verifying terrain physics and frozen sensor interfaces.\n'
  printf 'General Python suite is intentionally separate (five known 220-vs-232 examiner failures).\n'
  python -m pytest -q tests/test_terrain.py \
    tests/test_physics.py::test_versioned_sensor_actor_schema_keeps_pitch15_frozen
else
  python -m pytest -q
fi
if [[ $CUDA13 -eq 1 && $VIEWER_ONLY -eq 0 ]]; then
  emu doctor --domain-smoke
else
  emu doctor --no-require-gpu --no-mjx-smoke
fi
emu viewer-build
if [[ $VIEWER_ONLY -eq 1 ]]; then
  npm test --prefix emu/viewer_src
else
  emu ladder --stage plan
fi
if [[ $VIEWER_ONLY -eq 1 ]]; then
  printf '\nViewer acceptance setup complete. Launch the existing trained champion:\n'
  printf '  source .venv/bin/activate\n'
  printf '  emu viewer showcase/champion.emupolicy --no-open\n'
  printf '  open http://127.0.0.1:8765/ in your browser\n'
  printf 'No training, actor conversion or protected policy writes have occurred.\n'
else
  printf '\nSetup complete. In a separate terminal: source .venv/bin/activate && emu viewer emu/viewer_src/demo_zero.emupolicy --no-open\n'
  printf 'Viewer URL: http://127.0.0.1:8765/\n'
  printf 'Setup never starts training. On a supported GPU, run: emu doctor --v5015-smoke --domain-smoke\nThen: emu ladder --stage foundation --root "$HOME/robot_lab/emu_runs/new-study" --seed 733 --envs 2048 --device cuda\n'
fi
