#!/usr/bin/env bash
# Set up this EMU checkout in WSL. Installs dependencies but never trains,
# changes Git refs, deletes a virtualenv or alters protected policies.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CUDA13=0
INSTALL_NODE=0
VIEWER_ONLY=0
CHECK=0
usage() {
  cat <<'EOF'
Usage: bash scripts/setup_wsl.sh [--viewer-only] [--install-node] [--cuda13] [--check]
  (default)       Install EMU Python + browser viewer and run basic doctor/plan
  --viewer-only   Install EMU and build the viewer; skip training checks/plans
  --install-node  Install/use repository Node 24 via user-local nvm
  --cuda13        Install CUDA 13 JAX wheels for NVIDIA/WSL GPU experiments
  --check         Also run full Python and browser tests (never implicit)
No training is started, no Git state is changed.
EOF
}
for arg in "$@"; do
  case "$arg" in
    --viewer-only) VIEWER_ONLY=1 ;;
    --cuda13) CUDA13=1 ;;
    --install-node) INSTALL_NODE=1 ;;
    --check) CHECK=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; usage >&2; exit 2 ;;
  esac
done
if ((VIEWER_ONLY && CUDA13)); then
  echo 'Choose viewer-only or CUDA training setup, not both.' >&2
  exit 2
fi

if command -v python3.13 >/dev/null 2>&1; then
  PYTHON=python3.13
elif command -v python3 >/dev/null 2>&1 && python3 -c 'import sys; sys.exit(sys.version_info < (3, 13))'; then
  PYTHON=python3
else
  echo 'EMU needs Python 3.13+. Install it with venv support first.' >&2
  exit 1
fi

# The single source-checkout viewer uses Node, even in viewer-only mode.
[[ -f emu/viewer_src/package-lock.json ]] || {
  echo 'Missing committed viewer package-lock.json; refusing unlocked installation.' >&2
  exit 1
}
if ((INSTALL_NODE)); then
  command -v curl >/dev/null || { echo 'curl is required to install Node.' >&2; exit 1; }
  export NVM_DIR="$HOME/.nvm"
  if [[ ! -s "$NVM_DIR/nvm.sh" ]]; then
    temp_script="$(mktemp)"
    trap 'rm -f "$temp_script"' EXIT
    curl --proto '=https' --tlsv1.2 -fsSL \
      'https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.8/install.sh' -o "$temp_script"
    PROFILE=/dev/null bash "$temp_script"
    rm -f "$temp_script"
    trap - EXIT
  fi
fi
if [[ -s "$HOME/.nvm/nvm.sh" ]]; then
  export NVM_DIR="$HOME/.nvm"
  # shellcheck disable=SC1091
  source "$NVM_DIR/nvm.sh" --no-use
  if ((INSTALL_NODE)); then
    nvm install "$(cat .nvmrc)"
  else
    nvm use "$(cat .nvmrc)" >/dev/null 2>&1 || true
  fi
fi
if ! command -v node >/dev/null || ! command -v npm >/dev/null || \
   ! node -e 'const v=process.versions.node.split(".").map(Number); process.exit(v[0]>=24 || (v[0]===22 && v[1]>=12) || (v[0]===20 && v[1]>=19) ? 0 : 1)'; then
  echo 'Compatible Node/npm missing. Rerun with --install-node.' >&2
  exit 1
fi

if [[ ! -d .venv ]]; then "$PYTHON" -m venv .venv; fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -c 'import sys; assert sys.version_info >= (3,13), "Existing .venv uses older Python. Preserve it and recreate after review."'
if ((CUDA13)); then
  python -m pip install 'jax[cuda13]'
fi
if ((CHECK)); then
  python -m pip install -e '.[dev]'
else
  python -m pip install -e .
fi

# Keep installation distinct from acceptance. A viewer checkout should not
# silently execute the full CPU/JAX test suite.
if ((!VIEWER_ONLY)); then
  emu doctor --no-require-gpu --no-mjx-smoke
  emu train --help >/dev/null
fi
emu viewer-build
if ((CHECK)); then
  python -m pytest -q tests
  npm test --prefix emu/viewer_src
fi

printf '\nEMU setup complete (%s). No training was started.\n' "$([[ "$VIEWER_ONLY" == 1 ]] && echo viewer-only || echo full)"
printf 'Viewer: source .venv/bin/activate && emu viewer showcase/champion.emupolicy --no-open\n'
printf 'Open http://127.0.0.1:8765/ in your browser.\n'
printf 'Focused physical terrain proof: python -m emu.terrain_spike --worlds 2\n'
if ((CUDA13)); then
  printf 'GPU acceptance is separate: emu doctor --v5015-smoke --domain-smoke\n'
fi
