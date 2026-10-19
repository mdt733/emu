#!/usr/bin/env bash
# EMU CPU-only viewer for Ubuntu/WSL2. Installs tools INSIDE Linux only.
# No CUDA/JAX/MJX, training checkpoint, or Python MuJoCo installation.
set -Eeuo pipefail
BRANCH="${EMU_VIEWER_BRANCH:-main}"
REPO=https://github.com/mdt733/emu.git
DEST="$HOME/emu-viewer-cpu"
LAUNCH=1
say() { printf '\n[EMU WSL viewer] %s\n' "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }
usage() {
  cat <<'EOF'
Usage: bash install_viewer_wsl.sh [--no-launch] [--destination PATH] [--branch NAME] [--self-test]
Run from Ubuntu inside WSL2. All downloads and experiments stay in Linux.
Ctrl+C stops the viewer at http://127.0.0.1:8767/.
EOF
}
while (($#)); do
  case "$1" in
    --no-launch) LAUNCH=0 ;;
    --destination) (($#>1)) || die '--destination needs an argument'; DEST="$2"; shift ;;
    --branch) (($#>1)) || die '--branch needs an argument'; BRANCH="$2"; shift ;;
    --self-test) say 'Viewer bootstrap self-test passed; nothing installed.'; exit 0 ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown argument: $1" ;;
  esac
  shift
done
# Accept only branch names composed of ordinary Git ref path components.
# Never interpolate arbitrary shell or option-shaped values into git commands.
[[ "$BRANCH" =~ ^[A-Za-z0-9][A-Za-z0-9._/-]*$ && "$BRANCH" != *..* ]] ||
  die 'Invalid branch name.'
[[ "$(uname -s)" == Linux ]] || die 'Run inside WSL Ubuntu, not PowerShell.'
[[ -f /etc/os-release ]] || die 'Cannot identify Linux distribution.'
# shellcheck disable=SC1091
source /etc/os-release
[[ "$ID" == ubuntu ]] || die 'This bootstrap supports Ubuntu (including WSL2 Ubuntu).'

# The default keeps the repository under the WSL2 ext4 filesystem.
case "$DEST" in
  /mnt/*|/media/*) die 'Choose a Linux directory, not a Windows-mounted drive.' ;;
esac

apt_needed=0
for program in git curl python3; do
  command -v "$program" >/dev/null 2>&1 || apt_needed=1
done
[[ -f /etc/ssl/certs/ca-certificates.crt ]] || apt_needed=1
if ((apt_needed)); then
  say 'Installing Ubuntu Git, curl, Python and TLS certificates via apt.'
  if [[ "$(id -u)" -eq 0 ]]; then
    apt-get update
    env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      git curl python3 ca-certificates
  else
    command -v sudo >/dev/null || die 'sudo is required to install Ubuntu tools.'
    sudo apt-get update
    sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      git curl python3 ca-certificates
  fi
fi
python3 -c 'import sys; assert sys.version_info >= (3,10), "Python 3.10+ required"' ||
  die 'Python too old for showcase preparation.'
say "Using $(git --version) and $(python3 --version)"

# Node 24 via user-local nvm, only if not already present in Ubuntu.
has_node24=0
if command -v node >/dev/null && command -v npm >/dev/null; then
  node -e 'process.exit(+process.versions.node.split(".")[0]===24?0:1)' &&
    has_node24=1 || true
fi
if ((!has_node24)); then
  export NVM_DIR="$HOME/.nvm"
  if [[ ! -s "$NVM_DIR/nvm.sh" ]]; then
    say 'Installing user-local nvm 0.40.8 inside Ubuntu.'
    installer="$(mktemp)"
    trap 'rm -f -- "$installer"' EXIT
    curl --proto '=https' --tlsv1.2 -fsSL \
      https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.8/install.sh -o "$installer"
    PROFILE=/dev/null bash "$installer"
    rm -f -- "$installer"
    trap - EXIT
  fi
  [[ -s "$NVM_DIR/nvm.sh" ]] || die 'nvm installation failed.'
  # shellcheck disable=SC1090
  source "$NVM_DIR/nvm.sh" --no-use
  nvm install 24
  nvm use 24 >/dev/null
fi
node -e 'process.exit(+process.versions.node.split(".")[0]===24?0:1)' ||
  die 'Node.js 24 unavailable.'
say "Using Node $(node --version) and npm $(npm --version)"

# Refuse unexpected or dirty checkouts instead of resetting experiments.
if [[ -d "$DEST/.git" ]]; then
  say "Updating existing viewer checkout: $DEST"
  [[ "$(git -C "$DEST" branch --show-current)" == "$BRANCH" ]] ||
    die 'Existing checkout uses another branch.'
  if [[ -n "$(git -C "$DEST" status --porcelain)" ]]; then
    say 'Existing viewer checkout has uncommitted changes:'
    git -C "$DEST" status --short
    printf '\n'
    die "Preserving local changes. To install an independent clean viewer, run:
  bash /tmp/emu-viewer-install.sh --destination \"$HOME/emu-viewer-main\"
No files were changed or deleted."
  fi
  git -C "$DEST" pull --ff-only
elif [[ -e "$DEST" ]]; then
  [[ -d "$DEST" && -z "$(ls -A "$DEST")" ]] ||
    die "Destination exists and is not empty: $DEST"
  git clone --single-branch --branch "$BRANCH" "$REPO" "$DEST"
else
  git clone --single-branch --branch "$BRANCH" "$REPO" "$DEST"
fi
[[ -f "$DEST/showcase/champion.emupolicy" ]] ||
  die 'The trained champion policy is missing.'
[[ -f "$DEST/emu/viewer_src/package-lock.json" ]] ||
  die 'Locked npm dependencies are missing.'

say 'Preparing the committed 260M actor (Python standard library only).'
python3 "$DEST/scripts/prepare_showcase.py" --require-champion
say 'Installing browser-only Node dependencies.'
npm ci --prefix "$DEST/emu/viewer_src" --no-audit --no-fund
say 'Running viewer tests.'
npm test --prefix "$DEST/emu/viewer_src"
say "Everything stays in Ubuntu at $DEST"
say 'Open http://127.0.0.1:8767/ in Windows Chrome/Edge.'
if ((!LAUNCH)); then exit 0; fi
say 'Starting viewer; Ctrl+C stops it.'
cd "$DEST"
npm run dev --prefix emu/viewer_src -- --host 127.0.0.1 --port 8767 --strictPort
