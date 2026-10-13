#!/usr/bin/env bash
# Inspect (default) or reversibly quarantine ONLY recognized historic unpack/backup paths.
# Does not delete anything, change Git, or touch the repo or emu_runs.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
WORKSPACE="$(dirname "$ROOT")"
MODE="${1:-}"
case "$MODE" in ''|--dry-run) ;; --quarantine) ;; *) echo 'Usage: bash scripts/workspace_tidy.sh [--dry-run|--quarantine]' >&2; exit 2;; esac
if [[ "$(basename "$ROOT")" != emu ]]; then
  echo "STOP: expected repo named emu (got: $ROOT)." >&2; exit 1
fi
if [[ ! -d "$ROOT/.git" ]] && ! git -C "$ROOT" rev-parse --show-toplevel >/dev/null 2>&1; then
  echo 'STOP: not an EMU Git checkout.' >&2; exit 1
fi
candidates=()
unknown=()
shopt -s nullglob
for item in "$WORKSPACE"/*; do
  [[ -L "$item" ]] && { unknown+=("$item"); continue; }
  name="$(basename "$item")"
  case "$name" in
    emu|emu_runs) ;; # Protected, never candidates.
    emu_upgrade_backups|incoming-v5014*|v5013_results_backup.*|EMU_v5014_*.zip)
      candidates+=("$item") ;;
    *) unknown+=("$item") ;;
  esac
done
shopt -u nullglob
printf 'EMU checkout (never touched): %s\n' "$ROOT"
printf 'Training runs (never touched): %s\n' "$WORKSPACE/emu_runs"
printf '\nRecognized migration leftovers (candidate for reversible archive):\n'
if (( ${#candidates[@]} == 0 )); then
  echo '  none'
else
  for item in "${candidates[@]}"; do du -sh -- "$item" 2>/dev/null || printf '  %s\n' "$item"; done
fi
printf '\nOther workspace entries (NOT touched):\n'
if (( ${#unknown[@]} == 0 )); then echo '  none'; else printf '  %s\n' "${unknown[@]}"; fi
if [[ "$MODE" != --quarantine ]]; then
  printf '\nPreview only; nothing moved. To archive recognized leftovers:\n  bash scripts/workspace_tidy.sh --quarantine\n'
  exit 0
fi
if (( ${#candidates[@]} == 0 )); then echo 'Nothing to archive.'; exit 0; fi
printf '\nType ARCHIVE to move only the recognized candidates into a recoverable folder: '
read -r confirmation
if [[ "$confirmation" != ARCHIVE ]]; then echo 'Cancelled; nothing moved.'; exit 0; fi
STAMP="$(date +%Y%m%d_%H%M%S)"
ARCHIVE="$HOME/emu_workspace_archives/$STAMP"
mkdir -p "$ARCHIVE"
for item in "${candidates[@]}"; do
  dest="$ARCHIVE/$(basename "$item")"
  if [[ -e "$dest" ]]; then echo "STOP: destination exists ($dest)" >&2; exit 1; fi
  mv -- "$item" "$dest"
  printf 'Archived: %s -> %s\n' "$item" "$dest"
done
printf '\nRecoverable originals are in: %s\nDo not delete these archives until you have reviewed their contents.\n' "$ARCHIVE"
