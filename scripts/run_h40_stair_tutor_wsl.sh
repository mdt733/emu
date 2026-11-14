#!/usr/bin/env bash
# Small, controlled 50mm H40 tutorial A/B; old 16-arm study stays archived.
# Every fork starts from the same finished 60M D foundation policy/critic.
set -euo pipefail

ROOT="${EMU_TUTOR_ROOT:-$HOME/emu-data/runs/h40-stair-tutor-ab-s735}"
PARENT="${EMU_TUTOR_PARENT:-$HOME/emu-data/runs/h40-d-coach-lab-s735/foundation-60m/checkpoints/final.pkl}"
TARGET="${EMU_TUTOR_TARGET:-90000000}"  # 60M parent -> 90M; 30M per arm
MODE="${1:-help}"
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"
export EMU_PHYSICAL_PROFILE=h40
export EMU_UNIFIED_CONTINUOUS=cold-growth-approved
export XLA_PYTHON_CLIENT_PREALLOCATE=false
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing dirty source. Use a clean pinned Git worktree." >&2; exit 1
fi
COMMIT="$(git rev-parse HEAD)"
if [[ ! -f "$PARENT" ]]; then
  echo "Missing complete, immutable 60M foundation: $PARENT" >&2; exit 1
fi
PARENT_META="$(dirname "$(dirname "$PARENT")")/run_meta.json"
if [[ ! -f "$PARENT_META" ]]; then
  echo "Parent source provenance missing: $PARENT_META" >&2; exit 1
fi
PARENT_SOURCE="$(python - "$PARENT_META" <<'PY'
import json,sys
meta=json.load(open(sys.argv[1]))
print(meta.get('source_provenance',{}).get('git_commit',''))
PY
)"
[[ "$PARENT_SOURCE" =~ ^[0-9a-f]{40}$ ]] || {
  echo "Cannot verify original checkpoint source SHA" >&2; exit 1;
}
PARENT_HASH="$(sha256sum "$PARENT" | cut -d' ' -f1)"
if [[ -d "$ROOT" ]]; then
  for spec in "$ROOT/source.sha" "$ROOT/ancestor.sha256"; do
    if [[ -f "$spec" ]]; then
      expected="$(cat "$spec")"
      observed="$COMMIT"
      [[ "$spec" == *ancestor.sha256 ]] && observed="$PARENT_HASH"
      [[ "$expected" == "$observed" ]] || {
        echo "Source/ancestor changed: $spec; refuse mixing evidence" >&2; exit 1;
      }
    fi
  done
fi
COMMON=(
  --profile sim2real --device cuda --seed 735
  --terrain-actor --terrain-step-max-m .08
  --unified-capabilities --terrain-carpet --carpet-prisms
  --academy-stream --academy-height-cap-m .08
  --academy-course-episode-frames 1500
  --terrain-lesson coherent-careful
  --teaching-schedule staged --growth-recipe main-mature
  --foundation-steps 20000000
  --athletic-rehearsal-share .15 --retention-share .20
  --outcome-graduation --progressive-school --refined-school
  --bidirectional-school --science-step-finish-on-transfer
  --two-lip-continuation-school
  --science-step-goal-weight 0 --science-step-velocity-relief .80
  --stair-tournament-arm control
  --fixed-eval-every-steps 10000000
)

register_study(){
  mkdir -p "$ROOT"
  [[ -f "$ROOT/source.sha" ]] || printf '%s\n' "$COMMIT" > "$ROOT/source.sha"
  [[ -f "$ROOT/ancestor.sha256" ]] || printf '%s\n' "$PARENT_HASH" > "$ROOT/ancestor.sha256"
  [[ -f "$ROOT/ancestor-source.sha" ]] || printf '%s\n' "$PARENT_SOURCE" > "$ROOT/ancestor-source.sha"
}
make_args(){
  local arm="$1"; shift
  local prefix="$ROOT/$arm"
  FLAGS=("${COMMON[@]}" --run-dir "$prefix" --steps "$TARGET"
         --envs 2048 --stair-tutor-mode "$arm")
  if [[ -f "$prefix/training_config.json" ]]; then
    FLAGS+=(--resume auto)
  else
    FLAGS+=(--resume "$PARENT"
      --research-parent-source "$PARENT_SOURCE"
      --research-parent-sha256 "$PARENT_HASH")
  fi
}
check(){
  echo "Source $COMMIT; immutable parent $PARENT_SOURCE / $PARENT_HASH"
  echo "Runs: $ROOT, target absolute $TARGET, 50mm UP/20mm DOWN"
  for arm in legacy escrow; do
    make_args "$arm"
    python -m emu.cli train "${FLAGS[@]}" --check-resume-only
  done
}
case "$MODE" in
  help)
    echo "Usage: bash $0 preflight | smoke | arm legacy | arm escrow | both | report"
    echo "First do preflight and smoke. No training when called without an explicit mode."
    ;;
  preflight)
    check
    ;;
  smoke)
    register_study
    # A small genuine CUDA PPO trainer run, not an independent treatment.
    SMOKE="$ROOT/gpu-smoke"
    FLAGS=("${COMMON[@]}" --run-dir "$SMOKE" --envs 128
      --steps 60131072 --stair-tutor-mode escrow)
    if [[ -f "$SMOKE/training_config.json" ]]; then
      FLAGS+=(--resume auto)
    else
      FLAGS+=(--resume "$PARENT"
        --research-parent-source "$PARENT_SOURCE"
        --research-parent-sha256 "$PARENT_HASH")
    fi
    python -m emu.cli train "${FLAGS[@]}" --check-resume-only
    python -m emu.cli train "${FLAGS[@]}"
    ;;
  arm)
    [[ $# -eq 2 && ( "$2" == legacy || "$2" == escrow ) ]] || {
      echo "Usage: $0 arm legacy|escrow" >&2; exit 2;
    }
    register_study
    make_args "$2"
    python -m emu.cli train "${FLAGS[@]}" --check-resume-only
    python -m emu.cli train "${FLAGS[@]}"
    ;;
  both)
    register_study
    for arm in legacy escrow; do
      make_args "$arm"
      python -m emu.cli train "${FLAGS[@]}" --check-resume-only
      python -m emu.cli train "${FLAGS[@]}"
    done
    ;;
  report)
    python -m scripts.summarize_h40_stair_tutor --root "$ROOT"
    ;;
  *)
    echo "Unknown mode: $MODE" >&2; exit 2
    ;;
esac
