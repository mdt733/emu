#!/usr/bin/env bash
# Small, controlled 50mm H40 tutorial A/B; old 16-arm study stays archived.
# One walking bridge trains 60M -> 80M once. Both arms fork its exact full checkpoint.
set -euo pipefail

ROOT="${EMU_TUTOR_ROOT:-$HOME/emu-data/runs/h40-stair-tutor-ab-s735}"
PARENT="${EMU_TUTOR_PARENT:-$HOME/emu-data/runs/h40-d-coach-lab-s735/foundation-60m/checkpoints/final.pkl}"
BRIDGE_TARGET=80000000  # once-only walking graduation
TARGET=100000000       # 80M->100M matched 20M tutor arms
BRIDGE="$ROOT/bridge-80m"
BRIDGE_PARENT="$BRIDGE/checkpoints/final.pkl"
MODE="${1:-help}"
if [[ "$MODE" == help ]]; then
  echo "Usage: bash $0 preflight | smoke | bridge | arm legacy | arm escrow | both | exam | report"
  echo "Explicit modes only; help never reads a checkpoint or trains."
  exit 0
fi
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
  --fixed-eval-every-steps 10000000
)

register_study(){
  mkdir -p "$ROOT"
  [[ -f "$ROOT/source.sha" ]] || printf '%s\n' "$COMMIT" > "$ROOT/source.sha"
  [[ -f "$ROOT/ancestor.sha256" ]] || printf '%s\n' "$PARENT_HASH" > "$ROOT/ancestor.sha256"
  [[ -f "$ROOT/ancestor-source.sha" ]] || printf '%s\n' "$PARENT_SOURCE" > "$ROOT/ancestor-source.sha"
}
verify_bridge(){
  [[ -f "$BRIDGE_PARENT" ]] || {
    echo "Missing shared 80M bridge checkpoint. Run bridge first." >&2
    exit 1
  }
  local observed
  observed="$(sha256sum "$BRIDGE_PARENT" | cut -d' ' -f1)"
  if [[ -f "$ROOT/bridge.sha256" ]]; then
    [[ "$(cat "$ROOT/bridge.sha256")" == "$observed" ]] || {
      echo "Shared 80M ancestor changed. REFUSING both arms." >&2; exit 1
    }
  else
    printf '%s\n' "$observed" > "$ROOT/bridge.sha256"
  fi
  echo "Shared 80M bridge SHA256: $observed"
}
make_bridge_args(){
  FLAGS=("${COMMON[@]}" --run-dir "$BRIDGE" --steps "$BRIDGE_TARGET"
         --envs 2048 --stair-tutor-mode legacy)
  if [[ -f "$BRIDGE/training_config.json" ]]; then
    FLAGS+=(--resume auto)
  else
    FLAGS+=(--resume "$PARENT"
      --research-parent-source "$PARENT_SOURCE"
      --research-parent-sha256 "$PARENT_HASH")
  fi
}
make_args(){
  local arm="$1"
  local prefix="$ROOT/$arm"
  FLAGS=("${COMMON[@]}" --run-dir "$prefix" --steps "$TARGET"
         --envs 2048 --stair-tutor-mode "$arm")
  if [[ -f "$prefix/training_config.json" ]]; then
    FLAGS+=(--resume auto)
  else
    FLAGS+=(--resume "$BRIDGE_PARENT")
  fi
}
check(){
  echo "Tutor source $COMMIT; 60M parent source $PARENT_SOURCE SHA256=$PARENT_HASH"
  make_bridge_args
  python -m emu.cli train "${FLAGS[@]}" --check-resume-only
  if [[ -f "$BRIDGE_PARENT" ]]; then
    verify_bridge
    for arm in legacy escrow; do
      make_args "$arm"
      python -m emu.cli train "${FLAGS[@]}" --check-resume-only
    done
  else
    echo "Arms start only after shared 80M walking bridge."
  fi
}
run_bridge(){
  register_study
  make_bridge_args
  python -m emu.cli train "${FLAGS[@]}" --check-resume-only
  python -m emu.cli train "${FLAGS[@]}"
  verify_bridge
}
run_arm(){
  local arm="$1"
  register_study
  verify_bridge
  make_args "$arm"
  python -m emu.cli train "${FLAGS[@]}" --check-resume-only
  python -m emu.cli train "${FLAGS[@]}"
}
case "$MODE" in
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
  bridge)
    run_bridge
    ;;
  arm)
    [[ $# -eq 2 && ( "$2" == legacy || "$2" == escrow ) ]] || {
      echo "Usage: $0 arm legacy|escrow" >&2; exit 2;
    }
    run_arm "$2"
    ;;
  both)
    run_bridge
    run_arm legacy
    # A 20M tutor fork is pointless if the baseline never encountered
    # the 50mm uphill lessons. Explicitly stop before funding escrow.
    python -m scripts.summarize_h40_stair_tutor \
      --root "$ROOT" --require-up-exposure
    run_arm escrow
    ;;
  exam)
    # Fully frozen policy exams: 2 actors x 3 physical programmes x 2 seeds.
    # No PPO update; no root crossing alone counts as a climb.
    register_study
    for arm in legacy escrow; do
      FINAL="$ROOT/$arm/checkpoints/final.pkl"
      [[ -f "$FINAL" ]] || {
        echo "Missing completed arm checkpoint: $FINAL" >&2; exit 1;
      }
      mkdir -p "$ROOT/policies"
      POLICY="$ROOT/policies/$arm.emupolicy"
      python -m emu.cli export-policy "$FINAL" "$POLICY"
      for program in up mixed down; do
        for seed in 1701 2701; do
          DEST="$ROOT/exams/$arm/${program}_s$seed"
          if [[ -f "$DEST/exam.json" ]]; then
            echo "Existing physical evidence: $DEST"
          else
            [[ ! -e "$DEST" ]] || {
              echo "Incomplete exam, inspect and move manually: $DEST" >&2; exit 1;
            }
            python -m scripts.h40_d_two_lip_exam --policy "$POLICY" \
              --output "$DEST" --program "$program" \
              --worlds 128 --horizon 1050 --seed "$seed" --device cuda
          fi
        done
      done
      # Paired actual 50mm UP/DOWN versus identical zero-height seam worlds.
      # Same actor, command, seed and per-world reset keys. Read-only.
      STAIR="$ROOT/exams/$arm/paired_stairs_s1701"
      FLAT="$ROOT/exams/$arm/paired_flat_s1701"
      PAIR="$ROOT/exams/$arm/paired_comparison_s1701"
      if [[ ! -f "$STAIR/matrix_summary.json" ]]; then
        [[ ! -e "$STAIR" ]] || { echo "Incomplete $STAIR" >&2; exit 1; }
        python -m scripts.capability_matrix_exam --policy "$POLICY" \
          --output "$STAIR" --mode stairs --worlds 128 \
          --levels 1 --trials-per-level 64 --horizon 350 \
          --seed 1701 --device cuda
      fi
      if [[ ! -f "$FLAT/matrix_summary.json" ]]; then
        [[ ! -e "$FLAT" ]] || { echo "Incomplete $FLAT" >&2; exit 1; }
        python -m scripts.capability_matrix_exam --policy "$POLICY" \
          --output "$FLAT" --mode flat --worlds 128 \
          --trials-per-level 128 --horizon 350 \
          --seed 1701 --device cuda
      fi
      if [[ ! -f "$PAIR/matrix_compare.json" ]]; then
        [[ ! -e "$PAIR" ]] || { echo "Incomplete $PAIR" >&2; exit 1; }
        python -m scripts.capability_matrix_exam --compare-stairs "$STAIR" \
          --compare-flat "$FLAT" --output "$PAIR"
      fi
    done
    python -m scripts.summarize_h40_stair_tutor --root "$ROOT"
    ;;
  report)
    python -m scripts.summarize_h40_stair_tutor --root "$ROOT"
    ;;
  *)
    echo "Unknown mode: $MODE" >&2; exit 2
    ;;
esac
