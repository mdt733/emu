#!/usr/bin/env bash
# Experiment D: FULL 2^4 factorial investigation of teaching ideas.
# NOT a knockout tournament. No ranking, no winner, no finalist campaigns.
# The common 60M gait learner is cold trained once; every independent
# branch inherits the SAME saved policy/critic/optimizer/normalizer/governor.
set -euo pipefail

ROOT="${EMU_D_LAB_ROOT:-$HOME/emu-data/runs/h40-d-coach-lab-s735}"
BASE="$ROOT/foundation-60m"
PARENT="$BASE/checkpoints/final.pkl"
TARGET=160000000  # 60M parent + identical 100M additional steps per arm

ARMS=(
 "00-control:control"
 "01-progressive:progressive"
 "02-free-phase:free-phase"
 "03-foothold:foothold"
 "04-commands:commands"
 "05-progressive-free-phase:progressive+free-phase"
 "06-progressive-foothold:progressive+foothold"
 "07-progressive-commands:progressive+commands"
 "08-free-phase-foothold:free-phase+foothold"
 "09-free-phase-commands:free-phase+commands"
 "10-foothold-commands:foothold+commands"
 "11-progressive-free-phase-foothold:progressive+free-phase+foothold"
 "12-progressive-free-phase-commands:progressive+free-phase+commands"
 "13-progressive-foothold-commands:progressive+foothold+commands"
 "14-free-phase-foothold-commands:free-phase+foothold+commands"
 "15-all-four:progressive+free-phase+foothold+commands"
)

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD"
export EMU_PHYSICAL_PROFILE=h40
export EMU_UNIFIED_CONTINUOUS=cold-growth-approved
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export EMU_CONSOLE_MODE="${EMU_CONSOLE_MODE:-auto}"

if [[ -n "${EMU_RESEARCH_PARENT:-}" ]]; then
  echo "Refusing implicit alternate parent: unset EMU_RESEARCH_PARENT" >&2
  exit 1
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing dirty source. Use a detached, clean, source-pinned Git worktree." >&2
  exit 1
fi
COMMIT="$(git rev-parse HEAD)"
mkdir -p "$ROOT"
echo "Experiment D coach lab; Git source: $COMMIT"
echo "Runs: $ROOT"
echo "Physical: H40 / 8 real MJX prisms / 360-input actor / 2048 worlds"
echo "Purpose: 16 independent CAUSAL teaching experiments; NO finalists."

# Every switch not listed below inherits exactly the SAME recipe. Experiments
# use only the treatment selector, seeded JAX salts and optional predeclared
# foothold shaping, which one_run_config itself derives from the arm.
COMMON=(
  --profile sim2real --device cuda --envs 2048 --seed 735
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
  --fixed-eval-every-steps 20000000
)

# Fail BEFORE spinning up JAX when source/physical/ancestor differs.
run_checked(){
  python -m emu.cli train "$@" --check-resume-only
  python -m emu.cli train "$@"
}
run_foundation(){
  if [[ -f "$PARENT" ]]; then
    echo "Existing foundation final.pkl; validating (never re-training)"
    run_checked "${COMMON[@]}" --run-dir "$BASE" --steps 60000000 \
      --stair-tournament-arm off --resume auto
  elif [[ -f "$BASE/training_config.json" ]]; then
    echo "Resuming interrupted 60M foundation from its OWN snapshots."
    run_checked "${COMMON[@]}" --run-dir "$BASE" --steps 60000000 \
      --stair-tournament-arm off --resume auto
  else
    echo "Training common COLD foundation: 0 -> 60M (no ancestor)."
    run_checked "${COMMON[@]}" --run-dir "$BASE" --steps 60000000 \
      --stair-tournament-arm off
  fi
  [[ -f "$PARENT" ]] || {
    echo "Foundation has not produced valid final.pkl; REFUSING arm forks." >&2
    exit 1
  }
  lock_parent
}
lock_parent(){
  [[ -f "$PARENT" ]] || {
    echo "Missing 60M final.pkl: run foundation before any branch." >&2
    exit 1
  }
  local observed expected
  observed="$(sha256sum "$PARENT" | cut -d' ' -f1)"
  if [[ -f "$ROOT/ancestor.sha256" ]]; then
    expected="$(cat "$ROOT/ancestor.sha256")"
    if [[ "$expected" != "$observed" ]]; then
      echo "REFUSING: shared ancestor bytes changed! Expected $expected; got $observed." >&2
      exit 1
    fi
  else
    printf '%s\n' "$observed" > "$ROOT/ancestor.sha256"
  fi
  echo "Common immutable 60M ancestor SHA256: $observed"
}
run_arm(){
  local number="$1" arm="$2" path="$ROOT/arm-$number"
  [[ -f "$PARENT" ]] || {
    echo "Missing common final checkpoint: run 'foundation' first" >&2
    exit 1
  }
  lock_parent
  local flags=("${COMMON[@]}" --run-dir "$path" --steps "$TARGET"
               --stair-tournament-arm "$arm")
  if [[ -f "$path/training_config.json" ]]; then
    flags+=(--resume auto)
    echo "Resuming / checking completed arm: $number ($arm)"
  else
    flags+=(--resume "$PARENT")
    echo "Forking the SAME 60M learner: $number ($arm)"
  fi
  run_checked "${flags[@]}"
}
print_arms(){
  for entry in "${ARMS[@]}"; do
    printf '  %-38s  %s\n' "${entry%%:*}" "${entry#*:}"
  done
}
mode="${1:-all}"
case "$mode" in
  list)
    print_arms
    ;;
  report)
    python scripts/summarize_h40_d_coach_lab.py --root "$ROOT"
    ;;
  foundation)
    [[ $# -eq 1 ]] || { echo "Usage: $0 foundation" >&2; exit 2; }
    run_foundation
    ;;
  arm)
    [[ $# -eq 2 ]] || { echo "Usage: $0 arm 00-control" >&2; exit 2; }
    found=false
    for entry in "${ARMS[@]}"; do
      number="${entry%%:*}"; arm="${entry#*:}"
      if [[ "$number" == "$2" ]]; then
        run_arm "$number" "$arm"
        found=true
        break
      fi
    done
    [[ "$found" == true ]] || { echo "Unknown arm $2; run 'list'." >&2; exit 2; }
    ;;
  all)
    [[ $# -le 1 ]] || { echo "Usage: $0 all" >&2; exit 2; }
    run_foundation
    for entry in "${ARMS[@]}"; do
      run_arm "${entry%%:*}" "${entry#*:}"
    done
    python scripts/summarize_h40_d_coach_lab.py --root "$ROOT"
    echo "ALL 16 ARMS COMPLETE. Nothing eliminated; study the teaching effects."
    ;;
  *)
    echo "Usage: $0 [all|foundation|list|report|arm <00-control...15-all-four>]" >&2
    exit 2
    ;;
esac
