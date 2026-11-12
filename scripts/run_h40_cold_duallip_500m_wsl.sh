#!/usr/bin/env bash
# C: one absolutely cold 500M H40 learner. No B/A/420M checkpoint is loaded.
# Physical eight-prism two-lip Academy, full command after each lip.
set -euo pipefail

if [[ ${1:-} != "" && ${1:-} != "--resume" ]] || (( $# > 1 )); then
  echo "Usage: bash $0 [--resume]  # omit --resume for a NEW cold run" >&2
  exit 2
fi
MODE="${1:-cold}"
RUN="${EMU_COLD_C_RUN:-$HOME/emu-data/runs/h40-cold-duallip-500m-s735}"
if [[ "$MODE" == cold && -e "$RUN" ]]; then
  echo "REFUSING to overwrite $RUN. Use --resume to continue THIS SAME run." >&2
  exit 1
fi
if [[ "$MODE" == "--resume" && ! -d "$RUN" ]]; then
  echo "No previous cold C run at $RUN; omit --resume to start new." >&2
  exit 1
fi
if [[ -n "${EMU_RESEARCH_PARENT:-}" ]]; then
  echo "C is a COLD run. Unset EMU_RESEARCH_PARENT; no parent is permitted." >&2
  exit 1
fi
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"
export EMU_PHYSICAL_PROFILE=h40
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export EMU_UNIFIED_CONTINUOUS=cold-growth-approved
export EMU_CONSOLE_MODE="${EMU_CONSOLE_MODE:-auto}"
mkdir -p "$(dirname "$RUN")"

FLAGS=(
  --run-dir "$RUN" --profile sim2real --device cuda
  --envs 2048 --steps 500000000 --seed 735
  --terrain-actor --terrain-step-max-m .08
  --unified-capabilities --terrain-carpet --carpet-prisms
  --academy-stream --academy-height-cap-m .08
  --academy-course-episode-frames 1500
  --terrain-lesson coherent-careful
  --teaching-schedule staged --growth-recipe main-mature
  --foundation-steps 20000000
  --athletic-rehearsal-share .15 --retention-share .20
  --outcome-graduation --progressive-school --refined-school
  --bidirectional-school
  --science-step-finish-on-transfer
  --two-lip-continuation-school
  --science-step-goal-weight 0
  --science-step-velocity-relief .80
  --fixed-eval-every-steps 20000000
)
if [[ "$MODE" == "--resume" ]]; then
  FLAGS+=(--resume auto)
fi
echo "EMU C source: $(git rev-parse HEAD)"
echo "EMU C source branch: $(git branch --show-current || true)"
echo "H40 profile: $EMU_PHYSICAL_PROFILE"
echo "Fresh 0 -> 500M, 2048 GPU worlds, seed 735; NO pretrained actor."
echo "No first-lip terminal: real 2 lips + 0.5m follow-through + 0.4s vx."
echo "Output: $RUN"
echo "=== Validate exact source/config/checkpoint identity before GPU ==="
python -m emu.cli train "${FLAGS[@]}" --check-resume-only
echo "=== Beginning immutable-source cold C training ==="
exec python -m emu.cli train "${FLAGS[@]}"
