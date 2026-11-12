#!/usr/bin/env bash
# Experiment B: same 180M actor/PPO optimizer, same feet-only 8-prism MJX
# physics, paired uphill+downhill school. Never edits prior 300M/420M runs.
set -euo pipefail

OLD_PIN=853d19a63502413bb9c05049b9f9ac5ac1afad4c
PARENT="${EMU_RESEARCH_PARENT:-$HOME/emu-data/research-parents/h40-school-180m-s735/checkpoints/final.pkl}"
RUN="${EMU_PAIRED_RUN:-$HOME/emu-data/runs/h40-paired-school-420m-s735}"

if [[ ! -f "$PARENT" ]]; then
    echo "Missing verified 180M parent: $PARENT" >&2
    echo "Archive it as described in docs/INSTALL.md first; do NOT use the 420M model as parent." >&2
    exit 1
fi
if [[ ! -f "$(dirname "$(dirname "$PARENT")")/training_config.json" ]]; then
    echo "Missing corresponding historical 180M training_config.json" >&2
    exit 1
fi
if [[ -e "$RUN" ]]; then
    echo "Run path already exists: $RUN. Refusing to overwrite or silently resume." >&2
    exit 1
fi

export PYTHONPATH="$(pwd)"
export EMU_PHYSICAL_PROFILE=h40
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export EMU_CONSOLE_MODE="${EMU_CONSOLE_MODE:-auto}"
export EMU_UNIFIED_CONTINUOUS=cold-growth-approved
PARENT_HASH="$(sha256sum "$PARENT" | awk '{print $1}')"

FLAGS=(
    --profile sim2real
    --terrain-actor
    --terrain-step-max-m .08
    --unified-capabilities
    --terrain-carpet --carpet-prisms
    --academy-stream
    --terrain-lesson coherent-careful
    --teaching-schedule staged
    --growth-recipe main-mature
    --athletic-rehearsal-share .15
    --retention-share .20
    --science-step-velocity-relief .80
    --fixed-eval-every-steps 10000000
    --progressive-school --refined-school --bidirectional-school
)
TRAIN=(
    python -m emu.cli train --run-dir "$RUN" --device cuda
    --envs 2048 --steps 420000000 --seed 735
    "${FLAGS[@]}"
    --resume "$PARENT"
    --research-parent-source "$OLD_PIN"
    --research-parent-sha256 "$PARENT_HASH"
)

echo "=== B: VERIFY THE 180M SOURCE, CHECKPOINT HASH, H40 OBSERVATIONS AND FORK ==="
"${TRAIN[@]}" --check-resume-only
echo "=== B: PAIRED UP+DOWN, REAL 8 PRISMS, FEET-ONLY COLLISIONS ==="
"${TRAIN[@]}"
