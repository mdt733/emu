#!/usr/bin/env bash
# ONE H40 cold school. One teacher, random policy at birth, one PPO process.
# Usage: bash scripts/run_h40_school_wsl.sh plan|preflight|gpu-smoke|all|report
set -euo pipefail
MODE="${1:-plan}"
case "$MODE" in plan|preflight|gpu-smoke|all|report) ;; *)
  echo "Usage: bash $0 plan|preflight|gpu-smoke|all|report" >&2; exit 2 ;;
esac
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
PY="$REPO/.venv/bin/python"
[[ -x "$PY" ]] || { echo "Install the repo's .venv before running" >&2; exit 1; }
SEED="${EMU_SCHOOL_SEED:-735}"
[[ "$SEED" =~ ^[0-9]+$ ]] || {
  echo "EMU_SCHOOL_SEED must be a nonnegative integer" >&2; exit 2; }
ROOT="$HOME/emu-data/runs/h40-school-cold-s$SEED"
STEPS=180000000
export PYTHONPATH="$REPO"
export EMU_PHYSICAL_PROFILE=h40-spring-only
export EMU_UNIFIED_CONTINUOUS=cold-growth-approved
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# A single, PREDECLARED teacher recipe for every training seed. The
# nursery, stairs, and strength/recovery subjects all share one optimiser.
# Each learner starts at ZERO; age-based lessons are inside the teacher
# and cannot be changed manually during its lifetime.
RECIPE=(
  --profile sim2real --run-dir "$ROOT"
  --steps "$STEPS" --envs 2048 --seed "$SEED" --device cuda
  --pitch-deg 8 --terrain-actor --terrain-step-max-m .055
  --unified-capabilities --terrain-carpet --carpet-prisms
  --carpet-transition-cap-deg 3
  --academy-stream --stair-survival-curriculum
  --terrain-lesson coherent-careful
  --foundation-steps 20000000 --foundation-mix main
  --growth-recipe main-mature --teaching-schedule terrain-early
  --force-program gentle --athletic-rehearsal-share .15
  --retention-share .30
  --academy-intro-steps 60000000 --academy-start-steps 80000000
  --academy-ramp-steps 20000000 --academy-intro-share .10
  --academy-max-share .28 --academy-height-cap-m .055
  --academy-course-episode-frames 1200
  --fixed-eval-every-steps 20000000
  --science-step-goal-weight .12
  --science-step-velocity-relief .55
  --science-step-approach-m .22
  --science-bilateral-support-bonus 6
  --science-followthrough-bonus 7
)

if [[ "$MODE" == plan ]]; then
  "$PY" -m emu.cli train --plan "${RECIPE[@]}"
  echo "COLD SOURCE-PINNED school: seed $SEED, 0 -> $STEPS steps."
  echo "No warm forks, no mid-age teacher edits; no files created."
  exit 0
fi

# The source must be one immutable commit for the entire student's life.
[[ "$(git branch --show-current)" == research/h40-unified-school-cold-v1 ]] || {
  echo "Wrong school branch; no run started" >&2; exit 1; }
[[ -z "$(git status --porcelain)" ]] || {
  echo "Dirty Git source; commit or discard changes before launch" >&2; exit 1; }

# A restart uses only the CURRENT run's checkpoints. No external parent
# path, source-override or "research fork" flags are ever supplied.
RESUME=()
if [[ -d "$ROOT/checkpoints" ]] && \
   compgen -G "$ROOT/checkpoints/*.pkl" >/dev/null; then
  RESUME=(--resume auto)
fi

verify_model() {
  "$PY" - <<'PY'
import hashlib
from emu.contracts import ROBOT_XML
from emu.physical_variants import selected_profile
assert selected_profile() == "h40-spring-only"
want = "21b5856b7a248e5d5b988e2735255d57ec79bae1a7cab0bcb72f481bfc36f470"
assert hashlib.sha256(ROBOT_XML.read_bytes()).hexdigest() == want, \
    "Cold school physics differ from proven KS150 tendon-only model"
print("PHYSICS PASS: exactly Spring-only KS150, original motor/ROM preserved")
PY
}

if [[ "$MODE" == preflight || "$MODE" == gpu-smoke || "$MODE" == all ]]; then
  verify_model
  "$PY" -m emu.cli train --check-resume-only \
    "${RESUME[@]}" "${RECIPE[@]}"
fi
if [[ "$MODE" == preflight ]]; then
  echo "COLD SCHOOL PREFLIGHT PASS: no training or source writes."
  exit 0
fi
if [[ "$MODE" == gpu-smoke ]]; then
  "$PY" scripts/smoke_gpu_carpet.py --worlds 2048 --frames 2 \
    --device cuda --prisms --ppo --full-update \
    --stair-survival-curriculum
  echo "COLD SCHOOL GPU smoke PASS. No training started."
  exit 0
fi

if [[ "$MODE" == report ]]; then
  [[ -f "$ROOT/checkpoints/final.pkl" ]] || {
    echo "No completed school final checkpoint to report" >&2; exit 1; }
  "$PY" -m emu.results --root "$ROOT" --no-bundle
  echo "Read-only report refreshed; no GPU exams and no training."
  exit 0
fi

if [[ "$MODE" == all ]]; then
  # ONE continuous PPO target, same teacher/optimizer. Standard crash
  # checkpoints and final.pkl are the only checkpoint types; no research
  # stage copies, no 20M/40M human teacher swaps.
  "$PY" -m emu.cli train "${RESUME[@]}" "${RECIPE[@]}"
fi

FINAL="$ROOT/checkpoints/final.pkl"
[[ -f "$FINAL" ]] || {
  echo "No completed cold-school final checkpoint to evaluate" >&2; exit 1; }

# One truly held-out FIRST-LIFE physical seed for the complete final
# capability suite. 1701/2701 were used to DESIGN this teacher and may
# not be called independent out-of-sample qualification.
EXAM_SEED=4701
# Read-only final exams; one actor, no comparison clutter in the run root.
POLICY="$ROOT/final.emupolicy"
if [[ ! -f "$POLICY" ]]; then
  "$PY" -m emu.cli export-policy "$FINAL" "$POLICY"
fi
OUT="$ROOT/exam"
mkdir -p "$OUT"
if [[ ! -f "$OUT/stairs/matrix_summary.json" ]]; then
  [[ ! -e "$OUT/stairs" ]] || {
    echo "Incomplete stair exam: $OUT/stairs; inspect before rerun" >&2; exit 1; }
  "$PY" scripts/v5053_matrix_exam.py --policy "$POLICY" \
    --output "$OUT/stairs" --mode stairs --worlds 2048 \
    --horizon 350 --levels 20 --trials-per-level 50 \
    --seed "$EXAM_SEED" --device cuda
fi
if [[ ! -f "$OUT/pulse-s4701.zip" ]]; then
  "$PY" -m scripts.examine_h40_specialties --policy "$POLICY" \
    --output "$OUT/pulse-s4701.zip" --worlds 2048 \
    --horizon 600 --seed "$EXAM_SEED" --device cuda
fi
if [[ ! -f "$OUT/deep-s4701.zip" ]]; then
  "$PY" -m emu.physical_capability_exam --policy "$POLICY" \
    --output "$OUT/deep-s4701.zip" --worlds 8 \
    --seed "$EXAM_SEED" --device cuda
fi
"$PY" -m emu.results --root "$ROOT" --no-bundle >/dev/null
"$PY" - "$ROOT" <<'PY'
import os,sys
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
root=Path(sys.argv[1]).expanduser()
rel=[Path("training_config.json"),Path("run_meta.json"),
     Path("results_data.json"),Path("results.md"),
     Path("metrics.csv"),
     Path("exam/stairs/matrix_summary.json"),
     Path("exam/stairs/matrix_curve.csv"),
     Path("exam/stairs/matrix_worlds.csv"),
     Path("exam/pulse-s4701.zip"),Path("exam/deep-s4701.zip")]
for path in rel:
    if not (root/path).is_file():
        raise RuntimeError(f"Missing required physical evidence: {path}")
# Teacher quality requires the whole progression, not just a winning
# policy. Preserve fixed evaluations and throughput/health if available.
rel += [p for p in (Path("fixed_eval.csv"),
                    Path("performance.json"),
                    Path("capability_field_summary.json"))
        if (root/p).is_file()]
out=root/"school-evidence.zip"
if out.exists():
    print(f"Previous evidence bundle retained (not overwritten): {out}")
else:
    tmp=root/"school-evidence.zip.partial"
    if tmp.exists(): raise FileExistsError(tmp)
    try:
        with ZipFile(tmp,"x",compression=ZIP_DEFLATED) as z:
            for path in rel:
                z.write(root/path,arcname=path.as_posix())
        os.replace(tmp,out)
    finally:
        tmp.unlink(missing_ok=True)
    print(f"Ready for independent school-teacher audit: {out}")
PY
echo "School result: $FINAL; all actor promotions remain manual."
