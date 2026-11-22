#!/usr/bin/env bash
# H40-KS150 / STAIRS-BAL. One automated 120M->160M campaign.
# Modes: plan (default), preflight, all, report. No implicit GPU launch.
# Stage checkpoints are immutable; never touches 120M source or showcase.
set -euo pipefail
MODE="plan"
if (( $# )); then MODE="$1"; fi
case "$MODE" in plan|preflight|all|report) ;; *)
  echo "Usage: bash $0 plan|preflight|all|report" >&2; exit 2 ;;
esac
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
PY="$REPO/.venv/bin/python"
[[ -x "$PY" ]] || { echo "Missing .venv/bin/python" >&2; exit 1; }
export PYTHONPATH="$REPO"
export EMU_PHYSICAL_PROFILE=h40-spring-only
export EMU_UNIFIED_CONTINUOUS=cold-growth-approved
export XLA_PYTHON_CLIENT_PREALLOCATE=false
PARENT="$HOME/emu-data/runs/h40-spring-ab-s735/checkpoints/final.pkl"
ROOT="$HOME/emu-data/runs/h40-ks150-stairs-bal-s735"
if [[ "$MODE" == all ]]; then
  [[ "$(git branch --show-current)" == "research/h40-ks150-balanced-stairs" ]] || {
    echo "Wrong branch; refusing to train" >&2; exit 1; }
  [[ -z "$(git status --porcelain)" ]] || {
    echo "Dirty Git checkout; refusing unpinned training" >&2; exit 1; }
fi
# Exact source/actor/model/optimizer parent, not merely a compatible shape.
read -r PARENT_SOURCE PARENT_SHA < <("$PY" - "$PARENT" <<'PY'
import hashlib,sys
from pathlib import Path
from emu.contracts import ROBOT_XML
from emu.checkpoint_io import load_checkpoint
from emu.policy_export import build_bundle
from emu.training_config import TrainingConfig, training_sensor_contract
p=Path(sys.argv[1]).expanduser().resolve()
assert p.is_file(), f"Missing frozen 120M parent: {p}"
model_hash=hashlib.sha256(ROBOT_XML.read_bytes()).hexdigest()
assert model_hash=="21b5856b7a248e5d5b988e2735255d57ec79bae1a7cab0bcb72f481bfc36f470",model_hash
cfg=TrainingConfig.load(p.parent.parent/"training_config.json")
assert cfg.physical_profile=="h40-spring-only" and cfg.seed==735 and cfg.steps==120000000
ck=load_checkpoint(p,expected_sensor_contract=training_sensor_contract(cfg))
assert ck["meta"]["step"]==120061952
source=(ck["meta"].get("source_provenance") or {}).get("git_commit")
assert source=="5235ea687d04948b3dce88880e644eac5bb74cee",source
bundle=build_bundle(p)
assert bundle["source"]["actor_sha256"]=="cc1f0ab126126815f6cea907dbc3a8a2b3faa6caff8ccbe2f05133d7c090771d"
assert bundle["source"]["robot_model_sha256"]==model_hash
digest=hashlib.sha256()
with p.open("rb") as handle:
    for chunk in iter(lambda:handle.read(1048576),b""): digest.update(chunk)
print(source,digest.hexdigest())
PY
)
if [[ "$MODE" == report ]]; then
  "$PY" scripts/summarize_h40_ks150_stairs.py --root "$ROOT"
  exit 0
fi
# Same original PPO/sensors/motor controls. Changed: bounded 5.5cm signed
# stair training, outcome-based school and protected ordinary-control births.
set -- --profile sim2real --run-dir "$ROOT" \
  --envs 2048 --seed 735 --device cuda --pitch-deg 8 \
  --terrain-actor --terrain-step-max-m .055 \
  --unified-capabilities --terrain-carpet --carpet-prisms \
  --carpet-transition-cap-deg 3 \
  --academy-stream --terrain-lesson coherent-careful \
  --foundation-steps 20000000 --foundation-mix main \
  --growth-recipe main-mature --teaching-schedule terrain-early \
  --force-program gentle --athletic-rehearsal-share .15 \
  --retention-share .30 --academy-intro-steps 60000000 \
  --academy-start-steps 80000000 --academy-ramp-steps 20000000 \
  --academy-intro-share .10 --academy-max-share .28 \
  --academy-height-cap-m .055 --fixed-eval-every-steps 10000000 \
  --science-step-goal-weight .12 \
  --science-step-velocity-relief .55 \
  --science-step-approach-m .22 \
  --science-bilateral-support-bonus 6 \
  --science-followthrough-bonus 7
if [[ "$MODE" == plan ]]; then
  "$PY" -m emu.cli train --steps 160000000 --plan "$@"
  exit 0
fi
if [[ ! -e "$ROOT/training_config.json" ]]; then
  "$PY" -m emu.cli train --steps 130000000 \
    --resume "$PARENT" --research-parent-source "$PARENT_SOURCE" \
    --research-parent-sha256 "$PARENT_SHA" --check-resume-only "$@"
fi
if [[ "$MODE" == preflight ]]; then
  echo "PREFLIGHT PASS: frozen parent and Spring-only model verified; no training."
  exit 0
fi

# Automatic phases, same optimizer/governor/source. NOT four new learners.
for AGE in 130 140 150 160; do
  STAGE="$ROOT/stages/$AGE"m
  TARGET=$((AGE*1000000))
  echo "=== KS150 BALANCED STAIRS $AGE M ==="
  if [[ ! -f "$STAGE/stage_manifest.json" ]]; then
    if [[ -e "$ROOT/training_config.json" ]]; then
      CFG_TARGET="$("$PY" - "$ROOT" <<'PY'
import sys
from pathlib import Path
from emu.training_config import TrainingConfig
print(TrainingConfig.load(Path(sys.argv[1])/"training_config.json").steps)
PY
)"
      if (( CFG_TARGET > TARGET )); then
        echo "Missing archived $AGE M stage after a later training target; refusing false snapshot" >&2
        exit 1
      fi
      "$PY" -m emu.cli train --steps "$TARGET" --resume auto "$@"
    else
      "$PY" -m emu.cli train --steps "$TARGET" \
        --resume "$PARENT" --research-parent-source "$PARENT_SOURCE" \
        --research-parent-sha256 "$PARENT_SHA" "$@"
    fi
  fi
  # The root must first have a trainer-created config and run_meta.
  # Matched 120M reference exams are read-only and do not mutate parent.
  mkdir -p "$ROOT/reference"
  if [[ ! -e "$ROOT/reference/pulse-s1701.zip" ]]; then
    "$PY" -m scripts.examine_h40_specialties --policy "$PARENT" \
      --seed 1701 --device cuda --worlds 2048 --horizon 600 \
      --output "$ROOT/reference/pulse-s1701.zip"
  fi
  if [[ ! -e "$ROOT/reference/actor.emupolicy" ]]; then
    "$PY" -m emu.cli export-policy "$PARENT" "$ROOT/reference/actor.emupolicy"
  fi
  if [[ ! -f "$ROOT/reference/stairs/matrix_summary.json" ]]; then
    "$PY" scripts/v5053_matrix_exam.py --policy "$ROOT/reference/actor.emupolicy" \
      --output "$ROOT/reference/stairs" --mode stairs --worlds 2048 \
      --horizon 350 --levels 20 --trials-per-level 50 --seed 1701 --device cuda
  fi
  # Re-examine parent and child identically for 60s standing + 30s commands.
  if [[ ! -e "$ROOT/reference/deep-s1701.zip" ]]; then
    "$PY" -m emu.physical_capability_exam \
      --policy "$ROOT/reference/actor.emupolicy" \
      --output "$ROOT/reference/deep-s1701.zip" \
      --worlds 8 --seed 1701 --device cuda
  fi
  "$PY" scripts/summarize_h40_ks150_stairs.py --archive-stage "$AGE" \
    --root "$ROOT" --parent "$PARENT" --parent-sha256 "$PARENT_SHA"
  POLICY="$STAGE/actor.emupolicy"
  if [[ ! -e "$POLICY" ]]; then
    "$PY" -m emu.cli export-policy "$STAGE/final.pkl" "$POLICY"
  fi
  if [[ ! -e "$STAGE/pulse-s1701.zip" ]]; then
    "$PY" -m scripts.examine_h40_specialties --policy "$POLICY" \
      --seed 1701 --device cuda --worlds 2048 --horizon 600 \
      --output "$STAGE/pulse-s1701.zip"
  fi
  if [[ ! -f "$STAGE/stairs/matrix_summary.json" ]]; then
    "$PY" scripts/v5053_matrix_exam.py --policy "$POLICY" \
      --output "$STAGE/stairs" --mode stairs --worlds 2048 \
      --horizon 350 --levels 20 --trials-per-level 50 --seed 1701 --device cuda
  fi
  if [[ ! -e "$STAGE/deep-s1701.zip" ]]; then
    "$PY" -m emu.physical_capability_exam \
      --policy "$POLICY" --output "$STAGE/deep-s1701.zip" \
      --worlds 8 --seed 1701 --device cuda
  fi
  "$PY" scripts/summarize_h40_ks150_stairs.py --root "$ROOT"
done
echo "Finished. Inspect $ROOT/selection.md before any promotion."
