#!/usr/bin/env bash
# H40-KS150: independent held-out evaluation, seed 2701, NO PPO training.
# Modes: plan (default), preflight, all, report. All outputs separate.
# This does not choose or publish a champion. Requires completed V1 exports.
set -euo pipefail
MODE="\${1:-plan}"
case "$MODE" in plan|preflight|all|report) ;; *)
  echo "Usage: bash $0 plan|preflight|all|report" >&2; exit 2 ;;
esac
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
PY="$REPO/.venv/bin/python"
[[ -x "$PY" ]] || { echo "Missing $PY" >&2; exit 1; }
export PYTHONPATH="$REPO"
export EMU_PHYSICAL_PROFILE=h40-spring-only
export XLA_PYTHON_CLIENT_PREALLOCATE=false
SOURCE="$HOME/emu-data/runs/h40-ks150-stairs-bal-s735"
DEST="$HOME/emu-data/exams/h40-ks150-heldout-s2701"
SEED=2701

if [[ "$MODE" == plan ]]; then
  cat <<EOF
READ-ONLY held-out exam; no learning and no overwrite of V1:
  source: $SOURCE
  output: $DEST
  physical: h40-spring-only
  actors: 120M, 150M, 160M
  seed: $SEED (different evaluation RNG from 1701, NOT a training seed)
  each actor: 2048-world 12s pulse, 2048-world 4..80mm
  signed-prism matrix, and 8-world 60s/30s deep exam
Run preflight then all. Use report after the exams.
EOF
  exit 0
fi
if [[ "$MODE" == report ]]; then
  "$PY" scripts/score_h40_ks150_heldout.py --root "$DEST" --seed "$SEED"
  exit 0
fi
[[ "$(git branch --show-current)" == research/h40-ks150-safe-ascent-v2 ]] || {
  echo "Wrong research branch; refused" >&2; exit 1; }
[[ -z "$(git status --porcelain)" ]] || {
  echo "Dirty research checkout; freeze exact examiner source first" >&2; exit 1; }

# Check EVERY actor identity, model, actual source age, and source provenance
# before allocating any 2048-world CUDA memory.
"$PY" - "$SOURCE" <<'PY'
import sys,hashlib,json
from pathlib import Path
from emu.contracts import ROBOT_XML
from emu.policy_export import load_exported_actor_for_exam
from emu.physical_variants import selected_profile
root=Path(sys.argv[1]).expanduser()
MODEL="21b5856b7a248e5d5b988e2735255d57ec79bae1a7cab0bcb72f481bfc36f470"
assert selected_profile()=="h40-spring-only"
assert hashlib.sha256(ROBOT_XML.read_bytes()).hexdigest()==MODEL
identities={
 "120m":("reference/actor.emupolicy",
         "cc1f0ab126126815f6cea907dbc3a8a2b3faa6caff8ccbe2f05133d7c090771d",
         120061952, "5235ea687d04948b3dce88880e644eac5bb74cee"),
 "150m":("stages/150m/actor.emupolicy",
         "9a6e5f4bd09ff6fc258d8e32055f7b258c04ac1206f872d517d5a22821280240",
         150011904, "4504ac2c7219e5df8a106a224b77825fee219ead"),
 "160m":("stages/160m/actor.emupolicy",
         "cbe47e8225079ba67b899732f920998469320a22e26aa9d67f784c916a4cb5ac",
         160038912, "4504ac2c7219e5df8a106a224b77825fee219ead"),
}
for name,(rel,expected_actor,step,source) in identities.items():
    path=root/rel
    assert path.is_file(),f"Missing original immutable V1 export {path}"
    _,_,actor=load_exported_actor_for_exam(path)
    a=actor["source"]
    assert a["actor_sha256"]==expected_actor,(name,"actor identity changed")
    assert a["robot_model_sha256"]==MODEL
    assert a["step"]==step,(name,a["step"],step)
    assert a["git_commit"]==source,(name,"training source changed")
    assert actor["training"].get("terrain_carpet_prisms") is True
    print("VERIFIED",name,step,expected_actor[:15],path)
print("HELDOUT PREFLIGHT PASS: immutable 120/150/160 actors + KS150 physics")
PY
if [[ "$MODE" == preflight ]]; then
  echo "No GPU work launched."
  exit 0
fi

mkdir -p "$DEST"
for AGE in 120 150 160; do
  if [[ "$AGE" == 120 ]]; then
    POLICY="$SOURCE/reference/actor.emupolicy"
  else
    POLICY="$SOURCE/stages/\${AGE}m/actor.emupolicy"
  fi
  OUT="$DEST/\${AGE}m"
  mkdir -p "$OUT"
  if [[ ! -e "$OUT/pulse-s2701.zip" ]]; then
    "$PY" -m scripts.examine_h40_specialties \
      --policy "$POLICY" --output "$OUT/pulse-s2701.zip" \
      --seed "$SEED" --worlds 2048 --horizon 600 --device cuda
  fi
  if [[ ! -e "$OUT/stairs/matrix_summary.json" ]]; then
    # Do not mix partial old output directories into a rerun.
    [[ ! -e "$OUT/stairs" ]] || {
      echo "Incomplete stair result folder: $OUT/stairs; inspect it first" >&2; exit 1; }
    "$PY" scripts/v5053_matrix_exam.py \
      --policy "$POLICY" --output "$OUT/stairs" \
      --seed "$SEED" --mode stairs --worlds 2048 \
      --horizon 350 --levels 20 --trials-per-level 50 --device cuda
  fi
  if [[ ! -e "$OUT/deep-s2701.zip" ]]; then
    "$PY" -m emu.physical_capability_exam \
      --policy "$POLICY" --output "$OUT/deep-s2701.zip" \
      --seed "$SEED" --worlds 8 --device cuda
  fi
  "$PY" scripts/score_h40_ks150_heldout.py --root "$DEST" --seed "$SEED"
done
echo "Held-out seed2701 physical exams complete; NO training changes."
echo "Review: $DEST/heldout-report.md"
