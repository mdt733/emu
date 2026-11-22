#!/usr/bin/env bash
# KS150 SAFE ASCENT V2: one protected 150M->170M PPO continuation.
# plan, preflight, gpu-smoke, all, report, bundle. No implicit promotion.
set -euo pipefail
MODE="${1:-plan}"
case "$MODE" in plan|preflight|gpu-smoke|all|report|bundle) ;; *)
  echo "Usage: bash $0 plan|preflight|gpu-smoke|all|report|bundle" >&2; exit 2 ;;
esac
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
PY="$REPO/.venv/bin/python"
[[ -x "$PY" ]] || { echo "Missing $PY" >&2; exit 1; }
export PYTHONPATH="$REPO"
export EMU_PHYSICAL_PROFILE=h40-spring-only
export EMU_UNIFIED_CONTINUOUS=cold-growth-approved
export XLA_PYTHON_CLIENT_PREALLOCATE=false
V1="$HOME/emu-data/runs/h40-ks150-stairs-bal-s735"
PARENT="$V1/stages/150m/checkpoints/final.pkl"
PARENT_POLICY="$V1/stages/150m/actor.emupolicy"
ROOT="$HOME/emu-data/runs/h40-ks150-safe-ascent-s735"
PARENT_SOURCE="4504ac2c7219e5df8a106a224b77825fee219ead"
PARENT_SHA="c66d664d251d7ee441f42e6ca322cd3240e04f45087bad1a1543a7e2ce855d4f"
if [[ "$MODE" == all || "$MODE" == gpu-smoke ]]; then
  [[ "$(git branch --show-current)" == "research/h40-ks150-safe-ascent-v2" ]] || {
    echo "Wrong research branch" >&2; exit 1; }
  [[ -z "$(git status --porcelain)" ]] || {
    echo "Dirty source checkout; cannot reproduce the run" >&2; exit 1; }
fi
# Verify full optimizer parent and original export against the frozen V1 ZIP.
"$PY" - "$PARENT" "$PARENT_POLICY" "$PARENT_SOURCE" "$PARENT_SHA" <<'PY'
import sys,hashlib
from pathlib import Path
from emu.contracts import ROBOT_XML
from emu.checkpoint_io import load_checkpoint
from emu.training_config import TrainingConfig,training_sensor_contract
from emu.policy_export import load_exported_actor_for_exam
from emu.physical_variants import selected_profile
p,policy=[Path(x).expanduser() for x in sys.argv[1:3]]
source,parent_sha=sys.argv[3:5]
model="21b5856b7a248e5d5b988e2735255d57ec79bae1a7cab0bcb72f481bfc36f470"
assert selected_profile()=="h40-spring-only"
assert hashlib.sha256(ROBOT_XML.read_bytes()).hexdigest()==model
cfg=TrainingConfig.load(p.parent.parent/"training_config.json")
assert cfg.seed==735 and cfg.steps==160000000
assert cfg.physical_profile=="h40-spring-only"
assert cfg.terrain_step_max_m==cfg.academy_height_cap_m==.055
ck=load_checkpoint(p,expected_sensor_contract=training_sensor_contract(cfg))
assert ck["meta"]["step"]==150011904
assert ck["meta"]["source_provenance"]["git_commit"]==source
h=hashlib.sha256()
with p.open("rb") as handle:
    for chunk in iter(lambda:handle.read(1048576),b""):h.update(chunk)
assert h.hexdigest()==parent_sha
_,_,actor=load_exported_actor_for_exam(policy)
assert actor["source"]["actor_sha256"]=="9a6e5f4bd09ff6fc258d8e32055f7b258c04ac1206f872d517d5a22821280240"
assert actor["source"]["robot_model_sha256"]==model
assert actor["source"]["step"]==150011904
assert actor["source"]["git_commit"]==source
assert actor["training"]["terrain_carpet_prisms"] is True
print("VERIFIED full V1 150M parent, exact PPO/actor/model/source/checkpoint bytes")
PY

# V1 objective / actuator / PPO DOCTRINE retained. Only the live V2
# first-life safety teacher is new, and the 40mm UP fork clamp.
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
  "$PY" -m emu.cli train --steps 170000000 --plan "$@"
  echo "Plan only: 150M -> 170M; V2 safe uphill starts <=40mm."
  exit 0
fi
if [[ -f "$ROOT/training_config.json" ]]; then
  FORK=(--resume auto)
else
  FORK=(--resume "$PARENT" \
    --research-parent-source "$PARENT_SOURCE" \
    --research-parent-sha256 "$PARENT_SHA")
fi
if [[ "$MODE" == preflight || "$MODE" == gpu-smoke ]]; then
  "$PY" -m emu.cli train --steps 170000000 \
    --check-resume-only "${FORK[@]}" "$@"
  # Exercise *actual* V1 checkpoint -> V2 governor schema migration on CPU.
  # The CLI's generic rewind check verifies PPO tensors but not the new
  # safety-window migration, so fail fast before allocating 8GB GPU.
  "$PY" - "$PARENT" <<'PY'
from pathlib import Path
from emu.checkpoint_io import load_checkpoint
from emu.training_config import TrainingConfig, training_sensor_contract
from emu.curriculum import RegimeGovernor
import sys
p=Path(sys.argv[1])
cfg=TrainingConfig.load(p.parent.parent/"training_config.json")
ck=load_checkpoint(p,expected_sensor_contract=training_sensor_contract(cfg))
governor=RegimeGovernor(unified_capability_focus=True,
                        safe_ascent_enabled=True)
governor.load_state_dict(ck["curriculum_state"],
                         allow_safe_ascent_parent_fork=True)
old_down=governor.params.academy_height_down_m
governor.params.academy_height_up_m=min(.040,governor.params.academy_height_up_m)
governor.params.terrain_height_m=min(.040,governor.params.terrain_height_m)
governor.terrain_evidence=governor._empty_terrain_evidence()
governor.academy_evidence=governor._empty_academy_evidence()
assert governor.params.academy_height_up_m<=.040
assert governor.params.terrain_height_m<=.040
assert governor.params.academy_height_down_m==old_down
assert governor.safe_ascent_evidence==governor._empty_safe_ascent_evidence()
state=governor.state_dict()
again=RegimeGovernor(unified_capability_focus=True,safe_ascent_enabled=True)
again.load_state_dict(state)
assert again.state_dict()==state, "V2 checkpoint/resume roundtrip failed"
print("V2 fork migration PASS: full 150M governor, 40mm uphill, "
      "downhill unchanged, survival evidence restartable")
PY
  if [[ "$MODE" == preflight ]]; then
    echo "SOURCE-PINNED PREFLIGHT PASS; no training."
    exit 0
  fi
  "$PY" scripts/smoke_gpu_carpet.py --worlds 2048 --frames 2 \
      --device cuda --prisms --ppo --full-update
  echo "SPRING-ONLY CUDA full-PPO smoke passed; no training."
  exit 0
fi
if [[ "$MODE" == all ]]; then
  "$PY" -m emu.cli train --steps 170000000 "${FORK[@]}" "$@"
fi
[[ -f "$ROOT/checkpoints/final.pkl" ]] || {
  echo "No finished 170M learner; report unavailable" >&2; exit 1; }
# Training has completed: evaluate 150/160/170 frozen learners together
# on NEW seed3701 and same V2 source for apples-to-apples physics.
for AGE in 150 160 170; do
  OUT="$ROOT/exams/${AGE}m"
  mkdir -p "$OUT"
  if [[ "$AGE" == 150 ]]; then
    POLICY="$PARENT_POLICY"
  else
    if [[ "$AGE" == 160 ]]; then
      SNAP="$ROOT/checkpoints/milestone_160m.pkl"
    else
      SNAP="$ROOT/checkpoints/final.pkl"
    fi
    [[ -f "$SNAP" ]] || { echo "Missing exact $AGE M milestone" >&2; exit 1; }
    POLICY="$OUT/actor.emupolicy"
    if [[ ! -f "$POLICY" ]]; then
      "$PY" -m emu.cli export-policy "$SNAP" "$POLICY"
    fi
  fi
  if [[ ! -f "$OUT/stairs/matrix_summary.json" ]]; then
    [[ ! -e "$OUT/stairs" ]] || {
      echo "Partial stair directory, inspect before retry: $OUT/stairs" >&2; exit 1; }
    "$PY" scripts/v5053_matrix_exam.py --policy "$POLICY" \
      --output "$OUT/stairs" --mode stairs --worlds 2048 \
      --horizon 350 --levels 20 --trials-per-level 50 \
      --seed 3701 --device cuda
  fi
  if [[ ! -f "$OUT/pulse-s3701.zip" ]]; then
    "$PY" -m scripts.examine_h40_specialties --policy "$POLICY" \
      --output "$OUT/pulse-s3701.zip" --worlds 2048 \
      --horizon 600 --seed 3701 --device cuda
  fi
  if [[ ! -f "$OUT/deep-s3701.zip" ]]; then
    "$PY" -m emu.physical_capability_exam --policy "$POLICY" \
      --output "$OUT/deep-s3701.zip" --worlds 8 \
      --seed 3701 --device cuda
  fi
done
"$PY" scripts/score_h40_safe_ascent_v2.py --root "$ROOT" --seed 3701
if [[ ! -e "$ROOT/v2-evidence-s3701.zip" ]]; then
  "$PY" - "$ROOT" <<'PY'
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import sys,os
root=Path(sys.argv[1]).expanduser()
files=[Path("v2-exam-report.md"),
       Path("training_config.json"),Path("run_meta.json"),
       Path("results_data.json")]
for age in (150,160,170):
    d=Path("exams")/f"{age}m"
    files.extend([
        d/"stairs"/"matrix_summary.json",
        d/"stairs"/"matrix_curve.csv",
        d/"stairs"/"matrix_worlds.csv",
        d/"pulse-s3701.zip",
        d/"deep-s3701.zip",
    ])
files=[p for p in files if p.name!="results_data.json" or (root/p).is_file()]
for rel in files:
    assert (root/rel).is_file(),f"Missing complete evidence: {rel}"
out=root/"v2-evidence-s3701.zip"
temp=root/"v2-evidence-s3701.zip.partial"
if out.exists() or temp.exists(): raise FileExistsError("V2 evidence bundle exists")
try:
    with ZipFile(temp,"x",ZIP_DEFLATED) as z:
        for rel in files: z.write(root/rel,arcname=str(rel))
    os.replace(temp,out)
finally:
    temp.unlink(missing_ok=True)
print(f"UPLOAD AUDIT: {out}")
PY
fi
echo "Complete. No automatic promotion or source/checkpoint changes."
echo "Review $ROOT/v2-exam-report.md and $ROOT/v2-evidence-s3701.zip"
