#!/usr/bin/env bash
# ONE science experiment: Spring V1 120M -> 140M outcome-only stair commitment.
# Full rated torque, same spring/robot, same 360 actor, no gait clock payments.
# New lesson is ONLY Academy fixed-destination signed progress + optional
# temporary forward-speed relief and matched near-lip birth distance.
# Stage 122M smoke/gate, then exactly one 140M continuation and frozen exam.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
BRANCH="work/emu-stair-outcome-tutor"
if [[ "$(git branch --show-current)" != "$BRANCH" || -n "$(git status --porcelain)" ]]; then
  echo "Clean Git branch $BRANCH required; main and reference must remain untouched" >&2
  exit 1
fi
PY="$REPO/.venv/bin/python"
[[ -x "$PY" ]] || { echo "Missing proven EMU .venv/bin/python" >&2; exit 1; }
export EMU_PHYSICAL_PROFILE=h40-spring-v1
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export EMU_UNIFIED_CONTINUOUS=cold-growth-approved
export EMU_TUTOR_ROOT="$HOME/emu-data/runs/h40-stair-outcome-tutor-140m-s735"
export EMU_TUTOR_PARENT="$HOME/emu-data/runs/h40-spring-v1-outcomes-s735-repaired"
export EMU_TUTOR_REFERENCE="$HOME/emu-data/actors/h40-spring-v1-120m-reference"
export EMU_TUTOR_BASELINE="$HOME/emu-data/exams/h40-spring-v1-mechanics-s735"
ORIGIN_SHA="a7c6467c59f81c124cd76e3348a4f12baa6f438e"
ORIGIN_CKPT_SHA="a542722eaf0ddfc27194c68d77c47d907139f49c332f2d81e1b9b9fc40cb2a0b"
MODEL_SHA="7085b1732fe6b876b27ef9b485853d0127c70f6e1c1a9b52b57af82c34de6cfd"
ACTOR_SHA="26095f7f10084c815ee1f7ad24363834cdd017d4341bca9ec1d9f10e4372b16c"
PARENT="$EMU_TUTOR_PARENT/checkpoints/final.pkl"

# Fail before training if we cannot reproduce EXACT parent+baseline.
"$PY" - <<'PY'
import hashlib,json,os
from pathlib import Path
from emu.contracts import ROBOT_XML
from emu.training_config import TrainingConfig,training_sensor_contract
from emu.checkpoint_io import load_checkpoint
from scripts.v5053_matrix_exam import exam_source_fingerprint
orig=Path(os.environ["EMU_TUTOR_PARENT"])
reference=Path(os.environ["EMU_TUTOR_REFERENCE"])
baseline=Path(os.environ["EMU_TUTOR_BASELINE"])
expected_actor="26095f7f10084c815ee1f7ad24363834cdd017d4341bca9ec1d9f10e4372b16c"
expected_ckpt="a542722eaf0ddfc27194c68d77c47d907139f49c332f2d81e1b9b9fc40cb2a0b"
expected_physics="7085b1732fe6b876b27ef9b485853d0127c70f6e1c1a9b52b57af82c34de6cfd"
expected_git="a7c6467c59f81c124cd76e3348a4f12baa6f438e"
expected_exam="99b690b3c4aa0758157a9854e6442f5abc5572d1dd9f53c958bd2e4299ac4953"
def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()
if sha(ROBOT_XML)!=expected_physics:
    raise RuntimeError("Compiled-source robot XML changed: refusing new physical experiment")
if exam_source_fingerprint()!=expected_exam:
    raise RuntimeError("Independent exam changed since frozen Spring V1 reference; invalid A/B")
for p,checksum in (
    (orig/"checkpoints/final.pkl",expected_ckpt),
    (reference/"final.pkl",expected_ckpt),
    (orig/"h40-spring-v1-outcomes-120m.emupolicy",expected_actor),
    (reference/"h40-spring-v1-outcomes-120m.emupolicy",expected_actor),
):
    if not p.is_file() or sha(p)!=checksum:
        raise RuntimeError(f"Original 120M actor/optimiser changed or missing: {p}")
cfg=TrainingConfig.load(orig/"training_config.json")
meta=json.loads((orig/"run_meta.json").read_text(encoding="utf-8"))
if (cfg.steps!=120_000_000 or cfg.seed!=735 or cfg.envs!=2048
        or cfg.physical_profile!="h40-spring-v1"
        or (meta.get("source_provenance") or {}).get("git_commit")!=expected_git):
    raise RuntimeError("Original 120M provenance is not the validated Spring V1 study")
payload=load_checkpoint(
    orig/"checkpoints/final.pkl",validate=True,
    expected_observation_schema=cfg.observation_schema,
    expected_sensor_contract=training_sensor_contract(cfg))
if (int(payload["meta"]["step"])!=120_061_952
        or payload["meta"].get("robot_model_sha256")!=expected_physics):
    raise RuntimeError("Original 120M PPO state is not the trained H40 actor")
for mode in ("stairs","flat"):
    directory=baseline/mode
    manifest=json.loads((directory/"matrix_summary.json").read_text(encoding="utf-8"))["manifest"]
    for key,value in dict(
            policy_sha256=expected_actor,robot_model_sha256=expected_physics,
            exam_source_sha256=expected_exam,seed=1701,worlds=2048,
            horizon=350,mode=mode,assigned_worlds=2000).items():
        if manifest.get(key)!=value:
            raise RuntimeError(f"Original {mode} exam not comparable: {key}")
    if not (directory/"matrix_worlds.csv").is_file():
        raise RuntimeError(f"Original {mode} per-world evidence missing")
print("PARENT 120M POLICY + OPTIMISER + PHYSICS + EXACT BASELINE EXAMS: VERIFIED",flush=True)
PY

TRAIN_COMMON=(
  --profile sim2real --run-dir "$EMU_TUTOR_ROOT"
  --envs 2048 --seed 735 --device cuda
  --terrain-actor --terrain-step-max-m .08
  --unified-capabilities --terrain-carpet --carpet-prisms
  --carpet-transition-cap-deg 3
  --academy-stream --terrain-lesson coherent-careful
  --foundation-steps 20000000 --foundation-mix main
  --growth-recipe main-mature --teaching-schedule terrain-early
  --force-program gentle --athletic-rehearsal-share .10
  --retention-share .15
  --academy-intro-steps 60000000
  --academy-start-steps 80000000 --academy-ramp-steps 20000000
  --academy-intro-share .10 --academy-max-share .40
  --academy-height-cap-m .15
  --fixed-eval-every-steps 60000000
  # One coherent outcome-based tutor on real first prism lips.
  # No new gait clock, clearance/phase, hardware or motor-cap rewards.
  --science-step-goal-weight .22
  --science-step-velocity-relief .55
  --science-step-approach-m .22
)
CONFIG="$EMU_TUTOR_ROOT/training_config.json"
TARGET=""
if [[ -f "$CONFIG" ]]; then
  TARGET="$("$PY" -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["steps"]))' "$CONFIG")"
  if [[ "$TARGET" != "122000000" && "$TARGET" != "140000000" ]]; then
    echo "Existing tutor run is an incompatible target $TARGET; refusing" >&2
    exit 1
  fi
  INITIAL_RESUME=(--resume auto)
else
  INITIAL_RESUME=(
    --resume "$PARENT" --research-parent-source "$ORIGIN_SHA"
    --research-parent-sha256 "$ORIGIN_CKPT_SHA")
fi

if [[ ! -f "$CONFIG" || "$TARGET" == "122000000" ]]; then
  echo "===== SAME LEARNED BRAIN: 120M → 122M outcome tutor acceptance ====="
  "$PY" -m emu.cli train "${TRAIN_COMMON[@]}" --steps 122000000 \
    "${INITIAL_RESUME[@]}" --check-resume-only
  "$PY" -m emu.cli train "${TRAIN_COMMON[@]}" --steps 122000000 \
    "${INITIAL_RESUME[@]}"
  "$PY" - <<'PY'
import csv,json,math,os
from pathlib import Path
root=Path(os.environ["EMU_TUTOR_ROOT"])
with (root/"metrics.csv").open(newline="",encoding="utf-8") as stream:
    rows=list(csv.DictReader(stream))
if not rows:
    raise RuntimeError("Two-million-step tutor trial has no metrics")
last=rows[-1]
values={k:float(last[k]) for k in
        ("step","fall_rate","ep_len_mean","support","nan_grad_skips_total")}
bad=([k for k,v in values.items() if not math.isfinite(v)]+
     (["did not reach 122M"] if values["step"]<122_000_000 else [])+
     (["fall rate over 85%"] if values["fall_rate"]>=.85 else [])+
     (["episode length under 100"] if values["ep_len_mean"]<100 else [])+
     (["no physical foot support"] if values["support"]<.15 else [])+
     (["nonfinite PPO gradients"] if values["nan_grad_skips_total"]>10 else []))
result=dict(**values,pass_gate=not bad,blockers=bad,
            meaning="Only guards catastrophic 2M continuation; not proof of stair improvement")
(root/"tutor_2m_gate.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
print("122M SAFETY GATE:",json.dumps(result),flush=True)
if bad:
    raise RuntimeError("Stop rather than spend another 18M: "+"; ".join(bad))
PY
else
  # If 140M is already in progress, a gate from the same-source 122M
  # continuation must exist. Never skip the gate on an ambiguous retry.
  "$PY" - <<'PY'
import json,os
from pathlib import Path
p=Path(os.environ["EMU_TUTOR_ROOT"])/"tutor_2m_gate.json"
if not p.is_file() or json.loads(p.read_text())["pass_gate"] is not True:
    raise RuntimeError("Existing 140M job has no successful 122M safety gate")
PY
fi

echo "===== SOURCE-PINNED OUTCOME TUTOR: 122M → 140M ====="
"$PY" -m emu.cli train "${TRAIN_COMMON[@]}" --steps 140000000 --resume auto --check-resume-only
"$PY" -m emu.cli train "${TRAIN_COMMON[@]}" --steps 140000000 --resume auto

"$PY" -m emu.cli export-policy \
  "$EMU_TUTOR_ROOT/checkpoints/final.pkl" \
  "$EMU_TUTOR_ROOT/h40-stair-outcome-tutor-140m.emupolicy"

# Record the exact source and study semantics outside training metrics.
"$PY" - <<'PY'
import json,os,subprocess
from pathlib import Path
from scripts.v5053_matrix_exam import exam_source_fingerprint
run=Path(os.environ["EMU_TUTOR_ROOT"])
doc=dict(
    source_git_commit=subprocess.check_output(
        ["git","rev-parse","HEAD"],text=True).strip(),
    branch="work/emu-stair-outcome-tutor",
    parent_git_commit="a7c6467c59f81c124cd76e3348a4f12baa6f438e",
    parent_checkpoint_sha256="a542722eaf0ddfc27194c68d77c47d907139f49c332f2d81e1b9b9fc40cb2a0b",
    parent_step=120061952,target_steps=140000000,envs=2048,seed=735,
    model_sha256="7085b1732fe6b876b27ef9b485853d0127c70f6e1c1a9b52b57af82c34de6cfd",
    original_frozen_actor_sha256="26095f7f10084c815ee1f7ad24363834cdd017d4341bca9ec1d9f10e4372b16c",
    exam_source_sha256=exam_source_fingerprint(),
    tutor=dict(academy_fixed_destination_progress=.22,
               academy_forward_tracking_relief=.55,
               academy_birth_distance_to_lip_m=.22),
    restrictions="no change to spring, motor torque, obs, gait guidance after 40M, optimiser or original actor",
    evidence="frozen 2k physical stairs and 2k flat controls, then compare same 120M actor source")
path=run/"tutor_source_manifest.json"
if path.exists() and json.loads(path.read_text())!=doc:
    raise RuntimeError("Existing tutor manifest differs from current source")
path.write_text(json.dumps(doc,indent=2)+"\n",encoding="utf-8")
PY

POLICY="$EMU_TUTOR_ROOT/h40-stair-outcome-tutor-140m.emupolicy"
for MODE in stairs flat; do
  DEST="$EMU_TUTOR_ROOT/heldout/$MODE"
  if [[ -f "$DEST/matrix_summary.json" ]]; then
    echo "REUSING EXISTING $MODE RESULT (strict manifest checked by comparison)"
  else
    if [[ -e "$DEST" && -n "$(find "$DEST" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
      echo "Partial $MODE exam exists; inspect before retrying: $DEST" >&2
      exit 1
    fi
    EXTRA=(--trials-per-level 2000)
    if [[ "$MODE" == stairs ]]; then
      EXTRA=(--levels 20 --trials-per-level 50 --max-step-m .08)
    fi
    echo "===== INDEPENDENT FROZEN 2000-world $MODE physical exam ====="
    "$PY" scripts/v5053_matrix_exam.py --policy "$POLICY" \
      --output "$DEST" --mode "$MODE" --worlds 2048 \
      --horizon 350 --seed 1701 --device cuda "${EXTRA[@]}"
  fi
done
"$PY" -m scripts.compare_h40_stair_outcome_tutor --run "$EMU_TUTOR_ROOT"
echo "Original 120M actor and main remain UNTOUCHED."
