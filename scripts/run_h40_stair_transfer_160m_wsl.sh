#!/usr/bin/env bash
# One source-pinned physical-outcome experiment: 140M explorer -> 160M transfer tutor.
# Immutable ancestors; no cold reset, no motor/spring/actor-schema changes.
# 142M safety gate -> 160M -> exact same frozen 2k stairs and 2k flat controls.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[[ "$(git branch --show-current)" == "work/emu-stair-transfer-160m" ]] || {
  echo "Wrong branch: expected work/emu-stair-transfer-160m" >&2; exit 1;
}
[[ -z "$(git status --porcelain)" ]] || {
  echo "Dirty Git source; refusing altered learning lineage" >&2; exit 1;
}
PY="$ROOT/.venv/bin/python"
[[ -x "$PY" ]] || { echo "Use the existing EMU .venv/bin/python" >&2; exit 1; }
export EMU_PHYSICAL_PROFILE=h40-spring-v1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export EMU_UNIFIED_CONTINUOUS=cold-growth-approved
export EMU_TRANSFER_PARENT="$HOME/emu-data/runs/h40-stair-outcome-tutor-140m-s735"
export EMU_TRANSFER_REFERENCE="$HOME/emu-data/actors/h40-stair-outcome-tutor-140m-reference"
export EMU_TRANSFER_ROOT="$HOME/emu-data/runs/h40-stair-transfer-tutor-160m-s735"
PARENT_CKPT="$EMU_TRANSFER_PARENT/checkpoints/final.pkl"
PARENT_SOURCE="17f6fb17cc699342f2ff394e1c6eb8d237675023"

# Exact historical actor identity is from the uploaded real 140M exam ZIP.
# The full source checkpoint SHA is local-only: record and pin its ACTUAL bytes.
# Do not guess that the browser actor SHA equals the full PPO optimiser SHA.
"$PY" - <<'PY'
import hashlib,json,os,subprocess
from pathlib import Path
from emu.contracts import ROBOT_XML
from emu.training_config import TrainingConfig,training_sensor_contract
from emu.checkpoint_io import load_checkpoint
from emu.policy_export import build_bundle
from scripts.v5053_matrix_exam import exam_source_fingerprint
from scripts.examine_h40_spring_v1_mechanics import immutable_copy

origin=Path(os.environ["EMU_TRANSFER_PARENT"])
reference=Path(os.environ["EMU_TRANSFER_REFERENCE"])
child=Path(os.environ["EMU_TRANSFER_ROOT"])
actor_name="h40-stair-outcome-tutor-140m.emupolicy"
expected_actor="0324fa18d5e624c7a38a7085050cc2abc964f30e8eb453d00d5b280e6097e039"
expected_source="17f6fb17cc699342f2ff394e1c6eb8d237675023"
expected_model="7085b1732fe6b876b27ef9b485853d0127c70f6e1c1a9b52b57af82c34de6cfd"
expected_exam="99b690b3c4aa0758157a9854e6442f5abc5572d1dd9f53c958bd2e4299ac4953"
def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for buf in iter(lambda:f.read(1024*1024),b""): h.update(buf)
    return h.hexdigest()
if sha(ROBOT_XML)!=expected_model:
    raise RuntimeError("Spring V1 physics changed; refuse comparison")
if exam_source_fingerprint()!=expected_exam:
    raise RuntimeError("Frozen exam source changed; require exact same source fingerprint")
for p in (origin/"training_config.json",origin/"run_meta.json",
          origin/"checkpoints/final.pkl",origin/actor_name):
    if not p.is_file():
        raise RuntimeError(f"Missing required 140M parent artifact: {p}")
if sha(origin/actor_name)!=expected_actor:
    raise RuntimeError("140M parent policy bytes differ from uploaded held-out exam")
cfg=TrainingConfig.load(origin/"training_config.json")
meta=json.loads((origin/"run_meta.json").read_text(encoding="utf-8"))
if (cfg.physical_profile!="h40-spring-v1" or cfg.seed!=735 or
        cfg.envs!=2048 or cfg.steps!=140_000_000 or
        (meta.get("source_provenance") or {}).get("git_commit")!=expected_source or
        cfg.science_step_goal_weight!=.22 or cfg.science_step_velocity_relief!=.55 or
        cfg.science_step_approach_m!=.22 or cfg.retention_share!=.15):
    raise RuntimeError("Original 140M curriculum/provenance differs")
ckpt=origin/"checkpoints/final.pkl"
payload=load_checkpoint(ckpt,validate=True,
    expected_observation_schema=cfg.observation_schema,
    expected_sensor_contract=training_sensor_contract(cfg))
if (int(payload["meta"]["step"])!=140_050_432 or
        payload["meta"].get("robot_model_sha256")!=expected_model or
        (payload["meta"].get("source_provenance") or {}).get("git_commit")!=expected_source):
    raise RuntimeError("Not the complete 140M PPO and optimiser lineage")
# Bind the FULL optimiser checkpoint's actual actor weights to the
# user's exported 140M browser policy, not merely matching run metadata.
# The policy source hashes its NN tensors before export. No bytes are altered.
expected_nn_sha="c1e205f1c9d401649750fc8a67c661cf6a0d46a27b2ce66a574d89a13d6a629a"
checkpoint_actor=build_bundle(ckpt)["source"]
saved_actor=json.loads((origin/actor_name).read_text(encoding="utf-8"))["source"]
if (checkpoint_actor.get("actor_sha256")!=expected_nn_sha or
        checkpoint_actor.get("actor_sha256")!=saved_actor.get("actor_sha256") or
        checkpoint_actor.get("step")!=140_050_432):
    raise RuntimeError("Full 140M PPO checkpoint policy differs from exported 140M actor")
for name in ("stairs","flat"):
    directory=origin/"heldout"/name
    m=json.loads((directory/"matrix_summary.json").read_text(encoding="utf-8"))["manifest"]
    wanted=dict(policy_sha256=expected_actor,
                robot_model_sha256=expected_model,exam_source_sha256=expected_exam,
                seed=1701,worlds=2048,assigned_worlds=2000,
                horizon=350,mode=name,zero_optimizer_updates=True,first_life_only=True)
    for k,v in wanted.items():
        if m.get(k)!=v:
            raise RuntimeError(f"140M {name} frozen exam differs at {k}")
    if name=="stairs" and (m.get("max_step_m")!=.08 or m.get("levels")!=20):
        raise RuntimeError("Original 140M stairs must be the same exact 4–80mm challenge")
    if not (directory/"matrix_worlds.csv").is_file():
        raise RuntimeError(f"Missing original 140M per-trial {name} result")
reference.mkdir(parents=True,exist_ok=True)
immutable_copy(ckpt,reference/"final.pkl")
immutable_copy(origin/actor_name,reference/actor_name)
manifest=dict(
    parent_run=str(origin),
    frozen_parent_actor_sha256=expected_actor,
    frozen_parent_checkpoint_sha256=sha(reference/"final.pkl"),
    parent_steps=140050432,parent_training_source=expected_source,
    robot_model_sha256=expected_model,exam_source_sha256=expected_exam,
    restricted_changes="reward weights for authenticated bilateral support and followthrough, gentler signed progress, slightly more retention",
    original_run_immutable=True)
dest=reference/"reference_manifest.json"
if dest.is_file():
    if json.loads(dest.read_text(encoding="utf-8"))!=manifest:
        raise RuntimeError("Prior frozen 140M reference disagrees; refusing overwrite")
else:
    dest.write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
print("VERIFIED IMMUTABLE 140M ACTOR + PPO OPTIMISER + MATCHED PHYSICAL EXAMS",flush=True)
print("FROZEN PARENT PPO SHA256:",manifest["frozen_parent_checkpoint_sha256"],flush=True)
PY
PARENT_SHA="$(sha256sum "$PARENT_CKPT" | cut -d' ' -f1)"
[[ "$PARENT_SHA" =~ ^[0-9a-f]{64}$ ]] || { echo "Invalid PPO parent digest" >&2; exit 1; }
CONFIG="$EMU_TRANSFER_ROOT/training_config.json"
TARGET=""
if [[ -f "$CONFIG" ]]; then
  TARGET="$("$PY" -c 'import json,sys;print(int(json.load(open(sys.argv[1]))["steps"]))' "$CONFIG")"
  if [[ "$TARGET" != 142000000 && "$TARGET" != 160000000 ]]; then
    echo "Unexpected existing tutor target $TARGET; refuse overwrite" >&2; exit 1;
  fi
  INITIAL=(--resume auto)
else
  INITIAL=(--resume "$PARENT_CKPT" --research-parent-source "$PARENT_SOURCE"
           --research-parent-sha256 "$PARENT_SHA")
fi

# ONLY these deltas versus 140M:
#  - reduce Academy approach/charge potential .22 -> .12
#  - authentic 5-frame bilateral new upper tile support +6 one-time
#  - authentic post-transfer 50cm forward+heading follow-through +7 one-time
#  - preserve easy/command rehearsal by increasing retention .15 -> .20
# Both feet and follow-through events are already defined by physical MJX
# contacts; no new gait/phase processing or reward/event implementation.
TRAIN=(
  --profile sim2real --run-dir "$EMU_TRANSFER_ROOT"
  --envs 2048 --seed 735 --device cuda
  --terrain-actor --terrain-step-max-m .08
  --unified-capabilities --terrain-carpet --carpet-prisms
  --carpet-transition-cap-deg 3
  --academy-stream --terrain-lesson coherent-careful
  --foundation-steps 20000000 --foundation-mix main
  --growth-recipe main-mature --teaching-schedule terrain-early
  --force-program gentle --athletic-rehearsal-share .10
  --retention-share .20
  --academy-intro-steps 60000000 --academy-start-steps 80000000
  --academy-ramp-steps 20000000
  --academy-intro-share .10 --academy-max-share .40
  --academy-height-cap-m .15
  --fixed-eval-every-steps 60000000
  --science-step-goal-weight .12
  --science-step-velocity-relief .55
  --science-step-approach-m .22
  --science-bilateral-support-bonus 6
  --science-followthrough-bonus 7
)
if [[ ! -f "$CONFIG" || "$TARGET" == 142000000 ]]; then
  echo "===== 140M→142M: genuine two-foot transfer training smoke ====="
  "$PY" -m emu.cli train "${TRAIN[@]}" --steps 142000000 "${INITIAL[@]}" --check-resume-only
  "$PY" -m emu.cli train "${TRAIN[@]}" --steps 142000000 "${INITIAL[@]}"
  "$PY" - <<'PY'
import csv,json,math,os
from pathlib import Path
run=Path(os.environ["EMU_TRANSFER_ROOT"])
with (run/"metrics.csv").open(newline="",encoding="utf-8") as handle:
    rows=list(csv.DictReader(handle))
if not rows: raise RuntimeError("142M transfer tutor produced no rollout evidence")
last=rows[-1]
v={k:float(last[k]) for k in
   ("step","fall_rate","ep_len_mean","support","nan_grad_skips_total")}
bad=[k for k,x in v.items() if not math.isfinite(x)]
if v["step"]<142000000: bad.append("short of 142M")
if not 0<=v["fall_rate"]<.85: bad.append("gross fall collapse")
if v["ep_len_mean"]<100: bad.append("gross episode duration collapse")
if v["support"]<.15: bad.append("no foot support")
if v["nan_grad_skips_total"]>10: bad.append("PPO nonfinite gradient updates")
data=dict(**v,pass_gate=not bad,blockers=bad,
          caveat="A smoke viability gate only, not an efficacy or skill retention verdict")
(run/"transfer_2m_gate.json").write_text(
    json.dumps(data,indent=2)+"\n",encoding="utf-8")
print("142M SAFETY GATE:",json.dumps(data),flush=True)
if bad: raise RuntimeError("Refusing 18M further steps: "+"; ".join(bad))
PY
else
  "$PY" - <<'PY'
import json,os
from pathlib import Path
p=Path(os.environ["EMU_TRANSFER_ROOT"])/"transfer_2m_gate.json"
if not p.is_file() or json.loads(p.read_text())["pass_gate"] is not True:
    raise RuntimeError("Existing 160M experiment has no passed 142M gate")
PY
fi
echo "===== Same-source 142M→160M transfer/follow-through continuation ====="
"$PY" -m emu.cli train "${TRAIN[@]}" --steps 160000000 --resume auto --check-resume-only
"$PY" -m emu.cli train "${TRAIN[@]}" --steps 160000000 --resume auto

POLICY="$EMU_TRANSFER_ROOT/h40-stair-transfer-tutor-160m.emupolicy"
"$PY" -m emu.cli export-policy "$EMU_TRANSFER_ROOT/checkpoints/final.pkl" "$POLICY"
"$PY" - <<'PY'
import json,os,subprocess
from pathlib import Path
from scripts.v5053_matrix_exam import exam_source_fingerprint
root=Path(os.environ["EMU_TRANSFER_ROOT"])
ref=Path(os.environ["EMU_TRANSFER_REFERENCE"])/"reference_manifest.json"
ancestor=json.loads(ref.read_text(encoding="utf-8"))
source=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
manifest=dict(
    git_source=source,branch="work/emu-stair-transfer-160m",
    parent=ancestor,seed=735,envs=2048,target=160000000,
    intervention=dict(
        signed_academy_destination_progress=.12,
        bilateral_5frame_real_support_bonus=6,
        real_post_transfer_50cm_heading_followthrough_bonus=7,
        academy_speed_tracking_relief=.55,
        academy_spawn_distance_m=.22,
        retention_share=.20),
    identical_physics_exam_source=exam_source_fingerprint(),
    restrictions="no alteration of 360 actor, springs, servo caps, gait guidance, PPO, 140M reference or main",
    accountability="only heldout 4-80mm first-life true-prism exams determine improvement")
p=root/"transfer_source_manifest.json"
if p.exists() and json.loads(p.read_text(encoding="utf-8"))!=manifest:
    raise RuntimeError("Different source ancestry/teacher than previous transfer export")
p.write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
PY

# Never overwrite or silently merge partial, source-incompatible exams.
for MODE in stairs flat; do
  OUT="$EMU_TRANSFER_ROOT/heldout/$MODE"
  if [[ -f "$OUT/matrix_summary.json" && -f "$OUT/matrix_worlds.csv" ]]; then
    echo "FOUND $MODE RESULT; strict provenance verified by final comparator"
    continue
  fi
  if [[ -d "$OUT" && -n "$(find "$OUT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Incomplete $MODE frozen exam; inspect before retrying: $OUT" >&2
    exit 1
  fi
  EXTRA=(--trials-per-level 2000)
  if [[ "$MODE" == stairs ]]; then
    EXTRA=(--levels 20 --trials-per-level 50 --max-step-m .08)
  fi
  echo "===== FROZEN 2000-WORLD, FIRST-LIFE $MODE COMPARISON ====="
  "$PY" scripts/v5053_matrix_exam.py --policy "$POLICY" \
    --output "$OUT" --mode "$MODE" --worlds 2048 \
    --horizon 350 --seed 1701 --device cuda "${EXTRA[@]}"
done
"$PY" -m scripts.compare_h40_stair_transfer_tutor --run "$EMU_TRANSFER_ROOT"
echo "EXPERIMENT COMPLETE. Original 120M+140M saved references and main UNTOUCHED."
