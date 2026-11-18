#!/usr/bin/env bash
# Canonical 0->140M Spring V1 H40: ONE cold actor, ONE optimizer.
# All stage rewards are scheduled by the same JAX task packet in emu.training,
# not swapped launchers or inherited policies. No historical actor is read.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODE="${1:-plan}"
[[ "$MODE" == all || "$MODE" == plan ]] || {
  echo "Usage: bash scripts/run_h40_holistic_cold_wsl.sh [plan|all]" >&2
  exit 2
}
# Clean, named Git source; the original seed-735 run remains untouched.
[[ -n "$(git branch --show-current)" ]] || {
  echo "Detached checkout: switch to a named source branch" >&2; exit 1;
}
[[ -z "$(git status --porcelain)" ]] || {
  echo "Dirty Git source; refuse an unverifiable cold-start lineage" >&2; exit 1;
}
PY="$ROOT/.venv/bin/python"
[[ -x "$PY" ]] || { echo "Missing proven EMU .venv/bin/python" >&2; exit 1; }
export EMU_PHYSICAL_PROFILE=h40-spring-v1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export EMU_UNIFIED_CONTINUOUS=cold-growth-approved
export EMU_COLD_SEED="${EMU_COLD_SEED:-735}"
[[ "$EMU_COLD_SEED" =~ ^[0-9]+$ ]] || {
  echo "EMU_COLD_SEED must be a non-negative decimal integer" >&2; exit 1;
}
export EMU_COLD_RUN_DIR="${EMU_COLD_RUN_DIR:-$HOME/emu-data/runs/h40-main-140m-s$EMU_COLD_SEED}"
export EMU_COLD_SOURCE="$(git rev-parse HEAD)"
# Existing proven physical system: same 360-input observation, 150 N/m
# knee support, 2 N.m/rad hip centring, real signed 8-prism collider.
"$PY" - <<'PY'
import hashlib,os
from emu.contracts import ROBOT_XML
from emu.capability_focus import holistic_stair_stage,gait_guidance_strength
wanted="7085b1732fe6b876b27ef9b485853d0127c70f6e1c1a9b52b57af82c34de6cfd"
actual=hashlib.sha256(ROBOT_XML.read_bytes()).hexdigest()
if actual!=wanted:
    raise RuntimeError(f"H40 Spring V1 robot XML drifted: {actual}")
ages=(0,20_000_000,40_000_000,119_999_999,120_000_000,
      139_999_999,140_000_000)
for step in ages:
    stage=holistic_stair_stage(step)
    gait=gait_guidance_strength(step=step,foundation_steps=20_000_000,
                               fade_steps=20_000_000)
    print(f"age={step:,}: {stage.name}, gait={gait:.2f}, "
          f"goal={stage.goal_weight:.2f}, transfer={stage.bilateral_bonus}, "
          f"followthrough={stage.followthrough_bonus}, "
          f"retention={stage.retention_share:.2f}, "
          f"up_cap={stage.uphill_cap_m:.3f}, down_cap={stage.downhill_cap_m:.3f}")
if not (holistic_stair_stage(20_000_000).name=="gait_fade"
        and holistic_stair_stage(40_000_000).name=="outcome_school"
        and holistic_stair_stage(120_000_000).name=="commitment"
        and holistic_stair_stage(140_000_000).name=="transfer_retention"):
    raise RuntimeError("Frozen teacher stage boundaries changed")
# 140M is final. No after-140M transfer-rollout is trained by this recipe.
print("ONE COLD SOURCE:",os.environ["EMU_COLD_SOURCE"])
PY
COMMON=(
  --profile sim2real --run-dir "$EMU_COLD_RUN_DIR"
  --envs 2048 --seed "$EMU_COLD_SEED" --device cuda
  --terrain-actor --terrain-step-max-m .08
  --unified-capabilities --terrain-carpet --carpet-prisms
  --holistic-cold-curriculum
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
)
"$PY" -m emu.cli train "${COMMON[@]}" --steps 140000000 --plan
if [[ "$MODE" == plan ]]; then
  echo "PLAN ONLY. No run created or old policy opened."
  exit 0
fi
# Never collide with a previous experiment or an ambiguous working folder.
CONFIG="$EMU_COLD_RUN_DIR/training_config.json"
if [[ -d "$EMU_COLD_RUN_DIR" && ! -f "$CONFIG" && -n "$(
      find "$EMU_COLD_RUN_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Nonempty unrecognised run folder; refusing: $EMU_COLD_RUN_DIR" >&2
  exit 1
fi
# Short independent GPU physics/PPO checks before allocating overnight time.
# These are not a substitute for the actual 2048-env 140M learner.
if [[ ! -f "$CONFIG" ]]; then
  "$PY" -m emu.cli doctor --require-gpu
  "$PY" scripts/smoke_gpu_carpet.py --device cuda \
       --worlds 128 --frames 2 --prisms --ppo
  "$PY" scripts/smoke_gpu_carpet.py --device cuda \
       --worlds 2048 --frames 2 --prisms --ppo --full-update
fi
# Before escalating from the foundation / gait-fade stages, require
# evidence that the freshly initialized gait is physically viable. Unlike
# training reward, this gate remains meaningful across changing economics.
check_early_viability() {
  if [[ "$EMU_COLD_AGE" != 20 && "$EMU_COLD_AGE" != 40 ]]; then
    return 0
  fi
  "$PY" - <<'PY'
import csv,json,math,os
from pathlib import Path
root=Path(os.environ["EMU_COLD_RUN_DIR"])
age=int(os.environ["EMU_COLD_AGE"])
stage=root/"stages"/f"{age}m"
manifest=json.loads((stage/"stage_manifest.json").read_text(encoding="utf-8"))
actual=int(manifest["actual_step"])
with (root/"metrics.csv").open(newline="",encoding="utf-8") as stream:
    rows=list(csv.DictReader(stream))
matching=[row for row in rows if int(float(row["step"]))==actual]
if not matching:
    raise RuntimeError(f"Missing trained {age}M metrics row for actual step {actual}")
row=matching[-1]
names=("fall_rate","ep_len_mean","support","nan_grad_skips_total")
v={name:float(row[name]) for name in names}
bad=[name+" is not finite" for name,value in v.items()
     if not math.isfinite(value)]
if not bad:
    if not 0.<=v["fall_rate"]<.85: bad.append("fall fraction >=85%")
    if v["ep_len_mean"]<100.: bad.append("episode mean <100 frames")
    if v["support"]<.15: bad.append("foot support below 15%")
    if v["nan_grad_skips_total"]>10.: bad.append("too many skipped PPO gradients")
result=dict(age_million=age,actual_step=actual,source_git_commit=manifest["source_git_commit"],
            training_seed=manifest["training_seed"],metrics=v,
            passed=not bad,blockers=bad,
            caveat="Gross-collapse safety gate; NOT a reliable stair exam or guarantee of competence")
target=stage/"viability_gate.json"
if target.is_file():
    if json.loads(target.read_text(encoding="utf-8"))!=result:
        raise RuntimeError(f"Historical {age}M viability gate differs from current metrics")
else:
    target.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
print("COLD GROWTH VIABILITY:",json.dumps(result),flush=True)
if bad:
    raise RuntimeError("Do not spend another training stage on collapsed learner: "+
                       "; ".join(bad))
PY
}
for AGE in 20 40 120 140; do
  export EMU_COLD_AGE="$AGE"
  TARGET=$((AGE * 1000000))
  SNAP="$EMU_COLD_RUN_DIR/stages/${AGE}m"
  if [[ -f "$SNAP/stage_manifest.json" ]]; then
    # Every stage snapshot is immutable, even when the current run target
    # has grown beyond it. Verify instead of silently re-exporting it.
    "$PY" - <<'PY'
import hashlib,json,os
from pathlib import Path
p=Path(os.environ["EMU_COLD_RUN_DIR"])/"stages"/(os.environ["EMU_COLD_AGE"]+"m")
m=json.loads((p/"stage_manifest.json").read_text(encoding="utf-8"))
if (m["source_git_commit"]!=os.environ["EMU_COLD_SOURCE"] or
        m["target_steps"]!=int(os.environ["EMU_COLD_AGE"])*1_000_000 or
        m["training_seed"]!=int(os.environ["EMU_COLD_SEED"])):

    raise RuntimeError("Stage source/target changed; cannot resume")
for name,field in (("final.pkl","checkpoint_sha256"),
                   ("actor.emupolicy","actor_sha256")):
    if not (p/name).is_file() or hashlib.sha256(
            (p/name).read_bytes()).hexdigest()!=m[field]:
        raise RuntimeError(f"Frozen stage bytes corrupted: {p/name}")
print("REUSING VERIFIED COLD STAGE:",p,flush=True)
PY
    check_early_viability
    continue
  fi
  OLD_TARGET=""
  if [[ -f "$CONFIG" ]]; then
    OLD_TARGET="$("$PY" -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["steps"]))' "$CONFIG")"
    if (( OLD_TARGET > TARGET )); then
      echo "Already advanced beyond ${AGE}M but missing immutable stage snapshot: $SNAP" >&2
      exit 1
    fi
    RESUME=(--resume auto)
  else
    [[ "$AGE" == 20 ]] || {
      echo "Cold beginning missing 20M; no external checkpoint is permitted" >&2; exit 1;
    }
    RESUME=()
  fi
  echo "===== COLD, SAME-SOURCE PPO TO ${AGE}M (no historical parent) ====="
  "$PY" -m emu.cli train "${COMMON[@]}" --steps "$TARGET" "${RESUME[@]}" --check-resume-only
  "$PY" -m emu.cli train "${COMMON[@]}" --steps "$TARGET" "${RESUME[@]}"
  # Validate and freeze one full PPO checkpoint, actor and stage metadata
  # before advancing the target. No other stage ever overwrites this snapshot.
  "$PY" - <<'PY'
import hashlib,json,os,subprocess
from pathlib import Path
from emu.checkpoint_io import load_checkpoint
from emu.training_config import TrainingConfig,training_sensor_contract
run=Path(os.environ["EMU_COLD_RUN_DIR"])
age=int(os.environ["EMU_COLD_AGE"])
source=os.environ["EMU_COLD_SOURCE"]
origin=run/"checkpoints/final.pkl"
cfg=TrainingConfig.load(run/"training_config.json")
if not cfg.holistic_cold_curriculum or cfg.steps!=age*1_000_000:
    raise RuntimeError("Not a completed holistic stage target")
payload=load_checkpoint(origin,validate=True,
    expected_observation_schema=cfg.observation_schema,
    expected_sensor_contract=training_sensor_contract(cfg))
step=int(payload["meta"]["step"])
if not age*1_000_000<=step<age*1_000_000+cfg.envs*32:
    raise RuntimeError(f"Stage {age}M has wrong checkpoint age: {step}")
if (payload["meta"].get("source_provenance") or {}).get("git_commit")!=source:
    raise RuntimeError("Stage checkpoint is from a different source")
out=run/"stages"/f"{age}m"
out.mkdir(parents=True,exist_ok=True)
def frozen_copy(src,dest):
    if not dest.exists():
        temporary=dest.with_name("."+dest.name+".copying")
        if temporary.exists(): raise RuntimeError("Interrupted frozen copy: "+str(temporary))
        import shutil
        shutil.copy2(src,temporary)
        if hashlib.sha256(src.read_bytes()).digest()!=hashlib.sha256(
                temporary.read_bytes()).digest():
            raise RuntimeError("Cold stage checkpoint copied incorrectly")
        os.replace(temporary,dest)
        dest.chmod(0o444)
    if hashlib.sha256(src.read_bytes()).digest()!=hashlib.sha256(
            dest.read_bytes()).digest():
        raise RuntimeError("An existing frozen stage disagrees with final.pkl")
frozen_copy(origin,out/"final.pkl")
export=out/"actor.emupolicy"
if not export.exists():
    subprocess.run([str(Path.cwd()/".venv/bin/python"),"-m","emu.cli",
                    "export-policy",str(origin),str(export)],check=True)
manifest=dict(source_git_commit=source,target_steps=age*1_000_000,
              training_seed=int(os.environ["EMU_COLD_SEED"]),
              actual_step=step,
              parent_checkpoint=None if age==20 else "same learner previous stage",
              checkpoint_sha256=hashlib.sha256((out/"final.pkl").read_bytes()).hexdigest(),
              actor_sha256=hashlib.sha256(export.read_bytes()).hexdigest(),
              physical_model_sha256=payload["meta"]["robot_model_sha256"],
              random_initialization=age==20,
              no_historical_model_loaded=True,
              curriculum_stage=(
                  "foundation" if age<=20 else "gait_fade" if age<=40
                  else "outcomes" if age<=120 else "commitment"))
path=out/"stage_manifest.json"
if path.exists() and json.loads(path.read_text(encoding="utf-8"))!=manifest:
    raise RuntimeError("Frozen cold stage manifest conflict")
path.write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
print("FROZEN SOURCE-PINNED COLD STAGE:",out,flush=True)
PY
  check_early_viability
done

# Independently test the exact physical terrain (not Academy reward flags)
# at 120M and 140M. Every actor is from THIS cold run.
for AGE in 120 140; do
  for MODE in stairs flat; do
    ACTOR="$EMU_COLD_RUN_DIR/stages/${AGE}m/actor.emupolicy"
    OUT="$EMU_COLD_RUN_DIR/heldout/${AGE}m/$MODE"
    if [[ -f "$OUT/matrix_summary.json" && -f "$OUT/matrix_worlds.csv" ]]; then
      echo "VERIFYING existing $AGE $MODE with final source-authenticated comparison"
      continue
    fi
    if [[ -d "$OUT" && -n "$(find "$OUT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
      echo "Partial physical exam, inspect before continuing: $OUT" >&2
      exit 1
    fi
    EXTRA=(--trials-per-level 2000)
    if [[ "$MODE" == stairs ]]; then
      EXTRA=(--levels 20 --trials-per-level 50 --max-step-m .08)
    fi
    echo "===== COLD-TRAINED ${AGE}M: 2000 independent $MODE trials ====="
    "$PY" scripts/v5053_matrix_exam.py --policy "$ACTOR" \
      --output "$OUT" --mode "$MODE" --worlds 2048 \
      --horizon 350 --seed 1701 --device cuda "${EXTRA[@]}"
  done
done
"$PY" -m scripts.summarize_h40_holistic_cold --run "$EMU_COLD_RUN_DIR"
echo "Cold-start training and held-out evidence saved; NO promotion to main."
