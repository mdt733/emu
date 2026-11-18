#!/usr/bin/env bash
# H40 140M -> 180M SINGLE continuous PPO experiment, never on main.
# 140-160M commitment, 160-180M transfer/retention.
# Usage: bash scripts/run_h40_commitment_180m_wsl.sh [plan|train|exam|all]
set -euo pipefail
MODE="${1:-plan}"
case "$MODE" in plan|train|exam|all) ;; *) echo "Use plan|train|exam|all" >&2; exit 2;; esac
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[[ "$(git branch --show-current)" == "work/emu-commitment-180m" ]] || {
  echo "Wrong branch; use work/emu-commitment-180m, NEVER main." >&2; exit 1;
}
[[ -z "$(git status --porcelain)" ]] || {
  echo "Dirty source checkout; refusing unverifiable experiment." >&2; exit 1;
}
PY="$ROOT/.venv/bin/python"
[[ -x "$PY" ]] || { echo "Missing $PY" >&2; exit 1; }
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export EMU_PHYSICAL_PROFILE=h40-spring-v1
export EMU_UNIFIED_CONTINUOUS=cold-growth-approved
export SOURCE_SHA=22266821bdbfc82a529ef708e5171c82c6c9f29c
export SOURCE_RUN="${EMU_COMMITMENT_SOURCE_RUN:-$HOME/emu-data/runs/h40-holistic-cold-s735}"
export EXP_ROOT="${EMU_COMMITMENT_ROOT:-$HOME/emu-data/runs/h40-commitment-140to180-s735}"
export PARENT_DIR="$EXP_ROOT/parent-140m"
export RUN_DIR="$EXP_ROOT/experiment"
COMMON=(
  --profile sim2real --run-dir "$RUN_DIR"
  --envs 2048 --seed 735 --device cuda
  --terrain-actor --terrain-step-max-m .08
  --unified-capabilities --terrain-carpet --carpet-prisms
  --holistic-cold-curriculum --carpet-transition-cap-deg 3
  --academy-stream --terrain-lesson coherent-careful
  --foundation-steps 20000000 --foundation-mix main
  --growth-recipe main-mature --teaching-schedule terrain-early
  --force-program gentle --athletic-rehearsal-share .10
  --retention-share .15
  --academy-intro-steps 60000000 --academy-start-steps 80000000
  --academy-ramp-steps 20000000 --academy-intro-share .10
  --academy-max-share .40 --academy-height-cap-m .15
  --fixed-eval-every-steps 60000000
)
"$PY" -m emu.cli train "${COMMON[@]}" --steps 180000000 --plan
[[ "$MODE" != plan ]] || exit 0

if [[ "$MODE" == train || "$MODE" == all ]]; then
  # Stage the original immutable 140M FULL PPO learner under checkpoints/.
  "$PY" - <<'PY'
import hashlib,json,os,shutil
from pathlib import Path
src=Path(os.environ["SOURCE_RUN"])
dst=Path(os.environ["PARENT_DIR"])
manifest=json.loads((src/"stages/140m/stage_manifest.json").read_text())
original=src/"stages/140m/final.pkl"
def digest(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()
if (manifest["source_git_commit"]!=os.environ["SOURCE_SHA"] or
    manifest["target_steps"]!=140_000_000 or
    manifest["training_seed"]!=735 or
    not 140_000_000<=int(manifest["actual_step"])<140_065_536 or
    digest(original)!=manifest["checkpoint_sha256"]):
    raise RuntimeError("Source 140M checkpoint identity failed")
for source,target in ((original,dst/"checkpoints/final.pkl"),
                      (src/"training_config.json",dst/"training_config.json")):
    target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():
        temp=target.with_name("."+target.name+".copying")
        if temp.exists(): raise RuntimeError("Interrupted copy "+str(temp))
        shutil.copy2(source,temp)
        os.replace(temp,target)
        target.chmod(0o444)
    if digest(source)!=digest(target):
        raise RuntimeError("Staged 140M parent differs from source")
print("Verified 140M full optimizer and original source manifest.")
PY
  PARENT="$PARENT_DIR/checkpoints/final.pkl"
  PARENT_HASH="$(sha256sum "$PARENT" | cut -d' ' -f1)"
  if [[ -f "$RUN_DIR/training_config.json" && -f "$RUN_DIR/run_meta.json" ]]; then
    RESUME=(--resume auto)
  else
    RESUME=(--resume "$PARENT" --research-parent-source "$SOURCE_SHA"
            --research-parent-sha256 "$PARENT_HASH")
    # Generic CLI contract tests assert the default h40 profile; the separate
    # holistic experiment itself still runs with h40-spring-v1.
    EMU_PHYSICAL_PROFILE=h40 "$PY" -m pytest -q tests/test_core_run.py tests/test_core_stair_teaching.py
    "$PY" scripts/smoke_gpu_carpet.py --device cuda --worlds 128 --frames 2 --prisms --ppo
    "$PY" scripts/smoke_gpu_carpet.py --device cuda --worlds 2048 --frames 2 --prisms --ppo --full-update
  fi
  "$PY" -m emu.cli train "${COMMON[@]}" --steps 180000000 "${RESUME[@]}" --check-resume-only
  "$PY" -m emu.cli train "${COMMON[@]}" --steps 180000000 "${RESUME[@]}"
  echo "180M learner complete; snapshots and pre-transfer 160M checkpoint retained."
fi
[[ "$MODE" == exam || "$MODE" == all ]] || exit 0

# Verify original control evidence and select checkpoints; no retraining.
"$PY" - <<'PY'
import json,os,subprocess
from pathlib import Path
from scripts.v5053_matrix_exam import exam_source_fingerprint
from emu.checkpoint_io import load_checkpoint
from emu.training_config import TrainingConfig,training_sensor_contract
source=Path(os.environ["SOURCE_RUN"])
root=Path(os.environ["EXP_ROOT"])
run=Path(os.environ["RUN_DIR"])
sha=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
meta=json.loads((run/"run_meta.json").read_text())
if (meta.get("source_provenance") or {}).get("git_commit")!=sha:
    raise RuntimeError("Exam source differs from training source; checkout the exact commit")
fingerprint=exam_source_fingerprint()
for age in (140,160):
    for mode in ("stairs","flat"):
        original=json.loads((source/f"heldout/{age}m/{mode}/matrix_summary.json").read_text())
        if original["manifest"]["exam_source_sha256"]!=fingerprint:
            raise RuntimeError(f"Different frozen examiner for original {age}M {mode}")
ck=run/"checkpoints"
cfg=TrainingConfig.load(run/"training_config.json")
def read(path):
    return load_checkpoint(path,
        expected_observation_schema=cfg.observation_schema,
        expected_sensor_contract=training_sensor_contract(cfg))
final=int(read(ck/"final.pkl")["meta"]["step"])
boundary=int(read(ck/"milestone_160m.pkl")["meta"]["step"])
if not 180_000_000<=final<180_065_536:
    raise RuntimeError("No completed 180M full PPO learner")
if not 160_000_000<=boundary<160_065_536:
    raise RuntimeError("No exact pre-transfer 160M boundary checkpoint")
snapshots=sorted((int(p.stem.split("_")[-1]),p)
                 for p in ck.glob("snapshot_step_*.pkl"))
if len(snapshots)<18:
    raise RuntimeError(f"Only {len(snapshots)} saved 2M snapshots; cannot build trajectory")
choices=[]
for age in (145,150,155,160,165,170,175,180):
    if age==160: step,path=boundary,ck/"milestone_160m.pkl"
    elif age==180: step,path=final,ck/"final.pkl"
    else:
        step,path=min(snapshots,key=lambda row:abs(row[0]-age*1_000_000))
        if abs(step-age*1_000_000)>1_200_000:
            raise RuntimeError(f"No nearby snapshot at {age}M")
    choices.append(dict(age=age,actual_step=step,checkpoint=str(path)))
record=dict(source_git_commit=sha,parent_git_commit=os.environ["SOURCE_SHA"],
            exam_source_sha256=fingerprint,graduation_steps=160_000_000,
            stages=choices)
manifest=root/"timeline_sources.json"
if manifest.exists() and json.loads(manifest.read_text())!=record:
    raise RuntimeError("Timeline selection changed; refuse to rewrite study")
root.mkdir(parents=True,exist_ok=True)
manifest.write_text(json.dumps(record,indent=2)+"\n")
(root/"selected_checkpoints.tsv").write_text(
    "".join(f"{r['age']}\t{r['checkpoint']}\n" for r in choices))
print("Chosen checkpoint ages:",[(r["age"],r["actual_step"]) for r in choices])
PY

run_exam() {
  local tag="$1" actor="$2" mode="$3" out="$4"
  if [[ -f "$out/matrix_summary.json" && -f "$out/matrix_worlds.csv" ]]; then
    echo "Using existing $tag $mode examination"
    return
  fi
  [[ ! -e "$out" ]] || { echo "Partial exam at $out; inspect before retrying" >&2; exit 1; }
  local opts=()
  case "$mode" in
    stairs) opts=(--levels 20 --trials-per-level 50 --max-step-m .08) ;;
    flat) opts=(--trials-per-level 2000) ;;
    balanced) opts=(--levels 20 --trials-per-level 10 --max-step-m .08) ;;
  esac
  "$PY" scripts/v5053_matrix_exam.py --policy "$actor" --output "$out" \
    --mode "$mode" --worlds 2048 --horizon 350 --seed 1701 --device cuda "${opts[@]}"
}
mkdir -p "$EXP_ROOT/actors"
while IFS=$'\t' read -r AGE CHECKPOINT; do
  ACTOR="$EXP_ROOT/actors/commitment-${AGE}m.emupolicy"
  if [[ ! -f "$ACTOR" ]]; then
    "$PY" -m emu.cli export-policy "$CHECKPOINT" "$ACTOR"
  fi
  for TASK in stairs flat; do
    run_exam "${AGE}m" "$ACTOR" "$TASK" "$EXP_ROOT/heldout/${AGE}m/$TASK"
  done
  if [[ "$AGE" == 160 || "$AGE" == 180 ]]; then
    run_exam "${AGE}m" "$ACTOR" balanced "$EXP_ROOT/heldout/${AGE}m/balanced"
  fi
done < "$EXP_ROOT/selected_checkpoints.tsv"
for AGE in 140 160; do
  run_exam "original-${AGE}m" "$SOURCE_RUN/stages/${AGE}m/actor.emupolicy" \
    balanced "$EXP_ROOT/heldout/original-${AGE}m/balanced"
done

"$PY" - <<'PY'
import csv,os,zipfile
from pathlib import Path
source=Path(os.environ["SOURCE_RUN"])
root=Path(os.environ["EXP_ROOT"])
ages=[145,150,155,160,165,170,175,180]
def read(name,mode):
    folder=(source/f"heldout/{int(name.split('-')[1])}m/{mode}"
            if name.startswith("original-") else root/f"heldout/{name}m/{mode}")
    with (folder/"matrix_worlds.csv").open(newline="") as f:
        return list(csv.DictReader(f))
def pct(rows,field):
    if not rows: raise RuntimeError("Empty independent physical cohort")
    return round(100*sum(r[field].lower()=="true" for r in rows)/len(rows),2)
output=[]
for stage in ["original-140","original-160"]+[str(x) for x in ages]:
    stairs,flat=read(stage,"stairs"),read(stage,"flat")
    def band(direction,lo,hi):
        return [r for r in stairs
                if r["kind"]==("step_up" if direction=="up" else "step_down")
                and lo-1e-6<=abs(float(r["exact_step_m"]))<=hi+1e-6]
    medium=band("up",.044,.060)
    high=band("up",.064,.080)
    downhill=band("down",.044,.080)
    output.append(dict(stage=stage,
        flat_control_pct=pct(flat,"traversed_5frame"),
        up_44_60_pct=pct(medium,"traversed_upright_5frame"),
        up_64_80_pct=pct(high,"traversed_upright_5frame"),
        down_44_80_pct=pct(downhill,"traversed_upright_5frame"),
        medium_both_feet_pct=pct(medium,"both_feet_top_five"),
        medium_survival_pct=pct(medium,"survived")))
with (root/"comparison.csv").open("w",newline="") as f:
    writer=csv.DictWriter(f,fieldnames=list(output[0]))
    writer.writeheader();writer.writerows(output)
lines=["# H40 continuous commitment experiment","",
    "140-160M: unchanged approach/commitment; 160-180M: transfer/follow-through.",
    "All rows: original first-life physics exams, 2048 worlds, 350 frames, seed 1701.",
    "145-175M samples are nearest ~2M snapshots; 160M is PRE-transfer.",
    "Original 160M vs extended 160M is the primary comparison.","",
    "| Actor | Flat control | Uphill 44-60mm | Uphill 64-80mm | Downhill 44-80mm |",
    "|---|---:|---:|---:|---:|"]
for r in output:
    lines.append(f"| {r['stage']} | {r['flat_control_pct']:.1f}% | "
        f"{r['up_44_60_pct']:.1f}% | {r['up_64_80_pct']:.1f}% | "
        f"{r['down_44_80_pct']:.1f}% |")
lines+=["","Balanced command, slope and step exams at original 140M/160M and",
        "experimental 160M/180M are included under heldout/*/balanced/.",
        "180M is exploratory: the control has no matched 180M continuation.",
        "Never promote automatically; repeat promising findings with other seeds.",""]
(root/"comparison.md").write_text("\n".join(lines))
zip_path=root/"commitment_180m_results.zip"
with zipfile.ZipFile(zip_path,"w",zipfile.ZIP_DEFLATED) as z:
    for name in ("comparison.csv","comparison.md","timeline_sources.json"):
        z.write(root/name,arcname=name)
    for name in ("metrics.csv","training_config.json","run_meta.json"):
        z.write(root/"experiment"/name,arcname="experiment/"+name)
    for p in sorted((root/"heldout").rglob("matrix_*")):
        if p.is_file(): z.write(p,arcname=str(p.relative_to(root)))
    for p in sorted((root/"actors").glob("*.emupolicy")):
        z.write(p,arcname=str(p.relative_to(root)))
    for age in (140,160):
        for mode in ("stairs","flat"):
            for p in (source/f"heldout/{age}m/{mode}").glob("matrix_*"):
                if p.is_file(): z.write(p,arcname=f"original/{age}m/{mode}/{p.name}")
print("\n".join(lines))
print("RESULTS ZIP:",zip_path)
PY
echo "Physical examination finished. Main and original 140M learner untouched."
