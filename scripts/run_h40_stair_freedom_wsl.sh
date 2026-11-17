#!/usr/bin/env bash
# One controlled 140M -> 160M research fork. Original cold-run checkpoints untouched.
# Usage: bash scripts/run_h40_stair_freedom_wsl.sh [plan|pilot|run|exam|all]
set -euo pipefail
MODE="${1:-pilot}"
case "$MODE" in plan|pilot|run|exam|all) ;; *) echo "Use plan|pilot|run|exam|all" >&2; exit 2;; esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
BRANCH="$(git branch --show-current)"
[[ "$BRANCH" == "work/emu-stair-freedom-140to160" ]] || {
  echo "Switch to the single stair-freedom experiment branch (currently $BRANCH)" >&2; exit 1;
}
[[ -z "$(git status --porcelain)" ]] || {
  echo "Dirty Git worktree; experiment must be source-pinned" >&2; exit 1;
}
PY="$ROOT/.venv/bin/python"
[[ -x "$PY" ]] || { echo "Missing EMU .venv/bin/python" >&2; exit 1; }
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export EMU_PHYSICAL_PROFILE=h40-spring-v1
export EMU_UNIFIED_CONTINUOUS=cold-growth-approved

SOURCE_SHA=22266821bdbfc82a529ef708e5171c82c6c9f29c
SOURCE_RUN="${EMU_FREEDOM_SOURCE_RUN:-$HOME/emu-data/runs/h40-holistic-cold-s735}"
# Pilot is an independent 5M PPO fork with the same parent, not a silent
# change to the planned full 20M comparison. It cannot overwrite either.
if [[ "$MODE" == pilot ]]; then
  EXP_ROOT="${EMU_FREEDOM_ROOT:-$HOME/emu-data/runs/h40-stair-freedom-pilot-145m-s735}"
  TRAIN_TARGET=145000000
  AGE=145
  CANDIDATE=pilot145M
else
  EXP_ROOT="${EMU_FREEDOM_ROOT:-$HOME/emu-data/runs/h40-stair-freedom-140to160-s735}"
  TRAIN_TARGET=160000000
  AGE=160
  CANDIDATE=freedom160M
fi
PARENT_DIR="$EXP_ROOT/parent-140m"
RUN_DIR="$EXP_ROOT/experiment"
export SOURCE_RUN EXP_ROOT PARENT_DIR RUN_DIR SOURCE_SHA CANDIDATE AGE TRAIN_TARGET

# Stage the original immutable 140M learner under checkpoints/ so the
# existing strict cross-source fork verifier can authenticate it.
"$PY" - <<'PY'
import hashlib,json,os,shutil
from pathlib import Path
src=Path(os.environ["SOURCE_RUN"])
dst=Path(os.environ["PARENT_DIR"])
manifest=json.loads((src/"stages/140m/stage_manifest.json").read_text())
original=src/"stages/140m/final.pkl"
if (manifest["source_git_commit"]!=os.environ["SOURCE_SHA"] or
    manifest["target_steps"]!=140_000_000 or
    manifest["training_seed"]!=735 or
    hashlib.sha256(original.read_bytes()).hexdigest()!=manifest["checkpoint_sha256"]):
    raise RuntimeError("140M original checkpoint is not the authenticated cold stage")
for origin,target in (
    (original,dst/"checkpoints/final.pkl"),
    (src/"training_config.json",dst/"training_config.json")):
    target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():
        temporary=target.with_name("."+target.name+".copying")
        if temporary.exists(): raise RuntimeError("Previous copy interrupted: "+str(temporary))
        shutil.copy2(origin,temporary)
        os.replace(temporary,target)
        target.chmod(0o444)
    if hashlib.sha256(origin.read_bytes()).digest()!=hashlib.sha256(target.read_bytes()).digest():
        raise RuntimeError("Staged parent differs from original: "+str(target))
print("Verified immutable 140M external parent:",dst/"checkpoints/final.pkl",flush=True)
PY
PARENT="$PARENT_DIR/checkpoints/final.pkl"
PARENT_HASH="$(sha256sum "$PARENT" | awk '{print $1}')"
COMMON=(
  --profile sim2real --run-dir "$RUN_DIR"
  --envs 2048 --seed 735 --device cuda
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
echo "===== SOURCE-PINNED FREESTYLE STAIR FORK PLAN ====="
"$PY" -m emu.cli train "${COMMON[@]}" --steps "$TRAIN_TARGET" --plan
if [[ "$MODE" == plan ]]; then
  echo "Plan complete; no training started."
  exit 0
fi
if [[ "$MODE" == all || "$MODE" == run || "$MODE" == pilot ]]; then
  if [[ -f "$RUN_DIR/training_config.json" ]]; then
    # Resume of THIS source, never transplant the external ancestor twice.
    RESUME=(--resume auto)
  else
    RESUME=(--resume "$PARENT"
            --research-parent-source "$SOURCE_SHA"
            --research-parent-sha256 "$PARENT_HASH")
    "$PY" -m pytest -q tests/test_core_stair_teaching.py
    "$PY" scripts/smoke_gpu_carpet.py --device cuda --worlds 128 --frames 2 --prisms --ppo
  fi
  "$PY" -m emu.cli train "${COMMON[@]}" --steps "$TRAIN_TARGET" "${RESUME[@]}" --check-resume-only
  "$PY" -m emu.cli train "${COMMON[@]}" --steps "$TRAIN_TARGET" "${RESUME[@]}"
  if [[ "$MODE" == run ]]; then
    echo "Training complete. Next: bash scripts/run_h40_stair_freedom_wsl.sh exam"
    exit 0
  fi
fi

# Independent actor examinations (no PPO/optimizer updates), same 2,000
# labelled stair worlds and same 2,000 matched flat worlds as original run.
CHECKPOINT="$RUN_DIR/checkpoints/final.pkl"
[[ -f "$CHECKPOINT" ]] || { echo "No completed experimental checkpoint: $CHECKPOINT" >&2; exit 1; }
ACTOR="$RUN_DIR/actor-${AGE}m.emupolicy"
if [[ ! -f "$ACTOR" ]]; then
  "$PY" -m emu.cli export-policy "$CHECKPOINT" "$ACTOR"
fi
for task in stairs flat; do
  OUT="$RUN_DIR/heldout/$task"
  if [[ -f "$OUT/matrix_summary.json" && -f "$OUT/matrix_worlds.csv" ]]; then
    echo "Reusing completed $task examination; do not overwrite its evidence."
    continue
  fi
  [[ ! -d "$OUT" ]] || {
    echo "Incomplete $task examination at $OUT; inspect before retrying" >&2; exit 1;
  }
  if [[ "$task" == stairs ]]; then
    "$PY" scripts/v5053_matrix_exam.py --policy "$ACTOR" --output "$OUT" \
      --mode stairs --worlds 2048 --horizon 350 --seed 1701 --device cuda \
      --levels 20 --trials-per-level 50 --max-step-m .08
  else
    "$PY" scripts/v5053_matrix_exam.py --policy "$ACTOR" --output "$OUT" \
      --mode flat --worlds 2048 --horizon 350 --seed 1701 --device cuda \
      --trials-per-level 2000
  fi
done

"$PY" - <<'PY'
import csv,hashlib,io,json,os,zipfile
from pathlib import Path
base=Path(os.environ["SOURCE_RUN"])
root=Path(os.environ["EXP_ROOT"])
new=Path(os.environ["RUN_DIR"])
def read(path):
    with path.open(newline="",encoding="utf-8") as f: return list(csv.DictReader(f))
def stages(root,age):
    if age=="new":
        return (read(root/"heldout/stairs/matrix_worlds.csv"),
                read(root/"heldout/flat/matrix_worlds.csv"))
    return (read(root/f"heldout/{age}m/stairs/matrix_worlds.csv"),
            read(root/f"heldout/{age}m/flat/matrix_worlds.csv"))
def bands(stairs,flat):
    return {
        "up_4_40":[r for r in stairs if r["kind"]=="step_up" and .0<float(r["exact_step_m"])<=.040001],
        "up_44_60":[r for r in stairs if r["kind"]=="step_up" and .04399<=float(r["exact_step_m"])<=.060001],
        "up_64_80":[r for r in stairs if r["kind"]=="step_up" and float(r["exact_step_m"])>=.06399],
        "down_4_40":[r for r in stairs if r["kind"]=="step_down" and -.040001<=float(r["exact_step_m"])<0.],
        "down_44_80":[r for r in stairs if r["kind"]=="step_down" and float(r["exact_step_m"])<=-.04399],
        "flat":[r for r in flat if r["kind"]=="flat_control"],
    }
def fraction(rows,key):
    if not rows: raise RuntimeError("Missing independent worlds for "+key)
    return sum(r[key].lower()=="true" for r in rows)/len(rows)
rows={}
candidate=os.environ["CANDIDATE"]
for label,folder,age in (("140M",base,140),("160M",base,160),(candidate,new,"new")):
    stair,flat=stages(folder,age)
    groups=bands(stair,flat)
    rows[label]={}
    for task,subset in groups.items():
        primary="traversed_5frame" if task=="flat" else "traversed_upright_5frame"
        rows[label][task]=dict(n=len(subset),
            objective=fraction(subset,primary),
            both_feet=fraction(subset,"both_feet_top_five"),
            strict_lane_heading=fraction(subset,"traversed_5frame"),
            survival=fraction(subset,"survived"),
            mean_lateral_m=sum(float(r["max_abs_lateral_offset_m"]) for r in subset)/len(subset))
thresholds={
    "up_44_60_upright_at_least_50pct":rows[candidate]["up_44_60"]["objective"]>=.50,
    "up_64_80_upright_at_least_20pct":rows[candidate]["up_64_80"]["objective"]>=.20,
    "down_44_80_upright_at_least_60pct":rows[candidate]["down_44_80"]["objective"]>=.60,
    "flat_strict_at_least_45pct":rows[candidate]["flat"]["objective"]>=.45,
    "up_64_80_survival_at_least_55pct":rows[candidate]["up_64_80"]["survival"]>=.55,
}
out={"experiment":f"140M->{os.environ['AGE']}M late-stage stair freedom; same seed, frozen parent",
     "parent_source":os.environ["SOURCE_SHA"],
     "branch_source":os.popen("git rev-parse HEAD").read().strip(),
     "objective":"No lane/yaw requirement for stairs only; flat remains strict",
     "stages":rows,"gates":thresholds,
     "all_gates_pass":all(thresholds.values()) if candidate=="freedom160M" else None,
     "pilot_not_a_promotion_test":candidate!="freedom160M"}
(root/"comparison.json").write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8")
lines=[f"# H40 Stair freedom: same 140M parent, +{int(os.environ['AGE'])-140}M PPO", "",
       "Stairs use upright five-frame two-foot crossing; flat uses strict lane/heading control.",
       "No credit for a lead foot alone. Original 140M/160M held-out worlds are unchanged.","",
       f"| Physical goal | Original 140M | Original 160M | Freestyle {os.environ['AGE']}M |",
       "|---|---:|---:|---:|"]
for task in ("flat","up_4_40","up_44_60","up_64_80","down_4_40","down_44_80"):
    cells=[f"{rows[stage][task]['objective']:.1%}" for stage in ("140M","160M",candidate)]
    lines.append("| "+task+" | "+" | ".join(cells)+" |")
lines+=["","## Prespecified retention and exploration gates",
        "Exploratory only at 145M: gates do NOT authorise promotion." if candidate!="freedom160M" else "Full 160M criteria.",""]
lines += [f"- {'PASS' if passed else 'FAIL'}: {key}" for key,passed in thresholds.items()]
lines+=["","All per-world physical proof, strict lane metrics, lateral offsets and survival are in the CSVs.",
         "One parent seed, one exam seed; do not promote to main without independent cold-seed replication.",""]
(root/"comparison.md").write_text("\n".join(lines),encoding="utf-8")
bundle=root/"stair_freedom_results.zip"
files=["comparison.md","comparison.json",
       f"experiment/actor-{os.environ['AGE']}m.emupolicy","experiment/training_config.json",
       "experiment/run_meta.json"]
files += [f"experiment/heldout/{kind}/{filename}"
          for kind in ("stairs","flat")
          for filename in ("matrix_worlds.csv","matrix_curve.csv","matrix_summary.json")]
with zipfile.ZipFile(bundle,"w",zipfile.ZIP_DEFLATED) as z:
    for f in files: z.write(root/f,arcname=f)
print("\n".join(lines))
print("UPLOAD ZIP:",bundle,flush=True)
PY
echo "Completed isolated A/B research fork. Original run and PR #96 untouched."
