#!/usr/bin/env bash
# v5020: large, continuous, 2048-world uphill learning from validated 32M.
# Source branch ONLY. Never touch main, never restart from the failed 24M up.
set -Eeuo pipefail

usage() {
  cat <<'HELP'
EMU v5020: source-pinned overnight rated-step-up continuation
Usage: bash scripts/overnight_v5020_up.sh --plan|--run [--stop-after up64|up128|up224|up320]

Default: 2048 MJX GPU environments, seed 733, 32M -> 64M -> 128M
         -> 224M -> 320M ABSOLUTE steps (288M additional total).
A brand-new phase inherits the entire policy, optimizer, normalizer, field,
curriculum, commands, disturbances, reward and rated obstacle state.
The terrain physics is IDENTICAL across stages: only the runtime target
and output directory change. The 6mm figure is the hard physical safety
ceiling, not a forced curriculum difficulty or success certification.

Preflight: source/contract tests, clean branch, old champion exact SHA256,
source fingerprint, saved full network/optimizer/normalizer and rated level.
Each phase: continue, independent frozen evaluation, hard survival gate,
checkpoint digest, results.md + results_data.json + results_bundle.zip.
Stops on physics runaway or mass falls; quarantines a broken child.
The earlier good 32M parent is ALWAYS left untouched.
Reports can be rebuilt independently with:
  emu report --root ~/robot_lab/emu_runs/<overnight_run_folder>

--plan performs no training and creates no files.
--run stays in the foreground; overnight shell/session must remain running.
Optional overrides: STOP_AFTER, RUN_ROOT, PARENT_FROM, PARENT_SHA256.
HELP
}

MODE="${1:---plan}"
[[ "$MODE" == --run || "$MODE" == --plan || "$MODE" == --help ]] ||
  { usage >&2; exit 2; }
if [[ "$MODE" == --help ]]; then usage; exit 0; fi
STOP_AFTER="${STOP_AFTER:-up320}"
if (( $# > 1 )); then
  [[ $# == 3 && "$2" == --stop-after ]] || { usage >&2; exit 2; }
  STOP_AFTER="$3"
fi
case "$STOP_AFTER" in up64|up128|up224|up320) ;; *)
  echo "Invalid --stop-after: $STOP_AFTER" >&2; exit 2;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"
BRANCH="$(git branch --show-current)"
[[ "$BRANCH" == feature/v5020-coldstart-terrain ]] ||
  { echo "STOP: only feature/v5020-coldstart-terrain is permitted" >&2; exit 1; }
SHA="$(git rev-parse HEAD)"
SHORT="${SHA:0:12}"
RUN_ROOT="${RUN_ROOT:-$HOME/robot_lab/emu_runs/v5020_overnight_up_$SHORT}"
[[ "$RUN_ROOT" == /* ]] || { echo "STOP: RUN_ROOT must be absolute" >&2; exit 1; }
case "$RUN_ROOT/" in "$ROOT/"*) echo "STOP: output inside source checkout" >&2; exit 1;; esac
[[ "$RUN_ROOT" != "$HOME" ]] || { echo "STOP: cannot output in home root" >&2; exit 1; }

KNOWN_PARENT="$HOME/robot_lab/emu_runs/v5020_2048_6c22a5678442/up/checkpoints/final.pkl"
KNOWN_SHA="ffd9c2126cc5567826ed2820a0fbdb830edea52597dd6f2e5a7fb4c0bc75c344"
KNOWN_SOURCE="6c22a56784424c576c254c153b172a40849bf235"
PARENT_FROM="${PARENT_FROM:-$KNOWN_PARENT}"
PARENT_SHA256="${PARENT_SHA256:-$KNOWN_SHA}"

echo "=== EMU v5020 continuous rated-up overnight ==="
echo "Branch: $BRANCH"
echo "Source: $SHA"
echo "Output: $RUN_ROOT"
echo "Parent: $PARENT_FROM"
echo "Sequence: 32,047,104 -> 64M -> 128M -> 224M -> 320M"
echo "Terrain: same existing raised platform / same 6mm safety cap"
echo "Worlds: 2048; seed 733; stop after $STOP_AFTER"

if [[ "$MODE" == --plan ]]; then
  echo "PLAN ONLY: no training, checkpoint load, or file writes."
  exit 0
fi

[[ -z "$(git status --porcelain)" ]] ||
  { echo "STOP: Git checkout is dirty; source must be reproducible" >&2; exit 1; }
[[ -f .venv/bin/activate ]] ||
  { echo "STOP: .venv/bin/activate unavailable" >&2; exit 1; }
# shellcheck disable=SC1091
source .venv/bin/activate
command -v emu >/dev/null || { echo "STOP: emu command unavailable" >&2; exit 1; }

echo "=== Preflight: syntax, source regression and report contracts ==="
bash -n scripts/overnight_v5020_up.sh
python -m compileall -q emu scripts/overnight_v5020_up.sh
python -m pytest -q tests/test_v5020_overnight.py tests/test_v5020_report.py \
  tests/test_v5020_2048_launcher.py tests/test_v5020_rated_elevation.py ||
  { echo "STOP: preflight tests failed; GPU training was NOT started" >&2; exit 1; }

[[ -f "$PARENT_FROM" && "$PARENT_FROM" == /* ]] ||
  { echo "STOP: original 32M parent final.pkl missing: $PARENT_FROM" >&2; exit 1; }
[[ "$PARENT_SHA256" =~ ^[0-9a-f]{64}$ ]] ||
  { echo "STOP: expected parent digest must be a full SHA256" >&2; exit 1; }
actual="$(sha256sum "$PARENT_FROM" | cut -d' ' -f1)"
[[ "$actual" == "$PARENT_SHA256" ]] || {
  echo "STOP: saved 32M checkpoint bytes have CHANGED" >&2
  echo "expected=$PARENT_SHA256 found=$actual" >&2
  exit 1
}
echo "Parent identity: SHA256 $actual (confirmed)"
# Full 312-input checkpoint, optimizer, source step and config comparison.
python - "$PARENT_FROM" "$KNOWN_SOURCE" <<'PY'
import sys
from pathlib import Path
from emu.checkpoint_io import load_checkpoint
from emu.training_config import TrainingConfig, training_sensor_contract
from emu.contracts import OBS_SCHEMA_PREVIEW
p=Path(sys.argv[1]).resolve()
config=TrainingConfig.load(p.parent.parent/'training_config.json')
assert config.envs==2048 and config.seed==733
assert config.observation_schema==OBS_SCHEMA_PREVIEW
assert config.terrain_elevation_mode=='up'
assert config.terrain_height_safety_m==.006
assert config.terrain_height_initial_m==0.
payload=load_checkpoint(p,validate=True,expected_observation_schema=OBS_SCHEMA_PREVIEW,
                        expected_sensor_contract=training_sensor_contract(config))
meta=payload['meta']
assert int(meta['step'])==32047104, meta['step']
assert meta['source_provenance']['git_commit']==sys.argv[2]
rating=float(payload['curriculum_state']['params']['frontier_terrain_height_m'])
assert 0. <= rating <= .006, rating
print(f"Validated 32M final policy, optimizer, normalizer and 312-input sensor contract; restored rated step-up={rating*1000:.3f}mm",flush=True)
PY

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/lineage"
source_file="$RUN_ROOT/lineage/source_commit.txt"
if [[ -f "$source_file" && "$(cat "$source_file")" != "$SHA" ]]; then
  echo "STOP: overnight run already pinned to a DIFFERENT source commit" >&2
  exit 1
fi
printf '%s\n' "$SHA" > "$source_file"
trap 'code=$?; echo "Overnight v5020 exited $code. Logs: $RUN_ROOT/logs. Source 32M parent untouched." >&2' ERR

# The same options are passed for ALL stages, without escalating
# the command distribution, force, reward or sampled terrain difficulty.
PROFILE=(
  --profile terrain-bridge --forward-preview --pitch-deg 8
  --terrain-varied-placement --terrain-elevation-mode up
  --terrain-height-ceiling-mm 6 --terrain-height-initial-mm 0
  --terrain-height-exposure .40 --terrain-height-edge-fraction .30
  --terrain-height-growth-mm 1
  --terrain-progress-weight 0
)

gate_stage() {
  local name="$1"
  python - "$RUN_ROOT/$name" <<'PY'
import csv, math, sys
from pathlib import Path
from statistics import median
p=Path(sys.argv[1])
def read(name):
    with (p/name).open(newline='',encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))
metrics=read('metrics.csv')
fixed=read('fixed_eval.csv')
if not metrics or not fixed:
    raise SystemExit('STOP: no metrics or frozen evaluation for '+str(p))
last=metrics[-1]
eval_row=fixed[-1]
step=int(float(last['step']))
exam=int(float(eval_row['eval_policy_step']))
if step!=exam:
    raise SystemExit(f'STOP: frozen exam is not from exact final stage policy ({exam} != {step})')
def finite(row,key):
    x=float(row[key])
    if not math.isfinite(x): raise ValueError('nonfinite metric '+key)
    return x
pass_fraction=finite(eval_row,'eval_foundation_walking_pass')
core=finite(eval_row,'eval_core_survival')
fall=median(finite(r,'fall_rate') for r in metrics[-3:])
length=median(finite(r,'ep_len_mean') for r in metrics[-3:])
rating_mm=finite(last,'frontier_terrain_height_m')*1000.
reward=finite(last,'reward_mean')
print(f'Stage gate: step={step:,} frozen_walk={pass_fraction:.1%} core={core:.1%} last3_falls={fall:.1%} last3_len={length:.0f} rated_step={rating_mm:.3f}mm reward={reward:+.3f}',flush=True)
# A core walker has to survive. Small losses on difficult step attempts
# do NOT fail the stage; profound regressions do.
if pass_fraction<.75 or core<.75 or fall>.25 or length<400:
    raise SystemExit('STOP: walking retention regressed; prior stage remains safe. DO NOT launch the next stage.')
PY
}

report_now() {
  echo "=== Rebuilding shareable reports (NO checkpoint or GPU changes) ==="
  emu report --root "$RUN_ROOT"
  echo "Share in the morning: $RUN_ROOT/results_bundle.zip"
}

run_stage() {
  local name="$1" target="$2" parent="$3"
  local expected_source="$4"
  local directory="$RUN_ROOT/$name"
  local final="$directory/checkpoints/final.pkl"
  local log="$RUN_ROOT/logs/${name}_$(date +%Y%m%d_%H%M%S).log"
  [[ -f "$parent" ]] ||
    { echo "STOP: missing stage parent: $parent" >&2; exit 1; }
  if [[ -f "$RUN_ROOT/UNSAFE_RESUME.json" || -f "$directory/UNSAFE_RESUME.json" ]]; then
    echo "STOP: quarantined stage, DO NOT AUTO-RESUME." >&2; exit 1
  fi
  local hash="$(sha256sum "$parent" | cut -d' ' -f1)"
  local lineage="$RUN_ROOT/lineage/$name.parent.sha256"
  if [[ -f "$lineage" && "$(cat "$lineage")" != "$hash" ]]; then
    echo "STOP: $name parent checkpoint hash changed" >&2; exit 1
  fi
  printf '%s\n' "$hash" > "$lineage"

  local -a approval=() resume=()
  if [[ "$expected_source" != "$SHA" ]]; then
    approval=(--approve-parent-commit "$expected_source"
              --approve-parent-sha256 "$hash")
  fi
  if [[ -d "$directory/checkpoints" ]] &&
     find "$directory/checkpoints" -maxdepth 1 -type f -name '*.pkl' -print -quit | grep -q .; then
    resume=(--resume auto)
  else
    resume=(--resume "$parent")
  fi
  echo
  echo "=== $name: full 2048 worlds, absolute target $target transitions ==="
  echo "From: $parent (SHA256 $hash)"
  echo "Source: $SHA"
  echo "Log: $log"
  XLA_PYTHON_CLIENT_PREALLOCATE=false emu train \
    --run-dir "$directory" --steps "$target" \
    --envs 2048 --seed 733 --device cuda \
    "${PROFILE[@]}" "${approval[@]}" "${resume[@]}" \
    2>&1 | tee -a "$log"
  [[ -s "$final" ]] ||
    { echo "STOP: $name did not publish a final checkpoint" >&2; exit 1; }
  gate_stage "$name"
  local output_hash="$(sha256sum "$final" | cut -d' ' -f1)"
  local saved="$RUN_ROOT/lineage/$name.final.sha256"
  if [[ -f "$saved" && "$(cat "$saved")" != "$output_hash" ]]; then
    echo "STOP: existing $name final bytes differ from pinned result" >&2; exit 1
  fi
  printf '%s\n' "$output_hash" > "$saved"
  echo "=== $name accepted: $output_hash ==="
  report_now
}

run_stage up64   64000000 "$PARENT_FROM" "$KNOWN_SOURCE"
if [[ "$STOP_AFTER" == up64 ]]; then exit 0; fi
run_stage up128 128000000 "$RUN_ROOT/up64/checkpoints/final.pkl" "$SHA"
if [[ "$STOP_AFTER" == up128 ]]; then exit 0; fi
run_stage up224 224000000 "$RUN_ROOT/up128/checkpoints/final.pkl" "$SHA"
if [[ "$STOP_AFTER" == up224 ]]; then exit 0; fi
run_stage up320 320000000 "$RUN_ROOT/up224/checkpoints/final.pkl" "$SHA"

echo
echo "=== Overnight continuous uphill run complete ==="
echo "Checkpoint: $RUN_ROOT/up320/checkpoints/final.pkl"
echo "Results: $RUN_ROOT/results.md"
echo "Evidence: $RUN_ROOT/results_bundle.zip"
echo "Completion is NOT independent 6mm stair certification."
