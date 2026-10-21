#!/usr/bin/env bash
# V5020: one 312-input cold-start lineage, real slopes, resilience and steps.
# --run holds THIS console. No nohup, tmux, autolaunch or hidden jobs.
set -Eeuo pipefail
usage() {
  echo "Usage: bash scripts/run_v5020_coldstart.sh {smoke|foundation|slope|steps|mixed|stretch} [--run] [--arm sighted|masked]"
  echo "Without --run this only prints and validates the resolved training plan."
  echo "Environment: EMU_V5020_ROOT, EMU_V5020_ENVS (default 1024), EMU_V5020_SEED (default 733)"
}
if (($#==0));then usage; exit 2;fi
stage="$1";shift
run=0;arm=sighted
while (($#));do
  case "$1" in
    --run)run=1;;
    --arm)shift;arm="$1";;
    -h|--help)usage;exit 0;;
    *)echo "Unknown option $1" >&2;exit 2;;
  esac
  shift
done
case "$arm" in sighted|masked);;*)echo "Unknown arm $arm" >&2;exit 2;;esac
cd "$(dirname "$0")/.."
if [[ "$(git branch --show-current)" != feature/v5020-coldstart-terrain ]];then
  echo 'Use an isolated feature/v5020-coldstart-terrain checkout, not v5018/main' >&2;exit 1
fi
if ((run)) && [[ -n "$(git status --porcelain)" ]];then
  echo 'Training refuses a dirty checkout; pin the source commit first' >&2;exit 1
fi
root="${EMU_V5020_ROOT:-$HOME/robot_lab/emu_runs/v5020}"
envs="${EMU_V5020_ENVS:-1024}"
seed="${EMU_V5020_SEED:-733}"
[[ "$envs" =~ ^[0-9]+$ && "$seed" =~ ^[0-9]+$ ]] || { echo 'Invalid envs/seed' >&2;exit 2; }
((envs>=4)) || { echo 'At least four worlds required' >&2;exit 2; }
dir="$root/$arm/$stage"
sensor=(--forward-preview)
if [[ "$arm" == masked ]];then sensor+=(--forward-preview-masked);fi
parent=''
case "$stage" in
  smoke)
    [[ "$arm" == sighted ]] || { echo 'Smoke is a single physical acceptance, not an A/B arm' >&2;exit 2; }
    # A failed first compile leaves a source-pinned run_meta.json.
    # Give each source revision its own smoke directory automatically,
    # preserving failed logs without mixing checkpoint lineages.
    envs=64;dir="$root/smokes/$(git rev-parse --short=12 HEAD)"
    opts=(--profile sim2real --steps 4096 --pitch-deg 12
      --contact-terrain-radius .20 --terrain-height-ceiling-mm 15
      --terrain-height-initial-mm 3 --terrain-varied-placement --terrain-height-exposure .65)
    ;;
  foundation)
    opts=(--profile foundation --steps 20000000 --pitch-deg 8
      --contact-terrain-radius .16)
    ;;
  slope)
    parent="$root/$arm/foundation/checkpoints/final.pkl"
    opts=(--profile sim2real --steps 35000000 --pitch-deg 12
      --contact-terrain-radius .30)
    ;;
  steps)
    parent="$root/$arm/slope/checkpoints/final.pkl"
    opts=(--profile sim2real --steps 60000000 --pitch-deg 15
      --contact-terrain-radius .30 --terrain-height-ceiling-mm 25
      --terrain-height-initial-mm 3 --terrain-height-exposure .55
      --terrain-varied-placement)
    ;;
  mixed)
    parent="$root/$arm/steps/checkpoints/final.pkl"
    opts=(--profile sim2real --steps 85000000 --pitch-deg 20
      --contact-terrain-radius .55 --terrain-height-ceiling-mm 50
      --terrain-height-initial-mm 3 --terrain-height-exposure .60
      --terrain-varied-placement)
    ;;
  stretch)
    parent="$root/$arm/mixed/checkpoints/final.pkl"
    opts=(--profile sim2real --steps 105000000 --pitch-deg 25
      --contact-terrain-radius .75 --terrain-height-ceiling-mm 50
      --terrain-height-initial-mm 3 --terrain-height-exposure .65
      --terrain-varied-placement)
    ;;
  *)echo "Unknown stage: $stage" >&2;usage;exit 2;;
esac
resume=()
if [[ -f "$dir/checkpoints/latest.pkl" || -f "$dir/checkpoints/final.pkl" ]];then
  resume=(--resume auto)
elif [[ -n "$parent" ]];then
  resume=(--resume "$parent")
fi
cmd=(emu train --run-dir "$dir" --envs "$envs" --seed "$seed"
  --device cuda "${sensor[@]}" "${opts[@]}" "${resume[@]}")
echo "V5020 $stage / $arm | source $(git rev-parse --short HEAD)"
printf 'Command: ';printf '%q ' "${cmd[@]}";printf '\n'
if ((!run));then
  echo 'Planning only; no GPU run and no file writes.'
  "${cmd[@]}" --plan
  exit 0
fi
command -v emu >/dev/null || { echo 'Activate venv and pip install -e .' >&2;exit 1; }
if [[ -n "$parent" && ! -f "$parent" && ! -f "$dir/checkpoints/latest.pkl" && ! -f "$dir/checkpoints/final.pkl" ]];then
  echo "Missing approved parent final checkpoint: $parent" >&2;exit 1
fi
if [[ "$stage" == smoke ]]; then
  # On 8GB GPUs XLA's default ~75% reservation may fail to allocate
  # even when the eventual JIT graph fits. Honor an explicit user setting.
  export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
  echo "GPU smoke: XLA_PYTHON_CLIENT_PREALLOCATE=$XLA_PYTHON_CLIENT_PREALLOCATE"
fi
echo 'Training in this foreground console. Ctrl+C stops; rerun with --run to resume stage.'
# Keep the old EMU research habit: results.md, results_data.json and
# evidence ZIP belong to EACH run, outside the source checkout.
# Only report a completed/validated training stage. If training fails,
# set -e preserves its exit status and leaves diagnostics untouched.
"${cmd[@]}"
echo 'Writing offline results.md and evidence bundle for the completed stage...'
emu report --root "$dir"

