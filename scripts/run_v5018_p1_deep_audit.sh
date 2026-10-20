#!/usr/bin/env bash
# v5018 P1 ROOT-CAUSE audit: real collision surface normal PLUS a MOVING
# frozen-brain step course, unlike previous teleport-only top support smoke.
# NEVER modifies main/showcase/checkpoints, and does not train.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"
[[ "$(git branch --show-current)" == "feature/v5018-terrain-awareness" ]] || {
  echo "Wrong Git branch; v5018 deep audit is feature-branch-only" >&2; exit 2;
}
[[ -x .venv/bin/python ]] || { echo "No .venv; read docs/INSTALL.md" >&2;exit 2; }
source .venv/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
BASE="$HOME/robot_lab/emu_exams"
mkdir -p "$BASE"
OUT="$BASE/v5018_p1_deep_$(date +%Y%m%d_%H%M%S)"
[[ ! -e "$OUT" ]] || { echo "Result directory exists" >&2;exit 2; }
mkdir "$OUT"
printf 'stage\tstatus\texit_code\n' > "$OUT/stages.tsv"

archive_exit() {
  local status="$?"
  trap - EXIT
  if tar -czf "$OUT.tar.gz" -C "$BASE" "$(basename "$OUT")"; then
    echo "Archive: $OUT.tar.gz (exit $status)"
  else
    echo "Failed to archive results; preserve $OUT" >&2
  fi
}
trap archive_exit EXIT
exec 2> >(tee -a "$OUT/runner_stderr.log" >&2)
run() {
  local stage="$1";shift
  echo "========== $stage ==========" | tee -a "$OUT/campaign.log"
  set +e
  (cd "$ROOT" && "$@") 2>&1 | tee "$OUT/$stage.log"
  local code="${PIPESTATUS[0]}"
  set -e
  printf '%s\t%s\t%s\n' "$stage" "$([[ "$code" -eq 0 ]] && echo PASS || echo FAIL)" "$code" >> "$OUT/stages.tsv"
  if [[ "$code" -ne 0 ]]; then
    echo "FAIL $stage: preserve all partial JSON; no gate promoted" >&2
    return "$code"
  fi
}
{
  echo "Study: P1 contact-normal classification and REAL moving 260M frozen-actor A/B"
  date -Is
  echo "Git commit: $(git rev-parse HEAD)"
  echo "Branch: $(git branch --show-current)"
  git status --short
  python -c 'import jax,mujoco;print("JAX",jax.__version__,"MuJoCo",mujoco.__version__,"devices",jax.devices())'
  nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader || true
  sha256sum emu/robot.xml showcase/champion.emupolicy
} | tee "$OUT/environment.txt"

run source_syntax python -m py_compile emu/terrain.py emu/environment.py emu/ppo.py \
  scripts/v5018_p1_surface_smoke.py scripts/v5018_p1_dynamic_reward_audit.py
run adversarial_top_vs_riser_and_normal_tests python -m pytest -q \
  tests/test_v5018_terrain_truth.py
# Bounded initial actual-MJX test: same XML, 3-frame scalar reward and
# actor input equality, then actual contact normals at a controlled pose.
run mjx_static_real_contact_and_flat_parity python scripts/v5018_p1_surface_smoke.py \
  --worlds 4 --device cuda --output "$OUT/mjx_static_contact.json"
# IMPORTANT: this is now actual MOVEMENT under the frozen neural actor,
# including onset/loss of contact and possible dynamics/terminal divergence.
# It intentionally runs separate legacy and terrain-metric rollouts.
run mjx_frozen_actor_moving_step_ab python scripts/v5018_p1_dynamic_reward_audit.py \
  --seed 1701 --worlds 4 --horizon 240 --step-mm 10 --heading-kp .75 \
  --output "$OUT/mjx_dynamic_ab.json"

echo "=============== ALL GATES PASS (read the actual JSON) ==============="
cat "$OUT/stages.tsv"
echo "A technical PASS confirms traces/metrics, NOT robust step crossing or servo safety."
