#!/usr/bin/env bash
# EMU v5018: narrowly targeted NEW gait-v2 validation and historical 10mm
# course-repeatability check. Not a full 34-case frontier rerun, no PPO.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
if [[ "$(git -C "$ROOT" branch --show-current)" != "feature/v5018-terrain-awareness" ]]; then
  echo "Run only from the approved v5018 feature branch" >&2; exit 2
fi
if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  echo "Expected $ROOT/.venv/bin/python; see docs/INSTALL.md" >&2; exit 2
fi
source "$ROOT/.venv/bin/activate"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
BASE="$HOME/robot_lab/emu_exams"
mkdir -p "$BASE"
OUT="$BASE/v5018_gaitv2_acceptance_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
printf 'stage\tstatus\texit_code\n' > "$OUT/stages.tsv"

archive_on_exit() {
  local code="$?"
  trap - EXIT
  if tar -czf "$OUT.tar.gz" -C "$BASE" "$(basename "$OUT")"; then
    echo "Result archive: $OUT.tar.gz (exit $code)"
  fi
}
trap archive_on_exit EXIT

run() {
  local stage="$1"; shift
  echo "========== $stage ==========" | tee -a "$OUT/campaign.log"
  (cd "$ROOT" && "$@") 2>&1 | tee "$OUT/$stage.log"
  local code="${PIPESTATUS[0]}"
  if [[ "$code" == 0 ]]; then
    printf '%s\tPASS\t0\n' "$stage" >> "$OUT/stages.tsv"
  else
    printf '%s\tFAIL\t%s\n' "$stage" "$code" >> "$OUT/stages.tsv"
  fi
  echo "$stage exit=$code" | tee -a "$OUT/campaign.log"
  return "$code"
}

{
  date -Is
  printf 'branch='; git -C "$ROOT" branch --show-current
  printf 'source_commit='; git -C "$ROOT" rev-parse HEAD
  printf 'working_tree_status:\n'; git -C "$ROOT" status --short
  printf 'protected_policy_sha256='
  sha256sum "$ROOT/showcase/champion.emupolicy"
  python -c 'import jax,mujoco; print("JAX",jax.__version__,"MuJoCo",mujoco.__version__,"devices",jax.devices())'
  nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader 2>&1 || true
} | tee "$OUT/environment.txt"

# Do NOT spend GPU cycles or reinterpret earlier scoreboards if synthetic
# anti-cheat cases or exact original XML-course SHA parity fail.
if ! run v2_and_xml_parity python -m pytest -q \
  tests/test_direction.py::test_v5018_gait_v2_pairs_real_contacts_without_faking_long_single_swings \
  tests/test_direction.py::test_v5018_step_pilot_xml_yaw_and_declared_height \
  tests/test_direction.py::test_v5018_stride_diagnostics_identifies_strict_gait_rejection_without_changing_pass \
  tests/test_direction.py::test_v5018_crossing_gate_accepts_only_supported_world_fixed_crossings; then
  echo 'STOP: foundational gait or exact XML SHA test failed; preserve archive.' >&2
  exit 1
fi

# Real frozen-policy trace, old v1 and explicitly *nonofficial* new v2
# published together; identical operator commands and heading ON/OFF.
for surface in flat uphill10; do
  angle=0
  [[ "$surface" == uphill10 ]] && angle=10
  run "gait_${surface}" emu directional-exam \
    --policy "$ROOT/showcase/champion.emupolicy" \
    --output "$OUT/gait_${surface}" \
    --contact-slope --angle "$angle" --direction uphill \
    --worlds 4 --horizon 350 --settle-frames 25 \
    --speed .10 --seed 1701 --device cuda \
    --compare-heading --heading-kp .75 || true
done

# Only repeat the TWO 10mm frozen-control cases whose initial vs frontier
# outcomes disagreed under different XML hashes. The new zero-angle course
# uses the exact ORIGINAL 09:49 XML bytes (enforced by pytest above).
for seed in 1701 1702; do
  run "step_10mm_seed${seed}" python "$ROOT/scripts/v5018_champion_step_pilot.py" \
    --height-mm 10 --step-yaw-deg 0 --speed .10 \
    --heading-kp .75 --seed "$seed" --worlds 4 --horizon 350 \
    --output "$OUT/step_10mm_seed${seed}.json" || true
done

# Lightweight aggregation from the JSON actually written. Never synthesise
# missing outcomes or mark successful process execution as crossing success.
(cd "$OUT" && python - "$OUT" <<'PY'
import json, pathlib, sys
root=pathlib.Path(sys.argv[1])
result={'schema':1,'study':'v5018_gait_v2_research_and_xml_repeatability',
        'gait_cases':[],'step_cases':[]}
for course in ('flat','uphill10'):
    for mode in ('off','on'):
        path=root/f'gait_{course}'/mode/'directional_exam.json'
        if not path.exists():
            continue
        data=json.loads(path.read_text())
        worlds=data.get('worlds',[])
        result['gait_cases'].append({
            'course':course,'mode':mode,'worlds':len(worlds),
            'v1_strict_stride_pairs':[w.get('contact_stride_pairs') for w in worlds],
            'v2_research_pairs':[(
                w.get('gait_v2_research_only') or {}).get(
                    'candidate_alternating_pairs_v2') for w in worlds],
            'course_pass':[w.get('course_pass') for w in worlds],
            'v2_is_not_an_official_gait_pass':True})
for seed in (1701,1702):
    path=root/f'step_10mm_seed{seed}.json'
    if not path.exists():
        continue
    data=json.loads(path.read_text())
    result['step_cases'].append({
        'seed':seed,'course_sha256':data['world_xml_sha256'],
        'crossing_success_worlds':data['crossing_success_worlds'],
        'worlds':data['worlds'],
        'individual_pass':[w['crossing_pass'] for w in data['per_world']],
        'original_0949_crossings_expected_for_comparison':3})
(root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
PY
) 2>&1 | tee "$OUT/summary.log"

echo "============= STAGES ============="
cat "$OUT/stages.tsv"
if grep -q $'\tFAIL\t' "$OUT/stages.tsv"; then
  echo "Some stages failed; check log and archive; do not claim complete parity."
  exit 1
fi
echo "All scripts executed; read v2 diagnostics and repeated crossing counts before making any gait claim."
