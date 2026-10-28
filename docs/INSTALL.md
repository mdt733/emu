# Install, train and inspect EMU (WSL2 / Ubuntu)

The public `main` demonstration and 360-input research actor are
**different policies**. This page describes the current v5028 research
environment and how to inspect results without changing an ongoing
source-pinned experiment.

[Project overview](../README.md) ·
[Current handover](../HANDOVER.md) ·
[v5028 experiment](V5028_CONSIDERED_STEPS.md) ·
[physical exams](RELEASE_READINESS.md)

## Software prerequisites

Windows 11 + WSL2 Ubuntu, an NVIDIA GPU with compatible Windows and
WSL CUDA/JAX drivers, Python 3.13+ and a working MuJoCo/MJX
installation. The observed owner environment uses an **RTX 4060 Ti
8 GB**, **2,048 environments** and `~/emu-graduation/.venv`. The
new actor has 360 observation inputs, six action outputs and delayed
four-by-four terrain depth readings.

For a fresh machine, inspect/run `scripts/setup_wsl.sh` in a new clone
and follow its explicit dependencies. It does **not** start a GPU
training campaign. Don't overwrite an existing source-pinned venv or
checkout during a running experiment.

## Existing owner setup: preserve the running v5028 SHA

The owner launched `~/emu-v5028` on 29 October 2026.
**Do not `git pull`, rebase, cherry-pick, change source or install a
new editable checkout there while the GPU training run is live.**
The exact historical training SHA is
`38713791023681dc9883f6c374d92e46261b3baf`;
the independent `feature/v5028-merge-polish` branch is not its
resume source.

To inspect, **without launching**:

```bash
cd ~/emu-v5028
git branch --show-current
git rev-parse HEAD
git status --short
ls -lh ~/robot_lab/emu_runs/v5028_considered_steps_s735/
```

## Foreground training and honest failures

The source-pinned v5028 launcher is plan-only unless `--run` is
supplied. It checks source/checkout identity and then performs CPU and
full **2,048-world CUDA PPO** preflights before spending GPU time.
The owner venv was missing `pytest`, an optional testing package:
use the small dependency-free reward check directly rather than
silently skipping the test.

In a **new** v5028 training session, after confirming no other GPU run
is occupying that source and the checkout is clean:

```bash
cd ~/emu-v5028
source ~/emu-graduation/.venv/bin/activate
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python -m py_compile emu/environment.py emu/rewards.py \
    scripts/smoke_unified_focus.py scripts/v5028_considered_steps.py

python - <<'PY'
import runpy
scope = runpy.run_path("tests/test_considered_steps.py")
scope["test_step_relief_only_underspeed"]()
print("Reward check passed")
PY

# One console, no tmux; errors stay logged and visible.
python -u scripts/v5028_considered_steps.py --run \
    2>&1 | tee -a "$HOME/v5028_launcher_console.log"
rc=${PIPESTATUS[0]}
printf '\nEMU exit code: %s\n' "$rc"
```

Do **not** paste a leading `set -euo pipefail` into a disposable
WSL shell: a single missing dependency can exit that shell immediately.
For a plain interactive launch, `set +e; set +u` leaves the console
available for diagnosis. The Python launcher itself remains fail-fast
on wrong source, incompatible checkpoints or CPU/GPU preflight errors.

After a preflight-stamped interruption, use the **exact same SHA**
and the normal `--run --skip-smoke` option. Never overwrite a
manifest or pretend a partial run completed.

## Results and physical examination

v5028's output root:

```text
~/robot_lab/emu_runs/v5028_considered_steps_s735/
  campaign_manifest.json
  shared_foundation20/
  A_original_coherent/
  B_considered_up/
  physical_exams/
  physical_summary.md
  overnight_results.zip
  logs/
```

When collecting results, include the actual source SHA and entire
`overnight_results.zip`. The **original** pinned launcher used
`--worlds 8` per physical case, split into up/down worlds for steps;
it does **not** meet the promised sixteen-world replication.
Perform a separate read-only 16–32-world exam using the original
policy's compatible source and frozen conditions before declaring
3cm mastery. Distinguish **step-UP** from **step-DOWN** and supported
crossing from command-heading success.

A compatible policy can be exported and examined through `emu.cli`;
the fixed examiner uses the real physical contact geometry and
never supplies a training reward:

```bash
python -m emu.cli export-policy \
  "$HOME/robot_lab/emu_runs/v5028_considered_steps_s735/B_considered_up/checkpoints/final.pkl" \
  "$HOME/robot_lab/emu_runs/v5028_considered_steps_s735/B_considered_up.emupolicy"

python -m emu.cli capability-exam --terrain-matrix \
  --policy "$HOME/robot_lab/emu_runs/v5028_considered_steps_s735/B_considered_up.emupolicy" \
  --output "$HOME/robot_lab/emu_runs/v5028_considered_steps_s735/independent_exam" \
  --device cpu --worlds 16 --horizon 350 --speed 0.15 \
  --matrix-grades-deg 5 8 10 12 15 \
  --matrix-heights-cm 2 2.5 3 3.5
```

Run those last two commands **only after the intended final checkpoint
exists**, and with a source compatible with that archived policy.
Exam results are still simulation evidence, not hardware-safe refusal.

## Browser playground and manual surfaces

```bash
emu viewer-build
emu viewer showcase/champion.emupolicy --no-open
```

The published playground uses the **protected** approved 232-input
Pinnacle. The local experimental viewer and [surface laboratory](SURFACE_LAB.md)
may inspect 360 policies, but do not implicitly replace the public
champion. No public deployment without [release approval](RELEASE_READINESS.md).

For earlier launch recipes consult [historical research](history/README.md)
at their **original recorded Git SHA**, never copy an obsolete command
into the current checkout.
