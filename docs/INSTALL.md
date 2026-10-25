# EMU — install and operate

This guide documents the **current single-run entry point**, not retired ladder/Pinnacle campaigns. For the learning philosophy see [unified learning contract](UNIFIED_CAPABILITY_GROWTH.md); for current status see [handover](../HANDOVER.md).

## Install on Windows 11 + WSL2

Use a working Ubuntu distribution with Python 3.13 and NVIDIA WSL/CUDA drivers if GPU training is intended. In WSL:

```bash
git clone https://github.com/mdt733/emu.git
cd emu
git switch feature/v5023-capability-growth
bash scripts/setup_wsl.sh --cuda13 --install-node
source .venv/bin/activate
emu doctor --v5015-smoke --domain-smoke
```

For CPU-only inspection omit `--cuda13`, then use `emu doctor --no-require-gpu --no-mjx-smoke`.

## One source-pinned training lineage

**Main reference: the first 20M steps.** `foundation_config` retains the historical simple standing/walking bootstrap: seed 733, 2,048 worlds, 32-frame rollouts, 4 PPO epochs, 8 minibatches, LR 2e-4, original eight rewards, first 70 updates steady-only and restrained initial command requests.

```bash
RUN="$HOME/robot_lab/emu_runs/foundation20_example"
emu train --run-dir "$RUN" --profile foundation --steps 20000000 \
  --envs 2048 --seed 733 --device cuda --plan
```

Remove `--plan` **only for an explicitly intended real run**. This command uses the original 220-input contract, suitable for a working `main` foundation comparison. It is **not the new 360-input unified actor**. Interrupt/resume only with the **same commit and configuration** using `--resume auto`; checkpoints must remain outside the checkout. Export a verified checkpoint:

```bash
emu export-policy "$RUN/checkpoints/final.pkl" "$RUN/foundation.emupolicy"
emu viewer-build
emu viewer --live --root "$RUN" --no-open
```


## Unified continuous learning — preferred workflow

One source-pinned run, from zero through 20M, 40M, 100M and beyond. No ancestor or separate handoff is needed. The first 20M use main-style gait teaching; active specialist lessons then blend from 0% to 35% over the following 10M. Once introduced, each physical target changes based on its own evidence, not a training-age gate.

Before an extended run, review GitHub Actions for the exact source SHA, check the 12-world CPU smoke, and run a full 2048-world CUDA PPO preflight (shown in Checks below). Neither smoke proves walking competence or stability across a long run.

### Fresh cold run — complete WSL command (120M target example)

~~~bash
(
set -euo pipefail
cd ~/emu
if [ -n "$(git status --porcelain)" ]; then
  echo "STOP: working tree contains changes"
  exit 1
fi
git switch feature/v5023-capability-growth
git pull --ff-only
if [ -n "$(git status --porcelain)" ]; then
  echo "STOP: source still dirty"
  exit 1
fi
source .venv/bin/activate
nvidia-smi
git rev-parse HEAD

RUN="$HOME/robot_lab/emu_runs/unified_cold120_733"
if [ -e "$RUN" ]; then
  echo "STOP: run directory already exists; use resume instructions"
  exit 1
fi

XLA_PYTHON_CLIENT_PREALLOCATE=false \
EMU_UNIFIED_CONTINUOUS=cold-growth-approved \
emu train \
  --run-dir "$RUN" \
  --profile sim2real \
  --terrain-actor \
  --terrain-step-max-m 0.04 \
  --unified-capabilities \
  --steps 120000000 \
  --envs 2048 \
  --seed 733 \
  --device cuda
)
~~~

The fixed continuous launch gate requires the 360-input array actor, 2048 CUDA worlds, seed 733, unchanged sensor/physical configuration, and no external checkpoint. One training run includes both foundation and gradual specialist teaching.

### Resume interrupted 120M run — complete WSL command

Do NOT pull a new Git commit on resume. Keep the original source SHA, flags and run directory; the runner independently checks that source against run_meta.json, confirms the saved training config, and selects a valid same-run checkpoint.

~~~bash
(
set -euo pipefail
cd ~/emu
git switch feature/v5023-capability-growth
if [ -n "$(git status --porcelain)" ]; then
  echo "STOP: working tree contains changes"
  exit 1
fi
source .venv/bin/activate
RUN="$HOME/robot_lab/emu_runs/unified_cold120_733"
if [ ! -s "$RUN/training_config.json" ] || [ ! -s "$RUN/run_meta.json" ]; then
  echo "STOP: original run metadata is missing"
  exit 1
fi
export RUN
python - <<'PY'
import json, os, subprocess
from pathlib import Path
meta=json.loads((Path(os.environ["RUN"])/"run_meta.json").read_text())
source=meta["source_provenance"]["git_commit"]
current=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
if current != source:
    raise SystemExit("STOP: source mismatch. Restore original commit: "+source)
print("Exact source verified:", current)
PY

XLA_PYTHON_CLIENT_PREALLOCATE=false \
EMU_UNIFIED_CONTINUOUS=cold-growth-approved \
emu train \
  --run-dir "$RUN" \
  --profile sim2real \
  --terrain-actor \
  --terrain-step-max-m 0.04 \
  --unified-capabilities \
  --steps 120000000 \
  --envs 2048 \
  --seed 733 \
  --device cuda \
  --resume auto
)
~~~

Choose a larger absolute target before the fresh run if desired. Extending a run whose saved target is already complete is a separate source-safe lifecycle change; do not manually change training_config.json. Old checkpoint-specific 20M→40M and 40M→60M handoff procedures are historical, not required for this cold-run architecture.

## Checks and viewer

```bash
bash scripts/check.sh
python scripts/smoke_unified_focus.py --device cpu --envs 12
emu terrain-exam --help
emu viewer --no-open
```

CPU smoke/CI does not establish CUDA capacity or successful locomotion. The local viewer is read-only and **not authenticated**; do not expose its port publicly. See [PLAYGROUND](PLAYGROUND.md) for viewer behaviour and protected public release.

## Short CUDA acceptance without launching training

On the RTX 4060 Ti, explicitly opt in to **one** production-shaped PPO update:

```bash
python scripts/smoke_unified_focus.py --device cuda --envs 2048 --full-update
```

This reuses the existing physical/sensor smoke and uses 32-frame rollouts, four PPO epochs and eight minibatches. It does **not** open the unified campaign or write checkpoints. Capture its GPU memory/errors and throughput before attempting training. A successful isolated update does **not** prove long-run stability, resume/export or walking competence.

Once a source-pinned, trained 360-input array-only policy exists, the independent frozen assessment is:

```bash
emu capability-exam --policy /path/to/new360.emupolicy \
  --output /path/to/new-empty-exam --device cpu --worlds 4 --quick
```

Run with `--worlds 8` for all eight recovery push directions and without `--quick` for the complete 19-case assessment. It grades genuine physical support, not training reward or battery consumption.


## Before a serious GPU run

Ensure the latest branch CI is green. Use these existing checks in the same clean source:

~~~bash
(
set -euo pipefail
cd ~/emu
git switch feature/v5023-capability-growth
git pull --ff-only
test -z "$(git status --porcelain)"
source .venv/bin/activate
python scripts/smoke_unified_focus.py --device cpu --envs 12
XLA_PYTHON_CLIENT_PREALLOCATE=false \
python scripts/smoke_unified_focus.py --device cuda --envs 2048 --full-update
)
~~~

Stop for a CUDA OOM, invalid gradients, a broken physics smoke or an incompatible sensor contract. The preflight exercises real MJX and PPO but is not an extended training run. Fixed held-out exam cohorts in unified continuous mode use 32 worlds, not an additional 2048 on the 8GB GPU.

## Safety

No writes to `main`, showcase or approved Pinnacle without owner approval. Treat reward as training incentive, not real physical walking evidence. Simulated shaft work is not measured battery energy, and untested learned servo commands are **not hardware-safe**. Check source provenance and matched physical exams before any extended GPU campaign.
