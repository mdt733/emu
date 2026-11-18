# EMU installation and safe recovery · Windows 11 / WSL2

This guide supports **one maintained H40 learner**, its browser and a 2,048-environment RTX GPU training run. Installing does not start PPO, replace a published policy or delete existing run data.

## Prerequisites

- Windows 11 with WSL2 Ubuntu and a recent **Windows NVIDIA driver** supporting CUDA in WSL. Do **not** install generic Linux GPU drivers into WSL just to make JAX work.
- Python **3.13+**, with `venv` support. If Ubuntu does not provide it, use a user-local Python 3.13 installation (for example via `uv`).
- Git, curl, CA certificates and permission to install Node.js under your Linux user. `scripts/setup_wsl.sh --install-node` uses nvm and the version in `.nvmrc`.
- For the browser only, CUDA and a training GPU are not required.

## Fresh clone

```bash
sudo apt-get update
sudo apt-get install -y git curl ca-certificates python3-venv
git clone https://github.com/mdt733/emu.git
cd emu
bash scripts/setup_wsl.sh --cuda13 --install-node --check
```

If your distribution's `python3` is older than 3.13, install Python 3.13 first and verify `python3.13 --version`. The setup script uses a checkout-local `.venv` and installs the Python project, JAX CUDA13 wheels and the locked Node viewer dependencies. `--check` runs the maintained tests; without it installation is faster. It **never runs training**. To set up just the browser, use `bash scripts/setup_wsl.sh --viewer-only --install-node`.

Verify the GPU **inside WSL**:

```bash
source .venv/bin/activate
emu doctor --require-gpu
```

A GPU reported by `nvidia-smi` is not sufficient evidence of a functional JAX/MJX GPU backend. The training launcher runs its own 128-world and 2048-world PPO smokes before continuous learning.

## Train once, from zero

```bash
bash scripts/run_h40_holistic_cold_wsl.sh plan
bash scripts/run_h40_holistic_cold_wsl.sh all
```

The single script validates a clean named Git source and H40 Spring V1 physical hash. It starts a **new** seed-735 run in `~/emu-data/runs/h40-main-140m-s735/` (never the original validated `h40-holistic-cold-s735`), preserves 20/40/120/140M stage checkpoints, stops if early foundation health collapses, evaluates all final-stage frozen policies and writes the evidence bundle.

Data lives at `~/emu-data/runs/h40-holistic-cold-s735/`, **outside Git**. To conduct a second independent run from random weights on an unchanged source revision, set `EMU_COLD_SEED=736` in the launcher invocation; it uses a different output directory. Never re-use the seed-735 directory for a different experiment.

## Safe resume and source changes

A full checkpoint contains network, optimiser, observation normalisation, teacher state and physical/source provenance. Resuming through a stage uses the same source-validated `--resume auto`. **Don't switch a running learner from the original holistic experiment commit to a newer cleaned-up commit** even if the runtime training modules appear byte-identical; this still changes source provenance. Keep the original checkout and checkpoint directory untouched.

If you need to build/review the clean `main` candidate in parallel, use a **separate Git worktree or clone**, with its own `.venv` if necessary. Do not delete your running `~/emu-data` tree, reset a branch, or modify the user's saved actor just to fix an install.

To inspect logs or validated checkpoints without training:

```bash
source .venv/bin/activate
emu dashboard --run-dir ~/emu-data/runs/h40-holistic-cold-s735 --once --plain
emu report --root ~/emu-data/runs/h40-holistic-cold-s735
```

For an independent exam on a frozen export, see `emu prism-exam --help`. The holistic launcher runs the exact 2k-stair + 2k-flat held-out matrix at ages 120/140M automatically. Never confuse the browser's Meadow terrain with identical MJX collision physics.

## Browser and geometry lab

```bash
source .venv/bin/activate
emu viewer-build
emu viewer showcase/champion.emupolicy --no-open --port 8765
```

Open http://127.0.0.1:8765/. The checked-in champion is the previously approved 120M actor until a separately reviewed replacement is published. For untrained ToF/collision inspection: `bash scripts/start_sensor_lab_wsl.sh`.

## Validation and failure handling

Run `bash scripts/check.sh --quick` or `bash scripts/check.sh --full`. CI also runs actual small-world CPU prism contacts; it cannot train 140M on your RTX. Keep the exact source SHA, error and run directory on a failed run. Do **not** erase the run or copy historical checkpoints into the new study to get past a failure.

Repository/build policy and hardware assumptions: [maintenance](MAINTENANCE.md) · [core skeleton](CORE_SKELETON.md). An old 120M release policy is **not** the new cold-start model and must not be silently substituted.
