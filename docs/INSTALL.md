# EMU — install and operate

This guide documents the **current single-run entry point**, not retired ladder/Pinnacle campaigns. For the learning philosophy see [unified learning contract](UNIFIED_CAPABILITY_GROWTH.md); for launch blockers see [handover](../HANDOVER.md).

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

## Unified research: planned, not launchable

```bash
emu train --run-dir "$HOME/robot_lab/emu_runs/unified_candidate" \
  --profile sim2real --terrain-actor --terrain-step-max-m 0.04 \
  --unified-capabilities --steps 20000000 --envs 2048 --seed 733 \
  --device cuda --plan
```

That command prints a proposed configuration. **The matching live training invocation intentionally raises a pre-launch error** until true combined slope+step contact, independent held-out examination, GPU 2,048-world memory/PPO, checkpoint/resume/export, live viewer parity and cold-start review are accepted. Do not disable the guard to run overnight.

The six teacher-only cohorts and realistic **4×4 array-based 360-input sensor** use a separate actor contract; an old 220/232/360 checkpoint must not be blindly resumed under it.

## Checks and viewer

```bash
bash scripts/check.sh
python scripts/smoke_unified_focus.py --device cpu --envs 12
emu terrain-exam --help
emu viewer --no-open
```

CPU smoke/CI does not establish CUDA capacity or successful locomotion. The local viewer is read-only and **not authenticated**; do not expose its port publicly. See [PLAYGROUND](PLAYGROUND.md) for viewer behaviour and protected public release.

## Safety

No writes to `main`, showcase or approved Pinnacle without owner approval. Treat reward as training incentive, not real physical walking evidence. Simulated shaft work is not measured battery energy, and untested learned servo commands are **not hardware-safe**. Check source provenance and matched physical exams before any extended GPU campaign.
