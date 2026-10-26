# Installing and operating EMU

This is the **current 360-input research workflow**. It replaces the stale
v5023 checkout instructions that previously appeared here. The original
220/232-input Pinnacle route is a temporary compatibility/reference path,
not the default training procedure.

[Start page](../README.md) · [v5024 experiment](V5024_RUN_PLAN.md) ·
[release/retirement gates](RELEASE_READINESS.md) ·
[handover](../HANDOVER.md)

## 1. Fresh WSL2 installation, separate from a running run

From Ubuntu under WSL2, use a Linux filesystem directory. Prerequisites:
Python **3.13+** with `venv`, Git, compatible Node/npm, and for CUDA a
working NVIDIA GPU/WSL driver. The setup script installs dependencies and
builds the browser viewer but **never starts training**.

```bash
git clone --single-branch --branch feature/v5024-release-polish \
  https://github.com/mdt733/emu.git "$HOME/emu-polish"
cd "$HOME/emu-polish"
bash scripts/setup_wsl.sh --cuda13 --install-node
source .venv/bin/activate
emu doctor --no-require-gpu --no-mjx-smoke
```

Omit `--cuda13` for a CPU-only installation. When already installed, use
the existing checkout; do not clone over or reset local changes. No trained
policy or run directory belongs inside the repository.

**Critical:** the 3 × 60M experiment was started from
`feature/v5024-terrain-control` at
`d26ae0c2c4097d7c7124759197a94504abc44995`. Keep that existing
checkout at exactly this SHA for its entire interrupted-run resume
lifecycle. Do not switch it to `feature/v5024-release-polish`.

## 2. Software and physical acceptance

```bash
source .venv/bin/activate
bash scripts/check.sh
python scripts/smoke_unified_focus.py --device cpu --envs 12

# Optional, only on an appropriate NVIDIA GPU:
XLA_PYTHON_CLIENT_PREALLOCATE=false \
python scripts/smoke_unified_focus.py --device cuda --envs 2048 --full-update
```

The CUDA smoke runs a real production-shaped 2,048-world PPO update. It
does not train a long run or make a policy good at walking; it detects
compile, physical-simulation, optimiser and memory problems early.

A bounded checkpoint/export lifecycle can be tested under a **new output
root**, separate from live experiment directories:

```bash
ROOT="$HOME/robot_lab/emu_runs/accept_$(git rev-parse --short HEAD)_8"
python scripts/smoke_training_lifecycle.py first --root "$ROOT" --unified --envs 8 &&
python scripts/smoke_training_lifecycle.py continue --root "$ROOT" --unified --envs 8 &&
python scripts/smoke_training_lifecycle.py verify --root "$ROOT" --unified --envs 8
```

The lifecycle verifies persistence/normalisation/export, not locomotion.
Do not reuse an existing acceptance root.

## 3. v5024 3 × 60M controlled experiment — frozen code only

Use your **existing** `~/emu-v5024` checkout at the training SHA, not the
polish checkout. The launcher plans safely by default. The three arms
(`main_balanced`, `commandheavy_balanced`,
`commandheavy_steering`) each cold-start to 60M world steps at seed
733 and 2,048 worlds. The launcher runs CPU then 2,048-world CUDA preflight
before starting its sequential campaigns.

```bash
cd "$HOME/emu-v5024"
test "$(git rev-parse HEAD)" = "d26ae0c2c4097d7c7124759197a94504abc44995" || {
  echo "STOP: training source SHA mismatch"
  exit 1
}
test -z "$(git status --porcelain)" || {
  echo "STOP: dirty training checkout"
  exit 1
}
source .venv/bin/activate
python scripts/v5024_campaign.py                # preview ONLY
# The following opts into real training, after GPU/disk confirmation:
# python scripts/v5024_campaign.py --run
```

The actual run can be started inside `tmux` to survive a closed terminal:

```bash
tmux new -s emu-v5024
cd "$HOME/emu-v5024"
source .venv/bin/activate
python -u scripts/v5024_campaign.py --run
```

Detach: `Ctrl+B`, then `D`. Reattach: `tmux attach -t emu-v5024`.
**Do not launch a second copy if the first is already running.**
Campaign logs, checkpoints and the final `overnight_results.zip` stay under
`~/robot_lab/emu_runs/v5024_cold_steering/` (not Git).

### Interrupted run or completed-target extension

The campaign uses the same source/config/root on relaunch and automatically
requests `--resume auto` when a checkpoint exists. Verify the original
`tmux` session/process is no longer active **before** relaunching. Never
resume across a Git commit, change the `--foundation-steps`/schedule
or edit saved training metadata. A finished unified lineage may raise
only its cumulative target with matching source/configuration, but do not
silently extend a controlled 60M study before comparing its arms.

For a single-run 360-input cold learner, the entrypoint remains:

```bash
EMU_UNIFIED_CONTINUOUS=cold-growth-approved \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
emu train \
  --run-dir "$HOME/robot_lab/emu_runs/my_new_unified_run" \
  --profile sim2real --terrain-actor --terrain-step-max-m 0.04 \
  --unified-capabilities --foundation-mix main \
  --foundation-steps 20000000 \
  --steps 120000000 --envs 2048 --seed 733 --device cuda --plan
```

This is **plan only**. Remove `--plan` solely for a new, explicitly
intended GPU training lineage after checking its source and run-root
uniqueness. Its 20M foundation default intentionally differs from the
v5024 tournament's 10M setting.

## 4. Frozen physics exam (not training reward)

For an exported **360-input** policy from the new learner:

```bash
JAX_PLATFORMS=cpu emu capability-exam --terrain-matrix \
  --policy "/path/to/new360.emupolicy" \
  --output "$HOME/robot_lab/emu_runs/my_unique_terrain_exam" \
  --device cpu --worlds 8 --horizon 350
```

The physical matrix includes 1/2/3/5° uphill, downhill and cross-slope
contacts, plus 0.3/0.5/1/2 cm step-ups and downs. Evaluate stable supported
movement **separately** from commanded vx/vy/yaw tracking; report
gravity-induced drift rather than treating it as skilled locomotion.

The published Pinnacle uses the **232-input legacy contract** and cannot be
mislabelled a 360-input model. Historical exam routes remain temporarily for
reference comparison, then retire with the old public policy.

## 5. Local viewer

```bash
emu viewer-build
emu viewer showcase/champion.emupolicy --no-open
# http://127.0.0.1:8765/
```

The browser runs local MuJoCo physics, including configurable surface-lab
slopes and steps. Use `emu viewer --live --root /path/to/run --no-open`
to inspect a running source-matched policy in a **second terminal** without
mutating its trainer or checkpoints. The viewer's HTTP server is local and
unauthenticated; do not expose it publicly.

## Safety and source lifecycle

- A green CPU CI run does not establish full CUDA capacity, walking
  competence, checkpoint compatibility or servo hardware safety.
- Keep approved Pinnacle/showcase/Pages unchanged until a 360 policy wins
  appropriate independent exams and the owner approves promotion.
- Retire old actor schemas, CLI modes and tests *after* the new policy is
  accepted and the public release is replaced; do not maintain two systems
  forever merely because both are historically interesting.
- Preserve old policy bytes in Git history/release archive as a reproducible
  rollback, not as permanent code complexity.
