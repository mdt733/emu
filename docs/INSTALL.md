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

## Overnight readiness — exact sequence (same source SHA)

A clean full CI run and green physical CPU smoke come first. Keep the Git
checkout and all checkpoints source-pinned, with each acceptance root NEW,
outside the repository. The first real overnight train command is **still
deliberately blocked**, even after these diagnostics pass.

**1. Full 2,048-world GPU PPO preflight** (no file writes):

```bash
cd ~/emu
git switch feature/v5023-capability-growth
git pull --ff-only
source .venv/bin/activate
git status --short                    # must be empty
git rev-parse HEAD                    # record this exact SHA
python scripts/smoke_unified_focus.py --device cuda --envs 2048 --full-update
```

If there is CUDA OOM, a non-finite gradient, excessive memory use or a
significant speed regression, **stop**. Fix the source (or deliberately
redesign the accepted batch size) rather than launching training regardless.

**2. Two-update 360-input checkpoint lifecycle** (bounded and explicit):

```bash
ROOT="$HOME/robot_lab/emu_runs/unified_accept_$(git rev-parse --short HEAD)_8"
for STAGE in first continue verify; do
  python scripts/smoke_training_lifecycle.py "$STAGE" \
    --root "$ROOT" --unified --envs 8 || break
done
```

After the small lifecycle passes, repeat with a **fresh** root and
`--envs 2048` to verify real-sized checkpoint/restore/export. The smoke
cannot request more than two updates even when explicitly opted in; it
will not make a trained walking policy.

**3. Exercise the actual exported 360 policy in the frozen examiner**:

```bash
emu capability-exam \
  --policy "$ROOT/tiny_smoke_actor.emupolicy" \
  --output "$ROOT/quick_physics" --device cpu --worlds 4 --quick
```

Set `ROOT` to the relevant successful lifecycle root. An untrained
two-update policy should generally fail walking tasks; this stage only
checks that array-only sensor decoding, real two-slab physics, scoring and
artifact loading function end-to-end. **Do not interpret its pass fraction
as earned walking competence.** Open the exported policy locally in the
viewer too.

**4. Explicitly approved 20M cold unified foundation — now launchable on this branch.**

Owner approval covers this **one foundation run only**, not six-teacher growth.
Source must be a clean, passing feature-branch commit; keep `main`, the
approved 260M Pinnacle, and all existing policies untouched.

```bash
cd ~/emu
git switch feature/v5023-capability-growth
git pull --ff-only
source .venv/bin/activate
git status --short   # must be empty
RUN="$HOME/robot_lab/emu_runs/unified_foundation20_$(git rev-parse --short HEAD)"
XLA_PYTHON_CLIENT_PREALLOCATE=false \
EMU_UNIFIED_FOUNDATION=cold20m-approved \
emu train --run-dir "$RUN" --profile sim2real \
  --terrain-actor --terrain-step-max-m 0.04 --unified-capabilities \
  --steps 20000000 --envs 2048 --seed 733 --device cuda
```

The actual PPO boundary is **20,054,016** environment steps because each
update adds 2,048 × 32 steps. Normal training logs/checkpoints and held-out
exams are retained; the frozen 10-panel final examiner uses **32 worlds per
panel**, not another five 2,048-world allocations on the 8GB GPU. Through
the foundation, specialist terrain exposure remains inactive.

If interrupted, **use the identical source SHA and flags**, append
`--resume auto` and keep the same `RUN` directory; never delete a partial
run. The explicit `EMU_UNIFIED_FOUNDATION` opt-in is required again.
A larger absolute `--steps` value, arbitrary warm-start, different sensor
contract or world count is **still rejected**.

Once the checkpoint exists, measure actual supported gait, command tracking,
falls and effort against the preserved `main` reference. Source changes
invalidate this particular run lineage; never silently reuse it.

**5. Later overnight capability-growth continuation — NOT unlocked.**

After the foundation is genuinely walking, review the source and request a
separate owner-approved growth launch. Long growth runs and public policy
promotion remain blocked. Do not change `main` or the showcase.

## Approved 20M → 40M gait-maturation continuation (feature branch only)

This is a NEW 40M absolute-target run, **resuming the exact recovered 20M
ancestor**. No scratch training and no overwrite of the original 20M files.
The opt-in authorises **only this 20M → 40M continuation**. No 60M+, public
showcase promotion, or unrestricted six-capability experiment is authorised.

Before starting, verify all four GitHub Actions jobs are green at the new
feature-branch source SHA. The **source transition is explicit**: the parent
was trained with `470cf7f...`; the new code validates its original saved
`training_config.json`, complete PPO/checkpoint sensor-geometry contract,
network + optimiser/normalizer/health/teacher state, original source SHA,
and exact 20,054,016-step boundary. It also prints a hash of the parent
checkpoint. The original checkpoint is read-only.

```bash
cd ~/emu
git switch feature/v5023-capability-growth
git pull --ff-only
source .venv/bin/activate
test -z "$(git status --porcelain)" || { echo "STOP: dirty source"; exit 1; }
PARENT="$HOME/robot_lab/emu_runs/unified_foundation20_470cf7f/checkpoints/final.pkl"
test -s "$PARENT" || { echo "STOP: recovered 20M final.pkl not found"; exit 1; }
RUN="$HOME/robot_lab/emu_runs/unified_maturity40_$(git rev-parse --short HEAD)"
XLA_PYTHON_CLIENT_PREALLOCATE=false \
EMU_UNIFIED_CONTINUATION=20to40-approved \
emu train --run-dir "$RUN" --profile sim2real \
  --terrain-actor --terrain-step-max-m 0.04 --unified-capabilities \
  --steps 40000000 --envs 2048 --seed 733 --device cuda \
  --resume "$PARENT"
```

Use a new, clean `RUN` path for this branch and retain the 20M ancestor.
The guard pins the handoff to that exact source/path and requires the
*previous and new training contracts to match except runtime fields and
the total-step target*. The resumed run stores its own checkpoint metadata
and a reference to the original ancestor.

After an interruption on the **same feature source SHA**, resume that
40M run with identical flags but `--resume auto`. This finds a validated
checkpoint in the NEW run rather than re-reading the original 20M
ancestor. **Do not change source revision while an existing 40M run
is in progress.**

This 20–40M phase keeps actual collision geometry/sensors available but
the terrain/step specialist *lesson exposure* and specialist teacher mix
are held at zero through the 40M gait-maturation boundary; flat regime
balance, command practice and posture/recovery learning continue.
Frozen 10-panel fixed exams use 32 independent evaluation worlds per
panel to fit the 8GB RTX 4060 Ti. The target PPO boundary is
**40,042,496** environment steps. Inspect walking and genuine command
tracking before separately authorising further growth.

## Safety

No writes to `main`, showcase or approved Pinnacle without owner approval. Treat reward as training incentive, not real physical walking evidence. Simulated shaft work is not measured battery energy, and untested learned servo commands are **not hardware-safe**. Check source provenance and matched physical exams before any extended GPU campaign.
