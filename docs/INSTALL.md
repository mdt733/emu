# EMU — installation, validation and safe training (v5020 research)

**The working `main` branch and the approved public showcase are protected.**
These instructions apply exclusively to
`feature/v5020-coldstart-terrain`, which is experimental. See
[Architecture](EMU_ARCHITECTURE.md), [Research history](RESEARCH_HISTORY.md)
and [the current cleanup plan](V5020_CLEANUP_PLAN.md).

## 1. Set up one clean WSL2 checkout

Requires Python 3.13+, WSL2 (for Windows), Git, supported NVIDIA
drivers for CUDA training, and Node/npm for the browser viewer. Run as
the regular Linux user. The setup script installs the necessary Python
dependencies, builds/tests the browser and runs regression/doctor checks.

```bash
cd ~
git clone --single-branch \
  -b feature/v5020-coldstart-terrain \
  https://github.com/mdt733/emu.git emu-v5020
cd ~/emu-v5020
bash scripts/setup_wsl.sh --cuda13 --install-node
source .venv/bin/activate
```

Without a supported NVIDIA GPU, omit `--cuda13` and use
`emu doctor --no-require-gpu --no-mjx-smoke` for a limited read-only
Python inspection. Do not infer GPU acceptance from it.

**Already have a working checkout?** Use its own virtual environment,
not the separate viewer checkout's. If Git reports divergent local
history, do **not** force-pull or reset it blindly. Preserve that
working directory, then use a new folder and the single-branch clone
above. Source repository credentials are not required for public clone
unless your personal Git configuration overrides HTTPS authentication.

## 2. Check environment, source and experiment plan

```bash
pwd
git status --short --branch
python -m pytest -q tests/test_v5020_refork.py tests/test_v5020_campaign.py
emu doctor --v5015-smoke --domain-smoke
emu campaign-refork
emu campaign-refork --status
```

`campaign-refork` with no `--run` prints a plan and does **not**
launch training. The five teaching emphases are General, Turning,
Recovery, Efficiency and Steps. The default stop is `walk40`.

A source regression pass does not imply that a neural policy actually
learned to walk. See the independent examiner in
[`evaluation.py`](../emu/evaluation.py).

## 3. Inspect your actual foundation before starting a campaign

The new five-arm tournament resumes **a real, validated 312-input
foundation checkpoint** trained under the same environment count
(2048 or 4096). It will not synthesize one from the old 232-input
`main` policy.

Check the actual local run directory first:

```bash
find ~/robot_lab/emu_runs -path '*/checkpoints/*.pkl' -type f \
  | grep -E 'foundation|v5020' | sort
```

A final ~20M checkpoint is not necessarily the healthiest candidate;
inspect earlier surviving 12–22M snapshots and their fixed-evaluation
records. The user may already have a 4096-world foundation. **Do not
reuse a foundation from a different schema or another environment
count, and do not discard its original provenance.**

## 4. Guarded first overnight run

From a clean local checkout of the current feature ref, after reviewing the
foundation's fixed evaluation and ensuring GPU drivers are ready:

```bash
cd ~/emu-v5020 &&
test "$(git branch --show-current)" = "feature/v5020-coldstart-terrain" &&
git pull --ff-only origin feature/v5020-coldstart-terrain &&
bash scripts/overnight_v5020.sh --run
```

The script sources this checkout's own `.venv`, validates the current
312-input checkpoint and exact world count, runs focused regressions and
the GPU doctor, prints the current experiment plan, writes a dated log
**outside** the training output root, and starts only the first five-arm
20→40M generation. In the event of a rejected probe, the contender's
remaining steps are skipped. No 60M/80M continuations are launched.

To check everything without training: `bash scripts/overnight_v5020.sh --check`.
The default foundation checkpoint is the earlier fixed 4096-world lab's
`foundation20/checkpoints/final.pkl` if present. That is an **unreviewed
candidate**, not a guarantee of healthy gait; the script prints its
last recorded fixed exams. You may select a previously verified better
12–22M snapshot with `FOUNDATION_PATH=/full/path/to/checkpoint.pkl`.

Run it in a terminal that stays open or a **pre-existing** tmux session
if you intend to disconnect. Closing the WSL session may otherwise
interrupt training; this script does not silently detach. A compatible
interrupted campaign can resume only under the **same pinned source
commit, parent checkpoint and root**. Check progress in the logged output
and `walk40/*/probe_health.json`; a winner is recorded at
`walk40/scoreboard.md` and `walk40/selection.json`.

## 5. Manual training command (advanced)

This is an **example**, not a request to start training. Replace the
`--foundation` path with a confirmed, existing, healthy compatible
checkpoint. The run root must be fresh/empty.

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
emu campaign-refork --run \
  --foundation ~/robot_lab/emu_runs/v5020_lab_4096_fixed/foundation20/checkpoints/final.pkl \
  --envs 4096 --seed 733 \
  --screen-envs 32 --screen-horizon 300 \
  --stop-after walk40 \
  --root ~/robot_lab/emu_runs/v5020_refork
```

The five challengers share the same validated parent, each performs a
short collapse probe, surviving challengers continue towards 40M
absolute policy steps, and frozen actors face common independent tests.
A losing arm's weights are not averaged into the winning brain.

If a compatible source is **not** available, do not substitute an
arbitrary `.emupolicy` viewer asset: browser inference policies lack
the complete optimiser, normaliser and training-state lineage needed
for resume.

To continue to 60M or 80M, the coordinator requires the same pinned
source commit and campaign settings, an eligible prior winner, and
a compatible output root. A source-code change during an existing
campaign requires a *new root*, not a forced resume. Old runs can be
read and compared under their original revision.

## 5. Current repo maintenance checks

```bash
bash scripts/check.sh
```

This validates Python contracts, cheap doctor checks, supported CLI
help and the browser test suite. Real long-horizon GPU walking,
physics-domain robustness, trained policy quality and hardware-safe
servo dynamics are **separate acceptance tasks**.

See [V5020_CLEANUP_PLAN.md](V5020_CLEANUP_PLAN.md) for the gated
architecture/dead-code cleanup and [RESEARCH_HISTORY.md](RESEARCH_HISTORY.md)
for superseded command examples, experimental findings and known failures.

## 6. Troubleshooting boundaries

- `file not found: tests/test_v5020_refork.py`: you are probably in a
  different checkout (such as the viewer-only folder). Check `pwd`.
- `fatal: Not possible to fast-forward`: preserve your local branch and
  use a clean clone, rather than resetting your work.
- `ModuleNotFoundError: emu`: activate the checkout's own virtual
  environment and install it with `python -m pip install -e '.[dev]'`.
- CUDA out-of-memory: reduce exam/world settings as a separate *new*
  experiment, never re-label a 4096-world checkpoint as 2048-world.
- `--resume` rejected due to provenance: do not override the guard.
  Use the original source, or explicitly start a new experimental root.
