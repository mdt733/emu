# tinyEmu GPU campaign — 20M foundation, four +60M hypotheses

**Draft PR #3 / experimental branch only.** Accepted `main` at `40c5d0c` remains the known-working training baseline. Use a **separate worktree and brand-new output roots**; do not switch an in-progress long-run checkout, reuse earlier checkpoint directories, edit the champion, or manually dispatch Pages.

This recipe was selected to learn whether the simpler tinyEmu architecture can preserve walking while varying one meaningful condition at a time. There is **no assurance of walking quality** from simulation appearance or current step-lift scores.

## The exact training budget

Steps below are *absolute per-policy environment steps*, not PPO updates or wall-clock seconds:

| Stage | Lineage length | Added steps for that stage | Status |
|---|---:|---:|---|
| Shared foundation | 20M | 20M once | Review before forks |
| Four alternatives | 80M each | **60M per fork** from the same immutable 20M parent | Compare all four; pause |
| Optional two finalists | 100M each | 20M per selected fork | Only after read-only review |
| Optional one finalist | 120M | 20M | Only after read-only review |

**Total to the first real comparison: 260M environment steps** across all runs (20 + 4×60). The complete optional funnel consumes **320M** (260 + 2×20 + 1×20). With 2048 worlds and 32-step PPO rollouts, each run rounds *up* to the next full 65,536-step update boundary; hence displayed checkpoint steps differ slightly from nominal.

### Hypotheses (unchanged except for the experiment durations)

The 20M foundation uses the already-defined v5015 foundation profile with shadow teaching. Every 80M fork switches to the *same* v5015 mature base task/optimizer distribution and the **same source commit**.

- `baseline`: accepted low-dose 10% graph teaching, reference (no experiment knob override).
- `shadow`: field shadow/no teaching (mode `shadow`, influence `0`).
- `slope`: low-radius gravity-equivalent terrain domain practice (`domain_terrain_radius = 0.12`).
- `pitch15`: wider requested chassis-pitch command limit (15°, versus 8°).

No experiment changes the XML, actor inputs, PPO architecture or reward weights. The `slope` branch varies the learning physics **on purpose**; it is not a UI terrain change. The common held-out 15° pitch screening uses identical cohorts and environments for all four candidate policies, independent of what each was taught.

## 1. Make an independent checkout on WSL

Use an existing clone (`~/emu` in this example), from a terminal **not running the long trainer**:

```bash
cd ~/emu
git fetch origin
git worktree add --detach ~/emu-tiny-gpu origin/refactor/tiny-emu
cd ~/emu-tiny-gpu
git status --short
git rev-parse HEAD
```

`git status --short` must be empty. **Never** `sudo` the worktree/venv or recursively chown an existing experiment directory. If the target worktree directory already exists, choose a new one rather than deleting it. Keep this source commit pinned for the campaign; source changes require **a fresh output root**.

## 2. Install, validate and do a bounded real GPU lifecycle

```bash
cd ~/emu-tiny-gpu
bash scripts/setup_wsl.sh --cuda13 --install-node
source .venv/bin/activate
emu doctor --v5015-smoke --domain-smoke

SMOKE="$HOME/robot_lab/emu_runs/tiny_accept_001"
python scripts/smoke_training_lifecycle.py first --root "$SMOKE"
python scripts/smoke_training_lifecycle.py continue --root "$SMOKE"
python scripts/smoke_training_lifecycle.py verify --root "$SMOKE"
```

Use `--cuda13` only where the NVIDIA driver supports CUDA 13 JAX wheels. Fresh setup may already have been completed in the worktree; don't replace its `.venv` to “fix” a failed training run. The acceptance sequence trains exactly **256 then 512 cumulative environment steps** in isolated scratch directories; verifies GPU PPO, held-out exam, checkpoint ancestry, observation normaliser, no-op complete resume and browser actor export. It now also **fails** if the seven servo/spring telemetry columns are missing or nonfinite.

**STOP HERE** if either doctor or any acceptance stage fails. Read `$SMOKE/first/console.log` and `$SMOKE/continued/console.log`; don't jump to the 20M foundation. Only the real GPU lifecycle—not CI—confirms MJX stepping with the new physics metrics.

## 3. Inspect plan, then train only the shared foundation

```bash
RUNROOT="$HOME/robot_lab/emu_runs/tiny_fourway_001"
emu ladder --stage plan --root "$RUNROOT" --envs 2048
emu ladder --stage foundation --root "$RUNROOT" --envs 2048 --device cuda
```

Inspect `$RUNROOT/foundation20/checkpoints/final.pkl`, `$RUNROOT/foundation20/fixed_eval.csv`, `$RUNROOT/foundation20/metrics.csv` and `$RUNROOT/foundation20/console.log`. A validated 20M handoff is a *precondition*, not a guarantee of competent walking. Keep this final checkpoint immutable: the ladder records its SHA-256 in each arm's ancestry.

## 4. Four independent +60M continuations and common-screen comparison

Only after reviewing the foundation:

```bash
emu ladder --stage auto --stop-after round1 \
  --root "$RUNROOT" --envs 2048 --device cuda \
  --screen-envs 256 --screen-horizon 350
```

This runs four branches **sequentially**, not four simultaneous GPU processes. It reuses the completed foundation, resolves/checks each branch's parent and saved config, runs 20M→80M independently for `baseline`, `shadow`, `slope` and `pitch15`, then scores *all four* on common fixed held-out panels. If interrupted, re-run the **identical command from the identical clean source commit**; valid partial checkpoints resume, completed branches are not retrained, and scored checkpoint hashes prevent stale comparisons.

Results:
- **`$RUNROOT/results.md`** — full campaign-level report, auto-generated when a round is scored
- **`$RUNROOT/results_bundle.zip`** — one shareable ZIP with underlying branch CSV/JSON evidence, without checkpoints
- `$RUNROOT/results_data.json` — complete machine-readable summary
- `$RUNROOT/round1_scorecard.md` (human view)
- `$RUNROOT/round1_scoreboard.json` (machine-readable common exam, eligibility and checkpoint hashes)
- `$RUNROOT/round1/<arm>/metrics.csv` (physical effort and PPO training behaviour)
- `$RUNROOT/round1/<arm>/checkpoints/final.pkl` (untouched separate checkpoint lineages)

**Do not treat the highest survival-weighted score as proof of the best gait.** Current foot-lift `step_evidence` and height-based `support` are proxies, not verified completed strides, true ground contact force, motor electrical energy or calibrated thermal risk. Also compare command tracking and robustness; evaluate locomotion visually only as supporting evidence. If no arm is eligible, stop rather than bypassing the gate.

## Share your entire experiment

To regenerate a complete report, including any completed 80M arms, run:

```bash
emu report --root "$RUNROOT"
```

Upload **results_bundle.zip** to ChatGPT instead of collecting individual CSV
files. It includes the foundation and every branch, full training logs, common
screens, reward contributions, servo/spring data and checkpoint SHA-256 IDs.
It excludes checkpoint binaries and console.log files.

**Important for experiments already in progress:** do NOT update the training
worktree's Git commit merely to get this exporter. A fresh report-only worktree
can run the standard-library-only command
`python3 /path/to/new-worktree/emu/results.py --root "$RUNROOT"`
without installing EMU, CUDA or JAX. It reads completed older runs without
changing provenance, policies, scoreboards or the original training checkout.

## 5. Optional finalists (not automatic by default)

After reading the scorecard and discussing performance, select exactly **two eligible** round1 arms to continue to 100M, then manually screen and choose one eligible 120M finalist. Alternatively `emu ladder --stage auto` uses existing eligibility-based promotion; only choose this once you trust the measured gates. The published 60M policy is *not* part of this experiment and never gets overwritten.

## What has and has not been proven

GitHub's CPU and browser CI can verify contracts, tests and static packaging. It **cannot** claim RTX 4060 Ti throughput or true MJX motor-loading behaviour. Only the bounded WSL run above can establish that the current source commit passes its full trainer lifecycle. Never paste an old GPU PASS onto a different commit.

The broader goal remains **simpler and better than both accepted main and the exploratory Pass 9 ZIP**. The experiment asks which existing features are actually useful, instead of guessing based on file size or a single promising video.
