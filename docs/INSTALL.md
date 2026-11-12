# EMU — clean Windows 11 / WSL2 installation and recovery

This is the **single maintained Windows 11 / WSL2 setup guide** for
the **H40 120M mainline** (promoted in [PR #53](https://github.com/mdt733/emu/pull/53)).
Installing never starts PPO, changes the public GitHub Pages deployment,
or overwrites a run. The current `main` includes the actual H40
`showcase/champion.emupolicy`; its release approval is now recorded, but
public GitHub Pages deployment still needs successful verification and a
separate manual workflow dispatch. Preserve source-pinned training worktrees
and checkpoints when updating the ordinary `~/emu` checkout.

## 0. Before wiping a WSL distro (optional, your decision)

GitHub preserves source, history and public `.emupolicy` actors;
it **does not** contain the private optimiser/curriculum `final.pkl`
checkpoint. If an experiment is disposable, there is no need to keep
its checkpoint. Preserve any results you care about first, especially
`~/emu-data/runs/` summaries, exams, config and source SHA.

If you want a full safety export, run in Windows PowerShell **only
after training and WSL services are stopped**:

```powershell
wsl --list --verbose
$dest = Join-Path $env:USERPROFILE "Downloads\EMU-WSL-backup"
New-Item -ItemType Directory -Force -Path $dest | Out-Null
wsl --shutdown
wsl --export Ubuntu (Join-Path $dest "Ubuntu-before-reset.tar")
Get-Item (Join-Path $dest "Ubuntu-before-reset.tar")
```

Verify any export you intend to rely on before unregistering a distro.
Deleting WSL also removes Linux-side SSH/Tailscale and unrelated
project files; there is no automatic restoration from Git.
The old v5046 study paths remain in their [original handover](https://github.com/mdt733/emu/blob/experiment/v5046-75cm-four-tile-midpoint/HANDOVER.md),
not in the current run convention.

## 1. Windows: GPU driver and fresh Ubuntu

Use an up-to-date NVIDIA **Windows** driver compatible with WSL2,
not a Linux display driver installed inside Ubuntu. For JAX's pip
CUDA 13 wheels, upstream JAX presently calls for NVIDIA driver
version **580 or newer**. The RTX 4060 Ti supports the CUDA 13
architecture requirement (SM 7.5+).

```powershell
# After your deliberate Ubuntu reset; do not unregister if backups
# or another important WSL project have not been secured.
wsl --install -d Ubuntu
wsl --update
wsl --list --verbose
nvidia-smi
```

Complete Ubuntu's username/password initial setup, then in Ubuntu:

```bash
uname -m
nvidia-smi || /usr/lib/wsl/lib/nvidia-smi
```

Expect `x86_64`, WSL version 2 and an RTX 4060 Ti visible
inside WSL. **Never install Linux `nvidia-driver`, `cuda-drivers`
or generic driver meta-packages in WSL**. Pip CUDA wheels normally
need no additional Ubuntu CUDA toolkit.

## 2. Ubuntu: one main checkout, frozen training worktrees, one maintained venv

Ubuntu versions differ in bundled Python; the repo requires
**Python 3.13+**. Use `uv` to install user-local 3.13 rather than
changing Ubuntu's system Python.

```bash
sudo apt-get update
sudo apt-get install -y git curl ca-certificates pipx
pipx ensurepath
export PATH="$HOME/.local/bin:$PATH"
pipx install uv
uv python install 3.13
python3.13 --version

# Clone all normal remote branches: no more --single-branch tracking
# surprises when we deliberately switch between experiment and release.
git clone https://github.com/mdt733/emu.git "$HOME/emu"
cd "$HOME/emu"
git branch --show-current
git rev-parse HEAD

# Existing, maintained installer: local .venv, pip CUDA 13 JAX,
# EMU + pytest, Node 24 via nvm, viewer and core/browser checks.
bash scripts/setup_wsl.sh --cuda13 --install-node --check
source "$HOME/emu/.venv/bin/activate"
```

The install script never starts PPO or changes Git refs, and does
not need you to download a ZIP. It reads `pyproject.toml` and the
committed viewer lockfile. Use the Linux filesystem (`~/emu`),
**not `/mnt/c`**, for high-throughput simulations.

### Verify the exact accelerator, not merely `nvidia-smi`

```bash
cd "$HOME/emu"
source .venv/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python -m pip check
python - <<'PY'
import jax, mujoco
print("JAX:", jax.__version__, "MuJoCo:", mujoco.__version__)
print("GPU:", jax.devices("gpu"))
assert len(jax.devices("gpu")) >= 1
PY
emu doctor --require-gpu --no-mjx-smoke
# The --check installation step already ran the core and browser tests.
```

The maintained tests check physics, source/provenance, checkpoint
contracts and the browser. The old 34-arm, 140M, 20M and shared-parent
campaign scripts are **not production dependencies**; each original
recipe remains recoverable at its research Git SHA. GPU compilation and
frozen physical examinations are separate from ordinary Python checks.

### Existing checkout: safely recover from a diverged pre-squash `main`

The H40 promotion was **squash merged**. An existing local `main` may report
130 local commits and one different remote commit even though the desired
H40 source is now on GitHub. **Do not** force-pull, rebase the old 130 commits
on top of H40, or delete those commits. The following guarded one-time
recovery saves a named backup branch, an external self-contained Git bundle
and a comparison report **before** moving local `main`.

Run from WSL only, with no uncommitted or untracked working files:

```bash
(
  set -euo pipefail
  cd "$HOME/emu"
  git fetch origin
  if [[ "$(git branch --show-current)" != "main" ]]; then
    echo "STOP: ~/emu must be on local main" >&2; exit 1
  fi
  if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
    echo "STOP: stash/commit/inspect local changes first" >&2
    git status --short
    exit 1
  fi

  stamp="$(date +%Y%m%d-%H%M%S)"
  backup="backup/main-before-h40-$stamp"
  dir="$HOME/emu-data/git-backups"
  mkdir -p "$dir"
  git branch "$backup" main
  git bundle create "$dir/$stamp.bundle" "$backup"
  git log --oneline "origin/main..$backup" > "$dir/$stamp-local-only.log"
  git diff --stat "$backup" origin/main > "$dir/$stamp-tree-diff.txt"

  # Local-only commits remain on the backup branch AND in the bundle.
  git reset --hard origin/main
  git log -1 --oneline
  git status -sb
)
```

This operation **does not** alter `~/emu-h40-120m`,
`~/emu-h40-viewer` or any directory in `~/emu-data/runs`.
The before/after diff report is for reviewing whether any *local-only*
changes deserve a new, narrow PR. GitHub cannot examine commits held
only on this machine; retain the backup until reviewed. After alignment,
the expected `main` ancestor is H40 commit
`1b796144b128b326be1ac54f6a289ad87857cde6`, or a later
documented maintenance commit.

## 3. Prepare the next source-pinned training run

Do not create a second Python virtualenv named `emu-graduation` or
install an editable EMU package from a different checkout. The script
`scripts/smoke_gpu_carpet.py` verifies the import source path to
prevent the mixed-source error seen during v5046.

```bash
cd "$HOME/emu"
source .venv/bin/activate
git status --short
git rev-parse HEAD

# This command checks 2,048 CUDA worlds and a genuine full PPO update,
# but does not save a policy or start continuous training.
PYTHONPATH="$PWD" python scripts/smoke_gpu_carpet.py \
  --device cuda --worlds 2048 --frames 2 --ppo \
  --full-update --benchmark-updates 2 --prisms \
  --solver-iterations 2 --solver-ls-iterations 4
```

### The stable training and examination CLI

```bash
emu train --help
emu prism-exam --help
emu export-policy --help
bash scripts/check.sh --full
```

The production training entry point is `emu train` and the original
checkpoint safety contract lives in `emu/run.py`. To reproduce a
source-pinned experiment (including the v5048 shared-parent 50M/fork
tournament), check out **its original recorded Git SHA**, use its
committed run recipe, and keep its original source/config for every
resume. Do not treat a temporary launcher from an older research
commit as a permanent production script.

The deployment/release review is documented in
[RELEASE_READINESS](RELEASE_READINESS.md), separately from training.

## 4. Recovery, viewer and clean project layout

**Canonical H40+ layout (two home-level roots, optional frozen worktrees):**

```text
~/emu/                           # maintained main checkout + .venv; no run data
~/emu-h40-120m/                 # frozen training source (do not pull/reset)
~/emu-h40-viewer/               # optional separate viewer worktree
~/emu-data/
    runs/
        <study>-s<seed>/          # ONE directory owns ALL of a training run:
            training_config.json # source-pinned experiment settings
            run_meta.json        # Git SHA and physical model identity
            console.log          # one training transcript
            metrics.csv          # training progress
            fixed_eval.csv       # held-out physical assessment
            results.md           # generated summary in the same directory
            checkpoints/
                final.pkl        # only if you choose to retain the actor
    scratch/                      # throw-away GPU sweeps; can delete in bulk
```

A run identifier is a **specific immutable experiment recipe**, for example
the preserved `h40-1m-tof70-120m-s735` training run. New studies get
new names; never overwrite the H40 baseline.
Never put the Git branch name, a second virtualenv or arbitrary `results/`
folder inside the run root. Existing `emu train --run-dir`,
`emu report --root` and `emu prism-exam --output` accept explicit paths;
**all run-owned files, exports and exams should remain in that run directory**.

For a new experiment, create no directory manually; pass a fresh path to
the existing CLI, and let training own its contents:

```bash
cd ~/emu
source .venv/bin/activate
RUN="$HOME/emu-data/runs/<study>-s735"
# Substitute the exact approved source-pinned CLI recipe:
# emu train --run-dir "$RUN" <approved settings> --plan
# emu train --run-dir "$RUN" <approved settings>
# emu report --root "$RUN"        # evidence alongside its original run
```

No `~/robot_lab/emu_runs/`, `~/emu-graduation/`, `~/emu_runs/`,
`~/emu_gpu_audit/` or separate `overnight_results` roots for future
campaigns. Historical folders are not automatically migrated or deleted;
those paths still belong to their original source-pinned runs.

The repo's own `scripts/setup_wsl.sh` also installs Node 24 and
builds the viewer. After activation:

```bash
emu viewer showcase/champion.emupolicy --no-open --port 8765
# Open http://127.0.0.1:8765/ in Windows.
```

Use the checked-in H40 actor as the unchanged current baseline.
A `.pkl` checkpoint includes learned weights, normalizer, optimizer
and curriculum state; an `.emupolicy` is an inference export.
Never treat Git clone or a results CSV as a checkpoint backup.

For a reproducible restore, save `git rev-parse HEAD`,
`python -m pip freeze`, the exact run folder and the installer flags
in an external backup. Pin external package versions for a long
campaign once its preflight has passed. Don't modify or `git pull`
the checkout while that campaign is running; create another branch
or clone for parallel engineering work.

Troubleshooting:
- `No module named emu` / wrong source: activate `~/emu/.venv`,
  use `PYTHONPATH="$PWD"` and confirm Git SHA; never mix shared venvs.
- CUDA JAX unavailable: check driver, WSL2, `jax[cuda13]` and
  `LD_LIBRARY_PATH` (it can override pip's CUDA libraries).
- GPU OOM: no competing 2,048-world run on the 8 GB card;
  use `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
- Checkpoint source mismatch: preserve the original checkout
  and configuration; do not delete the manifest or force a resume.
- Missing saved policy: restore backed-up `final.pkl`; Git alone
  cannot recreate learned weights.

References: [JAX installation](https://docs.jax.dev/en/latest/installation.html),
[NVIDIA's CUDA-on-WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/),
[uv Python installation](https://docs.astral.sh/uv/guides/install-python/).


## Opt-in experimental H40 outcome graduation (draft PR #76)

This is **not yet promoted to main**. Follow the immutable-source workflow: keep `~/emu` unchanged, use a detached worktree, put output under `~/emu-data`, and do not confuse the checked-in 120M `.emupolicy` inference asset with a complete `final.pkl` PPO checkpoint. The latter must exist locally with its optimiser, normaliser and curriculum state.

```bash
cd ~/emu
git fetch origin
git worktree add --detach ~/emu-outcome-pr76 origin/feature/h40-outcome-stair-graduation
cd ~/emu-outcome-pr76
source ~/emu/.venv/bin/activate
export PYTHONPATH="$PWD" EMU_PHYSICAL_PROFILE=h40
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Minimal two-update GPU JIT/PPO acceptance (8 worlds, no claim of learned stairs).
EMU_UNIFIED_ACCEPTANCE=bounded-ppo-lifecycle python -m emu.cli train \
  --run-dir "$HOME/emu-data/runs/outcome-pr76-ppo-smoke" \
  --device cuda --envs 8 --steps 512 --seed 735 \
  --terrain-actor --unified-capabilities --terrain-carpet --carpet-prisms \
  --academy-stream --terrain-step-max-m .08 \
  --terrain-lesson coherent-careful --outcome-graduation

# Point at an EXISTING, complete, compatible H40 final checkpoint.
PARENT="$HOME/emu-data/runs/YOUR_VERIFIED_H40_RUN/checkpoints/final.pkl"
test -f "$PARENT" || { echo "Missing full PPO parent final.pkl"; false; }
read -r PARENT_STEP PARENT_SOURCE < <(python - "$PARENT" <<'PY'
from pathlib import Path
import sys
from emu.checkpoint_io import load_checkpoint
meta=load_checkpoint(Path(sys.argv[1]),validate=False)['meta']
print(meta['step'],meta['source_provenance']['git_commit'])
PY
)
PARENT_SHA256="$(sha256sum "$PARENT" | awk '{print $1}')"
RUN="$HOME/emu-data/runs/h40-outcome-pr76-s735-2m"

TRAIN=(python -m emu.cli train
  --run-dir "$RUN" --device cuda --envs 2048 --seed 735
  --steps "$((PARENT_STEP + 2000000))"
  --profile sim2real --terrain-actor --unified-capabilities
  --terrain-carpet --carpet-prisms --academy-stream
  --terrain-step-max-m .08 --terrain-lesson coherent-careful
  --growth-recipe main-mature --athletic-rehearsal-share .15
  --retention-share .20 --science-step-velocity-relief .80
  --outcome-graduation --outcome-walk-start-steps "$PARENT_STEP"
  --resume "$PARENT"
  --research-parent-source "$PARENT_SOURCE"
  --research-parent-sha256 "$PARENT_SHA256")

if "${TRAIN[@]}" --plan && "${TRAIN[@]}" --check-resume-only; then
  EMU_UNIFIED_CONTINUOUS=cold-growth-approved "${TRAIN[@]}"
fi
```

If this draft branch is updated before the *first* 2M run, refresh the
existing **detached**, clean worktree explicitly (not `git pull`):

```bash
cd ~/emu
git fetch origin
cd ~/emu-outcome-pr76
if [ -z "$(git status --porcelain)" ]; then
  git switch --detach origin/feature/h40-outcome-stair-graduation
  git rev-parse HEAD
else
  echo "Worktree has local edits; not switching."
fi
```

The original 120M `training_config.json` may contain neutral retired keys
`cloud_final_probes`, `cloud_probes_per_eval`, and
`science_reset_step_evidence_on_fork`. Only a source- and SHA256-verified
**external research fork** accepts these when they are exactly `0, 0, false`.
Unknown or non-default retired fields fail closed; an ordinary same-run resume
still rejects the older schema. Do not edit the parent config or checkpoint.

The independent CPU physical tests are also available via `python -m pytest -q tests/test_core_stair_teaching.py`. After the 2M fork, inspect actual height/footholds/chassis transfer, episode failures, gait retention, PPO terminal credit and the source-pinned report **before** increasing the target. A completed same-source lineage can be extended by repeating the same config with `--steps $((PARENT_STEP + 10000000)) --resume auto` in the **same** run directory and removing the external parent-hash arguments. Do not change the branch source commit when resuming an existing run.

`python -m emu.cli export-policy "$RUN/checkpoints/final.pkl" "$HOME/emu-data/actors/h40-outcome-2m.emupolicy"` exports only the inference actor for an independent frozen test. `emu prism-exam --mode step --lip-cm 5` samples a **frontier up to 5cm**, not a fixed exact 50mm cohort; inspect measured signed physical heights. Require separately demonstrated repeatable actual +50mm supported climbs against a frozen main actor before using `--outcome-stair-approved --outcome-approval-start-steps <absolute-step>` in a **new checkpoint fork**. That explicit, evidence-gated approval starts the 10M fade of dense stair shaping and independent downhill expansion. No automatic unseen PR is permitted to approve climbing.

## Follow-on H40 30M outcome school from the completed 122M checkpoint

The initial 2M research fork ended at 122,093,568 steps, source
`51c1d60ea734898496f0d26cbb7d946cbab22ca8`. It is an immutable
external parent, **not** an existing run to extend with newly changed code.
Create a NEW run directory, source-pinned to the updated PR #76 commit.
The new campaign target is `PARENT_STEP + 30000000` (about 152.1M
absolute world steps), 2,048 worlds, seed 735, with independent
held-out evaluations every 10M world steps.

Use the earlier explicit cross-source fork recipe, replacing the parent with:

```bash
PARENT="$HOME/emu-data/runs/h40-outcome-pr76-s735-2m/checkpoints/final.pkl"
RUN="$HOME/emu-data/runs/h40-outcome-30m-from122m-s735"
EXTRA=(--fixed-eval-every-steps 10000000)
```

Require `PARENT_STEP=122093568` and source
`51c1d60ea734898496f0d26cbb7d946cbab22ca8` before starting.
Use the *original* walking fade start `--outcome-walk-start-steps 120061952`.
Add `${EXTRA[@]}` to TRAIN before the plan/check-resume/train gate.
Derive the full parent source and SHA256 from the local immutable `final.pkl`.
Never rerun in the completed 2M directory or disable lineage checks.

Unlike the initial 2M run, successful nursery resets now bootstrap value
from the next reset observation (continuing task); physical falls remain
absorbing and timeouts use their pre-reset observation. The GAE done mask
still blocks gradients from separate episodes. This removes the old
artificial loss of all future walking value, but **does not promise positive
advantages**. Inspect the reported next-birth bootstrap values and
normalised completion advantages throughout the campaign.

Nursery reward and six bounded high-water progress payments are now
reported per rollout and cumulatively across ALL PPO updates via
`ppo_stair_audit_nursery_*`, not merely sampled every 10 updates.
Do not approve stair graduation or merge to main without independent
same-world frozen +50mm testing and retained core walking capability.

## H40 progressive school — ONE cold training run (draft PR #77)

Do **not** resume the 120M main actor, the 122M PR #76 fork or the 30M
adaptation. This is a new **zero-step** lineage on feature branch
`feature/h40-progressive-skill-school`. The trained actor shape, physics,
optimizer and observer remain the same: 360 inputs, one PPO, eight physical
convex prisms, H40 rated-torque motors, 2,048 shared GPU worlds.

### Learning stages (no automatic clock-based skill claims)

- 0–20M: raw standing/balance/gait foundation, no specialist STEP births.
- 20–60M: growing command, slope, resilience and control specialists;
  STEP is excluded, ordinary step exposure remains **zero**.
- 60–80M+: walk gait/phase shaping gradually fades over 20M, modulated by
  genuine core competence. After 80M, a core competence of at least
  **0.55** is required to lock walking graduation. Until earned, 5cm
  school is held off even if the counter passes 80M.
- Upon walking graduation: real +50mm (+/-4%) prism stair nursery, 15–30cm
  variable distance, 10% birth share rising toward 55%, independent
  real foot-plant and stable supported body-transfer reward.
- Proof gate: two **separate 100+ actual near-50mm uphill attempts**
  with >=70% verified supported climb success *in each window*.
  Failed windows reset the consecutive pass streak.
- AFTER proof only: phase out stair shaping over 20M, gradually introduce
  downhill episodes and expand physical uphill step heights toward
  the 80mm geometry ceiling. Keep rehearsing easier 50mm lips.

These are TRAINING curriculum gates, **not** held-out demonstrated skills.
Independent same-world matched frozen exams and walking retention remain
required before claiming +50mm capability or merging to main.

### Overnight run from scratch

Use a new clean detached worktree from this draft branch, and run `--plan`,
`--check-resume-only`, one 8-world CUDA smoke and the 180M continuous learner.
Never change branch source or reuse an existing run directory in the middle.

```bash
set -euo pipefail
cd ~/emu
git fetch origin
git worktree add --detach ~/emu-school-pr77 origin/feature/h40-progressive-skill-school
cd ~/emu-school-pr77
source ~/emu/.venv/bin/activate
export PYTHONPATH="$PWD"
export EMU_PHYSICAL_PROFILE=h40
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset EMU_UNIFIED_ACCEPTANCE

RUN="$HOME/emu-data/runs/h40-progressive-school-180m-s735"
TRAIN=(python -m emu.cli train
  --run-dir "$RUN" --device cuda --envs 2048 --seed 735
  --steps 180000000 --profile sim2real
  --terrain-actor --terrain-step-max-m .08
  --unified-capabilities --terrain-carpet --carpet-prisms
  --academy-stream --terrain-lesson coherent-careful
  --teaching-schedule staged --growth-recipe main-mature
  --athletic-rehearsal-share .15 --retention-share .20
  --science-step-velocity-relief .80
  --fixed-eval-every-steps 10000000
  --progressive-school)

"${TRAIN[@]}" --plan
"${TRAIN[@]}" --check-resume-only
EMU_UNIFIED_ACCEPTANCE=bounded-ppo-lifecycle python -m emu.cli train \
  --run-dir "$HOME/emu-data/runs/h40-progressive-smoke-s735" \
  --device cuda --envs 8 --steps 512 --seed 735 \
  --profile sim2real --terrain-actor --terrain-step-max-m .08 \
  --unified-capabilities --terrain-carpet --carpet-prisms \
  --academy-stream --terrain-lesson coherent-careful \
  --teaching-schedule staged --growth-recipe main-mature \
  --progressive-school
EMU_UNIFIED_CONTINUOUS=cold-growth-approved "${TRAIN[@]}"
```

Resume after a crash **only with the exact same source commit**, same
run folder and same flags; change only `--resume auto` and omit any external
parent-hash flags (there should be none). The trainer checks source SHA and
checkpoint schema, rejects changed learning knobs, and logs `SCHOOL` stage
transitions. Stage and physical mastery evidence are persisted in curriculum
checkpoint state. Requested target 180,000,000; PPO boundary 180,027,392.

Expected report diagnostics: `school_stage`, `school_walk_alpha`,
`school_stairs_enabled`, `school_stair_mastered_step`,
`school_up_good_windows`, `academy_height_up_m`,
`ppo_stair_audit_nursery_*` and actual freeze-evaluation retention.

## Future-run live console dashboard (display-only branch)

The parent training process now has an ANSI terminal dashboard that redraws
one screen instead of printing three long metrics lines per interval.
**The trainer, PPO, actor observations, curriculum, checkpoints, physics
and frozen evaluations are unchanged.** Original child output remains
unaltered in RUN/console.log; metrics.csv and fixed_eval.csv are canonical.

Choose display with EMU_CONSOLE_MODE in the environment for a FUTURE run:

| Mode | Effect |
| --- | --- |
| auto (default) | Dashboard on real interactive ANSI-capable terminal, otherwise compact |
| dashboard | Request dashboard on a real TTY; compact if not a TTY |
| compact | Append-only status milestones, including signed 5M trends |
| verbose | Historical long lesson/speed progress lines |

This is a parent-only environment variable, not a training-config or
checkpoint identity parameter. Do not change the source of an active
300M lineage; a later source revision requires an explicitly authorised
new-run research fork, never swapping source SHA in place.

The screen displays age/stage/progress, walking reward/falls/episode
length/core competence with signed Δ~5M-step differences, stair success
rate over the most recent 5M compared with the *preceding* 5M, birth
counts/physical milestones/frontiers, learner/wall throughput, and
the independent frozen walking score change since the preceding exam.

Stair success percentages require at least 20 sampled completed Academy
episodes spanning two or more logged rows per window. These are sparse
training-rollout samples, not exhaustive episode counts and NOT a frozen
stair exam. The screen says so. Missing or stale trend baselines show --.

No extra dependencies, GPU metrics transfers, or changes to the
results CSV schema. The dashboard uses standard ANSI terminal sequences
and restores the cursor on exit. The full raw log retains warnings,
tracebacks and training diagnostics. An unsuccessful child process still
emits its diagnostic tail when it exits.

## Refined stair school: a source-verified fork of 180M

The `--refined-school` switch requires `--progressive-school`. It is a
versioned, checkpointed upgrade to the SAME six-role PPO training process,
not a new environment, agent, optimiser or fork-specific trainer.

Intended research parent: source `853d19a63502413bb9c05049b9f9ac5ac1afad4c`,
exact validated 180,027,392-step H40 final.pkl. The older 300M result stays
untouched, and the newly trained successor must use a new run directory.
Cross-source forks require the original 180M bytes in a parent
`checkpoints/final.pkl`, a matching `training_config.json` adjacent to
that checkpoints directory, explicit 40-character `--research-parent-source`,
explicit checkpoint `--research-parent-sha256`, and an independent new
source commit. The parent configuration may be recovered from the archived
180M bundle, or derived from the unchanged 300M configuration by adjusting
only its original `steps` to 180000000. Never reuse the active 300M directory.

Stages are *physical-evidence-driven*, not specified at fixed 300M/320M
milestones. The existing 180M policy has earned walking graduation but no
5cm stair certification:

1. **Coached 5cm**: two consecutive 100-episode windows with >=70% true
   supported 15-frame uphill completions. Keep +50mm, no descent.
2. **Remove stair shaping** over 20M steps while holding **exactly +50mm**.
3. **Independent 5cm training proof**: after stair shaping is fully zero,
   discard coached-era evidence and require two *fresh* 100-attempt
   near-frontier windows with >=65% supported completion. No growing
   stair height before this pass.
4. **Slow uphill growth**: half of *actual physical uphill Academy births*
   remain at 50mm (+/- 4% tolerance); 35% probe the current frontier;
   15% bridge between. At most 0.5mm frontier advancement per 1M steps.
   At least two fresh 100-attempt uphill >=75mm windows with >=65%
   success are required before introducing descent.
5. **Downhill nursery**: freeze uphill at <=75mm while new downhill
   conditions enter at 10% of Academy births, with downhill frontier
   limited to 50mm. Two 100-attempt >=65% near-50mm downhill windows
   independently certify descent, then unlock signed heights up to 80mm
   and 20% downhill Academy share.

Regression handling: if two uncoached 5cm windows each succeed on fewer
than 40% of real completed attempts, briefly restore 25% of the dense
stair-only shaping for 5M steps, then fade it back over 5M steps; fresh
unassisted proofs are still required. After graduation, a new window of
100 *exact 50mm* births below 45% supported completion freezes harder
uphill promotions for at least 10M steps; already learned PPO state
and ordinary walking rewards are never reset or overwritten.

All windows, steps, state transitions, hold periods and proof counts are
serialized in `curriculum_state` with the policy/optimizer. The special
import of the old 180M governor is allowed only when `--refined-school`
uses a declared cross-source research parent; new proof counters begin
empty. The existing physical test checks the counter on true MJX prisms.

**Evidence scope:** these are source-grounded *on-policy physically
supported attempts*, not independent stair exams. Broad fixed evaluations
continue checking walking retention every 10M steps, but training promotion
must also be followed by a separate frozen true-prism stair exam before a
model is declared a reliable real-world stair walker. Do not interpret
Academy completions as multi-step course completion.

Use `EMU_CONSOLE_MODE=auto` for the updating terminal, or `compact` and
`verbose` for append-only output. The source SHA, physical profile and
parent checkpoint must always be verified before launching the fork.


## Experiment B — simultaneous uphill/downhill skill school

A new source-verified research fork of the *original* 180M walking
foundation. **Do not use the completed 420M policy as the parent** for the
controlled A/B result. A and B must share the same source checkpoint bytes.
Do not alter the H40 shin/torso/feet collision topology: B intentionally
retains the original **feet-only eight-prism MJX** solver. Shin collisions
are a separate proposed C arm, contingent on memory and throughput.

New switch: `--bidirectional-school` **requires** both
`--progressive-school` and `--refined-school`. Normal refined-school
behaviour is identical without this flag.

- Walk graduates at the established core/phase gate; Academy births now
  split **50% up / 50% down** from that point, inside the existing Academy
  allocation. All six other specialists retain their ordinary mix.
- Uphill starts at **5cm**, earns coached 2x100 @70%, phases out guidance
  over 20M steps, earns fresh uncoached 2x100 @65%, then expands slowly
  toward 7.5cm with 5cm retention. No change to the original uphill
  reward.
- Downhill starts alongside uphill at **2cm**, with an *independent*
  landing/weight-transfer teacher. It needs coached 2x100 @70%, fades
  coaching over its *own* 20M steps, then fresh uncoached 2x100 @65%.
  Only then can the downhill frontier expand by at most 0.5mm/1M toward
  5cm. Two uncoached 100-encounter windows @65% certify 5cm descent.
  Its own real 2cm anchor cohort is retained throughout growth.
- Both directions independently split their own birth distribution into
  **50% stable proven-height anchor, 35% frontier, 15% intermediate**,
  with the trainer's existing +/-4% physical-lip jitter.
- Critical fixed physical bug: the earlier uphill nursery required
  abs(lip)>3cm to arm teacher rewards/terminal successes. This experiment
  explicitly allows >1.2cm **only for the paired recipe**, otherwise a
  2cm DOWN lesson would have no graduation path.
- Downhill shaping rewards the actual lower-foot reach, the independently
  detected real contact on the *assigned* lower tile, and safe body
  transfer. No reward for upward foot lift or falling over the edge.
  The same 15-control-frame supported and upright absorbing terminal is
  required; moving 10cm past a lip without safe contact is not success.
- Downhill and uphill teaching are separate teacher-only task scalars;
  the **single 360-input actor, actions, optimizer, walking reward and
  8-prism solver remain unchanged**.
- School gates, 20M fades, proof windows, retention holds and growth
  timestamps are stored in curriculum checkpoints. The pinned original
  180M research parent imports with zero fabricated DOWN evidence.

### Launch on WSL2 (separate foreground console)

First preserve the original 180M `final.pkl` and its original
`training_config.json` using the **Refined stair school** archive
instructions above. An archive from the 420M result is *not* the same
initial condition. The checked script refuses an existing target run.

```bash
cd ~/emu
git fetch origin feature/h40-bidirectional-stair-school
git worktree add --detach ~/emu-school-bidirectional \
  origin/feature/h40-bidirectional-stair-school

cd ~/emu-school-bidirectional
source ~/emu/.venv/bin/activate
bash scripts/run_h40_bidirectional_school_wsl.sh
```

The script performs `--check-resume-only` before opening its new
`~/emu-data/runs/h40-paired-school-420m-s735` path and checks the
whole parent SHA256 and original source. For reproducibility pin the
worktree to the PR's tested HEAD SHA once checks pass. Result and
checkpoint names never conflict with the previous 420M training run.

**Acceptance:** 20M from 180M should show real 20mm downhill outcomes,
independent downhill guidance (not always zero), +50mm uphill exposure
and stable walking evaluation. A full 420M comparison requires fixed
signed-prism held-out exams, retention matrices, and a measured actual
downhill encounter count. The existing metrics remain on-policy proof,
not independent certification. The new per-step teacher scalar does
not enter the exported actor sensor/observation contract.

