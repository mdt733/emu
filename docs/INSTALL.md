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
  --academy-stream --terrain-step-max-m .08 --outcome-graduation

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

"${TRAIN[@]}" --plan
"${TRAIN[@]}" --check-resume-only
EMU_UNIFIED_CONTINUOUS=cold-growth-approved "${TRAIN[@]}"
```

The independent CPU physical tests are also available via `python -m pytest -q tests/test_core_stair_teaching.py`. After the 2M fork, inspect actual height/footholds/chassis transfer, episode failures, gait retention, PPO terminal credit and the source-pinned report **before** increasing the target. A completed same-source lineage can be extended by repeating the same config with `--steps $((PARENT_STEP + 10000000)) --resume auto` in the **same** run directory and removing the external parent-hash arguments. Do not change the branch source commit when resuming an existing run.

`python -m emu.cli export-policy "$RUN/checkpoints/final.pkl" "$HOME/emu-data/actors/h40-outcome-2m.emupolicy"` exports only the inference actor for an independent frozen test. `emu prism-exam --mode step --lip-cm 5` samples a **frontier up to 5cm**, not a fixed exact 50mm cohort; inspect measured signed physical heights. Require separately demonstrated repeatable actual +50mm supported climbs against a frozen main actor before using `--outcome-stair-approved --outcome-approval-start-steps <absolute-step>` in a **new checkpoint fork**. That explicit, evidence-gated approval starts the 10M fade of dense stair shaping and independent downhill expansion. No automatic unseen PR is permitted to approve climbing.

