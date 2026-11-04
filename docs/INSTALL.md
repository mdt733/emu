# EMU — clean Windows 11 / WSL2 installation and recovery

This is the **single maintained Windows 11 / WSL2 setup guide**.
It describes a future-main-ready installation, not a source-pinned
experiment launcher. Installing never starts PPO, changes the current
public actor, or overwrites an existing run. Until a separate release
decision is approved, `main` still ships the 260M Pinnacle actor.
Use the exact experiment branch/SHA only for an already-approved
research run; never switch that checkout mid-training.

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
The old v5046 study paths remain in their [original handover]
(https://github.com/mdt733/emu/blob/experiment/v5046-75cm-four-tile-midpoint/HANDOVER.md),
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

## 2. Ubuntu: ONE Python, ONE EMU checkout, ONE venv

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

**Canonical v5046+ layout (three types of paths, two home-level roots):**

```text
~/emu/                           # ONE Git checkout and its .venv; no run data
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
`v5047-early-lips-s735` or `v5047-early-lips-control-s735`.
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

Use the checked-in `main` champion only as an unchanged baseline.
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
