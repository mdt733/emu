# EMU — clean Windows 11 / WSL2 installation and recovery

**Updated 5 November 2026 for v5046.** This is the one maintained
installation guide. Installing never launches training, replaces the
released `main` policy or rewrites an existing checkpoint.
For the current six-servo GPU trainer, use an explicitly chosen feature
branch; `main` remains the preserved 260M showcase champion.

## 0. BEFORE wiping an old WSL distro

GitHub contains the source, **not** a trained checkpoint or your
20M-run metrics. The uploaded `results_bundle(20261104-201906).zip`
contains measurements and configuration **but not the neural policy
optimizer checkpoint**. Back up the entire run directory first:

- `~/emu_runs/v5046_20m_s735_solver2_ls4/`
  (especially `checkpoints/final.pkl`, `training_config.json`,
  `run_meta.json`, `metrics.csv`, `fixed_eval.csv`,
  `console.log`, any exported `.emupolicy` and saved reports).
- Any other `~/emu_runs/`, `~/robot_lab/emu_runs/`, `~/emu_gpu_audit/`,
  custom run roots, local uncommitted code, private credentials and
  non-EMU projects (e.g. Micromouse Lab, Meadow).
- If WSL hosts your Tailscale/SSH remote access, that installation and
  service will also be lost, even if the Windows Tailscale client survives.

**Safest backup:** Windows PowerShell, after **stopping training**:

```powershell
wsl --list --verbose
$dest = Join-Path $env:USERPROFILE "Downloads\EMU-WSL-backup"
New-Item -ItemType Directory -Force -Path $dest | Out-Null
wsl --shutdown
wsl --export Ubuntu (Join-Path $dest "Ubuntu-before-reset.tar")
Get-Item (Join-Path $dest "Ubuntu-before-reset.tar") |
    Select-Object FullName, Length
```

This creates a separate full Ubuntu export with the old run files,
environments, code and configuration. Ensure the TAR is nonzero,
there is enough disk space, and copy important checkpoints separately
to a safe location if practical. Do not delete or unregister Ubuntu
until the archive has been verified and the run directory is recoverable.
`wsl --unregister Ubuntu` **permanently destroys** that distro's files.
If you deliberately reset, run it as a separate, explicit action only
after backing up.

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

# Current experiment checkout. Swap only the branch when a new
# source-pinned promotion candidate is explicitly approved.
git clone --single-branch \
  --branch experiment/v5046-75cm-four-tile-midpoint \
  https://github.com/mdt733/emu.git "$HOME/emu"
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
python -m pytest -q tests
```

The v5046 development branch has **21 core pytest cases**, not 34.
The old 34-arm solver study was temporary research; its runner was
retired on the separate cleanup PR, retaining only measured findings.
There are also browser checks and one real-physics acceptance probe.
Do not disable these tests to shorten ordinary setup. They are separate
from costly first-use JAX compilation and frozen evaluation.

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

For **v5046's original 20M experimental recipe** only, while the
chosen branch remains `experiment/v5046-75cm-four-tile-midpoint`:

```bash
PYTHONPATH="$PWD" python scripts/run_v5046_20m.py plan

# Only for a NEW, EMPTY run directory — this starts real training:
PYTHONPATH="$PWD" python -u scripts/run_v5046_20m.py start
```

Its default output is `~/emu_runs/v5046_20m_s735_solver2_ls4/`.
Do not run `start` again if that folder exists. Use
`scripts/run_v5046_20m.py resume` only with the **same source
SHA and saved training configuration**. The 20M archive from
5 November was already completed: to inspect the trained bird,
**restore its original `final.pkl` and run artifacts** instead of
launching another 20M over the same directory.

A **future replacement-main run** should receive a freshly pinned
branch, recipe and run directory, with installation and acceptance
commands written alongside that run. Treat this 20M launcher as an
experiment, not as the general-purpose 140M or final release recipe.

## 4. Recovery, viewer and clean project layout

Suggested new WSL layout (one canonical location for each kind):

```text
~/emu/                     # one Git checkout + .venv (installed once)
~/emu_runs/<unique-run>/   # training configs, snapshots, final.pkl, metrics
~/emu_audit/               # temporary GPU research data, not trainer source
```

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
