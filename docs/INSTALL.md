# EMU — fresh install and recovery (Windows 11 / WSL2 Ubuntu / NVIDIA)

**Reviewed: 31 October 2026.** This is the maintained rebuild procedure.
The protected public `main` actor (232 inputs) and the experimental
360-input terrain actor are **different** policies. The current research
study is [v5031 verified ascent](V5031_VERIFIED_ASCENT.md), run on the
*frozen* source `feature/v5031-verified-ascent` at commit
`93cb5cf5a541416ab4366140b2205aca6bfb89a7`.

No step below launches training or replaces the published policy.
Never install into, change, pull, or overwrite an **active** training
checkout or virtual environment. Fresh machine setup uses a NEW directory,
a NEW local `.venv` and results stored OUTSIDE Git.

[Project overview](../README.md) ·
[Current handover](../HANDOVER.md) ·
[Release rules](RELEASE_READINESS.md) ·
[Official JAX installation guide](https://docs.jax.dev/en/latest/installation.html) ·
[NVIDIA CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/)

## 1. Windows host — GPU and WSL

1. Install a current NVIDIA **Windows** display driver that supports CUDA
   on WSL2. For JAX CUDA 13, NVIDIA/JAX document a driver **580+**
   requirement; verify the current JAX/driver guidance when rebuilding.
   The RTX 4060 Ti used for EMU supports CUDA 13.
2. In **administrator PowerShell**, install/update WSL2 and Ubuntu:

   ```powershell
   wsl --install -d Ubuntu
   wsl --update
   wsl --list --verbose
   nvidia-smi
   ```

   Restart Windows if requested. Confirm Ubuntu is WSL **version 2**.
   Configure the Ubuntu username and password on its first launch.
3. Enter Ubuntu with `wsl -d Ubuntu` and check:

   ```bash
   uname -m
   nvidia-smi || /usr/lib/wsl/lib/nvidia-smi
   ```

   Expect `x86_64` and a visible NVIDIA GPU. A `CUDA Version` reported by
   `nvidia-smi` is the driver's *capability*, not proof that JAX has been
   installed. **Do not install a Linux NVIDIA display driver in WSL**,
   or install a generic `cuda-drivers` meta-package: WSL uses the driver
   supplied by Windows. With JAX's pip-installed CUDA wheels you
   ordinarily need **no separate Linux CUDA toolkit**.

## 2. Ubuntu — Git, Python 3.13 and Node 24

Use the Linux filesystem (for example `~/emu-research`), not `/mnt/c`,
for a high-throughput MJX checkout or results.

Ubuntu 24.04 ordinarily ships Python 3.12, whereas `pyproject.toml`
requires **Python 3.13+**. This reproducible route installs a user-local
Python 3.13 without replacing Ubuntu's system Python. On a system that
already has working `python3.13`, skip the `pipx`/`uv` bootstrap.

```bash
sudo apt-get update
sudo apt-get install -y git curl ca-certificates pipx
pipx ensurepath
export PATH="$HOME/.local/bin:$PATH"
pipx install uv
uv python install 3.13
python3.13 --version
```

`uv` fetches a supported Python 3.13 distribution. Do not replace
`/usr/bin/python3` or use `--break-system-packages`. Python 3.13
installation can alternatively be provided by your distribution; the
version check above is what matters.

The repo pins the viewer's Node major version in `.nvmrc` (**24**).
The setup script can install user-local Node via nvm automatically;
there is no need to install Node manually here.

## 3. Clone the source, install CUDA JAX and all test tools

This *example* checks out the current experimental research code. For a
future study, replace only the branch with the then-current handover's
approved research branch. For the public viewer alone, use the
`main` branch or the [CPU-only viewer instructions](PLAYGROUND.md).

```bash
git clone --single-branch --branch feature/v5031-verified-ascent \
  https://github.com/mdt733/emu.git "$HOME/emu-research"
cd "$HOME/emu-research"
git branch --show-current
git rev-parse HEAD

# Creates this checkout's .venv; installs JAX CUDA 13 + EMU + pytest,
# installs Node 24, builds the viewer and runs CPU/browser tests.
bash scripts/setup_wsl.sh --cuda13 --install-node --check

source .venv/bin/activate
python --version
python -m pytest --version
```

**Important:** `pytest` is required by v5031 experiment preflights,
even though it is a development dependency. A complete GPU install must
include `python -m pip install -e ".[dev]"`. The improved
`setup_wsl.sh` on this documentation/maintenance branch installs the
dev extras with `--cuda13` even without `--check`; **older frozen
source checkouts do not yet have that fix**. The recommended command
above includes `--check`, which also installs dev extras on older
checkouts and explicitly runs the Python and browser suites. If you
omit `--check` on a historical branch, install `.[dev]` yourself.

If you choose to install manually instead of using the setup script:

```bash
cd "$HOME/emu-research"
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "jax[cuda13]"
python -m pip install -e ".[dev]"
```

The manual path is **training only**: building the browser viewer requires
Node 24 and the locked dependencies in `emu/viewer_src/package-lock.json`.
Do not mix the manual path with the script in an existing running venv.

## 4. Verify the actual GPU backend — not just a successful pip install

```bash
cd "$HOME/emu-research"
source .venv/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python -m pip check

python - <<'PY'
import sys
import jax
import mujoco
import mujoco.mjx
import optax
import pytest
print("Python:", sys.version.split()[0])
print("JAX:", jax.__version__)
print("MuJoCo:", mujoco.__version__)
print("Available JAX devices:", jax.devices())
print("GPU devices:", jax.devices("gpu"))
assert jax.devices("gpu"), "CUDA installation is NOT working"
PY

emu doctor --require-gpu
python -m pytest -q tests/test_v5031_verified_ascent.py
```

The command `jax.devices("gpu")` must show a CUDA device; merely seeing
`gpu` in `nvidia-smi` or having `jax` import successfully is insufficient.

**Acceptance checks, no training campaign:** run the real 360-input
sensor/terrain and PPO smoke on a small CPU batch, then on the intended
2048-world CUDA batch. The full CUDA smoke compiles and executes a
real PPO update, consumes substantial GPU memory and should be run only
when no other training process is using the GPU.

```bash
python scripts/smoke_unified_focus.py \
  --device cpu --envs 12 \
  --terrain-max-m .08 --growth-recipe main-mature \
  --terrain-lesson coherent-careful \
  --temporal-vx-reward-share .5 \
  --verified-ascent-reallocation 3

python scripts/smoke_unified_focus.py \
  --device cuda --envs 2048 --full-update \
  --terrain-max-m .08 --growth-recipe main-mature \
  --terrain-lesson coherent-careful \
  --fast-rehearsal-share .25 \
  --temporal-vx-reward-share .5 \
  --verified-ascent-reallocation 3
```

This proves the source can execute on the machine; it does **not**
prove that a new random policy can walk, that the trained checkpoint is
valid, or that a research campaign may skip its own source-specific
preflight. The [v5031 launcher](V5031_VERIFIED_ASCENT.md) has its own
mandatory two-arm acceptance and checksum checks.

## 5. Set up results, preserve versions, and start a *new* run

```bash
mkdir -p "$HOME/robot_lab/emu_runs"
cd "$HOME/emu-research"
git status --short
git rev-parse HEAD
python -m pip freeze > "$HOME/robot_lab/emu-python-packages.txt"
python -m emu.cli train --help
python scripts/v5031_verified_ascent_ab.py
```

The last line is **plan-only**. The current v5031 A/B campaign
**requires the completed v5030 B80 final checkpoint**. On a genuinely
new machine without that checkpoint, the v5031 continuation launcher
must refuse to start: it is *not* a cold-start installer. To reproduce
a full research lineage from zero, use a separate **source-pinned
cold-start launcher** at its original Git SHA, and follow its
experiment-specific readme and GPU preflights. Do **not** pretend a
missing ancestor can be recreated with random weights or silently
resume from a different training source.

All live checkpoints and exam results belong under
`~/robot_lab/emu_runs/`, **outside** the Git checkout. Keep:
the exact training SHA, run manifest, installed Python package
versions, seed, accepted GPU preflight, complete checkpoints,
normalization/optimizer/curriculum state, exported actors, and physical
exam protocols. The checkpoint files **are not stored in GitHub**.
Back them up separately if you want to migrate a trained bird.

Do not modify a checkout after starting a source-pinned run.
For code/report changes, use another Git branch and separate working tree.

## 6. Viewer, remote SSH and power management (optional)

The public `main`-lineage demo is deliberately protected. In the
installed checkout, after `emu viewer-build` (setup does this already):

```bash
emu viewer showcase/champion.emupolicy --no-open
```

Open `http://127.0.0.1:8765/` locally. Use
[the playground guide](PLAYGROUND.md) for a viewer-only installation
without training dependencies.

For remote unattended training via Tailscale SSH, use a persistent
terminal session such as GNU Screen (`sudo apt-get install screen`,
`screen -S emu`, detach with Ctrl+A then D, resume with
`screen -D -RR emu`). Keep Windows awake, not asleep or powered off.
The SSH/Tailscale service and Ubuntu distro must remain reachable;
Screen does not survive a host reboot, though an interrupted run may
be checkpoint-resumable with the exact original SHA.

## 7. Quick troubleshooting

| Symptom | Interpretation / remedy |
|---|---|
| `No module named pytest` | Wrong/incomplete venv. Activate this checkout's `.venv` and run `python -m pip install -e ".[dev]"`. |
| `No module named emu` | Activate the right environment and run `python -m pip install -e ".[dev]"` from the checkout. |
| `python3.13: command not found` | Ensure `~/.local/bin` is on `PATH`; use `uv python install 3.13` or your distribution's 3.13 package. |
| JAX has CPU only / `jax.devices("gpu")` fails | Confirm Windows GPU driver, WSL2, `nvidia-smi` inside WSL, `jax[cuda13]` and driver compatibility. Avoid overriding pip CUDA libraries with `LD_LIBRARY_PATH`. |
| CUDA / XLA out of memory on 8 GB | Ensure no concurrent GPU runs, use `XLA_PYTHON_CLIENT_PREALLOCATE=false`, and respect source-specific 2048-world preflights. Do not quietly halve world count in a matched A/B study. |
| Git error: not a repository | `cd` into the actual checked-out EMU directory, not `~`. |
| Manifest / source SHA mismatch | Stop. Never rewrite an ancestor SHA, bypass hashes or delete a manifest to force continuation. |
| No checkpoints on the new PC | Git stores *code*, not the learned neural policy. Restore backed-up run directories or cold-start using the original campaign contract. |

**Authoritative future references:** [JAX NVIDIA installation](https://docs.jax.dev/en/latest/installation.html),
[NVIDIA WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/),
[uv Python management](https://docs.astral.sh/uv/guides/install-python/).
Package versions and CUDA driver requirements may change; prefer
these upstream sources when provisioning a new machine.
