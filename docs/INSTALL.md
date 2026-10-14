# Install EMU (Linux / WSL2)

EMU has two independent environments: a **Python JAX/MJX trainer** and a **Node.js browser viewer**. The browser demo does not require CUDA; long GPU training does.

## Requirements

- Linux or WSL2, Python **3.13+** (including `venv`), Git and `curl`.
- Node.js **24** for Vite and JavaScript tests; the setup script can install the requested nvm/Node version.
- For GPU training, an NVIDIA GPU/driver exposed to WSL, and a JAX CUDA wheel compatible with the machine. Verify `nvidia-smi` before assuming a GPU is available.
- Internet access for the initial `pip` and `npm` installs.

Python dependencies are owned by `pyproject.toml`; browser dependencies by `viewer/package.json` and its committed lockfile.

## From blank Windows 11 to a ready Ubuntu shell

This section is the **Windows-side prerequisite** for a completely new machine. If WSL already exists, inspect it first; do not delete or unregister any distribution to follow this guide.

Open **PowerShell as Administrator**:

```powershell
wsl --list --verbose
wsl --install -d Ubuntu-24.04
```

Install Ubuntu **only if you do not already have the intended clean distro**. Reboot if Windows requests it, open the Ubuntu terminal and create its Linux username/password at first launch. Then from PowerShell confirm WSL 2:

```powershell
wsl --update
wsl --list --verbose
```

If you already have a suitable Ubuntu WSL instance, use it rather than installing a duplicate. Microsoft documents these controls in [Install WSL](https://learn.microsoft.com/en-us/windows/wsl/install) and [WSL basic commands](https://learn.microsoft.com/en-us/windows/wsl/basic-commands).

**Ubuntu 24.04 ships Python 3.12 by default; EMU needs 3.13+.** The repository's setup script deliberately refuses to install under 3.12. In the new **Ubuntu terminal**, install prerequisites and versioned Python 3.13:

```bash
sudo apt update
sudo apt install -y git curl ca-certificates software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.13 python3.13-venv
python3.13 --version
git --version
curl --version
```

[deadsnakes](https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa) is a third-party Ubuntu package repository, **not Canonical's default Python packaging**. Review its security/update policy before opting in. It installs a versioned interpreter alongside Ubuntu's Python; **do not change** `/usr/bin/python3` or use `update-alternatives` to replace Ubuntu's system Python. Other Ubuntu releases may already supply Python 3.13+ without the PPA; check their package availability instead of blindly adding it.

For a CUDA-capable Windows host, install/update the **Windows** NVIDIA driver first. Inside WSL check `nvidia-smi` (or `/usr/lib/wsl/lib/nvidia-smi` if it isn't in `PATH`). NVIDIA's [CUDA on WSL guide](https://docs.nvidia.com/cuda/archive/13.2.0/wsl-user-guide/index.html) warns **not to install the Linux display driver inside WSL**; WSL uses the host Windows driver. If no compatible GPU is visible, take the CPU install path below rather than claiming CUDA support.

### Pre-merge release-candidate check

Until the cleanup is merged, GitHub's default `main` does **not** contain its new `emu ladder` CLI. To verify the exact proposed source before merge, checkout the draft branch **before running setup**:

```bash
git clone https://github.com/mdt733/emu.git
cd emu
git switch --track origin/refactor/ladder-human-first
git rev-parse HEAD
```

This is a temporary release-candidate testing step. After the merge a normal `main` clone should work without switching branches. Avoid reusing any training data from an older checkout.

## Clone and configure

```bash
git clone https://github.com/mdt733/emu.git
cd emu
# For unmerged PR #2 validation only:
git switch --track origin/refactor/ladder-human-first
bash scripts/setup_wsl.sh --install-node
```

The example is the **CPU-only** install; it works without an NVIDIA card and is the safest first boot test. For a confirmed CUDA 13-compatible machine use `bash scripts/setup_wsl.sh --cuda13 --install-node`. Don't pass `--cuda13` unless your actual driver supports it; the setup script does not install NVIDIA drivers. It creates `.venv`, installs the editable package, runs contract checks and builds the packaged viewer. No long training run starts.

```bash
source .venv/bin/activate
emu --version
emu doctor --no-require-gpu --no-mjx-smoke
emu ladder --stage plan
```

For a GPU trainer, run `emu doctor --v5015-smoke --domain-smoke` before committing compute to a long campaign.

## Current workflows

| Task | Command |
|---|---|
| See planned experiment steps and arms | `emu ladder --stage plan` |
| Start/resume the automatic ladder on a tested GPU | `bash scripts/run_ladder.sh` |
| Stop after the first scored round | `bash scripts/run_ladder.sh --stop-after round1` |
| Watch completed validated checkpoints locally | `emu viewer --live --root ~/robot_lab/emu_runs/v5015_lab1 --no-open` |
| Run the zero-weight *local* rendering fixture | `emu viewer viewer/demo_zero.emupolicy --no-open` |
| Run Python tests | `python -m pytest -q` |
| Run browser tests | `npm test --prefix viewer` |
| Build canonical browser assets | `emu viewer-build` |

The local viewer listens on `127.0.0.1:8765`. The standalone demo fixture should **not** walk; it checks rendering. For public/static preview and the protected 60M champion see [PLAYGROUND.md](PLAYGROUND.md).

## Run directories and Git

Training source commits, environment counts and configurations are part of the experiment contract. Keep run roots outside the checkout (normally `~/robot_lab/emu_runs/`). Do not change source or ladder parameters mid-run. Do not reuse a checkpoint/run root under a different source commit; prepare a new run root instead.

The normal repository must **not** include `.venv/`, `node_modules/`, `viewer/dist/`, raw metrics dumps or schema-7 `.pkl` checkpoints. The public `.emupolicy` actor is a separate, intentionally tracked artifact.

## Browser dependencies and packaging

Use `npm ci --prefix viewer` to install the committed lockfile. The canonical browser source is `viewer/`; `emu/viewer_src/` is its generated Python-package mirror. When intentionally changing browser source or dependencies, run `emu viewer-build`, `npm test --prefix viewer`, and `npm run build --prefix viewer`; review the changed mirror and lockfile before committing.

Do not manually edit generated `viewer/dist/` or publish the local live checkpoint server.

## Troubleshooting

| Symptom | Check |
|---|---|
| Unexpected EMU version or missing `ladder` command | `git branch --show-current`, `git rev-parse HEAD`, `python -c "import emu.cli; print(emu.cli.__file__)"` |
| `PYTHONPATH` loads an old checkout | `unset PYTHONPATH`; verify the `emu.cli` source path |
| CUDA smoke fails | Verify `nvidia-smi`, installed JAX wheel and GPU backend; do not begin training |
| Optional Warp import warning | MJX may still work; trust the actual reset/step and PPO smoke results |
| Viewer reports an HTML response instead of JSON | Check whether you're running static Vite preview or the real local Python API and whether the manifest exists |
| `final.pkl` missing or corrupt | Do not delete checkpoints. Validate `latest.pkl` and rotating snapshots before attempting recovery |
| Workspace has obsolete migration backups | Preview `bash scripts/workspace_tidy.sh` first; its optional quarantine mode moves files, never blindly deletes runs |

The [validation guide](validation.md) describes the difference between fast tests, GPU integration and genuine evidence of walking.
