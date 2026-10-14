# Install EMU (Linux / WSL2)

EMU has two independent environments: a **Python JAX/MJX trainer** and a **Node.js browser viewer**. The browser demo does not require CUDA; long GPU training does.

## Requirements

- Linux or WSL2, Python **3.13+** (including `venv`), Git and `curl`.
- Node.js **24** for Vite and JavaScript tests; the setup script can install the requested nvm/Node version.
- For GPU training, an NVIDIA GPU/driver exposed to WSL, and a JAX CUDA wheel compatible with the machine. Verify `nvidia-smi` before assuming a GPU is available.
- Internet access for the initial `pip` and `npm` installs.

Python dependencies are owned by `pyproject.toml`; browser dependencies by `viewer/package.json` and its committed lockfile.

## Clone and configure

```bash
git clone https://github.com/mdt733/emu.git
cd emu
bash scripts/setup_wsl.sh --install-node
```

That is the **CPU-only** install. For a CUDA 13-compatible machine use `bash scripts/setup_wsl.sh --cuda13 --install-node`. Don't pass `--cuda13` unless your actual driver supports it; the setup script does not install NVIDIA drivers. It creates `.venv`, installs the editable package, runs contract checks and builds the packaged viewer. No long training run starts.

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
