# EMU — dynamically unstable six-servo biped

EMU learns **dynamic stepping, steering and recovery**, rather than merely standing upright. The v5014 production system uses JAX/MJX + PPO and a deployable actor with **55 observations × 4 frames (220 inputs), 6 actuator outputs**. A separate Three.js/MuJoCo WASM viewer lets you control or watch the same actor entirely in a browser.

**Release status:** The source and local viewer have passed meaningful tests, but this document does **not** certify completion of the 400-million-step run, the champion's held-out performance, or public GitHub Pages deployment. See [release evidence](RELEASE_VALIDATION.md) and the [project handover](HANDOVER.md).

## First time installing?

For a fresh **Linux/WSL2** checkout, follow [the installation guide](docs/INSTALL.md). Python **3.13**, Git and Node.js **24** are required for the documented full setup; GPU training additionally requires a compatible NVIDIA driver/JAX CUDA environment. From inside your Git clone:

```bash
cd ~/robot_lab/emu
bash scripts/setup_wsl.sh --cuda13 --install-node
```

Omit `--cuda13` for a CPU-only viewer/development setup. The script creates a virtual environment, installs Python dependencies, obtains Node via nvm when requested, builds the viewer using the **committed npm lockfile**, and runs checks. **It does not start training, delete files, or rewrite Git history.** Node/npm dependencies are in `viewer/package.json` and `viewer/package-lock.json`, **not** `requirements.txt`.

## Common commands

Activate the environment in each terminal:

```bash
cd ~/robot_lab/emu
source .venv/bin/activate
emu doctor --domain-smoke   # check GPU/MJX (requires a working JAX backend)
emu forge --plan            # inspect the long run without starting it
```

| Goal | Command / instructions |
|---|---|
| Start/resume the deliberate 400M production lineage | `emu forge` — [architecture and phase boundaries](docs/v5014-production.md) |
| Watch completed training checkpoints | `emu viewer --live --no-open` → `http://127.0.0.1:8765/` |
| Open a standalone policy | `emu viewer viewer/demo_zero.emupolicy --no-open` (untrained fixture) |
| Rebuild the viewer | `emu viewer-build` |
| Preview a public, fully static demonstration | [Playground guide](docs/PLAYGROUND.md), usually on port **4173** |
| Run held-out evaluation | `emu evaluate --help` and [evaluation methodology](docs/evaluation.md) |
| Restore context after a chat handoff | **[HANDOVER.md](HANDOVER.md)** |

**Live viewer and Playground are different services.** Port **8765** belongs to the local Python checkpoint watcher; port **4173** is the independent Vite static preview and cannot watch private training. Do not expose the local watcher to the internet. Launch the trainer and local viewer in separate terminals.

## Repository map

```text
emu/                     # Git repository
├── emu/                 # Python trainer, policy exporter, local viewer server
├── viewer/              # Canonical JavaScript viewer source + npm lockfile
├── emu/viewer_src/      # Packaging mirror of the viewer
├── scripts/             # Installation, validation, safe housekeeping
├── tests/               # Regressions
├── results/             # Small selected experiment evidence
├── showcase/            # Public actor-only champion export (when available)
├── docs/                # Operational documentation and research rationale
├── RELEASE_VALIDATION.md # Observed evidence versus remaining acceptance
└── HANDOVER.md          # Current state and safe continuation
../emu_runs/             # Large training runs/checkpoints, outside Git
```

Only source, tests, lockfiles, documentation and selected small results belong in Git. **Do not commit** `.venv/`, `node_modules/`, `dist/`, `webviewer_dist/`, generated `viewer/public/`, raw `*.pkl` checkpoints or entire GPU runs. An explicitly selected `showcase/champion.emupolicy` is intended to be public; review that exported actor before committing it.

For a guide to every document's purpose, use [docs/README.md](docs/README.md). For scientific lessons, see [docs/history.md](docs/history.md). For workspace cleanup, see [docs/INSTALL.md#workspace-cleanup](docs/INSTALL.md#workspace-cleanup)—preview first; never treat untracked files as disposable.
