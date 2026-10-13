# Installing and maintaining EMU v5014

This guide is for a **fresh clone** on Ubuntu/WSL2, or for recovering a known-good existing installation. The production project is a Python/MJX trainer plus a separately built browser viewer. None of these steps start the 400M-step job without an explicit `emu forge`.

## Prerequisites

- Git and `curl` (`sudo apt install git curl` on Ubuntu).
- **Python 3.13+**, including `venv` support. Check `python3.13 --version`. The exact installation method depends on your Ubuntu release; do not replace system Python to install EMU. The setup script detects `python3.13` or a compatible `python3`.
- For GPU training: NVIDIA driver and functioning GPU pass-through into WSL2. Verify with `nvidia-smi` in WSL. The example `--cuda13` installs the JAX CUDA 13 wheels; this **must match the GPU/driver support on the target PC**. JAX's CUDA package is independent of npm.
- Node.js **24**, preferably installed with [nvm](https://github.com/nvm-sh/nvm). `.nvmrc` records the desired major version. The setup script installs nvm/Node (with your opt-in flag) when they're absent.
- Network access for the first Python and npm dependency installation.

There is deliberately no second `requirements.txt` to drift out of sync with `pyproject.toml`. Python dependencies are defined in `pyproject.toml`. Browser dependencies are defined in `viewer/package.json` and the **committed** `viewer/package-lock.json`.

## First-time clone

Obtain the **actual** Git remote URL by running `git remote get-url origin` in the working checkout on the old machine. Clone that repository into `~/robot_lab/emu` using the returned URL; no remote hosting provider is assumed. If no remote exists yet, publishing the audited Git commit is a separate step that must happen before the new PC can clone it.

After the clone:

```bash
cd ~/robot_lab/emu
bash scripts/setup_wsl.sh --cuda13 --install-node
```

The script creates/uses `.venv`, installs `emu-biped` and pytest in editable mode, installs the CUDA 13 variant only when requested, runs tests and the domain smoke test, and compiles the viewer. If a GPU or CUDA-compatible driver is unavailable, omit `--cuda13` and use the CPU-only validation path. Production GPU training still requires a working GPU on the eventual trainer.

**An important portability gate:** `viewer/package-lock.json` must already be committed in Git. In the *existing* working v5014 machine, `emu viewer-build` previously ran `npm install`, which generated it. Stage and commit that working lockfile (and its mirror) with the release. The setup script refuses to fall back to floating dependency versions if the lockfile is missing.

The viewer's Python wheel source mirror (`emu/viewer_src/`) is refreshed from `viewer/` whenever `emu viewer-build` succeeds. It is intentional duplication for wheel packaging; make sure the two browser sources and lockfiles agree before committing. Browser outputs in `emu/webviewer_dist/` are generated, **not** Git content.

## Daily commands (separate terminals)

Start by checking the environment; none of these commands launches long training:

```bash
cd ~/robot_lab/emu
source .venv/bin/activate
emu doctor --domain-smoke
python -m pytest -q
emu forge --plan
```

To **train or resume** deliberately, in its own WSL terminal:

```bash
cd ~/robot_lab/emu && source .venv/bin/activate
emu forge
```

In a second terminal, watch the newest validated completed checkpoint:

```bash
cd ~/robot_lab/emu && source .venv/bin/activate
emu viewer --live --no-open
# Windows Chrome/Edge: http://127.0.0.1:8765/
```

For a **standalone rendering test instead of live watching**, stop the existing viewer first (`Ctrl+C`), then run `emu viewer viewer/demo_zero.emupolicy --no-open`. This demo is an untrained zero-output policy; the fact that the robot renders says nothing about walking. A successful `.wasm` HTTP response is necessary but not enough; inspect the actual browser canvas. Never port-forward the unauthenticated local checkpoint feed.

For the *public static* Playground on port 4173 use [PLAYGROUND.md](PLAYGROUND.md), not `emu viewer --live`. The trainer never depends on either viewer being open.

## Browser dependency locking

The canonical `viewer/package-lock.json` fixes the installed npm package versions, including MuJoCo, Three and Vite. `emu viewer-build` now uses **`npm ci`**, which installs from that lockfile and fails if `package.json` and the lock disagree. It never rewrites the lockfile during a normal viewer build.

To intentionally upgrade the browser dependencies on a development machine:

```bash
cd ~/robot_lab/emu/viewer
npm install
cd ..
emu viewer-build
python -m pytest -q
git diff -- viewer/package.json viewer/package-lock.json
```

Recheck the browser, then commit the updated `viewer/package-lock.json` and mirrored `emu/viewer_src/package-lock.json` together. Do not use `npm update` casually on the training computer. The bundled `package.json` presently lists `@mujoco/mujoco` as `latest`, so the **lockfile is the important source of the tested, concrete version**.

## Git workflow and source hygiene

Before a production run, commit the exact source and dependencies you intend to train:

```bash
cd ~/robot_lab/emu
git status --short
git add -A
git diff --cached --stat
git diff --cached --check
git diff --cached --name-only
# Only after reviewing staged files and running the actual browser smoke test:
git commit -m "v5014: production trainer, reproducible viewer, and installation"
# Only if v5014.0 has not been created already:
git tag -a v5014.0 -m "EMU v5014 production"
```

Large `.pkl` checkpoints and raw training outputs should live in `../emu_runs`, never in Git. `node_modules`, `.venv`, `dist`, generated bundles and GPU state are ignored. We intentionally track a compact demo `.emupolicy` fixture while ignoring arbitrary trained exports. Review changes before any `git add -A`, especially `results/`.

**Avoid committing on a branch whose status contains unexpected files.** To push, inspect `git branch -vv`, `git remote -v`, `git status --short` and release tags. The intended official mainline is `main`, but do not assume it exists or has already been pushed; don't force-push or rewrite the historical v500x/v5013 commits.

## Workspace cleanup (historical migration debris only)

A previous release installer created `~/robot_lab/emu_upgrade_backups/`. Previous troubleshooting also created `~/robot_lab/incoming-v5014-rc1/` and `~/robot_lab/v5013_results_backup.*/`. These folders are **outside Git**, so `git clean` won't help and could be dangerous if used recursively inside your checkout.

First preview:

```bash
cd ~/robot_lab/emu
bash scripts/workspace_tidy.sh
```

The preview labels a small whitelist of known migration leftovers and lists everything else as **not touched**. It protects `emu/` and `emu_runs/` unconditionally. If the proposed candidates are correct:

```bash
bash scripts/workspace_tidy.sh --quarantine
# Type ARCHIVE to confirm.
```

Files are **moved, not deleted**, to `~/emu_workspace_archives/<timestamp>/`. This reduces visual clutter while retaining an exact recovery path. The script never recursively purges backups and never chooses unknown folders. Inspect archived copies and compare any unique experimental evidence before you decide *manually* to reclaim disk space. Git and live checkpoints remain untouched.

## Common symptoms

| Symptom | Meaning / next check |
|---|---|
| `emu --help` shows only six v5013 commands | You're on the old checkout; verify `cat emu/VERSION`, `git branch --show-current` and `python -c 'import emu.cli; print(emu.cli.__file__)'` |
| `Node/npm are required` | Run `bash scripts/setup_wsl.sh --install-node` after Python prerequisites, or install Node 24 with nvm |
| Missing `package-lock.json` | Keep the lockfile from the **already-tested** build, stage it and its viewer mirror in Git; don't commit a random dependency resolution untested |
| Browser `.wasm` HTTP 200 but blank scene | Open browser console, verify JavaScript/physics and actual robot rendering before claiming acceptance |
| `warp` import missing in `emu doctor` | MJX can operate without optional Warp; check whether the MJX reset/step actually passed |
| CUDA timer warmup warnings | Often timing/measurement warnings; investigate only if GPU smoke fails or throughput is unexpectedly poor |
| Git shows `?? results/v5013_tournament/` | Compact research evidence is new to v5014. Review it; never wipe it merely to make status clean |
| Old backups fill `robot_lab/` | Use the preview/quarantine workflow above; no `rm -rf` needed |
