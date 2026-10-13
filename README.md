# EMU v5014 — production biped

**One dynamically stepping robot, one continuous learning lineage.** EMU trains a six-servo biped using JAX/MJX with a deployable **55-feature × 4-frame = 220-input, 6-output** neural-network policy. The production run uses a restrained 10% capability-field teacher and progressively imperfect robots/floors. It is not a static-standing project.

## New computer? Start here

Use **Linux or WSL2 Ubuntu**. The documented training path needs Python 3.13, Git, a working NVIDIA GPU/JAX CUDA setup, and Node.js 24 for the browser viewer. Python and JavaScript have **separate dependency managers**; npm does not belong in `requirements.txt`.

```bash
# Run from your cloned EMU Git repository (not the parent robot_lab directory).
cd ~/robot_lab/emu
bash scripts/setup_wsl.sh --cuda13 --install-node
```

This creates `.venv/`, installs the Python project/tests, installs Node 24 through nvm if needed, builds the browser from **locked** npm dependencies, runs the tests, and performs the MJX domain smoke test. It **does not start training, delete files, commit Git changes, or touch your existing run data**. `--cuda13` is for machines with a compatible NVIDIA driver/CUDA 13 JAX path; omit it for a CPU-only workstation. If Python 3.13 or a supported NVIDIA driver is missing, follow [the full installation guide](docs/INSTALL.md) first.

After setup, open two terminals as appropriate:

```bash
cd ~/robot_lab/emu && source .venv/bin/activate
emu viewer viewer/demo_zero.emupolicy --no-open  # viewer acceptance: http://127.0.0.1:8765/
# OR:
emu viewer --live --no-open                   # watches completed training checkpoints
```

```bash
cd ~/robot_lab/emu && source .venv/bin/activate
emu forge --plan                             # preview without starting anything
emu forge                                    # deliberate 400M-step production run
```

**Do not start `emu forge` just to test installation.** The first command to run is `emu doctor --domain-smoke`. See [Install & recovery](docs/INSTALL.md) for GPU/JAX notes, Node installation, troubleshooting, and exact fresh-clone expectations.

## Where everything goes

| Location | Purpose | Commit to Git? |
|---|---|---|
| `emu/` | Canonical Python trainer, checkpoints/exports, viewer server | Yes, source only |
| `viewer/` | Canonical browser source, `package.json` and **`package-lock.json`** | Yes |
| `emu/viewer_src/` | Mirrored viewer source for Python wheel installs | Yes; kept in sync by `emu viewer-build` |
| `tests/`, `docs/`, `scripts/` | Regression tests, decisions/history, setup/housekeeping | Yes |
| `results/` | Compact experiment summaries, v5013 tournament evidence | Yes, after review |
| `.venv/`, `node_modules/`, `dist/`, `emu/webviewer_dist/` | Local generated dependencies/build | **No** |
| `../emu_runs/` | Large checkpoints, training logs and generated policies | **Never** |

This source repository is meant to be portable. Don't copy `.venv`, `node_modules` or raw GPU runs from another PC to reproduce it: clone Git and run the setup script.

## Training and viewer contract

The default **400M** production run has phase boundaries at **20M/60M/140M/260M/400M**. The actor, critic, optimiser, normaliser, curriculum and capability field carry forward across phases. Checkpoints use atomic replacement and strict validation/resume checks. The default run root is `~/robot_lab/emu_runs/v5014_production/`.

The live viewer watches **completed, validated checkpoints** on loopback (`127.0.0.1:8765`), not partially written GPU weights. Physics and inference run in the browser; the viewer does not train the policy. Autonomous demonstration commands are training-derived self-assessments, **not held-out competence scores**. Do not forward the local viewer port to the internet.

**What this release does *not* prove:** a full 400M training run, physical-hardware transfer, full geometry randomisation, or successful lateral/yaw sign checks with a genuinely trained policy. The zero-output demo is only a rendering test.

## Project reference

- [Installation / clean-clone checklist](docs/INSTALL.md) — start here on a new machine.
- [Production architecture and training phases](docs/v5014-production.md)
- [Browser usage and control mapping](viewer/README.md)
- [Domain randomisation limits](docs/domain-randomization.md)
- [Scientific lessons that must survive cleanup](docs/history.md)
- [Audit results and outstanding acceptance](RELEASE_VALIDATION.md)
- [Release handover](docs/HANDOVER_5014.md)

## Cleaning `robot_lab/` safely

Run `bash scripts/workspace_tidy.sh` to **preview** recognized legacy extraction/backup folders outside this Git checkout. When comfortable, `bash scripts/workspace_tidy.sh --quarantine` **moves** those candidates into a dated archive **outside `robot_lab`**, without deleting any bytes or touching `emu/` and `emu_runs/`. See [the cleanup instructions](docs/INSTALL.md#workspace-cleanup) before using it.

## Public EMU Playground

The public Playground runs the MuJoCo browser simulator without a Python server
and starts with an explicitly untrained demo policy until you publish a champion.
See [Playground setup and deployment](docs/PLAYGROUND.md) for local previews,
viewer appearance controls, GitHub Pages and release instructions.
