# EMU · Learning to walk

**Six actuators. Two feet. No scripted gait.** EMU trains a dynamically unstable biped in thousands of parallel MuJoCo/MJX worlds. Its goal is to discover *how to teach walking*, not simply chase one lucky neural network.

**[Visit the EMU Playground](https://mdt733.github.io/emu/)** · **[Install and train](docs/INSTALL.md)** · **[Project handover](HANDOVER.md)** · **[Viewer and publishing](docs/PLAYGROUND.md)**

## How it works

```text
robot.xml + environment.py       The actual robot, physics, contacts and commands
        ↓
rewards.py                       Which behaviours are useful? (and which are costly?)
        ↓
ppo.py + training.py             Many MJX worlds, experience, gradients, checkpoints
        ↕
curriculum.py                    What should this robot practise next?
        ↕
branch_lab.py                     Shared foundation → competing teaching strategies
        ↓
results.py                       How did it learn? Which branches really improved?

             viewer_src/         Frozen .emupolicy, interactive MuJoCo/WASM;
                 ↑               NEVER changes the trainer
           export-policy
```

The deployed actor reads **220 observations (55 per frame × four frames)** and controls **six joint targets**. The checkpoint stores the full learning state; the smaller `.emupolicy` includes only what the browser needs. A nice-looking demo is not proof of safe hardware walking.

## Start

On Linux/WSL2 with Python 3.13, Git and optionally an NVIDIA CUDA-capable GPU:

```bash
git clone https://github.com/mdt733/emu.git
cd emu
bash scripts/setup_wsl.sh --cuda13 --install-node    # GPU / CUDA 13
source .venv/bin/activate
emu doctor --v5015-smoke --domain-smoke
emu ladder --stage plan
```

For CPU-only inspection, omit `--cuda13` and run `emu doctor --no-require-gpu --no-mjx-smoke`. For tested GPU training, begin with a new root outside this repo:

```bash
emu ladder --stage foundation --root "$HOME/robot_lab/emu_runs/my-study" --seed 733 --envs 2048 --device cuda
emu ladder --stage auto --root "$HOME/robot_lab/emu_runs/my-study" --seed 733 --envs 2048 --device cuda --stop-after round1
emu viewer --live --root "$HOME/robot_lab/emu_runs/my-study" --no-open
```

See [INSTALL.md](docs/INSTALL.md) for fresh Windows/WSL setup, GPU checkpoint acceptance, commands and troubleshooting. **Never mix source revisions or change a running campaign's configuration.**

## What the research found

A shared **20M-step foundation** fed four matched 80M-step teaching experiments, then two 100M-step finalists and a 120M-step winner (320M aggregate training steps). **Pitch15** (a 15° chassis-pitch command teaching ceiling) ranked first on the common frozen-policy screen: 0.684 at 80M → 0.712 at 100M → **0.723 at 120M**. The repository's `showcase/champion.emupolicy` is the genuine 120,061,952-step Pitch15 export. Public GitHub Pages may serve an older deployment until its manual workflow is run.

**Important limitation:** final training telemetry reported **74.1% of servo samples over the configured rated-torque threshold**. The model is a *simulation demonstration*, not a hardware-safe controller; signed distance, complete strides, real motor currents and contact transfer still require further study.

## Where things live

Everything is under `emu/`. Read these files in this order:

| Responsibility | Code |
|---|---|
| Physical robot, worlds, tasks, terrain | `robot.xml`, `environment.py` |
| Eight transparent training payments | `rewards.py` |
| Neural actor, PPO, JAX runtime | `ppo.py` |
| One run's update loop and saving | `training.py` |
| Curriculum, capability evidence and health | `curriculum.py` |
| Shared foundation, experimental arms and promotions | `branch_lab.py` |
| Fixed screens, gait, recovery and course examinations | `evaluation.py`, `directional.py` |
| Rollout facts, learning narratives and comparison reports | `results.py` |
| Local viewer, live watcher and policy export | `viewer_server.py`, `viewer_src/` |

Four support boundaries remain deliberately separate: `contracts.py` (actor/physics constants), `training_config.py` (all run knobs), `checkpoint_io.py` (complete validated learning state and recovery), and `cli.py` (one public command interface). `__init__.py` is Python packaging. The JavaScript/WASM viewer has its own locked dependencies.

This reduces the package from 42 Python files to 15 **without pretending the learning maths got smaller**. Further true size reduction means analysing and cutting unnecessary behaviour, not merely moving functions.

The only maintained long-form documentation is this front page, **[INSTALL](docs/INSTALL.md)**, **[HANDOVER](HANDOVER.md)** and **[PLAYGROUND](docs/PLAYGROUND.md)**. Old design attempts and one-off runs live in Git history, not in the working tree. Run `bash scripts/check.sh` before committing. Software tests don't replace a fresh CUDA acceptance and independent walking exams.
