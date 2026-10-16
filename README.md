# EMU · Learning to walk

**Six actuators. Two feet. No scripted gait.** EMU trains a dynamically unstable biped in thousands of parallel MuJoCo/MJX worlds. Its goal is to discover *how to teach walking*, not simply chase one lucky neural network.

**[Try the 120M Pitch15 robot](https://mdt733.github.io/emu/)** · **[Install and train](docs/INSTALL.md)** · **[Project handover](HANDOVER.md)** · **[Viewer and publishing](docs/PLAYGROUND.md)**

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
experiments                      Shared foundation → competing teaching strategies
        ↓
reports                          How did it learn? Which branches really improved?

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

A shared **20M-step foundation** fed four matched 80M-step teaching experiments, then two 100M-step finalists and a 120M-step winner (320M aggregate training steps). **Pitch15** (a 15° chassis-pitch command teaching ceiling) ranked first on the common frozen-policy screen: 0.684 at 80M → 0.712 at 100M → **0.723 at 120M**. The public [Playground](https://mdt733.github.io/emu/) contains the 120,061,952-step frozen Pitch15 export.

**Important limitation:** final training telemetry reported **74.1% of servo samples over the configured rated-torque threshold**. The model is a *simulation demonstration*, not a hardware-safe controller; signed distance, complete strides, real motor currents and contact transfer still require further study.

## Where things live

Start with `robot.xml`, `environment.py`, `rewards.py`, `ppo.py`, `training.py`, `curriculum.py`, then the experiment/report files. The browser has exactly one authoritative source: `emu/viewer_src/`. Its JavaScript/WASM dependencies are separate from training on purpose.

The only maintained long-form documentation is this front page, **[INSTALL](docs/INSTALL.md)**, **[HANDOVER](HANDOVER.md)** and **[PLAYGROUND](docs/PLAYGROUND.md)**. Old design attempts and one-off runs live in Git history, not in the working tree. Run `bash scripts/check.sh` before committing. Software tests don't replace a fresh CUDA acceptance and independent walking exams.
