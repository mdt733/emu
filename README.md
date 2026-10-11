# EMU v5011 — self-mapping biped trainer

EMU is a reinforcement-learning trainer for a dynamically unstable six-actuator biped in MuJoCo/MJX. Thousands of robots train in parallel on the GPU with PPO while a continuously decaying capability graph learns what the **current** policy can do and uses that knowledge to choose useful future practice.

The short version:

```text
                 2,048 MJX robots
                       │
                       ▼
                 PPO experience
                  /          \
                 ▼            ▼
              policy     capability graph
                 │            │
                 │            ▼
                 │       task sampler
                 │            │
                 └──────┬─────┘
                        ▼
                   next worlds

             HELD-OUT EVALUATORS
                     │
                     X   no feedback into training
```

Velocity capability lives on **nodes** in `(vx, vy, yaw-rate)` command space. Agility lives on **edges** between those nodes. Recovery is a command-conditioned layer on the same map.

## What changed in v5011

v5010 established that the graph-driven trainer beats the same trainer with the map in shadow mode and reproduced that result from an independent cold start. v5011 keeps that architecture but fixes the map's semantics and cleans the project for normal use:

- one **semantic node trial** per sustained command window instead of per-frame pseudo trials;
- one **recovery trial** per integrated resilience birth;
- one **maneuver trial** per complete command transition and settling period;
- competence probability is separate from continuous performance quality;
- conservative confidence bounds determine mastery;
- mastered-boundary expansion, under-covered-region exploration and stale-skill rehearsal can now work as intended;
- held-out reviews remain quarantined and test whether the self-map is honest;
- checkpoints carry explicit schema/model/observation metadata;
- a small `emu` CLI is the public front door, while the old research scripts remain available.

## Install

The known-good development environment is WSL/Linux with Python 3.13 and an NVIDIA GPU. Install the machine-specific JAX build first, then EMU in editable mode:

```bash
cd ~/robot_lab/emu
python -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel

# Choose the JAX CUDA build appropriate for your machine.
pip install -U "jax[cuda13]"
pip install -e .

emu doctor
```

`emu doctor` checks JAX/MJX, `robot.xml`, an environment reset/step, the capability-field event model and the basic source contract before you commit GPU hours. The editable install requires normal package-network access the first time so the build backend/dependencies can be installed; subsequent work is local.

## Train from zero

```bash
emu train
```

Defaults:

- seed `733`;
- 2,048 parallel environments;
- genuine random actor/critic start;
- 0→20M conservative nursery with the field observing in shadow mode;
- 20→120M recommended event-based graph trainer;
- held-out calibration every ~40M steps;
- sparse fixed regression checks plus a forced final common recovery exam.

A different independent seed gets its own default run root:

```bash
emu train --seed 1742
```

Interrupted runs are safe to rerun:

```bash
emu resume
```

The runner reuses completed stages and resumes the current branch from its latest checkpoint.

## Watch the live capability map

```bash
emu view
```

Open `http://127.0.0.1:8765/`.

The viewer auto-follows the newest v5011 run and can display:

- command competence;
- conservative competence bound;
- confidence / effective trials;
- continuous command quality;
- learning vs forgetting;
- staleness;
- integrated recovery frontier;
- maneuver coverage, competence and rate.

## Evaluate a checkpoint

```bash
emu evaluate
```

By default this finds the newest v5011 final/best/latest checkpoint and runs the physical atlas. You can also pass an explicit checkpoint:

```bash
emu evaluate --checkpoint /path/to/final.pkl
```

## Reproduce the trainer on two cold-start seeds

```bash
emu benchmark
```

This runs the recommended graph trainer independently for seeds 733 and 1742. Each seed receives its own run/results directory so one cannot accidentally reuse another seed's branch.

## Research interface

Nothing from the older experimental machinery has been thrown away. Advanced work can still use:

```bash
emu experiment --stage trainer --list
python run_ladder.py --help
python training.py --help
```

The difference is that a new user no longer needs to understand the entire project history before training a robot.

## Repository guide

```text
README.md                 this page
VERSION                   current source generation
pyproject.toml            editable install + `emu` command
robot.xml                 physical robot model
emu_cli.py                small public CLI

environment.py            MJX physics/task/reset model
training.py               PPO, governor, checkpoints and evaluators
capability_field.py       live event-based command/recovery/maneuver map
capability_cloud.py       quarantined held-out calibration/recovery probes
atlas.py                  physical-unit final capability atlas
field_viewer.py           read-only live map viewer
viewer.py                 policy/robot visual viewer
run_ladder.py             research campaign orchestration

docs/                     architecture, training, evaluation and history
tests/                    cheap source-level regression tests
```

## The important separation

| System | Updates policy? | Changes future training? | Purpose |
|---|:---:|:---:|---|
| PPO rollout/update | yes | yes | learn the actor/critic |
| Regime governor | no | yes | global safety, difficulty and allocation |
| Live capability graph | no | yes | decide **where/how** to practise |
| Held-out calibration/recovery exam | no | **no** | verify the self-map/generalisation |
| Physical atlas | no | no | measure final limits |

The held-out evaluator is intentionally not a teacher. If EMU claims a command is 90% reliable, the examiner is allowed to tell **us** whether that claim is true; it is not allowed to whisper the answer back into the curriculum.

## More detail

- [Architecture](docs/architecture.md)
- [Training lifecycle](docs/training.md)
- [Capability graph](docs/capability-map.md)
- [Evaluation](docs/evaluation.md)
- [Hardware/model notes](docs/hardware-model.md)
- [v5011 design/acceptance criteria](docs/v5011-design.md)
- [Validation status](docs/validation.md)
- [Project history](docs/history.md)
