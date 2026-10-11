# EMU

**A self-mapping reinforcement-learning trainer for a dynamically unstable six-actuator biped.**

EMU trains thousands of simulated robots in parallel with MuJoCo/MJX and PPO. The unusual part is that the trainer also builds a continuously changing map of what the *current* policy can actually do — where it can move reliably, where it is still learning, how quickly it can change direction, and how well it can recover from disturbances.

That map is then used to decide what the next generation of training worlds should practise.

<!--
MEDIA SLOT — enable after a good run is captured.
Suggested layout:

<p align="center">
  <img src="media/emu_prance.gif" width="760" alt="EMU biped performing dynamic locomotion in MuJoCo" />
</p>

A second, smaller GIF can sit in the Capability map section below:

<p align="center">
  <img src="media/capability_map.gif" width="680" alt="EMU live capability map evolving during training" />
</p>
-->

## The idea

EMU is not trying to learn a static standing pose and then shuffle around it. The target is **dynamic stepping and catching locomotion**: a robot that can move, turn, accelerate, absorb disturbances and continually recover its balance.

The training loop has four deliberately separate pieces:

```text
                 2,048 MJX robot worlds
                          │
                          ▼
                  fresh experience
                    /           \
                   ▼             ▼
                 PPO       capability graph
                   │             │
                   │             ▼
                   │        task sampler
                   │             │
                   └───────┬─────┘
                           ▼
                     next training worlds

                  HELD-OUT EVALUATION
                           │
                           X  no feedback into training
```

**PPO** learns the actor and critic.  
**The governor** decides how adventurous training is allowed to be.  
**The capability graph** decides where that training effort is most useful.  
**Held-out evaluators** only measure the result; they are deliberately prevented from teaching the policy.

That last separation matters. EMU is allowed to learn from its own training experience, but the exam is not allowed to whisper the answers back.

## A map of what the robot can do

The live capability model is a low-resolution 3-D lattice over commanded motion:

```text
node = [forward velocity, lateral velocity, yaw rate]
```

A node answers:

> Can the current robot operate here reliably?

Agility lives on the **edges** between nodes. An edge answers:

> Can the robot move from this command to that one quickly and stay in control?

Recovery is stored alongside the same command map, so the trainer can distinguish between a command that is easy in clean locomotion and the same command under a disturbance.

EMU v5012 records semantic events rather than pretending every simulation frame is an independent trial:

- about one second of stable command following → one node verdict;
- one complete disturbance/recovery attempt → one recovery verdict;
- one command ramp plus settling period → one maneuver verdict.

Each location therefore carries an interpretable estimate of **competence**, **confidence**, **quality**, **staleness** and **learning/forgetting trend**.

## Why this exists

A conventional curriculum often says things like “increase forward speed after N steps” or “make disturbances harder when the average fall rate is low.” That works, but it throws away a huge amount of information.

EMU already has thousands of little robots continuously trying different commands, transitions and recovery situations. Their ordinary training experience is itself an enormous experiment.

The capability graph tries to use that information directly:

- rehearse skills whose evidence is becoming stale;
- consolidate regions that are difficult but improving;
- expand outward from confidently mastered territory;
- spend less time proving the same easy command again;
- practise weak maneuver directions between otherwise good operating points;
- keep exploration inside the global safety limits earned by the curriculum governor.

The goal is not merely a good final policy. The goal is a **trainer that can repeatedly build a good robot from random weights under a finite GPU budget**.

## Current training lifecycle

The normal cold-start path is intentionally simple:

```text
0M ───────────── 20M ───────────────────────────── 120M
│                  │                                 │
│ conservative     │ self-mapping graph trainer     │ primary
│ nursery          │ becomes the curriculum locator │ benchmark
│                  │                                 │
└─ random weights  └─ same policy continues ────────┘

120M → 220M is available when exploring longer-run limits.
```

The first ~20M steps are mainly about reliably escaping random initialization. The interesting trainer comparison is what happens after that: how effectively the next ~100M experiences are spent building a broader, more agile and more robust robot.

## v5012: trustworthy measurement

v5011 produced a strong cold-start policy, but the run exposed two measurement bugs: the live capability field was too strict for an oscillating walking gait, and one-frame command jumps became invisible to the maneuver graph at the acceleration frontier. v5012 fixes those semantics before adding any new curriculum ideas.

- node competence uses a complete one-second gait window and **mean tracking error**, rather than demanding nearly every instantaneous frame sit inside the velocity tolerance;
- one-frame command jumps are explicit maneuver events;
- field calibration compares node predictions against a held-out **node-window exam**, while recovery remains a separate recovery exam;
- missing field metrics, incompatible field state and resume-contract failures stop loudly;
- `emu compare` runs matched `shadow` and `graph` arms from the same 20M foundation for each seed;
- the final trainer exam includes the longer hard suites so mature policies do not all look identical at ~100% survival.

The immediate experiment is deliberately boring and causal: **same 20M ancestor, shadow vs graph, two independent seeds.**

## Quick start

EMU is currently developed and tested on Linux/WSL with Python 3.13 and an NVIDIA GPU.

```bash
git clone https://github.com/mdt733/emu.git
cd emu

python3.13 -m venv .venv
source .venv/bin/activate

python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt

# JAX accelerator packages are machine-specific.
# This project is currently tested with JAX 0.11.2 + CUDA 13.
python -m pip install -U "jax[cuda13]==0.11.2"

python -m pip install -e .

emu doctor
```

A healthy setup should finish `emu doctor` with:

```text
Ready to train.
```

The current known-good development stack is Python 3.13, JAX 0.11.2 on CUDA, and MuJoCo 3.15.

## Train

Start the recommended trainer from random actor/critic weights:

```bash
emu train
```

Defaults include:

- seed `733`;
- 2,048 parallel MJX environments;
- 32 control frames per PPO rollout;
- a genuine cold start;
- a 0→20M conservative nursery;
- the self-mapping graph trainer from 20→120M;
- quarantined held-out checks during training;
- a common final recovery exam.

Use another seed for an independent reproduction:

```bash
emu train --seed 1742
```

If a run is interrupted, rerun it safely with:

```bash
emu resume
```

## Watch it learn

The live capability viewer is read-only; it cannot influence training.

```bash
emu view
```

Then open:

```text
http://127.0.0.1:8765/
```

The viewer can show:

- command competence;
- conservative competence/confidence;
- continuous command quality;
- effective recent evidence;
- learning versus forgetting;
- staleness;
- integrated recovery frontier;
- maneuver coverage, competence and rate.

When the map is working well, the interesting picture is not just a bigger green blob. You should be able to see **mastered regions, uncertain boundaries, stale skills returning for rehearsal, and weak roads between otherwise strong commands**.

## Evaluate a trained robot

Run the physical capability atlas on the newest checkpoint:

```bash
emu evaluate
```

Or specify one explicitly:

```bash
emu evaluate --checkpoint /path/to/final.pkl
```

The atlas is for measuring physical limits. Routine held-out checks are intentionally smaller and are mainly used to answer a different question:

> Is the trainer's internal picture of its own capability actually honest?

## Compare the trainer against its control

The main v5012 experiment is matched shadow-vs-graph training:

```bash
emu compare
```

By default this runs seeds `733` and `1742`. Each seed receives its own genuine 0→20M foundation, then forks that exact checkpoint into two 20→120M continuations:

```text
same 20M network + optimizer + governor + normalizer + field
                    │
             ┌──────┴──────┐
             ▼             ▼
          shadow          graph
       map observes     map teaches
```

That is the causal question v5012 is designed to answer.

For a simpler two-seed reproduction of the recommended graph trainer only:

```bash
emu benchmark
```

## Robot and observation model

EMU models a six-actuator biped in `robot.xml`. The policy is intentionally restricted to deployable, sensor-style observations rather than unrestricted simulator truth.

The current actor observation is a short temporal history built from quantities such as gravity direction, body angular rate, joint position/velocity estimates, previous controls, foot-relative state, velocity estimates, command/phase information and shin/load signals.

The physical model and observation contract are versioned with checkpoints so incompatible source/model combinations can fail clearly instead of silently producing nonsense.

## Repository map

```text
README.md                 project overview
VERSION                   current source generation
pyproject.toml            package metadata + `emu` CLI
robot.xml                 physical robot model
emu_cli.py                small public command-line interface

environment.py            MJX physics, resets and task generation
training.py               PPO, governor, checkpoints and evaluators
capability_field.py       live command/recovery/maneuver graph
capability_cloud.py       held-out calibration/recovery probes
atlas.py                  physical capability atlas
field_viewer.py           live capability-map viewer
viewer.py                 robot/policy viewer
run_ladder.py             research campaign orchestration

docs/                     architecture, training, evaluation and history
tests/                    cheap semantic/regression tests
```

For normal use, start with the `emu` command. The older research interfaces are intentionally still available for experiments:

```bash
emu experiment --stage trainer --list
python run_ladder.py --help
python training.py --help
```

## Checkpoints and reproducibility

New checkpoints are written with simple canonical names such as:

```text
latest.pkl
best_eval.pkl
final.pkl
```

They also carry metadata describing the checkpoint/field schema, observation/action contract, robot-model hash, training step, seed and parameter shapes.

The normal resume path restores the actor/critic, optimizer, observation normalization, governor/curriculum and capability graph. MJX physics buffers themselves are recreated as fresh episodes rather than treated as portable checkpoint state.

## Project status

EMU is an active research/hobby project. The current focus is the **trainer**, not just a single champion checkpoint: cold-start reliability, efficient use of simulation experience, self-calibrated curriculum selection and dynamic locomotion capability.

The project has already demonstrated that graph-guided training can outperform the same curriculum with the graph operating in observation-only mode, and that the approach can reproduce a strong policy from an independent random seed. v5012 is focused on making that self-map statistically interpretable and easier to inspect, reproduce and share.

## Documentation

- [Architecture](docs/architecture.md)
- [Training lifecycle](docs/training.md)
- [Capability graph](docs/capability-map.md)
- [Evaluation](docs/evaluation.md)
- [Robot/model notes](docs/hardware-model.md)
- [v5012 design and acceptance criteria](docs/v5012-design.md)
- [Validation status](docs/validation.md)
- [Project history](docs/history.md)

---

*Current mission: teach a small unstable robot to know where it can go, how fast it can get there, and what it should practise next.*
