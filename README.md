# EMU

**EMU is a reinforcement-learning project for a small, dynamically unstable six-actuator biped.**

The robot is not meant to solve locomotion by finding a static pose. It has small feet, springy mechanics and a deliberately limited deployable sensor set, so useful locomotion means continuously stepping, catching itself and adapting to commanded motion.

The research question is broader than “can PPO eventually produce one good policy?”:

> Can a compact trainer repeatedly build a capable robot from random weights, measure what it can actually do, and spend training effort where it is useful without teaching from the exam?

## The system

```text
MuJoCo / MJX environment
        │
        ▼
       PPO
        │
        ├──────────────► capability field
        │                     │
        │                     └── bounded teaching proposals
        │
        ▼
 curriculum governor
        │
        ▼
 next training distribution

held-out evaluation ───── read only ─────╳ training
```

The runtime responsibilities are intentionally separate:

- **Environment** — robot physics, deployable observations, commands, disturbances and reward.
- **PPO** — actor/critic optimisation and observation normalisation.
- **Curriculum** — global task difficulty and demonstrated physical frontiers.
- **Training health** — PPO stress, collapse detection and rollback requests.
- **Capability field** — training-derived evidence about commands, recovery and maneuvers.
- **Held-out evaluation** — independent exams that never teach.

## Actor observations

The policy receives a four-frame history of a 55-value deployable observation vector: **220 values total**. It includes gravity direction, gyro, joint state, estimated motion, previous control/action, relative foot geometry, estimated planar velocity, ToF height, confidence, command/phase and shin-compliance signals.

The actor does not receive privileged simulator state merely because MuJoCo has it.

## Capability field

The field is a low-resolution map over commanded:

```text
(vx, vy, yaw rate)
```

It records three kinds of semantic evidence:

- **Node event** — sustained operation at one command.
- **Recovery event** — one disturbance followed by recovery or failure.
- **Edge event** — one complete command transition followed by settling.

Evidence is event-based rather than frame-count based. Old evidence decays and confidence matters. Node tracking quality is retained separately from Bernoulli competence; recovery and maneuver layers keep only evidence that modern teaching/reporting actually consumes.

The field can run in two useful modes:

- `shadow` — observe only.
- `graph` — observe and propose bounded practice to the governor.

This makes a matched shadow-vs-graph fork the core causal experiment.

## Training shape

The standard experiment is deliberately simple:

```text
random weights
     │
     ▼
0 ─────── 20M target     shared foundation
                  │
                  ├──────── shadow ─────► 120M
                  │
                  └──────── graph  ─────► 120M
```

PPO advances in whole vectorised rollout updates, so the handoff is the **first complete update boundary at or beyond 20M**, not a partial update forced to land on the decimal number. With the default 2048 envs × 32 rollout this is step **20,054,016**. Within one seed, both mature arms resume that same validated checkpoint, including network, optimiser, curriculum, training-health, observation-normaliser and field state.

The foundation may retreat curriculum difficulty, but it does not rewind the network. Rollback is a mature-training emergency mechanism, not a nursery teaching strategy.

## Quick start

EMU is developed on Linux/WSL with Python 3.13, MuJoCo/MJX and a CUDA-enabled JAX install.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
# Install the JAX build appropriate to this machine first (CUDA example):
python -m pip install -U "jax[cuda13]"
python -m pip install -e .

emu doctor
```

`emu doctor` checks Python, JAX backend, MuJoCo/XML, field semantics and a tiny MJX reset/step.

### Train one graph policy

```bash
emu train
```

### Run the matched experiment

```bash
emu compare
```

### Run the efficient v5013 teacher tournament

Inspect the adaptive twelve-hypothesis funnel without starting JAX:

```bash
emu tournament --plan
```

Then run it. The tournament shares foundations, screens weak arms at 50/60M, promotes only four to 80M and only two challengers to 120M, then confirms the best challenger against shadow on seed 1742:

```bash
emu tournament
```

See [`docs/tournament.md`](docs/tournament.md) for the hypotheses, guardrails and restart-safe promotion rules.

For a simple second independent paired comparison instead:

```bash
emu compare --seed 1742
```

### Live capability map

```bash
emu view
```

### Physical capability/recovery atlas

```bash
emu evaluate
```

## Storage

Heavy run data is intentionally outside Git. By default a repository cloned as:

```text
/path/to/emu/
```

uses:

```text
/path/to/emu_runs/<version>_seed<seed>/
```

Override this with:

```bash
export EMU_RUN_BASE=/somewhere/fast/emu_runs
export EMU_RESULTS_BASE=/somewhere/emu_results
```

Small result summaries may be copied under `results/`; checkpoints and raw telemetry stay out of the repository.

## Evaluation philosophy

EMU keeps three questions separate:

1. **Can the policy perform the command?** — node competence.
2. **Can it recover from a disturbance?** — recovery competence.
3. **Can it execute a transition?** — edge/maneuver competence.

Those metrics must not be calibrated against one another as if they were the same event.

Physical capability is reported as curves/frontiers in real units where possible. Fixed regression suites remain useful, but an arbitrary legacy coordinate is never treated as a physical limit.

## Repository

```text
emu/
    cli.py              small public CLI
    paths.py            filesystem/resource policy
    contracts.py        exact observation + fixed JAX packet contracts
    training_config.py  foundation/mature profiles + fixed doctrine
    environment.py      MuJoCo/MJX robot + task environment
    ppo.py              network, normalisation, PPO + rollout kernels
    curriculum.py       task difficulty/allocation + physical frontiers
    health.py           PPO stress + collapse/rollback requests
    rollout_metrics.py  host reduction of training evidence
    training.py         training orchestration
    evaluation.py       deterministic read-only regression panels
    capability_field.py training-derived command/recovery/maneuver map
    tracking_contract.py shared locomotion competence thresholds
    capability_cloud.py sparse held-out node/recovery examination
    checkpoints.py      canonical checkpoint discovery
    checkpoint_io.py    strict schema-7 save/load + snapshots
    runtime.py          JAX/backend setup + performance bookkeeping
    artifacts.py        run artifact/log I/O
    atlas.py            physical command/recovery sweeps
    experiments.py      foundation + matched fork orchestration
    tournament.py       successive-halving 12-hypothesis experiment
    screening.py        cheap held-out tournament stress probes
    field_viewer.py     read-only live map viewer
    VERSION             package version identity
    robot.xml           robot model

tests/                  contract/regression tests
docs/                   current design + concise history
```

The current code intentionally does **not** carry every old experiment runner or compatibility path. Git history is the archive. If an old ablation becomes useful again, it can be reintroduced deliberately as a small isolated experiment instead of remaining permanent runtime baggage.

## Project rules

A few rules have become more valuable than any individual hyperparameter:

- Held-out evaluation never teaches.
- Resume either restores the complete contract or fails loudly.
- One definition of a metric lives in one place.
- Random-start foundations are first-class; old reference checkpoints are not architectural dependencies.
- Compare paired arms rather than declaring winners from opaque aggregate scores.
- Preserve physical-unit evidence and raw curves instead of hiding capability in a scalar.
- Prefer deleting obsolete experiment machinery over indefinitely supporting it.

See [`docs/history.md`](docs/history.md) for the lessons that led here.
