# Training lifecycle

The durable principle is **one learning engine, many replaceable experiment recipes**. The current `emu ladder` command is an experimental successive-halving runner; `emu/training.py` is the reusable PPO training orchestrator and never decides which research hypothesis deserves more GPU compute.

## Responsibilities

| Owner | Decides / produces |
|---|---|
| `ladder_recipe.py` | Absolute step targets and typed, named differences between hypotheses |
| `branch_lab.py` | Forking a shared ancestor, matched screens, hard eligibility and automatic/manual promotion |
| `training_config.py` | The exact serialised knobs for one training run |
| `training.py` | Lifecycle of updates, observations, governor/health, evaluations and artifacts |
| `ppo.py` | Vectorised rollout, actor/value optimisation, update diagnostics |
| `evaluation.py` | Fixed read-only questions about a specific saved actor/normaliser |
| `checkpoint_io.py`, `run_support.py` | Atomic save, validated resume, exact-target handoff and source provenance |

## One update boundary

A real PPO step advances by `envs × ROLLOUT_STEPS` environment transitions. Targets are **absolute** environment steps, rounded up to a complete boundary. The training process compiles and performs its first update, processes that rollout, and repeats until it reaches the target. A resume already at the target returns without applying another optimiser update.

Within that loop, EMU:
1. Collects MJX experience with the current command/curriculum parameters and a particular actor/normaliser.
2. Updates the actor and critic with PPO; retains the rollout-generating actor separately for accurately attributed rollout metrics.
3. Reduces command tracking, stepping and performance metrics; updates the health monitor and curriculum governor.
4. Lets the capability field propose bounded practice; the governor constrains permitted difficulty. Held-out exam results **never** teach the field.
5. Runs read-only fixed or cloud evaluations when due, on the *post-update actor* and matching normaliser.
6. Saves validated `latest.pkl`, rotating snapshots, `best.pkl` and `recovery.pkl` for distinct purposes. `final.pkl` is published atomically from a validated latest checkpoint.

`best.pkl` is an evaluation winner, `recovery.pkl` is a safe fallback for a degraded learner, `latest.pkl` is the most recent learner state, and `final.pkl` is the completed stage. **They are not interchangeable.**

## Resume semantics

Schema-7 checkpoints preserve actor/critic, optimiser, normaliser, governor/curriculum, health, capability field and learning-rate controller state. MJX episode buffers and simulator RNG are reset: this is a **warm learning-state resume**, not a byte-identical continuation of the prior physics episode.

On a complete resumed target, no new PPO update should execute. Checkpoint step, robot model, policy dimensions and training configuration must agree before resuming; incompatible states are rejected rather than guessed around.

## Teaching and experiment knobs

A candidate in `emu/ladder_recipe.py` declares `Arm(name, hypothesis, knobs=((field, value), ...))`. The runner validates **every candidate before starting** the shared foundation. Unknown/duplicate field names, incompatible/nonfinite values, unsafe arm names and changes to protected run identity are rejected.

Knobs include movement envelopes, curriculum/frontier limits, domain perturbation radii, capability-teacher influence/schedule, normaliser freeze, and PPO learning rate/entropy/value-loss weights. **The current arm defaults remain unchanged unless that arm explicitly overrides a field.** Network architecture, rollout shape and the 220-input/six-output actor contract are not general-purpose knobs.

The older `forge` and v5013 tournament implementations are archived in Git history, not parallel current execution paths. See [Experiments](tournament.md) and [History](history.md).

## What the trainer does not prove

High PPO reward, a green CI build or a browser robot that appears to walk is not a reliable locomotion metric. Fixed screens currently include survival, tracking and stepping telemetry, but stepping thresholds still need calibration. Long-horizon posture drift and real contact-ramp ability require independent testing.

[Architecture](architecture.md) · [Evaluation](evaluation.md) · [Validation](validation.md)
