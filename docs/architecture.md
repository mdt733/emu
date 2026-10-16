# EMU — how the system works

> **The aim:** build an understandable system that discovers good ways to *teach*
> a six-servo biped to walk. A capable robot is evidence that the teaching system
> works—not the only thing we're trying to optimise.
>
> **Read this page first.** It is the architectural contract for simplification,
> not a claim that the code already matches the ideal layout. See
> [the living worklist](SIMPLIFICATION.md) for mismatches and completed passes.

## The whole story

```text
Human chooses hypotheses                  ladder_recipe.py / branch_lab.py
        │                                   "What training should we compare?"
        ▼
Curriculum proposes next challenges        curriculum.py
        │                                   "What can this student practise?"
        ▼
Training runs many parallel worlds         training.py
        │              ┌──────────────────────┐
        ├──▶ Robot + universe                   robot.xml / environment.py
        │              │                       "What physically happened?"
        │              ▼
        ├──▶ Reward for each control step       rewards.py
        │              │                       "How useful was that behaviour?"
        │              ▼
        └──▶ PPO learner                        ppo.py
                       │                       "Update the actor and critic."
                       ▼
                 Saved checkpoints
                       │
              ┌────────┴───────────┐
              ▼                    ▼
       Reports + exams         Viewer/export
       results.py              policy_export.py
       evaluation.py           viewer_src/
       "What was learned?"     "What does it actually look like?"
```

The curriculum also learns from the rollout's **independently measured diagnostics**;
the experiment runner compares *complete learning histories*, not just final rewards.

## The seven responsibilities

| Layer | Should own | Should **not** own | Current code |
|---|---|---|---|
| **1. Universe** | Physical robot, collision and actuation; reset/step, sensors and measured events | Reward weights, PPO and experiment selection | `robot.xml`, `environment.py`, `domain_randomization.py` |
| **2. Rewards** | Pure function: measured event + command → one teaching signal and component payments | Physics, curriculum policy or winning a tournament | `rewards.py` |
| **3. PPO** | Actor/critic, rollout collection, optimisation, observation normalisation | Curriculum selection, reporting, run promotion | `ppo.py` |
| **4. Training loop** | Compose thousands of independent robot/world samples, manage PPO timing and durable training state | Embed reward maths or change winning criteria | `training.py` |
| **5. Curriculum** | Choose allowed tasks and difficulty from accumulated *training* competence, then sample commands/disturbances | Change actor weights or read held-out tests to teach | `curriculum.py` chooses difficulty; `tasks.py` samples each world's lesson |
| **6. Experiment design** | Shared foundation, independent branches, controlled teaching strategies, equal-budget comparisons, ancestry and promotion | Reimplement the trainer or change old checkpoints | `ladder_recipe.py`, `branch_lab.py` |
| **7. Understanding** | Diagnostic history, held-out examinations, comparisons, shareable report and optional frozen-policy viewer | Quietly influence learning or declare reward equal to walking | `rollout_metrics.py`, `evaluation.py`, `results.py`, `policy_export.py`, `viewer_src/` |

The **viewer sits beside the process**, not in the learning loop. It loads a
frozen actor, converts/export it as `.emupolicy`, and provides a browser
demonstration. A nice-looking demo is not a calibrated held-out test.

## Follow one robot world through its lifecycle

```text
reset()
  └─ _reset_impl → _reset_with_keys
                       ├─ tasks._sample_task         request a lesson
                       ├─ _birth_physics             initial physical pose/velocity
                       ├─ MuJoCo forward             compute initial physical state
                       └─ _base_obs + history        prepare the 220-input actor

step(action)
  └─ _step_impl
       ├─ _simulate_physics                         actuate → push → MJX
       │      └─ rollback nonfinite worlds          always reset them
       ├─ tasks._advance_commands                   NEXT lesson only
       ├─ _metrics + rewards.training_reward        score PREVIOUS command
       ├─ _base_obs                                 observe NEXT command
       ├─ _reset_with_keys                           replace failed/timeouts
       └─ StepOut                                   reward + pre-reset evidence
```

**Critical timing:** PPO's action came from the *previous* observation.
The teacher may advance its requested command while physics is stepping, but
the current reward is always measured against `state.task`, not the new
request. The *next* 55-channel observation includes the advanced command.
Episode bootstrap uses the observation **before** reset, while ordinary
next-state observation may be from a freshly reset world. Timeout episodes
do not receive the early-fall penalty.

All simulation methods remain JAX-traceable and share the same random key
splits, world variations and six-servo motor model. `PhysicalFrame` and
`CommandFrame` are lightweight named returns **within the same JIT trace**,
not independent simulators, controllers, or new learning state.

## The capability teacher, precisely

```text
TRAINING rollout metrics ──► CapabilityField.update_from_rollout
                                  │  optional field_taught exclusion
                                  ▼
                          Decayed node/recovery/edge evidence
                                  │
                          teaching_packet(step, mode)
                                  │
             ┌────────────────────┴─────────────────────┐
             ▼                                          ▼
  tasks._sample_task (birth)                  tasks._sample_cmd_target
  weighted node command                       node / local edge transition
             └───────────────────┬──────────────────────┘
                                 ▼
                    governor-clipped safe lesson
                                 ▼
                    physics + PPO rollout (not held-out)
```

The capability field provides *where to practise*, not permission to exceed
the physical frontier chosen by `RegimeGovernor`. Node anchors can bias birth
commands, graph edge anchors can bias mid-episode TRANSITION targets, and the
`field_taught` marker allows field-selected experience to be withheld from
field self-evidence while still teaching PPO. Held-out cloud results may query
the field for exam calibration, but cannot update its learned state. These are
executed mechanisms: switching off influence does not mean they are dead code.

## The data going between layers

1. **Experiment recipe** fixes source version, shared parent, random seeds,
   simulation budget, curriculum strategy and examination protocol *before*
   a comparison. Only named experimental variables are allowed to differ.
2. **Curriculum** describes desired movement and allowed terrain/robot/world
   variations. A requested velocity is not a reward or a completed movement.
3. **Environment** advances physical time with six actuator inputs, outputs
   sensory observations and factual measurements (positions, contacts,
   torque, falls, commands actually delivered). A contact-height proxy is
   not force; foot lift is not a confirmed walking stride.
4. **Rewards** turn the factual measurements into a scalar PPO payment, and
   report individual components without losing the physical measurements.
   This teaching signal may be gamed; it is never the exam result.
5. **PPO** learns from trajectory data and returns an updated policy. The
   curriculum receives a *reduced summary* of what the rollout actually did,
   not direct access to optimised weights.
6. **Training** saves enough state to continue the learning lineage
   (actor, critic, optimiser, normaliser, curriculum and related state).
   Physical MuJoCo episodes can restart when training resumes.
7. **Independent reports** use true signed course progress, command
   obedience, survival, gait evidence, load and time trends. They compare
   branches on **identical external examinations** and preserve enough
   data to challenge the interpretation.

**Observation versioning:** v5015 trained policies use 55 measurements × 4
frames = 220 inputs and six outputs. The v5016 61 × 4 = 244 candidate is
declared but NOT deployed. No number is sacred; an input change needs a new
well-documented actor/extractor/export/viewer path, not silent reinterpretation
of an older checkpoint.

## One training iteration, in human terms

```python
# Explanation only; this is NOT an alternative trainer or executable code.
teaching_task = curriculum.choose_tasks(training_evidence)
trajectories = universe.simulate_many(actor, teaching_task)
payments = rewards.evaluate(trajectories)
actor = ppo.learn(trajectories, payments)
capability = diagnostics.summarize(trajectories)  # independent facts
curriculum.observe(capability)
save_results_and_checkpoint_when_due()
```

In the real implementation, **each physics control frame** can earn reward.
Many frames across many worlds form a PPO rollout/update. Curriculum decisions
happen after observing rollouts; an experiment compares *many updates* across
millions of world-steps. This diagram is the explanatory order, not a demand
to add Python wrappers around JIT kernels or undo efficient fused MJX scans.

## Where teaching ends and evidence begins

The reward function is deliberately near the top of **`emu/rewards.py`**.
Reading it shows exactly what PPO learns from:

| Teaching payment | Coefficient | Physical proxy used |
|---|---:|---|
| Survival | 0.58 | Not terminated |
| Locomotion | 1.55 | Velocity/yaw/pitch obedience, sustained motion, posture and foot support |
| Posture | 0.68 | Body height/orientation and chassis-to-feet geometry |
| Contact | 0.50 | Feet near the floor and low body-contact proxies |
| Anti-slip | 0.18 | Estimated horizontal speed of feet near the floor |
| Clearance | 0.16 | Foot lift and emerging gait shape |
| Smoothness | 0.12 | **Change in PPO action**, not motor current or power |
| Joint safety | 0.17 | Proximity to joint limits and leg scissoring |

Posture and contact payments are reduced by the **episode-long stepping
activity gate**. A separate age-dependent **early-fall penalty** applies to
genuine failure, not episode timeouts. The environment scores an action
against the command the actor knew *before* command transition sampling.

**Servo power is not currently rewarded.** `environment._metrics` measures
torque, nominal shaft mechanical power, spring energy and rated-torque
pressure; `rollout_metrics.summarize_rollout` reports them as independent
diagnostics. Electrical draw, servo temperature, actual joint contact forces
and verified footfalls require further modelling or physical data. Adding
energy efficiency belongs in a controlled future experiment, not a hidden
change during refactoring.

The **stable data boundary** is:
`environment._metrics` (world facts)
→ `rewards.training_reward` (teaching scalar and payments)
→ `ppo.py` (learning), while
`rollout_metrics.summarize_rollout` independently reduces the world facts
and labelled payments for reports/curriculum. The lists of reward-only
transient fields and weighted payment names live in `rewards.py`, not
in the other owners. True locomotion quality still needs held-out exams.

## Design rules for future passes

- **One obvious owner per concept.** To change slope sampling, go to the
  curriculum/world task boundary; to change the cost of servo use, go to
  `rewards.py` after verifying physical telemetry; to change PPO mathematics,
  go to `ppo.py`.
- **Read the story before the details.** The public function for each
  responsibility should explain its inputs, outputs and steps in order.
  Prefer named coherent chunks to either a 900-line mystery function or
  a forest of one-line wrappers.
- **Physics facts, teaching reward and independent evidence must not blur.**
  Old slope tests at roughly 1° did not prove uphill locomotion. Requested
  direction stays fixed independently of robot yaw in held-out exams.
- **No historical compatibility scaffolding by default.** Keep experiments
  and old brain formats available at their original Git revisions. Remove
  unused runtime branches when evidence proves they are superseded.
- **Never spend a long GPU budget to test a refactor.** Use 20 small software
  checks and then a **fresh, isolated, short real GPU lifecycle/exam** before
  treating modified training source as trustworthy.
- **One current plan, one current report.** Experiment results get a
  generated `results.md` and evidence ZIP; this project gets one living
  [simplification checklist](SIMPLIFICATION.md). Old pass notes are historical,
  not extra layers of instruction.

## Current gaps (not hidden)

- `tasks.py` owns lesson sampling **and command-ramp advancement**, while `environment.py` owns servo delivery, disturbances, MJX stepping, physical birth, observations and reward integration. The stateless sampler shares the environment's JAX context, physical limits and keys; it adds no duplicate mutable teacher. The remaining `_metrics`/`_base_obs` block is large and still merits a focused readability pass.
- Reward owner and physical telemetry boundary have been clarified; posture,
  support, step shape and slip are still **proxies** and should be compared
  against independent real gait evidence. Torque/power remain diagnostic-only.
- `training.py` still owns fixed/cloud **exam cadence and policy snapshot selection**; `evaluation.py` and `capability_cloud.py` now execute and record the exams. These examine frozen policies without feeding evidence back into PPO or curriculum.
- `artifacts.py` now owns the exact metrics.csv row, envelope report and latest/final field exports; training passes PPO run settings explicitly, so reporting does not import the learner's configuration. Training retains scheduling and checkpoint publication order.
- `run_support.py` now owns complete collapse recovery: validate rewind candidate, restore all learner/teacher state, and reset physics using a step-derived JAX key; training rebinds the successful result and chooses its subsequent lesson.
- `health.py` owns PPO KL-stress expansion gating, global-collapse detection, rollback candidacy and the soft KL learning-rate adjustment; the trainer explicitly passes measured training KL and controller bounds.
- `training.py` is still a long function combining loop control, curriculum,
  learning health, snapshot recovery, fixed exams, logging and finalisation.
- `ppo.py` still houses **read-only** exam kernels alongside training maths.
  This may be practical for shared JAX compilation but should be explained.
- The physical-frontier decision in `curriculum.py` is now a named method of the same governor; it is driven only by settled training-rollout tracking evidence and preserves the pre-existing hysteresis, retreat and safety ceilings.
- Capability-field teachers and periodic evaluation add complexity whose
  actual value should be demonstrated before they become permanent doctrine.
- v5016 direction/heading/ground-plane modules are experimental examination
  machinery, NOT newly trained terrain-aware PPO observations.

See [Simplification](SIMPLIFICATION.md) for the next actual code change, and
[Testing](TESTING.md) for the 20 retained software checks.
