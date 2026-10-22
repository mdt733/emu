# EMU architecture — current experimental research branch

**Status:** v5020 is experimental. A green Python/CPU/MJX regression suite does
not establish successful 312-input locomotion. **Do not change `main` or
the public showcase until a stronger policy has been demonstrated and reviewed.**
The reference implementation is available at
[the immutable historical code on `main`](https://github.com/mdt733/emu/tree/main).

## One robot; three feedback loops

```text
                         Five-arm Champion Ladder
                 (frozen exam, compare to incumbent, promote)
                                  |
                common source checkpoint + five recipes
                                  |
  ┌──────────────────── evidence-driven curriculum ─────────────────┐
  │  governor: current lesson, difficulty, command/terrain frontiers│
  │                             |                                   │
  │   thousands of MJX worlds -- shared actor -- 32-frame rollouts  │
  │                  |                          ^                   │
  │                  v                          |                   │
  │        physical measurements           PPO gradients           │
  │                  |                          ^                   │
  │                  ├--> teaching rewards -----┘                   │
  │                  └--> aggregated training evidence --> governor │
  └─────────────────────────────────────────────────────────────────┘
                       |
               frozen policy snapshot
                       |
        independent, fixed, held-out physical examiner
                       |
             certified capability + eligibility
                       |
                 ladder promotion
```

The shared `scoring.py` now owns completed obstacle-trial accounting and exact-height exam validation. It rejects missing completed attempts and events without completed trials. Broader gait/tracking/support definitions are **not yet unified**.

The 2M probe is a separate **numerical safety and training-health gate**:
it reads all logged samples, preserves any per-world physics peaks,
weights final-window fall/episode length by completed trial counts,
and records recovered early collapse. It never teaches PPO or supplies
evidence to a frozen certification exam.

The same *definitions of physical events* must be used by all consumers;
their evidence sources and decisions must **not** be conflated.

| Mechanism | What it owns | Must **not** own |
|---|---|---|
| Physics, sensors (`robot.xml`, `environment.py`, `terrain.py`, `rolling_tof.py`) | Collision, actuators, on-robot observation, physically observable facts | Champion selection or training schedule |
| Reward (`rewards.py`) | Bounded per-step learning payments, including opt-in verified crossing | Proof of capabilities or held-out promotion |
| PPO (`ppo.py`) | Network and value updates from rollouts | Training difficulty or examiner selection |
| Metrics (`scoring.py`, `results.py`) | Episode counts, command tracking, stance, support, falls, effort, *physical crossing* | Scoring weights and policy promotion |
| Governor and capability field (`curriculum.py`) | Pool training evidence; choose next task envelope and retreat on failure | Certify unseen conditions with training rewards |
| Frozen examiner (`evaluation.py`) | Matched independent physical tests with counts, task context and confidence | Teach PPO or modify governor |
| Champion Ladder (`campaign_refork.py`) | Five independent arms, short health probes, identical exams, single winner / keep incumbent | Define physical truth or implement PPO |
| Lifecycle (`train_cli.py`, `training.py`, `checkpoint_io.py`, `experiment_io.py`) | Source identity, isolated runs, full-state resume and atomic reports | Interpret reward as demonstrated walking |

## The three notions of difficulty

1. **Physical safety ceiling:** legal bounds of this simulator/hardware;
   e.g. the existing adaptive box collision supports up to 150mm. Frozen
   5/10/20/40/80/150mm exam labels are now exact physical heights, not
   the upper limit of a random-height training sampler.
2. **Training frontier:** current challenge distribution, set from *on-policy
   physical evidence* by the governor and retained in the learning checkpoint.
3. **Certified capability:** what a **frozen** actor succeeds at with
   statistically adequate, held-out, same-condition examination.

Never label the frontier as a certified skill. The robot can sometimes pass an
unseen harder task without having trained on it. Report that generalisation,
including its sample size and success criterion.

**Read-only observation of this gap:** `python -m emu.competence_shadow
--metrics /path/to/metrics.csv --config /path/to/training_config.json`
reports the current training command limits, configured safety bounds, logged
on-policy edge tracking, and completion-weighted recent fall rates. It
produces stdout JSON only; does not change lessons, checkpoints, rewards,
run directories or CI training. Neither on-policy tracking nor a training
height frontier is an independently **certified** capability, and the
output explicitly reports certification as unknown.

**Current gap:** the Champion Ladder uses `open_frontier=False` and stages
`vx/vy/yaw` ceilings at 40/60/80M. The inherited adaptive command-frontier
logic exists (enabled in the mature `main` recipe), while the *separate*
physical step-height frontier is already evidence-driven. Converge on one
governor interface, initially in shadow mode; do not switch on every growth
axis during an active experiment.

## A physical fact has three different uses

Example: one **first successful real box crossing** has one physical definition.
It may (a) cause a small learning payment, (b) count toward aggregated training
height-frontier advancement, and (c) count as a success in an independent
held-out test. Those are *different datasets*. Physics event semantics may be
shared, but held-out exam evidence cannot become on-policy teaching evidence.

Training examples must not secretly expose teacher truth through the 312-input
actor; observation schema, history, sensor delay and extrinsics are checkpoint
contracts. `main` retains the 232-input reference and older 220-input
legacy actor. Their weights are not interchangeable with 312-input policies.

## Current experimental campaign

```text
              validated compatible ~20M foundation
                             |
                 five 20→40M continuations
      general | turning | recovery | efficiency | steps
                             |
           matched held-out physical exams + incumbent
                             |
                      ONE winner / STOP
                             |
              five 40→60M continuations
                             |
                      ONE winner / STOP
                             |
              five 60→80M continuations
                             |
                    optional 80→100M
```

Each candidate uses its own 2M-step **health probe** before its remaining
continuation. One candidate's 20M increase costs 20M environmental steps, so
five full survivors cost ~100M total environmental steps per generation.

The Steps arm begins with low obstacles and grows *from completed crossing
evidence*, with mixed easier/flat rehearsals and automatic hold/retreat.
Frozen exams try all five arms at 5, 10, 20, 40, 80 and 150mm: these heights
are **test levels, not training limits**. Moving past an obstacle without
realistic supported crossing must not count.

## Permanent engineering invariants

- Do not write to `main`, existing demo, archived checkpoints or existing
  run roots. All source experimentation occurs on the already selected v5020
  branch. Only promote to main after explicit user approval.
- Train from validated snapshots with the entire learning state: policy,
  optimiser, critic, observation normalisation, governor, field, health and
  source provenance.
- Keep old snapshot semantics through strict checkpoint schema and physical
  sensor-contract validation; any change to observations or physical model
  meaning requires explicit compatibility/retraining decisions.
- A valid study must distinguish absent episodes, failed attempts, one-time
  success, task exposure and independent testing. No evidence ≠ 0% success.
- Safety, core survival and stable gait are eligibility gates. A weighted
  improvement in steps or energy must not excuse an unusable walker.
- Use matching seeds/exam configurations across all contenders, and version
  their frozen exam doctrine alongside every score.
- Short probes catch early catastrophic failures, but a probe pass does not
  certify later training stability.
- Training source commits and evidence records are immutable; results and
  checkpoints stay out of Git.

## Read the system in this order

`contracts.py` → `robot.xml` → `environment.py` / `terrain.py` →
`rewards.py` → `ppo.py` → `results.py` →
`curriculum.py` → `evaluation.py` → `campaign_refork.py`.

For historical versions, old experiment decisions and lessons, read
[`RESEARCH_HISTORY.md`](RESEARCH_HISTORY.md), **not stale version-by-version
stories embedded in current code comments**.

For staged consolidation, see [`V5020_CLEANUP_PLAN.md`](V5020_CLEANUP_PLAN.md).
