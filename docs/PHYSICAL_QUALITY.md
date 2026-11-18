# EMU physical quality: what is a good robot?

**Independent examiner contract (originating 31 October 2026).** Read-only
physical evidence; **not** a PPO reward function or training schedule.
The current H40 120M `main` actor has **not** yet passed matched
held-out signed-step exams. Historical v5032/v5048 results remain
research baselines, not evidence for H40.

## The north-star definition

**A good robot reliably completes the movement we actually asked it to
perform, across increasingly hard physical situations, while remaining
stable, controllable, recoverable, graceful and economical.**

In particular:

1. **Survival is necessary, not sufficient.** Standing still, reversing
   away from the step, sliding downhill or walking in a circle are not
   complete locomotion missions.
2. **The requested challenge is not a capability.** Credit a 3cm step
   only for verified 3cm collision geometry, genuine far-side supported
   contact, and continued commanded movement. Step UP and DOWN are
   completely separate physical outcomes.
3. **The route is in world coordinates.** Good body-frame velocity
   tracking can coexist with a gradual unintended turn. Compare against
   an independently integrated *commanded* trajectory, not the actor's
   actual heading as the definition of the desired route.
4. **Alternating gait yaw is not necessarily loss of control.** Measure
   signed yaw drift, accumulated world-frame heading/path error and
   stride-related jitter **separately**. Do not penalise natural gait
   oscillation merely for existing.
5. **Efficiency is per useful, supported, directed progress.** A robot
   should not improve its rating by stopping or deliberately walking
   a long zig-zag to inflate its path-length denominator. Simulated
   motor shaft J/m is not battery energy; spring-storage changes do
   not prove recovered mechanical work.

## Our six-dimensional physical profile

| Dimension | Real test tasks | Primary proof |
|---|---|---|
| **Locomotion** | +/−forward, +/−lateral, +/−yaw, mixed commands | Useful, sustained travel at achieved requested speeds |
| **Control** | Heading, turns, braking, start/stop, transitions | Actual path and turn accuracy; small *persistent* bias |
| **Terrain** | Four slope bearings, real step up/down, oblique lips, mixed | Verified geometry, genuine supported transfer and progress |
| **Resilience** | Pushes from different bearings, posture, kick, recovery | Applied disturbance **and** return to commanded movement |
| **Grace** | All useful walking tasks | Good stance contact, controlled foot/torso motion, little needless actuator thrashing |
| **Efficiency** | Matched, completed locomotion tasks | Mechanical work per *useful* metre without speed/survival regressions |

**Robustness/generalisation** is an essential cross-cutting seventh
perspective: repeat independent seeds and inspect weakest direction
and failure tail, not just the average.

The teacher's six *training focus groups* are **not** these six
independent exam dimensions. Training exposure, reward, learned
difficulty and physical proficiency are deliberately different.

## When is slowing down actually the RIGHT answer?

There are **two distinct tests** and we must not mix them:

- **Low-level velocity-command examiner:** when we request 0.15m/s
  downhill, can the learned motor controller track that command and
  steer without drifting? Slowing to 0.04m/s is an intentional *miss*
  of this exact command, even if it survives.
- **High-level destination/safety examiner (future):** when we ask
  the robot to reach a target beyond a steep descent, **choosing a
  safer speed or a brief preparatory crouch is GOOD** if it still
  completes the route in finite time with genuine support and acceptable
  progress. Stalling, walking away from the destination or cycling
  in place cannot earn safety credit.

This distinction applies to step-ups too. Pre-extending the legs,
loading springs, carefully testing a foothold and adjusting stride
timing can be competent behaviour; none should be penalised just
because the torso departs from its nominal height. The *outcome*
matters: safe, supported arrival, not obedience to one idealised
walking pose.

Current v1 reports the existing **low-level commands** plus
world-frame route diagnostics. It does NOT falsely claim to test
autonomous terrain-aware route planning or discretionary slowing yet.

## From individual cases to an honest capability envelope

Each frozen case reports:

- All-world survival; all-world movement tracking; signed yaw bias
  and jitter; world-frame route endpoint/RMS error; verified support;
  actual step contact or grade; full mission; historical strict MAE.
- Per-direction, per-severity **missions passed / all attempted worlds**,
  and a Wilson 95% interval. 8/8 is not proof of 100% reliability.
- If **two or more severities were actually tested** in the same
  direction and physical unit, a *trapezoidal tested-range area*
  under the observed mission-success curve. Show interval endpoints
  and the raw integrated area. Do **not** extrapolate above/below
  the sampled range. A single point has **no AUC**, not zero AUC.
- Achieved physical speeds (not mere requested envelopes), real
  supported crossing height/direction, and mechanical J/m for
  **physically supported useful directed motion**. This efficiency
  diagnostic does not vanish just because the old yaw-jitter threshold
  rejected an otherwise supported walk; its heading failure remains
  visible in a separate column. Missing observations are
  **null / not measured**, never invented as a passing or failing zero.
- Separate per-seed outcomes and the complete underlying world rows.

There is **no intrinsic cap at 1m/s, 15° or 4cm**. Those are current
exam fixture limits, *not* maximum definitions of robot quality.
When a stronger robot emerges, safely expand the independent task grid.
Worlds with different fixture physics, clock horizon, command schedule
or seeds cannot be silently ranked as if paired.

## Goodness is a profile, not a magic number

The robot's **competence** is the area under its successful physical
capability curves, qualified by control and support, with no hidden
hard ceilings. But averaging six arbitrary percentages into one
number can hide a robot that's great uphill and unable to descend.
Therefore the first examiner deliberately provides *no global score*.

A future release decision should (1) verify same physical fixture and
source identity; (2) reject non-progress/unsupported tricks;
(3) compare each signed dimension on identical frozen cases with
repeated seeds; (4) identify important regressions and weak tails;
(5) inspect graceful and efficient execution **conditional on useful
successful work**. Only then consider an explicit, reviewable
aggregate—not one that can erase a catastrophic weakness.

## Current implementation: real prisms, frozen actor

The historical `emu capability-exam` and `emu physical-quality` commands
belonged to a two-slab physical examiner and are **not executable in
v5045**. Their original scoring protocols, reports and exact command
recipes remain in the [pinned pre-cleanup source](https://github.com/mdt733/emu/tree/a2f59bb31bec049204b60a3eb86500b100db039f).

The maintained `emu prism-exam` instead evaluates a frozen, true-prism
360-input actor with raw MJX contact and observed triangle evidence.
It measures actual grade, supported signed up/down crossings,
command-following velocity, survival and positive mechanical shaft
work per directed progress. **A requested frontier is not a guaranteed
fixture**, and these procedural results must not be compared to old
two-slab fixed-angle pass rates as though they were identical exams.

For installation, the frozen-actor validation gate and new v5045
usage, see [the current physical release gates](RELEASE_READINESS.md).
The **current H40 120M** source and inference actor are on `main`; do
not transfer the previous 260M actor's independent exam scores to it or
fabricate matched-prism results.

See [issue #17](https://github.com/mdt733/emu/issues/17) for research
questions and the [retained research findings](RESEARCH_FINDINGS.md) for
the earlier 260M physical baseline and release card (linked by immutable
Git source).

---

# EMU Physical Capability Examination v1 — 18 November 2026

**Status: research / uncalibrated. Not a champion selection system.**

Research branch: `research/physical-capability-exam-v1`. Base source:
`78a261c22487871137c4c957a136080586842694`. Frozen
`showcase/champion.emupolicy` is the owner-approved 120M H40 actor.

This document is an **audit of the named code at that base commit**, not a
claim that these experiments have been executed or that Spring V1 has passed.

## Why the previous numbers were misleading

The earlier `scripts/v5053_matrix_exam.py` uses a **350-control-frame
default** (7 seconds at `CONTROL_DT=0.020`), with a permitted horizon of
80–700 frames (1.6–14 seconds). Its nominal `flat_control` explicitly commands
**+0.25 m/s**, not stillness. Its `stand` task injects a signed initial
velocity perturbation (up to 0.30 m/s) but has no bounded-position
requirement. In the matrix reducer `survived` means that the first physical
life lasted to the requested horizon; it is not a measure of commanded speed,
upright endurance, zero-command balance or travel direction. Independent
destination and traversal gates are more demanding but answer different
questions. A 7-second survival percentage can be very high for a robot that
falls regularly over 60 seconds.

The legacy matrix **does** intentionally stop credit at the first terminal,
retain world-level records, measure exact physical seam height, and record
source/actor/model fingerprints. These are valuable and should be retained.

The old `emu/evaluation.py` is **training retention**, not a physical
certification. Its four panels (`core`, `dynamic80`, `combo70`,
`resilience75`) default to the same 350 frames. It calls
`ppo.evaluate_summary`, which stores first fall but continues stepping the
auto-reset environment and averages tracking/support/posture across subsequent
lives. Its broad weighted score may therefore blend a failed initial
episode with later reset episodes. This may be useful for retention but cannot
answer first-life endurance. The panels also randomise commanded conditions
rather than fixing a zero-command stillness test.

`emu/prism_exam.py` performs physical contact inspection of procedurally
generated true prism terrain; its default horizon is 300 frames (6 seconds),
with an allowed maximum of 650 (13 seconds). It masks everything after the
first terminal for independent crossings, and gives per-world gait, contact
and energy diagnostics. Actual encountered terrain severity is measured,
not assumed to equal the teacher's requested frontier. It is a useful
research tool, not a long-term generalist exam.

The visual Infinite Meadow uses its own browser CPU terrain/heightfield
workflow. It is **qualitative** until per-frame observation, motor and
collision parity with the exact MJX scene is verified. The new visual world
must not be replaced or silently treated as the GPU ground truth.

## Exam v1 — first delivered slice: flat balance, locomotion, transitions

Entry point: `python -m emu.physical_capability_exam`.

The exam uses the **existing** `make_trainer()["evaluate_trace"]`, which
runs the immutable actor's deterministic mean action on the actual MJX
environment. It reuses the actor sensor/normalizer contract, physical model
and servo target/actuation pipeline. No policy optimiser update, PPO loss or
curriculum promotion occurs. `main`, viewer, and training code are unchanged.

The examiner lengthens `max_episode_steps` **beyond its longest test** and
disables Academy success terminals. Initial command and per-frame command
channels are deliberately applied through the established trace evaluator,
rather than modifying the reward engine.

| Scenario | Duration | Operator instruction |
|---|---:|---|
| Zero-command standing | 60 s | vx=0, vy=0, yaw=0 |
| Forward | 30 s | vx=+0.25 m/s |
| Reverse | 30 s | vx=-0.20 m/s |
| Left/right lateral | 30 s each | vy=±0.15 m/s |
| Left/right yaw | 30 s each | yaw=±0.50 rad/s |
| Combined | 30 s | vx=+0.20, vy=+0.10, yaw=+0.30 |
| Stop/reverse | 30 s | 10s +0.25, 10s zero, 10s -0.20 m/s |

Each world gets its own deterministic reset key. **The first terminal ends
its measured life forever**; later automatic resets inside the inherited
environment are discarded by the host scorer, even when the underlying
physics kernel keeps running. First physical post-step positions and headings
are used; commanded trajectories integrate requested yaw, not actual yaw,
to prevent an actor from steering off course without penalty. Sensor and
command corrections internal to the policy are not substituted for the
operator's desired trajectory.

The standing pass requires **all** these provisional gates for the full
60 seconds: no terminal/fall, maximum birth-relative XY radius ≤0.20 m,
maximum birth-relative heading error ≤0.35 rad, 95th-percentile torso tilt
≤20°, measured foot-contact coverage ≥75%, and bad-contact coverage ≤5%.
Corrective stepping is allowed; wandering away is not. These tolerances are
not yet calibrated against both reference actors.

For each trial we record survival, first-life duration, time to first fall,
failure class, maximum/final XY drift, heading drift, tilt, contact coverage,
bad contacts, achieved vx/vy/yaw, operator-frame velocity tracking errors,
desired-path tracking error, and positive servo shaft work. The stop interval
has its own speed measure so forward/reverse velocities cannot average out
and masquerade as a successful stop.

**No aggregate weighted championship score is assigned.** Summary rows
retain the denominator of the number of evaluated worlds. A policy does not
pass as a generalist merely because a stair or speed specialist is strong.

### Execute on the source-matched H40 workstation

From this research branch, with WSL/JAX/MJX installed:

```bash
export PYTHONPATH=$PWD
export EMU_PHYSICAL_PROFILE=h40
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Cheap contract/unit tests, no GPU or actor required:
python -m unittest tests.test_physical_capability_exam -v

# Small compilation/loader diagnostic; NOT a full qualification:
python -m emu.physical_capability_exam \
  --policy showcase/champion.emupolicy \
  --output ~/emu-data/exams/h40-v1-stand-diagnostic.zip \
  --worlds 2 --device cuda --scenario stand_zero_60s

# Frozen initial flat capability campaign:
python -m emu.physical_capability_exam \
  --policy showcase/champion.emupolicy \
  --output ~/emu-data/exams/h40-physical-v1-flat-s1701.zip \
  --worlds 8 --seed 1701 --device cuda
```

**Do not** reuse an output ZIP path. Every run records source-file SHA-256,
actor weights and artifact identity, robot XML, compiled solver settings,
sensor contract, sampling seed and exam settings. Test only actor/physics
source combinations accepted by the existing strict actor loader.

The ZIP contains `manifest.json`, `summary.json`, `summary.md`,
`worlds.csv` and `capabilities.csv`. There are no invented results or
replays: **replay export is not yet implemented**. Keep console logs and
the exact ZIP along with the source commit.

### Important initial limitations

- The implementation uses the retained full-trajectory MJX trace, at
  2–32 worlds, to make precise first-life analysis feasible on 8 GB VRAM.
  A future streaming on-device accumulator is needed for larger samples
  and 5–15-minute endurance without giant host trace transfers.
- Contact coverage records actual foot/terrain contact, **not** the centre
  of pressure, load distribution or dynamic stability margin. The current
  torso tilt metric is based on the MJX gravity vector.
- No fresh GPU rollout was possible through the GitHub-only connection.
  No 120M/140M calibration results have yet been generated in this branch.
- No confidence interval or cross-training-seed generality is inferred from
  multiple world resets of one trained actor.
- Internal source simulator auto-reset still executes after a terminal but
  **cannot affect the measured first life**. A future kernel should freeze
  dead worlds to reduce computation and enable stronger parity comparisons.

## Next work, in strict order

1. **Calibrate the baseline.** Run the 120M actor with v1. Inspect world rows
   and at least three success/failure trajectories. Align measured stability
   with the existing viewer, or identify a concrete source of disagreement.
2. **Spring V1 source/authentication.** The 140M Spring V1 actor/checkpoint
   is **not** compatible with current `main`'s `h40` model. Recover its
   original model from training source
   `22266821bdbfc82a529ef708e5171c82c6c9f29c`; run with its own
   physical/motor target contract. Port the v1 scoring contract with
   exactly documented differences; compare *capability outcomes* between
   source-authenticated environments, not pretend equal mechanical physics.
3. **Disturbances.** Add exact birth velocity/tilt and timed push impulses,
   with recovery time, maximum excursion, contact safety and first-life
   survival. Keep a no-perturbation standing gate mandatory.
4. **Terrain.** Add source-matched signed +X lip/grade levels, explicit
   first destination top contact for each foot, transfer, continuation
   and full multi-step course completion. Distinguish up/down and
   include trajectory/lateral drift. Verify the physical geometry.
5. **Endurance.** Add 5–15 minute continuous, mixed-command, mixed-terrain
   first-life exams; stream compact JAX stats and selected replay frames
   rather than storing unbounded [time, world, contact] arrays.
6. **Cross-engine comparison.** Replay fixed first-state pose, commands,
   terrain prisms, servo target/actuation histories and sensor packets in
   MuJoCo CPU/browser and MJX GPU. Compare per-frame positions, velocities,
   impulses, ToF returns, command delivery, joint angles and first terminal.
   Browser Meadow aesthetics are kept separate from authoritative parity.
7. **Benchmark acceptance.** Require stable first-life results across several
   seeds, reproducible fresh ZIPs, explicit contact/fall definition review,
   and calibration to both reference actors before using any pass gate to
   accept a replacement policy.

## Archive policy

Historical matrix/prism/teacher exams remain evidence under their **original
source commit, physics model, actor, trial configuration and date**. Do not
combine them into a v1 leaderboard, modify their original score meaning, or
overwrite their ZIPs. The old source code is retained temporarily for
forensics, not reused as a current champion scoreboard.

