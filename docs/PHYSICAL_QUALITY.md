# EMU physical quality: what is a good robot?

**Examiner contract v1 · 31 October 2026.** Independent, read-only physical
evidence; NOT a PPO reward function or training schedule. The released
v5032 260M actor remains untouched.

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
usage, see [the current runbook](V5045_LEAN_PRISM_RUNBOOK.md).
The shipped 260M public champion must retain its original,
source-specific physical baseline; do not fabricate matched prism results.

See [issue #17](https://github.com/mdt733/emu/issues/17) for proposed
diagnostics and the [released 260M profile](V5032_PINNACLE_RELEASE.md).
