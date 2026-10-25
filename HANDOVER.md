# EMU — active engineering handover

**Date:** 26 October 2026  
**Work branch:** `feature/v5023-capability-growth`  
**Protected:** `main`, public showcase, approved 260M Pinnacle. **Do not change or merge without explicit user approval.**  
**Design contract:** [docs/UNIFIED_CAPABILITY_GROWTH.md](docs/UNIFIED_CAPABILITY_GROWTH.md)  
**Physical capability inventory:** [docs/CAPABILITY_DIMENSIONS.md](docs/CAPABILITY_DIMENSIONS.md)  
**Complete WSL commands:** [docs/INSTALL.md](docs/INSTALL.md)

## Mission

Create **one endlessly useful cold-start EMU learner**, rather than several
specialist policies or source-specific 20M→40M→60M experiments. The actor has
360 array-only observations, six actuator outputs and ONE PPO optimizer. It
must use real MuJoCo/MJX contact, realistic delayed sensor observations and
physically checked, direction/severity-attributed learning outcomes.

The owner explicitly prioritises a **systematic, reliable, unstoppable
teacher** over maximum rate of advancement. A target is where to practise,
not a certificate of maturity. Sample easier/near/harder challenges.
Qualifying near-target success nudges it up; failed hard probes normally
hold; persistent easier failures nudge it down. The slow-moving target is
the stability filter; no unrelated cohort may unlock or block another.

Don't overinterpret 20M or 40M locomotion as a mature walker. 20M is barely
standing, 40M is still an infant. Do not design around any particular old
checkpoint. Fresh-start quality matters more than preserving old migrations.

## Training lifecycle (current intended path)

- One clean, source-pinned `emu train` run using
  `EMU_UNIFIED_CONTINUOUS=cold-growth-approved`, `--profile sim2real`,
  `--terrain-actor --terrain-step-max-m 0.04 --unified-capabilities`,
  `--envs 2048 --seed 733 --device cuda`, with a chosen cumulative target
  (the documented example is 120M steps).
- The original foundational gait lessons dominate until 20M world steps.
  Specialist activation blends from 0% to 100% of each allocated cohort
  during the next 10M steps. Non-focus difficulties remain ~half of their
  current targets; grace/control uses gentler cross-training.
  This schedules **practice share only**, not a mastery threshold.
- One run contains all later learning; no 40M milestone or source transition.
  After interruption use `--resume auto` with EXACTLY the same Git SHA,
  config and output path. A completed final checkpoint can extend the target
  (e.g. 120M→240M) with the same source/parameters and `--resume auto`.
- Safety: strict schema/robot/sensor/finite-checkpoint validation, unchanged
  source/config on resume, off-repo checkpoints, emergency PPO health and
  source-matched recovery state. Frozen evaluation cohorts use 32 worlds
  (8GB GPU budget). No grandfathered, cross-source warm-start exemptions.

## Current implementation

**Working code on this feature branch; NOT YET GPU campaign-proven:**

- Six concurrent birth roles: commands, resilience, slopes, steps, transitions,
  control/efficiency, all teaching ONE policy. Most ordinary gait learning
  remains in non-specialist worlds.
- Commands: independent forward/reverse frontiers, symmetrical ±vy/±yaw
  frontiers, plus mixed radius. Small target adjustments use completed
  episode near/easy evidence; *easy failure windows remain independent of
  near-probe windows*.
- Inclined collision: two real contact slabs, four measured directions.
  Near-maximum slope success advances the shared angle; failed easier
  inclines retreat it; hard failures hold. Physical grade is read from
  the actual collision normal, not only the teacher request.
- Discontinuous contact step: actual upper/lower surface displacement,
  supported up/down crossings, severity-relative easy/near cohorts. Shared
  height target rises on near-height up AND down support, holds on hard
  failure and retreats on easier failure. Typical training edge yaw is
  ~±20°, **not yet a trained ±45° dimension**.
- The legacy original `resilience_difficulty` still bundles pose, position,
  joint, linear and angular birth disorder. Separate push-force frontier is
  present but not yet based on a direction-resolved recovery target.
- The transition teacher still couples hold time and ramp speed; the control
  teacher has motor-effort/smoothness shaping and work-per-distance
  diagnostics, but no independent adjustable grace/energy limit.
- Sensor-noise frontier exists; broader physical domain randomization is OFF.
- Removed obsolete source-SHA-specific 20M/40M/60M launch gates, and the
  redundant secondary `_share_unified_growth` limiter. Original core
  gait regime governor, PPO, physical world and actor contract stay intact.

**Do not conflate challenge directions with independent targets.**
See the linked capability inventory: up to about 26 conceivable physical
target dials, far fewer independently controlled today, plus a grace/control
specialist **without** an independent target. Grow the evidence plumbing first,
then split each direction only when it is worth the sampling overhead.

## Next engineering work (rough order)

- [ ] **GitHub Actions must pass at CURRENT source SHA.** Resolve real CI
      errors before more speculative restructuring. Previous passing commit
      `193dc59` precedes the most recent cleanups.
- [ ] Check actual 2048-world CUDA PPO compile/update and memory use on
      owner's RTX 4060 Ti 8GB. CPU MJX smoke proves contact and PPO shape
      only, not GPU capacity or long-run learning.
- [ ] Verify teacher reports **allocated vs activated vs physically challenged
      and completed** episodes in signed command, four slope and up/down
      step cohorts. No falsely claimed progress on missing episodes.
- [ ] Verify full cold 20M foundation plus the 20M→30M specialist blend,
      including stable metrics CSV headers and same-run resume/extension.
- [ ] Improve recovery by distinct physical birth-disturbance outcomes
      (joint, tilt, position, linear/angular momentum, pushes), without
      multiplying PPO algorithms or making a 6D Cartesian monster.
- [ ] Give control-efficiency a genuine target based on mechanical work
      per **usefully tracked and supported** distance, with smoothing/jerk
      and a retreat option; do not reward motionlessness.
- [ ] Only then split symmetrical lateral/yaw, directional slope and step
      targets if evidence shows asymmetric ability.
- [ ] Confirm meaningful cold-run locomotion and target movement from
      owner's GPU logs/exams before any policy promotion.

## What NOT to do

Do not modify `main`, public GitHub Pages or approved Pinnacle; don't
introduce a new actor observation format, extra PPO networks, unrelated
curriculum timers, or separate 40M checkpoint special cases. Preserve
resilience/step/ascent physical evidence, useful cold gait, source-pinned
resumability and a clean comprehensible project. Optimise the code for
understanding, not line-count theatre; remove obsolete systems as they are
replaced. Keep source changes exclusively on this feature branch.

## Operating instructions

The fully copy-pastable **fresh run, same-source resume and completed-target
extension WSL blocks** live in [docs/INSTALL.md](docs/INSTALL.md). Treat
source-pinned `--plan`, 12-world CPU smoke and 2048-world CUDA full-update
smoke as separate operations from starting a real training campaign.
Never tell the owner a long campaign is ready based solely on CPU CI.
Old experimental details and decommissioned handoffs are in the Git history
of this file and the older source revisions.
