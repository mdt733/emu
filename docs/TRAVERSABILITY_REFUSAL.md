# EMU — learned traversability, cautious navigation and controlled refusal

**Design decision, 29 October 2026; not yet implemented or trained.**
Preserve the frozen `feature/v5027-coherent-retention` actor at source
`5dec60c4`, all saved checkpoint bytes and protected `main`.
This is a proposed follow-up for the **one 360-input, six-servo actor**
and the existing forward 4×4 ToF array. No second locomotion policy.

## Requirement

When an operator or waypoint requests motion toward terrain which is
**unlikely to be negotiable by this particular policy at this speed,
approach direction and state**, EMU should detect it early enough, reduce
speed, stop in supported dynamic balance or back away, and report that
the route is blocked. It should **not** deliberately walk into the
edge until it falls, nor blindly override a climbable obstacle.

During PPO training it should **still encounter attempted near-frontier
crossings**, including many failed ones, and learn from physically
supported completion. A traversal capability earned tomorrow must not
remain blocked by a fixed rule learned today. This is *risk-based
navigation* plus *ongoing deliberate experimentation*, not a hard-coded
2/3cm maximum step height.

A dynamically unstable biped may require stepping in place or a
controlled retreat, not rigidly holding six actuators motionless.

## What exists and why it is not enough

- The single schema-9 actor already receives delayed and noisy
  **16 ToF ranges + 16 validity bits**, plus IMU, servo/phase and
  estimated motion; no perfect ground-truth height or collision labels.
- `emu/environment.py::_specialist_task` gives coherent STEP/SLOPE
  cohorts a forward crossing request (0.10–0.22m/s before the original
  auxiliary attenuation). The direct arm preserved more forward
  command on **both** slope and step cohorts; its expanded 5–15°
  exam results were much worse on downhill and inclines than the
  original A. This is **not** an established upgrade.
- `emu/rewards.py::training_reward` rewards locomotion and tracking
  and discounts stationary non-stepping behaviour. The existing
  step-crossing criterion recognises five consecutive supported
  far-side frames. **There is no independent safe-refusal success
  label or early stop/retreat curriculum.**
- `emu/viewer_src/src/goal-navigation.js` already commands zero
  velocity while waiting for a route and can replan for geometric
  trees; it does not currently estimate edge traversability from ToF.
- Expanded physical exam: 3cm crossing occurred 5/16 A,
  5/16 B, 6/16 C (UP/DOWN pooled), but only 0/16, 1/16,
  1/16 completed full 3cm missions; at 15° downhill all arms
  passed 0/16 *movement* missions. These are motivation for
  discriminating *proceed*, *slow/inspect*, and *decline*.

## Proposed architecture (keep motor learning separate from navigation intent)

### 1. Perception: honest, observable estimate

Use current 4×4 ToF and its validity, delay, ego motion, feet geometry
and confidence to estimate an **obstacle approach**: distance to edge,
signed discontinuity (up/down), lateral footprint clearance,
pitch/roll and uncertainty. Do not read perfect MJX terrain height
inside the controller or actor. Estimate must remain conservative
for occlusion, glossy/noisy returns and insufficient pre-edge coverage.
A raw height reading alone is **not** a guarantee that a crossing can
or cannot be completed.

Before touching training, inspect actual episode histories: Does
the existing 4×4 see the rise **early enough to brake at 0.15m/s**?
Measure estimated height/distance versus ground truth *offline only*,
from flat, 1/2/2.5/3/3.5/4cm UP and DOWN, different edge azimuths.
If not, sensor placement, field of view or speed must be revisited.
No controller can infer an obstacle that its sensor never observes.

### 2. Capability: condition on the robot's **demonstrated** abilities

Build a *revisable*, versioned empirical traversability estimate from
held-out physically verified completions across step sign, height,
approach speed, grade, bearing, recovery and mechanical margin.
Include uncertainty and unknown/out-of-distribution cases. Use
repeated trials, not a single optimistic 3cm crossing. Do not store
a universal fixed `MAX_TRAVERSABLE_STEP_CM` or infer ability from
the teaching frontier.

A useful risk estimate is

`P(supported crossing AND maintained control | sensed scene, speed, heading, robot capability)`.

This may first be a small calibrated table linked to the policy SHA,
later trained as an auxiliary estimator **if evidence shows it is
necessary**. There is still **one locomotion brain**; no second gait or
privileged actor input. A predictor must not be treated as magically
certain when a surface is unobserved.

### 3. Supervisor: cautious navigation, no secret servo override

In normal navigation, map risk to **PROCEED**, **SLOW / RECHECK**,
or **REFUSE / ROUTE AROUND** using a confidence threshold, braking
distance and a hysteresis timer. The supervisor issues the ordinary
`vx/vy/yaw` commands to the same actor, never changes servo
targets directly. A refusal should halt **before** entering the
physical edge, with adequate support, or trigger a controlled
reverse/turn if a safe retreat exists. The user should see an
honest status such as `untraversable / uncertain`, not a fake
success event. Navigation can replan for another route.

A positive forward command must remain the *requested intent*;
a deliberate refusal must be reported separately from tracking
failure. Manual controls need an explicit operator policy for
caution/override later; do not silently disable the operator's
commands in the current feature branch.

### 4. Training: optimism with an escape option

Keep the existing command/terrain/control specialists and
**continue near-frontier exploratory step attempts**, including
crossings the policy presently fails. Do not penalise an attempted
climb just because it was ambitious. Continue paying genuine
supported crossings and progress when physically possible.

Introduce a distinct small allocation of **decision episodes**
containing easy, near-frontier and clearly beyond-current-frontier
obstacles. Reward a supported **early stop / safe retreat** for a
challenging obstacle *only when appropriate*, and penalise giving
up on an obstacle repeatedly proven traversable. Never pay for
indefinite zero-command stillness; score controlled spacing,
low-slip support, obstacle avoidance and an unblocked next step.

Separate *exploration* from *deployment caution*: the training
teacher must force a meaningful fraction of real attempts so
the estimated frontier can grow. A failure-to-advance score
must not set up a self-confirming refusal spiral. If the
same observation/command is expected to produce different
behaviour in exploratory and cautious modes, **make that mode
observable to the policy or keep the distinction in the
higher-level supervisor**; do not teach contradictory hidden
objectives under identical actor inputs.

Proposed first version: keep one existing six-action actor
unchanged, teach supported controlled slow/stop/reverse commands
via regular inputs, and implement risk-based choice in the
high-level navigator. Only consider new actor output or additional
observation fields after proving the existing contract insufficient.

## Hard acceptance gates before a big PPO run

1. **Sensor gate:** real 4×4 delayed ToF detects a discontinuity
   with useful confidence before measured braking distance at
   0.15m/s; false terrain detections on flat and slopes are bounded.
2. **Motor gate:** the 360 actor can slow, halt safely in dynamic
   balance, or reverse without falling, with and without a prior
   commanded walk. Empty/occluded sensor evidence does not
   masquerade as traversability.
3. **Navigation gate:** a clearly beyond-capability step triggers
   refusal **before contact**, while easy 1cm and ordinarily
   manageable 2cm steps still get attempted; false refusal
   rate is independently reported.
4. **Exploration gate:** the curriculum still samples and attempts
   obstacles beyond the current certified ability, and the
   empirical traversability frontier expands as real
   supported-and-controlled crossings become repeatable.
5. **Retention gate:** compare familiar flat/lateral/yaw/recovery,
   2cm crossing, 5° and 12° directed grade and shaft J/m;
   guard against blanket immobility and irreversible stop
   policies.
6. **Physical exam:** report separately
   `approach_detected`, `safe_refusal`, `false_refusal`,
   `unsafe_contact`, `attempted_crossing`, `supported_crossing`,
   `post_cross_control`, `recovered_after_abort`, as well as
   true distance to edge at first slowdown. No strict pass
   percentage should bury useful behaviour.

Suggested initial research: inspect the three **already-written**
16-world-per-condition extended per-world JSON reports and sensor
coverage first, then create a small new feature-branch
sensor+stopability experiment. **Do not** yet claim this design is
implemented, or launch a source-incompatible checkpoint
continuation.

## What not to do

- Hardcode a step-height cutoff at 3cm; capability will change.
- Reward standing still on all terrain instead of walking.
- Let a refusal artificially count as a successful crossing.
- Give the actor exact physics ground-truth, policy difficulty
  or terrain task labels it cannot observe on hardware.
- Assume a stable stationary stance exists for a dynamically
  unstable six-servo biped.
- Modify `main`, the public viewer champion, or frozen
  `~/emu-v5027` / `~/emu-v5027-eval` to prototype this.
