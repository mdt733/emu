# Browser-only autonomous pose navigation — v5017 prototype

**Branch:** `feature/v5017-goal-navigation-demo`. The approved v5016
Pinnacle brain on `main` is unchanged; the planner never trains it.

## What it does

The demo can now choose **random targets in a circle of radius 4 m**:
each target is a world (x, y) position **and a target final heading**.
The target appears as a gold position ring and facing arrow. An outlined
4 m arena and a cyan planned path show where the robot intends to go.

The browser's **time-weighted SE(2) lattice planner** evaluates
forward, reverse, sideways, diagonal, pivot and simultaneous steering
primitives. It uses command-speed *estimates* bounded below the checkpoint's
published (vx, vy, yaw-rate) frontier, and checks predicted motion against
the solid tree-trunk locations with clearance. A 16-heading grid and 22 cm
position lattice keep the search usable on desktop and mobile.

Each leg of the plan is tracked by a **feedback controller** using robot
position and world heading; near the goal it brakes, corrects remaining
position and turns to the final facing direction. A mission counts as
parked only if the actual body is within ~19 cm position and ~16° heading,
with low speed for 0.75 seconds, then samples another random target.
The router retries from real robot position if it drifts far off its path.
The planner's outer-loop yaw is applied **directly** as a rate command;
the manual world-heading P loop is bypassed only while navigation is
active so the two controllers never fight.

## How to use

Open local EMU viewer with any trained schema-2 policy. In **Settings →
Demo controls**, keep Autonomous demonstration checked and select
**Navigate random poses** (the default in public showcase).
**Random command practice** is currently the safe default.
Choose **Navigate random poses (experimental)** to opt into navigation.
The planner first commands a gentle 1.2-second walking warm-up,
avoids untrained stationary pivots and uses continuous stepping.
Arrivals are dynamic pass-throughs at the requested heading, not
a verified static parking hold. Uncheck Autonomous demonstration
to use manual joystick control. **New random target**
skips to another mission. Orbit camera frames the area when navigation
starts; Chase, POV and level POV are still available. Touch stick or
WASD/Q/E input immediately cancels the autonomous demo.

All three modes run entirely in browser-local MuJoCo/WASM. The viewer shows
planned estimated time, manoeuvre name, distance/angle to the goal and
number of successfully parked poses. Trees outside the very centre
become obstacles, not just visual decoration.

**Limitations:** The lattice is weighted A* with a conservative command
velocity envelope, not a mathematically time-optimal trajectory under true
actuator dynamics, thermal limits or heading-control latency. Obstacle
clearance uses static trunk footprints only. The learned sensor actor
does not get a new goal observation; only the browser-level command
controller knows the pose target. This is a compelling navigation demo,
*not* a validated autonomy/performance score. Measured speed envelopes
from repeated physical/simulation waypoint trials would be the next
research upgrade, followed by predictive turn-and-brake planning.

## Release safety

This feature remains branch-only for review. Neither the public showcase
model, release approval hash nor the legacy v5016 checkpoint was modified.
Run `npm test --prefix emu/viewer_src`, browser build/preview, and test
both desktop/mobile before a separate PR promotion.

Unreachable or stalled missions are explicitly timed out and counted as
failures, not successes. Stuck waypoints trigger a fresh route search,
while target-search failures back off rather than monopolising frames.

To preview in a separate WSL directory with your existing viewer virtual
environment (no training checkpoint needed):

```bash
git clone --single-branch --branch feature/v5017-goal-navigation-demo \
  https://github.com/mdt733/emu.git ~/emu-goal-demo
cd ~/emu-goal-demo
source ~/emu-viewer-5016/.venv/bin/activate
export JAX_PLATFORMS=cpu
PYTHONPATH="$PWD" emu viewer-build
PYTHONPATH="$PWD" emu viewer showcase/champion.emupolicy --port 8767 --no-open
```

Open http://127.0.0.1:8767 and choose **Settings → Demo controls**.
The latest committed 260M inference policy is already included.

## v5017 frozen-actor regression: root cause and guard

First preview showed the 260M model's last pose frozen when navigation
started, whereas random command mode advanced inference normally. The
first goal planner synchronously explored up to 5 × 6500 lattice nodes
inside `requestAnimationFrame`, and the next animation frame was queued
only after navigation returned. A slow search or JavaScript exception
could therefore stop all `stepControl` and freeze leg animation.

**Repair:** expensive goal search and replanning now execute in a Vite
module Worker. The main render loop submits requests and continues
calling MuJoCo and the policy at normal control rate, with gentle walking
while a result is pending. A request-generation guard discards stale
answers after reset or model hot-swap. Timeouts and Worker errors are
reported instead of preventing physics steps. The next animation frame
is queued at the beginning of each frame, and navigation exceptions
restore the known-working random-command mode. The navigation HUD shows
an increasing `actor control frames` counter so a stalled policy is
immediately distinguishable from an unreachable waypoint.

These are browser-runtime safety checks, not proof of goal-navigation
competence in the full MuJoCo biped. The feature remains experimental on
a draft PR; merge only after actual Chrome/mobile behaviour is checked.


## 2026-10-19 additional correction: same proven command pipeline

The first attempt to repair the frozen robot moved A* to a Worker but left
an **unnecessary alternate yaw/control path** in `sim.stepControl()`.
The user confirmed Random command practice continued to walk, while
Navigate random poses remained frozen. Navigation now feeds the *exact*
same `sim.setDemoCommands({vx,vy,yaw,pitch})` callback as the working
random demonstration. There is **no navigation-specific yaw, policy
inference, actuator, or physics path**: only the upstream selection of
desired command velocity differs. The navigation-mode setter affects
visual presentation only. The global frame error handler no longer
silently sets `sim.paused=true`, so optional UI exceptions can't
disguise the issue as a dead neural model.

The heading P setting has the same meaning in both demo modes; it
translates the commanded yaw rate into a heading setpoint. Off-thread
A* remains optional trajectory selection rather than an alternative
simulation system. Confirm the live actor-control frame counter advances
when testing navigation in an actual browser.

Command updates are now delivered with a 650 ms zero-order hold, mirroring
the proven random command demonstration's practice of holding each velocity
request between neural control frames. Route evaluation remains independent
of this command cadence. HUD diagnostics report both continuously
incrementing actor inference frames and the count of delivered command
updates; either staying at zero distinguishes a control-loop fault from a
navigation-planning fault.

## Cruise controller revision (after real-browser actor validation)

The first goal driver enforced heading and position at every 22 cm A* lattice
node, capped forward speed to **0.30 m/s**, and constantly strafed to align.
The actual v5016 Pinnacle browser policy declares `vx_max=1.042937 m/s`
and `vx_min=-0.529896 m/s`; it is not inherently limited to 0.30.

The revised planner uses an estimated **0.84 × learned forward speed**
(up to 0.95 m/s) rather than a hardcoded 0.48 m/s. Its route follower is
a **velocity-limited pure-pursuit controller** with a 1.1–1.7 m
speed-dependent preview, no intermediate heading requirements, fast
forward cruising near 90% of the checkpoint's commanded `vx_max`
when aligned, and simultaneous sweeping turns while walking. Sideways
error correction is only allowed within 0.64 m of the FINAL goal, and
the final heading becomes increasingly important during approach.
The planner still checks trunk clearance before taking a preview shortcut.
Route progress is monotonic by position instead of stopping for each
quantized heading. The original actor/normalization/inference pipeline,
motion physics and command delivery cadence are unchanged.

The speed limits are **command limits**, not measured real ground speeds.
We need to measure actual progress, arrival rates, falls and average
time-to-goal before calling these near-optimal or deploying to main.

Command updates now run at 4 Hz in cruise and 5 Hz in approach through
the same `setDemoCommands()` path (the actor remains at its normal control
frequency). The navigation HUD shows both the commanded forward velocity
and observed body-frame forward velocity so it is immediately apparent
whether a high-speed request actually produces ground speed.

## Prototype: joystick-motion bidirectional lattice (2026-10-19)

Initial proposal: nine left-stick directions crossed with three yaw
choices. The 2026-10-19 revision replaces that with a more useful
**forward/reverse/pivot × 9 yaw rates** arrangement. This is still a clean
27-option finite-duration control set, but every forward or backward
choice corresponds to a different physical turning radius. The robot CAN
stand and pivot; the initial claim that it couldn't was mistaken.
Eight in-place turning options are eligible, and only the ninth
pivot-with-zero-yaw option is skipped as a do-nothing command.
No primary search primitive strafes. Small sideways inputs remain available
to the *existing arrival controller* for final position correction.

This is in the state-lattice / kinodynamic-A* family used by Nav2's
Smac State Lattice planner, extended with bidirectional search. The
predecessor tree searches backwards from the required full pose
`(x,y,heading)` to meet the forward tree from the robot. A curvature
bridge connects near meetings without teleporting between bins.

The state-space search is CONTINUOUS: a 26 cm spatial / 24-bin heading
quantization is used **only for visited-state hashing**, not to force grid
edges or stopping at tile centres. Reverse-time expansion uses negated
body twist to discover predecessors; when the resulting route is
reconstructed, the actual recorded original forward action is used.
Both trees must match pose position AND heading, and an independently
validated short curvature segment joins their near-matching states.

Cost is estimated seconds plus light steering/reversal/sideways penalties.
The real trained command limits determine translational and turn speeds.
An alternative continuous Dubins forward/reverse candidate planner is in
`curvature-planner.js`. The lattice is computationally bounded and has
no authority over PPO, viewer actuator commands, or production showcase.
Do not activate either new planner in the viewer until the model-backed
browser test and route quality comparisons pass.

**Important limitation:** full MuJoCo gait physics is NOT time-reversible.
The backward search is an inverse of the *kinematic command model* only.
Learned locomotion acceleration, true slope behaviour and servo thermal
constraints require later empirical speed/cost calibration. Bidirectional
search may fail to meet within a limited budget even when a feasible
solution exists; a direct curvature path remains a useful baseline.

### Variable-curvature motion fan

The primitive chooser in `bidirectional-lattice.js` now uses nine yaw
fractions `[-1, -.75, -.5, -.25, 0, +.25, +.5, +.75, +1]`
applied to the policy's bounded yaw command, independently for forward,
reverse, and in-place pivot. The walking cruise command is intentionally
near the model's trained forward and reverse envelopes. Curvature is
`yaw/vx`; a straight step has zero yaw; an in-place turn has zero
translation. These are **controls**, not grid-neighbour coordinates.

This matches the general design of Nav2's configurable minimum control
set for differential-drive and state-lattice planning: include rotation
in place and reversing when the robot can actually perform them.

Choose routes by estimated time (including reversal and pivot transitions),
not by shortest Euclidean metres. The planner has NO strafing edges.
The follower may independently issue gentle `vy` trim close to the
goal; it doesn't need to pay for a sideways traverse across the arena.

This remains an **offline experimental solver**. The browser Worker
continues to use the earlier known-working route planner until the new
candidate can be compared using actual full 260M-policy MuJoCo rollouts.
This protects the existing demonstration and makes regressions measurable.
