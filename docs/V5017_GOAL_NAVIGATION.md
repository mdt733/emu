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
