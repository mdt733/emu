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
Choose **Random command practice** for previous behaviour, or uncheck
Autonomous demonstration for manual joystick control. **New random target**
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
