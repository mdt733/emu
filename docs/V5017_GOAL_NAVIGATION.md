# EMU browser viewer — v5017

A public-facing, browser-only MuJoCo / WASM playground for EMU's trained
six-servo biped. **Only the pose-lattice autonomous navigation demo ships**.
The viewer does not train or modify the network.

## What happens at startup

1. Load the published trained `.emupolicy` from the static showcase, or the
   selected policy from the local WSL viewer.
2. Start autonomous pose navigation immediately. A Web Worker chooses a
   position *and* target heading and searches a time-weighted pose lattice.
3. Follow the continuous route line, not each discrete lattice node:
   cross-track prediction, up-to-30° interception, speed reduction for
   large current/predicted lateral error, measured yaw-rate damping.
4. Set commands using **the same** `setDemoCommands()` / heading controller
   / PPO actor / MuJoCo chain as before. PPO inference remains 50 Hz.
5. Show the commands by moving the two joystick knobs, without fabricating
   human input events. Touch either joystick or use WASD / Q-E / R-F
   to take manual control immediately.
6. When the human releases the controls and leaves them idle for **30
   seconds**, restore autonomous navigation and plan a fresh route from the
   *current* physical pose. Holding a stick or key prevents that timeout.

The trained policy's command limits define the manual and automatic
joystick range. No per-policy movement calibration is required.

## Controls

- **Left joystick:** up/down = forward/reverse, left/right = lateral.
- **Right joystick:** left/right = turning, up/down = chassis pitch.
  The knobs *display* the currently requested autonomous commands.
- **W/A/S/D, Q/E, R/F:** keyboard equivalents. Space = pause; Backspace
  = reset.
- **New destination:** pick another random position and facing.
- **Settings:** simulation terrain, trees, camera follow, resets,
  rendering options, heading assist and live policy updates.
- **Load policies / drag-drop:** switch to another `.emupolicy`. Same
  start-in-autonomy behaviour, no changes to neural weights.
- **Camera:** orbit, chase, raw chassis POV and levelled POV.
- **Telemetry:** observed versus requested motion. Held-out evaluation is
  available in Settings when the bundle contains the relevant metrics.

The removed height/roll/step-height/cadence sliders were not a useful public
interface. The actor **still receives its required observation fields** and
the automatically chosen episode cadence; those were not removed.

## Architecture

```text
goal target / park trees
        |
pose-lattice A* in goal-worker.js
        |
continuous path geometry (path-follower.js)
        |
cross-track / yaw feedback + final parking
        |
setDemoCommands() or manual setJoystickTargets()
        |
existing heading controller + PPO actor + MuJoCo/WASM
        |
actual pose and velocity back to controller and HUD
```

Essential source:
- `index.html`, `src/style.css`: compact public UI
- `src/main.js`: viewer lifecycle and manual/autonomous handoff
- `src/joystick.js`, `src/demo-handoff.js`, `src/controls.js`:
  input and display, including non-interactive joystick mirroring
- `src/goal-navigation.js`, `src/goal-worker.js`,
  `src/goal-planner-client.js`, `src/path-follower.js`: route planning
  and tracking
- `src/sim.js`, `src/policy.js`, `src/sensor.js`: actual robot,
  neural inference, sensor emulation, physics and visuals

The experimental continuous-curvature and bidirectional-arc backends,
the random-command demo, unused waypoint follower and their dedicated
tests are deleted from this branch. Their earlier versions remain in Git
history. Other independent viewer facilities (model loading, telemetry,
terrain, camera and live update support) are retained.

## Launch / tests (Ubuntu WSL)

```bash
curl -fsSL \
  https://raw.githubusercontent.com/mdt733/emu/feature/v5017-goal-navigation-demo/scripts/install_viewer_wsl.sh \
  -o /tmp/emu-viewer-install.sh
bash /tmp/emu-viewer-install.sh --destination "$HOME/emu-viewer-cpu-clean"
```

The installer pulls the branch only if the checkout is clean, prepares the
champion policy, runs `npm test`, and serves the local viewer at
`http://127.0.0.1:8767/`. No training GPU dependencies are needed.
Optionally run `npm run build --prefix emu/viewer_src` for a production
asset build.

### Honest validation boundary

Unit tests cover planner geometry, actor command limits, manual/display
mapping, handoff timeout, path tracking, arrivals and background Worker
protocol. Full browser MuJoCo/WASM testing is needed to check actual gait
wander, stability, touch behaviour and responsive layout. Never infer
physical success from ideal kinematic tests alone.

## Minimal viewer test contract (12 tests, one file)

The browser viewer deliberately keeps just twelve core checks in
`emu/viewer_src/tests/core.test.mjs`. They cover: policy formats and
asset safety; physical terrain; sensor timing; manual yaw and command limits;
one-demo startup and unchanged trained observations; visual-only joystick
mirroring; 30-second manual/autonomy handoff; basic pose-lattice routes;
obstacle keep-outs; continuous-line tracking, drift and manoeuvres;
arrival/hold bookkeeping; and off-thread planner requests.

These are the shared viewer's **baseline**, not invitations to grow dozens
of narrowly overlapping tests. When replacing an implementation, update
or replace one of the twelve rather than appending a new test. The installer
still calls the same `npm test` command, which now runs this single file.

Cleanup also removed a dangling assignment to `this.phaseOverride` from
the MuJoCo viewer: the public cadence override slider was already removed,
and the policy's automatic episode cadence is preserved.


## Route-quality refinement (experimental)

Keep one pose-lattice A* in the background Worker, with its established
forward/reverse/pivot/sideways manoeuvres. After planning:

- Subdivide the exact integrated motion primitives into ~12 cm / ~10-degree
  samples instead of joining sparse A* nodes by long straight chords.
- Apply a bounded, 12-pass elastic smoother **only within uninterrupted
  moving runs**. Never smooth across reverse cusps, pivots or the terminal
  target. Keep modifications within 10 cm of their swept source.
- Collision-check both adjacent chords during smoothing and verify the
  entire final polyline, including the terminal connection. Failed
  refinement falls back to the original route. A* also verifies that its
  final approach to the goal is unobstructed.
- Increase nominal obstacle clearance to **62 cm** from obstacle centres
  (formerly 53 cm) to allow some room for real gait wander. This is NOT a
  physical collision guarantee: the robot may leave the requested route.
- Add a lightweight forward/backward acceleration and braking envelope
  along each movement run, without mandatory policy-specific calibration.
  The trained actor's command limits bound speeds. The follower uses
  that profile directly instead of applying duplicate end-of-run braking.
- Ease the final few decimetres into the existing closed-loop parking
  controller. A separate final parking correction remains necessary for
  residual errors; it is blended rather than switched abruptly.

The route overlay shows the refined centreline. Nominal travel time is
still a heuristic based on requested commands, **not** measured speed or
a time-optimal trajectory. There is no LQR, MPPI, or another search planner.

All changes are covered by the **existing 12 core tests**, with stronger
assertions inside the existing path, obstacle and lattice tests. Browser
MuJoCo tests are still needed to measure line crossings, detour clearance,
falls and final docking in the real learned gait.

The obstacle interface currently accepts stationary circular keep-out
centres based on the park's solid trees. Optional obstacle presets
(Off / Sparse / Course) would need matching *physical MuJoCo geometry*
and collision-aware planner input; merely drawing new scenery is not enough.


## Arc-length trajectory tracking (experimental follow-up)

The route-refinement branch now has a **profile-aware follower** rather than
multiplying planned cruise speed by three unrelated throttles. A* already
returns movement samples containing commanded \`vx\`, \`vy\`, \`yaw\` and
segment \`dt\`; the refinement stage provides a bounded commanded-speed and
yaw profile. The follower:

1. Projects the real robot onto the current route run, preserving monotonically
   increasing arc-length progress even when the gait wanders.
2. Interpolates the planned twist between adjacent samples at that progress;
   checks a short distance ahead for upcoming braking.
3. Uses the profile as feedforward, plus a small bounded correction from the
   **measured body velocity** (at most 0.13 m/s of command trim). This is
   not a fitted dynamic model.
4. Preserves 30-degree line interception, sideways drift prediction and yaw
   damping; reduces speed substantially **only when line/heading errors
   become significant**, so a normal on-path profile isn't multiplied down.
5. Preserves reverse, lateral movement, stationary pivots and the existing
   blended parking correction. The original outer-loop/heading-hold/PPO
   command interface is unchanged.

**Why spatial progress, not absolute playback time?** A six-servo learned
walker doesn't follow an ideal clock. If it takes longer to execute a
motion or wanders off the line, replaying \`x_ref(t)\` can run metres ahead
of the real robot. Arc-length-indexed reference speed maintains the
benefits of planning and lets the controller adapt to actual motion. The
per-segment \`dt\` is still recorded for planner timing and diagnostics.

The status/HUD now reports mean absolute cross-track, heading and speed
tracking errors; reference vs observed speed; plan time vs real elapsed
time; and time spent in each manoeuvre mode. Those measurements can inform
later viewer tuning or a **small optional** in-browser calibration if
measurements show a real problem.

No mandatory per-policy calibration sweep, LQR, MPPI, second planner or
additional tests. There remain **12** core viewer tests. Predicted
kinematic arrival and collision-clearance checks are not a substitute for
testing the actual MuJoCo actor's wandering gait and braking distances.
