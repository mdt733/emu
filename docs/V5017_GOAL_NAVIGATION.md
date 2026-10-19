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
