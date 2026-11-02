# v5041 — live training-equivalent solid-box viewer

**Research branch:** `feature/v5041-unified-terrain-carpet`. The
released champion and Terrain Park on `main` are unchanged.

## v5041 viewer refresh: two worlds, one inspector

The visible top navigation now has exactly two world modes:

- **Field:** the established natural grass/trees/radial-slope MuJoCo
  world, with an optional numbered, graduated physical challenge course.
- **Training Carpet:** the nine 50cm tilted MuJoCo SOLID boxes with sharp
  vertical faces and safe row/column recycling around the robot.

**Settings** is the sole configuration drawer. The former floating
`FRONT DEPTH · 4×4` laboratory is now a collapsed **Sensor inspector**
within it, off by default; optional advanced legacy surface tests remain
inside this diagnostic area rather than as competing top-level modes.
The floating telemetry HUD shows just policy, age, exercise, body
height and measured/requested velocity plus support grade; click
**More telemetry** for yaw, heading, pitch and FPS.

Training Carpet's specialist drop-down is an *exercise inspector*.
Choose **Auto, Standing, Commands, Resilience, Slopes, Steps,
Transitions or Control**. The selected role changes both the physical
terrain envelope (the actual GPU role's slope/step coefficients) and
the representative exercise's velocity/yaw requests. Select
**Run specialist commands** or **Manual joystick/WASD**. Moving a
joystick/key during an automatic exercise switches to Manual rather
than fighting your input. Automatic exercise resets the browser robot
after falling, so repeated attempts remain visible. Field's normal
autonomous navigation stays exclusively in Field.

The **Match training frontiers** toggle uses the local
`/api/live/terrain` read-only metrics for slope, signed up/down lip
frontiers, command vx forward/reverse, lateral and yaw limits and
resilience push frontier. Role factor and exposure multiplier are
applied using the actual carpet training envelope logic. Before
the Academy (~60M default), ordinary STEP births use
`max(terrain_height_m,academy_height_up/down_m)`; once Academy
births begin, it previews the independently measured Academy
frontiers. This is a representative birth, **not** an exact clone of
the randomly sampled 2048-world GPU cohort. Up/down signs and
difficulty vary by position; step heights may grow to 1.5x the
per-birth reference farther along the world.

Uncheck **Match training frontiers** to expose manual slope, signed
step heights, example speed and push limit. This deliberately makes an
*illustrative* world; it doesn't claim those values were learned.
It can be useful for testing how the current trained checkpoint
behaves against a bigger step than its actual current teacher
assigns.

### Real browser shoves and arrows

In the **Resilience** exercise, the spectator can apply an actual
horizontal torso force using `MjData.xfrc_applied`, held over five
50Hz control frames and the same MuJoCo substeps as normal motion.
A visible amber 3D arrow points in the **actual applied world force
direction**. The strength is limited by the reported global push
frontier and the configured physical push-force ceiling. You can
trigger a single shove with **Test one shove**. Automatic Resilience
preview intentionally repeats visible shoves at an **illustrative
six-second cadence**, rather than pretending to show the trainer's
per-world rare stochastic timing or hidden seed. A manual arrow/force
is viewer-only and cannot modify the training policy, teacher or
saved checkpoints.

The live inspector continues to poll every five seconds, but
only substantial terrain changes rebuild/reset the viewer. Different
specialists are inspected one at a time, sharing the latest validated
policy, while their real GPU counterparts continue learning
simultaneously. The trained policy age (in the HUD) can lag live
curriculum age (in Settings) between checkpoints.

**Source isolation:** Run this viewer from a **separate checkout**
of `feature/v5041-unified-terrain-carpet`; never update or reinstall
the active 140M trainer checkout/venv. The browser viewer is
CPU/WebAssembly for physics and inference; it does not start an
additional 2048-world GPU trainer.

## Which world does the viewer show?

Training Carpet in the EXISTING EMU 3D MuJoCo viewer now instantiates
**nine actual 50cm tilted solid-box collision geoms**, with the same
`0.258 × 0.258 × 0.270 m` half-sizes as the GPU terrain. Every box has
real top and vertical step side contacts. The 3×3 ring is recycled by
moving its MuJoCo mocap bodies as EMU crosses cell boundaries; an
outgoing tile **never moves when a sphere foot could overlap it**.
The visible 50cm tile grid follows the actual ring. Robot policies,
kinematics, viewer headings, ToF, depth displays and manual controls
remain on the existing EmuBrowserSim code path.

The standalone browser DOES NOT replay an identical JAX world. The
trainer samples a distinct random terrain seed, starting pose and
specialist allocation per training environment, and runs all 2,048
environments concurrently. The browser picks ONE deterministic
representative environment from the measured curriculum. It ports the
trainer's signed step profile, smooth gradient potential, local
top-plane and real box dimensions, but it does NOT reproduce the JAX
per-world random birth or sensor domain randomisation. Do not treat a
viewer crossing as an independent held-out GPU physical exam.

The original Terrain Park (trees and eight obstacle fixtures) remains
an **independent selectable mode**. Training Carpet deliberately
doesn't place decorative trunks outside its nine-physical-box ring.

## Automatic development stages

Open **Settings → Terrain → Training Carpet**, with Specialist set to
**Auto** (the default). With a real source-tagged live run:

| Age / curriculum | Representative world |
|---|---|
| **Before 20M foundation cutoff** | Flat physical 50cm tiles, even if STEP/SLOPE is manually selected; training exposure is genuinely zero |
| **20M to ~60M** | Representative SLOPE specialist, with the actual measured frontier (often near zero when still learning) |
| **From Academy introduction (~60M by default)** | Representative STEP world, real vertical UP/DOWN lips and the cross-specialist mild slope |

A single viewer cannot show all six concurrently sampled roles. Select
COMMAND, RESILIENCE, SLOPE, STEP, TRANSITION or CONTROL manually to
inspect their respective projected envelopes. Specialist frontiers can
continue growing independently. These ages represent *birth
allocation*, not mastery gates: historically standing first emerges
around 5M, at 20M the gait can still be immature, and ~40M can be
early walking.

The **read-only live endpoint** `/api/live/terrain` supplies actual
`metrics.csv` frontiers, source SHA, foundation cutoff, Academy
intro/start/ramp ages, and slope-transition *configured geometric
cap*. It never changes the PPO optimiser. Automatic refresh is every
5 seconds; because moving frontiers should not reset a good walk every
time metrics change slightly, the viewer only rebuilds the scene for a
meaningful ~0.5° or ~2mm change, phase change or changed role.
Rebuilding resets the **viewer robot only**, never the trainer.
Uncheck **Follow training stage automatically** to hold a scene and
apply updates manually.

### Running on your WSL computer

After pulling the feature branch and activating its existing
GPU-enabled `.venv`:

```bash
cd ~/emu-v5041-gpu-preflight
git pull --ff-only
git status --short
emu viewer-build
emu viewer --live --root "$HOME/robot_lab/emu_runs/v5041_carpet_s735_5m" --no-open
```

Open `http://127.0.0.1:8765/`. The local server automatically selects
Training Carpet for source-tagged `terrain_carpet` training runs,
when metrics become available. The watcher loads the latest validated
policy snapshot; policy ages may lag the latest curriculum metrics
between checkpoints. If no run exists yet, you can still select
Training Carpet manually and inspect representative default terrain.
For non-live runs import `metrics.csv` or the live snapshot JSON;
the former lacks full configuration provenance, so choose STEP manually
to inspect a late Academy run.

### Test coverage and caveats

```bash
npm test --prefix emu/viewer_src
npm run build --prefix emu/viewer_src
python -m pytest -q tests/test_terrain_carpet.py
```

Node tests verify genuine nine solid-box XML, stage gating, different
slope/step shapes, signed lips, modulo-three recycling, no-foot-on-
abandoned-tile safety, and separation of terrain from robot sensor
self-occlusion. Python tests check the live metadata contract. Actual
**interactive browser WASM contact behaviour** remains to be verified
in the user's viewer on the same source SHA; the Node tests don't
simulate MuJoCo. The 2048-world CUDA PPO test verifies only the real
trainer, not the browser renderer.

For safety, training source and checkpoints remain untouched, and
`main` stays frozen.
