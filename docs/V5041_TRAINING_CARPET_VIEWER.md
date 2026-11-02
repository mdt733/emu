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
within it, off by default; its ground is simply the selected Field
or Carpet. The retired surface-lab menu, step presets and old Ground
selector have been removed from the UI.
The floating telemetry HUD shows just policy, age, exercise, body
height and measured/requested velocity plus support grade; click
**More telemetry** for yaw, heading, pitch and FPS.

Open **Settings** while in Training Carpet. The FIRST control is the
specialist selector, followed by the actual projected challenge sliders.
Choose **Auto, Standing, Commands, Resilience, Slopes, Steps,
Transitions or Control**. The selected role changes both the physical
terrain envelope (the actual GPU role's slope/step coefficients) and
the representative exercise's velocity/yaw requests. Select
**Run specialist commands** or **Manual joystick/WASD**. Moving a
joystick/key during an automatic exercise switches to Manual rather
than fighting your input. Automatic exercise resets the browser robot
after falling, so repeated attempts remain visible. Field's normal
autonomous navigation stays exclusively in Field.

**Follow live specialist values** uses the read-only
`/api/live/terrain` data for slope, transitions, up/down lips, forward
and reverse velocity, strafe, yaw, push limit and resilience intensity.
The sliders show **specialist-scaled effective values**, not the raw
global maximum, so switching from SLOPE to STEP immediately shows
their different terrain weightings. The origin note explains the
role's percentage of the global slope/step frontiers.

Move any slider to enter a **viewer-only customised scenario**.
The physical MuJoCo carpet and representative command exercise use
your selected values. You can retain separate experimental settings
for different specialties and return with **Reset sliders to this
specialist's training values**. If you adjust STEP during the original
flat foundation, the deliberately unearned challenge can still be
inspected; the actual trainer's foundation remains unchanged.

The ordinary STEP birth uses `max(terrain_height_m,
academy_height_up/down_m)` before Academy allocation begins around
60M, then the Academy-specific signed fronts. The browser runs ONE
deterministic representative birth. It's not an exact copy of any
random GPU environment, and achieved skill requires independent exams.

### Real browser shoves and arrows

The viewer applies genuine horizontal torso forces through
`MjData.xfrc_applied`, with random directions, magnitudes up to the
shown push limit, and random 2–8 control-frame durations. These are
real 50Hz-browser physics impulses (held over normal MuJoCo substeps)
and appear as bright amber directional arrows while active. A muted
arrow briefly marks a *completed* shove, with no extra force applied.
The probability of a shove is checked every control frame and
scales with the **Disturbance intensity** slider.

The default **Inspection 8×** option multiplies the stochastic
training-style event probability to make rare recovery behaviour
easier to watch. Switch to **Training-like event frequency** for the
unaccelerated 0.0015-per-frame baseline × resilience intensity.
The browser doesn't know the exact sampled per-world GPU disturbance
state, so this is only a *representative* stochastic exercise.
**Apply one random shove** forces an immediate random test even
between automatic events. No viewer action reaches the live learner.

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

Choose **Training Carpet** at the top of the viewer, open
**Settings** and keep **Training specialist** on **Auto** (the default). With a real source-tagged live run:

| Age / curriculum | Representative world |
|---|---|
| **Before 20M foundation cutoff** | Flat physical 50cm tiles in live-follow mode even when STEP/SLOPE is selected; manually overridden challenges are viewer-only |
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
Uncheck **Update terrain when live limits change** under
Training evidence & live connection to hold the current physical scene.

### Start the viewer while the 140M GPU trainer is running

Use a **SEPARATE viewer-only checkout and CPU Python environment**.
Do not update, install into or `git pull` the training checkout
that owns source-pinned checkpoints. You need Python 3.13+ and Node
(see `docs/INSTALL.md` if either is missing).

```bash
VIEW="$HOME/emu-v5041-viewer"
if [ ! -d "$VIEW/.git" ]; then
  git clone --single-branch --branch feature/v5041-unified-terrain-carpet \
    https://github.com/mdt733/emu.git "$VIEW"
else
  git -C "$VIEW" status --short
  git -C "$VIEW" pull --ff-only
fi
cd "$VIEW"

# New viewer-only venv in THIS separate checkout (no JAX CUDA setup,
# no 2048-world GPU update, never changes the trainer's venv).
bash scripts/setup_wsl.sh --viewer-only --install-node
source .venv/bin/activate

emu viewer --live \
  --root "$HOME/robot_lab/emu_runs/v5041_carpet_s735_140m_fork10m" \
  --no-open --port 8765
```

Open `http://127.0.0.1:8765/` in the Windows browser. Select
**Field** or **Training Carpet** using the prominent top buttons;
the **Training specialist** selector is the first control in
Settings, above the editable sliders. Neither it nor the retired
navigation HUD obscures the 3D view. The local server
automatically selects Carpet for source-tagged `terrain_carpet` runs,
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
