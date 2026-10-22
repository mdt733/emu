# EMU v5022 — isolated forward-sensing spike

Source: `main` commit `2ec9e9db0a816c658131311129ef89ec7a595a12`.
Branch: `feature/v5022-forward-terrain`. No change to the public showcase.

## Baseline: what must not regress

- `pitch15_220`: 55 base inputs × 4 frames; legacy checkpoint schema 7.
- `sim2real_232`: 58 base inputs × 4 frames; three downward ToF,
  IMU/optical flow, asynchronous acquisition and schema-8 fingerprint.
- Proven cold-start foundation is in `training_config.foundation_config`
  (`allow_rollback=False`, no athletic rehearsal, small pushes, gentle
  command distribution); mature training is a distinct opt-in profile.
- `environment.py` constructs physics and observations, `training.py`
  delivers batches to `ppo.py`, `contracts.py` fixes observation order,
  `checkpoint_io.py` checks resume, `viewer_server.py` exports policies
  and `viewer_src/src/sensor.js` simulates approximate browser sensors.
- `main`'s baseline champion is an independently retained acceptance
  reference, not a 232-channel parent checkpoint.

## Phase 1 prototype: three additional forward/downward ToF beams

The original three-ray system is retained as a **comparison prototype**,
not an approved hardware shopping list. Keep the downward sensor tripod and
its estimators unchanged. The comparison arrangement places three
additional narrow-range modules around body X=+90 mm, Z=-18 mm, at
Y=-60/0/+60 mm. Approximate forward/downward beam vectors are
(+.70,-.12,-.70), (+.71,0,-.71), (+.70,+.12,-.70), normalized.
The center of the actual torso box is X=+26 mm, with half-width 50 mm:
its front surface is X=+76 mm. Sensor optics must sit beyond that surface,
not inside the robot collision volume. Geometry is still provisional and
requires clearance, cable and leg-swing/CAD checks.

At neutral torso Z≈258 mm on level ground, the forward fan strikes the
ground approximately 0.33 m ahead of the root. This is **not** a guaranteed
preview distance: tilt, platform height, riser occlusion, or missing surfaces
change the first return. A narrow beam cannot infer the entire terrain
surface. A ray can strike a vertical riser rather than its top, which
is informative but does not directly measure step height.

Early three-ray candidate actor channels (not yet implemented):
`preview_right_range_m`, `preview_centre_range_m`,
`preview_left_range_m` and their three independent validity channels.
That **candidate only** would make an explicitly NEW 64×4 = 256-input contract, not a
reinterpretation of `sim2real_232`. Dimensions will be frozen only after
sensor validity, actuation geometry and real MJX feasibility are verified.
Existing 220/232 weights and checkpoint identity remain unchanged.

The isolated `emu/forward_sensing.py` prototype intersects rays with
**solid boxes**, including vertical faces. It does not grant future world
state to the actor and represents no return by 0 plus false validity. A
caller must add stochastic sensor noise, timing, delay and dropout.
The prototype supports simultaneous worlds with distinct platform heights.
It is not used by production training at this milestone.

## Acceptance ladder

1. **Geometry reference (this milestone):** unit-test level-ground
   readings, physical up/down difference, riser interception, invisibility
   before the preview horizon, missing ranges, and batch/JAX shape parity.
2. **MJX hard gate (not passed):** build actual solid collidable platforms,
   verify box/foot contacts, non-fall support and distinct physical heights
   in at least two simultaneous worlds. Compare ray results against actual
   MuJoCo/MJX geometry and measure JIT compile speed and memory. Do NOT
   conflate a visual heightfield with training contact physics.
3. **Actor gate (not started):** add delayed, noisy, dropout-aware preview
   readings in a dedicated sensor queue; explicitly version the observation
   schema; update PPO/export/checkpoints/browser as a single contract change.
4. **Learning gate (not started):** fresh gentle foundation, then measurable
   step crossings. Blind matched control and frozen-policy ablation must
   demonstrate anticipation ahead of contact.

## Local software checks

```bash
python -m pytest -q tests/test_v5022_forward_sensing.py
python -m pytest -q tests
emu doctor --no-require-gpu --no-mjx-smoke
```

No experiment should be called GPU-ready until MJX physical support and the
new observation/checkpoint/export/viewer contract have passed their gates.


## Hardware selection update: prefer ONE 4×4 multizone front module

A single forward multizone ToF module (reference candidate ST VL53L5CX)
is more practical than adding three independent overlapping single-zone
modules to the existing three downward sensors. It can capture a horizontal
and vertical patch ahead with **one sensor optical centre**, fewer mounting
points and a more natural interface for crests, troughs and ledges.

Manufacturer references:
- https://www.st.com/en/imaging-and-photonics-solutions/vl53l5cx
- https://www.st.com/resource/en/datasheet/vl53l5cx.pdf
- https://www.st.com/content/dam/dc21/conference-sessions/tt_6_kvam.pdf

Nominal module FoV is 45° horizontal × 45° vertical (65° diagonal);
4×4 can operate up to 60 Hz and 8×8 up to 15 Hz. We propose **4×4 at
<=50 Hz control** initially; actual integration time and observed timestamps
must be measured. Data from an external driver must retain zone status,
measurement age and signal confidence. Optical flow remains a separate
downward source. Do not assume all sensor modules have compatible power or
I²C interfaces without a chosen breakout/board design.

This branch now includes a 16-zone, five-subray-per-zone geometry prototype
in \`forward_sensing.py\`. All zones share the same mounting centre:
body (X=+90 mm, Y=0, Z=-18 mm), down-tilt around 50°. Each zone's median
range and geometric visibility confidence are only **simulation
approximations**, NOT hardware-equivalent histogram returns. Test actual
field edges, multi-target readings, ambient-light sensitivity, reflectance,
cover-window crosstalk, vibrations, and body/self-occlusions.

Avoid treating 16 fixed zone centres as perfect pencil beams: real zone
Footprints are broad and a step can yield a mixed or missing reading.
No sensor can provide exact surface type from a single depth image.
A missing value must not be interpreted as flat ground.

The tentative 256-input three-ray schema is NOT frozen and must not be
used for policy training. Later choose a compact reduction of physically
measured zones and validities (or use all zones if justified); version the
final *meaning* and sample timing, not merely the channel count.

## Unified terrain generator with distinct seams

\`terrain_profiles.py\` now describes consecutive physically supporting
segments by top elevation and slope. Between adjacent segments, it records
a signed **height jump** and **slope-angle change** independently:

- zero height jump + angle change = continuous crest, valley, slope onset
  or slope exit (the grade may go uphill to level or downhill).
- positive jump = true step-up even if two adjacent slopes match.
- negative jump = true step-down with measurable lower support.
- both angle and jump changes = combined terrain (advanced, not the newborn
  curriculum).

Each segment can be represented by a genuinely collidable **rotated solid
box**. This is NOT a decorative ramp or a terrain-truth actor channel.
The examiner owns profile parameters. The sensor gets only returns from
real colliding geometry. Rectangular slab boundaries can create contact
seams or underside protrusions; MJX must explicitly test such artifacts
before promotion. We must validate constant-height and tilted support in
separate worlds and on both sides of crests and troughs.

Curriculum difficulty should progress in **two or three measured axes**
rather than an all-purpose integer: absolute slope, change of slope over
distance, and signed step height. Keep flat/command rehearsal throughout.
Do not increase obstacles until an independent examiner confirms that
actual surfaces were traversed with support and post-feature walking.

## CI finding on protected historical tests (22 October 2026)

Our first full CI run reported 84 passing and seven failing tests. Two were
new \`mjx.ray\` checks: MuJoCo-JAX's ray implementation indexed an *empty*
material table in the minimal XML scene. The spike now adds an opaque
terrain material so the ray test can execute without that scene defect.

Five other failures are longstanding **legacy schema-1 exam assumptions**:
some old tests pass the current approved 232-input/schema-2 Pinnacle
showcase into a deliberate 220-input/schema-1-only examiner, producing
\`not an EMU schema-1 exported actor\`. The checker must remain strict;
we should explicitly test schema-2 in a compatible examiner or use a
truly legacy policy for old schema-1-only tests. Never rewrite the approved
Pinnacle just to satisfy these stale tests.

The new physics / observation path is not ready for a GPU training run.
