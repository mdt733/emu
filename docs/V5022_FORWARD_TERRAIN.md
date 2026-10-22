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

## Phase 1 choice: three additional forward/downward ToF beams

Keep the downward sensor tripod and its estimators unchanged. Place three
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

Proposed new actor channels (not yet implemented):
`preview_right_range_m`, `preview_centre_range_m`,
`preview_left_range_m` and their three independent validity channels.
This would make an explicitly NEW 64×4 = 256-input contract, not a
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
