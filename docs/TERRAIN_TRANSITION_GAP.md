# EMU terrain transitions: viewer crest bug and curriculum gap

**31 October 2026 — investigation and proposed next experiment.**
**Research is source-pinned:** the active v5031 fork, its B80 parent,
existing manifests and the public `main` policy must NOT change.
This document is a diagnostic/next-study plan, not permission to
modify an ongoing PPO run or to promote a candidate.

## Immediate bug: park's disconnected crest

The browser park's `crest` feature was assembled from three solid
MuJoCo boxes at world Y:
- `crest_rise`: ramp from `2.20` to `2.64` m, with height 0→6 cm.
- `crest_top`: local X half-length `0.18` m rotated 90 degrees at
  `Y=2.83` m, physically supporting only `Y=2.65..3.01` m.
- `crest_fall`: ramp from `3.02` to `3.46` m, height 6→0 cm.

The **two missing centimetres of support** were real 1-cm gaps at the
two transitions, dropping to the underlying park hfield about 6 cm
lower. These are not a slope-to-flat competence test. The browser
only *looked* continuous.

Patch: crest top half-length `0.195` m gives
`Y=2.635..3.025`, 5-mm overlap at each join. The resulting
height difference at the edges is less than 1 mm at the ramp's
approximate grade (6 cm / 44 cm = 7.8 degrees). The actual simulated
walking outcome still needs browser validation. A static JS geometry
regression checks the support footprint and rejects renewed gaps.
The prior [optical-flow park hfield fix](../emu/viewer_src/src/sim.js)
is preserved.

Do not evaluate whether the neural walker can crest slopes using a
viewer surface known to have actual support gaps.

## Audit: the training domain is broad in *orientation* but sterile in *curvature*

`emu/environment.py` `add_unified_terrain` creates exactly two enormous,
rigid, constant-normal support slabs. At reset, the environment changes
their common physical normal, sharp lip elevation and edge yaw.
The actual slope direction is random around 360 degrees. The step
edge yaw has a narrower random range, and the held-out examiner can
specify precise angles. These are **real colliding** floors and
steps, not a gravity fake; the actor sees delayed 4×4 ToF/flow/IMU,
not privileged teacher geometry.

What the current physics does **not** teach within a single
continuous traversal is a surface normal that changes with XY:
`flat → incline → flat`, `flat → incline → crest → descent`,
or a curved/cambered hillside that transitions naturally.
There is no continuous terrain curvature on those two slabs.
The 80M B-arm's 3-cm upstep stalls also remain an independent
known capability gap; adding organic hills must not excuse poor
step climbing.

## Proposal: controlled read-only *transition* suite first

Keep existing flat, planar slope and precise step exams. They are
valuable *controls*: adding hard terrain must not hide regressions in
the clean fundamentals.

Using a frozen B80, C100/B100, and later Pinnacle candidate, test
*matched* geometry and commands, no PPO update or new reward.
Use real support collisions, the 360-input physical sensor model,
complete traversals and seeded world variation. Suggested cases:

1. **Crest join:** flat→uphill (2°, 5°, 8°, 12°)→level. Explicitly
   examine the ramp/plateau junction with a continuous support mesh
   or carefully checked seam geometry.
2. **Whole hill:** flat→uphill→crest→downhill→flat, with heights and
   lengths that require genuine balance adaptation over several steps.
3. **Valley:** reverse the curvature: downhill→bottom→uphill. No
   clever turnaround or episode reset can count as success.
4. **Cross-slope transitions:** modest roll changing sign or a
   diagonal slope at a different azimuth from the approach.
5. **Slope + vertical discontinuity:** a *true* 1–3 cm step on an
   inclined landing, with supported far-side continuation.
6. **Gentle irregularity:** smooth, correlated XY height variation
   over foot-scale-and-larger spatial wavelengths; avoid independent
   white-noise vertices and accidental millimetre cliffs.

For the first comparison, use fixed independent commands, e.g.
`vx=0.15` and `0.30 m/s`, no autonomous navigator, and matched seeds.
Later include higher supported speed, sideways, heading hold, and
combined commands as separate cases. Preserve baseline clean-flat
and inclined-plane exams in every comparison.

Report **supported crossing + survival** along with
distance past the crest, forward tracking, sideways drift, heading
bias, body attitude/pitch rate, foot contact/slip, actuator J/m and
percentage of valid ToF/flow. Do not award a pass for stationary
balance, avoiding the obstacle, or falling just after reaching
the crest. Report exactly what is measured; distinguish MJX results
from interactive browser-only observations.

## Geometry and performance cautions

New real hill terrain is *not* just changing the rendered mesh.
Collision, contact normals, ToF rays, optic-flow depth, support-height
telemetry and spawn configuration must all agree. The present MJX
two-slab layout is cheap, fixed-topology and proven at 2,048 worlds
on an 8-GB GPU. Do not assume arbitrary per-environment dense hfields
will remain fast or correctly batched.

First prototype several **static or fixed-topology** hills/transition
templates and smoke them on CPU (8–32 worlds), then 256 and 2,048
world GPU preflight if feasible. Investigate edge seams, contact
penetration, 4×4 sensor consistency, reset clearance and compiler
memory before turning on training. True smooth heightfields might be
appropriate if MJX supports the required per-world variation efficiently;
otherwise prefer a bounded set of validated piecewise planar segments,
with carefully controlled short transitions rather than unverified
triangle/convex-hull terrain.

Only after read-only tests establish a real transition failure should
we run a paired curriculum study:
- **Control:** current same-source training recipe/continuation.
- **Treatment:** identical run, same reward and command coverage,
  replacing a bounded portion of terrain specialist exposure with
  progressive organic transition worlds. Keep flat, straight
  speed, slopes, sharp 2–3 cm steps and recovery rehearsal in rotation.
- Evaluate *both* on identical old and new suites. Extra complexity
  must earn its keep by improving real supported locomotion without
  reducing speed, tracking, energy efficiency or resilience.

A larger multi-surface **cold-start** campaign should happen only once
this experiment is physically validated. The public main remains frozen
until replicated physical + browser success and explicit release approval.
