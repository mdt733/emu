# v5046 — only the two feet solve real terrain contacts

**Status:** experiment on `experiment/v5046-75cm-four-tile-midpoint`.
No change to `main`. The old torso-contact MJX world remains selectable
using `--torso-contacts` in the GPU smoke benchmark. No CUDA speedup is
claimed until that A/B is measured on the RTX 4060 Ti.

## Why

The operator's XProf trace of the existing 8-prism trainer captured
4.75 seconds for a full 2048×32 update. Four real MJX substeps account
for ~3.66 seconds (77% of rollout scan time), and a separate redundant
all-world MJX forward after each control frame costs ~0.81 seconds (17%).
The available 8-prism MJX data had **51 padded contact slots and
218 constraint rows per world**, despite only two actual 18mm ball feet.

The original robot XML already disables all thin thigh and shin
rod collisions. The derived carpet XML also disables the underlying
flat infinite plane. However the torso box still participates in
prism contact solving and can inflate every world's statically sized
constraint budget, even when the torso is far above the ground.

## Implemented physics reductions

1. **Actual foot-sphere vs convex prism remains MuJoCo/MJX physics.**
   Four 75cm moving squares, eight precisely deformed convex triangular
   prism halves, true hard friction and impact contacts, six servos,
   passive shin springs, all joint inertias unchanged.
2. **Remove the torso box from the MJX contact graph** (only in the
   v5046 candidate, controllable via constructor `feet_only_contacts`
   and smoke option `--torso-contacts`). Its 300g
   mass, exact size, inertia and physical position remain in the model.
   Thin thigh and shin rods were already noncontacting.
3. **Use virtual ground strikes, not virtual ground forces.**
   One vectorized exact-surface query examines the oriented lowest
   torso-box corner, both knee joint positions, thigh midpoints,
   middle shins, and lower-shin midpoints. It accounts for the
   legs' 8mm capsule radii, comparing against the live triangles
   rather than a level plane or a requested lesson.
   Penetration deeper than 5mm gets a strong nonfoot-contact penalty;
   penetration deeper than 20mm terminates the episode as a
   physical fall, to discourage walking knees through risers.
   These thresholds are hypotheses to validate, not certified
   hardware clearance limits. It does not fake reaction forces.
4. **Consolidate repeated terrain sampling:** the previous code
   separately queried 1 torso, 2 foot and 5 body origins. The
   feet-only path now queries all those plus the virtual-sentinel
   positions in a single vectorized four-tile calculation.
5. **Compact the expensive MJX forward on episode resets:** default
   `reset_forward_capacity=128` packs the exact terminating worlds
   into a fixed-size batch. A frame with no deaths skips MJX forward.
   Over-capacity frames automatically run the exact reference
   2048-wide forward, preserving every reset. Every world retains
   its domain-randomized MJX model and exact fresh task/sensor/terrain.
   `--reset-forward-capacity 0` restores the original dense path.

### What this does *not* change

- The 360-input actor's meaning or schema, PPO reward weights,
  actual six-servo dynamics or episode curriculum.
- Exact foot/prism contact materials, independent step-up/down top
  surfaces, or the true support-foot evidence contract.
- The protected `main` champion or its source.
- Viewer physics: a viewer using the standalone XML has not yet
  been aligned to the new optional contact contract; do not
  assume a browser-only rollout validates these changes.

### Physical acceptance

The GitHub CPU checks include: identical body mass/inertia and geom
dimensions versus the reference collision topology; only feet able
to physically contact the prism; fewer padded MJX contact slots and
constraint rows; exact contact mesh/ToF top agreement; 4-world real
MJX trajectories; foot-preserving recycling; compact reset parity
(no deaths / sparse deaths / overflow), and a deep knee penetration
that must trigger the virtual fall sensor. These are necessary,
not sufficient, before a long cold-start run.

Remaining acceptance requires:
- A genuinely **matched 2048-world GPU A/B**, plus recorded
  `CONTACT SOLVER` dimensions, `RESET BATCH STATS`, PPO update
  throughput and memory peaks on the same installed GPU stack;
- Validating no false contact-fall at initial birth, high-speed
  steps, up/down vertical lips and oblique slopes;
- Comparing held-out physical locomotion and falling rates against
  the torso-contact reference model before promoting this policy.

## GPU A/B — minimal two commands

Activate the GPU virtualenv in a clean working tree at the current
v5046 branch head. No checkpoint is written.

**Control:** 8 prisms; original torso + feet collision and original
full-world reset forward (identical topology and algorithm to
the first 8-prism XProf run):

```bash
PYTHONPATH="$PWD" python scripts/smoke_gpu_carpet.py \
  --device cuda --worlds 2048 --frames 2 --ppo --full-update \
  --benchmark-updates 3 --prisms \
  --torso-contacts --reset-forward-capacity 0
```

**Candidate:** 8 prisms; only the two ball feet collide, the
geometric body-strike sentinel applies, and the exact sparse reset
solver is enabled (128-world compact capacity):

```bash
PYTHONPATH="$PWD" python scripts/smoke_gpu_carpet.py \
  --device cuda --worlds 2048 --frames 2 --ppo --full-update \
  --benchmark-updates 3 --prisms
```

The A/B requires two separate executions, each with first-update
compilation; compare only **steady state PPO PASS** (post-compile)
and the last JSON line, not the compile time. The smoke now
prints `CONTACT SOLVER: feet_only=... padded_contact_slots...`,
plus deaths/frame and fallback frequency after timing.

If performance is surprising, independently isolate the variables
using `--reset-forward-capacity 0` without
`--torso-contacts` (feet-only vs torso) and/or
`--torso-contacts` alone (compact vs full resets).
Do not launch overnight training until the physical exams agree.

**Important:** the control run may compile differently from
the earlier v5046 XProf measurement because of any other
source changes. Use its contemporaneous steady-state timing.
