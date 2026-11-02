# v5040 — streaming STEP specialist

Branch: `feature/v5040-multi-lip-academy-world`. Released 260M `main` is unchanged.

## Three real platforms, no end-of-course pattern

The Academy uses three 50cm-long *real* MJX box colliders, each wide enough
for safe lateral correction. They rotate through current, next and ahead
roles. The original long flat/step slabs are buried in Academy worlds and
remain untouched in normal generalist worlds. The same fixed three
colliders are recycled independently per environment, with no model rebuild.

Each new lip independently draws UP or DOWN (50/50); height has its own
random jitter. There is no prescribed two-up/two-down sequence. A small
per-episode course yaw (up to ±6°) requires visual edge alignment; bending
terrain paths are not claimed or implemented.

A trailing platform can be recycled **only** after five consecutive frames
of real far-side top-foot support, the root at least 13cm past the edge,
both feet 3.5cm past it, and no remaining trailing-platform foot contact.
Only then does its collision body move beyond the look-ahead platform.
There are indefinitely many successive steps; the episode ends only
on fall or its own time limit, not after crossing a fixed number of lips.

The actor still sees only the real 360-input 4×4 ToF, IMU, flow and
proprioception. The platform geometry and supported-contact ledger are
strictly teacher-only. The Academy gets no auxiliary CONTROL specialist
effort weight; the usual walking, balance and motor physics remain.

## Independent uphill and downhill growth

The governor stores two separate frontiers:
`academy_height_up_m` and `academy_height_down_m`, both seeded at
3mm. Each has its own approached-lip evidence window and promotes by
0.5mm once near-frontier success is at least 55% across 12 encounters.
Repeated easy-band failures retreat only that direction by 0.5mm.
A successful descent cannot promote ascent, or vice versa.

Each *newly generated* step is approximately 50% of its own directional
frontier early in an episode, 100% halfway, and 150% late in the episode,
plus small independent height jitter. Challenges above the current
frontier are learning opportunities, NOT evidence of mastery.
The default **earned frontier ceiling is 15cm**, configurable to 25cm.
This is aspiration/headroom, not proof real hardware could climb a gutter.

Verified uphill progress can earn one-off clearance, target foothold,
weight transfer and crossing rewards. Descents earn real supported
crossings without imaginary upward lifting. Every four real crossings
produce a *milestone counter*, never an artificial finish.

## Generalist curriculum stays

One actor, one PPO, six specialties. The proven first 20M flat walking
foundation and ordinary single-lip STEP worlds are preserved. Academy
births ramp 0→10% between 60–80M, then 10→50% between 80–100M, without
deleting fast walking rehearsal or other skill allocations.
An Academy episode may last 1200 frames (24s); ordinary ones 700
(14s), so occupancy can be higher than the share of births.

## Before a long cold-start train

From a clean `v5040` checkout, with matching CUDA/JAX/MJX environment:

```bash
python -m emu.cli doctor --require-gpu --no-mjx-smoke
PYTHONPATH="$PWD" python scripts/smoke_step_course.py \
  --device cuda --worlds 128 --frames 3 --ppo
PYTHONPATH="$PWD" python scripts/smoke_step_course.py \
  --device cuda --worlds 2048 --frames 3 --ppo --full-update
```

The 2048×32 update may OOM an 8GB GPU. It saves no training weights.
If it passes, normal `emu train` with `--academy-stream`,
`--academy-height-cap-m .15` and
`EMU_UNIFIED_CONTINUOUS=cold-growth-approved` can launch a new,
source-pinned cold run in a **new** results directory. Frozen policy
exams use `scripts/exam_step_course.py` and report actual approached
up/down lips separately. Successful smoke tests do NOT prove learning.
