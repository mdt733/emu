# v5046 — 34-arm 4060 Ti solver campaign results (5 November 2026)

> **Source**: operator-uploaded `overnight_results(20261104-194041).zip`,
> from `scripts/overnight_solver_campaign.py` at
> `7f2edd988fcb2e637577b7c5aa23b3e274aac856`.
> This is measured evidence, not a proposal or a code change.

## Outcome

**34/34 arms PASS**, two seeds (735, 761), 17 configurations each,
8 real full PPO updates per arm (first JIT/compile excluded), 2,048
GPU worlds, 32-frame rollouts: nominal **17,825,792 world-steps**
including compilation warmups. Eight real dynamic convex triangular
prisms, 360 inputs, standard feet-only physics except one explicit
torso-collider A/B arm.

Results were accumulated between 2026-11-04 11:47 and 13:33 UTC.
The passing preflight was **CPU**; CUDA-specific physical-contact and
compact-reset parity remain distinct open investigations. Every GPU
arm independently passed its own 2-frame geometry/recycle/reset smoke
and full, finite PPO update.

### Measured steady throughput

Throughputs below are medians across the two seeds (because each
configuration has exactly one run per seed, these are also the
arithmetic mean). **These runs do not measure actual sustained
learning or earned step mastery.**

| Configuration | World-steps/s | x default | Purpose |
|---|---:|---:|---|
| Default MuJoCo 100 / 50 | **31,974** | 1.00x | reference |
| 8 / 8 | 66,669 | 2.09x | conservative limited solve |
| 4 / 4 | 89,921 | 2.81x | earlier validated speed candidate |
| 2 / 4 | **119,662** | **3.74x** | **next physical validation candidate** |
| 2 / 2 | 121,665 | 3.80x | similar speed, compare quality |
| 1 / 1 | **162,930** | 5.10x | fastest, physical warning signals |
| 4 / default LS (4 / 50) | 45,243 | 1.42x | isolate linesearch burden |
| default Newton / 4 LS (100 / 4) | 75,130 | 2.35x | isolate Newton burden |
| 4 / 4, reset cap 192 | 95,029 | 2.97x | capacity A/B, inconclusive |
| 4 / 4, reset cap 256 | 93,193 | 2.91x | capacity A/B, inconclusive |
| 4 / 4, full reset cap 0 | 89,846 | 2.81x | reset codepath A/B |
| 4 / 4, torso collisions on | 51,429 | 1.61x | topology/constraint cost |
| repeated default | 30,160 | 0.94x | timing drift reference |
| repeated 4 / 4 | 90,447 | 2.83x | timing drift reference |

Other observed arms: 4/8=81,532; 8/4=73,421; 2/4
beats 4/2=93,451. The repetition and per-seed spread argue against
treating single-digit percent timing differences as established.

### Contact quality signals: averaged across seeds

The last 32-frame PPO rollout of each arm sampled a **newly
initialised, partially updated policy**; these are *not* repeated
frozen-action trajectories or a genuine learned walking exam.

| Solver | Forward vx (m/s) | Stance slip proxy | Virtual nonfoot strike | Fell/event fraction |
|---|---:|---:|---:|---:|
| Default 100/50 | 0.3491 | 0.09720 | 0.001480 | 0.03431 |
| 8/8 | 0.3491 | 0.09756 | 0.001373 | 0.03426 |
| 4/4 | 0.3478 | 0.09721 | 0.001434 | 0.03406 |
| **2/4** | **0.3434** | **0.09743** | **0.001526** | **0.03399** |
| 2/2 | 0.3481 | 0.09798 | 0.001511 | 0.03421 |
| **1/1** | **0.3192** | **0.11636** | **0.003113** | **0.03345** |

1/1 versus 4/4: **+19.7%** stance slip, **+117%**
virtual nonfoot strikes, **-8.2%** forward speed,
**+9.2%** bad-contact proxy, and **+124%**
virtual-contact fall signal. Its aggregate fall/event fraction being
slightly lower does **not** imply improved stability. Treat 1/1 as
**provisionally rejected for production** pending explicit contact
parity testing.

2/4 versus 4/4: **+0.22%** stance slip proxy, **+6.4%**
virtual strike proxy, **-1.3%** vx. These differences are small in
this smoke experiment, but neither statistically significant nor
enough to certify solver correctness. 2/4 is a candidate for controlled
physics validation, **not a new training default**.

`terrain_cross_new` was **0** in every sampled final rollout.
`academy_cross_new` did register nonzero events, but the campaign
provides **no demonstration of successful repeated step-climbing**.
No frozen walking actor was evaluated. This distinction is important.

### Reset capacity: no convincing increase yet

For 4/4, 128-cap baseline 89,921 steps/s,
192-cap 95,029 (+5.7% aggregate), 256-cap 93,193,
0/full-cap 89,846. Relative to baseline 128:

- Seed 735: 192 **+14.2%**; 256 +10.3%.
- Seed 761: 192 **-1.9%**; 256 -2.3%.
- The final 32-frame reset report for both 4/4 baseline seeds
  showed 0 overflows of cap 128 (observed max 84 / 97).
  Earlier short-run work had rare >128 events. There is no
  robust overnight evidence of a capacity-192 benefit here.
- The 4/4 repeat varied by over 10% for individual seeds.
  Use a substantially longer single-process run and aggregate
  overflow statistics across all 224 measured frames per arm.

### Collision topology: the strongest non-solver result

Re-enabling the torso geom increases padded contacts
**17 -> 51** and constraint rows **82 -> 218** per world;
4/4 throughput drops **89,921 -> 51,429**
(**-42.8%**). Memory peak increases from ~1.09 GB to
~1.28 GB. This confirms why collision topology reduction is
worth investigating. However `virtual_nonfoot_strike` becomes zero
by definition with torso collisions switched on, so metric parity
must not be inferred from comparing those zeros to feet-only mode.
Physical policy training can still exploit a missing torso collider;
retain geometric strike/fall sentinels and test full-body stability.

### Initial failed preflight: actionable CUDA discrepancy

The archived failed preflight
`preflight_failed_a2c0ee182ed8.log` shows that on the operator GPU
the compact vs full `mjx.forward` probe failed in **packed mode,
pytree leaf 13** for 2 of 28 compared elements, with maximum absolute
disagreement **0.00027889** (relative maximum 0.00035754), against
probe tolerances `rtol=2e-4, atol=2e-5`. The subsequent forced-CPU
preflight passed and was the same backend as GitHub Actions. This
might be harmless GPU reduction-order rounding or a real forward
batch-size dependence. **Do not relax tolerances or call it harmless
without identifying the leaf, field and physical units.** Build a
named-field CUDA parity report on a fixed deterministic state, record
absolute and relative deviations and evaluate impact on foot impulses,
step clearance and resets.

### Recommended decision and next gated work

1. Keep current production robot.xml and main unchanged. The entire
   solver A/B sweep is **temporary benchmark XML**, not a committed
   physics-model change. Keep PR #35 draft.
2. Promote **2/4 and 4/4 only to a deterministic physical exam
   bracket**, with 8/8 as conservative comparator and 1/1 as
   deliberately negative control: signed 0, 3mm, 4cm, 15cm
   and 40cm sidewalls, varying lateral approach, 15deg slopes,
   both-feet support, joint/slip/contact impulses, energy, sensor
   parity and trajectory divergence. Pass on real CUDA MJX,
   not only CPU `mjx.forward` geometry.
3. Compare 2/2 versus 2/4 only if 2/4 passes physical
   accuracy; the 2% speed difference is too small to justify
   a weaker solver without evidence.
4. Instrument named leaf output for compact-reset/full-forward
   parity on GPU and count overflow events across full measured
   runs before changing reset capacity.
5. Afterwards test four independent heightfield collision geoms
   versus eight prisms as a separately gated physical branch,
   NOT as an unobserved change hidden inside the solver sweep.
6. Only after physical parity and a longer held-out trained
   walking exam, consider a pinned new physics recipe and
   fresh training/learning comparison. No 140M release claim yet.

## Measurement boundaries

The current throughput is full PPO and synchronized, **excluding**
JIT compilation, logging/checkpoint storage and high-level
curriculum orchestration. Eight PPO updates per fresh process are
not interchangeable with a single long run training 17.8M
sequentially. The source SHA is the archived `7f2edd...`
regardless of newer report commits. Repeating at matched
temperatures / GPU idle would improve small-difference precision.
