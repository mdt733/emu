# First evidence-based baseline — released EMU Pinnacle 260M

**Measured, not newly simulated.** This is a read-only analysis of the
original 260M frozen physical exams, from the operator's retained
`overnight_results(9).zip`; the model and training code were **not**
changed or run again.

## Exact evidence identity

- Policy 360-input export SHA256:
  `ef929c0220c38e5519391320e5f0629a200e1ef37a5808086f6383cfb63f7ec8`
- Original trained source: `9b11bbe596aea7201d29a6beb2d192922f8e2e9a`,
  final step **260,046,848**, seed 761. The release actor in main has
  the exact same SHA.
- Frozen examiner: CPU MJX with real paired collision slabs, **seed 1701**,
  16 worlds/case, 350 frames/7s terrain, 700 frames/14s speed.
  Terrain requests **0.15 m/s**. For paired steps, each of UP and DOWN
  has **eight** worlds.
- Original JSON SHA256 for independent byte checking:
  - `physical_exams/terrain_matrix/unified_exam.json`:
    `87c770ddd23077daf15f188df516614c9833689eb92ed66fc68681bc590edd5c`
  - `physical_exams/speed_ladder/unified_exam.json`:
    `013a0240755557b8e96d5d2f3026c9706b5042f4bcfda330cfe7a0f51f2d9d24`
- These are **previously completed** physical exams. They do **not**
  contain the new PR #18 world-frame path calculations, so route-error
  metrics remain genuinely **unknown** until read-only re-examination.

## We need three separate kinds of success

1. **Physical:** did EMU actually get a supported foot safely beyond
   the sharp step edge, or stay physically supported on a verified
   real slope?
2. **Locomotion:** did it keep useful movement with requested
   velocity and genuine support, rather than park, slide or retreat?
3. **Control:** did it maintain the requested heading/turning
   without persistent course drift? The historical mission gate
   also enforces a yaw-jitter limit that may reject benign gait
   oscillation. That is an **explicit candidate for calibration**,
   not a reason to rewrite old results.

Our new quality examiner reports these separately. Do not call
"2cm UP 8/8" a complete *controlled* capability. It is
**8/8 genuine crossing and 3/8 full mission**.

## What can it actually do? (first frozen seed)

### Real supported sharp steps

| Height | UP supported /8 | UP controlled /8 | DOWN supported /8 | DOWN controlled /8 |
|---|---:|---:|---:|---:|
| 1.0 cm | 8 | 8 | 8 | 8 |
| 2.0 cm | **8** | **3** | **8** | **7** |
| 2.5 cm | 4 | 3 | 8 | 8 |
| 3.0 cm | **0** | **0** | **8** | **4** |
| 3.5 cm | 0 | 0 | 1 | 1 |

At **3cm UP**, all eight survive but none passes the
physical crossing or movement check. Mean achieved forward
speed is **0.022 m/s** on 0.15 requested; that is almost
certainly a *terrain negotiation / forward-progress problem*,
not merely the gait-yaw threshold. A deeper torso/spring/foot
trace will be needed to identify the exact mechanism.

At **2cm UP**, all eight survive, physically cross and pass
movement tracking. **Five** miss heading qualification;
individual frozen results show five exceeding the historical
0.20 rad/s yaw-jitter threshold. That may include acceptable
stride oscillation, which the new world-route exam will
quantify separately.

At **3cm DOWN**, all eight physically cross, but four miss
heading control, so it is misleading to call either 0/8 or
8/8 the *complete* answer.

### Verified slopes at 0.15 m/s

| Grade | Uphill mission /16 | Downhill mission /16 | Cross-left /16 | Cross-right /16 | Downhill survived /16 |
|---|---:|---:|---:|---:|---:|
| 5° | 16 | 16 | 15 | 16 | 16 |
| 8° | 16 | 13 | 16 | 16 | 16 |
| 12° | 15 | **4** | 16 | 16 | **16** |
| 14.9° | 12 | **2** | 14 | 14 | **16** |

The 12° downhill robot did **not** simply fall down:
all 16 survived and passed movement tracking; 12 missed
heading qualification. Its mean signed yaw-rate bias was
**−0.099 rad/s**, with mean per-world gait yaw jitter
**0.154 rad/s**. This strongly motivates a real course/path
drift test, and a downhill speed × grade sweep. It does not
prove what the best descent speed should be.

### Sustained flat forward movement

| Requested | Achieved average | Survived /16 | Movement passed /16 | Full mission /16 |
|---|---:|---:|---:|---:|
| 0.10 m/s | 0.094 | 16 | 16 | 16 |
| 0.20 m/s | 0.188 | 16 | 16 | 16 |
| 0.30 m/s | 0.280 | 16 | 16 | 16 |
| 0.40 m/s | 0.377 | 16 | 16 | 11 |
| 0.50 m/s | 0.478 | 16 | 15 | 0 |
| 0.60 m/s | 0.571 | 16 | 15 | 0 |
| 0.80 m/s | 0.747 | 12 | 8 | 0 |
| 1.00 m/s | 0.931 | 9 | 0 | 0 |

At 0.6m/s, the robot is genuinely moving near the
requested speed and surviving 16/16. The complete-mission
failure is NOT evidence that it cannot run at 0.6; it
exposes an unresolved **control/heading problem**.
At 0.8–1m/s, survival and useful movement degrade too.

## The actual tested-range capability curves

Trapezoidal mean mission-success fractions across **only** the
observed levels; this is *not* a single global robot score,
not a new hard frontier and not comparable across ranges
without rerunning the same grid.

| Skill, signed direction | Actual sampled levels | Controlled mission AUC | Physical support/crossing AUC |
|---|---|---:|---:|
| Uphill | 5–14.9° | 0.942 | — |
| Downhill | 5–14.9° | **0.544** | 1.000 survival |
| Cross-slope LEFT | 5–14.9° | 0.972 | — |
| Cross-slope RIGHT | 5–14.9° | 0.982 | — |
| Step UP | 1–3.5cm | **0.388** | 0.600 |
| Step DOWN | 1–3.5cm | 0.775 | 0.912 |
| Forward flat | 0.1–1m/s | **0.354** | 0.896 survival |

This shows why the physically capable, balanced robot is not
yet fully controllable on steep descents or higher-speed
requests—and why the step-up weakness merits targeted analysis.

## What remains unknown, not zero

- World-frame route RMS/endpoint error and final heading offset;
  the old exams only measured yaw-rate bias/jitter.
- Whether an adaptive robot can appropriately slow down on
  steep terrain and still reach a waypoint: the current exams
  are *velocity-command obedience* tests, not route-planning tests.
- Spring preload, crouch/rebound, actuator saturation and
  verified real 3cm step-up failure mechanism.
- Independent seed-to-seed reliability (these are seed1701
  worlds; repeat at least seed1702).
- Recovery over all push headings, braking and command changes
  in these two archived panels.
- Energy per **useful directed** metre; the old positive-shaft
  J/m uses total travel, which could reward wasteful zig-zagging.

**Next experiment: none.** Next *measurement*: rerun this same
actor on the PR #18 independent exam code, gather world-path
traces, compare old mission vs actual route quality, then
calibrate a separately versioned control qualification if
the data warrants it. Keep the trained model, main and
published viewer unchanged.

[What makes a good EMU?](PHYSICAL_QUALITY.md) ·
[Current release card](V5032_PINNACLE_RELEASE.md) ·
[Future research backlog](https://github.com/mdt733/emu/issues/17).
