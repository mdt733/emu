# v5048 shared-parent experiment — measured findings

**Date:** 5 November 2026 (Sydney). **Experiment source:** `088a29be47f624be9e2c322c504ec98049b3f9bf`.
**Evidence:** operator-supplied `results_bundle(20261105-070042).zip`:
`state.json`, `SUMMARY.md`, 148 independently frozen `prism_exam.json` panels,
training metrics, source manifests, and ten trained-policy exports
(one 50M parent, eight 100M branches, plus one 200M export).

**Research parent:** 50,003,968 steps, seed 735, 2048 CUDA MJX worlds,
same-source final.pkl SHA256
`8047f4bd4d75894c627e8689e0c645a6bde23496caf6a807f408fa8c88a4085f`.
Each of eight branches loaded that exact 50M parent (source/age in
`run_meta.json`) and ran to 100,007,936 steps. The selected
`temporal_speed` policy **continued in the SAME run** to
**200,015,872 steps**. All trained models use one 360-input/six-output
policy and real four-75cm-tile/eight-convex-prism simulation.
All but the explicit `solver4_control` experiment used Newton/LS
**2/4**; that arm used **4/4**. Total **550M unique world-steps**,
ten train stages and **148/148** frozen 500-frame GPU exam cases
(16 worlds, up to two held-out seeds 1701 and 1702). Campaign elapsed
approximately 5h40m including compilation and exams.

## Major result — Academy helps with physically supported steps

The examiner requires the body to enter a distinct adjacent source tile,
a real prism-foot top-contact near the correct triangle surface, and
five consecutive supported frames. Signed rise is computed at the
*actual tile boundary*, not copied from the requested curriculum
height. These are real supported transitions, not privileged PPO
reward counters. However this protocol does **not** yet demand
one metre of continued heading-controlled walking after each crossing.

The nominal **3cm** step challenge produced actual supported lip
heights roughly **1.8–2.9cm**; none of the recorded transitions
proved full **3.0cm** ascent or descent. Positive steps in the
200M candidate ranged at most **2.624cm**. The nominal 2cm panel
often gave actual **1.2–1.8cm**, and the 5mm panel **0.3–0.46cm**.
**Never call a requested 3cm exam a verified 3cm climb.**

| 100M research arm | Distinct worlds with ≥2.0cm supported UP, nominal-3cm panel (of 32) | ≥2.5cm UP (of 32) | Survived nominal-3cm worlds | Fast achieved vx (0.70 requested) | Absolute fast yaw bias rad/s | Still-command drift over ~10s |
|---|---:|---:|---:|---:|---:|---:|
| No Academy | **2** | 0 | **3/32** | 0.626m/s | 0.284 | 14.4cm |
| Early-lips Academy | **19** | 6 | **26/32** | 0.622m/s | **0.100** | 14.2cm |
| Precise 1mm frontier | **21** | **7** | 24/32 | 0.646m/s | 0.278 | 12.4cm |
| Zero-command balance | 17 | 6 | 19/32 | 0.637m/s | 0.375 | 11.4cm |
| Temporal velocity tracking | 16 | 5 | **26/32** | 0.586m/s | 0.346 | **11.1cm** |
| Fast rehearsal | 18 | 1 | 21/32 | **0.653m/s** | 0.284 | 22.5cm |
| Heavy Academy 65% | 14 | 3 | 24/32 | 0.630m/s | 0.188 | 21.2cm |
| Newton/LS 4/4 | 13 | 3 | 16/32 | 0.599m/s | 0.109 | 18.7cm |
| **Temporal policy continued to 200M** | **16** | 5 | **23/32** | **0.595m/s** | **0.180** | **11.7cm** |

These fractions are **test-world incidence**, not exposure-normalised
mission success probabilities: some worlds contain multiple rises
and other worlds may not offer the same signed height/opportunity.
The two exam seeds were held out from training; the research eight
branches share seed 735 and an exact saved parent. For signed
DOWN ≥2.0cm, the 200M policy had **22/32 worlds**; the 100M
early-lips policy had **25/32**.

The matched-parent comparison is persuasive: dedicated early-lips
and the 1mm frontier substantially outperform the non-Academy
branch on harder supported uphill encounters and 3cm-*requested*
survival. The 50M parent already had 14 supported up-transition
events in its nominal 2cm (actual 1.2–1.8cm) preliminary 16-world
exam, so there was **some pre-Academy ability**; do not attribute all
step learning exclusively to post-50M teaching.

## The selected temporal-speed finalist and the scoring trap

The automated score chose **temporal_speed at 100M** (136.231)
just ahead of **early_lips** (135.977), a **0.254-point**
difference. All eight arms passed the broad `qualified`
survival/fast-motion gate. Crucially the score's supported-step term
is `8 * min(12, supported_up + supported_down)`: **every arm exceeded
12 crossings and received the same maximum step credit**. Hence the
headline score cannot distinguish 2cm/2.5cm stair competence or
post-lip progress. This is an **experimental scorer-design
limitation**, not evidence temporal reward specifically solves stairs.

The 200M temporal continuation did learn some useful control:

| Independent frozen metric | Temporal 100M | Same policy 200M |
|---|---:|---:|
| Slow achieved vx (0.06 requested) | 0.043m/s | **0.054m/s** |
| Cruise achieved vx (0.20 requested) | 0.172m/s | **0.187m/s** |
| Fast achieved vx (0.70 requested) | 0.586m/s | 0.595m/s |
| Fast survival | 31/32 | 31/32 |
| Fast absolute yaw bias | 0.346rad/s | **0.180rad/s** |
| Fast same-foot forward stride | 16.0cm | 16.8cm |
| Still-command net displacement | **11.1cm** | 11.7cm |
| Nominal-3cm survival | **26/32** | 23/32 |
| Nominal-3cm worlds with actual ≥2cm uphill support | 16/32 | 16/32 |
| Nominal-3cm worlds with actual ≥2.5cm uphill support | 5/32 | 5/32 |
| Composite research score | **136.231** | 135.728 |

At zero command it still made about **8.5 detected footfalls per
second across both feet**, so it has *not* learned a quiet stationary
pose. A command near zero is being dynamically maintained with
rapid steps. The longer stride at 0.7m/s is promising but does not by
itself prove a more efficient dynamic gait; heading/turn performance
is not yet certified. At 200M the fast mechanical drive-work proxy
averaged ~54.4J/useful m, versus ~36.9J/m for the 100M early-lips
arm on its own matching procedure. The 100M temporal arm's mean
energy ratio is distorted by a few low-progress outliers.

**Bottom line:** temporal_speed is a legitimate *automatically
selected research finalist*, but **not proven the best stair policy**.
Early-lips 100M is a more compelling all-round harder-step/heading
candidate; precise-frontier 100M had the most distinct worlds with
≥2cm uphill support. The 200M extension improved slow-command
tracking and yaw bias, but it **did not improve the ≥2cm/≥2.5cm
uphill test-world counts**, and 3cm-requested survival declined.

## Throughput and physical limitations

Across the primary 2/4 branch runs, steady `training_fps` was
about **117k–122k world-steps/s**; the 4/4 control finished at
about **91k world-steps/s** (~25% less) without a demonstrated
step-quality improvement. The throughput gain over the former
~6k-era trainer is a major engineering benefit. The old CUDA
packed/full reset `.qacc` mismatch and possible underresolved
2/4 contact fidelity are still unresolved and must not be presented
as hardware-verified physics.

The training rollout's latest `fall_rate` (for example 0.463 at
200M) is an Academy-heavy training population metric, **not** the
same as the frozen 0.7m/s fast-panel survival 31/32. Keep these
denominators separate.

No direct comparable v5032 `main` examination was run in this bundle.
The incumbent's historical two-slab fixture has separate exposure,
physics and mission criteria; neither a nominal-height string nor
new procedural crossing counts proves a head-to-head win.

### Frozen 200M candidate identity

- Experiment source SHA: `088a29be47f624be9e2c322c504ec98049b3f9bf`
- Original training seed: `735`; checkpoint age **200,015,872**
- Exported file: `v5048-temporal_speed-s735/exports/step_200000000.emupolicy`
- Export file SHA256:
  `36508ecca685cb7f113f62502541594f8f9da345dda3025dbb4c9e7fac6668c6`
- Actor-weights SHA256:
  `29219e34e149159df1d050281903c79752d665fa13dcafe303bafcc45bfbdce1`
- Physical robot SHA256:
  `c1af72485e83de277efa1914a476519cebbebdc2beb1f03a10c475c9fadf3a64`
- *Final 200M `final.pkl` checkpoint SHA256 is not included in the
  upload.* Only its output actor, provenance and experiments are available.

## Next engineering/release decision

1. **Retain** the four-tile/eight-prism physics, 360-input single
   policy, source-pinned checkpoints, compact independent examiner
   and measured GPU throughput as valuable engineering.
2. **Retain only proven teaching changes:** early dedicated steps
   clearly help; fine 1mm step frontier merits confirmation. More
   Academy share is not automatically better. Temporal speed is
   promising for slow tracking and yaw, not yet a stair solution.
   Keep zero-balance as opt-in until truly standing behaviour exists.
3. **Upgrade physical certification rather than score another
   oversized tournament:** deliberately construct *fixed actual*
   1, 2, 2.5, 3 and 3.5cm signed step walls, two bearings, record
   eligible approaches, attempts, fall avoidance, top-of-next-tile
   support, **sustained post-step progress and heading**; compare
   *equivalent* v5032 and current actors. Include zero-velocity
   stability and turn/strafe command exams.
4. **Do not automatically replace the public 260M policy.** The
   new robot has a faster research trainer and genuine supported
   2cm encounters, but head-to-head controlled mission parity,
   3cm actual ascent, robust stillness, yaw/path control and
   2/4 contact parity have not been demonstrated.
5. **Cleanup:** the disposable `scripts/v5048_shared_parent_tournament.py`
   has already been removed from the future-main release-prep
   [PR #39](https://github.com/mdt733/emu/pull/39); it remains
   source-pinned in the original v5048 commit for reproduction.
   Keep this measured findings card and issue #17; don't retain
   another permanent experimental framework.

**Decision:** Gate A engineering promotion deserves owner review
after final source CUDA acceptance. Gate B public-model promotion
remains **HOLD** pending matched physical exams and explicit approval.
