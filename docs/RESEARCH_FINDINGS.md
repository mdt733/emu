# EMU — retained research findings

**Current source:** [H40 handover](../handover.md). This is a concise **evidence ledger**, not a second set of runbooks. The full 2026 experiments, source, commands, reports and discarded proposals remain in [the pre-cleanup Git snapshot](https://github.com/mdt733/emu/tree/ded2a111c9ab50c304e1b1a70736a84b32449887/docs) and their original PRs. Each result below identifies **its own** physics, source, seeds and exam fixtures. Historical v50xx results must **not** be treated as H40's measured capability; the matched frozen-H40 CUDA study below is the current measured stair baseline.

## Cold school 120M sanity run — 23 November 2026 (uploaded evidence)

**Status: complete from-birth execution/physical evaluation, NOT teacher qualification.**
User-supplied `school-evidence.zip` (~5.8 MB, verified archive including embedded
pulse/deep ZIPs), project run `h40-school-cold-120m-s735`, **120,061,952 steps**,
training seed **735**, **2048 MJX worlds**, zero external ancestor and
no recorded resume; frozen training source
`1fa832fc7cd9241e2772b44ad434613c6080adfa` (clean branch),
`h40-spring-only` exact generated robot SHA
`21b5856b7a248e5d5b988e2735255d57ec79bae1a7cab0bcb72f481bfc36f470`,
frozen actor SHA `2a5f56992dc715470d9adf49ef06f7063f3df437890131353c97256994fc4336`.
The release `main` H40 shares the nominal **400 mm morphology**
but has **different knee-spring mechanics, generated model hash and actor**.
The current research branch has advanced **past this training commit**; never
attribute this learner to newer cleanup commits. Complete checkpoint bytes
are *not* in the uploaded evidence ZIP; full restartable state stays with
the operator.

**Pipeline passed:** 1832 PPO rollouts, no recorded gradient-NaN skips
or rollbacks. Independent read-only examinations ran on **evaluation seed
4701** with zero optimiser updates; actor/robot/source identities agree
across the signed-prism matrix, six-specialty pulse and deep 30/60s exam.
The fixed held-out retention panel improved 20M→120M from broad
score **0.012→0.847**, reaching 100% 350-frame core, dynamic80 and
combo70 survival and 85.9% resilience75 survival at the end.
This is solid pipeline/gait learning, **not** proof of robust task completion.

**The true first-life signed-prism exam** (`exam/stairs/matrix_curve.csv`,
**50 trials per exact height**, 4–80mm in 4mm increments, seed4701):

| First-life outcome | UP 4–40mm / 500 | UP 44–80mm / 500 | DOWN 4–40mm / 500 | DOWN 44–80mm / 500 |
| --- | ---: | ---: | ---: | ---: |
| Leading foot reaches actual destination top | 352 | 0 | 500 | 485 |
| Each foot independently supported ≥5 frames | 140 | 0 | 266 | 133 |
| Controlled 5-frame downstream/lane traversal | **39** | **0** | **138** | **71** |
| Full-life survival to exam horizon | 495 | 262 | 499 | 404 |

**Uphill first-contact cliff:** +28mm 17/50 first top contacts, +32mm
2/50, +36mm through +80mm **0**. Zero strict parked-arrival
completions throughout the matrix are *not* evidence of zero controlled
walking; the training school's global-XY parking gate was intentionally
disabled. Conversely 95–100% survival at some higher steps can mean
**stopping before the obstacle**, not climbing.

**The teacher's bottleneck:** recorded Academy uphill frontier **began
and ended at 3mm**, with only three short promotions to 3.5mm
(~66M/~77M/~117M) followed by safety retreats. Academy DOWN quickly
earned its **55mm cap**. The last logged rollout's ordinary Academy
near-up successes were **23/23**, but actual **completed-first-life**
near-up evidence was **12 births, 8 survived, 5 bilateral transfers,
only 4 safe bilateral transfers**. These two counters measure different
events and must never be compared as synonymous or accumulated across
logged rows. The uphill governor requires ≥24 completed near-height
births, ≥55% safe bilateral crossings in the same life and ≥90%
full-life survival to expand; ≥24 births below 85% survival trigger
retreat. Its 1200-frame nursery may penalize falling *later* even after
a first-lip success. Could be insufficient curriculum signal,
genuinely unsafe skill or both; **do not lower safety gates just to
make the frontier number grow**. The original study never deliberately
taught 8cm Academy ascent.

**Other physical capabilities:** 12s/2048-world pulse:
command **357/384** passed, transition **254/256**,
control **237/256**, slope **222/384**, resilience **179/256**,
step **140/512** (these are separate specialist gates, not the signed
matrix traversal criterion). Longer 8-world deep exam had **8/8
30s/60s survival in every scenario**; yet **zero/8 bounded 60s
station-keeping**, forward 0.25m/s mission **1/8**, reverse
0.20m/s **0/8**, left/right 0.15m/s **0/8**, turn-left
**7/8**, turn-right **1/8**, combined **6/8**, and
forward-stop-reverse **0/8**. Staying upright is not adequate evidence
of obeying heading/route/stop commands; eight worlds give wide
uncertainty. The capability-field ledger stayed `shadow` with
zero teaching influence.

**Interpretation / next test:** The 120M experiment **did not break**
GPU PPO, replayable source identity or independent examination. It
**did not produce reliable uphill stairs**; course survival and general
locomotion are the strengths, uphill acquisition/controllable
continuation the deficits. Review why the same 3mm uphill cohort
oscillates between expansion and fall-driven retreat, then improve
the **existing** physical teaching/credit decision in place—no extra
teacher, branches or stacked rewards. For a causal teacher comparison
freeze one revised recipe and train multiple **fresh cold seeds** at
equal budgets against a frozen baseline. Because seed4701 results have
now been inspected to guide improvements, reserve **new previously
unseen physical examination seeds** for evaluating the revised teacher;
4701 remains valid as this candidate's preregistered test, not a
permanently pristine future holdout. No automatic promotion.

---

## H40 2,000 same-world frozen-actor signed stair vs flat — 11 November 2026

**Read-only measured CUDA evidence; no PPO updates.** Frozen H40 120,061,952-step actor (training source `301d18b7b1cd1b165e6578fc53f9f8cda94fb680`, export SHA256 `178c201af661d3172d701a7d1d8a7711f9b2106e1f7058bc4902aa9c64a37c7a`, weights SHA256 `d4e35f814baef5e46afa6e96e942c86ab100ef25a4ae1913c76bbdb22615e673`, robot XML SHA256 `b513bb3326a73d9ca661f761f1352b22c36d0f1b6d51be72c2eca15d8712ef5b`). Exact physical 1 m signed-lip prism seams and zero-lip control, 2048 MJX GPU worlds, **2000 same-world indexed pairs**, 20 height levels each direction from ±2.5 to ±50mm, **50 repeated trials per sign/level**, seed 1701, 350 physical frames. [Paired comparison PR #69](https://github.com/mdt733/emu/pull/69) rejects mismatched actor, model, commands, seed, batch, horizon and examiner-source fingerprint. Full user-supplied aggregate `matrix_compare.json`, `matrix_paired_curves.csv` and analysis are retained in [master issue #62](https://github.com/mdt733/emu/issues/62); `matrix_pairs.csv` was not attached in this chat so only supplied aggregate pair counts have been independently rechecked.

| First-life outcome (each signed direction: 1000 stair worlds, 1000 matched flat worlds) | UP stair / flat | DOWN stair / flat |
| --- | ---: | ---: |
| First foot supported by destination prism | **587 / 1000** | **1000 / 1000** |
| Each foot independently supported for five frames | **327 / 574** | **454 / 581** |
| Upright downstream crossing, five frames | **322 / 574** | **453 / 581** |
| **Controlled lane+heading traversal, five frames** | **182 / 364 (−18.2 pp)** | **285 / 381 (−9.6 pp)** |
| Strict stable parked arrival (separate from walking) | **8 / 17** | **8 / 12** |

**UP: sharp new physical failure near 30–32.5mm.** At **+2.5–27.5mm**, first-foot contact **549/550**, independent both-foot support **310/550** versus paired flat **323/550**, controlled traversal **179/550 (32.5%)** versus flat **192/550 (34.9%)** (**−2.4 percentage points**). This is approximately flat-equivalent under the examiner's restrictive continuation/foot-contact gates; it does not certify perfect physical stepping. At **+30mm**, first-foot supported **36/50**, fully controlled crossing **3/50** versus flat **23/50** (speed MAE ~0.117m/s worse). At **+32.5mm**, first foot **2/50**, controlled crossing **0/50**. At **+35–50mm**, first foot **0/350** and controlled crossing **0/350**, versus **132/350** flat controlled crossings. Aggregate **+30–50mm** traversal is **3/450** versus matched flat **172/450** (**−37.6 pp**). A repeatable cliff in *destination-foot support* is observed but no specific mechanical, sensory or learned policy cause is yet proven.

**DOWN: different, gradual loss.** First foot supported on lower destination **1000/1000** across 2.5–50mm descents. Controlled traversal DOWN **2.5–27.5mm: 171/550** vs paired flat **207/550 (−6.5 pp)**; DOWN **30–50mm: 114/450** vs paired flat **174/450 (−13.3 pp)**. Trailing-foot support and lane stability lose ground with increasing drop rather than hitting the same first-foot cliff.

**Interpretation:** The original flat exam itself only passes controlled continuation **745/2000 (37.25%)** despite 100% survival and goal-line crossing. Strict stable parking passes **29/2000 (1.45%)**; don't train toward parking as if it were ongoing gait. The main incremental tall-step issue is **upper foothold acquisition**; competing causes for targeted investigation: ToF lookahead and edge timing, ballistic clearance/phase timing, joint reach, C018 rated torque, passive spring, collision and learned refusal. Next **source-pinned frozen-policy threshold autopsy** should compare +27.5/+30/+32.5/+35mm first-contact/foot-clearance/ToF/actuator traces before choosing *one* PPO experiment. Broader commands/slope/recovery/shaft and real-hardware uncertainty remain open. This evidence covers **one policy training seed and one evaluation seed**, not a sim2real or cross-policy replication.

An earlier **64-world CUDA smoke** (one trial per height/sign) validated execution and exact signed geometry but was too sparse to characterize capability; it is superseded by these repeated matched trials. Raw trial files remain at the owner's local `~/emu-data/exams/` directories; don't rerun/overwrite completed outputs.

## Findings that changed the design

| Investigation | Evidence and retained decision |
|---|---|
| GPU/MJX cost, v5045–v5046 | The solver/contact path dominated rollout cost. On the measured 2,048-world overnight study, default 100/50 solver achieved **31,974 steps/s**; 8/8 reached **66,669**, and 4/4 **89,921**. These are **that study's** throughput measurements, not guaranteed H40 speed. Foot-only contacts and eight prisms improved cost, while reset-forward / solver parity required continued scrutiny. See [results](https://github.com/mdt733/emu/blob/ded2a111c9ab50c304e1b1a70736a84b32449887/docs/V5046_OVERNIGHT_RESULTS_20261105.md) and [workload audit](https://github.com/mdt733/emu/blob/ded2a111c9ab50c304e1b1a70736a84b32449887/docs/JAX_GPU_WORKLOAD_AUDIT.md). |
| Shared-parent tournament, v5048 | A 50M parent, eight forks, and a 200M selected continuation yielded **148 independent frozen exam panels**; Academy exposure helped some authentic foot-supported steps but did **not** establish full reliable 3 cm climbing or post-step heading completion. Speed rewards alone gave a misleading winner signal. See [results](https://github.com/mdt733/emu/blob/ded2a111c9ab50c304e1b1a70736a84b32449887/docs/V5048_SHARED_PARENT_RESULTS_20261105.md). |
| Clock and tippy-tap, v5051 | A stationary-command activity gate could reduce posture/contact payments by up to **0.826 per 20 ms frame** under the documented idealized comparison. Phase observations persist at low speed. This is a *source-grounded hypothesis* for restless gait, **not proof** that the clock caused stair failure. See [gait autopsy](https://github.com/mdt733/emu/blob/ded2a111c9ab50c304e1b1a70736a84b32449887/docs/V5051_GAIT_AUTOPSY.md). |
| Step manoeuvre vs examiner, v5052–v5054 | Historical training saw **776 authentic top-contact events** but **zero strict completed post-contact missions** in one stair study. The exact-height matrix differentiated leading-foot top contact, both-feet support, forward traversal and precise park-and-hold. In an independent **20 mm** screen, **89/100** attained five-frame both-feet top support and **86/100** got the body downstream: a tight parking score is not the same as zero physical crossing ability. See [step school](https://github.com/mdt733/emu/blob/ded2a111c9ab50c304e1b1a70736a84b32449887/docs/V5052_STEP_SCHOOL.md), [matrix](https://github.com/mdt733/emu/blob/ded2a111c9ab50c304e1b1a70736a84b32449887/docs/V5053_GPU_MATRIX.md), and [scoring reset](https://github.com/mdt733/emu/blob/ded2a111c9ab50c304e1b1a70736a84b32449887/docs/V5054_STAIR_SUCCESS_RESET.md). |
| More training can make hard steps worse, v5055 | A reported 2,000-trial matched-height comparison found five-frame both-feet support **1,348→1,096** and downstream traversal **531→421** between the historical 80M and 182M actors. At **32.5–40 mm**, controlled traversals fell **70/400→11/400**; do not interpret a longer curriculum as uniform competence growth. Hip-roll saturation and action-credit direction remained concerns. See [credit audit](https://github.com/mdt733/emu/blob/ded2a111c9ab50c304e1b1a70736a84b32449887/docs/V5055_LEARNING_CREDIT_AUDIT.md). |
| Physical feasibility, v5056–v5058 | Historical motors mixed voltage/speed assumptions, and an early 40 cm result allowed **±2.941995 N·m stall torque** rather than the C018's **±0.980665 N·m rated bound**. The older 200M result reported **57.7% over-rated exposure**, so it could not certify rated-torque feasibility. Later work enforced the lower cap and re-identified the 40 cm physical model. See [12 V design](https://github.com/mdt733/emu/blob/ded2a111c9ab50c304e1b1a70736a84b32449887/docs/V5056_KNEE_STRUT_C018_STAIR_PLAN.md), [height matrix](https://github.com/mdt733/emu/blob/ded2a111c9ab50c304e1b1a70736a84b32449887/docs/V5057_HEIGHT_PHYSICAL_MATRIX.md), and [torque correction](https://github.com/mdt733/emu/blob/ded2a111c9ab50c304e1b1a70736a84b32449887/docs/V5058_C018_RATED_TORQUE_HARDCAP.md). |

## Current H40 baseline: what these studies do **not** prove

The owner-approved H40 mainline is the **120,061,952-step**, seed-735, 40 cm / six chassis-mounted C018 / rated-torque model with a 45°×45°, 70°-down 4×4 sensor and four rolling 1 m tiles. Frozen training SHA: `301d18b7b1cd1b165e6578fc53f9f8cda94fb680`. Its exact trained `showcase/champion.emupolicy` and checkpoint SHA live in the [handover](../handover.md) and [release readiness](RELEASE_READINESS.md).

The owner reports pleasing viewer walking and slopes approaching 15°; these are **qualitative**. A **matched 2000-pair signed MJX step-vs-flat physical exam is now measured**, with nearly flat-equivalent controlled continuation up to +27.5mm and a sharp first-upper-foothold loss at +30–32.5mm; see results above. Repeatable high-step ascent, full command/slope/recovery/effort envelope, actual hardware actuation, remote knee transmission losses, motor thermal behaviour, battery sag and mass **remain unproven**. Retired experiments cannot be ranked as if they shared the same robot, sensor, reward, evaluation and physics.

## Recovery instead of permanent experiment runbooks

- [Historical dimension map](https://github.com/mdt733/emu/blob/ded2a111c9ab50c304e1b1a70736a84b32449887/docs/CAPABILITY_DIMENSIONS.md) and [unified growth contract](https://github.com/mdt733/emu/blob/ded2a111c9ab50c304e1b1a70736a84b32449887/docs/UNIFIED_CAPABILITY_GROWTH.md) explain the old reward/curriculum rationale. Today's single actor/curriculum and ownership are described in the [core skeleton](CORE_SKELETON.md).
- [old 260M independent physical baseline](https://github.com/mdt733/emu/blob/ded2a111c9ab50c304e1b1a70736a84b32449887/docs/PHYSICAL_BASELINE_260M.md) and [v5032 release card](https://github.com/mdt733/emu/blob/ded2a111c9ab50c304e1b1a70736a84b32449887/docs/V5032_PINNACLE_RELEASE.md) retain the exact earlier reference geometry and results.
- [old 960M research plan](https://github.com/mdt733/emu/blob/ded2a111c9ab50c304e1b1a70736a84b32449887/docs/V5051_TRAINING_CAMPAIGN.md), [contact simplification proposal](https://github.com/mdt733/emu/blob/ded2a111c9ab50c304e1b1a70736a84b32449887/docs/V5046_FOOT_CONTACT_OPTIMIZATION.md) and [old viewer park](https://github.com/mdt733/emu/blob/ded2a111c9ab50c304e1b1a70736a84b32449887/docs/VIEWER_GRADUATED_PARK.md) are historical, not commands for current `main`.
- All 18 retired pages are recoverable from the [source-pinned old docs tree](https://github.com/mdt733/emu/tree/ded2a111c9ab50c304e1b1a70736a84b32449887/docs). GPU data and the original run bundles are not embedded in this repository.

**Rule for future studies:** one concise measured result in this ledger, plus a source-pinned commit/run and the relevant externally retained evidence. Remove one-off run instructions and inactive branches when a study ends. Git is the archive, not the working tree.

## KS150 spring-only stair-school evidence — 22–23 November 2026

**Research evidence, not the approved browser champion and not hardware
validation.** In two independent *evaluation* seeds (1701 and 2701), we
compared the same KS150 spring-only physical robot (`h40-spring-only`,
model SHA256 `21b5856b7a248e5d5b988e2735255d57ec79bae1a7cab0bcb72f481bfc36f470`)
using its independently trained 120M source and then its V1 150M/160M
warm continuations. The physical test was the v5 exact signed 4–80mm,
50 trials/height/direction, 350-frame first-life prism matrix; short
six-specialty and eight-world 60s/30s deep exams also matched. [V1 frozen
evidence/qualification](https://github.com/mdt733/emu/pull/101) and
[V2 seed-2701 replication](https://github.com/mdt733/emu/pull/102)
preserve complete source and examiner fingerprints.

| Seed2701, physical trials | KS150 120M | KS150 150M | KS150 160M |
|---|---:|---:|---:|
| Uphill 4–40mm: controlled traversals /500 | 120 | **236** | 151 |
| Uphill 4–40mm: full first-life survival /500 | 485 | **488** | 471 |
| Uphill 44–56mm: traversals /200 | 0 | 19 | 21 |
| Uphill 44–56mm: first-life survival /200 | **196** | 104 | 91 |
| Downhill 44–80mm: full first-life survival /500 | 131 | **377** | 359 |

Seed1701 showed the same **directional trade-off**: 150M uphill
4–40mm crossings 245/500 versus 120M 111/500, with survival 490/500
for both. At 44–56mm uphill, 150M survived only 107/200 versus the
120M 196/200; 160M was worse (89/200). Good stair *traversal* must be
distinguished from upright survival after the attempt and from strict
world-XY precision parking. H40 has velocity feedback, not global-XY
station-keeping feedback; the 60s zero-command test had 8/8 dynamic
balance for all and unwanted mean horizontal speeds around 3–4cm/s.

**Teacher-design lessons:** Keep the one learned 360-input actuator policy
and the measured Spring-only tendon physics. A cold foundation, gradual
Academy births, authentic supported two-foot transfers and continued
flat/command/slope/recovery rehearsal improve trainability; but an
uphill teacher which advances solely on transfers can buy more crossings
at the price of dangerously many falls. Uphill promotion therefore needs
**completed first-life survival and supported crossing from the SAME
physical height cohort**, with evidence windows saved in checkpoints.
Downhill advancement is independent. The **nominal ordinary-world**
signed frontier is capped at 40mm; the procedural generator deliberately
offers some later lip challenges above the nominal height. Academy up
and down frontiers are independently allowed to earn a nominal 55mm.
The teacher uses measured physical **first-birth** height and does not
mix a cohort from an older frontier with a newly graduated height.

The actor has no global-XY position feedback: neither precision parking
nor static world-coordinate station keeping is a graduation condition.
The teacher may retain bounded forward-progress shaping, but **strict
world-XY parked-arrival reward is disabled in the school**. Its actual
graduation evidence is full first-life survival with **both feet
independently supported on the first assigned destination tile**, not
a single upper foothold or merely crossing a world coordinate.

**The current hypothesis is ONE consolidated cold study**, not a
150M/170M hand-selected student's continuation. The general training
tool still supports explicitly authenticated research warm starts,
including frozen source-pinned milestone checkpoints; this study
simply selects step zero to measure teaching from birth.
See the single active
`scripts/run_h40_school_wsl.sh` runner on the cold-school research branch:
0→180M PPO steps on exactly one source/config, no external warm parent
and no teacher change at 20M intervals; then repeat whole cold runs with
other training seeds using that SAME source and budget. Age-dependent
lesson scheduling inside the saved governor is automatic and allowed.
One initial seed-735 cold study and one revised seed-736 cold study
have now completed, **on different source revisions**, and were evaluated
with different independent exam seeds. This is evidence of real cold
execution and promising small-step transfer, **not** a matched-seed
causal test or a multi-seed qualification. Compare completed cold-seed
cohorts, including failures and the worst learner, against equal-budget
baseline-teacher cold runs before claiming the education method is
superior. No automatic champion promotion.

## Revised cold-school qualification — seed 736, 23 November 2026

The operator-supplied `school-evidence(1).zip` verifies a single cold
0→120M student, source commit `4048e8450c36882501cbb624f4ca656e11da9419`,
training seed **736**, **120,061,952** environment steps, 2,048 MJX
worlds, no warm-start ancestor. Profile `h40-spring-only` uses the
original 400 mm H40 morphology and six-motor/360-input interface with
KS150 knee support; its generated XML SHA256 is
`21b5856b7a248e5d5b988e2735255d57ec79bae1a7cab0bcb72f481bfc36f470`.
Held-out first-life real-prism stair exam **5701** used 50 trials per
4 mm level and direction, 350 control frames and 2,000 assigned worlds.

| Signed real-prism traversal | Initial cold seed 735 (exam 4701) | Revised cold seed 736 (exam 5701) |
| --- | ---: | ---: |
| Uphill +4–40 mm, five-frame traversal | 39/500 | **64/500** |
| Uphill +44–80 mm, five-frame traversal | 0/500 | **0/500** |
| Downhill −4–40 mm, five-frame traversal | 138/500 | **132/500** |
| Downhill −44–80 mm, five-frame traversal | 71/500 | **84/500** |

The revised uphill Academy frontier finished at **3.5 mm** and
downhill at **55 mm**. The recorded `academy_short_climb_successes`
counter was always zero **by configuration**:
`science_step_finish_on_transfer=false` explicitly disables that
short-completion event in the survival school. Zero here is **not**
evidence that no partial climbs happened. Short command pulses passed **360/384**,
recovery pulses **218/256**; 30–60s deep survival was **72/72**
across nine 8-world scenarios, but 60s stationary position control
was **0/8**, and several long command missions were **0/8** despite
survival. The apparent small-step gain is not evidence of solved tall
uphill climbing. Student source revisions, training seeds, and exam
seeds differ, so this is **descriptive**, not a controlled proof of
teacher improvement. The released `main` actor uses different knee
physics and has not been ranked against this student on an identical
authenticated mission protocol.

Preserve the complete `final.pkl` and exported `.emupolicy` separately:
the submitted evidence ZIP contains provenance/results/exams but not
the bytes needed for exact checkpoint replay. This candidate is not
approved to replace the immutable `main` actor or its public release.


### Seed-736 first-lip funnel autopsy — source and evidence, not a new policy

Post-hoc breakdown of the *same* independently frozen seed-5701 exam,
350 frames at a fixed +0.25 m/s command:

| Positive real lip | Leading top support | Both feet supported for five frames | Controlled five-frame traversal |
| --- | ---: | ---: | ---: |
| +4 to +28 mm (350 worlds) | **350** | **184** | **64** |
| +32 to +80 mm (650 worlds) | **1** | **1** | **0** |

On small steps, *first* foothold acquisition is not the observed
bottleneck; bilateral transfer and controlled continuation are. On tall
steps, a distinct first-foot acquisition cliff starts near +32 mm.
The 350-frame exam is **not** a full Academy lesson: it has a different
command and duration from the training cohort.

To separate raw lip-count events from the governor's actual safe-ascent
evidence, the **184 periodically saved metrics rows** contain these
*sampled-rollout* event counts (not full 120M totals or a single closed
24-birth governor window):

| Evidence source / gate | Observed sampled count |
| --- | ---: |
| Raw near-height uphill lip encounters / successes | **1,356 / 1,294** |
| First-life near-height uphill births | **254** |
| Those births that survived the complete course | **187 / 254** |
| Those births with verified assigned two-foot transfer | **107 / 254** |
| Those births with both transfer and full-life survival | **106 / 254** |

The last distinction matters: 106 of the 107 observed transferred births
survived. In this limited log sample, **failure to reach two-foot transfer**
and falls **before** transfer appear more important than late post-transfer
falls. The governor rightly does not treat 1,294 raw per-lip crossings as
1,294 independent safe completed lessons. Its promotion requires at least
24 same-frontier births, >=55% *same-life safe* transfers and >=90%
survival; <85% near-frontier survival can cause retreat. These sampled
logs are not a substitute for its actual checkpointed decision windows.

**Specific untested hypothesis:** the nursery teaches around **0.10
m/s** requested forward command for **1,200 frames**; the independent
stair exam uses **0.25 m/s** for **350 frames**. Faster approach or a
shorter exposure may improve observed crossing without teaching durable
course survival. Compare the same frozen seed-736 actor with the same
KS150 model, assigned first-lip geometry, matched seeds and both
command/horizon variants as a **diagnostic**. Report first-top,
bilateral contact, full-life falls and controlled continuation
separately. This is not a qualifying holdout once the design is chosen
from seed-5701 evidence. Do not weaken the governor or stack a new
reward before that experiment.

Reproduce the static logged-rollout-versus-exam funnel without loading
checkpoints or doing PPO:
`python -m emu.results --school-audit ~/emu-data/runs/h40-school-cold-120m-s736`
or pass the saved `school-evidence.zip`. The report labels its sparse
rollout counts explicitly; it is descriptive and cannot rank policies.
