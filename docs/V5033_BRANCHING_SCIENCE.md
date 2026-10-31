# v5033 — 24-arm branching science before the next cold Pinnacle

**Status: experiment design, NOT IMPLEMENTED, NOT LAUNCHED.**
One temporary research branch `research/v5033-24-arm-science`; one
shared source commit after implementation, **24 individually labelled
training cases** under an external results root. No 24 Git branches.
Preserve frozen `main` and the v5032 260M source/checkpoint untouched.

## What actually failed in the 260M physical tests?

The completed v5032 260M research actor (`terrain_360`) was trained
with `temporal_vx_reward_share=0.5`, 25% fast-flat rehearsal and
**`verified_ascent_reallocation=0.0`**, not the launcher/manifest's
requested 3.0: `emu/run.py` omitted the CLI-to-config forwarding.
Fix this first and test the **resolved** training config, actual
`EmuMJXEnv` attributes, and checkpoint's saved config. A CLI echo
or separate GPU smoke is not adequate proof of arm assignment.

The independent slope matrix uses real constant-normal colliding
slabs, 16 worlds, **0.15 m/s** body-forward command, zero lateral/yaw,
**350 frames (7 s)** and no reward/curriculum updates. The
read-only examiner requires support, useful movement, bounded
forward/lateral error, and for a straight course mean signed
yaw-rate bias ≤0.08 rad/s and yaw jitter ≤0.20 rad/s. The
historical *strict* all-axis score additionally gates yaw MAE;
do not confuse these distinct scores.

| Downhill | Survived | Supported stable movement | Movement command pass | Heading pass = mission |
|---|---:|---:|---:|---:|
| 5° | 16/16 | 16/16 | 16/16 | 16/16 |
| 8° | 16/16 | 16/16 | 16/16 | 13/16 |
| 12° | 16/16 | 16/16 | 16/16 | **4/16** |
| 14.9° | 16/16 | 16/16 | 15/16 | **2/16** |

At 12° the walker averages **0.159 m/s** on a 0.15 request;
mean signed yaw-rate bias is **−0.099 rad/s**. It is **not**
falling off hills or failing to move; this is chiefly downhill
directional control. At 14.9° mean forward is ~0.180 m/s and
yaw jitter increases as well. Rhythmic leg-driven yaw oscillation
is not the same as sustained unwanted turning.

Therefore before any training:
- Add a **read-only downhill speed × grade sweep** using the same
  robot/policy/physics at speeds 0.05, 0.10, 0.15, 0.20, 0.30 m/s
  and grades 5°, 8°, 12°, 14.9°. Keep heading bias, gait yaw
  jitter, support, slip, actual vs requested speed and travel.
- Distinguish a commanded slow pace from a policy silently refusing
  to move. An intentionally slower **external** safe-speed governor
  is acceptable only if evaluated as its own explicitly named
  controller; do not give the autonomous policy free credit for
  stopping or ignoring 0.15 m/s.
- Separate absolute heading/path drift from mean signed yaw-rate
  and gait-period jitter. Keep the strict and mission scores.

## Fixed research ancestor and protection

**Ancestor:** immutable completed
`~/robot_lab/emu_runs/v5032_pinnacle_s761_v2/pinnacle_cold260/checkpoints/final.pkl`,
source SHA `9b11bbe596aea7201d29a6beb2d192922f8e2e9a`,
checkpoint SHA-256 recorded in report:
`238b49134a055c70e483fe993acdd86b2d1b5994cc14ef186fd6b05329e94f0b`,
cumulative step **260,046,848**. The exported `.emupolicy` is
for inference only; forks need full `.pkl` optimizer/teacher state.
Verify the actual checkpoint bytes and metadata on GPU before
starting. The user's ZIP contains a hash but **not** the `.pkl`.

Frozen robot XML, 360-input/4×4 sensor revision, six actions,
PPO structure, contact geometry, and source checkpoint for EVERY
arm. Use the existing **explicit cross-source research fork**
with both ancestor source SHA and checkpoint SHA-256. Never
alter the parent. Every arm trains in its own directory.
Each arm must advertise **one named intervention**, with no
default-value leakage and all other fields controlled.

## The proposed 24 distinct arms

All experimental arms inherit the exactly same v5032-final
`260,046,848` step full learner state. Arm 00 is an otherwise
unchanged continuation. Interventions marked **native** already
have a related user-facing knob; they still need CLI-to-config
end-to-end verification. Interventions marked **new** require a
small, explicit, tested teacher/reward implementation. All must
pass reward math, real-contact behavior and schema continuity
checks before 2,048-world CUDA.

| ID | Area | SINGLE treatment / comparison | Primary question | Implementation |
|---|---|---|---|---|
| **00** | Control | Exactly unchanged 260M policy + 8M continuation | What improvement occurs merely with more training? | native |
| **01** | Step frontier | Cap each evidence-earned **promotion** to +1 mm; preserve other logic | Does finer target growth help ≥3cm uphill? | new, curriculum |
| **02** | Step frontier | Cap routine **easy-failure retreat** to −1 mm instead of ×0.82 | Does less violent retreat improve frontier retention? | new, curriculum |
| **03** | Step evidence | Require 12 verified near attempts **per direction** rather than 4 | Is the frontier reacting to noisy tiny batches? | new, curriculum |
| **04** | Step reward | Actually enable verified upper-foot ascent reward reallocation **1.5** | Is mild support-transfer teaching enough? | native after wiring fix |
| **05** | Step reward | Actually enable verified ascent reward reallocation **3.0** | Does the intended v5032 feature resolve 2.5–3cm ascent? | native after wiring fix |
| **06** | Step reward | Reallocate **4.0** to verified upper support, total budget preserved | Is stronger contact evidence useful or too restrictive? | native after wiring fix |
| **07** | Step reward | With arm 05's 3.0 baseline, delay support-payment eligibility until a **3-frame consecutive true upper contact** | Is a persistent foothold better than a fleeting touch? | new; paired against 05 |
| **08** | Downhill pacing | Descending teacher requests limited to **0.10 m/s** on steep grades | Is careful pacing teachable without freezing motion? | new, teacher |
| **09** | Downhill exposure | Double **downhill** lesson fraction within slope worlds, no other reward changes | Is there simply too little signed descent practice? | new, teacher |
| **10** | Downhill steering | Small **25-frame signed yaw-bias** payment on downhill only; don't penalise stride jitter | Can we remove persistent left/right drift? | new, reward |
| **11** | Downhill braking | Extra mild downhill **overspeed** penalty, retaining distance-progress requirement | Can it control 14.9° speed without stalling? | new, reward |
| **12** | Steering | `--teaching-schedule steering` instead of balanced | Does more general command practice improve slope headings? | native |
| **13** | Speed rehearsal | Fast-flat rehearsals **40%** instead of 25% | Does more high-speed gait exposure fix >0.6m/s heading? | native |
| **14** | Speed rehearsal | Fast-flat rehearsals **10%** instead of 25% | Are speed lessons starving terrain/recovery? | native |
| **15** | Velocity feedback | Velocity integral trim ceiling **1.5×** instead of 1.0× (off) | Better command tracking or downhill overspeed? | native |
| **16** | Temporal reward | 25-frame speed reward share **0.25** instead of 0.5 | Is less slow-window bias better for precision? | native |
| **17** | Temporal reward | 25-frame speed reward share **0.80** instead of 0.5 | Is more averaged tracking better for fast gait? | native |
| **18** | Speed frontier | Raise upper commanded-vx training envelope from **0.60 to 0.80 m/s**, with same actual velocity gate | Does the actor lack exposure to sustained faster requests? | native/config |
| **19** | Transitions | Increase command-transition focus **10→20%**, reallocating from steady command worlds, all else matched | Does speed-changing/braking practice cure drift and control? | new, teacher |
| **20** | Resilience | `--force-program aggressive`, versus gentle 0.7 N cap | Can harder pushes preserve walking or do they damage gait? | native |
| **21** | Resilience | Recovery focus **20→30%**, reallocating from command worlds | Does more recovery practice improve robustness without speed loss? | new, teacher |
| **22** | Efficiency | Enable small bounded motor-cost reward **0.03**, measuring shaft J/m (NOT battery energy) | Can power-efficiency improve without slow/stalled cheating? | native, verify knob |
| **23** | Gait smoothness | Increase action-delta penalty **slightly** in control-focused worlds only, gated on actual walking | Can we reduce unhelpful servo thrashing without sacrificing steps? | new, reward |

**Note:** 07 deliberately nests one incremental change on 05; its
proper direct comparator is 05 rather than 00. All others compare
directly with 00, and some parameter families support dose-response
tests (04–06, 13–14, 16–17). Pre-register that nesting before runs.

The 24 are **not all implemented in current CLI**. A plan is NOT
an executable experiment. The validated final registry must
explicitly distinguish native switches from new code; it must
fail on missing/ignored settings.

## Two-stage overnight funnel

**Stage A — explore:** 24 arms × **8M additional world steps**,
each from the *same frozen 260M parent*, all at 2,048 envs, with
a unique result directory and a shared **single immutable science
source SHA**. Each targets ~268M *cumulative* steps, not “train
for 8M from zero.” Don't silently extend one arm's checkpoint
into the next. Only one CUDA trainer active at any instant.

At ~22k measured world steps/s, total training exposure of
~192M steps corresponds to ~2.4 h *pure rollout throughput*.
Compilation, 24 independent PPO startups, exams and CPU work
can be substantial; never promise an exact overnight ETA.

**Stage B — confirm:** shortlist about six arms on an immutable,
pre-defined diagnostic scorecard. Continue only those six by
**another 16M world steps** (to ~284M cumulative), retaining
each arm's exact new source and config, then repeat exams with
at least two fixed independent test seeds. Stage B adds ~96M
world steps (~1.2 h ideal throughput); total science compute
~288M world steps **plus** preflights, compilations and
independent evaluation. Do not pick a winner from a single
lucky 8-world binary result.

Avoid expensive full terrain matrix exams for all 24. Use a
**small, common frozen diagnostic panel** (same seeds and
commands) on all 24, then use full 16-world speed ladder,
four-bearing 5–15° incline and paired real step 1–3.5cm on
the six finalists and the unchanged control, with exact
saved JSON/CSV and actor/source SHA hashes.

**Key read-only diagnostic panel:**
- Downhill at 5/8/12/14.9° × five speeds 0.05–0.30, including
  0.15 as the reference; count stable, movement, mission,
  strict yaw, signed yaw bias and jitter separately.
- 2.0, 2.5, 3.0, 3.5cm up/down at 0.10–0.15, recording
  5-frame far-side support streak and remaining movement.
- Flat 0.3/0.6/0.8/1.0m/s *actual*, 700-frame sustained
  survival, signed heading and commanded-vs-achieved velocity.
- Up/down/cross slopes, recoveries by azimuth and cost J/m.
- Explicit uphill vs downhill tradeoff so easy downhill
  improvements cannot erase all supported uphill progress.

## Stop/go criteria and avoiding 24 false discoveries

Report **one fixed common scorecard** with (i) support-qualified
UP steps; (ii) downhill 12° and 14.9° movement plus yaw bias,
jitter and braking; (iii) flat 0.6/0.8/1.0 tracking, heading,
survival; (iv) held-out resilience and J/m; (v) regressions
against untouched control. Keep **strict** all-axis and
**mission** scores separately, and a min-across-capabilities
safety floor so one axis cannot buy the win by ruining another.

Example practical floors against frozen initial 260M:
- At least 7/8 reliable 2cm UP and no collapse at 0.3m/s flat.
- New slope controller never gets credit for parking or turning
  around; require supported nontrivial travel.
- 0.8m/s survival not markedly below 12/16 and no excessive
  1.0m/s regressions; important to inspect actual velocity and
  yaw bias rather than binary grades alone.
- Compare exact initial 260M policy, control continuation 00
  and same-age trained arms; preserve common 16-world seeds.
- Do not claim hardware power from simulated positive shaft J/m.

For candidate selection, prefer meaningful **paired** physical
differences repeated across seeds and a coherent mechanism;
do not rank 24 arms solely by a higher training reward or the
best of many noisy test scores. Document both negative results
and code-revision-specific sensor/lesson failure modes.

## Mandatory launch hygiene

1. Freeze research source SHA, verify clean Git and Python
   import location. Protect published main and old checkpoint.
2. Before **any** GPU run, validate exact parent SHA256,
   full training checkpoint step, schema-9/terrain_360 sensor
   contract, six outputs and compatible full learner state.
3. Require `--research-parent-source` + `--research-parent-sha256`
   on each new cross-source arm; do not silently trick
   `--resume auto` into accepting a different source.
4. Verify each arm's *resolved* `TrainingConfig` and in-environment
   values before PPO: e.g. arm 05 **must read 3.0**, arm 00
   **must read 0.0**, and zero treatment must preserve
   the baseline reward arithmetic. Test reward physics too.
5. Run bounded CPU contact/reward smoke for every distinct
   treatment, then real CUDA 2,048-world full-update gates
   for representative paths / all genuinely new JAX kernels.
6. Run each arm in its own directory, one GPU process,
   independent logs, hashes and associated read-only exams.
   Resume ONLY the exact same source/config and arm ID.
7. No automatic cherry-picking of “best” checkpoints from
   unequal ages or editing the control's run after seeing results.
8. Retain every arm and all negative evidence. Science branches
   are temporary; new main graduation should be a clean,
   tested cold start from random weights, not a merged optimizer.

## After the science run

Choose no more than 2–3 **non-interfering** improvements with
clear replicated physical evidence, verify their interaction
in a separate matched combination-vs-controls study, and
then build the next **cold-start Pinnacle**, aiming again
at 260M+ with updated strict source/recipe checks.
Organic terrain is still postponed; step-UP, downhill yaw,
and sustained speed are the target weaknesses. Do not merge
science weights or an untested reward mixture into main.

**Important scope boundary:** This document registers experiments
and evidence rules. It does not itself implement the 24 arm
controls, create 24 PPO runs, or validate any future result.
