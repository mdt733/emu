# EMU graduation: two experiments, one continuous training run

**Status:** design proposal, **not** a launched training run or a implemented
cross-schema benchmarking command. Updated 27 October 2026.  
**Owner's objective:** replace the published 260M, 232-input Pinnacle with
one clean 360-input array-only unified EMU *if physical evidence warrants it*.  
**Engineering branch:** `feature/v5024-release-polish` (draft PR #15).
The previous GPU experiment is frozen at `d26ae0c2c4097d7c7124759197a94504abc44995`.
**Never change that source to resume its checkpoints.** Each future campaign
must pin its own clean source SHA, full settings, Python/JAX versions, seeds
and run roots, and save its reports outside Git.

## What the completed v5024 study tells us

Source: owner's `overnight_results(1).zip` from the exact three-arm
v5024 experiment. All three ran to **60,030,976** world steps, produced
final checkpoints, and show zero PPO rollbacks and successful CPU/CUDA
preflight. These are *frozen generalist* results, not a physical terrain
matrix or Pinnacle cross-schema comparison.

| 60M measurement | A: main+balanced | B: command-heavy+balanced | C: command-heavy+steering |
| --- | ---: | ---: | ---: |
| Frozen broad score | 0.642 | 0.594 | **0.666** |
| Core survival | 100% | 100% | 100% |
| Dynamic80 survival | **79.7%** | 70.3% | 78.1% |
| Resilience75 survival | 76.6% | 70.3% | **79.7%** |
| Growth110 survival | **57.8%** | 39.1% | 48.4% |
| Growth110 yaw MAE (rad/s, lower better) | 0.548 | 0.520 | **0.376** |
| Last taught slope angle | 0.92° | 0.87° | 0.92° |
| Last taught step height | 0.34 cm | 0.30 cm | 0.25 cm |

A and C are the two candidates worth repeating. C offers stronger
steering/generalist score, A harder-command survival and some step height.
A versus C differs in **two** curriculum settings (foundation and
subsequent emphasis): it compares complete recipes, **not an isolated
causal effect**. The earlier B arm isolated the separate components for
seed 733 only. Slope frontiers below 1° and small step frontiers are
practice limits, not certified incline/step competence.

### Gate 0 — read-only baseline exams before spending more GPU time

Export the already saved A, B and C final `.emupolicy` files using the
**original exact-source** v5024 installation. Do not rewrite checkpoints.
Run `emu capability-exam` **both** ordinary and
`--terrain-matrix` on each 360 policy: same 8+ worlds, horizon, speed,
seed and explicit output directory per actor. Check 1/2/3/5° four
directions and 0.3/0.5/1/2cm step up/down.

Record uphill **signed** achieved vx, downhill/lateral drift, yaw
tracking, supported moving stability, foot-supported step crossing,
settled fall rate, servo shaft work per useful metre, and confidence
from *completed* attempts. If examination compilation/geometry or
export fails, fix it on a new review branch before GPU replication.

**Pinnacle does not belong in this 360-only exam**: its legitimate
232-input tripod/flow observation contract requires a common physical
benchmark harness (specified below). Do not compare numbers from
two incompatible exam implementations.

## Experiment 1 — repeat the two viable v5024 recipes

**Question:** is C's steering benefit repeatable, and does either recipe
keep improving beyond 60M without losing supported locomotion?

- **Source:** release-polish learning implementation, unchanged actor,
  PPO, physical surfaces, learning rewards and evidence rules. Pin a
  clean new experiment SHA before starting; no train-time code edits.
- **Two independent COLD starts**, sequential CUDA, 2,048 worlds,
  seed **734**, **80M world steps per arm = 160M total**.
- **A2:** first 10M `--foundation-mix main`, then
  `--teaching-schedule balanced`.
- **C2:** first 10M `--foundation-mix command-heavy`, then
  `--teaching-schedule steering`.
- Identical robot/sensor contract 360 array-only, original gait
  payments, PPO hyperparameters, source SHA, random seed,
  4 cm terrain ceiling, eval definitions and saved checkpoints.
- Frozen exams at **20M, 40M, 60M and 80M** (10M cadence acceptable
  if current runner retains it). The 60M point compares with seed 733.
  An 80M checkpoint tests whether late gains are stable.
- No source-crossing warm start, inherited A/C checkpoint, automatic
  training-arm champion selection, or GPU experiments sharing one run
  directory. Use a fresh `v5025_repeat_seed734` data root.

**Decision at the gate:** choose the *recipe*, not the highest training
reward. Prioritise held-out flat/command tracking, hard survival,
recovery, four-way incline control and actual step-up/down crossings;
work per useful distance is a secondary constraint. Compare seed
733 and seed 734 **at the same 60M age**. Treat a seed-specific reversal
as uncertainty, not a reason to cherry-pick. If neither is an obvious
all-round winner, keep both as a Pareto shortlist but choose one
conservative baseline for Experiment 2.

Implementation note: the existing polish `scripts/v5024_campaign.py`
still launches THREE arms; **it does not yet implement a two-arm filter**.
Before Run 1, implement and test an explicit two-arm runner or launch
two separately pinned `emu train` commands. The documented campaign
is *not* presently a runnable one-command 160M launcher.

## Experiment 2 — one controlled terrain-control intervention

**Question:** can we convert stable slope contact into *commanded uphill
motion*, and improve supported step-ups, without regressing flat walking?

**Design is intentionally provisional until Experiment 1's physical matrix.**
Choose the dominant weakness and implement **one** teaching change, not
a basket of unrelated PPO, reward, sensor and architecture modifications.

**Preferred intervention P — gravity-resisting command practice**

In slope-focused worlds, deliberately rehearse feasible signed
forward/sideways/yaw commands in each of four incline directions.
Keep near/easy/stretch slope exposure, but record signed actual progress
against gravity and lateral/downhill drift separately from no-fall
stability. Use the existing shared actor, observations, physical contacts
and base rewards. The independent physical command-tracking metric
must remain strict; simply remaining upright cannot win the lesson.

**Alternative Q — useful supported step-up practice**, only if the
Gate-0 exams show steps are the more urgent bottleneck: increase early
real-contact opportunities to approach, plant a foot on the upper
surface, transfer load and fully cross. Reward truthful supported
crossing/progress, never teleport/root-past-edge, tiny nonfunctional
foot lift or stationary pose. Preserve step-down competence.

Run P **or** Q, not both together without an additional isolating arm.
No privileged teacher ground truth is ever appended to the policy's
360 observations; MJX can use simulator state only to evaluate/reward
actual outcomes.

- A **control arm** uses Experiment 1's selected winning recipe with the
  teaching intervention OFF.
- A **treatment arm** uses the *identical* source SHA, seed **735**,
  cold start and learner, but the one explicitly opt-in teaching
  intervention ON from the **appropriate curriculum phase**. Record
  flag and physical meaning in saved training config/checkpoint metadata.
- 2,048 CUDA worlds; **80M per arm = 160M** total, sequential, in
  fresh `v5026_controlled_terrain_seed735` directories.
- Same 10M foundation selection as the winning recipe. Ensure OFF
  branch is behaviourally unchanged, and that no new cold-start
  penalty destabilises the first 70 original steady-only updates.
- Frozen common physical cases every 20M and terrain matrix at
  40/60/80M, plus hard commands/resilience and true step up AND down.
- Compare treatment vs control at identical age/seed/source and against
  the matching seed-733/734 recipe results. Repeatability of the
  treatment beyond seed 735 would still be unproven at this point.

**Go/no-go interpretation:** prefer treatment only for meaningful
repeatable directional tracking or supported crossing gains **without**
clear regression in flat supported gait, reverse/lateral/yaw, recovery or
work per useful distance. As indicative review tolerances (not a
mathematical "passed" label), flag a >5 percentage point loss in
basic/hard survival, >10% worse ordinary command MAE or a substantial
increase in shaft J/m for similar achieved motion. If there is little
physical improvement or the effect is noisy, choose the **control**;
do not force a new idea into the final run.

## Run 3 — one clean, continuous graduation learner

**Goal:** discover how far ONE selected design grows when allowed to
train continuously, not search across an ever-growing branch tournament.

- Freeze the chosen configuration, core learner and safety/evidence
  checks on a clean versioned release-candidate commit.
- Fresh seed **736** and **fresh actor from random initialisation**;
  never transplant a 232/360 checkpoint or a treatment checkpoint
  from a different source revision.
- 2,048 worlds, CUDA, **400M cumulative environment steps** in a
  single source-pinned run root. The 10M or 20M foundation length and
  subsequent schedule are chosen based on Experiments 1 and 2, not
  an artificial arbitrary rule applied mid-run.
- One continuous shared policy; independent dimension-specific
  easy/near/stretch lessons and retreat/advance remain active throughout.
  Do not impose new time-gated challenge unlocks or end training merely
  because one specialist stalls.
- **Observe but don't interfere:** immutable 20M cadence generalist
  exams, rotating 2M snapshots, best/last checkpoints, metrics,
  source SHA, runtime manifest, output bundle and read-only physical
  exams at useful ages (e.g. 60, 120, 200, 300, 400M).
- On bounded numerical failure or corrupt policy checkpoint, halt
  cleanly. Short-lived fall-rate/reward regressions or slope-frontier
  retreats alone are **not** reasons to abort or change teaching.
- Record both **best independently examined checkpoint** and the
  final 400M policy; avoid choosing based only on training reward.
  If independently measured progress continues, consider a
  **same-source, same-config target extension** toward 600M after
  evaluating 400M. That's an extension of this one learner, not a
  fourth experiment or a rewritten checkpoint lineage.

The long run is **not started by this plan**. Its config and source SHA
must be frozen after Experiment 2 and CPU/2048-world CUDA acceptance.

## Fair Pinnacle vs new EMU: common physical benchmark v1

This is a **required new implementation**, not an existing CLI flag.
Current `capability-exam` is 360-only; the legacy `directional-exam`
and `terrain-exam` use different case sets and scoring. Their scores
must NOT be compared as if they were the same exam.

Proposed read-only entrypoint:
`emu benchmark --reference showcase/champion.emupolicy --candidate
/path/to/final360.emupolicy --protocol graduation_v1 --output NEW_DIR`.

### Same world, separate legitimate sensors

- Verify model/actuator/XML contract equivalence, six action order,
  control dt, motor limits, spring dynamics, collision geometry and
  physical parameters. Record hashes and **fail**, rather than silently
  claiming fairness, if physically meaningful robot/engine settings
  differ. If model semantics are not equivalent, fix the exam setup
  explicitly or report the difference as a limitation.
- Load independently approved/validated exported actors: Pinnacle
  **schema 2, 232 obs, 260,112,384 trained steps** and the
  candidate **schema 3, 360 obs, array-only**. Each gets its own
  saved observation normaliser, real sensor latency/noise/extrinsics
  and its own inference path. **No extra privileged information**,
  external heading assist or teacher signals.
- Same exogenous physical tasks, starting poses, seeds, requested
  body-frame velocity trajectory, reset rule, disturbance direction
  and contact surfaces. Pair trials by scenario+seed and stop
  scoring at the first physical reset. No reward/curriculum influence.
- Use ONE shared policy-agnostic trace scorer for all actors:
  survival and first fall, settled signed achieved vx/vy/yaw vs command,
  uphill progress, lateral/downhill drift, pitch/height, supported
  far-side step-up/down, push recovery, slips/action jerk and simulated
  servo **shaft** work per useful completed distance.
- Guard against gaming: surviving but stationary is not a successful
  walking test; a fall is not rescued by subsequent reset progress;
  always report "not attempted / unsupported / unknown" distinctly.
- Run matched flat/forward/reverse/strafe/yaw, combined commands,
  starts/stops, recovery pushes, four grade directions at
  **1/2/3/5°** (optionally 8° as stretch), real **0.3/0.5/1/2cm**
  up/down steps and incline+step transitions. Both actors see the
  same terrain, not one actor's historical course.
- Screening: ~8 worlds × 2 held-out seeds per case. Final review:
  more seeds/trials (e.g. 4 seeds × 8 worlds = 32 per case) and
  paired bootstrap intervals or uncertainty bands. Report actual
  world counts; avoid overclaiming from tiny samples.
- Save per-policy per-case raw physical evidence, final summary CSV,
  paired difference CSV, policy SHA256s, physics/exam source SHA,
  seed/case protocol and readable Markdown. Produce a compact
  *Pareto view*, not a single flattering weighted reward.

### Graduation / release decision

A candidate should at least retain Pinnacle's **ordinary walking,
tracking and robust survival** within defensible sampling uncertainty,
and demonstrate **material physical gains** in the goals that motivated
the replacement: uphill/camber command tracking, genuinely supported
step crossings, combined commands and/or recovery, without
unacceptable efficiency or falls. Stronger performance on one axis
does not erase disastrous regression elsewhere.

Report BOTH:
1. **Absolute deployment competence** vs the published 260M Pinnacle;
2. **Learning efficiency** among new 360 policies at matched 60/80M
   and long-run checkpoints. Do not compare training rewards from
   different curricula or call a 60M score directly comparable
   to Pinnacle's historical exam.

If the 400M candidate loses the common benchmark, keep Pinnacle
published and improve the candidate in another explicitly scoped
experiment. No promise of promotion after an arbitrary run count.
If it wins, get explicit owner review for `main` and **separate**
public-release approval. Then archive Pinnacle and prune temporary
220/232 input inference/export/training/test compatibility.

## Engineering order of work

- [ ] Export and freeze A/B/C 60M policy hashes and read-only terrain exams.
- [ ] Design/implement common policy-neutral benchmark and prove
      Pinnacle-vs-Pinnacle identity (same scenario+seed gives same scores).
- [ ] Verify the 232- and 360-input policies run under identical physical
      model/control settings with only observation paths differing.
- [ ] Add a two-arm *plan-only by default* source-pinned replication
      launcher (or audited two-command runner) without adding another
      general training framework.
- [ ] Complete Experiment 1 at seed 734; select one baseline recipe.
- [ ] Inspect physical failures; implement **one opt-in** Experiment 2
      teaching change and test OFF vs ON, same source and seed.
- [ ] Freeze a single configuration for the 400M graduation run,
      with no midstream code/source/config drift.
- [ ] Final common benchmark against published Pinnacle; only then
      decide on `main` promotion and a separately authorised Pages release.

**Rule:** Keep the 232-input compatibility code long enough to evaluate
Pinnacle fairly. Delete it **after** we have switched the reviewed
public benchmark, not before the only trustworthy comparison.
