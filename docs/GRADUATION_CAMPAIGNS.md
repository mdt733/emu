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

## Proposed recovery-first research programme

**Updated decision:** Treat the apparent velocity-versus-resilience
tradeoff as a *hypothesis*: test the same low-speed recovery cases
across checkpoints before calling it forgetting. Preserve qualified
easy/medium dynamic walking and recovery with ongoing rehearsal while
expanding the harder speed/step/slope/force frontier independently.
Read the explicit [P0–P3 controlled research plan](RECOVERY_FIRST_GROWTH.md):
three matched post-foundation mature/retention trials, three
coherent-terrain/force-isolation trials, an estimated 140M total
world steps including shared 20M cold birth, then integrate and
consider a 400M fresh cold-start only after physical gates pass.
This is a plan, not training code or a launched run.

## Root-cause comparison with protected main

**New conclusion:** See [source-versus-telemetry root-cause analysis](MAIN_VS_V5025_ROOT_CAUSES.md).
Main's mature speed/phase/rehearsal training was not carried into the
continuous 360 run, STEP/SLOPE assigned geometry without suitable
goal-conditioned task/command coherence, step progress reward was
often inaccessible at frontier vx <0.08m/s, terrain grading repeatedly
eliminated practice, and 4N push target saturated very early.
Exact causal attribution still needs isolated matched interventions.
These findings supersede the notion that a higher promotion factor
alone would fix the robot.

## Experiment 1 — completed 290M-screen outcome

**Completed at source `ab29c027`:** common 50M foundation + 12 matched
50→70M forks. All finished; see the [result audit](EXPERIMENT1_FINDINGS.md).
This replaces the earlier provisional claim that the experiment had not
run. Best fixed generalist exam = staged OFF **0.7343**; balanced
400ms/2× integral **0.7231**, versus same-lesson OFF **0.7139**.
**All 12 finished with 1.5mm step-height practice; most slope practice
fell back to 0.35°; 4N push and fast yaw frontier were not independently
certified capabilities.** The 400M cold graduation run is NOT authorised
by this evidence. Experiment 2 begins with equal-physics frozen exams,
then isolated terrain/command/resilience/control repairs.

## Experiment 1 — broad research tournament (supersedes A2/C2 plan)

The narrow A2/C2 160M repetition is **superseded by owner direction**.
Instead, train **one learner to 50M** (20M proven steady-first foundation and
30M of adaptive six-specialist growth), then branch its 50M checkpoint
into **twelve matched 20M continuations (50→70M)**.
This totals **290M environment steps**. This is a
controlled short-horizon exploration, **not** twelve independent cold starts
or a proof of seed robustness. A and C's 60M results motivate the arms.

| Fork | Teaching | Velocity-I ramp | Ceiling |
|---|---|---:|---:|
| 1 balanced OFF | balanced | off | 1× |
| 2 steering OFF | steering | off | 1× |
| 3 terrain OFF | terrain-early | off | 1× |
| 4 staged OFF | staged | off | 1× |
| 5/6/7 balanced I | balanced | 0.2/0.4/0.8s | 1.5× |
| 8/9/10 balanced I | balanced | 0.2/0.4/0.8s | 2× |
| 11 steering I | steering | 0.4s | 2× |
| 12 terrain I | terrain-early | 0.4s | 2× |

The **user's 400ms / 2×** proposal is a real arm, not merely an
offline notebook example. The source implements sensor-gated velocity-I
in **live training**; teacher tracking/error stays referenced to raw
operator velocity, while the actor receives its corrected internal
command. Assisted slope stability and operator-command accuracy are
different measurements. A multiplier cannot correct zero-vy lateral
camber drift (0×2=0), so examine it honestly and consider a separate
additive offset in Experiment 2 if evidence demands it.

The original `d26ae0c2` 60M checkpoint belongs to a different source
revision: **do not warm-start it** under changed controller code. The
shared 50M learner is trained on the exact experiment SHA, with
20M steady-first foundation + 30M of full multi-dimensional practice,
and is the valid common checkpoint ancestor for all forks. Each fork
inherits the same source/actor/optimizer/normaliser/curriculum, with its
own post-fork training configuration, output directory and fresh physics
episodes. Source, lineage, seed and controller settings are recorded.
Research forks are now first-class: `--resume /path/to/checkpoints/final.pkl`
into a **new run directory** is permitted after validating checkpoint,
source SHA and 360 actor/sensor/command semantics; changed task/physics
distribution knobs remain legal, explicitly reported experiments; no
arbitrary parent age, seed or branch label restrictions. Same-run
`--resume auto` remains strict.

**Implemented launcher:** `scripts/v5025_screen.py`, **plan-only**
unless `--run`; source must be clean on
`feature/v5024-release-polish`. Fresh checkout/virtualenv, then:

```bash
python scripts/v5025_screen.py          # plan
python -u scripts/v5025_screen.py --run # real GPU
```

Outputs go to `~/robot_lab/emu_runs/v5025_aggressive50_seed734/`;
`overnight_results.zip` holds comparisons/console references, not
the large checkpoints themselves. First inspect code/CI and GPU smoke.
**The source above is an implementation candidate**, not evidence
that 290M CUDA steps have actually run successfully.

**Shortlist rule:** keep 2–4 genuinely different Pareto candidates.
Compare physically observed command tracking (especially uphill and
camber), genuine support/step ascent, recovery, falls, stop/reverse
transitions and shaft work per useful distance. Controller ON must
not win by redefining the requested velocity; compare its actual
policy demand and saturation against raw operator demand. If a fork
only raises training reward or a teacher frontier, do not promote it.

### New accelerated six-capability curriculum — restart experiment

At ~32M the previous learner had ≤0.08m/s certified vx despite very
low global falls. The near-edge evidence gate, not demonstrated top
speed, was the likely limiting factor.

On the new SHA and entirely fresh root, the first **20M retains the
proven main-style gait curriculum**. Afterward the governor expands
practice aggressively on *completed physical evidence*, and retreats
whenever easier challenges start failing:

- vx+/vx−/vy±/yaw±: minimum 4 near-edge completions (was 8),
  ≥60% success (was 72%), +15% per promotion (was +2.5%).
  Mixed commands +10%; more pure-axis commands and extra forward
  samples. Easy-command failures retreat 12%.
- Four signed incline directions: ≥4 near-grade completions each,
  ≥65% supported stability, +max(0.25°,16%) per promotion (was
  +0.08°). Easy-slope failure retreats 12%; stable support is
  **not** automatically accurate velocity tracking.
- Step up AND down: ≥4 near-height real supported crossings each,
  ≥55% success, +max(1mm,18%) (was +0.25mm). Easy failures retreat
  18%. Real collidable geometry has **8cm** experiment headroom;
  this is not a guaranteed achieved height.
- Resilience push growth +8% per qualified update rather than 0.15%,
  stronger retreat on persistent failure; practice headroom 4N.
- Mixed requests, transition timing, sensor disturbance and four
  conventional difficulty scalars now push faster after 20M.
  Core retreat, KL/recovery guards and per-specialty grading remain.
- Control/efficiency remains evaluated on moving shaft J/m, not idle.

This is an *experiment*, not proven to halve learning time. New source
and robot-world setup require cold-start training and a separate
results root `v5025_aggressive50_seed734`. Do not combine the
previous ~32M checkpoint with this changed source.

## Experiment 2 — one controlled terrain-control intervention

**Question:** which Experiment-1 gains survive integration and longer
training—controller parameters, teaching allocation or terrain-specific
practice—without sacrificing normal control?

**Design is intentionally provisional until Experiment 1's physical matrix.**
If velocity-I wins the screen, keep it ON from the selected training
boundary and evaluate using the saved controller parameters. Its slope
tracking pass/fail is judged against raw operator vx/vy, not the
boosted policy command. The frozen examiner now reads controller metadata
from exported policies. The public browser currently does NOT implement
velocity-I; publishing assisted actors is deliberately blocked until
browser/simulation parity is built and examined.
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
- [x] Add `--arms A C` to the existing guarded campaign launcher.
      It remains plan-only unless `--run` is explicitly provided.
- [x] Complete established 50M + 12 × 20M Experiment 1 at seed 734;
      all 13 training records verified from supplied overnight ZIP;
      document limitation and rankings in `docs/EXPERIMENT1_FINDINGS.md`.
- [ ] Inspect physical failures; implement **one opt-in** Experiment 2
      teaching change and test OFF vs ON, same source and seed.
- [ ] Freeze a single configuration for the 400M graduation run,
      with no midstream code/source/config drift.
- [ ] Final common benchmark against published Pinnacle; only then
      decide on `main` promotion and a separately authorised Pages release.

**Rule:** Keep the 232-input compatibility code long enough to evaluate
Pinnacle fairly. Delete it **after** we have switched the reviewed
public benchmark, not before the only trustworthy comparison.
