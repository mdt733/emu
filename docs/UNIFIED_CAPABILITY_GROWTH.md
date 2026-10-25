# EMU — unified learning contract

**Detailed dimension inventory:** [CAPABILITY_DIMENSIONS.md](CAPABILITY_DIMENSIONS.md).

**Purpose:** one durable explanation of *what we are building and why*. For current progress and the next task, use [HANDOVER.md](../HANDOVER.md). Older experiments are preserved in [historical research](CAPABILITY_GROWTH.md) and Git history.

## The idea

**One robot. One brain. Many complementary lessons. Earn the next difficulty.**

Train one six-actuator biped from scratch with **one shared actor and PPO optimizer**, learning simultaneously in **2,048 MJX worlds**. The objective is not the highest scalar reward, speed, step height or slope. It is the *breadth and quality of reliable locomotion*: how many combinations of commands, terrain, disturbances and transitions the robot can handle **while walking with support, accuracy and reasonable mechanical effort**.

No fixed timetable automatically unlocks hard tasks. We maintain **evidence-backed capability frontiers**, sample near them, and expand, hold or retreat as real competence warrants. All worlds contribute to the same network every update.

## Five distinct responsibilities

```text
Physical world + robot  →  realistic observations  →  ONE actor
          ↑                                          ↓
   Task/curriculum teacher  ←  completed-episode evidence
          ↑                                          ↓
   Safety/learning governor       PPO + eight locomotion rewards
                             ↘
                    independent frozen-policy exam
```

1. **Physics** decides real contact, ground shape, support, displacement, step crossing and falls. Do not replace terrain contact with visual labels or tilted-gravity tricks.
2. **Actor** receives only physically plausible delayed measurements and user commands; it produces six actuator targets. No curriculum label, physical ground-truth normal or specialist ID leaks into its inputs.
3. **Reward/PPO** teaches useful behaviour *within each sampled task*. Keep the proven eight locomotion payments and one optimizer; reward is a teaching signal, never proof of mastery.
4. **Curriculum** chooses the next task distribution from actual completed-episode attempts, quality and severity. One governor owns decisions but **each specialty advances independently**, with its own closed physical evidence and capped step size. Preserve main's original four-regime budget; cap specialist families *separately*, so slope and step work never stop vx/vy/yaw command frontier growth. Retreatment is faster than growth.
5. **Examiner** independently measures frozen-policy physical competence on fixed, paired, held-out challenges. It reports strengths and failure modes; exam outcomes do not silently update the live training curriculum.

The browser viewer is an inference/debugging tool, not a representation of all GPU training worlds.

## Six simultaneous lesson emphases

These are **teacher-side emphases within one policy**, not separate brains or exclusive modes. The allocation is the *target*. The original 20M cold gait foundation has zero activated specialists. After 20M, activation blends linearly from zero to 35% over 10M steps, retaining a large ordinary-gait majority. **Only lesson frequency follows this blend**, never competence or permission to grow an individual target. Once active, each capability changes difficulty based solely on its own episodes.

| Emphasis | Target worlds | Harder lesson |
| --- | ---: | --- |
| Commands | 20% (~410) | Signed forward/reverse, lateral, yaw and mixtures |
| Resilience | 20% (~410) | Pushes, recovery and awkward starts |
| Continuous slopes | 20% (~410) | Actual uphill, downhill and both cross-slope directions |
| Discrete steps | 20% (~410) | Real step up/down, rotated lips and supported landings |
| Transitions/posture | 10% (~204) | Start/stop, acceleration, reversals and chassis attitude |
| Control/efficiency | 10% (~204) | Smooth, economical **moving** gait on easier supported tasks |

Every specialist still encounters easier doses of the other challenges. Neither zero-exposure silos nor maximum-difficulty mixtures from birth are acceptable. Mix challenges more strongly **after** component competence, while keeping a small early crossover. Proportions may later adapt to information needs, but a capability must not silently starve.

Control worlds are not idle rest worlds. They reward supported motion, accurate tracking, smoothness, low slip and **bounded** simulated mechanical shaft work. Freezing must never be the efficient solution. Work/metre is a useful mechanical proxy, **not measured battery energy**.

## One evidence-and-growth rule

Use the same compact logic for each capability rather than inventing six governors:

1. **Birth:** sample a teacher-only focus, direction, severity and supporting challenges from the *earned* frontiers. Record what was actually presented.
2. **Attempt:** the actor receives realistic observations, walks, and produces measurable physical outcomes.
3. **Close the episode:** record attempted severity, supported progress, command error, survival/falls and relevant quality. An interrupted/failed attempt is failure evidence, not a success inferred from partial displacement.
4. **Attribute locally:** pool sufficient completed episodes in the *same meaningful direction and severity context*. No single lucky win, no recycled evidence after a frontier changes, no mixing easier attempts with frontier proofs.
5. **Decide independently:** advance gently on adequate repeated success; hold with sparse/ambiguous evidence; retreat faster on sustained failure. One struggling specialty does not block another's exploration. A global emergency gait/PPO-health brake is reserved for genuine shared-policy collapse, not ordinary difficulty progression. Each family has a bounded **local** growth budget; main's four core-regime scalars retain their existing shared budget.

**Directions matter.** For velocity commands, the signed `vx/vy/yaw` limits define an **L1 octahedral envelope** (`|vx|/Vx(sign) + |vy|/Vy + |yaw|/Yaw <= 1`). Probe interiors, edges and modestly beyond, including pure-axis vertices. A successful diagonal is evidence for *that combination*, not permission to increase every independent axis. Both signs of lateral/yaw motion need evidence before expanding a symmetric limit.

For terrain, uphill does not equal downhill. An actual step ascent requires **supported upper-foot contact**, not merely the body passing a lip. A slope lesson needs real inclined **foot-ground contact** in all four directions, independently of step crossings. Report commanded **and achieved** velocities.

Keep **configured physical limits**, **sampled challenge severity**, **earned curriculum frontier**, and **independently demonstrated competence** as four separate facts. Never call one the other.

## Cold start: preserve what already worked

Historical `main` foundation: `2ec9e9db0a816c658131311129ef89ec7a595a12`.

- **20M steps; 2,048 worlds; seed 733; 32-frame PPO rollout; 4 epochs × 8 minibatches; LR 2e-4; maximum 700 control frames/episode.**
- First **70 updates (~4.59M steps) steady-only**. Initial `vx` requests were **+0.020…+0.060 m/s**. Core regime growth remains competence-driven; specialist emphases are inactive for the cold 20M and blend in gradually thereafter.
- Preserve the eight original gait/contact/clearance/smoothness/safety rewards, pooled completed-episode fall EMA, faster retreat, shared regime growth limit and PPO recovery behaviour.
- **Historical 20M SHA `470cf7f`** did indeed suppress new specialist births for the foundation. **Running 40M source `54345b2` likewise suppresses them until 40M; neither can change mid-run.** This *was an architectural mistake*, because it froze vx/vy/yaw as well as terrain. Going forward, all six fixed cohorts exist from birth but **specialist lessons activate gradually between 20M and 30M**; exposed slopes/steps start tiny, the majority of worlds retain main's easy gait, and each target grows or retreats by its **own physical performance**. The 20M observation point is not an unlock, and the source recipe change is untrained/unproven until a new explicitly authorised lineage is run. Preserve the original actor and ToF contract throughout. 20M remains merely the earliest standing milestone, with control and recovery reasonably requiring 20–50M additional steps.

These numbers are a reference to reproduce deliberately, not permission to copy mature training-profile defaults. The new 360-input actor will **not** reproduce the historical 220/232-input weights. Compare *walking quality* after its own fresh 20M foundation, not weight identity.


## Continuous target dials (25 October implementation)

A training target is a question, **not a proven capability**. Each focused lesson samples easier practice, near-target episodes and harder exploratory probes. The teacher pools completed episodes in a small checkpointed local window. A successful near-target cohort nudges its own signed command limit up by 2.5% (mixed radius 1%); a poor exploratory attempt simply holds; sustained failure on easier commands nudges the limit down by 2.5%, never below the cold probe seed. The small adjustment itself is the low-pass filter: it can wander, retreat and discover again without a graduation gate. Symmetric lateral/yaw limits require both signs for expansion but either sign can trigger repair. Eight meaningful attempts is a provisional compromise, not an achievement threshold or magic number. Physical exams remain separate.

The specialist blend follows one reproducible cold-start timeline: 20M basic foundation, then a 10M gradual teaching-exposure ramp. This is a sampling schedule, **not** a competence gate. There are no special-case checkpoints, ancestor SHAs or teaching-age overrides in the curriculum.


Current implementation covers the **command family's** continuous dials and the post-foundation activation blend. Step/slope/push/transition/control frontiers still use their prior evidence controllers until migrated; do not claim all six already share this exact rule. A future physical severity-aware migration must keep the independent up/down and incline directions.

**Growing the run target is now permitted as well.** A finished, source-pinned continuous run can raise only its requested absolute step target and resume its own validated final checkpoint using --resume auto; no 20M/40M checkpoint-era fork, no source change, and no manual JSON edit.

## Continuous incline target (26 October)

Slope evidence now uses real collision normals, separating easier and near-target completed episodes in each of four directions. Successful near-target slope episodes nudge the incline target up; difficult exploratory failures hold; repeated failure on easier slopes nudges it down. Old pooled success cannot certify a harder physical grade. The target can retreat *below its initial 0.7-degree seed* (down to a cautious 0.35-degree floor), and real slopes below the old 0.25-degree recognition threshold still count in the lesson evidence. The local exposure fraction remains modest. Step-ups and step-downs now use exactly the same principle: physical upper/lower slab displacement, rather than the requested height, determines easy/near cohorts. Both directions must demonstrate near-target supported crossings for a small upward nudge; repeated easier failures gently retreat the shared step-height target. Descent success cannot stand in for missing ascent skill. Physical push recovery, transitions and control efficiency are the remaining controller migrations.

## One honest physical world and sensor contract

Unified training uses **two solid, oriented collision slabs** representing adjoining terrain. Per-episode poses vary actual pitch/roll, step rise/drop and independently rotated edge; feet and rays encounter **the same geometry**, with no per-world mesh recompilation. Flat/control episodes must have no ghost collisions. Physics and ray/height/support calculations must agree, especially at sloped step lips and descending edges.

The new actor remains **90×4 = 360 observations, six actions**. Its terrain sensing is a delayed **4×4 = 16-cell ToF array** with validity, body IMU and optical flow; local grade/confidence is *inferred* from sensor returns. The three old tripod ToFs are not part of this new actor. **8×8 is not implemented** and would need an explicit versioned redesign. Existing 220-, 232-, and earlier 360-input policies retain their own schema/replay semantics; same tensor length does not mean interchangeable sensor meanings.

## How we know it is better

A credible examiner reports at least:

- **Gait:** actual displacement, supported locomotion, falls, slip, tracking, posture and movement smoothness.
- **Envelope:** separately evidenced pure and mixed signed commands, accelerations, pushes and recovery.
- **Terrain:** real incline grade/direction, paired up/down step heights, upper-foot support, edge angle and failures; compare combined slopes+steps, not just isolated successes.
- **Efficiency:** mechanical work per *achieved* metre alongside movement, tracking, stability and action jerk.
- **Comparison:** the same frozen tasks/seeds for old vs new, with valid sensor/physics protocols and explicit confidence/attempt counts.

The **first fresh 20M unified actor** is a major checkpoint: compare it with main's proven foundation on matched flat tasks and inspect earned physical capabilities. If the basic gait deteriorates substantially, fix the teaching distribution before committing to 100M+. A later ~10-hour run is a **single resumed lineage of the same shared actor**, not parallel bespoke speed/slope/step networks.

## Non-regression contract — what must never be traded away

Read this before accepting a curriculum, sensor, reward, physical-world or
checkpoint change. **Code compatibility, a green smoke test and a larger
training reward do not prove that we preserved a working walker.** Distinguish
the protected main *recipe* from the new actor's independently measured gait.

### Keep from `main` (reference `2ec9e9db`)

| Contract | Acceptance |
| --- | --- |
| **Gait from cold birth** | A fresh run works with no imported policy or special initial pose; first 70 PPO updates remain steady-only (~4.59M sampled world frames at 2,048 × 32). Later transition/resilience/posture lessons are earned, not abruptly imposed. |
| **Same optimizer economics** | 2,048 worlds, seed 733, rollout 32, PPO 4 epochs × 8 minibatches, LR 2e-4, 700-frame episodes, PPO health/KL protection and finite gradient/recovery behaviour. |
| **The eight gait payments** | Alive, locomotion, posture, contact, anti-slip, clearance, smoothness and joint safety retain their weights, early-fall penalties and standing-still deterrence. No new cost may quietly dominate the first 20M. |
| **Gentle initial commands and disturbances** | Main's initial +0.020–0.060 m/s forward requests, foundation caps (+0.40/−0.12 vx, ±0.14 vy, ±0.90 yaw), phase gain 0.75, regime-command fraction 0.60 and **effective** early push burden capped around the main 0.70N physical lesson—not the mature 3.2N headroom. |
| **Responsive, unified retreat** | Completed-episode fall evidence, core gait competence, faster retreat than expansion, anti-collapse behaviour, shared difficulty budget and actual offered task mix. Avoid hard-coded success from reward or partial movement. |
| **Reproducible ownership** | Real six-actuator robot XML, springs/contact, one-frame actuator delay, source-pinned checkpoints, exact normalizer/schema, resumable clean run and local inference/viewer. Preserve old 220/232/360 lineages as distinct protocols. |

Some items are numerical source contracts, enforced by small checks.
**The 360 actor cannot inherit main's learned weights:** it has new sensor
semantics, so main-like 20M **walking performance** remains a separate,
unproven behavioural acceptance gate. The new hardware-like 4×4 observations
must exist from step zero without changing the teaching distribution.

### Add through unified growth—without losing the above

| Contract | Acceptance |
| --- | --- |
| **One policy, six teachers** | Commands/resilience/slopes/steps/transitions/control target 20/20/20/20/10/10%; roles never enter actor observations. Each cohort samples easier background challenges alongside its hard specialty. All six contribute gradients to one optimizer. |
| **Protected handoff** | No active specialist lessons before 20M. Blend activation from 0% to 35% over the following 10M steps independently of earned gait. The schedule changes only teaching dose, never mastery or physical target. |
| **Proper command geometry** | Signed (vx,vy,yaw) **L1 octahedron**, not a Cartesian rectangle. Sample interior/boundary/modestly beyond; account for signed pure-axis limits and mixed-sector evidence separately. A diagonal victory cannot unlock every axis. |
| **Honest terrain** | Two actual colliding tilted slabs; uphill, downhill, both cross-slopes, step up/down, rotated lip and incline+step mixture. Training contact, ToF measurements and examiner geometry refer to the **same** physics. Supported far-side foot contact is required for crossing credit. |
| **Evidence before expansion** | Completed episodes, attempted severity and direction, movement/heading tracking, support and falls. Small near-target success nudges difficulty up; sustained easier-task failure nudges it down. Growth budgets are local to each family, not a global throttle. |
| **Grace and practical efficiency** | Make accurate **moving** gait smoother with fewer unnecessary corrections, slip and *simulated* shaft joules per achieved metre; zero movement is never efficient success. Do not claim shaft joules are electrical battery energy. |
| **No hidden oracle** | 90×4=360 versioned observations, delayed 4×4 ToF + validity, IMU and flow; simulator slope, contact truth, curriculum IDs and exact root velocity do not enter actor observations. Keep old viewer/exports correctly versioned. |
| **Independent scorecard** | Freeze the actor. On matched tasks/seeds report flat gait quality, command-sector tracking, 4 incline directions, paired supported up/down steps, mixtures, disturbances, starts/stops, J/m, slip, jerk and failure reasons. Do not promote from PPO reward, prettier viewer animation or a favorable subset alone. |

### Hard gates before a long run or promotion

- [ ] **Source/CPU gate:** current HEAD passes CI, physical MJX contact, real 360-input PPO gradient, frozen examiner and browser replay. A passing historic SHA does not count for changed code.
- [ ] **CUDA gate:** measured 2,048-world RTX 4060 Ti **production-shaped** 32×4×8 PPO update without OOM/divergence; checkpoint/resume/export and frozen/interactive inference from the exact same source/actor contract.
- [ ] **20M gait gate:** one fresh **360** cold run compared to main on matched, frozen **flat** gait exams (survival, true achieved command tracking, supported strides, fall causes, posture, shaft work). Fix a substantial regression before growing more capabilities.
- [ ] **Growth gate:** after foundation, observed command envelope and each directional terrain/recovery/transition frontier can advance, hold and retreat **without degrading the flat baseline**. No good grade should hide lost basic walking.
- [ ] **Approval gate:** owner approval before a long unified GPU campaign, `main` merge, protected 260M Pinnacle replacement or showcase promotion.

**Stop/retreat rule:** if the new source forgets how to walk, falls into
reward hacking, starves a specialist, invents a step success or loses provenance,
the correct move is to **hold the promotion, fix the cause and re-examine the
same frozen baseline**—not to train another 200M steps and hope.

## Boundaries

- Development stays on `feature/v5023-capability-growth`. **Do not edit or merge `main`**, its approved **260M Pinnacle**, or the public showcase without explicit approval.
- Keep historical source/checkpoint lineages pinned. A new actor contract or source requires its **own cold-start lineage** unless a migration is deliberately proved.
- Prefer the existing `environment.py`, `curriculum.py`, `rewards.py`, `ppo.py`, `training.py`, `results.py` and viewer/evaluation owners. Replace duplication; do not create new `v502x` managers/launchers to paper over it.
- **No unified overnight GPU campaign until** full physical-contact acceptance, joint examiner, actual RTX 4060 Ti CUDA PPO/checkpoint/resume/export/viewer lifecycle and cold-start settings are validated. The source deliberately fails closed for now.

**That is the design.** Status, verified commit, outstanding acceptance gates and *the next single engineering action* belong only in [HANDOVER.md](../HANDOVER.md).
