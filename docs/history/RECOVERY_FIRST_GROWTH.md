# EMU — competence-preserving capability growth

**Status:** proposed engineering design and controlled research plan; **no trainer code changes or GPU runs authorised by this document**  
**Date:** 27 October 2026  
**Active branch:** `feature/v5024-release-polish`; protected `main` and published Pinnacle remain untouched.  
**Evidence:** [main-vs-360 forensic audit](MAIN_VS_V5025_ROOT_CAUSES.md) and [290M screen results](EXPERIMENT1_FINDINGS.md), exact screen source `ab29c027`.

## Mission and hypothesis

**Grow new abilities without sacrificing reliable previously learned behaviour.** High-speed walking may initially have poorer recovery than moderate-speed walking; that is an acceptable unmastered operating point. Worsening at an **identical previously mastered** speed, turn, grade and shove is a regression, not payment for frontier progress. This is a *policy selection and curriculum objective*, **not** a mathematical guarantee that neural-network weights can never forget.

The owner's suspicion is **testable**, not yet established: did high-speed command practice cause forgetting of low-speed recovery, or did later episodes simply include harder pushes, poor commands, and mismatched terrain challenges? A large global fall rate at 70M cannot distinguish these explanations. The 4N push **practice target** reached its ceiling by ~24M, while commanded walking and real terrain competence remained immature. The existing training did not measure a stable force-conditioned recovery surface across model ages.

### Four non-negotiable design invariants

1. **One actor, one PPO optimiser** with realistic hardware-like observations. Do not introduce a separate standing controller, policy bank, scripted gait or privileged terrain/velocity inputs.
2. **No hard growth ceiling.** Physical limits, geometry compilation and approved test envelopes are implementation constraints, not permanent mastery ceilings. New requested capabilities can expand beyond a qualified boundary; qualified mastery cannot expand from request numbers alone.
3. **Retain useful dynamic movement.** Low-speed rehearsal still includes stepping, commanded vx/vy/yaw, starts/stops, mild slopes and recoverable shoves. Do not award survival, energy efficiency or resilience for freezing in place.
4. **A mastered condition remains examined.** Speed expansion may reduce success on *new* frontier states. A sustained loss at a previously qualified identical condition must trigger additional rehearsal, rollback/restore to the last best checkpoint or rejection of the candidate.

## The correct state representation: three independent surfaces

For physical condition `c = (requested vx, requested vy, requested yaw, slope direction/angle, step direction/height, shove direction/force, initial disturbance)`, retain:

- **Qualified capability Q(c):** independent held-out survival, commanded tracking, signed progress, supported contact, recovery and useful shaft J/m. Separate *probability of crossing* from *survival after crossing*. Qualification is based on actual movement with uncertainty/confidence, not the maximum sampled command or a teacher score.
- **Teaching distribution T(c):** the proportion and severity of episodes *currently assigned* to this condition. It intentionally includes familiar states, new boundary states and moderate overshoot, so a failed graduation does not destroy practice.
- **Exploration frontier X(c):** an explicit harder challenge that may fail frequently. Its failure does not directly demote Q(c), and should not zero T(c). No dangerous real-hardware adaptation is implied; these are simulated trials.

For practical code keep **per-axis/per-direction tables or bins** (vx+/vx−/vy+/vy−/yaw+/yaw−, four incline bearings, up/down heights and conditional push), not a dense enormous N-dimensional tensor. Use *small held-out slices at selected combinations* to catch interference/coupling. Stop reporting the single largest push/speed/grade request as 'competence'.

### Speed-conditioned resilience: the critical test

Use paired fixed episodes, with unchanged physical initial states, distinct command speed bins and disturbance levels. Track both **absolute recovery success** and *incremental damage from a disturbance*: at each command bin compare `success(no push)` with `success(force)`. The latter controls for difficulty of the movement before the push.

A practical first matrix, adjusted to demonstrated feasible speeds:

| Requested forward speed | 0N | 0.5N | 1N | 2N | 4N |
|---|---|---|---|---|---|
| 0.05m/s | baseline | easy retention | near | stretch | stretch |
| 0.15m/s | baseline | retention | near | stretch | stretch |
| 0.30m/s | baseline | near | stretch | stretch | stretch |
| 0.50m/s | near | stretch | stretch | stretch | stretch |
| 0.80–1.0m/s | aspirational | aspirational | aspirational | aspirational | aspirational |

Categories above are **proposed test scheduling**, not any claim of measured success. If a speed is not achievable without a push, do **not** interpret a failed pushed case as defective resilience. Test lateral and turning commands in separate small paired matrices, with forward progress and correct heading verified.

A regression requires same commands, physical severity, horizon, seeds, sensors, controller ON/OFF state and evaluation setup. More demanding training mixtures cannot legitimately be compared through aggregate fall rates.

## Architecture: controlled growth, permanent useful rehearsal

### 1. Reproduce main's actual maturation — not just its 20M nursery

Keep the working 20M cold foundation, then progressively introduce the **mature** command, gait-phase, transition and dynamic athletic rehearsal economics from protected main. Main mature defaults include +0.60/−0.20m/s, ±0.22m/s lateral, ±1.50rad/s yaw, phase-speed gain 1.25 and ~10% athletic rehearsal; earlier 360 run kept +0.40/−0.12, ±0.14, ±0.90, gain .75 and no athletic rehearsal. Don't assume these are immutable caps: graduate their requested practice bands based on measured tracking and stable stepping, not elapsed age. Preserve the distinction between user/operator command and boosted internal velocity-I command.

Avoid an abrupt 20M switch: blend ordinary practice over successive checkpoints while retaining early competent command bands. Freeze/normalise observation statistics only after verifying main parity and sensor signal stability; do not blindly assume the new 360 sensor channels share the main 232 normaliser behaviour.

### 2. Reserve training capacity for old skills

Suggested initial allocation *after 20M*, subject to controlled A/B:

- **20–25% lifelong retention/rehearsal:** fixed set of already-qualified flat walking, forward/reverse, lateral/yaw, transition and recoverable disturbances, with enough movement to rule out standing still.
- **55–60% focused practice:** coherent assignments across COMMAND, SLOPE, STEP, RESILIENCE and CONTROL/efficiency; one genuine primary skill per episode.
- **15–20% frontier stretch:** intentionally harder commands/terrain/shoves near and beyond present qualification, bounded by actual simulation/hardware plausibility and with independent outcome telemetry.

These are *difficulty mix allocations*, not six separate neural networks. Keep genuine diverse specialist world shares within the practice population; avoid a compulsory 50% resilience regime simply because its fall rate is high. Enforce minimum episode counts in each primary skill before reallocation; concentrate additional sampling on **informative attempts**, not necessarily the worst fall cohort.

When retention exam shows repeated degradation in a mastered cell, temporarily increase that cell's teaching share and stop promoting competing harder challenges. Keep enough other lessons alive to avoid a frozen curriculum. Use previous best full learner checkpoint as a fallback; a rejected new checkpoint is *not* the same thing as deleting harder practice.

### 3. Make a lesson physically coherent

- **COMMAND:** deliberate pure forward/reverse/sideways/yaw attempts plus mixed motion; request speeds from known-easy through just beyond competence. Better edge evidence from *more actual signed trials*, not 100 tiny promotions.
- **SLOPE:** four physical directions, predictable sustained movement and meaningful heading/velocity commands; first learn steady supported travel, then require accurate uphill/camber tracking, then combine with turning and pushes.
- **STEP:** spawn at a traversable distance, prescribe meaningful positive signed crossing command, low yaw and controlled initial disturbance. Critically, remove/adjust the current ascent-reward eligibility contradiction (**requires vx≥0.08m/s while early lessons usually ask ≈0.04–0.06m/s**). Reward monotonic physical lip approach, useful supported foot placement and far-side supported travel; independently record post-cross survival. Distinguish ascent and descent competence.
- **RESILIENCE:** start with *actually achievable* flat dynamic walking at a known command, apply measured shove/initial pose disruptions with controlled directions, track subsequent recovery to commanded movement. Practice larger shoves at appropriately supported speeds only when force-conditioned exams justify them.
- **TRANSITION / CONTROL:** starts, stops, reversals, heading changes, step cadence and smooth torque/shaft work while actually moving. No efficiency achievement for immobility.

Introduce compound terrain / disturbances **after** coherent single-skill competence, but continue low-dose familiar mixing to maintain transfer. No 'all dimensions must wait for the slowest' global growth gate.

### 4. Resilience must be measured by force, not requested force

A 4N curriculum target is not 4N recovery ability. The old governor allowed the requested push strength to reach 4N by ~24M while it used regime-averaged fall/competence signals. Replace this with **force-conditioned** success at comparable commanded speeds and defined directions, recording whether a shove actually occurred and its impulse (force×duration), not just the possible upper bound.

Don't escalate default background pushes in STEP/SLOPE worlds in lockstep with RESILIENCE's ambitious frontier. Use a bounded small background nuisance there, alongside a dedicated resilience focus capable of testing much harder disturbances. Reduce severity or exposure after repeated bad outcomes *without discarding qualification in already easier cells*.

### 5. Admission, regression and candidate-selection logic

- **Practice can expand on failure** when the attempt is informative and safe in simulation; **qualification only expands on actual successful results**. A single short bad easy cohort does not destroy all old slope/step qualification or clear every signed-direction window.
- **Hold the previously qualified surface:** at each significant training milestone, compare to the accepted previous checkpoint on the same fixed, seeded cells. As a *provisional engineering gate*, investigate repeat losses larger than 5 percentage points in previously reliable survival/crossing, or >10% increase in command MAE at identical conditions. These are triggers, **not proof** from tiny cohorts.
- Use a first 32-world paired screen per representative cell, then repeat borderline/core regressions with ≥64–128 comparable episode outcomes and additional seeds. A confidence interval crossing the threshold means '**inconclusive**', not 'pass'. Check fall/recovery, signed distance, tracking and movement simultaneously.
- **Do not trade away the easy core for one speed headline.** Pick Pareto-improving checkpoints: retain easy/medium speed recovery and expand the reliable velocity/terrain volume. At genuinely newly mastered high speeds some reduced resilience can be acceptable, explicitly reported as such.
- Keep raw operator-command performance in every integral-controller test. A controller boost that simply raises the actor's demanded vx without improving *actual* movement is not a pass. Do not enable an unverified velocity estimator frame transform on cambers.

## Sequential causal programme — no more blind 12-arm marathon

### P0 — Frozen physical examinations, **zero training steps**

- [ ] Run fixed, equal-physics supported command/terrain/recovery tests for the accessible main 260M actor, new 20M and 50M common checkpoints, 70M staged-OFF, balanced-OFF and 400ms/2x assisted branches. Same physical cases but each actor's legitimate 232/360 observation and matching controller.
- [ ] At fixed low/medium commanded speeds compare no-push versus 0.5/1/2/4N recovery at *every age*. Determine if old-cell resilience truly **degraded** after maturation or training simply grew harder.
- [ ] Audit ToF lip visibility, measured optic-flow estimates versus physical motion (diagnostics only), actuator/collision parity and reward accessibility. Do not touch `main`.

### P1 — Restore mature locomotion and test retention (one new shared 20M cold start)

Same new source SHA and shared 360 input/robot. Fresh 20M main-style foundation, then **three matched 20M continuations** (all 20→40M):

- **A — control:** previous 360 post-foundation ordinary/teacher curriculum with capped gentle background pushes and same test cases.
- **B — main-mature:** A plus restored mature command/gait-phase/rehearsal training; same background pushes.
- **C — main-mature + retention:** B plus constant qualified-skill rehearsal and conditional regression protection.

`B−A` primarily tests the mature locomotion recipe, `C−B` tests skill retention. This is a **recipe package test** for A→B, not identification of the single best numeric hyperparameter. Observe signed *achieved* speed, dynamic gait, stable recovery at fixed lower speeds, and measured energy—not taught envelope values. Source guard/provenance, checkpoint parent and teacher config must be recorded; do not corrupt old archived runs. Total **80M aggregate steps** including foundation.

**P1 gate:** useful achieved speed improves relative to A **without** a repeatable drop in the previously reliable fixed-speed recovery/command exam. If not, investigate reward/phase/observation mismatch before adding terrain complexity.

### P2 — Isolate coherent terrain lessons versus force interference

Fork from the selected qualified **40M** checkpoint (same source/actor), three further **20M** trials (40→60M each):

- **D — mixed-terrain control:** old random regime/command terrain lessons; low background force.
- **E — coherent terrain:** physically deliberate STEP and SLOPE goals, decoupled reward/crossing/retention evidence; **same low background force** as D.
- **F — coherent terrain with aggressive original push dosing:** same as E except fast push exposure/escalation.

`E−D` tests coherent task assignments and teaching. `F−E` tests damaging force interference. If E grows real supported up/down steps and slope-tracking skill while F loses it, the owner's interference hypothesis gains strong causal support. Conditions and independently fixed seeds match; high difficulty is a treatment, not proof of fitness.

Aggregate **140M world steps** for full P1+P2 (~half previous 290M). These are planned trial lengths, not a commitment to continue unproductive arms. Prefer selecting P1 before running P2. If 20M isn't long enough to resolve a promising trend, extend *the finalist*, not every arm.

### P3 — Integrate, replicate, graduate

- [ ] If P1 and P2 isolate useful effects, combine them into **one simple coherent curriculum** with explicit baseline retention, speed-conditioned force growth, permanent old-skill rehearse, and distinct practice/probe/qualification semantics.
- [ ] Test bounded 400ms/2x and 800ms/2x velocity-I against OFF with matched actor and **actual operator motion**, after coherent locomotion training is restored.
- [ ] Replicate the finalist with a different seed and a true cold-start, preserve exact run source; verify CPU collision and **one full 2048-world CUDA PPO update** then a real sustained result. Do not create a long pile of brittle synthetic tests.
- [ ] Only then commit to one uninterrupted **400M fresh cold-start** run. Examine at 20/40/60/100/200/300/400M; preserve prior acceptable model checkpoints for rollback and common-physics main-vs-new comparison.
- [ ] Never publish/merge to protected `main` or overwrite Pinnacle without owner-approved common benchmark and browser controller parity.

## Acceptance / rejection summary

**Accept as on-track only when**: normal gait still works; actual commanded forward and lateral motion advances in held-out cases; resilience at previously learned speeds does not regress materially; real signed grade walking and supported step crossing begin improving beyond millimetres and subdegree targets; reported 4N / 1m/s requests are not confused with passed exams; and useful shaft J/m does not worsen without acknowledged speed/ability gains.

**Reject/pause** when: changing challenge mixtures alone lowers measured old-cell survival, slope/step teachers repeatedly retreat to floors, a near-step task cannot earn its own progress signal, force frontier maxes without demonstrated recovery, the network protects itself by standing still, or the same-age new candidate loses against the restored-mature 360 control.

**Engineer the smallest mechanisms capable of testing this model.** No second neural network, no new grand generic curriculum framework, no giant test suite, no tmux. Run foreground WSL commands only after P0/P1 code review. Maintain this checklist in `HANDOVER.md`; this document itself implements none of the training changes.

## Tonight's executable P1/P2 recovery-first campaign

**Implementation committed on this feature branch:** `scripts/v5026_recovery_first.py`
with new CLI training knobs `--growth-recipe`, `--retention-share`,
`--terrain-lesson`, `--force-program`.

- **New clean seed 735**, source-pinned 360 actor, fresh result root
  `~/robot_lab/emu_runs/v5026_recovery_first_s735`.
- **20M** original low-force/low-speed cold nursery, then **3×20M**
  matched P1 comparisons: old 360 control (gentle real force);
  main mature commands, gait phase and athletic rehearsal; and the
  same with **25% low-disturbance, actually moving skill rehearsal**.
  The core exercise includes signed reverse, lateral and yaw commands
  and bounded mild shoves; its control worlds have no terrain
  in the later coherent lessons.
- At 40M compare frozen eval core, dynamic80, resilience75, vx MAE,
  broad score. Fail the new candidate back to the measured control
  when it fails old-skill safeguards; save `p1_selection.json` with
  exact rule and all figures. This is screening, not a speed-force
  physical P0 certification.
- **3×20M** further forks 40→60M: random-terrain/gentle-push
  control, coherent STEP/SLOPE/gentle push, coherent STEP/SLOPE/high
  push. In coherent training STEP/SLOPE get positive 0.10–0.22m/s
  forward operator commands, no yaw/lateral request and no
  simultaneous step+grade for their primary lesson. Real supported
  contact and original raw command remain the reward/exam criterion.
- Total **140M aggregate training steps**. All runs are in the
  **foreground terminal**, sequential, resume-aware, with CPU
  checks and one full 2048 CUDA gradient update before foundation.
  Ctrl+C forwards termination to child; restart the same command
  on the exact unchanged source to resume.
- The old 290M source and protected `main` remain untouched.
  The 260M 232-schema Pinnacle **is not silently included in this
  frozen 360 exam**; common 232/360 P0 is still separate and required
  before any main replacement.
- This is implemented but local CUDA has not been verified from
  ChatGPT. CPU and Python CI are distinct from the user's CUDA
  preflight; do not announce success prematurely.

Foreground commands after `git pull --ff-only`:
```bash
cd ~/emu-graduation
source .venv/bin/activate
python scripts/v5026_recovery_first.py
python -u scripts/v5026_recovery_first.py --run
```

## Execution: new 2026-10-27 overnight source-pinned implementation

This P1/P2 plan is wired to the actual CLI and environment via
`--growth-recipe`, `--retention-share`, `--terrain-lesson` and
`--force-program`, plus `scripts/v5026_recovery_first.py`.
The 20M foundation is intentionally a **single unchanged legacy 360
cold start**. Every subsequent arm is a same-source validated fork
with its own explicit learning configuration and full optimizer/policy
and teacher checkpoint. P1 isolates main mature settings and a 25%
genuinely moving control rehearsal cohort. P2 isolates coherent
supported STEP/SLOPE tasks and high vs gentle force exposure.
Running the experiment does NOT certify anything about Pinnacle or
replace the separate original-source read-only P0 benchmark.

P1 compares actual held-out fixed core/dynamic/resilience results,
not merely requested frontier speed, and auto-selects a defensible
40M source (or safe baseline) before P2. Report all comparisons
as causal **recipe bundles** rather than proof of specific knob efficacy.
All attempts and source SHA are recorded in the run's manifest and
`overnight_results.zip`. The runnable source needs CPU/2048-world
CUDA acceptance before serious training, and old result roots never
receive new code. Never run under tmux unless user explicitly changes
preference: run directly in the foreground WSL console.
