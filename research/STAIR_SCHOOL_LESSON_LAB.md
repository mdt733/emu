# EMU KS150 Stair Teaching: Lessons Laboratory v1

**Status:** preregistered **design**, not trained evidence. **Purpose:** discover which teaching methods are worth building into a *new, from-scratch* EMU school. This is **not** an actor tournament to select a champion from the forks.

**Source and ancestry:** the currently completed `h40-spring-only` seed-736 student at **120,061,952** PPO environment steps was trained at Git SHA `4048e8450c36882501cbb624f4ca656e11da9419`. Its exported actor SHA256 `f1e808b2c1fd3cb6433ddbcea9f1c9e1afce69c69eb768fabf678907e7deecbc` is **not** a training checkpoint. The tournament requires the original, full, immutable `checkpoints/final.pkl` outside Git. The local launcher pins *its exact SHA256* and prevents a parent change. It also freezes the **new tournament Git source**, which may differ from the parent's original source; the existing authenticated cross-source research fork gate still applies.

**Physical contract:** original Spring-only KS150 variant, not original `main` H40; 360 inputs, 6 servos, one PPO actor, 2,048 MJX environments, source-matched sensors and prism geometry. All forked children retain policy, optimizer, observation normalizer, curriculum/field/health state and checkpoint provenance. No branch changes robot physics, sensor meanings, PPO algorithm, or exported `main` asset.

## The actual question

Which *generalizable educational principles* can improve real uphill toe acquisition, **two-foot transfer**, stable follow-through and **long-duration survival** without unlearning flat gait, commands, slopes, standing balance or recovery?

We want evidence about **what to teach, how to teach it, how often to teach it, when to reward it, and which seemingly sensible teaching devices should be dropped**. A treatment that loses stair crossings but exposes a dangerous reward shortcut, overexposure or a missed credit-assignment signal is scientifically valuable. One observed branch result is a **hypothesis-specific finding**, not necessarily a portable law. There is no promise that 12 branches yield 12 positive discoveries.

## Preregistered comparison

All 13 arms start from **the same verified 120,061,952-step checkpoint**, run on the **same frozen tournament source**, with identical episode RNG seed, optimizer state, world count and PPO schedule. The ultimate requested target is **160,000,000 steps**, about **39.98M additional steps per arm** after full-batch rounding. **Training must stop at 130M, 140M and 160M requested steps** (real PPO boundaries **130,023,424**, **140,050,432** and **160,038,912**). An interrupted stage may auto-resume that stage; advancing to the *next* stage is a separate, explicitly reviewed same-source full-state continuation. All arms including control experience the same stage boundaries, so episode-buffer reinitialization is a common condition, not an unbalanced treatment. The control makes the *exact same* teacher decisions and reward choices as the parent continuation. Each of the 12 treatments alters **one** existing CLI training configuration field. Same-arm partial restarts use only `--resume auto` and their own checkpoint, never a second fork from the ancestor.

**Why one parent first:** it is a relatively economical *mechanism screen*. It can isolate treatment differences conditional on this already-trained KS150 student. It cannot establish that a new teacher works from scratch, that the effect generalizes to other training seeds, or that the actor beats original-H40 `main`. Repeat the strongest *lessons*, not just the best policies, across multiple cold parent seeds before incorporating them into a new mainline teaching skeleton.

**Important comparability caveat:** some treatments will change the stochastic lesson sequence once training restarts; shared seed and parent are useful common controls but are not a guarantee of trajectory-by-trajectory paired randomness. Outcomes must be evaluated on separately generated **matched frozen physical exams**, not on training rewards alone. The same tested source also ensures the 350-frame view is taken *inside the same 1200-frame rollout* whenever evaluating speed/horizon effects.

## Selected 12 explanatory arms, plus the control

One field per arm, so a positive/negative effect can be attributed more clearly than a bundle of simultaneous changes. **All values are explicitly within existing `emu train` flags and validation ranges**. These are proposals to investigate, not predictions of guaranteed improvement.

| Arm | Baseline → treatment | Question to answer | Primary precommitted readout | Counter-risk / falsification |
| --- | --- | --- | --- | --- |
| **C00** continuation | Identical teacher, +40M | How much improvement comes from extra PPO time alone? | All stair and non-stair gates | No treatment effect can be claimed without control-relative gain |
| **H01** command fidelity | `coherent-careful` → `coherent-direct` | Is attenuating coherent step commands slowing useful physical transfer? | Early/late bilateral transfer by commanded speed | Increased speed but falls/command interference; affects STEP **and SLOPE**, not just Academy |
| **H02** stronger destination credit | Step goal `.12` → `.22` | Is supported progress under-rewarded? | 5-frame supported traversal / long-route completion | Foot movement without stable support, drift or reward hacking |
| **H03** velocity-relief ablation | Relief `.55` → `0` | Does forgiving speed tracking teach lingering and indecision? | Time-to-transfer; tracking and first-life survival | Strict tracking prevents safe obstacle negotiation |
| **H04** earlier encounter | Lip approach `.22 m` → `.15 m` | Is time spent approaching diluting stair learning? | Assigned first-lip encounters and completed transfers per life | Birth too close causes crashes or bypasses anticipation |
| **H05** first-foothold credit | First contact `0` → `4` | Does specific credit help at the +32 mm acquisition cliff? | Lead-top at ≥32 mm **and** conversion to two feet | One-foot hover without subsequent transfer |
| **H06** bilateral credit | Two-foot bonus `6` → `9` | Does stronger verified support credit fix the transfer bottleneck? | Both-feet five-frame / lead-top ratio | Sticky stance with weak following movement |
| **H07** follow-through ablation | Follow-through `7` → `0` | Is post-transfer reward distracting from learning the transfer itself? | Same-life transfer and long-path continuation | Improved near-lip score but worse route completion |
| **H08** greater Academy dose | Birth share `.28` → `.40` | Does the stair skill simply need more physical exposure? | Per-eligible-birth improvement and core skill retention | More near-lip encounters but catastrophic forgetting |
| **H09** smaller Academy dose | Birth share `.28` → `.16` | Is the existing Academy overtraining a narrow gait? | Generalization + slope/command retention | Not enough attempts; slow acquisition |
| **H10** stronger general retention | Core rehearsal `.30` → `.38` | Can we protect walking while stairs learn? | Core commands, recovery and signed slopes vs control | Protected generalist but lower stair conversion |
| **H11** shorter course | Academy `1200` → `900` frames | Are overlong lessons slowing completion/safe graduation? | Same-life transfer, completion, independent 1200-frame survival | Trains to survive 900 frames but fails after frame 900 |
| **H12** stepping-gate ablation | Gate `legacy` → `zero-only` | Does the established contact/posture gate suppress useful obstacle preparation? | Real toe clearance, supported transfer and fall rate | Worse support, motor effort or general balance |

This deliberately includes **things to add** and **things we may be wrong to keep**. No treatment changes the `safe_ascent` promotion requirement to hide failures. That governor is another valuable scientific topic, but relaxing a first-life safety gate before we understand transfer and survival can give seductive yet dangerous frontier numbers.

## Twelve more candidates considered, not included in the first cohort

These are *reserve hypotheses*, not unplanned mid-run knob changes. Choose only in a separately versioned second study, based on what v1 teaches.

| Reserve | Change / question |
| --- | --- |
| R13 | Weaken near-zero phase advance (`--science-phase-lowcmd-scale .6`): is the clock forcing stationary steps? |
| R14 | Weaken phase synchronization (`--science-phase-sync-strength .5`): does organic lift guidance suppress asymmetrical climbs? |
| R15 | Allow 2 cm extra crouch (`--science-crouch-allowance-m .02`): is strict body height blocking foot clearance? |
| R16 | Add stride efficiency (`--science-stride-efficiency-weight .05`): can inefficient foot cycling be reduced? |
| R17 | Add effort guidance (`--motor-guidance-weight .025`): are motor effort limits influencing uphill stalls? |
| R18 | Sensor-based teacher hint (`--science-sensor-teacher-bonus 4`): is perception, rather than contact, the limiting signal? |
| R19 | Supported-crossing payment (`--science-supported-cross-bonus 5`): does direct crossing credit generalize beyond foothold credit? |
| R20 | Intermediate Academy share (`--academy-max-share .34`): investigate nonlinear exposure effects after H08/H09 |
| R21 | Remove destination goal (`--science-step-goal-weight 0`): does progress shaping invite shortcuts? |
| R22 | Increase velocity relief (`--science-step-velocity-relief .8`): if H03 fails, where is the helpful range? |
| R23 | Longer initial approach (`--science-step-approach-m .30`): does anticipation need more time? |
| R24 | Stronger follow-through (`--science-followthrough-bonus 11`): if H07 fails, test additional continuation credit |

**Not randomly permuted factors:** this cohort measures 12 deliberately selected single-field perturbations on a shared parent. It is a mechanistic screen, not a full factorial or an unbiased estimate of every potential teaching interaction.

## Keep the generalist alive: co-primary physical outcomes

**Frozen examinations, not Academy reward or trainer-reported frontier, determine learning.** Pre-register the exact exam seeds **6901 and 6902** for discovery and keep **7901 uninspected** for a later locked integrated-school validation. The previously inspected seed 5701 is **design-time evidence only**, never an independent final holdout.

At the parent checkpoint and **+10M / +20M / +40M** branch development stages, retain: complete source/checkpoint/actor/model SHA; observed teacher birth counts and exposure; the actual gated uphill and downhill curriculum and their first-life survival evidence; 4–80 mm exact signed stair and matched-flat curves; speed 0.10 and 0.25 m/s; leading-upper-foot support, 5-frame both-feet support, 5-frame controlled continuation; first-life survival and full-course endpoint; independent command tracking, yaw, cross-slopes, dynamic balance, resilience and motor effort. Compare the frozen actor to the **same checkpoint-age control** and parent. Use exact matched exam worlds, fixed seed pairs and physical model across branches. Reporting the entire signed-height curve is required: never optimize a single headline scalar.

### Lesson ledger, not leaderboard

Each treatment closes with a **one-page causal-learning entry**: (1) the original question; (2) which signal was altered; (3) observed dose (births/events), not just intended knob; (4) measured effect versus the same-age control, including paired-world uncertainty; (5) first-foot → two-foot → traversal → course-survival funnel by height and speed; (6) what non-stair behaviour improved or degraded; (7) whether the predicted mechanism was supported, contradicted or undecidable; and (8) **proposed change to the next teacher**, including when *not* to apply it. Keep inconclusive/no-effect entries rather than forcing 12 positive conclusions.

**Safety is a hard veto, not a minor score penalty:** a stair bonus that gains transfers while causing major losses in general walking, standing, resilience or long-horizon survival does not qualify for the new school. The nominal screen flags >5 percentage-point loss versus control in any core survival/command panel for mandatory review; this is a caution threshold, not statistical significance or automated deployment approval. **No arm automatically advances beyond the 130M or 140M stage**: an explicit `--reviewed` acknowledgement is required, after checking frozen command/slope/recovery panels and stair funnel evidence against the same-stage control. A safety failure should pause that arm for investigation, even if its stair metric rises. High stair performance alone is insufficient.

### After the 13 forks: interactions and cold-start proof

The initial twelve treatments are **single-factor probes**. Study a *small, preregistered set of combinations* only after examining the mechanism ledger, for example command fidelity × two-foot credit, first-contact credit × continuation reward, Academy dose × core retention, or encounter distance × course duration. A useful combination must be compared with its **two individual components and the continued control**, otherwise adding gains can conceal antagonistic interactions. Run at least two fresh parent seeds for the most informative findings; do not infer repeatability from a single seed-736 ancestor.

Finally construct **one coherent new teacher**, without twelve simultaneous ad-hoc reward knobs, and run **fresh 0→120M cold students with untouched evaluation seeds**. Compare it to the original KS150 teacher with matched seeds and the same hardware, and separately evaluate any proposed `main` replacement under an authenticated same-task/same-hardware policy before release. No tournament fork is automatically a champion; provenance remains immutable.

## Operator interface (no training unless explicitly requested)

On the dedicated lab Git branch, the one cohesive helper is `scripts/stair_school_lesson_lab.py`. It never launches all arms implicitly. `list`/`plan` are read-only; `lock` writes a new immutable study manifest **outside Git**; `run` verifies source identity, checkpoint SHA and existing EMA/config contracts, then launches exactly one named arm. GitHub CI handles software regression checks.

```bash
# Inspect all hypothesis names (read-only)
python scripts/stair_school_lesson_lab.py list

# When the COMPLETE original 120M checkpoint exists, register the parent
# (no PPO or GPU execution; refuses an existing study root)
python scripts/stair_school_lesson_lab.py lock \
  --parent ~/emu-data/runs/h40-school-cold-120m-s736/checkpoints/final.pkl

# See the exact pinned commands before any fork (read-only)
python scripts/stair_school_lesson_lab.py plan

# Later, on explicit approval, run ONE CONTROL to the FIRST review stop.
# The same command may resume only that child's checkpoint if interrupted.
python scripts/stair_school_lesson_lab.py run --arm C00_control --to-steps 130000000

# Only AFTER reviewing matched frozen stair/general-skill exams at 130M,
# explicitly advance this arm to the NEXT stage; never skip 140M.
python scripts/stair_school_lesson_lab.py run --arm C00_control \
  --to-steps 140000000 --reviewed
```

The helper pins the **new source Git SHA** at lock time. The study lock is JSON-round-trip tested against the real CLI's full control recipe, with checkpoint SHA mismatch rejection before any training. Any later code/docs commit changes that SHA and will deliberately prevent training until the exact frozen experiment source is checked out again. A completed training run cannot be resumed from an evolving PR head. For another independent parent seed, a new preregistered study/version is required.

**Budget:** 13 × 39,976,960 = **519,700,480 additional training environment steps**, plus independent physical exams and two compulsory intermediate human reviews per arm. Training source remains frozen across stage extensions. Actual baseline 120M telemetry recorded ~3,530 seconds of steady PPO-update time; a 40M continuation on comparable hardware is roughly one third of that *before* compilation/exams, so the full thirteen-arm design is material but not inherently absurd. If 130M stage evidence is uninterpretable, pause and diagnose rather than automatically spending 519.7M. Budget the full study explicitly; if early stopping is permitted, preregister **identical safety-based stopping rules** before looking at outcomes. Never stop or extend just the exciting arms based on a mid-run leaderboard.


### Audit correction: source-locked exams for every child

The original four-condition 120M diagnostic was deliberately locked to
seed 736 at source `4048e845`; it **cannot** evaluate a new fork. The
lesson laboratory therefore authorizes the same 350-inside-1200 MJX
snapshot method **only** when supplied a study lock, named arm, frozen
export from that arm's validated *final* checkpoint and a completed
130M/140M/160M stage. The examiner checks Git source, 2048-env seed,
arm/run name, KS150 model, exact PPO boundary, original 360-input
contract and exam seed. The previous parent-only diagnostic remains
locked to seed 736. No arbitrary actor is accepted by the new route.

After completing an arm's 130M stage, export its frozen checkpoint
(example: C00; use a *new* evidence output folder):

```bash
EMU_PHYSICAL_PROFILE=h40-spring-only python -m emu.cli export-policy \
  ~/emu-data/studies/ks150-stair-teaching-lessons-v1/runs/C00_control/checkpoints/final.pkl \
  ~/emu-data/studies/ks150-stair-teaching-lessons-v1/runs/C00_control/actor-130m.emupolicy
EMU_PHYSICAL_PROFILE=h40-spring-only python scripts/v5053_matrix_exam.py \
  --school-diagnostic-snapshot-suite \
  --lab-study-lock ~/emu-data/studies/ks150-stair-teaching-lessons-v1/study_lock.json \
  --lab-arm C00_control \
  --policy ~/emu-data/studies/ks150-stair-teaching-lessons-v1/runs/C00_control/actor-130m.emupolicy \
  --output ~/emu-data/studies/ks150-stair-teaching-lessons-v1/evidence/C00_control/130m-s6901 \
  --seed 6901 --device cuda
```

Repeat with **6902** for the second preregistered discovery cohort,
and with each treatment at the *same stage*. The untouched seed 7901
remains sealed for integrated-school proof, not exploratory branch
scoring. Full signed uphill/downhill 350-frame matrix exams and ordinary
command/slope/recovery retention exams are also required before
`--reviewed` stage approval; the speed/horizon study alone is not a
complete generalist-retention assessment.


For **generalist retention**, the existing frozen 46-task pulse is
also required on the same actor and both discovery seeds; it scores
commands, control, transitions, signed slopes, steps and resilience:

```bash
EMU_PHYSICAL_PROFILE=h40-spring-only python scripts/examine_h40_specialties.py \
  --policy ~/emu-data/studies/ks150-stair-teaching-lessons-v1/runs/C00_control/actor-130m.emupolicy \
  --output ~/emu-data/studies/ks150-stair-teaching-lessons-v1/evidence/C00_control/pulse-130m-s6901.zip \
  --seed 6901 --worlds 2048 --horizon 600 --device cuda
```

To assess **both signed stair directions** at each stage, use the
independent 350-frame matrix (up and down, 4–80 mm, 10 repeats per
height, 400 active worlds) with the same frozen policy and seed:

```bash
EMU_PHYSICAL_PROFILE=h40-spring-only python scripts/v5053_matrix_exam.py \
  --policy ~/emu-data/studies/ks150-stair-teaching-lessons-v1/runs/C00_control/actor-130m.emupolicy \
  --output ~/emu-data/studies/ks150-stair-teaching-lessons-v1/evidence/C00_control/signed-130m-s6901 \
  --mode stairs --worlds 512 --levels 20 --trials-per-level 10 \
  --horizon 350 --seed 6901 --device cuda
```

Repeat with seed 6902 and each treatment. These are **physical
experiments**, not the ordinary GitHub regression checks; their results
must be inspected before stage advancement. The pulse is a short
screen, not proof of 60-second station keeping or sustained recovery.


## Prelaunch audit and go/no-go (reviewed before any GPU branch)

**Verdict: GO for a staged mechanism screen; NO-GO for automatically spending
all 519.7M steps or claiming a new champion.** The first exercise is
C00 control, then H06 bilateral credit, H01 direct commands, H03 velocity
relief ablation and H07 follow-through ablation at the **130M review stop**.
This order targets the observed small-step two-foot-transfer problem without
privileging an interesting-looking branch with extra training time. The other
arms retain their preregistered control-relative questions and the same stage
boundaries. Do not discard their negative results or improvise new thresholds.

| Audit item | Outcome and limitation |
| --- | --- |
| Source and shared parent | The recorded seed-736 120M configuration matches all control teaching choices. Real local `final.pkl` bytes, exact PPO rewind state and SHA remain to be checked by `lock` and source-approved `--check-resume-only`; the evidence ZIP is **not** the checkpoint. |
| Launcher operability | Corrected a JSON-array-vs-Python-tuple lock comparison that previously made `plan`/`run` unusable. CI now includes a lock → read-back → checksum-mutation rejection test, plus thirteen real CLI recipe parses. |
| Stage integrity | All arms stop at the same 130M/140M/160M PPO boundaries. No stage skips or silent extension. A subsequent stage requires a validated prior `final.pkl` and explicit `--reviewed` acknowledgement; this acknowledgement is **manual**, not an automated physical-result certificate. |
| Frozen stair measurement | The original parent-only snapshot examiner rejected new fork actors. The lesson-lab path now requires the study lock, exact arm/seed/physical model/source/stage, and only exam seeds 6901/6902. The old seed-736 examiner stays restricted. |
| Generalist retention | Short command/slope/transition/recovery pulse and full signed stair matrix must be run on the same exported actor at the same stage. Training reward or own Academy frontier cannot certify retention; the pulse is not long-endurance proof. |
| Attribution | One changed config field does not always mean one changed physical mechanism: H01 affects STEP **and SLOPE**, Academy dose rebalances other births, and H11 changes exposure and course-survival criteria. Report actual births, attempts, survival and physical outcomes rather than attributing all changes to a single intuitive explanation. |
| Replication | Same-parent matched training seed is a **local intervention screen** (one learner lineage). Paired eval worlds reduce measurement noise but do **not** estimate variability across learning seeds. Replicate informative mechanisms on another parent and then in fresh zero-start schools. |

**Why these arms are worth a bounded attempt:** the current school already
obtains an upper foothold on many small lips, yet its independent first-life
transfer, controlled continuation and full-course survival are weak. H01/H03,
H06/H07, H04/H11 and H08/H09/H10 probe different plausible failures:
command economics, credit assignment, encounter timing, dose and retention.
Ablations can disprove an existing incentive as usefully as additions can
improve it. H05 first-foothold credit is **lower priority**: easy-height
footholds may already be saturated while the 32 mm acquisition cliff lies
beyond the tiny current Academy uphill frontier. No apparent H05 benefit at
+32 mm can be inferred without actual high-step physical exposure. H12 is
**high risk** because a global gate change can sacrifice ordinary gait; stop
for retention review before extending it.

**What we will not claim:** this is *late-school remediation* of a KS150
student that already spent 120M steps in a stair curriculum, not a pristine
120M generalist-only foundation. It cannot isolate what should have happened
at 20M/60M/80M birth, guarantee twelve positive findings, or establish
superiority over original-H40 `main`. We are buying explanatory clues to
the next teacher, not twelve independent discoveries by fiat.

**Stop/continue review at each common stage:** verify the saved checkpoint
and output metadata; inspect per-height uphill/downhill first-foot → two-foot
→ traversal and 350/1200 first-life survival; inspect actual Academy dose
and near-frontier births; compare independent core command, balance, slope
and resilience outcomes with the *same-age control*. Treat a missing or
underexposed physical event as **inconclusive**, not a zero-performance
proof. The nominal 5-percentage-point retention-loss caution triggers
manual investigation, not automatic significance or rejection. A training
collapse, corrupted rewind, missing source identity, or large core regression
is a reason to stop that arm before any `--reviewed` continuation.

**Evidence not yet performed:** an actual WSL `lock` against the complete
120M final checkpoint, an operator-side 2048-world PPO continuation, and
full frozen 6901/6902 physical exams of a newly exported branch actor.
GitHub's Python/Node regression suite tests APIs and guards but cannot
substitute for these GPU and science acceptance steps. No branch training
is authorized merely because the draft research PR exists.
