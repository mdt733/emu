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
