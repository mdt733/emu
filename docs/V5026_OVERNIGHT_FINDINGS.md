# v5026 recovery-first experiment — complete 140M audit

**Audited:** 28 October 2026  
**Owner-provided input:** \`overnight_results(3).zip\`; \`campaign_status.json\` shows all 7/7 complete.  
**Actual GPU training source SHA:** \`ab64cac18637222924f3c79680e30011830f7c4a\`; seed 735.  
**Experiment:** 20,054,016 foundation transitions + 3 × (40,042,496−20,054,016) + 3 × (60,030,976−40,042,496) ≈ **140M aggregate transitions**. Every run contains a final file, no reported rollback, and successful full-2048-world preflight. Training outputs are not policy checkpoints in this ZIP; physical replay requires locally retained source-pinned checkpoints.

## 28 October: matched physical examiner — strict-score masking discovered

**Evidence:** owner supplied three complete frozen `unified_exam.json` results
for P2 D, E and F, run with exact archived source `ab64cac`, CPU,
8 worlds per case, 350 frames, 0.15m/s command, matched physical topology
and seed 1701. No new PPO steps and no evaluation feedback to policy.

**Important scoring distinction:** the original examiner's `case_pass`
requires three-axis mean absolute tracking error, including zero-yaw
yaw-rate MAE <=0.08rad/s. All three policies had flat yaw-rate MAE
approximately 0.115–0.126rad/s, due substantially to gait oscillation,
despite mean signed yaw-rate biases of only +0.004 (D), +0.033 (E),
+0.012rad/s (F). All three survived flat 8/8; mean forward error
was 0.023, 0.024 and 0.022m/s, respectively. Hence **flat strict 0/8
does not mean flat walking failed**. Keep strict heading precision as a
distinct research objective rather than dropping it.

### Genuine supported crossing, counted separately from yaw precision

| Step height | D mixed | E coherent | F coherent + push |
| --- | ---: | ---: | ---: |
| 3mm | 8/8 | 8/8 | 8/8 |
| 5mm | 8/8 | 8/8 | 8/8 |
| 10mm | 4/8 (0/4 up) | **8/8 (4/4 up)** | **8/8 (4/4 up)** |
| 20mm | 4/8 (0/4 up) | **7/8 (3/4 up)** | **7/8 (3/4 up)** |

Real physical evidence: verified terrain geometry, started near side,
reached far side and **five consecutive supported contact frames**,
with full-horizon survival. At 20mm one E and one F case failed;
training-time frontier 28–29mm remains *unexamined*.

### Verified grade stability and commanded movement are different

Across the frozen 1/2/3/5° matrix, every grade has real physics,
support and measured body velocity. Most cases survived, and almost
all passed physical grade **stability** even for D. But that alone is
not accurate locomotion. The coherent recipes show markedly lower
forward tracking MAE on matched 5° grades:

| 5° direction | D vx MAE | E vx MAE | F vx MAE |
| --- | ---: | ---: | ---: |
| Uphill | 0.263m/s | **0.039m/s** | **0.027m/s** |
| Downhill | 0.183m/s | **0.017m/s** | **0.017m/s** |
| Left camber | 0.169m/s | **0.031m/s** | **0.025m/s** |
| Right camber | 0.151m/s | **0.022m/s** | **0.021m/s** |

D can remain balanced while being dragged/downhill or losing uphill
progress: this directly supports the old concern that stability
alone is not directional command competence. The 5° results strongly
favour coherent teaching, **at this speed and these eight worlds**.

**Reporting repair:** the development-branch frozen examiner now retains
its original strict `case_pass` and adds separately named `mission_pass`
(physical task plus bounded translational MAE, mean yaw bias and yaw jitter),
`crossed`, and `stable`; schema versioned `v2_dual_scores`.
This is a *new diagnostic definition*, not retroactive reclassification
of the frozen v1 results or a substitute for longer-range path accuracy.
No training code or archived source/checkpoint changed.

**Next:** repeat 1cm/2cm uphill and 3°/5° directional cases with more
seeds and worlds; improve heading tracking as its own goal, preserve
generalist recovery, and examine larger absolute steps before claiming
28mm or 15° mastery. Do not relaunch the previous 400M campaign based
on source-frontier values alone.

## Executive result

**The most useful experiment to date, but no overall new champion yet.**

- Restoring main's mature teaching **improved equal-age fixed generalist examination**: P1 B 0.7088 versus P1 A 0.6317. P1 B dynamic80 survival 96.9% versus 84.4%; both easy-core 100%. This is a *recipe-package* test, not isolation of the phase gain from the command envelope or athletic rehearsal.
- P1 C's **25% moving, manageable-disturbance retention** yielded better terminal on-policy average vx (0.0883m/s vs B 0.0513), lower on-policy fall rate (8.5% vs B 10.9%) and better simulated mechanical shaft cost (77.7 vs 96.0J/m). However, **C's fixed exam broad score (0.6786) and dynamic80 (92.2%) fell below B (0.7088 and 96.9%)**. C's 68.8% versus B's 65.6% resilience75 survival is only two worlds in a 64-world cohort; it does NOT prove retention prevented forgetting.
- The launcher's conservative P1 screening selected **P1 B (main-mature, no extra retention)** as ancestor for **all P2 arms**, verified from each P2 \`run_meta.json\`. Therefore **P2 did not test retention combined with coherent terrain**.
- At 60M **D/mixed gentle** led the frozen broad exam **0.7803** and resilience75 **85.9%**. **E/coherent gentle** scored 0.7590 and resilience75 78.1%. **F/coherent aggressive** scored 0.7505 and resilience75 81.3%. D had 100% easy-core and dynamic80 survival; E/F each 98.4% dynamic80, F only 98.4% easy core. These are the same fixed panel definitions, but not a dedicated directional step/grade or speed-conditioned shove exam.
- **Coherent terrain teaching clearly changed the lesson outcomes**: D ended at 0.35° and 1.5mm teacher frontier, while E and F both reached the **15° practice ceiling**, with step practice **28.0mm (E)** and **29.0mm (F)**. Unlike the prior millimetre trap, the coherent episodes yielded many supported far-side crossing successes. Nonetheless the near-height **UP** pass rate declined at the growing harder end; neither 15° nor 3cm is independently certified mastery.

## Comparable P1 fixed exams at 40M

| Arm | Broad | Dynamic80 survival | Resilience75 survival | Core vx MAE | Final training fall / actual vx / control J/m |
| --- | ---: | ---: | ---: | ---: | --- |
| A: older 360 recipe, gentle force | 0.6317 | 84.4% | 62.5% | 0.0136m/s | 11.4% / 0.0595m/s / 91.4 |
| B: main-mature, gentle force | **0.7088** | **96.9%** | 65.6% | 0.0144m/s | 10.9% / 0.0513m/s / 96.0 |
| C: main-mature + 25% retention | 0.6786 | 92.2% | **68.8%** | 0.0140m/s | **8.5% / 0.0883m/s / 77.7** |

All three had 100% fixed easy-core survival. Last-window on-policy data were gathered on DIFFERENT curricula and should not be interpreted as matched performance; frozen exams are more comparable, but sample sizes are limited.

## Comparable P2 fixed exams at 60M (all from P1 B)

| Arm | Broad | Dynamic80 | Resilience75 | Core survival | Core vx MAE | Target step / slope / push |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| D: mixed, gentle | **0.7803** | **100%** | **85.9%** | **100%** | 0.01418m/s | 1.5mm / 0.35° / 0.7N |
| E: coherent, gentle | 0.7590 | 98.4% | 78.1% | 100% | **0.01385m/s** | 28.0mm / 15° / 0.7N |
| F: coherent, aggressive | 0.7505 | 98.4% | 81.3% | 98.4% | 0.01511m/s | 29.0mm / 15° / 3.02N |

The force frontier is a **practice target**, not measured 3.02N force-conditioned recovery competence. The P2 last on-policy fall fractions are D 26.0%, E 26.5%, F 22.4%; this does NOT make F more reliable because task mixes and individual cohorts differ.

## Evidence underneath the much bigger terrain numbers

These are summed **completed training** evidence events over 40→60M, **not a held-out fixed-terrain exam**:

| Evidence over 20M P2 segment | D mixed | E coherent | F coherent+force |
| --- | ---: | ---: | ---: |
| Supported near-height step-UP successes/attempts | 27/66 (40.9%) | 14/39 (35.9%) | 17/40 (42.5%) |
| Supported near-height step-DOWN successes/attempts | 23/70 (32.9%) | 27/33 (81.8%) | 28/35 (80.0%) |
| Easier step-UP successes/attempts | 26/87 (29.9%) | 83/103 (80.6%) | 84/107 (78.5%) |
| Easier step-DOWN successes/attempts | 34/89 (38.2%) | 94/103 (91.3%) | 81/95 (85.3%) |
| Near-slope **full command tracking** successes/attempts, four directions pooled | 49/94 (52.1%) | 48/58 (82.8%) | 50/59 (84.7%) |

**Severity is moving across these windows**, so D's 40.9% uphill near-pass at ~1.5mm is not equal to E's 35.9% while height grows toward 28mm. This is evidence of better task coherence, not proof of a fair 28mm fixed-height pass percentage.

**Late-run diagnostic:** in the last ~5M, E's near-height step-UP evidence was only **3/22 (13.6%)** and F's **4/22 (18.2%)**; near-down was 3/6 (E) and 9/15 (F). The teacher reached an ambitious height target but **uphill mastery is still sharply limited**. The near-incline tracking data late in the run are also small; do not claim 15° command following from the angle dial.

### Important code-level confound remains

The coherent STEP/SLOPE lesson samples operator forward commands of 0.10–0.22m/s, then generic **auxiliary specialist attenuation** pulls them halfway toward +0.04, yielding approximately 0.07–0.13m/s. Meanwhile ascent-progress shaping in \`environment.py\` requires **cmd_vx ≥0.08m/s**. Some supposedly deliberate step lessons still fail this eligibility gate. This can be corrected in a **new reviewed source/experiment**; do not edit the completed source-pinned training checkpoint or silently treat previous success as independent proof.

## What we know / do not know

**Supported conclusions:** The 20M main-style nursery still worked; mature teaching improved fixed generalist walking at 40M; a dedicated moving retention curriculum produced attractive on-policy speed/effort outcomes but not higher broad held-out score; coherent step/slope teaching avoided the near-zero teacher target collapse and greatly improved easier crossing evidence; aggressive force did not clearly outperform gentle force on fixed broad outcome.

**Not supported yet:** No demonstrated permanent no-forgetting effect, no proven 15° slope following or ≥28mm step-up mastery, no actual 3N shove-recovery certification, and no controlled comparison to the 260M/232-input Pinnacle. Tiny held-out panel differences are not necessarily meaningful and a single seed is not replication.

## Next actions before any 400M campaign

- [ ] **P0 physical matrix first (no PPO steps):** use original SHA \`ab64cac\` and saved local checkpoint/policy exports to give D/E/F equal REAL up/down steps **3/5/10/20/30mm** and four incline bearings **0/1/2/3/5/8/10°**, each with actual signed requested vx, stable supported body travel, far-side crossing and post-cross survival. Include P1 B/C for regression baseline. Separate a *qualified* threshold from an exposed teacher target. Avoid guessing ability above the tested geometry.
- [ ] Test fixed same-speed no-push vs 0.5/1/2/3N pushes (plus 4N if suitable) on **at least two viable speed bands**. A rising difficulty number is not measured resilience; the F−E difference is not conclusive.
- [ ] Next design should **combine** coherent terrain with a demonstrably moving retention cohort, not simply continue E/F unchanged. Preserve deliberate forward crossing commands through the auxiliary sampler and align their reward eligibility; verify true physical lip crossings. Keep **easy anchor, near practice and harder exploration** separate from a graded competence map, especially for step-UP.
- [ ] Consider 2–3 small same-age comparisons: coherent + retention vs coherent without, with command attenuation repaired as its own controlled variable if checkpoint contracts permit. Preserve one 360 actor, one PPO; no scripted gait or second robot brain.
- [ ] Run the same physical observer with legitimate observation schemas for protected Pinnacle before any main replacement. Real champion promotion requires multiple seeds, clear speed-command progress and retained recovery at already mastered states.

**Source management:** This report and related handover/README links are documentation only; original training artefacts and protected \`main\` untouched. Source-pinned resume of the completed \`ab64cac\` learner requires an **unchanged exact checkout**, not the later polish branch HEAD.
