# EMU capability-growth charter and running handover

**Canonical design and running checklist for `feature/v5023-capability-growth`.**
This branch starts at tested `main` commit `2ec9e9db0a816c658131311129ef89ec7a595a12`. Do not edit or merge `main` without explicit approval. Do not copy the experimental v5022 learner wholesale.

## The mission (do not drift)

Grow **the range and quality of situations in which one cold-start EMU policy can follow its commands, remain stable, recover, and move with mechanical grace**. A better result is a wider *reliably competent* capability envelope, not one highest speed, step height or reward.

One physical world → physically plausible sensors → actor → PPO → reward teacher → independent examiner → evidenced capability map → one curriculum director. Keep their responsibilities separate. Retain the accepted PPO, rewards, curriculum, capability field, evaluations, training health, rollback, checkpoint and resume machinery of `main` unless a measured issue requires a reviewed change.

### Rules for every capability

1. **Physical definition:** identify the variable, units, limits and *actual* effect on collision, forces, commands or sensor measurements. A labelled "step" with no MJX contact is not a step.
2. **Actor honesty:** only realistic, delayed/noisy measured inputs enter the actor. Truth about future terrain, precise foot identity, contacts and task success belongs exclusively to teaching/diagnostics.
3. **Teaching:** preserve command-following and locomotion reward while teaching supported progress; don't let circling, reversing, freezing or falling count as a completed crossing. Safe refusal/avoidance is a separate future commanded behaviour.
4. **Evidence:** separate success (completion, support, accuracy), grace (impact, slip, smoothness, effort) and confidence (enough independent examples). Task success and safety gate efficiency; low power from doing nothing is not success.
5. **Growth/retreat:** earn small frontier increases through demonstrated performance, sufficient samples and healthy PPO/core performance; retreat failed dimensions quickly. Use a **shared expansion budget**, with local retreat where possible and global intervention for core collapse.
6. **Coverage:** keep steady/transition/resilience/posture and diverse vx/vy/yaw commands in the mix while adding terrain. All training-exposure allocations sum to a bounded budget. Most lessons remain ordinary walking until competence justifies wider coverage.
7. **Truthful examinations:** score fixed *held-out* difficulties not controlled by the training frontier; publish curves plus a quality-aware coverage summary, never hide weak dimensions inside an average. Exams must not directly teach or advance curriculum.
8. **Safe evolution:** cold start remains possible on a new PC. Version observation/physics/checkpoint contracts. Same seed and model size for A/B arms, preserve failure visibility and true resume lineage, never overwrite champion/main as a shortcut.

### Capability families

| Family | Expandable variables | Mandatory evidence |
| --- | --- | --- |
| Commands | forward/reverse vx, vy, yaw, combinations | tracking, progress, support, falls |
| Changes | acceleration, deceleration, reversals, command hold/ramp | overshoot, recovery time, falls |
| Resilience | push size/direction/duration; perturbation at birth | regain requested movement, not mere survival |
| Posture | chassis pitch/height and foot-midpoint attitude | body stability and command tracking |
| Continuous ground | uphill/downhill/side slope, crest, changes of grade | same-world collision, traction, tracking |
| Discontinuous ground | step up/down, ledge/log/gap, relative height/width, approach direction | physically pass requested corridor alive and supported |
| Sensor uncertainty | noise, lag, dropout and self-occlusion | stable behaviour, realistic return distribution |
| Grace / hardware | actuator force/shaft-work proxies, slip, impact, jerk, safety | cost **conditional on actual task progress**, separately reported |

Do NOT make a dense eight-dimensional grid. Preserve the existing command/recovery/transition capability field. Add sparse conditional context slices (ground category/severity and quality) only after evidence and runtime measurements justify their value.

### Foundation and lifecycle

- The **first 20M environment steps** follow `main`'s foundation on physically flat ground. No early real steps, including buried obstacles accidentally producing contact or ToF returns. Existing command/frontier, posture, resilience and health training remains enabled. The optical sensor may observe flat ground and moving legs if comparing from cold start, but its semantics must be truthful.
- Step/slope lessons open **no earlier than 20M AND only after core competence passes**. Frontier height/angle and exposure share begin small, increase by evidence and retreat promptly. Preserve easy-world practice throughout. A failed step must not automatically lower a healthy unrelated vx frontier.
- Stronger combined challenges appear only after their component skills are measured. Do not increase all axes simultaneously. Later add safety/refusal as an explicitly commanded higher-level choice, not a PPO loophole.
- Choose a champion on independent capability coverage, quality and weak-area regressions. No policy promotion from a positive training reward alone.
- Prefer one 20M cold-start foundation **forked into same-size visible/blinded terrain-learning arms** to isolate sensor benefit. Preserve optimizer, normalization and curriculum lineage on fork; new observation field values can be introduced gradually without changing width.

## Ownership: as few coherent modules as possible

- `robot.xml`: robot hardware/physics.
- `environment.py` + small `terrain.py` when warranted: real batched world, control, measured actor packets.
- `rewards.py`: bounded teaching only.
- `ppo.py`: same general learning algorithm.
- `curriculum.py`: one director, global growth/retreat budget, skill allocation; retain `RegimeGovernor`, `TrainingHealth` and `CapabilityField`.
- `evaluation.py` and `results.py`: independent exams and capability evidence.
- `training.py` and existing checkpoint code: single lifecycle; do not clone its entire trainer for a v-number.
- `viewer_src/`: diagnostic viewing, never a substitute for MJX contact evidence.

## Permanent stage checklist (update this SAME file each working session)

The stage gates are **serial**. Mark complete only with a linked commit/test/evidence; a feature that looks plausible is not a pass.

- [x] **G0 — decide the principles.** Charter accepted in conversation, rooted in verified green `main` (`2ec9e9d`); keep older v5022 as an inspectable experiment, not the new base.
- [ ] **G1 — clean foundation + useful viewer.** Port only browser diagnostics from v5022 onto `main`; preserve all learner physics, rewards, PPO and recovery. **Viewer/browser checks passed**; approved 232-input champion previously exposed five outdated schema-1 exam failures. The versioned examiner and five real tests have been repaired on this branch, with full Python CI still the acceptance gate. Never hide failures or alter champion bytes.
- [ ] **G2 — evidence contract.** Report independent task success, observed quality, conservative evidence confidence, unknown bins, core-regression checks and source separation. **Implemented and focused CI green:** `results.py` emits read-only existing-field evidence, compares matched held-out `fixed_eval.csv` rows and refuses missing/mismatched protocol inputs. Existing `metrics.csv` and `envelope_summary.json` retain the teacher's decisions and regime-allocation trace. **Still requires full green CI and verified exam-protocol identities before gate close.** No PPO/reward/governor changes.
- [ ] **G3 — physical terrain spike.** Two truly different batched MJX worlds (flat and a finite raised step), validate stepping/collision, floor and foot sensor agreement, JIT speed/memory and fixed model topology. No training yet. Include direction/angle and retreat from step.
- [ ] **G4 — bounded terrain frontier.** Per-world height/approach/exposure with 0–20M flat, competency-gated introduction, shared growth budget, local retreat, preservation of command/resilience practice. Test correct checkpoint/rollback/resume of the frontier; no secret floor-height oracle to actor.
- [ ] **G5 — versioned 4×4 observations.** Use provisional 0mm forward / 70° down mount, verify bracket clearance, physically honest foot occlusion, queue/age/validity/noise, actor size and schema. Test blind arm matches architecture.
- [ ] **G6 — foundation/fork PPO gates.** CPU real optimizer, 256-world and full 2048-world CUDA shapes, true checkpoint resume, 20M foundation with core performance reviewed *before* fork.
- [ ] **G7 — controlled expansion.** Matched sensor-visible/blinded terrain arms, growing/retreating height and exposure, preserved other frontiers, shared-budget audits and fixed-difficulty evaluation. No large overnight run before G0–G6 evidence.
- [ ] **G8 — multidimensional independent exams.** Core, commands, resilience, slopes, steps, transitions, grace/servo proxies, held-out combinations, confidence and regressions; select next experiment by weak coverage rather than maximum single-dimension score.
- [ ] **G9 — review/promotion.** Clean diff, tests, reproducible cold install, public viewer/export compatibility and explicit human approval before touching `main`.

### Work-session protocol (the cross-chat memory)

1. Read **this file first** and check current branch, clean working tree, CI and latest commit. Treat unfinished checkboxes as unfinished.
2. State which single gate is being advanced and its acceptance evidence. Choose branch-only minimal code, test it, and keep baseline behaviours intact.
3. Before ending, **edit this very checklist**: mark proven checks only, add 1–4 sentences of concrete outcomes/blockers/next task under Session log below, with commit/CI identifiers. Any future chat can continue from that.
4. If a gate fails, stop promotion and record it. Do not paper over failure by changing a protected champion, reducing test coverage, lowering acceptance criteria or silently disabling a physics feature.

### Session log

- **2026-10-23 — reset decision:** The old `feature/v5022-forward-terrain` built a useful 4×4 ray viewer but accumulated an experimental parallel terrain learner and unresolved PPO smoke failure. The new branch starts at green `main` and will only reintroduce code that meets the charter. Next: complete G1, then build G2 before training physics.

- **2026-10-23 — G1 viewer-only restart implemented:** Created `feature/v5023-capability-growth` from green `main` (`2ec9e9d`), draft [PR #14](https://github.com/mdt733/emu/pull/14). Brought over **only seven `viewer_src` files** (4×4 square ToF, 0mm/70° controls, ray-count costs, physical obstacle previews and associated JS tests). No `emu/terrain.py`, training policy contract, trainer, PPO, reward, checkpoint or v5022 experimental environment changes. Browser and Ubuntu CPU-viewer passed in CI run `39920103547`; Python suite was still in progress when this log was written. The earlier v5022 branch/PR #13 is retained for forensic reference.
- **2026-10-23 — G2 evidence reporting partially implemented:** `results.capability_evidence_report` distinguishes observed success, quality, conservative lower bound, adequate evidence and **unknown bins**; explicitly separates training from held-out sources. `capability_field_evidence_snapshot` generates report-only `capability_evidence.json` from `main`'s existing command/recovery/transition evidence; none of these summaries enters PPO or curriculum. Tests added to existing `tests/test_learning.py`. The report does NOT claim servo electrical energy, independent examination, new terrain skills, or an integrated performance score. CI for the latest commits remains to be confirmed.

- **2026-10-23 — G2 exam regression guard:** Added a read-only `compare_fixed_exam_evidence` and `python -m emu.results --compare-fixed REFERENCE.csv CANDIDATE.csv --reference-protocol ID --candidate-protocol ID --output comparison.json`. It catches loss of core survival/support/velocity/yaw/pitch/height quality even when high-difficulty survival improves. Identical externally verified frozen exam/physics IDs and equal cohort/sample counts are mandatory; unknown is not a pass or a zero. Thresholds are provisional regression alerts, not statistical confidence intervals or reward payments. New tests verify boundaries, missing evidence and command behaviour.
- **2026-10-23 — baseline examination repair:** A full Python CI run `39926669293` found **79 passed and five inherited failures**: schema-1 examiner tests had been using the now-approved 232-input schema-2 champion. Added strict, versioned 220/232 reader and matching MJX evaluation, preserving the old 220 path; updated physical and tamper tests to inspect the true current 260,112,384-step champion. Added a fast focused CI step **without omitting the full suite**. Focused gate, browser and CPU viewer passed on `e69240f` (run `39928751958`); full Python suite was still in progress when recorded.
- **Next G3 acceptance design:** After G1/G2 are *proven* green, build a tiny MJX physics spike with two physically distinct worlds of **identical topology** (flat/no active riser vs a real finite raised step), verify 0mm/variable-height MJX contacts with a settling probe and that ToF beams see the same geometry, then benchmark 256/2048 worlds. No curriculum, PPO architecture or new learner may depend on G3 until this proof passes.

### Next-session entry point (read before touching code)

1. `git switch feature/v5023-capability-growth && git pull --ff-only`. Inspect PR #14, the exact latest SHA and full CI. Do NOT resume v5022's abandoned learner or touch `main`.
2. Check the **fast capability/exam gate**, browser and CPU viewer, then the *full* Python suite. If all pass, mark G1/G2 accepted here with the exact workflow SHA. Any remaining failure must be recorded and corrected, not ignored or suppressed.
3. Independently verify the frozen **exam protocol fingerprint** before comparing old/new `fixed_eval.csv` across policy generations. This comparison must stay read-only, not silently feed the governor or select a champion.
4. Only when G1/G2 are accepted, start **G3: two-world physical MJX step spike** (same topology, distinct real contact, honest ToF, memory/throughput). No fresh full training-run command until G6 proves cold start and GPU memory.
5. Record source SHA, accepted/failed tests, next single actionable task in this same document. Keep one PPO, one governor, one trainer, one examiner.

