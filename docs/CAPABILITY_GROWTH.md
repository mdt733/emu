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
- [ ] **G1 — clean foundation + useful viewer.** Branch from `main`, port *only* helpful viewer/terrain visualisation without any v5022 PPO, observation, reward or training forks. Confirm browser build and `main` baseline CI; document any unported viewer shortcomings.
- [ ] **G2 — evidence contract.** Define minimal uniform `task outcome / quality / confidence` records, core-regression checks, training/exam separation, curriculum decision trace. First test on existing command/transition/recovery data *without changing rewards or PPO*.
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
