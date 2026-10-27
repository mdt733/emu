# EMU — active engineering handover

**Updated:** 27 October 2026  
**Release-polish branch:** `feature/v5024-release-polish`  
**Frozen GPU training branch:** `feature/v5024-terrain-control` at
`d26ae0c2c4097d7c7124759197a94504abc44995`  
**Draft review:** [PR #15](https://github.com/mdt733/emu/pull/15)  
**Public/main:** protected — **no merge, new champion or release without explicit owner approval**.

## Mission, in one paragraph

Build **one cold-startable, continuously teachable six-actuator biped**
using real MJX contacts, delayed 4×4 forward ground sensing, **one 360-input
actor with six actions**, one PPO optimiser and completed physical episode
evidence. Six lesson emphases (commands, resilience, slopes, discontinuous
steps, transitions/posture and control/efficiency) share that actor.
Difficulty advances, holds or retreats from local directional near/easy
evidence, not reward, elapsed steps, unrelated curriculum gates or optimism.
The actual goals are accurately commanded, stable, recoverable and
mechanically sensible walking. A frontier is a *training request*, never
certified competence.

**Design contract:** [UNIFIED_CAPABILITY_GROWTH](docs/UNIFIED_CAPABILITY_GROWTH.md) ·
[Physical target inventory](docs/CAPABILITY_DIMENSIONS.md).

## Completed v5024 180M study — source frozen

Owner-provided `overnight_results(1).zip` confirms **three completed
cold-start arms × 60,030,976 world steps**, zero PPO rollbacks, valid
final checkpoints and passed CPU + 2,048-world CUDA preflight, from
`feature/v5024-terrain-control` at
`d26ae0c2c4097d7c7124759197a94504abc44995`.
Frozen 60M broad scores: **A/main-balanced 0.642,
B/command-heavy-balanced 0.594, C/command-heavy-steering 0.666**.
A did better on growth110 survival (57.8% vs C's 48.4%);
C tracked difficult yaw better (0.376 vs A's 0.548 rad/s MAE).
Final taught slope limits remained about 0.9° and step heights
0.25–0.34 cm; the true terrain-matrix and Pinnacle common physical
benchmark are **still missing**. No winner is declared.

The original GPU checkout and run root
`~/robot_lab/emu_runs/v5024_cold_steering/` remain archived and
should not be touched by cleanup or future scripts. A checkpoint from
that exact source may resume only on that exact clean commit.

**Current graduation path:** [Two experiments + one long run +
Pinnacle cross-schema benchmark](docs/GRADUATION_CAMPAIGNS.md).

## What v5024 changes — measured facts versus assumptions

- The physical slope practice target advances from actual completed
  supported **moving stability** on each of four incline directions.
  Full `vx/vy/yaw` command tracking is still a *different*, stricter
  result; stable downhill drifting does not count as correct steering.
- Commands use forward/reverse and ±lateral/yaw physical frontiers;
  steering attention is an experiment, not a second network.
- Slopes use actual oriented collision surfaces. Steps require real
  far-side *foot support*, not root position past an edge.
- Nonfocus worlds practise gentler cross-training; control/grace cannot
  gain credit by standing still to minimise shaft effort.
- The fixed read-only `emu capability-exam --terrain-matrix` measures
  1/2/3/5° uphill, downhill and cross-left/right slopes and
  0.3/0.5/1/2 cm up/down steps, reporting support, full tracking and
  directional drift separately.
- The local browser viewer already includes adjustable **real MuJoCo**
  inclines and steps. It is useful for manual diagnosis, not a substitute
  for matched frozen MJX exams.

PPO/cold-start quality, reliable slope tracking and superior generalist
walking **remain experimental**. CI cannot prove any of them.

## Release-polish branch — completed engineering work

Forked at the exact frozen training SHA. In [draft PR #15](https://github.com/mdt733/emu/pull/15):

- Replaced stale README/install instructions and renamed current CI wiring;
  added a concise documentation index and explicit promotion checklist.
- Removed the superseded v5023 260M launcher. Made the **polish copy**
  of `scripts/v5024_campaign.py` self-contained, with its own output root
  `v5024_polish_study`, source guard and maintained three-arm plan test.
- Added independent guarded public approval for **360-input schema-9
  array-only** 4×4 physical sensor policies, while keeping the existing
  approved 232-input Pinnacle release unchanged.
- Removed the untrained zero-action browser fixture and fallback; the
  static build requires a trained candidate. Added release integrity
  regression tests for 360 array geometry/normalizer contract, provenance
  and current Pinnacle approval.
- Removed superseded experiment write-ups from the active docs; their
  procedures and results remain in Git history.

**Do not resume a training checkpoint from `d26ae0c2` on polish.**
Even documentation commits change the required source SHA.

## Current architecture & what to preserve for now

```text
robot.xml                         physical six-actuator robot
environment.py + contracts.py     MJX, sensors, observations, episode facts
rewards.py                        foundational locomotion incentives
ppo.py + training.py              shared actor and PPO optimisation
curriculum.py + capability_focus.py
                                  adaptive local evidence and lesson allocation
run.py + training_config.py       source-pinned run/resume/extension
evaluation.py + results.py        physical exams and reports
viewer_server.py + viewer_src/    policy export/serve/replay, browser MuJoCo
scripts/v5024_campaign.py         explicit three-arm research campaign
```

The currently published **232-input Pinnacle** is still the benchmark and
rollback asset, **not** a forever-supported second product. Its 220/232
actor/export/checkpoint paths and the corresponding tests are temporary
until we have **approved a better 360 actor, verified browser parity, and
actually switched the public showcase**. Then retire those paths and
historical CLI/test machinery in one audited migration; preserve old bytes
and source provenance in a tagged release/Git history.

The old demonstration has not been replaced, and GitHub Pages is a separate
manual/owner-approved workflow.

## Next engineering work

- [x] Polish CI on `bf04f33e` was green: Python/MJX CPU/browser/fresh viewer.
  Later documentation commits need their own CI. CUDA is independently
  verified only for the original v5024 study SHA.
- [x] Owner's original three-arm 180M ZIP reviewed; all completed.
  Paired frozen terrain matrix and cross-schema benchmark remain outstanding.
- [ ] Compare frozen generalist scores AND separate supported/stable versus
  commanded/accurate uphill, downhill and camber physics, true step crossings,
  recovery, yaw/lateral drift and energy per *useful* distance.
- [ ] **Revised Experiment 1:** 20M shared source-pinned foundation
  plus 12 × 20M same-source checkpoint forks (260M total), including
  balanced/steering/terrain/staged and six velocity-I gain/ramp
  combinations. Launcher: `scripts/v5025_screen.py --run`.
  New training controller commands are boosted but teacher/reward
  success uses the original operator velocity; no cross-source reuse
  of the archived v5024 60M checkpoint.
- [ ] The former A/C seed-734 80M replication is superseded. Its
  original launcher remains separately available but not the current
  experiment target.
  Two-arm runner: `scripts/v5024_campaign.py --arms A C --steps 80000000
  --seed 734 --root ~/robot_lab/emu_runs/v5025_repeat_seed734 --run`
  from a **separate clean polish checkout**. Both train I=OFF; replay
  with velocity-I settings 0.2/0.4/0.8s × 1.5/2× after checkpoints
  (examiner micro-sweep **not yet implemented**). If useful, Experiment 2
  enables I during training and scores raw operator requests separately
  from corrected internal policy commands.
- [ ] Experiment 2: selected recipe OFF vs one terrain-control intervention
  ON at seed 735, 80M each; decision depends on Exam 1.
- [ ] Run 3: single fresh-seed 736, 400M cold-start candidate.
- [ ] Verify cold training, 2,048-world full PPO update, same-source
  checkpoint lifecycle, policy export and actual 360 browser replay.
- [ ] Review remaining production call sites and tests, then prune
  legacy-only paths when the public champion has migrated.
- [ ] Decide on a public **LICENSE** before calling the project
  open-source; public visibility by itself does not grant reuse rights.
- [ ] Explicitly review/approve PR to main; separately approve any
  new policy publication. No automatic promotion.

**Operating rule:** maintain only the code that still serves the active
360 learner or its necessary pre-promotion safety net. Prefer deleting
obsolete experiment machinery over preserving it indefinitely. No
second PPO, no invented gait gates, no fake success telemetry, no
unreviewed edits to `main`, showcase or the frozen GPU checkout.
