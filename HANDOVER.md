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

## In-flight 180M experiment — do not perturb

The source-pinned v5024 experiment compares three **independent cold
starts** on the same seed 733, 2,048 CUDA worlds, and 60M steps each:

| Arm | First 10M | Following 50M |
|---|---|---|
| A · main_balanced | Main-style foundation | Balanced six-way growth |
| B · commandheavy_balanced | 90/5/5 command-heavy foundation | Balanced growth |
| C · commandheavy_steering | Same command-heavy foundation | Steering-focused growth |

The dedicated checkout is `~/emu-v5024`. The data root is
`~/robot_lab/emu_runs/v5024_cold_steering/`; expected consolidated bundle:
`overnight_results.zip`. The experiment owner controls the WSL GPU.
**Completion/result status has not been observed in this chat.**
Do not claim which arm wins until its logs and held-out exams arrive.

The original v5024 launcher first runs 12-world CPU and full
2,048-world CUDA PPO preflight, logs source/runtime manifests and 10M
frozen generalist evaluations, and permits only **exact-SHA/config/root**
checkpoint resume. It still imports reporting helpers from the
old tournament script **on that original training branch**. Do not
delete or update files in `~/emu-v5024` while any run is active or
resumable; polish is entirely separate.

See [V5024_RUN_PLAN](docs/V5024_RUN_PLAN.md) for the unchanged study
design and [INSTALL](docs/INSTALL.md) for safe WSL commands.

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

- [ ] **Latest polish CI green** (Python + MJX CPU + browser + fresh viewer).
  Do not assume CUDA acceptance from CPU CI.
- [ ] Collect original **three-arm 180M logs, metrics, source SHA and ZIP**.
  Check missing/failed runs before ranking.
- [ ] Compare frozen generalist scores AND separate supported/stable versus
  commanded/accurate uphill, downhill and camber physics, true step crossings,
  recovery, yaw/lateral drift and energy per *useful* distance.
- [ ] Repeat the promising recipe(s) at seeds 734/735; avoid selecting
  from one stochastic seed or inflated curriculum limits.
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
