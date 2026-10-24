# EMU — active handover

**Branch:** `feature/v5023-capability-growth` · **Updated:** 24 October 2026  
**Design and permanent non-regression contract:** [Unified learning contract](docs/UNIFIED_CAPABILITY_GROWTH.md#non-regression-contract--what-must-never-be-traded-away)  
**Protected reference:** `main` at `2ec9e9db0a816c658131311129ef89ec7a595a12`

## Mission

**One cold-startable 360-input actor, one PPO optimizer, 2,048 MJX worlds.** Preserve the working **20M-step main-style foundation** (original gait physics, eight reward payments, 70-update steady-only bootstrap, conservative commands, growth/retreat and checkpoint safety), then grow coherent command, recovery, real slope, real step, transition/posture and economical control capabilities. All experiences train the **same** robot brain; specialist labels remain teacher-only.

The six mature emphases target **20/20/20/20/10/10%**. Every world still sees gentler doses of other challenges. Every completed episode adds directional, severity-aware evidence; frontiers expand only on enough real success and share one bounded growth budget. No invented step crossings or PPO reward masquerading as independent walking ability.

**Foundation→growth now implemented (unverified walking):** the unified 360 actor starts with main's command ramp and foundation resilience burden; its teacher-only specialist dose is zero through 20M, then blends in over ~2M steps weighted by earned steady difficulty. This controls role assignments, real slope/step births, and the switch toward evidence-backed command limits. Sensor schema remains 360 throughout; no 220-to-360 checkpoint migration.

## What is implemented

- **Learning:** one governor with the original four core difficulties, separate closed-window signed L1 command, four-direction slope and genuine step-up/step-down evidence. Specialist expansions now pass **one combined growth arbitration**, with full retreats and a PPO-health expansion veto. One `advance_curriculum` entry point handles normal and final rollout decisions.
- **Robot/world:** 6 actions, delayed **4×4 ToF array** and IMU/flow within a versioned **90×4 = 360** actor. Two real inclined solid collision slabs permit sampled pitch/roll, rise/drop and rotated edge. Old 220/232/360 actor replay is a separate preserved contract.
- **Single run:** `emu train` → `emu/run.py` → `training.py`; source-pinned plan, checkpoint/resume, reporting and viewer. `foundation_config` is intact. The public 260M Pinnacle and guarded Pages release remain protected.
- **Repository cleanup:** removed old ladder/lab5015, Pinnacle campaign+upgrade runners, multi-arm evaluator and directional-batch runner, their exclusive tests/scripts, oversized dated historical documents, and dead viewer stage sorting. Old code and research remain available in Git history; `main` was not changed. Active docs are README, this handover, the learning contract, INSTALL and PLAYGROUND. Tournament results replaced by a small single-run on-policy evidence report.

## Verified versus pending

- **Previous integrated engineering SHA `b7893fd`: four CI jobs passed** (Python, 12-world real MJX/PPO CPU, browser, clean Ubuntu viewer). The larger cleanup commits require their **own latest-HEAD CI**; don't describe old green checks as proof of newer source.
- **Not yet verified:** unified 2,048-world RTX 4060 Ti memory/PPO + checkpoint/resume/export; actual cold unified 20M first-stage gait parity vs main; accurate frozen **combined** slope+step+command+recovery examiner and browser parity for a freshly trained array-only actor.
- The unified trainer **intentionally fails closed**. There is no authorised ten-hour unified GPU campaign and **no trained unified successor** to Pinnacle yet. Existing frozen older 360 runs climbed poorly: supported ascent stayed 0/12 in matched +0.15 m/s exams despite better flat survival and descent. Do not label far-side root motion as an ascent.

## Live preservation checklist (details in the design contract)

**From main — preserve, not merely resemble**

- [x] Code-level **20M foundation** core: 70 steady-only updates, 32×4×8 PPO, LR2e-4, seed733/2048 envs, eight weighted gait payments, phase/command cap and effective early 0.70N push exposure. Existing `tests/test_learning.py` now checks source-level numerical parity. **Does not prove gait quality.**
- [x] Intended 360 actor born with same versioned realistic sensors; no specialist or hard physical terrain before 20M; gentle competency-driven fade thereafter. **Physical/training distribution only verified in small CPU smoke.**
- [ ] **Real first-20M parity:** a fully trained cold 360 actor must demonstrate support, achieved vx/vy/yaw tracking, stability, commanded pace and acceptable motor burden comparable to main on *matched flat frozen exams*.
- [ ] Verify that early **posture, command schedule, normalizer and capability-field shadow behavior** remain observationally equivalent enough to main; similar configuration values alone do not establish this.

**From our unified vision — earn, don't invent**

- [x] Six teacher emphases, one actor/optimizer, signed L1 command probes, direction-aware closed windows for slope/steps and shared expansion arbitration implemented.
- [ ] Demonstrate each teacher's **actual** task mixture and evidence attribution from completed episodes. Check underrepresented directions, sparse evidence, and whether successful diagonal commands wrongly expand unproven pure axes.
- [ ] Validate frozen array-only **19-case** physical exam on a real exported unified policy: all four signed slopes, rotated/paired supported up/down steps, mixed surfaces, commands, pushes, transitions, grip/slip, action changes and mechanical J/m. No such trained policy exists yet.
- [ ] Prove full **2048-world CUDA** memory/FPS, checkpoint/resume/export and viewer contract from one source revision; only then consider launch approval, first 20M and growth continuation.

**Never claim a replacement is better** using reward, a green CPU smoke, an examiner with zero trained-policy runs, or a single terrain case. Preserve `main`, approved 260M Pinnacle, public showcase and archived checkpoints. **Next action:** finish current-HEAD CI, then isolate GPU lifecycle and frozen policy examination. 

## Next actions — code before complexity

- [ ] **Check latest HEAD CI**, including the newly pinned numerical foundation contract, real physics and legacy replay. Do not claim another commit's green checks as validation. Fix real failures in existing modules.
- [x] **Code-level foundation parity:** align starting 360-actor command caps (0.40/−0.12/0.14/0.90), early 0.70N physical push teaching, steady-only 70 updates, LR, rollout, rewards, normalizer and early no-rollback behaviour with `main`; zero specialist/physical terrain dose until 20M, then gradual earned blend. **Real 20M walking parity remains unproved.**
- [x] **One array-only frozen physical examiner implemented:** `emu capability-exam --policy <new360.emupolicy> --output <new-dir> --quick --device cpu`. Fixed flat, four real directions, equal-height up/down contact slabs, inclined steps, command mixtures, start/stop, perturbations and shaft J/m. Reads frozen real MJX telemetry; **no trained unified policy has yet been tested**. CPU execution and scoring must pass CI, then verify on source-pinned GPU output.
- [ ] **Freeze one source and prove real CUDA lifecycle:** the existing `scripts/smoke_unified_focus.py --device cuda --envs 2048 --full-update` now performs a 32-frame, 4-epoch, 8-minibatch real 2,048-world CUDA PPO update without writing a training run. **Not executed here on the owner's GPU.** Then independently validate source-pinned checkpoint/resume → export → new `emu capability-exam --policy <new360.emupolicy> --output <new-dir> --quick` → local viewer. Only then request permission for fresh 20M and continuation.

**Never edit or merge `main`, promote/replace `showcase/champion.emupolicy`, overwrite historical pinned runs, or bypass the long-run launch guard without explicit approval.** Prefer improving existing `environment.py`, `curriculum.py`, `training.py`, `rewards.py`, `ppo.py` over creating another `v502x` subsystem. Update this file briefly when evidence changes; keep research in Git history.
