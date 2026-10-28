# EMU — active engineering handover

**Updated:** 27 October 2026  
**Release-polish branch:** `feature/v5024-release-polish`  
**Frozen GPU training branch:** `feature/v5024-terrain-control` at
`d26ae0c2c4097d7c7124759197a94504abc44995`  
**Draft review:** [PR #15](https://github.com/mdt733/emu/pull/15)  
**Public/main:** protected — **no merge, new champion or release without explicit owner approval**.

## v5027 — new independent feature-branch follow-up (28 Oct 2026)

- [x] Created `feature/v5027-coherent-retention` from the release-polish branch; **no main edits** and original `ab64cac` 140M v5026 data stays immutable.
- [x] Added isolated `--terrain-lesson coherent-direct`: focused STEP/SLOPE keep their sampled 0.10–0.22m/s forward intent rather than generic half-strength attenuation. Phase rate updated for this command. Existing `coherent` control remains unchanged.
- [x] Added physical post-sampler command assertion in CPU/full CUDA smoke (focused STEP exactly 0.10–0.22m/s).
- [x] Added `scripts/v5027_coherent_retention.py`, plan-only default; shared 20M seed735 cold start and 3×40M matched forks (old coherent, direct, direct+25% moving retention), full 2048 CUDA smoke, fixed 10M exams, final 0.15m/s matched CPU physical matrices and report bundle.
- [ ] **NEXT:** run clean branch on WSL using [one-page launch guide](docs/V5027_EXPERIMENT.md), preserve source SHA and collect resulting `overnight_results.zip`. The implementation has **not** trained on the owner's GPU yet.
- [ ] After results: assess real 1/2cm up/down, 3/5° directed grade, fixed core/dynamic/recovery, precision yaw, J/m; decide whether to combine, replicate or abandon. No 400M or public replacement on teacher numbers alone.

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

## Root-cause diagnosis after Experiment 1

- [x] Audited protected-main maturation against the exact 290M-screen
  source `ab29c027` and owner's complete training logs.
  [Root-cause investigation](docs/MAIN_VS_V5025_ROOT_CAUSES.md).
- [ ] **No 400M yet.** The 360 actor's 20M flat gait learned well, but
  the mature learner never adopted main's broader/stronger athletic
  command teaching. STEP/SLOPE cohorts inherit random incompatible
  commands/regimes and near-lip ascent reward requires vx≥0.08m/s
  before young step-focused worlds reliably request it. Real terrain
  certification retreats to 0.35°/1.5mm; 4N push target maxes by ~24M.
  Evaluate signed physical progress against Pinnacle with matched world
  before claims of superiority.
- [ ] Run isolated **main-parity mature-command** and **coherent-terrain**
  causal comparisons; test low-vs-high push interference. Preserve
  20M cold foundation, compatible policy/source lineage, branch-only rule.

## 28 October — genuine terrain learning was masked by an overly strict exam headline

- [x] Owner ran old frozen-source `ab64cac` physical matrix on all P2
  policies: `unified_exam(5/6/7).json`, eight matched physical worlds
  per terrain. All 3 survived 8/8 on flat; all strict 0/8 because
  zero-command yaw MAE approximately 0.115–0.126rad/s exceeds 0.08,
  despite mean signed yaw biases usually much smaller.
- [x] Verified **real supported five-frame crossings**: D 3/5/10/20mm
  = 8/8,8/8,4/8,4/8; E = 8/8,8/8,8/8,7/8;
  F = 8/8,8/8,8/8,7/8. Both E/F managed 3/4 step-UP
  at 20mm, versus D 0/4. The old strict `case_pass` remains 0/8,
  so do not say either number is false: the criteria are different.
- [x] E and F genuinely tracked forward vx across 3–5° inclines
  substantially better than D. D uphill-5° mean vx MAE ~0.263m/s
  versus E ~0.039 and F ~0.027; D downhill-5° ~0.183
  versus E/F ~0.017. These are equal-physics measured motion,
  **not** merely training slope targets.
- [x] Source-level exam presentation updated on development branch
  (not old `ab64cac`!) to report original strict all-axis score
  and a separate provisional physically meaningful mission score,
  true supported crossing and verified grade stability. The new
  mission score bounds mean yaw drift and yaw jitter, not a full
  heading-path exam. It does not rewrite any archived output or
  train any network. The archived old source remains exact.
- [ ] Repeat fixed 10/20mm and 3/5° results across different
  seeds/32 worlds before claiming robust mastery. Keep real uphill
  skill and generalist retention as independent selection objectives.
  Fix forward-step command attenuation only in a new source-pinned
  training version, never on saved P2 source.

## Completed v5026 recovery-first 140M audit — 28 Oct 2026

- [x] All seven seed-735 jobs finished, zero reported rollbacks;
  training source `ab64cac`, first 20M common foundation and six
  20M continuation trials; owner archive `overnight_results(3).zip`.
  See [v5026 findings](docs/V5026_OVERNIGHT_FINDINGS.md).
- [x] P1 B/main-mature improved 40M fixed broad exam **0.7088** vs
  A/control **0.6317**; C/mature+retention improved on-policy
  useful movement and J/m but didn't exceed B's fixed broad result.
  P1 B was the actual parent of P2 D/E/F.
- [x] P2 D/mixed gentle won 60M fixed broad **0.7803** and
  resilience75 **85.9%**, yet step/slope frontier retreated to
  **1.5mm/0.35°**. E and F coherent terrain ended at **28/29mm**
  and **15° teaching targets** with many supported training crossings,
  but worse held-out broad scores and very low late uphill near-height
  pass rates (~14–18%). These are NOT certified 3cm/15° skills.
- [ ] **Next priority:** run fixed physical terrain/force matrix on
  existing local P1 B/C and P2 D/E/F policy checkpoints *before new
  training*. Then combine coherent terrain with useful retention,
  preserving step command eligibility (coherent 0.10–0.22m/s is
  later attenuated toward 0.04; ascent reward needs ≥0.08m/s).
  No 400M launch until physical tests support competency and retained
  lower-speed recovery.
- [ ] Do not resume the archived `ab64cac` training source on the
  later docs-only polish HEAD; use exact-source checkout.

## Recovery-first capability-growth redesign (proposed)

**Current next mission:** [Competence-preserving growth plan](docs/RECOVERY_FIRST_GROWTH.md).
The owner's speed/resilience tradeoff hypothesis is not yet proven:
compare **same-speed, same-force** recovery at 20/50/70M and against the
published champion. We will not accept losing previously stable
movement/recovery merely to increase a frontier request.

- [ ] P0: frozen physical matrix; distinguish real forgetting from higher
  difficulty and from absent mature main commands
- [ ] P1: 20M cold birth, then three 20M matched continuations: current
  360 post-foundation control; main-mature speed/phase/rehearsal; same
  main-mature plus permanent trained-skill retention; **80M aggregate**
- [ ] P2: fork best 40M checkpoint into three 20M trials: old mixed
  terrain (low push); coherent STEP/SLOPE (low push); coherent
  STEP/SLOPE (rapid push growth); **140M P1+P2 aggregate**
- [ ] P3: keep the best simple architecture, test bounded velocity-I
  controller, repeat with fresh seed, then consider 400M only after
  proof of retained core skills and growing signed physical competence
- [ ] Do not confuse training frontier with qualified resilience or speed;
  no new 4N target promotion until force-conditioned held-out recovery.
  Preserve old physical and command abilities with retained-easy/near/stretch
  practice, and track separate certified/teaching/probe surfaces.

**No source code or GPU training was changed as part of this plan.**

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

## Next engineering work

- [x] Polish CI on `bf04f33e` was green: Python/MJX CPU/browser/fresh viewer.
  Later documentation commits need their own CI. CUDA is independently
  verified only for the original v5024 study SHA.
- [x] Owner's original three-arm 180M ZIP reviewed; all completed.
  Paired frozen terrain matrix and cross-schema benchmark remain outstanding.
- [ ] Compare frozen generalist scores AND separate supported/stable versus
  commanded/accurate uphill, downhill and camber physics, true step crossings,
  recovery, yaw/lateral drift and energy per *useful* distance.
- [x] **Experiment 1 290M completed and audited** from owner's
  `overnight_results(2).zip`; exact source `ab29c027`, all twelve 70M
  forks finished, no reported rollback. See
  [Experiment 1 findings](docs/EXPERIMENT1_FINDINGS.md).
  Staged OFF best broad exam 0.7343; balanced I 400ms/2× 0.7231 vs
  balanced OFF 0.7139. **Not physical mastery**: all twelve step
  frontiers at 1.5mm; most slopes at 0.35°; resilience all 4N;
  commanded speed frontiers outrun observed motion.
- [ ] Experiment 2: FIRST common real physical matrix on frozen 70M
  (staged OFF, balanced OFF, balanced I 400ms/2x, I 800ms/2x);
  THEN 2-4 isolated short teacher/controller repairs. Fix 0.35° slope
  and 1.5mm step retreat-floor traps; separate real resilience recovery
  evidence from merely requesting 4N. Keep raw commanded motion as
  the criterion for assisted control; browser parity before publishing.
  Preserve checkpoint source provenance with reviewed migration if needed.
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
