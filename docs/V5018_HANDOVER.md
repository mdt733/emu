# EMU v5018 — clean restart / handover

> **START HERE in a fresh chat.** This is an orientation document, **not a
> second progress checklist**. All dated evidence, `[x]/[~]/[!]/[ ]`
> statuses, acceptance gates, decisions and next work belong ONLY in
> [`docs/V5018_TERRAIN_PLAN.md`](V5018_TERRAIN_PLAN.md).
> For a genuinely blank Ubuntu WSL installation use
> [`docs/INSTALL.md`](INSTALL.md), first section, and
> `scripts/bootstrap_wsl_v5018.sh`.

## The project and non-negotiable boundary

- Repository: https://github.com/mdt733/emu
- Development branch: **`feature/v5018-terrain-awareness`**.
- The branch began from reviewed `main` commit
  `2ec9e9db0a816c658131311129ef89ec7a595a12` on 19 Oct 2026.
  It has several subsequent branch commits: **always fetch and record the
  current `git rev-parse HEAD` rather than trusting a SHA frozen in a handover**.
- **Explicitly approved:** inspect, implement, commit and push on this feature
  branch. **Not approved:** edit/merge into `main`, replace
  `showcase/champion.emupolicy`, publish Pages, or disturb another checkout's
  active GPU process, experiment or checkpoint root. Reviewed PR before
  any later user-approved promotion.
- Robot: ~0.5 kg dynamically unstable six-actuator ball-foot EMU; passive
  shin springs; 18 mm-radius ball feet. Simulated servo torque/damping/limits
  are **not measured hardware safety**.
- Preserve 220-input `pitch15_220` and 232-input `sim2real_232`
  actors and six outputs. The protected showcase is currently a
  **260,112,384-step schema-2, 232-input actor**; checkpoint schema 8.
  The 120M-step 220-input Pitch15 champion was historical, and caused the
  first clean-WSL compatibility-test failures. The schema-1/2 examiner
  compatibility patch is committed but its **full-suite rerun is pending**.
- v5018 is **sim-first research**, not a real-hardware deployment claim.
  PPO/control on a physical EMU has not been established by this effort.

## What we're actually building

Train a single neural six-servo locomotion controller to **anticipate**
raised/lowered steps, diagonal edges and low capsule logs using a
**physically plausible forward-looking ToF/range scanner** plus IMU/flow
odometry; preserve its proven flat gait, 232-input sensor convention,
v5017 navigation and joystick UX, and user-observed 10° slope traversal
in the published viewer.

The existing three downward tripod ToFs measure immediate local
ground grade. They **cannot simply be rebranded as look-ahead sensors**:
the new forward scan must cover left/right future foot corridors and
deliver timestamped first-hit rays. A small fixed-size rolling memory
reprojects those hits using inertial/optical-flow motion, with explicit
validity/age/confidence. Unknown elevation + confidence zero is **not
known flat terrain**. Step *faces*, logs and occluded/unsafe landings
are hazards, not invented walkable surface heights. Keep direct tripod
grade and confidence independent from rolling-map confidence.

The tentative 5-bin/two-lane height+confidence extension produces
`(58+20)×4=312` observations, **ONLY if P2 evidence chooses 5 bins**.
First compare 3/5/8 bins and physical range/coverage; do not silently
alter either existing actor contract, or freeze the new one prematurely.
All realistic actor inputs must come from simulated *legal sensors*.
Use exact terrain truth only for physics, scoring and an explicitly
labelled, non-deployable oracle upper bound.

## What had been proven before the big campaign

The locally uploaded `v5018_p0a_spike_v3_results.json` demonstrated
**5/5** physically supported constrained-ball contact fixtures
(flat, raised 30 mm box, genuinely lowered 30 mm shelf, diagonal ramp,
capsule log); CPU versus MJX final foot-centre heights within
**0.000590 mm (0.59 μm)**. CPU raycasts versus JAX box/plane/capsule
primitive first-hit intersections matched **12/12** tested rays
(within **0.000046 mm**); notably step risers, log top and side.
Two simultaneously batched same-course worlds landed at their respective
step and flat heights. **These are not dynamic whole-robot gait/force
or hardware tests.**

The uploaded `emu_v5018_batch_256_512.json` reported the actual
232-input EMU GPU environment:
- 256 worlds: flat **9,541** env-steps/s; added step **9,717**.
- 512 worlds: flat **15,290** env-steps/s; added step **16,690**.

**Caveat:** the step obstacle registered **zero foot contacts** at the
recorded final frame, so the comparison showed geometry-presence
overhead, not colliding-step workload. The GPU was `cuda:0` with
Python 3.13.16, JAX 0.12.0 and MuJoCo 3.16.0; these are
*environment-step*, not PPO gradient/update FPS. The GPU allocator's
reported peak is not total physical VRAM. The browser's
`@mujoco/mujoco` is now pinned to **WASM 3.15.0**, so actual
CPU/MJX/WASM parity remains an open gate.

The initial clean Python regression reported **73 passed, 5 failed**
against obsolete 220/120M policy expectations. Code was patched to
handle the current 232/sensor champion but a fresh full-suite PASS was
**not** established in this conversation.

The user reports the existing published champion *visibly* climbed
and descended **10° slopes** in the viewer. **Preserve and independently
grade that capability** rather than starting slopes at 1° as though
nothing was learned. Viewer observation is meaningful, not a measured
held-out contact-based training or hardware result.

## Current tested evidence and next continuation

**First large P0 archive:**
`v5018_p0_20261020_081806.tar.gz`, *tested commit*
`6a32841c73042ffa380d197e953f31b196f60857`.
**12/13 stages exited zero**, with one old Python regression stage
76/78 tests passed and two stale/runner issues subsequently patched
on the feature branch. Physically collidable plane/box/capsule
contact/raycast checks passed, including both log-side and riser
rays, moving ball-foot impacts, per-world obstacle pose variation,
static WASM/CPU comparison, 12 browser tests and 2048-world actual
foot-contact-loaded GPU **environment stepping** at approximately
24,533 env-steps/s. **Not PPO training FPS, proof of locomotion
over obstacles or real-hardware safety.**

**Second frozen-champion archive:**
`v5018_heading_20261020_090712.tar.gz`, *tested commit*
`d525ab662990405bc2ab6c8cd3e12cf14cdb60a7`,
protected champion SHA-256
`d3d319b82fbc5de331752f0f06753c57a776ac3212bcb6d386116c4f5f8a4862`.
**11/12 independent stages exited zero**; the lone failure was a
zero-course-axis exception message assertion, fixed on feature branch
*after the run*. The previously missing **real 232-input PPO update
smoke PASSED** (2 worlds, 2-frame rollout, one optimizer update).
All **80/80** frozen-champion 350-frame simulator trajectories
survived on matched real-plane flat/10° grades, reverse and turn
courses, with outer heading hold OFF/ON. **None of those
80 trajectories met the strict contact-qualified walking pass**,
including flat, even though measurable locomotion occurred. This
is NOT yet proof the actor doesn't walk; investigate the
contact event detector and genuine swing-foot lift/path.

Across matched seeds, OFF→ON `kp=.75` mean heading error:
flat **12.5→3.7°**, 10° uphill **20.2→3.9°**,
10° downhill **40.4→6.1°**, 10° cross-left **64.3→24.5°**.
Heading P helps orientation but **no 10° run passed the strict
course examiner**. The user previously saw 10° ascent/descent in
the viewer; the simulator survival and visual observation are
compatible with the new, more demanding failure on precise
heading/progress/contact-qualified walking.

**Hardware-realism concern:** some frozen-policy trials
spent **74–78%** of sampled servo channels above the *simulated*
0.638 Nm nominal continuous torque reference, sometimes
reaching the configured 1.913 Nm limit. This is a simulation
effort warning **not a measured winding temperature/current
or safe hardware duty cycle**. No physical hardware run is
authorized by this work.

**Third uploaded archive (20 October 09:49, fully reviewed):**
`v5018_gait_step_20261020_094915.tar.gz`, tested feature commit
`45f0655f687fa4628f0c9b002c5af202362e2896`, same protected
champion SHA-256
`d3d319b82fbc5de331752f0f06753c57a776ac3212bcb6d386116c4f5f8a4862`.
**All 15 stages PASS**, including **6/6 targeted Python tests**.
Frozen actor measured real collidable steps in copied XML, with
0mm/off-course-box flat control, 5mm step and 10mm step. Every
real-step variant physically registered actual ball-foot/box
contacts (5mm **914**, 10mm **1,031** world-frame occurrences
summed over its 3 configurations); no off-course-box contact
on flat.

**The exciting finding:** no terrain policy training needed
to see *some* physical threshold traversal:
- OFF heading correction (seed 1701): **0/4** strict
  fixed-world corridor crossings for each flat/5/10mm course.
- ON (kp=.75, seeds 1701+1702): **4/8 flat**,
  **5/8 with 5mm box**, **6/8 with 10mm box**.
- All failing pilot scores were **corridor_detours**,
  often torso drift 0.13–0.29m vs ±0.12m allowed.
  Successful scores required torso + both feet beyond
  actual obstacle far edge, genuine subsequent far-side
  support and finish runout. The contact counts prove
  collision interaction but do NOT prove an intentionally
  adapted high-step gait; spherical feet may bump/roll
  over shallow edges. Do not infer 10mm universally
  easier than 5mm; there are only two ON seeds and
  one OFF seed.
- Critically, strict contact-pair *walking* gait score
  remains **zero on flat** despite authentic liftoffs,
  touchdown transitions and airborne intervals.
  The new rejection counters show the
  **25mm torso motion required within each individual
  foot airborne interval** rejects nearly every
  touchdown on flat. A naturally short-step robot may
  fail this criterion despite useful movement.
  Do **not** edit pass criteria until per-event displacement,
  true support/foot travel/slip evidence is measured.
- Baseline actuator effort remains high (~74–76% of
  scored samples above simulated nominal continuous
  torque). No real hardware motor thermal proof.

**Branch implementation AFTER 09:49 archive (not WSL executed):**
- Added per-event real-contact airborne interval diagnostics
  including frame numbers, foot/body travel, clearance and
  p10/median/p90 distributions; existing conservative gait
  pass/fail rule is unchanged.
- Expanded the same immutable frozen-champion pilot to
  **0/5/10/15/20mm steps**, rotated step faces up to **±35°**,
  forward speeds **0.05–0.18m/s**, conservative projected
  near/far scoring bounds, spatial ball-foot-vs-box height
  diagnostics; still no NN terrain preview or reward changes.
- Added `scripts/run_v5018_obstacle_frontier.sh` with
  targeted unit tests, four real flat/10° directional-stride
  investigations, controlled two-seed height sweeps,
  ±25° face-angle trials, slower/faster 0.07/0.14m/s
  trials and heading OFF comparisons. Every experiment
  logs its own exit and actual crossing result.
- The event-probe, angle/height/speed extension and batch
  runner are **committed but not executed on user's GPU**
  yet. They may discover issues requiring targeted fixes.

**Fourth frontier attempt — partial, NOT a failed physics result:**
At 10:26 AEDT, user executed
`scripts/run_v5018_obstacle_frontier.sh` on
`c6a5dab5b0ea9b7c6d34db2b6e85599ae015314a`.
Protected policy hash remained unchanged; 4/4 targeted
Python tests and all four matched real flat/10° directional
studies (four worlds each, heading OFF/ON) completed.
Flat course passing 2/4→3/4; 10° uphill, downhill,
cross-left course passing 0/4 both modes. Heading
mean absolute errors OFF→ON 14.21→4.57°, 22.01→5.37°,
45.29→7.39°, 68.77→27.94°, respectively. The cross-left
yaw-rate error and slip worsened despite better orientation.
**No step-height/angle/speed frontier trials ran**: the
shell runner stopped with `code_y: unbound variable`
while expanding the first step trial filename.

**Corrected only on the feature branch after the attempt:**
fixed `code_y`→`code_yaw`, added
`--resume EXISTING_OUTPUT_DIR`, skip only
stage-PASS + extant logs/artifacts, original policy hash
verification, original and resumed commit history in
`revision_history.tsv`, and EXIT-trap tar.gz even if
an unexpected error occurs. This fix has not been
verified by a subsequent user-run yet. **Use the same
output directory and do not repeat the four GPU studies**.
This is a *mixed-commit resumed campaign* by design;
interpret logs with their stored revision history.

**Second resume attempt — archive reviewed, still no new
obstacle measurements:**
User uploaded `v5018_frontier_20261020_102641.tar.gz`
after a 10:43 resume on feature commit
`cad93f94fbb026da5feade1f15786af9fc25b8ee`.
The archive contains *only* the original five PASS stage rows
and four prior flat/10° heading study folders;
`campaign.log` confirms all five were skipped as intended.
**No step_*.json and no aggregate summary exist.** The runner
still had a second stale `code_y` reference inside the stage
filename despite the earlier assignment fix. This is a
Bash `set -u` orchestration error and must NEVER be
reported as robot failing 15/20mm steps or bad heading.

**New code after that uploaded archive:** on
`feature/v5018-terrain-awareness`, the second
`code_y` reference was replaced with `code_yaw`;
a preflight expands and tests **180 distinct experiment
names** prior to any GPU calls, and stderr is saved to
`runner_stderr.log` for future unexpected shell errors.
The real run and preflight share ONE `step_stage_name()` formatter
(commit `3c029c256614adf35f2db93d4dbc21a324a287fc`);
an isolated `bash -u` check of this exact formatter passed
all 180 combinations. The end-to-end WSL campaign remains unverified.
A standalone `bash -u` isolated naming simulation
covered 180 combinations and 3 representative edge cases
PASS, but the full user-WSL campaign has not run since
this latest correction.

**Important:** Resume the *same* saved directory
`~/robot_lab/emu_exams/v5018_frontier_20261020_102641`
on the newest feature branch. The original stride files
are from `c6a5dab...`; first resume recorded
`cad93f94...`; new step experiments will be
from a later commit. Preserve all three source identities.

**Completed frontier archive (20 October, 11:43–12:05):**
`v5018_frontier_20261020_102641.tar(1).gz` has 34/34
successful stage executions: four targeted tests and four
flat/10° heading exams from original commit `c6a5dab...`,
plus 28 real physical 0–20mm, ±25° face, speed and
heading-OFF/ON course variants from the later resumed commit
`69daa3d253cf92d0e3c40fbc060d54c4cbb549cf`,
and an aggregate report. Revision history is embedded;
**this is mixed-commit evidence**. Champion SHA-256 remains
`d3d319b82fbc5de331752f0f06753c57a776ac3212bcb6d386116c4f5f8a4862`.
All evidence is independently recorded with its source.

**Read-only objective results:** 49/112 fixed-world
course crossings across all variants (includes flat
8/20); 41/92 physical obstacle trials passed. For
straight 0.10m/s heading P-ON, *matched seed 1701/1702*:
flat 4/8, 5mm 5/8, 10mm 4/8, **15mm 6/8** and
**20mm 3/8**. One 20mm world fell/reset with a
heading reversal; no policy terrain training took
place. At 10mm face yaw +25° passed **7/8**,
yaw -25° passed **3/8**, a suggestive left/right
asymmetry. At 0.07m/s, both 5/10mm step courses
scored 0/4; at 0.14m/s, 5mm 2/4 and 10mm 3/4
(one seed). Fixed 7s/finish constraints may
confound the low-speed result.

**Major bottleneck:** **59/112** total worlds left
the allowed ±12cm lateral corridor; of 63 failed
worlds, 59 reported that reason (fail causes
can overlap). Heading P helps yaw pointing but
cannot hold lateral track by itself. Crucially,
course crossing proves movement of torso+both
feet with valid subsequent support; it does
NOT establish low-impulse, high-clearance,
torque-efficient deliberate step-over motion.
At 20mm, many foot-in-step-XY samples had ball
bottom below box top; side-face contact or
contact solver details need analysis.

**Historical "no real steps" alarm is now
well explained:** Four-world flat trace OFF
reported 196 genuine contact-bounded airborne
intervals; ON 197. Median torso travel **11.9mm
OFF / 11.2mm ON** during one swing, while
median foot travel ~25.6/23.7mm and
clearance ~15mm. **Zero out of all 393**
events reached the historical requirement of
25mm torso travel during a SINGLE swing,
even though 351/393 had ≥3 airborne frames,
≥4mm clearance and ≥12mm relocated foot.
Do not rewrite old results: keep old v1
walker scoreboard immutable and evaluate
a versioned alternative on fresh traces.

**New feature-branch code AFTER completed frontier
(not WSL tested):**
- `emu/directional.py` research-only v2
  `diagnose_alternating_contact_pairs_v2` and
  `gait_v2_research_only` JSON for original operator
  signed progress over TWO alternating supported
  foot intervals, checking true takeoff/landing,
  opposite-foot support, 3-frame air, 4mm
  clearance and 12mm forward foot relocation.
  Does NOT change original course/walking pass
  or policy/reward.
- `scripts/v5018_champion_step_pilot.py`
  now **omits euler=0 0 0** for straight
  courses, to preserve byte-identical original
  09:49 step XML; focused tests assert all
  three original 0/5/10mm XML SHA-256 values.
  Earlier seed1701 10mm result drifted from
  3/4 to 1/4 in the later frontier under a
  different course XML hash. This is a
  repeatability warning, *not evidence the
  zero-Euler token caused the policy difference*.
- Added synthetic v2 tests rejecting sideways
  walking, missing opposite-foot support, and
  backward foot relocation. These and XML
  hash lock remain to be executed on WSL.

**NEXT DEVELOPMENT GATE:** Instead of launching
another massive 28-course GPU study, run the
committed and still **unexecuted**
`scripts/run_v5018_gaitv2_acceptance.sh`:

```bash
cd ~/emu-v5018
git pull --ff-only origin feature/v5018-terrain-awareness
source .venv/bin/activate
bash scripts/run_v5018_gaitv2_acceptance.sh
```

It first validates the gait-v2 anti-cheat synthetic
tests and original 0/5/10mm exact XML SHA-256 hashes,
then uses the frozen 232 actor in only **two**
matched OFF/ON 0°/10° read-only directional
exams and two 10mm seed1701/1702 repeats.
Reports source/actor hash and old official
v1 vs *new research-only* v2 gait diagnostics;
does not change v1 pass, PPO or any reward.
Output archive:
`~/robot_lab/emu_exams/v5018_gaitv2_acceptance_*.tar.gz`.
If the foundational tests fail, it SKIPS all GPU
runs and archives diagnostics.

After evidence review, stage an **opt-in
physically correct P1 support/height/stance-slip
reward-only path** for collidable terrain,
and test exact old flat reward parity before
bounded PPO updates. Finish P0B motor-under-load
and moving-leg ToF occlusion/latency/dropout
and six-servo CPU/MJX/WASM parity before
freezing `terrain_312` or attempting long
terrain teaching. The full 34-case frontier
is ALREADY complete; do not run it again.

## New 12:48 gait-v2 evidence and cold-start architecture decision

**Uploaded and verified:** `v5018_gaitv2_acceptance_20261020_124851.tar.gz`,
exact tested source `622d515b352abae67b0065b643912f151c4afa78`,
protected champion SHA-256
`d3d319b82fbc5de331752f0f06753c57a776ac3212bcb6d386116c4f5f8a4862`
unchanged. **All five executable stages passed**: 4/4
synthetic/XML-hash regression tests, flat and 10°
uphill heading OFF/ON examinations, and 10mm
box pilot seed1701/1702.

- **Flat:** conservative official v1 qualified 0 strides;
  separately versioned research-only gait-v2 reports
  `[8,1,7,8]` candidate alternating pairs heading
  OFF and `[2,2,9,8]` heading ON in four worlds
  each. Course passes **3/4 OFF, 3/4 ON**.
  This agrees with observed genuine short stepping.
  **Do not silently replace historical v1 scores** or
  declare hardware-safe gait from these proxies.
- **Real 10° uphill:** both modes had **0/4
  course passes and 0/4 v2 candidate-pair positives**.
  Measured progress was nonzero; forward foot
  relocation and directional tracking were deficient.
  The new metric is not a magic fix for slope performance.
- **10mm independent repeat:** identical
  original XML SHA-256
  `f5ca6525a5d2ef8e608121d88fb613b5cc56e92b76bbbacba1b895ad17d0b754`;
  both seeds reproduced their original **3/4**
  crossing rates and identical per-world success
  masks. This controls one critical XML input;
  not proof of deterministic training across GPUs.

**New explicit user requirement:** keep the
project capable of learning from **absolutely
nothing**—random initial weights, fresh
normalizer/optimizer/teacher/capability field,
new seed, zero prior brain files—even as
short research experiments continue using
the frozen 260M champion. The champion remains
a convenient research control, **never a required
dependency, initialisation assumption, or
part of the future production training recipe**.

Code audit confirmed the current standalone
`emu train --profile foundation --sensor-actor
--run-dir NEW_ROOT --steps N --envs W --seed S`
already goes through `make_initial_state` with
random JAX seeded network and no parent when
`--resume` is absent. Likewise the 220 legacy
path is supported. New safeguards committed
AFTER the 12:48 archive (not WSL executed):

- `tests/test_v5018_cold_start_contract.py`:
  empty-run initialization of both 220 and 232
  profiles and sim2real opt-in, with no
  checkpoint load or implicit champion dependency.
- `scripts/run_v5018_cold_start_acceptance.sh`:
  OPTIONAL, bounded 8-world/256-environment-step
  fresh PPO-update, new checkpoint and
  `.emupolicy` export **for BOTH actor schemas**,
  with scratch results outside Git. It does not
  read the champion, is not a training plan, and
  is not proof of learning how to walk.
- `.gitignore` now ignores the four *existing*
  root-level early v5018 probe JSONs so Git's
  clean-provenance gate doesn't reject fresh
  `emu train`; **files are not deleted**.
- The living `docs/V5018_TERRAIN_PLAN.md` now
  requires every future new schema/terrain
  curriculum to pass BOTH actual cold-start
  PPO→checkpoint→export→browser and optional
  warm-init compatibility smoke. A future
  312-style actor remains provisional and
  **not implemented**.

**Next substantive development:** no need
for another 34-course frontier run. Build
opt-in collision-truth, terrain-relative
support/foot-clearance/stance-slip metrics
with EXACT legacy flat reward equivalence;
then mount/occlude and timestamp physical
lookahead rays and test estimator output.
Continue quick experiments against the
frozen brain where useful, but teach
newcomer networks flat/basic
walking before terrain, using a
capability-gated curriculum. Never make
future training rely on a previous trained
policy. `main`, protected showcase and
Pages remain unchanged.

## 20 Oct afternoon — P1 actual terrain truth implementation (NOT GPU run)

The user explicitly said **"Get into it"** and reminded us to
keep both the living checklist and fresh-chat handover current.
Work was done **only on `feature/v5018-terrain-awareness`**,
not on main, protected `showcase/champion.emupolicy` or Pages.
The 260M brain is a **reference for experiments**, never a
required training foundation. The cold-start tests and optional
one-update 220/232 fresh-policy lifecycle checker are retained.

**Code changes now on the branch:**

- `emu/terrain.py`: small pure NumPy/JAX physics-truth component,
  `box_surface_truth()`, handles a single fixed real yaw-rotated
  worldbody box above a plane. Combines *true geom-pair*
  near-contact evidence with contact geometry. Separates
  floor contact, conservative box-top contact candidates,
  vertical riser/side touch, unsupported feet. Reports
  body height relative to top when above it, signed
  ball-bottom top clearance, 3D contact-tangent stance
  motion and support fraction. A contact pair is NOT
  a measured normal force or physically proven load.
- `emu/environment.py`: new opt-in env args
  `terrain_contact_mode='legacy'|'box_surface'`
  (default legacy) and
  `terrain_box_geom='v5018_obstacle_step'`.
  Invalid/missing/moving/non-horizontal boxes and real
  inclined collision planes are rejected rather than
  incorrectly scored. Near the physical box, use
  top/floor contact support and corrected body/foot
  heights/clearance/slip for the existing reward.
  OUTSIDE its footprint/contact, a per-world mask
  preserves exact old legacy scalar economics.
  No changes to `_base_obs`, 220/232 policy
  shapes, PPO code, trained checkpoint or reward
  coefficients. Teacher-only `terrain_*` metrics
  are exposed ONLY as eval/rollout diagnostics.
- `tests/test_v5018_terrain_truth.py`: independent
  pure top-vs-riser, no-pair ghost stance, below-top
  signed gap, genuine tangent slip, yaw footprint
  and JAX-JIT/numpy agreement checks.
- `scripts/v5018_p1_surface_smoke.py`: actual
  JAX/MJX same-XML 4-world 0mm off-course flat
  control compares **bit-for-bit** old/new PPO
  rewards and identical 232 actor observations
  for three steps. Then a separate true 10mm
  geometry probe must activate finite terrain
  relative metrics without importing any policy.
- `scripts/run_v5018_p1_terrain_acceptance.sh`:
  independent fail-fast syntax, new physical-truth,
  cold-start regression and historical float32
  reward regression; then actual CUDA/MJX smoke.
  Writes all logs and errors to fresh
  `~/robot_lab/emu_exams/v5018_p1_terrain_*.tar.gz`.
  **NEW CODE UNTESTED IN THE USER'S WSL** at this
  handover; do not state PASS until archive arrives.

**Exact next WSL command after pulling branch:**

```bash
cd ~/emu-v5018
git pull --ff-only origin feature/v5018-terrain-awareness
source .venv/bin/activate
bash scripts/run_v5018_p1_terrain_acceptance.sh
```

If it fails, fix the exact failing case and rerun a
**targeted** smoke rather than the finished 34-stage
frontier or 5-stage gait-v2/repeatability campaigns.

**Strict remaining limits:** this first measurement
supports only ONE static yawed box on a level floor.
No normal impulse, actual electrical motor effort,
body-low terrain termination, steps DOWN, ramps,
logs, arbitrary per-world obstacle pose, full
six-servo browser parity or moving-leg forward-ray
occlusion has been proven or implemented in P1.
The legacy forward range estimator still assumes a
floor plane and is **not** a terrain preview for the
actor. `terrain_312` remains a hypothetical
future schema, not a frozen contract.
The new branch mode is deliberately **not yet
offered by public `emu train`** until it survives
the flat/riser/MJX gates and a clearly scoped
curriculum. The full future product must always
be teachable from new random weights without
a prior checkpoint.

**Authoritative current implementation status and
gate checklist:** `docs/V5018_TERRAIN_PLAN.md`.
Do not copy this prose into another parallel
progress tracker.

## 20 Oct 13:45 — P1 failure reviewed; challenge-ring viewer implemented

**User's latest actual WSL test, source SHA
`d683aa4f9dfa69bed53e2067886bae938a6ddfad`:**
uploaded archive `v5018_p1_terrain_20261020_134537.tar.gz`.
Python 3.13.16, JAX 0.12.0, MuJoCo 3.16.0,
RTX 4060 Ti 8GB; robot.xml SHA-256
`c1af72485e83de277efa1914a476519cebbebdc2beb1f03a10c475c9fadf3a64`.
**P1 syntax PASS, 6/6 true-terrain NumPy/JAX tests
PASS, 6/6 legacy/220+232 cold-start contract tests PASS.**
The independent GPU/MJX reward-parity stage **FAILED**:
`v5018_p1_surface_smoke.py` raised
`box off-course changed actor observations` while
comparing exact float32 actor observations.
This was a bitwise equality assertion with
NO printed numerical discrepancy. A previous
scalar reward equality check passed on the
frame leading to the exception, but the rest
of the three-frame experiment and active 10mm
contact probe did not finish. **DO NOT claim
P1 real GPU validated**.

The feature branch smoke has since been
modified (UNRUN in WSL) to record observation
world/frame/channel/difference and reject
material discrepancy. It still insists on
**exact scalar reward equality** and only
permits tiny `atol=2e-6, rtol=3e-6`
float32 differences in separately compiled
GPU actor observations. This is an
**unverified diagnosis**, not evidence
of a solved numerical physics discrepancy.
The updated diagnostic script writes a
machine-readable failure JSON as well as
the offending numeric channel to stdout
even if the next GPU attempt fails.

**Viewer request:** existing autonomous demo
draws a circle and chooses random destinations.
User requests coloured physical obstacles
lining that exact circle plus a second
autonomous demo that travels ordered waypoints
through every challenge, and live viewing
of the learning progression if possible.

**Committed viewer source changes (NOT yet
executed by Node/Vite/WASM/real browser):**

- New `emu/viewer_src/src/challenge-course.js`
  generates eight deterministic stations
  exactly on the EXISTING 4m circle radius;
  each step is a real collidable MuJoCo
  worldbody BOX, tier-selected 0/5/10/
  15/20mm and colour coded green/blue/
  amber/magenta. Optional orange CAPSULE
  logs are clearly experimental:
  no trained log-crossing claim.
  Each station produces tangential ENTRY
  and opposite-side CROSSING waypoints
  with an unambiguous world course;
  16 ordered waypoints total.
- `terrain.js` supports
  `withTerrain(xml,'course')`, retains
  real floor and clean physical geometry,
  leaving original flat/ramp/radial
  heightfield park untouched.
- `sim.js` mounts course XML in the real
  browser MuJoCo/WASM solver, while
  floating coloured names and station
  halos are purely render-only visual
  aids for tiny obstacles; all are
  re-created on changes of policy/terrain,
  disposed appropriately. A new
  **View: arena** camera preset frames
  the whole 4m challenge circle and
  auto-activates when course mode is selected
  from orbit; render-only, no policy changes.
  Compatible actor hot-swaps preserve
  selected terrain and demo mode.
- `main.js`, `index.html`, CSS show
  dropdowns in the existing mobile
  settings drawer: Terrain→Challenge
  ring; obstacle tier; optional log;
  **Demo route → random destinations or
  challenge circuit**. Circuit is an
  *autonomous movement demo* using the
  SAME normal policy action pipeline,
  not a privileged teleport or a
  terrain-crossing capability exam.
- `goal-navigation.js`, `goal-worker.js`,
  and `goal-planner-client.js` use
  existing background A* for exact
  caller-chosen circuit targets, never
  replace them with random points,
  and reset stale routes when the
  robot resets / human takes the sticks.
- New `tests/challenge-course.test.mjs`
  verifies physical XML, colours, circle,
  ordered entry/exit stations, async
  target request and UI wiring. The new
  `scripts/v5018_wasm_challenge_smoke.mjs`
  probes actual pinned browser MuJoCo
  3.15 static step/log contact and
  writes a JSON report; the setup is
  deliberately a constrained falling
  sphere, not six-servo validation.
- New fail-fast
  `scripts/run_v5018_viewer_course_acceptance.sh`
  installs locked npm deps, runs all
  viewer tests, tests actual WASM
  collision, Vite build and
  `emu viewer-build`, and archives
  `~/robot_lab/emu_exams/v5018_viewer_course_*.tar.gz`.
  An isolated pure JS generated-course
  check passed all 0–20mm levels,
  and a V8 parse check passed modified
  main/sim/terrain/nav modules; these
  are **not the full npm/WASM/browser
  end-to-end acceptance**.

**Two independent next commands from a
healthy WSL checkout:**

```bash
cd ~/emu-v5018
git pull --ff-only origin feature/v5018-terrain-awareness
source .venv/bin/activate
bash scripts/run_v5018_p1_terrain_acceptance.sh
bash scripts/run_v5018_viewer_course_acceptance.sh
```

If P1 still fails, do NOT allow that
failure to obscure independent Node/WASM
course results; the two stage runners
each produce an archive. For a
local static-policy demonstration after
successful browser build:

```bash
emu viewer showcase/champion.emupolicy --no-open
# Open http://127.0.0.1:8765/ and choose Challenge ring.
```

To watch real learning **only if a current
trainer is saving validated checkpoints**,
use `emu viewer --live --root
/path/to/existing/training/run --no-open`.
It polls saved checkpoints; **not live
MJX GPU positions** and does not
interfere with PPO training. No new
training lineage was started here.

**Status:** v5018 source branch only;
`main`, protected champion, public Pages
untouched. No terrain-related PPO training
or frozen new actor schema introduced.
Continue honoring the fully independent
fresh-random-brain cold-start requirement.
The living detailed status is in
`docs/V5018_TERRAIN_PLAN.md`.

## 20 October 14:43 — newest P1 MJX probe result and next correction

**Second user-run P1 archive:**
`v5018_p1_terrain_20261020_144341.tar.gz`,
tested Git SHA **`b93f0cf3d250deee2a39a38f164f96e3d58972b3`**;
Python 3.13.16/JAX 0.12.0/MuJoCo 3.16.0 on
RTX 4060 Ti CUDA, unchanged `robot.xml` SHA
`c1af72485e83de277efa1914a476519cebbebdc2beb1f03a10c475c9fadf3a64`.

**Observed:** syntax PASS, terrain truth **6/6** PASS,
legacy/cold-start **6/6** PASS. Crucially the
four-world **three-control-frame exact flat-reward
comparison and float32 deployable actor-observation
checks now completed**; no scalar reward error,
off-course terrain activation, or numeric
observation-warning message printed. Therefore
this 3-frame flat parity gate is substantially
better grounded than the older 13:45 failing
archive. There is no archived parity JSON from
that run because the script only wrote it on
full success.

**What interrupted next:** active 10mm box
probe called `state.data._replace(qpos=q)`.
In MuJoCo3.16/MJX `Data` is NOT a
NamedTuple; it exposes `replace(qpos=...)`.
The error was `AttributeError: Data object
has no attribute '_replace'` at
`scripts/v5018_p1_surface_smoke.py:146`.
**Not a model/physics failure.** Normal
EMU `EmuMJXEnv` code already uses
`data.replace()` everywhere else.

**Fixes committed AFTER the archive (NOT
YET WSL/GPU TESTED):**
1. Change the exact line to
   `state.data.replace(qpos=q)` (the
   correct installed MJX Data API).
2. Write successful 3-frame flat-parity
   status and full numeric drift
   details to `mjx_surface_truth.json`
   *before* active 10mm tests start, so
   any subsequent exception cannot erase
   completed evidence.
3. Use the actual per-world MJX
   right-foot sphere `geom_xpos`
   and physical box pose to relocate
   the free root: sphere is placed
   **1mm into the real 10mm box top**
   and tested after `mjx.forward`
   to within 1mm. It is a controlled
   CONTACT FIXTURE, NOT a plausible
   walk or stepping demonstration.
   The post-translation foot position
   is the slip reference, preventing
   a fabricated ~10m/s teleport-slip.
4. Log actual MJX foot/box **geometry
   contact-pair candidates**, top/riser
   fractions, supported stance and
   signed toe gap. Require at least one
   box/foot pair, not just torso XY
   overlap, before claiming real active
   obstacle collision evidence. A missing
   contact must be diagnosed, never
   awarded as a fabricated success.
5. Update the sole living status
   checklist `docs/V5018_TERRAIN_PLAN.md`.
   Public/Pages, main, champion and
   220/232 from-zero learning path
   untouched.

**Next WSL step:** pull latest
`feature/v5018-terrain-awareness`,
then run **only** the bounded P1 test
(or the separate viewer acceptance),
not the old 34-case campaign:

```bash
cd ~/emu-v5018
git pull --ff-only origin feature/v5018-terrain-awareness
source .venv/bin/activate
bash scripts/run_v5018_p1_terrain_acceptance.sh
```

**Independent viewer validation** remains
`bash scripts/run_v5018_viewer_course_acceptance.sh`
for the coloured real-box/log 4m ring,
ordered circuit route, pinned WASM collision
and Vite build. This has not yet been
validated by WSL/browser. Browser live
mode polls completed validated policies,
not live GPU robot positions.

## 20 October 15:44 — root-cause physics and viewer audit (UNTESTED new branch fixes)

**Actual user's latest successful P1 log:**
at branch commit `0f7f02b2eadb78ac8816d60f07619bde37463c77`,
four stages all PASS: syntax; terrain NumPy/JAX tests
**6/6**, old reward and independent 220/232 cold-start
contracts **6/6**; true CUDA 4-world **3-frame** flat
reward and all 232 observations **bitwise identical**
(max difference 0); controlled 10mm box contact
pair present **4/4**, geometry active 4/4, top
contact fraction 1.0, riser 0.0, support 1.0,
slip 0.0, signed min toe gap ~-1mm.
10mm course XML SHA `f5ca6525...`.
This proves the NARROW test passed and does
not prove terrain PPO or real stepping.

**Critical diagnostic root cause:** that
test TELEPORTS the free root so a foot is
deliberately 1mm inside the box, resets the
previous foot position to the new position,
then immediately reads the collision field.
Zero slip is **by construction**; synthetic
geometry proximity, even with a contact-pair,
is not normal load-bearing support or a
moving footfall. More dangerously, the code
used ONLY contact geom-pair and ball-centre
XY/Z to decide **top vs riser**. A foot
pressing against a step corner/riser can
have a near-top ball centre and the same
pair as a valid top landing. That could
silently credit wrong `support`,
`swing_clearance`, `stance_slip`, body
height and reward.

**New branch corrective work AFTER 15:44
(NOT YET EXECUTED IN WSL):**

1. `emu/terrain.py::box_surface_truth`
   now requires TRUE contact `frame`
   normal (first 3 float components)
   vertical relative to box top
   (abs cosine >= .88), and TRUE
   contact `pos` within 4mm of
   the horizontal top face/within
   rotated footprint. Geom pair,
   near-zero distance and ball centre
   Z alone no longer earn top
   support. Reversed normal sign
   is allowed because MuJoCo can
   swap contact geom order.
   No positive contact impulse
   was measured; still a
   conservative support candidate.
2. `emu/environment.py` passes real
   MJX `contact.frame` and `contact.pos`
   to the pure classifier; never
   passes these privileged truths
   to a 220/232 actor.
3. New tests in
   `tests/test_v5018_terrain_truth.py`
   cover an actual adversarial
   *near-top horizontal riser normal*
   and a vertical normal whose
   contact position is not on top.
   There are now 8 tests, NOT
   WSL-rerun.
4. `emu/ppo.py::evaluate_trace`
   gained a STATIC OPTIONAL
   `record_terrain_truth=False`
   research diagnostic. Default
   exam behavior/fields and all
   actor weights/inputs remain
   unchanged. The new tracing
   option logs frame-wise physical
   top/riser/floor candidate
   fractions, contact-active masks,
   scalar rewards and frozen
   actions during real moving MJX
   rollouts.
5. New `scripts/v5018_p1_dynamic_reward_audit.py`
   and fail-fast
   `scripts/run_v5018_p1_deep_audit.sh`
   actually execute the same
   immutable 260M 232 frozen actor
   twice on REAL 10mm MJX course,
   heading=.75, 0.10m/s, seed1701,
   4 worlds/240 control frames.
   Compare legacy-vs-opt-in top/
   riser/contact transitions,
   support/gaps, actor action,
   scalar reward, pose and
   terminal divergence. This
   DOES NOT train and is optional
   research only: from-zero
   learning remains a strict
   independent path. **No GPU
   test for this new code has run**.

**Separate real viewer user run,
source commit `0f7f02b2...`:**
attached `Pasted text(20261020-044353).txt`
contains direct console from
`bash scripts/run_v5018_viewer_course_acceptance.sh`.
Source syntax 3 stages PASS,
npm ci PASS, browser Node tests
**16/17 PASS**. The only
failure at `emu/viewer_src/tests/core.test.mjs:111`
was a **stale legacy assertion**
`doesNotMatch(...id="demoMode"...)`.
It erroneously prohibited the
new deliberate Random/Circuit
selection. Because of fail-fast,
**actual WASM collisions, browser
build and viewer installation
DID NOT RUN**; those status
are UNKNOWN, not passing.

**Viewer branch fix (NOT WSL rerun):**
the old core test now explicitly
requires the new random-default
and circuit modes and forbids
only obsolete multi-planner/
advanced-actor controls. Added
a geometry-aware pure Node test:
real existing pose-lattice
must solve routes through all
8 challenge obstacle footprints,
not merely accept their waypoint
strings. The exact source-level
lattice was executed in a V8
smoke and **8/8** obstacle
approach-to-exit routes both
solved and crossed box-XY.
This does **not** establish
real MuJoCo robot walking or
browser rendering success.

**Deep source audit expanded after the initial normal fix
(ALL corrections still awaiting WSL/MJX validation):**

- True support now requires **signed
  `contact.dist <= 0`**, not just a
  near-geometry pair at **+1mm/+2mm**
  positive separation. This is valid
  for the canonical *zero-margin and
  zero-gap* model; box_surface
  constructor rejects unsupported
  nonzero margin/gap semantics.
  Nonnegative normal-force magnitude
  is still not measured—do NOT call
  a near-zero candidate load-bearing.
- Previous terrain body-height reward
  followed the torso's XY footprint,
  even when both feet were still on
  floor. That also missed actual
  high-step support when the torso
  moved off the box footprint.
  New scalar `support_reference_elevation_m`
  is the physical raised-surface
  elevation averaged over the
  actually classified supporting
  feet. It is not load weighted.
  The legacy flat height remains
  exactly unchanged in default
  mode. Opt-in teacher-only
  `terrain_support_reference_elevation_m`
  exposes this choice for audit.
- The new adversarial terrain
  suite has **11 tests** (six
  previously run, five new and
  UNRUN in user's WSL). The moving
  P1 trace suite has **two new
  CPU tests** for foot-vs-world
  frame counting and excluding
  post-terminal contamination.
  The deep driver executes all
  these before engaging MJX/CUDA.
- The dynamic audit now logs
  support-surface height as well
  as top/riser normals and
  reward/action/physical trajectory
  divergence, and uses distinct
  world-contact-frame vs
  foot-contact-frame counters.
  The research-only `ppo.evaluate_trace`
  flag defaults OFF, so existing
  training, old scoreboards and
  232 actor observations are
  untouched.

**NEXT TWO INDEPENDENT WSL RUNS**,
current feature-branch HEAD only:

```bash
cd ~/emu-v5018
git pull --ff-only origin feature/v5018-terrain-awareness
source .venv/bin/activate
bash scripts/run_v5018_viewer_course_acceptance.sh
bash scripts/run_v5018_p1_deep_audit.sh
```

Both write archives in
`~/robot_lab/emu_exams/`.
Viewer checks true pinned MuJoCo
3.15 WASM step/log colliders,
Node and Vite build. Physics
runs new 8 adversarial tests,
true box contact-normal probe,
then a REAL moving 232 frozen
reference (not teleport-only)
on opt-in vs legacy reward
metrics. Send BOTH archived
logs/JSON, including failures.
**Do not re-run 34/34 frontier,
do not train from scratch just
to verify this research fix,
and never modify main, Pages,
protected showcase or active
GPU training.** The actor's
220/232 schema, new-policy
cold-start obligation and
unevaluated future 312 contract
remain unchanged.

Detailed canonical status:
`docs/V5018_TERRAIN_PLAN.md`.

## Challenge curriculum / anti-cheat design, clarified with the user

**Heading and command following come before giving crossing credit.**
Use the already-existing gyro-integrated heading P-controller as an
optional aid: it corrects **issued yaw rate**, never scripts feet and
never redefines the original operator command being graded. Run the
frozen 232 champion through matched flat and established 10° slope
tests with heading hold **ON/OFF**, maintaining the same original
command. Study heading, signed world-frame progress, yaw/reversal
tracking and falls; do not claim the controller enforces crossing.

Once P0 and physically correct terrain contact metrics pass, build a
small deterministic corridor crossing exam:
- Spawn on a stable approach side of a **real** step/shelf/log or
  diagonal grade, varying distance, approach yaw, lateral offset,
  footfall phase and requested speed (within measured mechanics).
- Require actual **signed world-course progress**, torso **and both
  ball feet** beyond the far edge, continuation/stable locomotion in
  the requested direction, and no fall/timeout.
- Detour beyond the corridor, 180° turn-away, perpetual hesitation,
  sideways escape, high-step reward farming and repeated touching
  do **not** count as passing; score against original commands and
  a frozen world-fixed goal/corridor. Corridor width derives from
  observed normal sway (±0.10–0.15 m is only a starting trial).
- Use obstacle truth only in the **evaluator and reward**; the realistic
  actor sees only its legal sensor estimate. First correct physically
  wrong support/clearance/stance-slip metrics **without** increasing
  bonuses. Then ablate **small bounded** approach/crossing/progress
  shaping, not a giant forced knee-lift reward.
- Cap difficulty according to dynamic servo/foot-clearance evidence;
  retain flat and previously successful slopes through every level.
  Start with modest easy lessons but permit a controlled minority
  to explore just beyond current capability; pass/fail advancement
  uses held-out crossing and retention, **not raw PPO scalar reward**.
- Compare matched A (masked/no preview), B (instantaneous scan),
  C (rolling map), R0 (archived champion), R1 (corrected-metrics
  retrained flat). D is **non-deployable oracle** and delayed until C
  architecture/comparisons are frozen. Use small staged training budgets,
  independent seeds and real observed crossings before promoting.

## Code ownership and order after P0

- `emu/robot.xml`: mechanical robot/collision model.
- `emu/environment.py`: batched MuJoCo/MJX physics, reset, observations
  and terrain-relative state.
- `emu/directional.py`, `emu/evaluation.py`: independent heading,
  metrics, physical crossing/course exams.
- `emu/rewards.py`: transparent bounded coaching terms.
- `emu/ppo.py`, `emu/training.py`: neural actor and updates, lifecycle.
- `emu/curriculum.py`: difficulty progression and physical capability.
- `emu/contracts.py`: schema/timing/hashes; **don't freeze 312 yet**.
- `emu/viewer_src/`, `emu/viewer_server.py`: actual browser
  MuJoCo/WASM, sensor emulator, release-preserving interactive viewer.
- `scripts/v5018_*.py`, `scripts/run_v5018_p0_campaign.sh`:
  feasibility tests, not the eventual teacher/evaluator architecture.
- `docs/V5018_TERRAIN_PLAN.md`: **sole source of status/gates**.

Order: **P0A/P0B physics, sensing, batch and engine-gap gates → P1
terrain truth + strict no-detour course + corrected-flat reference →
P2 3/5/8-bin sensor evaluation → P3 explicit compatible actor/lifecycle
vertical slice → P4 reward/curriculum anti-cheat audit → P5 short
matched GPU arms → P6 browser parity/challenge course → P7 release
evidence and user-approved PR.**

## Fresh-chat prompt

> Continue EMU v5018 strictly on
> `feature/v5018-terrain-awareness`
> at https://github.com/mdt733/emu.
> Read CURRENT branch
> `docs/V5018_TERRAIN_PLAN.md`,
> `docs/V5018_HANDOVER.md` and
> `docs/INSTALL.md` first.
> Preserve ONE living checklist
> TERRAIN_PLAN. Never touch main,
> the published 260M 232-input
> champion, Pages or an active
> training process without user
> approval. Champion is OPTIONAL
> for research; the full learning
> system MUST also build from
> random network/optimizer/
> normalizer/curriculum without
> ANY earlier brain or private
> .pkl. The 220/232 contracts
> are unchanged, terrain_312
> remains UNDECIDED.
>
> Important actual P1 evidence:
> 20 Oct 15:44 source `0f7f02b2`.
> All FOUR P1 narrow stages PASS;
> 3 control frames old-vs-opt-in
> flat scalar reward + 232 actor
> observations bitwise MATCH,
> 6/6 original terrain tests,
> 6/6 cold-start/reward tests,
> and four-world forced 10mm
> *stationary* foot/box geom
> near-contact PASS. But top
> fractions 1 and slip 0 came
> from artificially placing the
> foot 1mm INSIDE box and
> setting previous foot position
> equal to current. Not proof
> of dynamic supported stepping.
> ROOT DEFECT FOUND AFTER RUN:
> terrain.py classified BOX TOP
> contact from pair + foot-centre
> XY/Z alone; a riser near
> a step CORNER can satisfy it.
> Further bugs: +1mm contact
> candidates were treated as
> support despite zero margin;
> body-height reward followed
> TORSO XY instead of actual
> supporting FOOT contact.
> New source demands true contact
> frame NORMAL aligned with top
> plus real contact point ON
> the top face; adversarial
> false-top tests and a
> research-only moving-champion
> read-only A/B course audit
> were added, NOT YET RUN
> on user GPU.
>
> Separate viewer console at
> 15:43 on same source:
> syntax and npm ci PASS,
> Node tests 16/17; 1 FAIL
> due to old core.test
> mistakenly FORBIDDING
> id="demoMode" even though the
> user explicitly wants Random
> vs Challenge circuit modes.
> This was corrected to assert
> both modes + single planner
> and actor unchanged. All 8
> actual source-level lattice
> station crossing paths have
> been demonstrated in isolated
> V8 tests to intersect real
> obstacle footprint. NO WSL
> WASM collision/build/browser
> result yet: old fail-fast
> stopped before those stages.
>
> NEXT RUNS after pulling current
> feature HEAD:
> `bash scripts/run_v5018_viewer_course_acceptance.sh`
> and
> `bash scripts/run_v5018_p1_deep_audit.sh`.
> The P1 deep runner tests
> real normal+contact point
> semantics, flat reward parity
> and TWO actual MOVING 260M
> read-only 10mm MJX A/B
> rollouts; it is NOT a PPO
> training run. Archive partial
> failure JSON; do not report
> success before WSL evidence.
> Live browser still polls
> completed validated policies,
> not every training GPU world
> frame. Challenge circuit is
> navigation demo, NOT held-out
> strict crossing certification.
> Keep docs and handover current
> after each new archive.
