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

**NEXT USER-RUN (when not sharing GPU with other training):**

```bash
cd ~/emu-v5018
git status --short
git pull --ff-only origin feature/v5018-terrain-awareness
source .venv/bin/activate
bash scripts/run_v5018_obstacle_frontier.sh --resume \
  ~/robot_lab/emu_exams/v5018_frontier_20261020_102641
```

The script resumes a `~/robot_lab/emu_exams/v5018_frontier_*.tar.gz`
containing per-experiment source/actor identities, exact stride
displacement and crossing logs and JSON. Upload the archive even
if some stages fail. **Do not rerun the previous three complete
batches.** The sole authoritative historical measurements
and gates are in `docs/V5018_TERRAIN_PLAN.md`.

**Still P0B blockers:** dynamic, torque-limited true swing
clearance and hardware energy budget; realistic moving-leg
forward-scan occlusion/dropout/IMU+flow drift; identical actual
six-servo contact course CPU/MJX/WASM transfer. P1 training
physics has not yet been corrected for terrain-relative support,
swing, stance slip and reward. The current 0/5/10 crossing
pilot demonstrates partial pre-existing physical obstacle
capability, **not a trained terrain-aware 312-input actor**.
Do not advance P1 learning or promote a champion based on these
small samples. Preserve the existing 220/232 interface, main,
the showcase, Pages and any ongoing GPU run.

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

> Continue EMU v5018 strictly on `feature/v5018-terrain-awareness`
> at https://github.com/mdt733/emu. Read the actual
> `docs/V5018_HANDOVER.md`, `docs/V5018_TERRAIN_PLAN.md`,
> and `docs/INSTALL.md` from the CURRENT feature-branch
> Git HEAD. Three 20 October archives were reviewed:
> 08:18 P0A GPU/contact probes, 09:07 frozen 232 champion
> heading 10° physical-plane ON/OFF, 09:49 real 0/5/10mm
> obstacle crossing/stride diagnostics. Current physical
> frozen policy actually crossed 5mm (5/8) and 10mm (6/8)
> box obstacles with heading P ON, 0/4 OFF each seed1701,
> using no-detour objective world-fixed course scoring;
> many failed solely on lateral corridor drift. These
> are small frozen-policy tests and do not prove a terrain
> NN preview or true high-foot stepping. Strict gait
> qualification remains zero on flat because the 25mm
> per-airborne-interval torso travel threshold rejects
> nearly all genuine touchdown cycles. Do NOT relax
> the score without quantitative per-event evidence.
> The first frontier run (10:26 AEDT, original commit
> `c6a5dab`) passed 4/4 Python tests and four flat/10°
> heading OFF/ON diagnostics, then stopped on a Bash
> `code_y` variable typo. A first resume at 10:43 on
> `cad93f94` correctly skipped all five stages and
> archived the results, but another stale `code_y` in
> the filename stopped it before ANY frontier step cases.
> The latest feature fix eliminates both variable
> references, adds a 180-case no-GPU naming preflight,
> and logs unexpected runner stderr. Execute:
> `bash scripts/run_v5018_obstacle_frontier.sh --resume
> ~/robot_lab/emu_exams/v5018_frontier_20261020_102641`
> after pulling feature branch, and send its new archive.
> Both early attempts collected ZERO new height/angle/
> speed experiments; preserve revision_history.tsv.
> Update the ONE authoritative living checklist
> `docs/V5018_TERRAIN_PLAN.md`. Keep experimental
> source/actor SHA evidence. Never change main, champion,
> GitHub Pages or a running training process. Avoid
> premature terrain_312 freeze or huge PPO runs before
> physically correct terrain-referenced metrics,
> realistic forward sensors and hardware load constraints.
