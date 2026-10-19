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

**New branch code after the heading archive (NOT YET RUN):**
- Corrected the exception-message unit assertion.
- Kept the existing conservative walking score intact and
  added **reason-level diagnostic counts** for liftoff,
  touchdown, airborne time, clearance, foot relocation and
  torso progress in the read-only course evaluator, plus
  focused synthetic tests.
- Added a named obstacle-contact mask to
  `ppo.evaluate_trace` for evaluation ONLY; actor input,
  checkpoint shape, policy weights and reward functions
  are unchanged.
- Implemented an independent frozen-policy
  `scripts/v5018_champion_step_pilot.py` that copies
  the robot XML, adds physically collidable **5mm or 10mm**
  steps (plus an off-course-box flat control), and scores
  genuine 2-foot-plus-torso course completion inside a
  world-fixed corridor, with actual ball-to-obstacle
  contacts recorded. No training or auto-promotion.
- Batched these into
  `scripts/run_v5018_gait_and_step_pilot.sh`: target the
  previous regression, collect exact stride rejection
  evidence on flat/10° grades, run frozen actor ON/OFF
  through small steps across two seeds, and generate a
  `.tar.gz` and compact summary.

**NEXT USER-RUN after current branch update:**

```bash
cd ~/emu-v5018
git status --short
git pull --ff-only origin feature/v5018-terrain-awareness
source .venv/bin/activate
bash scripts/run_v5018_gait_and_step_pilot.sh
```

Output: `~/robot_lab/emu_exams/v5018_gait_step_YYYYMMDD_HHMMSS.tar.gz`.
Send that archive, even if a new probe fails. The new scripts
are committed, not yet WSL-tested. **Do not rerun either
previous big campaign** just to address a stale test message.
The live detailed evidence and next gate status remain
exclusively in `docs/V5018_TERRAIN_PLAN.md`.

**Still open:** dynamic servo under-load clearance, attached
forward scanner moving-leg occlusion/dropout/odometry and
full six-servo MJX–CPU–WASM commanded trajectory equivalence.
The new step experiment is a **frozen-policy baseline**, not
a newly trained terrain controller. The 220/232 exports are
preserved; `terrain_312` is not frozen. No changes to main,
showcase, Pages or a running GPU training process without
explicit permission.

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

> Continue EMU v5018 exclusively on `feature/v5018-terrain-awareness`
> at https://github.com/mdt733/emu. Read the actual
> `docs/V5018_HANDOVER.md`, `docs/V5018_TERRAIN_PLAN.md`,
> and `docs/INSTALL.md` from the *current feature-branch HEAD*.
> The 20 Oct 08:18 big P0 and 09:07 frozen-champion heading
> archives have been reviewed. Physical primitive collisions/rays
> work at up to 2048 EMU GPU environments, but these are
> NOT verified terrain walking; the frozen 232 champion's
> 80 tests all survived while strict gait-qualified passes
> remained zero, and heading P (.75) markedly reduced yaw
> error. A real 232 PPO smoke passed. The currently
> committed next test is
> `bash scripts/run_v5018_gait_and_step_pilot.sh`,
> producing strict stride-rejection diagnostics plus true
> 0/5/10mm box-step no-detour exam JSON. No output has
> been reviewed for this new script yet; interpret real
> crossing scores rather than script exit status.
> User observed 10° viewer slope capability; preserve it
> while diagnosing actual controlled gait. Keep source
> SHA and champion hash in every archive. Update only
> the ONE living checklist `docs/V5018_TERRAIN_PLAN.md`.
> Work/commit only on feature branch: never alter main,
> showcase, Pages, or active training; do not freeze
> terrain_312 until physical forward scanner studies;
> do not claim unexecuted tests passed.

