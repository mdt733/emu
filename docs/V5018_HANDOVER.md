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

## What's currently running / still UNPROVEN

As of the last conversation (20 Oct 2026), the user had launched the
**combined P0 feasibility campaign**, and we were awaiting its output.
Do **not** mark any of the following as passed without the uploaded
archive: complete updated pytest suite, npm CI/build/Node tests, GPU
doctor/PPO smoke, MuJoCo/WASM gap, moving toe strikes, per-world
`geom_pos`, scanner coverage, kinematic clearance, contact-stressed
256→2048 scaling. **The terrain policy has not been trained.**

Command for a future **new** run only (do not restart an active run):

```bash
cd ~/emu-v5018
source .venv/bin/activate
bash scripts/run_v5018_p0_campaign.sh
```

It writes a dated directory and matching `.tar.gz` under
`~/robot_lab/emu_experiments/`. The shell script continues independent
tests after a stage failure; individual `stages.tsv` results and logs
must be inspected. Only the GPU batch benchmark stops increasing world
count when the current size fails. It contains a small PPO-update smoke
but **no long terrain training run**.

When the user's archive arrives:
1. Read the **actual** branch `docs/V5018_TERRAIN_PLAN.md`, current Git
   status/HEAD, and the archive's `environment.txt` commit SHA;
   distinguish the revision being tested from any later documentation edits.
2. Inspect `stages.tsv`, failing logs, JSON traces and independent ground
   truth. Review the real GPU memory and collision-loaded throughput.
3. Fix genuine failures only on the feature branch; execute CPU/static
   checks wherever possible, provide user-run GPU commands only for blocked
   GPU work, and keep P0A/P0B gated until evidence actually meets exit
   criteria. Update the **single** living checklist and commit.
4. Do not ask to rerun a 7-minute suite or all 2048-world probes merely
   because a standalone stage failed; target the observed failure.

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

## Fresh-chat prompt (paste as-is)

> Continue EMU v5018 on `feature/v5018-terrain-awareness` in
> https://github.com/mdt733/emu. Read
> `docs/V5018_HANDOVER.md`, `docs/V5018_TERRAIN_PLAN.md` and
> `docs/INSTALL.md` from the **current branch** first; inspect actual
> branch HEAD, source and the newest test archive. The only living
> checklist is `docs/V5018_TERRAIN_PLAN.md`. Work/test/commit/push
> **only on this feature branch**; leave `main`, released champion,
> Pages and any active WSL training untouched. The previous P0 v3
> contact/ray probe passed; 256/512 environment-step benchmark passed
> **without obstacle contacts**. The subsequent combined P0A/P0B
> campaign was running; its results must be checked before any gate
> promotion. Preserve the existing 220/232 actor and the viewer's
> user-observed 10° slope ability. Next: interpret the campaign's
> stage logs, fix actual failures, finish P0 gates, then build a
> corridor/no-detour physical crossing evaluator with matched heading
> assist ON/OFF before terrain training. Treat reward/scanner/PPO
> success as hypotheses until physically measured. Update the one
> branch checklist after each substantive pass and never claim
> an unexecuted test passed. Do not ask me to repeat information
> already captured in this handover.
