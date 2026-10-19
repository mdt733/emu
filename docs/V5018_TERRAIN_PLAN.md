# EMU v5018 — Terrain-Aware Stepping: living branch checklist

**Last evidence review:** 20 October 2026. **Ref:** `feature/v5018-terrain-awareness`. **Status:** P0A partial / P0B partial; P1–P7 gated. **Never merge main, replace a champion or deploy GitHub Pages without explicit approval.** This is the **only tracked living v5018 progress/gate checklist**. The full reviewed master-plan acceptance criteria are condensed below; [V5018_HANDOVER.md](V5018_HANDOVER.md) is a stable, practical restart guide, not a second progress log. [INSTALL.md](INSTALL.md) is the reproducible setup reference.

## Reviewed technical contract (source: master plan Review 2, 19 Oct 2026)

This section preserves the **original technical requirements and measured
exit conditions** inside the sole living plan. It describes what *must*
be tested, **not what has already passed**. See the dated evidence
section below for completed/unexecuted status.

**Goal:** A six-servo, dynamically unstable ball-foot EMU that can
anticipate and negotiate low physically collidable step-ups/step-downs,
diagonal edges/cross-slopes and rounded logs from two compact,
confidence-rated *physically observable* forward ground profiles. Keep
flat gait, the current useful 10° slope behaviour **observed by the
user in the published viewer**, and v5017 manual/autonomous controls.
The 10° observation deserves a matched frozen-policy physical exam;
it isn't a certified held-out test result or physical hardware proof.

**Hard limits:**
- Work **only on the feature branch**; never modify `main`,
  `showcase/champion.emupolicy` or Pages without user approval.
  Don't move an active training checkout or overwrite old run roots.
- Keep **legacy 55×4=220** and **sim2real 58×4=232** policy interfaces
  and six actions loadable, with their original normalizer/history,
  timing, actuator and XML identities.
- Simulated true surface/contacts can drive scoring and simulated
  sensor rays, but **must not leak as perfect terrain actor inputs**.
  Perfect terrain input is a separate, explicitly non-deployable oracle
  arm, delayed until the real estimator design is frozen.
- Original tripod three-downward-ToF **instantaneous** grade, height
  and validity/confidence remain separate from the aged rolling map;
  a three-point fit across a discontinuity can be fictitious slope.
  Map unknowns use zero value **with zero confidence**, never "flat".
- Keep code understandable: reuse physics/reward/contract owners,
  fixed-shape bounded JAX buffers, parameterized tests and no CPU
  callbacks inside MJX rollout. No SLAM/global obstacle mapper.

### P0A — physical feasibility exit
- Confirm exact installed JAX/MJX/MuJoCo/CUDA device and benchmark
  identical-training-path **flat versus contact-loaded** 256→512→
  1024→2048 worlds where 8GB VRAM allows. Record first JIT time,
  steady physics env-steps/s, actual PPO update steps/s separately,
  allocator peak versus total GPU VRAM and fail reasons.
- Verify true contact *geom pairs, normals/load-bearing forces or
  impulses, dynamic toe strikes, swing-vs-stance classification*;
  do not treat ball-centre Z or plane height as adequate support.
  Match CPU MuJoCo, MJX-JAX for box top, riser, lowered shelf,
  diagonal support, capsule log, far floor and occlusions.
- Use collision **and** first-hit ray geometry consistently.
  MJX hfield collision support **does not** imply JAX hfield raycasting.
  Prefer measured exact plane/box/capsule primitives or a small
  course bank over invented oracle distance fields. Probe MJX
  per-world `geom_pos`/size changes, fixed topology and recompiles;
  avoid assuming arbitrary terrain meshes batch cheaply.
- Gate remains **open** until actual collision, ray visibility,
  solver stability and realistic training-sized GPU performance
  are measured. If not feasible, document fixed-course fallback.

### P0B — mechanical, forward sensor and engine boundary
- Determine both **unconstrained kinematic** and **dynamically
  achievable** swing clearance (joint range, ball radius, actuator
  lag/rate, torque, spring/support, self-collision); derive *candidate*
  obstacle heights only from evidence, not a casual 20 mm guess.
- Test scanner class/mount/field of view: forward/downward
  multi-zone or a handful of fixed angled rays, possibly a moving
  scan if truly needed. Beams must cover **both future foot lanes**,
  including chassis bounce, ±pitch/roll, time-to-impact, occlusion
  from robot legs and sensor dropout. **Ten map bins do not imply
  ten simultaneous beams**. Measure natural foot-lane excursion
  before fixing lane widths.
- Pin the actual Node/npm/browser MuJoCo/WASM version and test
  **identical real six-servo pose/command/contact trajectories**
  across CPU MuJoCo, MJX and WASM. Preliminary constrained-ball
  final height alone cannot establish parity.
- Robot-in-hardware policy deployment is **not demonstrated**;
  v5018 is a sim-first milestone with later guarded hardware
  tests for sensor FOV/extrinsics, delays, gear lash, servo
  voltage/current/thermal/torque and safe toe impacts.

### P1 — physical terrain integration and the anti-escape examiner
- Introduce a minimal verified course family to **real MJX training**:
  a shallow raised/lowered shelf, tilted/asymmetric approach and
  rounded obstacle only if P0 permits. Correct `_floor_signed_heights`
  and dependent support, clearance, phase, swing, stance slip, foot-mid
  pitch/height and termination in the new mode. Preserve old actor
  semantics and compare old-vs-corrected flat metrics frame-by-frame.
  Include a **separately retrained corrected-flat baseline R1**.
- Build a deterministic **no-detour crossing test** with a
  **fixed world-frame start, course axis, crossing obstacle and finish**.
  Predeclare corridor based on measured flat sway (provisional
  ±0.10–0.15 m), timed forward *signed progress*, body **and both
  feet** beyond obstacle, resumed motion, stability, real contact,
  load/servo cost. Fail/withhold credit for stepping around its ends,
  sidestepping outside lane, turning back, hesitation, looping,
  "touch obstacle" farming, and perpendicular escape.
- Heading assist is an **already available gyro-integrated yaw P
  controller**, to be tested with the *frozen 232 showcase policy*
  on actual flat and previously observed ~10° ascents/descents,
  **heading hold ON vs OFF** with matched commands and seeds.
  It corrects *issued* yaw rate, never scripts foot movement.
  Always score against the **original operator command** and
  world-frame destination, not a corrected request that hides drift.
  Keep heading-off/perturbed robustness assessments so a policy
  isn't dependent on a perfect external controller.
- Gate progression on held-out **crossing + flat/slope retention**,
  not reward or millions of steps alone; retain some familiar
  flat terrain and measured easy slopes in every training tier.
  Allow limited next-frontier exploration once baseline following
  is satisfactory; don't demand mathematically perfect walking
  before introducing any small challenge.

### P2 — physically observed sparse rolling estimate (before bins)
- Coordinate datum: X=0 is torso's gravity projection onto
  *estimated local support plane*; +X is gravity-levelled heading;
  heights are relative to local support elevation, not raw torso Z.
  Re-anchor changes on ascent/descent coherently and lower confidence
  as needed. Two lane centres near Y=±0.08 m are *hypotheses*,
  not exact future swinging-foot placements.
- Timestamp actual ray returns, compensate IMU attitude and optical
  flow/gyro odometry, retain a fixed-size recent-hit ring, group
  by lateral tolerance, preserve discontinuities (no indiscriminate
  smoothing), and classify step faces/log sides as hazards.
  No-return, stale, out-of-FOV, occluded, dropout and noisy contacts
  need explicit calibrated low-confidence states.
- Compare **3, 5, 8 bins per lane** with measured mm height/edge error,
  missed-step/false-positive rate, physical time-to-impact preview,
  confidence calibration, lane misses, latency, memory and CPU/GPU
  compute under slopes, steps, reverse, sidestep and pitch variation.
  Height normalization proposed ±0.10m is **not frozen**.

### P3 — policy/lifecycle and opt-in explicit schema
- Decide bin count/reference/units/clipping and temporal strategy
  **before** freezing schema. Only for *5 bins/lane*,
  `terrain_312=(58+10 heights+10 confidences)×4`.
  Other bin counts require independently named schema/dimension.
- Authoritative SHA-256 contract includes field order, mounts/beams,
  bin positions, lane geometry, datum, scaling, estimator revision,
  sensing clock/delay/dropout, control frame, history/normalizer
  and actor size. Validate **against locally reconstructed meaning**,
  not checkpoint self-certified JSON.
- Test new network fresh and controlled 232→expanded warm-start,
  tiny genuine PPO gradient update, save/resume/recover, export,
  exact WASM inference, and old 220/232 regression. Compare
  repeated maps per frame to latest-only/decimated strips under
  *identical neural input shape*. Explicit all-zero-confidence
  survival/finiteness and sensor-loss fallback; no promise of
  perfect stop behaviour from invalid inputs.

### P4 — reward/curriculum integrity
- Snapshot old reward terms first: vx/vy/yaw, command step-height,
  swing/alternating phase, body height/foot-mid, clearance, falls,
  bad contacts, shaft power/torque/jerk, spring energy.
  **Correct physical geometry first at unchanged weights**.
- Then, only if needed, test bounded *outcome* shaping for
  verified corridor crossings and signed progress; ablate it
  against corrected-physics baselines. Never pay for infinite
  high foot lift or mere top contacts; reject looping/backwards
  reward farming and mismatched P-controller scoring.
- Curriculum expands by demonstrated ability: flat retention /
  already-demonstrated slopes → small shelves up/down →
  diagonal/asymmetric footfalls → low logs → mixed terrain,
  with speed, entry angle, distance, offsets, command changes
  and dropout sweeps. Choose heights from P0 mechanics, not
  arbitrary fixed severity leaps.

### P5 — matched learning comparisons, staged by evidence
- **R0** frozen historical 232 champion (different-shape reference
  only if new contract); **R1** post-contact-fix retrained flat.
  **A** preview masked, **B** current-only forward scan,
  **C** legal rolling profile; all A/B/C on *identical*
  network shape/training budgets/seeds/courses. **D** perfect
  preview (nondeployable) is delayed until C architecture and
  evaluation thresholds are fixed; never initialise C from D.
  E no-motion-compensation / F bounded reward are optional ablations.
- Screen ~**2.1M** steps per A/B/C at seed 733; extend only
  promising selected arms to ~**8M** at 733/734/735;
  reproduce winner vs A/R1 at ~**20M+** with ≥3 seeds
  **only when justified**. Save checkpoint/source hashes,
  actual FPS/VRAM and realistic wall time. This is a *ceiling*,
  not a mandatory overnight marathon.
- Primary held-out **within-corridor crossing rate**, with
  uncertainty over independent episodes/seeds; report progress,
  actual contacts, fall/toe strike, actuator work, command tracking,
  map reliability and flat/slope retention. **C−A** isolates
  legal preview benefit, **C−B** isolates memory benefit;
  not C vs archived R0 alone. No winner from PPO reward only.

### P6–P7 — browser and release
- Share compact **course specifications** (type, seed, size,
  transform, friction) between MJX and browser MuJoCo/WASM
  collision, not just Three.js meshes. Viewer sensor obeys same
  legal geometry, delay/occlusion and confidence; optional labelled
  oracle overlay must not feed actor. Preserve two-joystick mobile,
  v5017 pose-lattice navigator, manual takeover and heading controls;
  examination route must **not detour**.
- Publish benchmark evidence and known engine/hardware limits,
  including unsuccessful cases. Prepare reviewed branch PR only.
  **No main merge, champion promotion, Pages deploy or physical
  actuation without explicit additional approval.**

## What is measured and verified

### P0 campaign — user-run archive at 08:18 AEDT, 20 October 2026

**Source actually tested:** `6a32841c73042ffa380d197e953f31b196f60857`,
`feature/v5018-terrain-awareness`. **Archive:**
`v5018_p0_20261020_081806.tar.gz`, including `environment.txt`,
`stages.tsv`, JSON and per-stage logs. Do **not** claim the later
documentation/test changes were present during this frozen-source run.

- [x] **12/13 campaign stages had successful exit codes**. PASS:
  Python syntax compilation, npm ci, **12/12 browser tests**,
  Vite production build, GPU doctor/PPO 220-input smoke, v3
  static contact/rays, headless WASM/CPU comparison, moving impacts,
  actual per-world `geom_pos` variance, scanner coverage, clearance
  *kinematics* script and contact-loaded 256→2048 GPU stepping.
  The **only failed campaign stage** was full Python regression.
  NOTE: individual probe PASS != completion of hard P0 gates.
- [!] Full Python suite at tested SHA: **76 passed, 2 failed, 60
  warnings in 637.80 s**. Both failures are identified:
  1. `tests/test_learning.py::test_training_config_roundtrip` imports
     `scripts.smoke_training_lifecycle`, but the campaign launched pytest
     with CWD outside the repo, hiding local `scripts` from the import path.
  2. `tests/test_learning.py::test_v5016_schema8_viewer_bundle_roundtrip`
     still expected schema-2/232 examiner loading to **raise**, even after
     examiner support had been deliberately added. This was a stale
     assertion, **not rejected sensor policy data**.
  [~] Fixed the script's pytest CWD and changed the schema-2 test
  to assert successful roundtrip plus *rejecting tampered measurement
  clocks*. These fixes are **committed after the archive and have not yet
  been executed**. No 78/78 claim until rerun.
- [x] GPU environment: NVIDIA RTX 4060 Ti **8188 MiB**, Windows
  driver **610.88**, JAX **0.12.0**, MuJoCo/MJX **3.16.0**,
  Python **3.13.16**, `cuda:0`; Node **24.22.0**,
  npm **11.19.1**. Initial `nvidia-smi` used **1820 MiB**.
  JAX's `bytes_limit` **6,440,353,792** and allocator peak are
  allocator scope, **NOT complete host+GPU memory accounting**.
- [x] `emu doctor --v5015-smoke --domain-smoke` reported genuine
  4-world 220-input PPO optimizer update + 11 fields, 8-world MJX
  reset/step and randomization pass. **A dedicated 232-input PPO update
  is still unrun**; next campaign uses `--v5016-smoke`.

### Exact geometry, ray and impact evidence

- [x] Constrained-ball five surface contacts flat/30mm step-up/true
  30mm step-down/tilted ramp/capsule log agree CPU–MJX within
  **0.000590 mm** final centre height. Not gait or load-bearing
  force equivalence.
- [x] All **12** idealized JAX analytic static-primitive first-hit
  range comparisons agree with CPU MuJoCo `mj_ray`, worst
  **0.000046 mm**, including vertical step risers and both log
  top/side. This is *not* an actual mounted ToF module.
- [x] **4/4** constrained moving ball-foot impact probes: slow,
  fast toe strike, negative (high-clearance) control, and
  vertical touchdown. CPU/MJX first impact **same 5-ms frame**,
  maximum full-position trajectory mismatch **0.00221 mm**.
  CPU impact-force magnitudes and integrated force proxy are
  recorded, but **MJX physical force equality was not measured**.
  Dynamic multibody balance, passive shins and load-bearing gait
  remain unproven.
- [x] Actually varying **static box position per MJX world**
  using batched `geom_pos` passes: elevated/floor sphere heights
  swap correctly. First compile+run **2.495 s**; second same-shape
  call **0.352 s**. Does not prove arbitrary geometry dimensions,
  topology or many-course throughput.
- [x] Headless WASM **3.15.0** vs CPU MuJoCo **3.16.0**:
  all five vertical-slider fixtures pass, resting-Z differences
  at numerical-roundoff scale (`<2e-11 mm` reported). **Not**
  six-servo trajectory/policy/browser parity.
- [!] The dynamic moving-contact log contains initial JAX
  allocation failures attempting **6.00, 5.40, 4.86, 4.37 GiB**
  preallocations; JAX recovered and all probes produced valid
  results. Avoid misreporting these recoverable warnings as a
  completed memory feasibility proof or as fatal test failures.

### Full 232-input robot, measured GPU environment-stepping throughput

All 12 batch rows were reported PASS at the frozen source SHA.
5 repeats of 8 control frames each, 256→2048 worlds; each
flat/step/contact variant in a fresh subprocess. These are
**simulation environment steps/sec, NOT PPO training/update FPS**.

| Worlds | Flat FPS | Step ahead FPS (0 foot hits) | Shallow 5mm contact stress FPS |
|---:|---:|---:|---:|
| 256 | 8,190 | 5,248 | 5,197 |
| 512 | 9,787 | 10,373 | 8,201 |
| 1,024 | 16,666 | 15,150 | 19,556 |
| 2,048 | **25,799** | **23,383** | **24,533** |

- [x] The inserted *shallow contact pad* actually touched
  ball feet in **2,026 / 4,051 / 8,110 / 16,240**
  world-frame samples in the final repeat at
  256/512/1024/2048 worlds, respectively. The 2048 contact
  stress was approximately **4.9% slower** than flat in this
  noisy short-run comparison. The *step ahead* had **zero**
  contacts; its numbers only measure geometry-presence overhead.
- [x] Reported XLA allocator peak for 2048 worlds **221–224 MB**,
  but **must not be mistaken for GPU physical memory consumption**.
  Full real PPO backward pass with actual complex courses and
  memory margins is still required. The apparent anomalous cases
  where contact is faster than flat are short-run timing
  variability, **not an optimization discovery**.
- [!] Current reward and swing/stance metrics still use legacy
  flat-plane conventions. Positive reward here is **not**
  evidence of an obstacle learnt or crossed.

### Preliminary sensor footprint and mechanical reach

- [x] In idealized static CPU geometric scans across 3 body
  heights × 3 pitch × 3 roll × 2 lanes = **54 cases per design**:
  single 45° beam: **54/54 detected**, minimum **136 mm**
  first-detection lead (median **257 mm**); three beams
  28°/43°/58°: **54/54 detected**, minimum **208 mm**
  (median **364 mm**). Single-beam design had **3/54
  cases without step-face hits** even though it saw the top;
  three-angle design had some invalid/out-of-range beams
  (~93.2% mean valid-ray fraction). **No moving-leg
  occlusion, realistic beam footprint, noise, gyro/flow drift,
  timestamp error or dropout**. `any_detection` cannot certify
  correct step height or safe foothold preview.
- [x] Exploratory torso-fixed leg forward-kinematics sampler,
  2,500 poses per nominal unloaded speed window, found
  forward-advancing ≥25mm joint poses with observed
  **77/166/215/216 mm** maximum right-ball lift relative
  to the nominal left foot for 0.08/0.16/0.32/0.60s.
  **IMPORTANT: these are unconstrained optimistic kinematic
  samples, NOT achievable stepping heights or safe obstacles.**
  The model omits torque under load, stability, self-collision,
  passive shin movement, battery response and servo lag.
  Do **not** select a 100mm obstacle from these numbers.
- [x] User reports existing showcase champion visibly walks
  up and down ~10° slopes in the browser viewer. This is
  valuable experience to **preserve**, not a matched physical
  retained-skill exam. The normal examiner had a historical
  5° cap. [~] Branch now has **opt-in read-only 10°/12°**
  inclined-plane test allowance without broadening the
  production/training 5° limit; awaiting local execution.

### Interim result from next frozen-champion batch (not completed)

- [~] At the user's 20 Oct morning run, the `corrected_contracts`
  stage reported **5 passed, 1 failed**, in **0.49s**.
  The single failure was **test assertion wording only**:
  `test_v5018_crossing_gate_rejects_incomplete_or_fabricated_support`
  expected exception text `world-fixed` for a zero-length
  `course_axis_xy=(0,0)`, but the evaluator correctly rejected it with
  `ValueError: course axis must be nonzero`.
- [~] Corrected the assertion in feature-branch commit
  `85d36c79f91c9ca3328f86ee81d4068e53726618`.
  **Not present in the already-running local campaign and not rerun.**
- [ ] The user snippet ended just as `sensor232_ppo_smoke`
  began. **No outcome for this or any later heading tests was supplied.**
  Do not infer an outcome. Allow that campaign to complete and preserve
  the archive before pulling the newer commit.
- [ ] After it completes, run only the one corrected test and/or a
  short targeted suite; do **not** rerun the GPU slope campaign just
  to fix an exception-message assertion.

### Next branch-only implementation (committed, not yet verified by WSL)

- [~] Fixed the two pytest regressions **after** this archive.
- [~] `scripts/run_v5018_frozen_champion_baseline.sh` now
  schedules corrected targeted pytest, real **232-input PPO**
  smoke and matched frozen-champion
  **heading P-controller OFF/ON** 0° flat, real 10° uphill,
  real 10° downhill, cross-left 10° (seeds 1701, 1702),
  plus flat reverse and turn-stop. All independent outcomes,
  hash-identified, no actor modification or long training.
  This command is **not yet executed**.
- [~] Added pure `emu/directional.py::examine_obstacle_crossing`
  no-detour/two-foot-support evaluator and adversarial unit
  examples (turn-around, backtracking, bypass, hover, long stall,
  reset/teleport). **Only a scorer contract, not wired into
  training or P1 terrain rewards.** Its new tests have not
  yet been run, and P1 integration remains gated on P0.
- [~] `scripts/bootstrap_wsl_v5018.sh` installs a blank Ubuntu
  WSL Python 3.13 via uv, CUDA JAX, pinned npm/Node;
  committed but **not separately rerun clean-room**.
- [x] `@mujoco/mujoco` viewer pinned to npm package **3.15.0**
  in manifest and lock; actual npm CI, **12 tests**, build
  and minimal WASM fixture executed PASS in this archive.

## P0 blocking gates before schema freeze or PPO terrain training
- [ ] Check *moving* collisions/toe strikes and actual contact/support force in MJX against CPU, not merely final sphere poses.
- [ ] Confirm per-world terrain pose variation is JIT-efficient or adopt fixed static course-bank fallback.
- [ ] Obtain **collision-loaded** full robot GPU throughput, then measure one real PPO update and VRAM. 2048 is a goal, not mandatory; reduce safely when needed.
- [ ] Prove physically plausible forward-scanner coverage of both future foot lanes for selected pitch/roll heights, including risers/occlusion/dropout. Choose actual sensor class/FOV and compare 3/5/8 bins later in P2.
- [ ] CPU–MJX–WASM discrepancy for same actor commands and terrain; account for wasm 3.15.0 vs CPU/MJX 3.16.0.
- [ ] Derive mechanical *dynamically achievable* swing clearance with actuator speed, torque, drag and support, not only kinematic reach. Physical hardware deployment not demonstrated; sim-first only.
- [ ] Retest full 220/232 loader, viewer/mobile navigation regressions; preserve old policy behaviour.
- [ ] Fix any failed probe rather than relaxing numeric gates silently. Do not infer real steps/climbing from the initial 30 mm sphere collision.

## Later gates (remain intentionally locked)
- **P1:** Integrate verified collidable terrain into actual resets/worlds; terrain-relative support/swing/reward-only metrics, corrected flat control and no-detour corridor crossing exam. Preserve old policy semantics.
- **P2:** Realistic timestamped two-lane beam model, inertial/flow motion compensation, bounded confidence-rated rolling profiles, quantified 3/5/8-bin choice, separate instantaneous tripod slope.
- **P3:** Explicit schema only after P2 (312 if and only if 5 bins/lane), strict fingerprint and normalizer checks; fresh/warm PPO smoke, save/resume/export/actual WASM inference, old-policy compatibility.
- **P4:** Bounded anti-cheat reward/curriculum audit with feasible obstacles; no hidden terrain truth into actor.
- **P5:** Controlled matched A masked/B current-scan/C rolling, R0 archived/R1 corrected-flat and delayed D oracle; staged GPU seeds/steps with held-out no-detour crossing and flat retention.
- **P6:** Physical MuJoCo/WASM challenge courses, full-sensor consistency, preserve v5017 joystick/navigation/mobile.
- **P7:** Independent validation/scoreboard and reviewed PR; **approval required** before main/showcase changes.

## Next user-run: targeted champion heading + 232 PPO evidence

After the frozen P0 campaign archive was **fully preserved**, the following
starts the next independent read-only baseline experiment on the
**updated branch** (not a rerun of the 2048-world P0 campaign):

```bash
cd ~/emu-v5018
git status --short
git pull --ff-only origin feature/v5018-terrain-awareness
source .venv/bin/activate
bash scripts/run_v5018_frozen_champion_baseline.sh
```

Outputs are in `~/robot_lab/emu_exams/v5018_heading_YYYYMMDD_HHMMSS/`
with `stages.tsv`, per-case controller OFF/ON reports, summary
and matching `.tar.gz`; send the archive. This is
**frozen-policy evaluation only**, with a small 232 PPO gradient smoke,
no prolonged training. It covers the user's observed 10° viewer
slope capability in a **different infinite inclined-plane test**,
which must be labelled accordingly.

The original 20 Oct P0 campaign used
`bash scripts/run_v5018_p0_campaign.sh`. **Do not rerun that
full campaign simply to resolve two stale tests.**
Targeted regression execution will establish whether the fixes
actually work. Preserve all source SHA/checkpoints and external
archives. Runtime GPU OOM can require smaller independent exam
world counts in a new output root, not silent protocol editing.

**Post-exam engineering decision:** confirm at least a modest,
repeatable flat/10° heading-and-walking baseline; then close
remaining P0B **dynamic** servo-clearance, limb-occluded scanner
footprints and identical six-servo MJX/CPU/WASM trajectory gap
before wiring true stepped terrain and corrected support into
P1. At present, the required ray/primitive family and 2048-world
contact workloads are promising, but a terrain-trained actor,
true crossing and safe hardware remain **unproved**.
