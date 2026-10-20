# EMU v5018 — Terrain-Aware Stepping: living branch checklist

**Last evidence review:** 20 October 2026, ~20:11 AEDT (user-defined course geometry and waypoint refinement; current feature source awaiting WSL/browser acceptance). **Ref:** `feature/v5018-terrain-awareness`. **Status:** P0A/P0B partial; **P1 deep moving-MJX 5/5 stages PASS** (11/11 terrain + 2/2 trace tests, flat parity, controlled contact and real frozen-actor motion); **Viewer course 8/8 stages PASS** (18/18 Node tests, six WASM colliders, build + install). These are validated experiments, **not certified supported walking, motor loading, same-state reward superiority, or real-browser visual acceptance**. P2–P7 gated. **Never merge main, replace a champion or deploy GitHub Pages without explicit approval.** This is the **only tracked living v5018 progress/gate checklist**. The full reviewed master-plan acceptance criteria are condensed below; [V5018_HANDOVER.md](V5018_HANDOVER.md) is a stable, practical restart guide.

## 20 Oct evening — eight half-gap waypoints, continuous CCW movement, mixed obstacles (UNTESTED on WSL)

**Latest user instruction:** retain the existing destination/waypoint
navigation system, but place goals **halfway between neighbouring
obstacles** in their existing circle order, never reversing
the traversal direction. This refines the user's preceding
request for wider angled step bars, step-ups, step-downs,
half-buried logs and real local uneven park terrain.

### Actual feature-branch implementation (source committed, not yet browser/WASM tested)

- [~] **Eight gap destinations**, not 24 intermediate stopping goals:
  `generateChallengeCourse().waypoints` now contains precisely
  one pose at polar angle `2π(i+½)/8`, radius 4m,
  with CCW heading `angle+π/2`. The final → first
  transition wraps in the **same anticlockwise direction**.
  The navigator's original arrivals/dwell, Worker-backed
  pose-lattice planner, human override and six-servo
  command pipeline are retained.
- [~] **Real passage through challenges:** a straight
  midpoint-to-midpoint chord would miss the obstacle
  by roughly 0.30m radially. Each subsequent leg
  therefore uses the intervening obstacle centre as
  an **INTERNAL route-shaping via**, not an additional
  navigation/parking goal. The real A* planner solves
  both sides; its route includes one positively driven
  forward crossing point and terminates only at the
  next midway gap. The initial leg goes from spawn
  to the first gap without a forced centre-crossing.
  On crossing/replanning after the obstacle, the
  via is dropped rather than driving backwards.
  Timed-out circuit missions retry the SAME gap
  instead of silently skipping ahead. Ordinary
  Random destinations never receive a via.
- [~] The visible teal `circuitPath` now joins
  alternating obstacle centres and between-obstacle
  destinations; it is NOT a misleading polygon
  of straight midpoint chords. The actual driver
  may deviate and is not guaranteed to complete
  a ring until real visual tests. Heading remains
  ring-tangent CCW even when the obstacle's own
  yaw is rotated.
- [~] **80cm full-width** collidable barriers
  (previously 40cm) and deterministic obstacle
  yaw offsets in degrees
  `[0,+15,-15,+25,-25,+10,-10,+20]`.
  Eight stations now include two each of
  **basic sharp step**, **sharp step-up plus
  long gentle ramp DOWN**, **gentle ramp UP
  plus sharp step-down** (the true reverse
  geometry; no impossible pit dug through
  a solid MuJoCo heightfield), and **half-buried
  capsule log**. Logs are default-enabled
  only in viewer challenge mode, are
  toggleable and explicitly unproven.
  All coloured labels name the actual family
  and nominal height; tier controls 5/10/15/20mm
  frontier upper bound and mixes smaller challenges.
- [~] **Actual seeded variable ground**, not
  flattened ring: the park `Float32Array`
  hfield retains its height distribution beyond
  the central 2m flat spawn region at the
  4m circle. The browser's physical hfield,
  visual grass triangles, trees, local ground
  height scanner and obstacle Z agree on the
  SAME sampled final surface. Flat step-bar
  top uses the centre ground + nominal height;
  composite plateau uses local plateau ground;
  sloped ramp endpoints use actual separate
  local elevations; log centreline uses sampled
  surface height at BOTH ends so approximately
  HALF of the capsule radius is buried.
  Natural hills can change effective difficulty
  across any given 80cm span; a 5mm bar
  is not an isolated flat-lab 5mm obstacle.
- [~] Mixed-course browser tests now verify:
  exactly eight halfway goals in strictly
  CCW heading order including wraparound;
  double-width and varying yaw; actual XML
  box/ramp/log collision geoms; nonflat
  hfield and locally-seated height; a
  buried capsule log; trees within sight
  but outside walking lane; internal
  crossing via exactly once per leg;
  initial spawn-to-gap and post-crossing
  replan without backward return; and
  strict fixed Worker targets without
  random fallback. Updated headless
  WASM test now includes **11 physical
  probes** (flat, 5/10/15/20mm step
  bars, both step/ramp families'
  top and ramp surfaces, half-buried log,
  and nonflat hill). It reports actual
  height error/contact per fixture.
- [x] **Isolated source-level V8 checks**
  successfully parsed all changed JS
  modules, checked generated real
  physical XML, found 12 obstacle geoms
  (2 bars, 4 step-up pieces, 4 step-down
  pieces, 2 logs), preserved 17 actual
  trees and confirmed nonzero and varying
  terrain height at the 4m stations
  (~30.6–36.4mm for this seed). A real
  A* route was found from spawn to the
  first gap, through each of seven further
  obstacle centres, and across the
  last → first wraparound; no planned
  crossing point stopped inside an
  obstacle. This is source-level
  evidence only, **not** Node/npm/MuJoCo
  WASM collision or live robot traversal.
- [ ] **UNTESTED on WSL in this latest source:**
  full Node regression suite, combined
  uneven hfield + tilted-box + log
  MuJoCo/WASM compilation/contact,
  Vite build and local viewer reinstall;
  interactive Windows/mobile circle,
  automatic switching to CCW circuit
  and successful 8-station locomotion.
  Former 8/8 acceptance applies ONLY
  to older source `f0150734...`,
  not the widened/angled/uneven build.

**Next command after pulling current feature HEAD:**

```bash
cd ~/emu-v5018
git pull --ff-only origin feature/v5018-terrain-awareness
source .venv/bin/activate
bash scripts/run_v5018_viewer_course_acceptance.sh
```

It archives to
`~/robot_lab/emu_exams/v5018_viewer_course_*.tar.gz`
and installs the local viewer only after
Node + WASM tests pass. Then serve
`emu viewer showcase/champion.emupolicy --no-open`
and hard-refresh `http://127.0.0.1:8765/`
with Ctrl+Shift+R. Select **Park + challenge ring**,
confirm **Challenge circuit**, verify a single
goal in each gap is visited CCW, and check
grass/trees/terrain and collider signage.
Any failed test should be diagnosed
and fixed **on feature branch**, not
disabled or treated as a visual-only
misunderstanding.

**Protected boundaries unchanged:** no
edit or merge of `main`, protected champion,
public Pages, actor/NN schema, PPO rewards
or ongoing GPU training. Cold-start from
no prior brain remains mandatory for
future true terrain learning.

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

### 20 Oct 09:07 frozen-champion heading/10° exam — reviewed results

**Archive:** `v5018_heading_20261020_090712.tar.gz`.
**Exactly tested commit:** `d525ab662990405bc2ab6c8cd3e12cf14cdb60a7`;
**protected champion SHA-256:**
`d3d319b82fbc5de331752f0f06753c57a776ac3212bcb6d386116c4f5f8a4862`.
The local branch subsequently advanced; do **not** claim new tests/code
were executed under that archived SHA.

- [x] **11/12 independent stages exited zero**. The only failed stage
  was `corrected_contracts`: **5 passed / 1 failed**, an assertion that
  expected `world-fixed` on a zero-length axis while the scorer
  correctly emitted `ValueError: course axis must be nonzero`.
  The assertion wording was fixed later in feature commit
  `85d36c79f91c9ca3328f86ee81d4068e53726618`; rerun pending.
- [x] **Real 232-input PPO gradient/update smoke PASS** on
  CUDA: 2 worlds, 2-frame rollout, one optimizer update, finite params
  and rewards. This closes the earlier missing *minimal* 232 PPO smoke,
  **not** 2048-world PPO terrain training throughput or memory.
- [x] **10 matched heading comparisons all executed** using
  frozen 260M 232-input champion: flat, *real* inclined 10°
  uphill/downhill/cross-left (seeds 1701+1702, four worlds each),
  plus flat reverse and turn-stop (seed 1701, four worlds each).
  All original operator commands preserved; no servo scripts or
  policy mutation, outer `kp=0.75` ON versus OFF.
- [x] **80/80 complete 350-frame (7s) trajectories survived**,
  across all 10 cases × heading OFF/ON × four worlds. This is
  simulator survival only, not demonstrated challenge completion.
- [!] **0/80 strict contact-qualified walking passes**, despite
  clear signed movement. Some downhill trajectories produced
  1–5 alternating touchdown pairs, but no world simultaneously
  met all strict course+stride conditions; flat cases registered
  zero qualifying pairs. This could be overly stringent contact
  event scoring *or* genuine sliding/low-clearance gait. **Do not
  weaken thresholds or declare "not walking" until raw contact
  traces are diagnosed.**
- [!] Real 10° incline results: **0/8 strict course passes per
  heading mode per slope direction** (up/down/cross).
  Positive motion and survival do **not** certify controlled
  sustained 10° incline traversal. The user-observed 10° viewer
  ability remains an important but different observation.
- [x] Heading controller improved yaw orientation substantially.
  Across both seeds (8 worlds per slope/flat), mean heading
  absolute error OFF→ON and signed movement m OFF→ON:
  - flat: **12.5°→3.7°**, **0.532→0.558m**; strict course
    pass **5/8→6/8**, strict gait pass 0.
  - uphill 10°: **20.2°→3.9°**, **0.164→0.228m**; course pass 0.
  - downhill 10°: **40.4°→6.1°**, **0.915→1.070m**;
    course pass 0 (large path error/overspeed), not success.
  - cross-left 10°: **64.3°→24.5°**, **0.358→0.579m**;
    course pass 0 (large path/heading error).
  - reverse flat: 4/4 strict *course* passes both ON/OFF;
    turn-stop flat: **2/4→3/4** strict course passes.
  The matched ON condition is promising **but not automatically
  better on every other metric or a guarantee against sideways
  drift**.
- [!] **Servo-duty flag:** simulated shaft torque exceeded
  the configured nominal **0.638 N·m continuous** reference
  for roughly **74–78% of scored servo samples**, with
  per-world peak reaching the configured **1.913 N·m**
  limit in representative modes. These are *simulation* actuator
  outputs, not motor current, temperature, reliable torque duty
  for physical hardware or a safe transfer criterion. Before
  real deployment, address actuator realism and effort/heat.

### 20 Oct 09:49 — independent first physical step/stride pilot, reviewed

**User-run archive:** `v5018_gait_step_20261020_094915.tar.gz`.
**Exactly tested feature commit:**
`45f0655f687fa4628f0c9b002c5af202362e2896`.
**Protected 260M actor SHA-256:**
`d3d319b82fbc5de331752f0f06753c57a776ac3212bcb6d386116c4f5f8a4862`.
The post-archive branch is newer; do not conflate newly committed
instrumentation with this old-source test result.

- [x] **All 15/15 independent stages exited zero**; six targeted
  pytest cases **6/6 PASS in 0.38s**. Four matched
  0°/10° directional-gait probes, nine frozen actual 0/5/10 mm
  courses, objective report summary. The prior assertion-message
  error is resolved **on the tested source**.
- [x] Read-only 232 frozen actor, commanded **+0.10 m/s** for
  **350 × 20 ms = 7s** in **four worlds per case**,
  heading P=0 or P=.75 (no motor scripting), two seeds for ON,
  one seed for OFF. All step geometries are actual *static
  collidable boxes* in a transient copy of robot XML; untouched
  `showcase/champion.emupolicy`, policy observation, weights,
  reward and old training roots.
- [x] Objective course crossing gate checks fixed world-frame
  forward course, torso and BOTH ball feet beyond the far edge,
  genuine subsequent foot-ground-or-obstacle support, 0.42m finish,
  heading reversal, timeouts, teleport, stable runout and
  **±0.12m torso corridor**. The physical obstacle spans
  ±0.20m in Y (so ordinary body detours outside the corridor
  cannot simply sidestep a 0.08m longitudinal box).
- [x] **Course gate successes (per tested pilot, not gait-certified):**

  | Scenario | Heading OFF, seed 1701 | Heading ON, seed 1701 | Heading ON, seed 1702 | Heading ON total |
  |---|---:|---:|---:|---:|
  | Flat with off-course control box | 0/4 | 1/4 | 3/4 | **4/8** |
  | Actual 5mm raised box | 0/4 | 2/4 | 3/4 | **5/8** |
  | Actual 10mm raised box | 0/4 | 3/4 | 3/4 | **6/8** |

  Every failed pilot crossing in this dataset was labelled
  **`corridor_detour`**, with actual torso max lateral excursions
  often 0.13–0.29m. Other scorer gates passed in those worlds.
  This supports evaluating heading hold as a useful stabiliser,
  **not** treating it as an automatic guarantee of accurate lateral
  path following. Unequal OFF/ON seed coverage prevents simplistic
  0/12 vs 15/24 inferential claims.
- [x] Real ball-foot/box *world-frame* contact counts across 3 pilot
  configs: **5mm: 314+315+285=914**; **10mm:
  365+379+287=1,031**. Flat control: **0**. Every step trial
  touched the real obstacle with the simulated ball feet.
  **Contact by itself does not certify a controlled step-over
  manoeuvre**: sphere feet may contact/roll against the edge,
  and the current pilot did not record step-top vs riser normals,
  peak penetration, true impulse or force transfer.
- [!] The surprising empirical **10mm 6/8 vs 5mm 5/8 vs flat 4/8**
  is *not proof higher obstacles are easier*. Sample is small;
  contact can affect lateral guidance/foot motion and scoring
  margin. The next controlled frontier should change one
  dimension at a time with matched seeds/commands/same corridor.
- [!] The strict *walking/gait* gate still recorded zero
  qualification on flat and uphill, with **some** downhill and
  cross-slope alternating touchdown pairs. Across four
  flat-control worlds the contact diagnostic recorded **around
  100 liftoffs per foot aggregate**, contact fractions ~0.56–0.62,
  and real airborne intervals up to 6–7 × 20ms.
  The strongest rejection was **`body_motion_under_25mm`**
  during each specific air interval: ~96–100 of ~100
  completed flat touchdowns per foot. Air duration and
  4mm toe clearance were often adequate; foot displacement
  sometimes under 12mm. A fixed 25mm-per-airborne-event
  body-travel requirement may not suit an actual *short-step*
  gait, but **do not quietly relax the scoring gate**;
  first inspect per-event displacement distributions, support
  transfer and foot slip/chatter. Qualifying touchdown
  counters and alternating *pairs* are separate.
- [!] Servo over-nominal-continuous model torque in the actual
  step pilot was typically around **0.74–0.76 of samples**,
  confirming existing efficiency/hardware realism warning.

### Implemented AFTER the 09:49 archive (not yet run on WSL)

- [~] Extended **read-only** `diagnose_contact_stride_gate` with
  every independent contact-backed airborne event's start/end
  frame, toe-clearance, per-step foot/torso displacement and
  p10/median/p90 distribution. Strict previous pass criteria
  and physical contact source are unchanged; focused synthetic
  tests were expanded. This is designed to identify real
  small-stride walking versus contact chatter/sliding without
  upgrading failed evidence into a false pass.
- [~] Extended `scripts/v5018_champion_step_pilot.py` to a
  bounded **0/5/10/15/20mm** geometric course and **±35°**
  step front-face orientation, optional **0.05–0.18m/s** forward
  command, and recorded actual ball-feet spatial overlap with
  the top of the obstacle. The world-frame near/far/finish
  checks account for rotated box projected footprint.
  **No new NN terrain preview, servo torque training or
  full browser actor parity**.
- [~] `scripts/run_v5018_obstacle_frontier.sh` batches the
  targeted tests, fresh contact-backed stride diagnostics,
  two-seed straight 0/5/10/15/20mm comparisons, ±25° face
  approaches, 0.07/0.14m/s speed perturbation and matched
  heading-OFF ablations; results/archive outside Git.
  This campaign has **not** been run yet.
- [~] Existing CLI evaluator and P0 campaign assets remain
  compatible; the step extension and new diagnostics must be
  validated on the user's WSL before performance claims.
- [x] `@mujoco/mujoco` pinned to npm/WASM **3.15.0** with
  prior 12/12 browser Node tests and ball fixture pass.
  Full six-servo actor/terrain contact parity is not certified.
- [~] `scripts/bootstrap_wsl_v5018.sh` remains committed
  but has not been independently clean-room rerun.

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

### 20 Oct 10:26 — first frontier attempt interrupted by Bash typo

- [x] Started frozen 232-input actor frontier at source commit
  `c6a5dab5b0ea9b7c6d34db2b6e85599ae015314a`;
  champion SHA-256 remained
  `d3d319b82fbc5de331752f0f06753c57a776ac3212bcb6d386116c4f5f8a4862`,
  JAX 0.12.0 / MuJoCo 3.16.0 / CUDA RTX 4060 Ti 8GB.
- [x] `targeted_python_tests` **4/4 passed in 0.15s**.
  All four 4-world heading comparison runs (flat, real
  10° uphill, downhill, cross-left) finished with exit 0.
  The four attached per-surface `heading_comparison.md`
  reports come from **seed 1701 only**, 4 worlds/mode:
  - Flat: heading error **14.21° OFF → 4.57° ON**,
    strict course pass **2/4 → 3/4**,
    progress 0.5261m → 0.5192m, strict gait 0/4 each.
  - Uphill 10°: heading error **22.01° → 5.37°**,
    progress 0.1321m → 0.2108m, strict course pass
    0/4 in both.
  - Downhill 10°: heading error **45.29° → 7.39°**,
    progress 0.9121m → 1.0687m, strict course pass
    0/4 in both.
  - Cross-left 10°: heading error **68.77° → 27.94°**,
    progress 0.2847m → 0.5413m, strict course pass
    0/4 each; yaw-rate MAE and contact slip speed
    **worsened** under heading hold.
  Full survival was 1.0 in all four OFF/ON reports.
  Correct original operator commands were used for scoring.
  This is partial frontier evidence, not an independent
  training improvement or complete slope-walking qualification.
- [!] Before the **first step-frontier world** was launched,
  Bash exited at runner line 72:
  `code_yaw="${code_y/+/p}"` referenced undefined
  `code_y` under `set -u`. This is an experiment
  orchestration typo, **not a failed MuJoCo collision,
  neural policy or terrain algorithm**. No 15/20mm,
  ±25° or speed-frontier data were collected yet.
- [~] Fixed the typo in feature branch commit
  `6a53dbadfa5170cafe05c1c8bd5709dff802f423`;
  added `--resume EXISTING_OUTPUT_DIR` to skip
  already-PASS stages only when logs/output exist,
  preserve `environment.txt` original tested SHA,
  append `revision_history.tsv` with updated source SHA,
  require matching protected policy SHA, and write a
  tar.gz via EXIT trap **even on an unexpected shell abort**.
  This bugfix itself has **not been executed** in WSL.

### 20 Oct 10:43 — resumed frontier interrupted before physics steps

**User-uploaded actual archive:** `v5018_frontier_20261020_102641.tar.gz`,
containing **46 tar entries and only five PASS stage rows**.
Source recorded for first attempt:
`c6a5dab5b0ea9b7c6d34db2b6e85599ae015314a`.
Resume history recorded:
`2026-10-20T10:43:30+11:00`,
commit `cad93f94fbb026da5feade1f15786af9fc25b8ee`.
Protected champion SHA-256 remained
`d3d319b82fbc5de331752f0f06753c57a776ac3212bcb6d386116c4f5f8a4862`.

- [x] Resume correctly **skipped 5 completed stages**, retaining all
  old `stride_*/heading_comparison.json` and original
  `environment.txt` (4/4 targeted tests + four slope/flat
  matched heading exams). Those are from the ORIGINAL commit.
- [!] **Zero** `step_*.json` frontier results or
  `aggregate_reports` output exist in the uploaded archive.
  The abort came from a **second** stale variable reference in
  the runner's stage filename: the assignment used
  `code_yaw`, but `local stage="...yaw${code_y}..."`
  still referenced undefined `code_y` with `set -u`.
  This is **not** evidence of any robot failing terrain.
- [~] Corrected the second unbound variable in branch commit
  `17414a31d8b7c0aa62bc07ea1bd4261e1c469d16`.
  Added a **pre-GPU** Bash loop to render and verify 180
  unique parameter combinations, plus stderr capture to
  `runner_stderr.log` for unexpected shell failures.
  The variable expansion/naming loop was independently
  exercised under `bash -u` (3 edge cases + 180 unique
  names) locally with PASS. The full WSL campaign has **not**
  been rerun; real 15/20mm/±25°/speed outcomes remain UNKNOWN.
  The runner was then further hardened in commit
  `3c029c256614adf35f2db93d4dbc21a324a287fc`:
  **one shared `step_stage_name()` routine** now generates
  filenames for BOTH the real runner and its preflight,
  avoiding divergent duplicated string interpolation.
  This exact formatter was separately exercised with 180
  unique combinations under strict Bash variable checking
  (PASS). This is still not a completed WSL physics run.
- [ ] The same directory is still the correct resume target:
  `~/robot_lab/emu_exams/v5018_frontier_20261020_102641`.
  On the next resume, new stage results must record their
  later source SHA in `revision_history.tsv` alongside
  the preserved original stages.

## 20 Oct 11:43–12:05 — completed frozen-champion frontier, review

**Actual uploaded artifact:** `v5018_frontier_20261020_102641.tar(1).gz`
(475 KB), containing `stages.tsv`, per-world physical step JSONs,
four earlier directional exams, summary and revision history.
**Completed status: all 34/34 stages have PASS execution codes.**
Breakdown: 4/4 targeted Python tests, four read-only flat/10°
heading/stride comparisons (from the **original** 10:26 commit
`c6a5dab5b0ea9b7c6d34db2b6e85599ae015314a`),
**28 distinct physical-course configurations**, and
one aggregate summary. New step cases were executed after the
final resume recorded at 11:43 AEDT with feature commit
`69daa3d253cf92d0e3c40fbc060d54c4cbb549cf`;
`cad93f94...` was an earlier aborted resume. This is a **mixed
source-revision archive** and was NOT all executed on one commit.

The 260M frozen-policy SHA-256 remains
`d3d319b82fbc5de331752f0f06753c57a776ac3212bcb6d386116c4f5f8a4862`.
Every step used real MuJoCo/MJX contact boxes; no actor weights,
observations, PPO reward, `main`, `showcase` or Pages changed.

### Objective crossing scoreboard (not walking gait certification)

- [x] **28/28 unique step configurations ran**, comprising **112
  separate four-world trials**; **49/112** satisfied the physically
  supported fixed-world corridor/finish gate, including **8/20
  flat-control** trials. Physical-obstacle-only configurations
  achieved **41/92**. This does not prove preview sensing or
  collision-free high-stepping.
- [x] For exactly matched straight course, `vx=.10m/s`,
  heading P **ON**, seeds **1701+1702**, four worlds/seed:

  | Real course | Fixed-world crossing success |
  |---|---:|
  | 0mm (identical off-course control box) | **4/8** |
  | 5mm box step | **5/8** |
  | 10mm box step | **4/8** |
  | 15mm box step | **6/8** |
  | 20mm box step | **3/8** |

  The 20mm case included **one simulator fall/reset, one
  heading reversal** and additional incomplete/detouring trials.
  15mm >10mm here is **not monotonic learning**; a small,
  variable set of stochastic contact/directional outcomes
  cannot establish a mechanical height limit.
- [x] At 10mm with **±25° box face yaw**, same
  seed1701+1702 and heading P ON:
  **+25° = 7/8, −25° = 3/8**.
  For 5mm (one seed): +25° 2/4, −25° 3/4;
  for 15mm (one seed): +25° 1/4, −25° 2/4.
  Strong *observed* asymmetry at 10mm is worth targeted
  left/right foot-timing and lateral-slip investigation;
  8 trials per direction is not enough to certify asymmetry
  across seeds/course lengths.
- [x] At seed 1701, straight box, heading P ON,
  changing ONLY operator vx:

  | Requested vx | 0mm | 5mm | 10mm |
  |---|---:|---:|---:|
  | 0.07m/s | 2/4 | **0/4** | **0/4** |
  | 0.10m/s | 1/4 | 2/4 | 1/4 |
  | 0.14m/s | 2/4 | 2/4 | **3/4** |

  At slower speed, 5/10mm misses often had
  `no_stable_finish_and_runout`; this may reflect
  fixed 7s/0.42m finish timing rather than inability
  to contact a threshold. Compare fixed-distance/time-
  normalized crossings before declaring slow gait unfit.
- [!] **59/112 total trajectories violated the ±0.12m
  lateral corridor**, accounting for **59 of the 63 failed
  crossing worlds** (some also failed other criteria).
  `no_stable_finish_and_runout` occurred 16 times;
  `no_torso_and_two_feet_crossing` 5; heading reversal
  once; fall/reset once. Fail reasons overlap.
  This supports **lateral course-keeping as a major
  constraint** at 0–15mm, not a general claim that feet
  cannot reach these heights. Heading P corrects yaw but
  does NOT guarantee lateral path control.
- [x] Step-box actual ball-foot contact world-frames
  occurred at every height. Across matched 2-seed
  straight P-ON cases: 5mm **653**, 10mm **671**,
  15mm **714**, 20mm **793**.
  For 20mm, **335/780** foot-in-box-XY frames had
  ball-bottom height >3mm **below the box top**.
  These footprint probes are *not validated solver
  penetration or load-bearing top-contact normals*:
  they may include side-wall contacts or interpolation
  effects. **Do not label every passing course a clean
  toe-cleared step-over**.

### Measured natural gait and the misleading historical gate

- [x] The original four heading-diagnostic stages
  from 10:26 are retained, with **actual per-touchdown
  airborne event records** in the current archive.
  At flat grade 0°, heading OFF: **196 real contact-
  qualified airborne intervals** in four worlds;
  heading ON: **197** intervals. All were strictly
  below the historical **25mm torso travel PER SWING**
  threshold, despite real liftoff and touchdown.
- [x] Aggregate flat event p10/median/p90:
  - OFF torso movement: **8.0/11.9/15.5mm**;
    swing-foot travel **11.7/25.6/40.0mm**;
    clearance **12.4/15.3/19.4mm**.
  - ON torso movement: **7.6/11.2/15.0mm**;
    swing-foot travel **11.8/23.7/37.6mm**;
    clearance **12.5/15.4/19.4mm**.
  - All 393 events had ≥3 airborne frames and
    ≥4mm clearance. **351/393** also had
    ≥12mm foot relocation. **0/393** had
    ≥25mm torso movement during that one airborne
    interval. Consequently the historical gait v1
    is **incapable of crediting this actual short-step
    gait on the measured flat run**, even though this
    does not automatically establish non-slipping
    or energy-efficient walking.
- [~] Committed **separately versioned RESEARCH-ONLY**
  `diagnose_alternating_contact_pairs_v2` to
  `emu/directional.py`, requiring real takeoff and
  touchdown, ≥3 airborne frames, ≥4mm clearance,
  ≥12mm forward foot relocation, actual opposite-
  foot support, and **≥25mm NET SIGNED operator-
  course progress across an alternating two-foot
  pair**, not during each single swing. Its output
  lives alongside historical v1 in exam JSON but
  **does not change official gait/course pass,
  historical scoreboards, policy or PPO**. Focused
  anti-sideways, anti-contact-free and negative-swing
  unit tests were added, NOT YET RUN in WSL.

### Reproducibility and next implementation controls

- [!] Two nominally identical frozen-champion 10mm
  seed1701 step pilot reports produced different
  crossing scores: earlier **3/4** (09:49) versus
  frontier **1/4** (11:43). Both used the same policy
  SHA, nominal seed and 0.1m/s command, but the
  **generated XML hashes differed** because the
  later course included an explicit `euler="0 0 0"`
  attribute absent in the original. Non-identical XML
  artifacts mean these are **not** a clean repeatability
  experiment. The cause of physical outcome variation
  remains to be proven, not attributed definitively
  to the zero-angle attribute.
- [~] After frontier evidence, corrected
  `scripts/v5018_champion_step_pilot.py`: at
  **zero yaw**, omit the unused `euler` attribute
  and preserve *byte-for-byte* original 09:49
  course XML. Added tests requiring exact recorded
  SHA-256 hashes for the original 0/5/10mm cases
  (`cbc81bde...`, `7968b117...`, `f5ca6525...`).
  **The new XML hashes/physics still await WSL
  execution**; no reproducibility claim yet.
- [ ] Before staging terrain-trained PPO, validate
  `v2` with genuine course/fall/stance controls and
  **physical support, foot-slip and energy metrics**;
  inspect the ±25° asymmetry and corridor control.
  Preserve the frozen actor and both the old and new
  metrics in score reports.
- [ ] P1 next: implement opt-in actual collision-
  terrain-relative foot/torso/support/stance-slip
  measurements, maintain existing 220/232 actor
  observation contracts, and compare byte-for-byte
  old flat reward path before enabling small real-step
  training. Then test timestamped forward ray
  visibility and confidence on actual moving gait
  without privileged terrain heights in the actor.

## 20 Oct 12:48 — focused gait-v2 and byte-identical repeat PASS

**Uploaded evidence:** `v5018_gaitv2_acceptance_20261020_124851.tar.gz`
(user's WSL, **tested source commit
`622d515b352abae67b0065b643912f151c4afa78`**).
Protected showcase actor SHA-256 unchanged:
`d3d319b82fbc5de331752f0f06753c57a776ac3212bcb6d386116c4f5f8a4862`.

- [x] **5/5 stages PASS:** 4/4 focused synthetic and original XML
  hash tests (0.14s); matched real flat and real 10° uphill
  directional OFF/ON evaluations; genuine 10mm collision
  step course at seeds 1701+1702.
  **No policy learning or weight transfer occurred.**
- [x] All four flat worlds recorded genuine alternating
  physical-contact gait-v2 candidate pairs, unlike
  strict historical v1 (zero everywhere). OFF: **8,1,7,8
  =24 total**; ON: **2,2,9,8 =21 total**.
  Flat **course passes 3/4** under each heading mode.
  This is a diagnostic candidate, NOT automatic gait-v2
  certification or a rewritten historical v1 result.
- [!] Real 10° uphill has **0/4 course passes and
  0/4 v2 alternating-pair positives** in BOTH heading
  modes. The trace still shows measured positive
  forward displacement and actual contacts, but
  fewer qualifying forward foot relocations, frequent
  backward foot motion and weak uphill command
  execution. A flat-gait fix does NOT solve uphill.
- [x] Original zero-yaw 10mm XML SHA-256 was restored
  to `f5ca6525a5d2ef8e608121d88fb613b5cc56e92b76bbbacba1b895ad17d0b754`.
  Both seeds **1701 and 1702 reproduced 3/4**
  fixed-world obstacle crossings; exactly the original
  09:49 pilot scores, with per-world success
  seed1701 [yes,no,yes,yes], seed1702
  [no,yes,yes,yes]. All 8 robot worlds
  physically contacted the obstacle. This is
  useful *input-contract-controlled repeatability*
  evidence, not a universal proof of determinism.
- [x] Existing foundational cold-start capability
  was audited: `emu train` without `--resume`
  and with a fresh external run directory uses
  `make_trainer(...).make_initial_state()`,
  draws weights from JAX seed, initializes optimiser,
  normalizer, teacher and physical worlds without
  reading the showcase actor. Supports 220-input
  legacy and `--sensor-actor` 232-input pathways.
  **This is source-path verification, not a new
  clean-room multi-million-step training result.**
- [~] Following user direction, committed
  `tests/test_v5018_cold_start_contract.py` asserting
  empty-parent training setup for BOTH 220/232 and
  opt-in sim2real, and optional
  `scripts/run_v5018_cold_start_acceptance.sh`
  which builds fresh 220/232 models, runs ONE
  actual PPO update, saves a checkpoint and
  exports a newly trained actor to a *scratch
  directory*. Both new assets are **not yet
  WSL-executed**, and do not read or overwrite the
  protected champion. Newly generated local
  root-level probe JSONs are `.gitignore`d,
  **not deleted**, to prevent untracked diagnostic
  files blocking clean-source `emu train`.

### Permanent requirement: ability to build a new brain from nothing

**The protected 260M policy is a reference for fast experiments,
never a required ingredient in the finished EMU system.**

1. **Cold**: fresh random actor/value networks + fresh observation
   normalization + a fresh curriculum/capability field, with
   independently chosen seed, no checkpoint, no showcase file
   and no preloaded terrain oracle. Start on easy flat-ground
   walking/command tracking (including heading/scorer exams),
   expand into previously measured slopes and low obstacles
   **only as held-out capability improves**.
2. **Warm**: optional compatible exact-checkpoint continuation
   for short R&D contests. If future terrain schema adds
   channels (e.g. a measured 312-input choice), warm-start
   must explicitly transfer compatible encoder layers
   and initialise new inputs safely; **never** direct-resume
   incompatible 232 weights and optimizer as if schemas
   matched. Evaluate warm and cold **independently**.
3. **Acceptance for every new observation schema**:
   brand-new-policy init → real PPO update → finite outputs →
   checkpoint/resume/normalizer → independently validated
   held-out walking progression → export → actual browser
   inference, **without a champion file**. Repeat both
   cold and warm paths in CI/local smoke. A one-update
   smoke checks lifecycle only, not learned walking.
4. **Release must not depend on a private foundation
   .pkl, archived .emupolicy, existing runs or hardcoded
   trained policy paths**. Showcase examples remain
   optional inference demos, never required for learning.

**Next engineering work still uses the frozen champion** for
quick controlled terrain/contact experiments; don't waste
training millions just to change a physics measurement.
When P1+P2 are ready, keep a separate cold-start path
through the *same* terrain curriculum, with baseline gait
gates before obstacle progression.

## 20 Oct 13:08 onward — P1 terrain-surface truth, feature implementation (GPU UNVERIFIED)

**User instruction:** continue using the protected 260M frozen champion as a
research benchmark, but keep every training/physics improvement capable of
cold-start learning from a brand-new seeded actor; keep this living checklist
and `docs/V5018_HANDOVER.md` updated on the branch.

**What is committed to `feature/v5018-terrain-awareness` (NOT `main`):**

- [~] New `emu/terrain.py::box_surface_truth` is one small, pure NumPy/JAX
  contact+geometry measurement module for a single fixed, yawed
  worldbody box and two real 18mm sphere foot geoms. It uses real
  MuJoCo/MJX `contact.geom1/geom2/dist` geometry-pair evidence,
  box local coordinates (actual `geom_xpos/xmat`),
  foot radius and near-top geometry checks. Distinguishes:
  **real floor-pair contact**, **box TOP contact candidate**,
  **riser/side contact**, and unsupported free swing. Side contact
  never counts as top support. Computes signed ball-bottom gap
  (negative below top), true per-surface tangential stance motion,
  terrain-relative body height, foot support fraction, and swing
  clearance. Contact pair/geometry is **not actual solver normal
  force/impulse**; the top flag is a *conservative support candidate*.
- [~] `emu/environment.py::EmuMJXEnv` has the last-argument opt-in
  `terrain_contact_mode='box_surface'` (default **`legacy`**)
  and `terrain_box_geom='v5018_obstacle_step'`. Constructor fails
  closed on a nonexistent/non-box, moving or non-horizontal
  obstacle and incompatible real-floor slopes. The opt-in
  reward/physical proxy consumes real top/floor support, box-relative
  height/clearance and tangential slip while preserving exact
  legacy measurements in worlds with **no box contact or footprint
  overlap**. This is *reward-only and diagnostics-only*, not an
  extra actor channel. Explicit sensor inputs and actor/network
  schemas remain **220/232 unchanged**; old `training_reward()`
  coefficients/arithmetic, PPO, doctor, policy and showcase untouched.
- [~] Added `terrain_geometry_active`, top/riser/floor candidate
  fractions, foot overlap and signed minimum toe gap to opt-in
  telemetry; these quantities are NOT fed into `_base_obs`.
  Body-low soft proxy still uses original floor geometry, and
  current forward ToF/flow estimation still assumes a single
  floor plane—**not yet a deployable obstacle preview**.
- [~] New synthetic physics tests in
  `tests/test_v5018_terrain_truth.py`: top vs riser,
  a ghost support with no contact pair, signed negative toe gap,
  genuine tangent slip, yawed footprint exclusion, and JAX/JIT
  parity. New `scripts/v5018_p1_surface_smoke.py` performs
  **actual GPU/MJX** comparison with the SAME off-course-box
  XML and seed: legacy versus box opt-in must produce
  **bitwise-identical** scalar rewards and 232 actor
  observations for three steps, then a 10mm box probe
  must activate finite physically relative diagnostics.
  **Neither this GPU smoke nor the new tests has yet been
  executed in the user's WSL.**
- [~] One fail-fast archive-producing driver
  `scripts/run_v5018_p1_terrain_acceptance.sh` invokes
  syntax, terrain JIT unit tests, the newly committed
  220/232 cold-start contract, historical float32 reward
  regression and actual four-world GPU/MJX parity/active-box
  tests. The entire script is an *experimental acceptance
  request*, not evidence of passing. All outputs go to
  `~/robot_lab/emu_exams/` away from Git and pretrained
  champion files.

**Important constraints deliberately still open:**

- This geometry pass is **one yawed immovable box on a level
  floor**, not an arbitrary course-bank, log/capsule, sloped
  floor + box stack or per-world adaptive geometry.
- "Top support candidate" requires a near-zero true
  contact-pair distance and sphere centred plausibly
  above the face, but no normal-force magnitude, support
  impulse, heat or actuator capability was measured.
- Default mode is legacy and the opt-in is not yet
  exposed by `emu train`: do **NOT** claim terrain PPO
  training or cold-start obstacle competency.
- Real moving-leg sensor occlusion/latency, per-world
  obstacle pose GPU throughput, CPU–MJX–WASM full
  six-servo parity and hardware servo safety remain
  blockers before P1 teaching or schema freeze.

**Next user WSL acceptance, optional while no GPU training is running:**

```bash
cd ~/emu-v5018
git pull --ff-only origin feature/v5018-terrain-awareness
source .venv/bin/activate
bash scripts/run_v5018_p1_terrain_acceptance.sh
```

Expected new evidence path:
`~/robot_lab/emu_exams/v5018_p1_terrain_YYYYMMDD_HHMMSS.tar.gz`.
Review ALL failures and diagnostics before enlarging the
terrain curriculum; do not interpret the *test script exiting*
as proof of a trained terrain-crossing brain. The separate
optional cold-start full PPO lifecycle proof is
`scripts/run_v5018_cold_start_acceptance.sh` (also not yet run);
it does not need the champion.

## 20 Oct 13:45 onward — actual P1 smoke results and browser challenge-ring branch

**User-run P1 archive physically inspected:**
`v5018_p1_terrain_20261020_134537.tar.gz`,
**tested source commit
`d683aa4f9dfa69bed53e2067886bae938a6ddfad`**.
Python **3.13.16**, JAX **0.12.0**, MuJoCo **3.16.0**,
NVIDIA RTX 4060 Ti CUDA **8188MiB**, canonical
`emu/robot.xml` SHA-256
`c1af72485e83de277efa1914a476519cebbebdc2beb1f03a10c475c9fadf3a64`.
The archive includes stage logs, source identity and the full Python
exception; no trained brain was loaded by these tests.

- [x] **Python syntax stage PASS.**
- [x] **6/6 physical terrain NumPy/JAX synthetic tests PASS (0.97s)**:
  genuine top vs riser candidate, no-contact hover, signed negative
  clearance, tangent slip, yaw footprint and JAX/JIT behaviour.
  Not actual contact force certification.
- [x] **6/6 historical reward and 220/232 from-zero setup
  contract tests PASS (0.10s)**. This proves the contract tests,
  not that a new random brain trained to walk.
- [!] **Real four-world MJX parity stage FAIL**, at
  `scripts/v5018_p1_surface_smoke.py:85`:
  `RuntimeError: box off-course changed actor observations`.
  The strict **bit-for-bit** observation comparison failed
  after one or more real GPU simulation steps; the original
  script did **not** print differing channel indices or
  magnitudes. Earlier in that stage, a comparison of the
  paired scalar reward on that frame did not trigger.
  Therefore **the exact cause is NOT yet established**.
  No actual active 10mm box-probe result is present.
  Do not call P1 GPU validated.
- [~] After the failed archive, the opt-in surface smoke was
  modified to log the *exact* frame, world, observation
  index, max absolute discrepancy and differing scalar
  count. Keep the PPO **scalar reward comparison bitwise
  strict**. For independently compiled CUDA kernels,
  allow only `rtol=3e-6, atol=2e-6` for 232-input
  observations; numerically significant observation
  drift is still a hard failure and triggers diagnostics.
  This **narrow tolerance is a hypothesis, not a
  verified fix** until the user's rerun. Never adjust
  physics score thresholds based on a result that
  hasn't been measured. If these limits are exceeded,
  inspect raw frame/channel values and rectify code.

### Browser v5018 challenge ring and sequential route — branch work

**User proposal:** reuse the existing 4m navigation circle; place
physically collidable obstacles of selectable demonstrated
difficulty along its **actual circumference**, brightly marked,
and add a second autonomous demo mode that follows successive
entry/exit checkpoints to go over the obstacles, while preserving
random-goal navigation and **read-only live policy loading**.

- [~] Added pure `emu/viewer_src/src/challenge-course.js`.
  Generates eight deterministic stations on the **same 4m radius**
  as `goal-navigation.js`; each has a tangential entry and far-side
  exit waypoint (16 ordered poses), and real MuJoCo worldbody
  box contact geoms at explicit selectable **0/5/10/15/20mm**,
  each with a distinct colour. Optional capsule logs
  have separate orange **experimental/unproven** labelling.
  The 20mm crossing seen in the user-run GPU frontier was only
  **3/8**, not a reliable "unlocked" motor capability.
- [~] `terrain.js::withTerrain(...,'course')` injects actual
  collision geoms with the original flat floor retained;
  the existing flat, 1° ramp and radial hfield park
  are unchanged. `sim.js` constructs a real WASM course,
  retains it on compatible policy hot-swap / rebuilds it
  for incompatible policies, and adds visible station rings
  and floating coloured labels (render-only annotations,
  not surrogate collision). A new **View: arena** camera
  preset frames the whole 4m ring and auto-selects when the
  user switches from orbit to the challenge course; it is
  render-only and does not change the robot's policy, IMU,
  collision or command targets.
- [~] The public/local/live `main.js` and `index.html`
  offer **Terrain → Challenge ring**, **step height**
  and **optional capsule log** settings, and a demo selector
  **Random destinations / Challenge circuit**. Circuit uses
  the SAME `GoalNavigator`, Web Worker pose planner,
  bounded yaw feedback and `sim.setDemoCommands` → normal
  policy inference. `goal-worker.js` explicitly handles
  `kind:'target'` without replacing a checkpoint/waypoint
  with a random goal. A world reset or manual override
  resets the route and releases the normal joysticks.
  Settings and legend live inside the mobile drawer.
- [~] A pure Node test
  `emu/viewer_src/tests/challenge-course.test.mjs`
  covers station locations, tangent geometry, deterministic
  challenge tiers/colours, injected MuJoCo XML, a
  no-silent-skipping planner contract and UI wiring.
  A pinned MuJoCo/WASM 3.15.0 constrained sphere test
  `emu/viewer_src/scripts/v5018_wasm_challenge_smoke.mjs`
  checks actual static box/log collisions at the raised
  contact heights; this is **not** six-servo policy/WASM
  walking parity. The acceptance driver
  `scripts/run_v5018_viewer_course_acceptance.sh`
  runs syntax + `npm ci` + all browser tests + headless
  collision probe + `npm run build` + `emu viewer-build`,
  and archives stage reports under
  `~/robot_lab/emu_exams/v5018_viewer_course_*.tar.gz`.
- [~] At time of this update, the generated obstacle
  geometry/entry→exit layouts were exercised through an
  isolated pure JS source check for all five tiers,
  **PASS**, and changed browser JS files had a
  syntax-only V8 parse **PASS**. **The complete Node/Vite
  bundle, actual WASM collision probe, rendered browser
  view and end-to-end circuit waypoint route have
  NOT yet been WSL or mobile tested.** Do not claim
  the demo proves strict no-detour crossing, performance
  across seeds or terrain-specific learning.
- [!] Live training is still a *read-only poll of
  COMPLETED and validated checkpoints*, not live
  per-frame GPU simulation mirroring, and this viewer
  has not added a privileged simulator-truth actor
  channel or changed the public protected champion.

**Next local commands, each stage independently useful:**

```bash
cd ~/emu-v5018
git pull --ff-only origin feature/v5018-terrain-awareness
source .venv/bin/activate
bash scripts/run_v5018_p1_terrain_acceptance.sh
bash scripts/run_v5018_viewer_course_acceptance.sh
```

The P1 scalar parity and 232 input drift should be
investigated separately from browser or WASM course failures.
If a command fails and returns nonzero, preserve the
archive and continue with the **other** independent test
when appropriate. No prior 34-stage frontier rerun is
needed. For a local visual trial after the viewer
acceptance, serve the **unchanged protected champion**:

```bash
emu viewer showcase/champion.emupolicy --no-open
# Open http://127.0.0.1:8765/ in browser
```

For a currently running training lineage, instead use
`emu viewer --live --root /path/to/that/runs/root --no-open`.
The viewer updates from validated checkpoints at its
existing polling interval; it does **not** control training
or fabricate a brain.

## 20 Oct 14:43 — second P1 CUDA/MJX run, fixed harness API

**Actual user-run uploaded evidence:** `v5018_p1_terrain_20261020_144341.tar.gz`;
**tested source commit**
`b93f0cf3d250deee2a39a38f164f96e3d58972b3`,
Python3.13.16/JAX0.12.0/MuJoCo3.16.0 CUDA RTX4060Ti,
same robot XML SHA-256
`c1af72485e83de277efa1914a476519cebbebdc2beb1f03a10c475c9fadf3a64`.
Archive was read directly; 4 stages logged with 3 PASS and last FAIL.

- [x] Python syntax PASS; terrain truth **6/6 PASS**; older
  reward + independent random-weight 220/232 setup
  **6/6 PASS** again. Not evidence of a complete cold-trained
  walking brain.
- [x] The real four-world zero-height **FLAT-control**
  legacy-vs-opt-in CUDA rollout completed all **three
  physics control frames** without triggering **exact
  scalar reward mismatch** or the strict tiny-precision
  actor observation gate. There were no printed
  `NOTE: off-course observation difference` diagnostics;
  this is consistent with bitwise identity in this run,
  but there is no successful machine-readable
  `mjx_surface_truth.json` because it was written only
  at the very end. No privileged geometry activation
  was detected off-course in those frames.
- [!] The NEXT, independent 10mm active obstacle
  **probe DID NOT RUN**: `state.data._replace(qpos=q)`
  raised `AttributeError: 'Data' object has no attribute
  '_replace'. Did you mean: 'replace'?` in MuJoCo 3.16
  `DataJAX`. This is a test-script API error, **not
  evidence of a failed physical-contact model**.
  It is different from the previous first-run
  observation assertion failure.
- [~] Fixed script on feature branch:
  `state.data.replace(qpos=q)`, matching the existing
  legitimate MJX `data.replace(...)` calls in
  `emu/environment.py`; preserved `EnvState._replace`
  (NamedTuple) elsewhere, unchanged.
  Persist exact 3-frame flat-parity report *before*
  running the separate 10mm active probe, so even an
  unexpected later exception preserves the completed
  flat comparison's numeric differences and SHA.
  The teleport-only probe now computes the actual
  right-foot sphere and static box pose *per world*,
  translates the free root so that the foot lies
  **1mm into the real 10mm box TOP**, and checks
  that the re-forwarded foot pose matches the
  requested fixture to within 1mm. Its
  **post-translation foot position** is used as
  the previous foot position to avoid a fabricated
  ~10m/s stance-slip penalty. The report captures
  top/riser contact-pair candidate fractions,
  supported stance, signed toe gap, and requires
  **at least one real sphere/box MJX contact
  pair**; torso XY-overlap alone is NOT enough.
  This is a collision diagnostic pose, absolutely
  **not** proof of physically feasible robot
  stepping. If real MJX contact allocation still
  reports none, investigate the contact model
  rather than relaxing the requirement.
  **The new source changes have NOT been run on WSL.**
- [ ] Immediate next gate: re-run
  `bash scripts/run_v5018_p1_terrain_acceptance.sh`
  on latest feature HEAD, upload the fresh archived
  JSON and logs. Do not rerun the finished 34-stage
  obstacle frontier or make a release claim from
  only the off-course comparison.

## 20 Oct 15:44 — narrow P1 PASS and root-cause audit of misleading metrics

### Actual new user evidence (not inferred)

**Source tested:** `0f7f02b2eadb78ac8816d60f07619bde37463c77`
on `feature/v5018-terrain-awareness`.
P1 archive named
`v5018_p1_terrain_20261020_154411.tar.gz` by runner;
user pasted complete console output. RTX4060Ti 8188MiB,
Python3.13.16, JAX0.12.0, MuJoCo3.16.0,
`emu/robot.xml` SHA-256
`c1af72485e83de277efa1914a476519cebbebdc2beb1f03a10c475c9fadf3a64`.

- [x] **4/4 runner stages PASS**: syntax, terrain NumPy/JAX
  **6/6**, cold-start/old reward contracts **6/6**,
  actual four-world GPU/MJX `actual_mjx_reward_parity`.
- [x] **Three true CUDA frames** of *off-course box* produced
  **bitwise identical old and new reward and all 232 actor
  observations** (`max difference=0`); correct identical
  XML SHA `cbc81bde...`.
- [x] Controlled 10mm contact fixture verified four geometrically
  targeted sphere poses (4–17nm target error),
  `terrain_geometry_active` in **4/4** and box–foot
  near-contact pairs in **4/4**; old 10mm course XML
  SHA `f5ca6525...`. All four reported
  top-contact fraction **1.0**, riser fraction **0.0**,
  support fraction **1.0**, stance slip **0.0**,
  and signed minimum toe gap approx **−1mm**.
- [!] These values do **NOT** demonstrate correctly learned
  stepping or stable load-bearing contact. The experiment
  *forcibly teleports* an unloaded robot so its right-foot
  sphere sits 1mm inside a stationary box, while it
  deliberately sets previous foot pose EQUAL to the
  translated current pose. Therefore **0 slip is
  guaranteed by construction** and 1.0 support merely
  reflects both ball geoms contacting or nearly contacting
  the box in this artificial state. It does not measure
  natural stride transitions, contact normal impulse,
  joint power, or dynamic falling/catching under motion.

### Actual viewer evidence (separate attached WSL console)

**Viewer run:** `v5018_viewer_course_20261020_154330.tar.gz`,
same source/unchanged champion SHA-256
`d3d319b82fbc5de331752f0f06753c57a776ac3212bcb6d386116c4f5f8a4862`;
Node24.22/npm11.19.1.
- [x] source, Worker and course-module syntax stages PASS,
  `npm ci` PASS; browser Node test stage **16/17** passed.
- [!] The ONE failed test
  `tests/core.test.mjs::viewer starts in lattice demo without obsolete controls...`
  contained a **historical assertion**
  `assert.doesNotMatch(html,/id="demoMode"|.../)`.
  That prohibition was added before the deliberate 2026
  Random/Circuit mode and is now contradictory.
  This is a **test-contract conflict**, not evidence
  of defective MuJoCo collider physics. Fail-fast runner
  stopped **before** headless WASM collision/build/install
  so browser runtime and rendering are still UNVERIFIED.
- [~] Updated that exact old core test to require **random
  selected by default, the circuit option, one existing
  neural actor/planner, and no obsolete advanced actor
  mutation controls**. Added a stronger *actual lattice*
  test requiring routes across every station's real
  projected box footprint. Isolated current source-level
  lattice execution with normal speed limits verified
  **8/8** routes solved and intersected their respective
  obstacle footprint. Still **no WASM or rendered browser
  result yet**. `esbuild` install script warning did not
  fail `npm ci`, and should not be mistaken for the
  actual test failure.

### Root defect found in P1 support classification

**This is substantive and not fixed by changing a test threshold.**

Previous `emu/terrain.py::box_surface_truth` called any
near-zero box/foot geom-pair a potential **top** support
when foot XY and sphere centre Z were close to the
raised surface. It did **NOT** consult the real collision
**normal** or the **contact point**. At a step edge,
a sphere pressing into the **vertical riser** can have
the same box-pair ID, near-top Z and XY footprint as a
true landing on the horizontal face. Thus a horizontal
collision could incorrectly award top support, affecting
actual PPO `support`, `height`, `swing_clearance`,
`stance_slip` and reward/termination. The four-world
stationary forced-top smoke cannot falsify that case.

- [~] Changed opt-in `box_surface_truth` to consume
  `data.contact.frame` and `data.contact.pos`.
  Uses MuJoCo frame's FIRST row (world-space normal),
  tests `|dot(normal,box_top_normal)| >= .88`
  regardless of contact geom ordering, and separately
  requires contact position within **4mm of the true
  top plane**, within the rotated box face's XY
  bounds. Invalid side/riser pairs remain reported
  as riser but **cannot** earn top support.
  Contact pair + geometry and normal remain *support
  candidates*, NOT measured positive normal force.
- [~] Added adversarial NumPy/JAX tests for (a)
  near-top sphere with **horizontal** collision
  normal, (b) vertical normal with contact point
  NOT on the box top, and (c) reversed valid
  normal orientation. New tests also reject a
  **+1mm separated** geom-pair (no touching under
  zero-margin physics), torso-over-box with both
  feet supported on the flat floor, and one
  box-supported foot after the torso moves
  PAST the box footprint. The terrain suite
  now contains **11 tests** (six prior,
  five new); none of the new ones have
  been executed in user WSL yet.
- [~] Further PHYSICS root causes addressed on the branch:
  (a) A MuJoCo pair within **+2mm positive
  separation** was previously treated as a
  supported stance even though the canonical
  geom margin is zero. The new P1 contact
  candidate requires `contact.dist <= 0`
  before using frame/point normals. The
  opt-in constructor fails closed on nonzero
  floor/foot/box margins or gaps, rather
  than silently applying the wrong rule.
  Still a candidate, NOT measured
  normal impulse/torque.
  (b) The old `torso_height` reference
  subtracted obstacle elevation whenever
  the **torso XY** crossed its 40mm strip,
  even if BOTH feet supported only flat
  ground; it also ignored a raised
  supported foot once torso XY passed
  beyond that strip. The corrected
  height reference weights the genuine
  classified support-surface elevations
  under the feet (equal count, NOT
  load-weighted), with
  `terrain_support_reference_elevation_m`
  separately reported. This does not
  affect default legacy reward.
- [~] Also added two CPU reporting tests in
  `tests/test_v5018_p1_dynamic_audit.py`,
  confirming that two contacting feet in
  one world-frame count as **one time
  frame**, and that post-reset contacts
  cannot fabricate pre-fall progress.
  Both tests are UNEXECUTED in WSL.
- [~] Added a STATIC OPTIONAL
  `record_terrain_truth=False` flag to
  `emu/ppo.py::evaluate_trace`, default unchanged.
  Only when true, adds *read-only* per-control-frame
  scalar reward, frozen actor action, real
  support, top/riser/floor fractions, signed toe
  clearance, and physical geometry activation to
  the exam trace. **No actor input channels,
  policy weights, base PPO training recipe or
  default evaluate_trace fields change**.
- [~] New `scripts/v5018_p1_dynamic_reward_audit.py`
  and `scripts/run_v5018_p1_deep_audit.sh`.
  Uses the **unchanged 260M 232-input** frozen champion
  as an OPTIONAL controlled research reference,
  real 10mm collidable XML, four active MJX worlds,
  one moving run with historical metrics and one
  with opt-in terrain physics; same operator command,
  seed, heading correction and sensor contract.
  Records genuine contact ONSETS, repeated real
  box contact world/foot frames, top vs riser candidate
  frames, physical support/gaps, actor-action/
  reward differences and time of any trajectory/
  terminal divergence. Requires actual contact
  during MOVEMENT before concluding success.
  This audit does **not** perform an RL update and
  does not modify or require a brain for cold-start
  learning. **NOT yet tested on the user's GPU.**
- [ ] Do not promote reward changes for terrain
  learning until these 8 new adversarial
  tests, actual normal/point MJX contact smoke,
  and moving frozen-actor A/B dynamic audit
  pass and their evidence is reviewed. A
  stationary near-zero collision score is
  necessary but **insufficient**.

### Next two INDEPENDENT WSL acceptance commands

```bash
cd ~/emu-v5018
git pull --ff-only origin feature/v5018-terrain-awareness
source .venv/bin/activate
bash scripts/run_v5018_viewer_course_acceptance.sh
bash scripts/run_v5018_p1_deep_audit.sh
```

Viewer: validates all Node tests, pinned actual WASM 3.15
sphere collisions, browser bundle and local installation;
course was NOT merged to public Pages.
P1: validates real contact normals/point, flat parity,
and actual moving 260M champion over 10mm in matched
read-only legacy/terrain scoring.
Both scripts archive to `~/robot_lab/emu_exams/`
and stop with partial results when a gate fails.
Use the fresh archived failure evidence, never fill
in missing results or report unexecuted success.

## 20 Oct 18:18–18:19 — two independent ACTUAL USER ARCHIVES RECEIVED

**Source verified in both archives:** Git commit
`f01507344589ee22f54f5ca6d1588b7e6582ae76`,
branch `feature/v5018-terrain-awareness`, canonical
robot SHA `c1af72485e83de277efa1914a476519cebbebdc2beb1f03a10c475c9fadf3a64`,
protected champion SHA
`d3d319b82fbc5de331752f0f06753c57a776ac3212bcb6d386116c4f5f8a4862`.
No files were mutated by the acceptance runners.

### P1 deep MJX archive — 5/5 stages PASS

File: `v5018_p1_deep_20261020_181914.tar.gz`,
real Python3.13.16/JAX0.12.0/MuJoCo3.16.0 CUDA
RTX4060Ti GPU with ~2.87GiB used at start.

- [x] syntax; **11/11 adversarial terrain-truth
  tests** (including positive-gap, real normal/point
  and torso-vs-foot reference), **2/2 moving-trace
  accounting/reset tests**.
- [x] `mjx_static_real_contact_and_flat_parity`:
  all 232 actor inputs and scalar rewards **bitwise
  equal for three flat control frames**; exact
  comparison difference **0**. Four-world artificially
  imposed 10mm top contact was recognised;
  support-reference elevation **10mm**.
  Its zero slip is imposed by the fixture, not gait.
- [x] `mjx_frozen_actor_moving_step_ab`: real
  immutable 260,112,384-step 232 actor, 4 MJX
  worlds, **240 control frames**, 0.10m/s vx,
  heading P gain 0.75, seed 1701, actual
  collidable **10mm** box. No PPO update and
  no terminal reset during the 240 frames.
  **Legacy** total: 394 world-frames with
  box contact, 425 contacting foot-frames,
  59 contact onsets.
  **Corrected P1** total: 388 world-frames,
  421 foot-frames and 60 contact onsets;
  396 terrain-active world-frames; 116 frames
  with top-normal candidates and **272**
  with riser-normal candidates. Per-world
  contacts legacy `131/90/105/68`,
  corrected `131/87/102/68`.
- [x] The corrected contact classifier is actually
  reacting to moving foot/step encounters,
  not simply tallying a manufactured stationary
  collision. Mean of per-world `support`:
  **0.889 legacy vs 0.691 corrected**;
  mean `swing_clearance`:
  **12.47mm legacy vs 9.73mm corrected**.
  Mean per-world reference elevation for the
  box-surface run **~1.17mm**,
  maximum per-world support elevation **10mm**.
  Many contacts are risers, so do not
  interpret these results as reliably
  walking OVER the step.
- [!] **Important causal-comparison limitation:**
  the two datasets are independent closed-loop
  simulations, not two rewards evaluated on
  the SAME measured physical trajectory.
  First reward mismatch at frame 39,
  first RAW actor network output mismatch
  above 1e-4 at frame 41, first XY mismatch
  above 1mm at frame 84. Maximum XY
  divergence **0.0709m**, maximum scalar
  reward difference **1.373**, and
  **693/960** world-frames with a changed
  reward. The reported max actor output
  difference **5.9076** is the **unclipped
  actor network mean** from `ppo.evaluate_trace`,
  NOT an actual servo/command difference:
  environment clips actions to [-1,+1].
  Since no episodes terminated during
  this experiment and scalar reward isn't
  fed to the actor, the large positional
  divergence is NOT proof that different
  reward values caused new behaviour.
  Different compiled numerical graphs or
  feedback amplification are possible,
  but **unproven**; a common-trajectory
  /same-state counterfactual is needed
  before concluding effect size.
- [~] Source audit after reading archive:
  the optional moving audit was clarified
  to compute both **raw actor mean** and
  **clipped command** discrepancies and
  to label its independent-closed-loop
  causal limitation. **These new fields
  are NOT present in the above user
  archive and are NOT WSL tested**.
- [ ] Physical force/impulse support,
  6-servo contact parity and actual
  course gate success NOT established.
  Normal/point and distance are better
  contact *candidates*, not positive
  load-bearing contact-force measurements.

### Viewer obstacle-course archive — 8/8 stages PASS

File: `v5018_viewer_course_20261020_181856.tar.gz`,
Node **24.22.0**, npm **11.19.1**,
`@mujoco/mujoco` pinned **3.15.0**,
Vite **7.3.7**.

- [x] three source syntax checks, npm clean install,
  all **18/18** Node tests (including new real
  lattice footprint-crossing and updated Random/
  Circuit UI contract), actual pinned-WASM
  collision probe for **floor, steps 5/10/15/20mm
  and experimental capsule log** (all **6/6**),
  production Vite build and local `emu viewer-build`
  install. **8/8 stage PASS**.
- [x] Physics fixture reported exactly one
  sphere contact in each WASM scene, with
  equilibrium sphere-height error
  **0.37mm** on flat and **~0.79mm**
  on each 5–20mm step/log. These are
  numerical contact constraints with
  acceptable stated **4mm test tolerance**;
  they are NOT precise CPU–MJX–WASM
  six-servo equivalence or reliable
  walking through the ring.
- [x] The local build was installed to
  `emu/webviewer_dist` in user's
  feature checkout. This does NOT
  publish Pages or overwrite the
  frozen showcase policy.
- [!] npm emitted nonblocking
  `esbuild@0.28.2` install-scripts
  approval warning; Vite also warned
  of a **737.68kB** uncompressed
  application JS chunk and externalized
  `module` import, but the build
  itself PASSED. Do not confuse
  warnings with errors.
- [ ] No real interactive desktop/mobile
  screenshot, policy navigating the full
  ring, or local `--live` checkpoint
  hot-swap run is in the archive.
  These remain manual browser-acceptance
  gates, not a prerequisite for
  claiming headless WASM stage success.

### Updated next meaningful engineering gates

1. **Visual browser check** of physical
   station colours, height labels, arena
   camera and Random/Circuit controls
   from `emu viewer showcase/champion.emupolicy --no-open`;
   test manual handoff and actual crossing
   progress. Strict no-detour physical
   crossing remains a separate scored exam.
2. **Same-state reward counterfactual:**
   use ONE frozen physical MJX trajectory,
   evaluate legacy vs corrected terrain
   scoring against the IDENTICAL physics
   frame/action, and report per-term deltas
   without independent actor divergence.
   Keep the from-zero policy lifecycle.
3. Extend physical support semantics with
   actual solver impulse/normal reaction,
   contact persistence, edge and mixed
   surface evaluation before teaching
   terrain reward to fresh PPO.
4. No 312-input freeze or large obstacle
   curriculum yet; maintain opt-in
   mode and protect champion/Pages/main.

## 20 Oct ~19:02 — actual desktop feedback: viewer circuit/scenery regression FIXED ON BRANCH

**User directly observed two issues AFTER the previous 8/8
headless viewer acceptance:** Challenge ring had **no grass or
trees**, and selecting that terrain still produced random targets
inside the circle instead of following the obstacle course.
This real browser observation OVERRIDES any implication that
the previous Node/WASM acceptance proved those UI behaviours.
Prior 8/8 acceptance was for source
`f01507344589ee22f54f5ca6d1588b7e6582ae76`,
NOT the new fixes. The corrected viewer has **NOT
yet been run in WSL or a browser**.

### Confirmed source root causes

- `terrain.js::withTerrain(mode='course')` **replaced**
  the entire park with challenge boxes on a bare
  original floor, while `sim.js::loadModel`
  generated `this.park`, grass mesh and tree crowns
  **only if** `mode==='park'`. That is why scenery
  vanished; no rendering setting or texture bug.
- `index.html` naturally defaults
  `demoMode` to `random`; the
  `terrainMode` dropdown's `change`
  handler previously only rebuilt terrain,
  **never selected the circuit**. Thus its
  existing target worker legitimately kept
  running `kind:'next'` random sampling.
  A hidden/permissive fallback in the
  initial mode wiring made this confusing
  despite the `kind:'target'` worker itself
  being implemented correctly.
- The original 16-point circuit had only
  approach/cross poses; the route planner
  could chord through the interior between
  stations, not keep a visually continuous
  circuit along the ring.

### Implemented branch-only correction (UNTESTED WSL)

- [~] Challenge mode now **combines the
  original actual MuJoCo hfield/grass mesh and
  tree cylinders/canopies** with the
  5/10/15/20mm boxes and optional real
  capsule logs. The original colliding
  `floor` plane is removed ONCE, as in
  park mode; no duplicate ground colliders.
  A gently feathered flat 4m annular track
  prevents small boxes from being buried in
  rising terrain. Physical box and capsule
  z positions sample that SAME generated
  park heightfield. The 10° outer hills
  and real tree collisions remain.
- [~] Forest generator retains 17 trees
  in deterministic positions: four inner
  grove trees at radius **2.25m**, visible
  from the whole-arena camera, and 13
  further trees outside the 4m lane.
  No trunk lies within **1.08m radial**
  distance of the obstacle circumference.
  `sim.js` now creates park grass/trees in
  BOTH park/course, and updates route
  overlays, sensor height references,
  terrain-slope indicator and obstacles'
  3D signs using physical park elevation.
  Tree switch and landscape slope control
  remain available in challenge mode.
- [~] New pure
  `emu/viewer_src/src/demo-mode.js` makes
  an **explicit terrain dropdown change
  to Challenge ring select `circuit`**;
  selecting other modes returns to
  `random`. A user can still deliberately
  select Random within the course. If the
  UI specifies circuit but no valid course
  is loaded, route setup now throws rather
  than silently falling back to random.
  Navigation uses the existing policy and
  Worker `kind:'target'`, never a scripted
  joint-action shortcut.
- [~] Circuit expanded from 16 to
  **24 ordered waypoints**: approach
  each obstacle, cross its actual
  footprint, then follow a midpoint
  arc on the 4m circumference.
  A persistent teal full-circuit line
  is displayed in addition to the
  immediate pose-lattice segment;
  HUD explicitly labels circuit targets
  and planning. The demo remains
  **not** a held-out no-detour test.
- [~] Browser tests now cover the real
  combined XML heightfield and 17
  collidable trees, visible inner grove,
  no trunks obstructing the ring,
  exact ground-relative obstacle height,
  course selection auto-setting
  circuit, manual random override,
  missing-course fail-closed,
  24 ordered worker targets and 24
  physically tree-aware A* legs.
  The pinned WASM smoke now includes
  a **7th physical contact fixture on
  a genuinely non-flat grassy hill**,
  beyond flat/4 boxes/capsule log.
- [x] An isolated pure-source V8 check
  of the updated JS **PARSED 10/10
  relevant modules**, generated true
  park+hfield+17 trunks+6 boxes+2
  capsule logs, found no duplicated
  floor, preserved 4 visible inner
  trees and 13 outer trees, and
  **planned 24/24 successive circuit
  legs** with physical park trees.
  This is NOT real WASM collision
  testing, rendering or an end-to-end
  browser route execution.
- [ ] **Mandatory next WSL run:**
  `bash scripts/run_v5018_viewer_course_acceptance.sh`.
  It rebuilds and INSTALLS the new
  `webviewer_dist`, not just tests source.
  Then start `emu viewer
  showcase/champion.emupolicy --no-open`
  and hard refresh `http://127.0.0.1:8765/`
  (Ctrl+Shift+R) so the browser does not
  reuse the old bundle. Manually verify
  grass, tree crowns/trunks, station
  markers, automatic **CIRCUIT**
  heading/teal route, 24-waypoint
  progression and manual joystick
  handoff. Screenshots or logs of
  failures are useful. The robot
  may fail to complete the ring if
  its locomotion capability is poor;
  that is **not** a random-target bug.

**Protected boundaries:** no main merge, Pages
deployment, champion modification, actor schema
change, PPO reward/training change or interruption
of a running experiment.

## Immediate next gates — move beyond an artificial P1 contact

**P1 old four-stage smoke is now 4/4 PASS,
but not sufficient for physical support certification.**
The new code adds actual contact-normal/top-point
classification, adversarial false-top cases, and a
genuine MOVING frozen-policy A/B rollout. Run the
bounded `scripts/run_v5018_p1_deep_audit.sh`
to test that real physics supports the intended
classification and to quantify true actor/reward/
trajectory divergence. Do **not** replace this
with another forced stationary top contact test.

The previous viewer source at `f0150734...` passed
8/8 stages including WASM and Vite build, but a
later **real browser test revealed** that course mode
discarded park scenery and kept Random targets.
New combined-grass/trees and auto-Circuit code has
been committed, pure V8/source-route-checked,
and MUST undergo a NEW Node+WASM+browser run.
See the 19:02 entry above. Do not reuse an
earlier green stage as evidence for this build.

**Viewer gate is separately UNVERIFIED on Node/WASM.**
The eight-station circle and two demo modes have been
implemented on the feature branch, with pure source
generation checks; launch
`bash scripts/run_v5018_viewer_course_acceptance.sh`
to validate Node, pinned MuJoCo/WASM geoms,
original browser regressions and Vite output.
Then visually verify in Windows/mobile that the
tiny physical step shapes, coloured signage,
random-goal and circuit routes are legible,
and that manual input still interrupts autonomy.
Use a current locally served policy or live
**validated checkpoint replay**, not the PPO
GPU simulation frame stream.

**After those independent gates**, audit the
frozen champion on the same 0/5/10mm real
collidable geometry with old-vs-new reward
metrics, held-out world-fixed crossing and
servo limits. Then extend support truth beyond
a single box, implement truly physical ToF
ray occlusion and latency, and compare
warm-start research against the independently
cold-started foundation/curriculum path.

**Do not** build a 312-input policy or begin
large terrain PPO until servo load, dynamic
contact forces, real forward scanning and
six-servo CPU–MJX–WASM parity gates advance.
No direct main/showcase/Pages changes.
