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

### Fresh WSL baseline
- [x] Python **3.13.16**, MuJoCo/MJX **3.16.0**, JAX **0.12.0**, device **cuda:0**, from user-run v3 JSON.
- [x] Initial clean install ran 78 Python tests: **73 passed, 5 failed, 24 warnings in 428.56 s**. All five failures traced to stale 120M/220-input protected policy assumptions; actual immutable showcase is 260,112,384-step **232-input schema-2** policy with checkpoint schema 8.
- [~] Branch has reader + examiner support for both 220/232 and corrected test fixtures with schema/fingerprint validation. **The corrected full test suite has not been rerun and is not a PASS.**
- [x] Showcase's embedded robot XML byte-for-byte matched the repository model during inspection. Its weights and `showcase/` have not changed.

- [x] Existing showcase policy was **reported by the user** to ascend/descend slopes of approximately 10° in the MuJoCo/WASM viewer. This is practical viewer experience; a controlled 10° held-out MJX/CPU exam and command-following retention threshold are **not yet measured**. Don't start the terrain curriculum by declaring mild slopes unlearned.
- [~] A gyro-integrated heading-hold P-controller exists in earlier EMU code; matched frozen-champion heading ON/OFF slope/directional exams, no-detour corridor and anti-cheat crossing rewards are **design requirements**, not implemented/validated terrain success.
- [~] Current branch has new `scripts/bootstrap_wsl_v5018.sh` and updated `docs/INSTALL.md` for fresh Ubuntu; the bootstrap script is **committed but not yet rerun clean-room**, so documentation does not constitute an installation PASS.
- [!] User currently awaits the big `run_v5018_p0_campaign.sh` results. No uploaded stage archive exists in this conversation as of the 20 Oct documentation pass. **Do not re-run or `git pull` on that actively running source checkout** until its existing archive is preserved. Later doc-only commits cannot retroactively change the tested SHA.

### Collidable contact and ray physics (user-run v3 JSON)
- [x] **5/5** contact cases with correct CPU and MJX geom pairs: flat, 30 mm raised box, 30 mm real lower landing, tilted box, low capsule log. CPU/MJX final centre-Z maximum discrepancy **0.000590 mm** (0.59 µm). This is constrained-ball resting position evidence, not gait/dynamic load agreement.
- [x] **12/12** CPU `mj_ray` versus JAX plane/oriented-box/capsule first-hit distances matched within **0.000046 mm**. Includes actual step faces and both log top and log side. This uses collision primitives; no oracle heightfield shortcut. These rays still have idealized static extrinsics; sensor calibration/occlusion by robot limbs unmeasured.
- [x] CPU contact support magnitudes near 0.0981 N for the 10 g sphere (incline ~0.1000 N). **MJX force equivalence is unmeasured.**
- [x] Two spheres at separate X positions in the **same shared terrain topology** landed correctly on raised step and floor. **This does not establish varying geometry between worlds.**

### Full EMU MJX GPU *stepping* (user-run 256/512 JSON)
- [x] Native `sim2real_232` EMU environment completed **256** and **512** worlds on GPU in isolated subprocesses:
  - 256 flat: **9,541 env-steps/s**; step geometry present: **9,717 env-steps/s**
  - 512 flat: **15,290 env-steps/s**; step geometry present: **16,690 env-steps/s**
- [!] Both configurations reported **zero contacts with the inserted step at the last frame**; these results only establish overhead of geometry presence in short 8-frame sequences, **not collision-loaded throughput**. Step and flat rewards matched; reward is not yet terrain-relative. The small apparent speed advantage is noise, not a physics optimization claim.
- [x] Reported XLA allocator peak ~201.4 MB in all cases, `bytes_limit=6,440,353,792` bytes. This is **not** total VRAM usage and does not prove 2048-world training fits.
- [ ] Quantify 1024 and 2048-world GPU stepping, **actual obstacle foot contacts**, VRAM and *PPO update* FPS independently.

## Implemented in feature branch; **not yet executed**

- [~] `scripts/v5018_dynamic_contact.py`: four genuine time-resolved moving-sphere controls (slow/fast step-face strikes, high-clearance no-hit control, vertical onto-top hit), first-impact timing, CPU impulse proxy, MJX contact flags. Constrained slider, no learned gait.
- [~] `scripts/v5018_per_world_terrain.py`: vary actual `geom_pos` on two MJX worlds with identical collision topology; swap positions without changing compiled shape; CPU references. If unsupported, choose a small static course bank. Does not prove arbitrary mesh topology is cheap.
- [~] `scripts/v5018_scanner_coverage.py`: CPU MuJoCo-first-hit two-lane forward scan over 3 chassis heights × 3 pitches × 3 rolls × 2 lanes and two beam proposals, with valid-range, riser/top hazard labels and first-detection lead. **Candidate beams only**, no physical module claim, and currently no articulated leg self-occlusion/noise.
- [~] `scripts/v5018_clearance_envelope.py`: exploratory robot.xml FK of actuated joints, forward reach and rising foot relative to support under optimistic no-load speed limits; **not a dynamic clearance envelope or servo torque proof**.
- [~] `scripts/v5018_batch_benchmark.py` now includes an added **'contact'** mode: a shallow 5 mm pad deliberately under the right ball foot, requiring true contact samples, and the earlier 'flat'/'step' controls; tests 256→512→1024→2048 while capacity allows. All three use the full robot environment; **not PPO update throughput or walk-over success**.
- [~] `emu/viewer_src/scripts/v5018_wasm_contact.mjs`: headless actual MuJoCo/WASM 3.15.0 five-fixture final-height comparison to the CPU 3.16.0 JSON, preliminary engine-gap evidence only (not full browser/actor trajectory).
- [x] Pinned the viewer's `@mujoco/mujoco` in **package.json and package-lock.json to 3.15.0**. This preserves the previous resolved wasm release and prevents future implicit `latest` drift; npm runtime regression remains untested.
- [~] `scripts/run_v5018_p0_campaign.sh`: a **single local sequential campaign**: Python syntax/full regression, npm ci/tests/build, GPU doctor with a small real PPO-update smoke, v3 ray/contact, WASM check, moving impacts, heterogeneous-position MJX, sensor scanner, clearance FK, and 256→2048 native GPU benchmarks. Independent steps continue after failures; GPU batch escalation stops at first failing size. The script logs every exit and creates a compressed archive outside the source tree. **No campaign stages run by assistant** (local runtime lacks MuJoCo and network access).

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

## Run the combined local campaign

```bash
cd ~/emu-v5018
git pull --ff-only origin feature/v5018-terrain-awareness
source .venv/bin/activate
bash scripts/run_v5018_p0_campaign.sh
```

For a **future new campaign after the current run ends**, the command above starts a fresh experiment; don't run a second copy concurrently. The campaign saves logs, `stages.tsv` and JSON under `~/robot_lab/emu_experiments/v5018_p0_YYYYMMDD_HHMMSS/`, plus a matching `.tar.gz`. **Upload that archive** (including failures) for the next review. No long training campaign starts (the GPU doctor does execute a small PPO-update smoke); the script never modifies main or the protected showcase. Do not run alongside another GPU training job.

**Next implementation decision:** after local results, choose the simplest proven course primitive family and batch ceiling, then close remaining P0 gates before implementing terrain truth in P1.
