# EMU v5018 — Terrain-Aware Stepping: living branch checklist

**Last evidence review:** 20 October 2026. **Ref:** `feature/v5018-terrain-awareness`. **Status:** P0A partial / P0B partial; P1–P7 gated. **Never merge main, replace a champion or deploy GitHub Pages without explicit approval.** This is the only tracked living v5018 plan. The full initial motivation/experimental protocol is in the reviewed handover document, not an additional branch status file.

## What is measured and verified

### Fresh WSL baseline
- [x] Python **3.13.16**, MuJoCo/MJX **3.16.0**, JAX **0.12.0**, device **cuda:0**, from user-run v3 JSON.
- [x] Initial clean install ran 78 Python tests: **73 passed, 5 failed, 24 warnings in 428.56 s**. All five failures traced to stale 120M/220-input protected policy assumptions; actual immutable showcase is 260,112,384-step **232-input schema-2** policy with checkpoint schema 8.
- [~] Branch has reader + examiner support for both 220/232 and corrected test fixtures with schema/fingerprint validation. **The corrected full test suite has not been rerun and is not a PASS.**
- [x] Showcase's embedded robot XML byte-for-byte matched the repository model during inspection. Its weights and `showcase/` have not changed.

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

The campaign saves logs, `stages.tsv` and JSON under `~/robot_lab/emu_experiments/v5018_p0_YYYYMMDD_HHMMSS/`, plus a matching `.tar.gz`. **Upload that archive** (including failures) for the next review. No training starts; the script never modifies main or the protected showcase. Do not run alongside another GPU training job.

**Next implementation decision:** after local results, choose the simplest proven course primitive family and batch ceiling, then close remaining P0 gates before implementing terrain truth in P1.
