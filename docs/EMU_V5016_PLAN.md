# EMU v5016 — Directional locomotion and small-slope competence

**Status: first CPU-only implementation underway. Not yet integrated with PPO or GPU/MJX.** See [Pass 1 status](V5016_PASS1_STATUS.md); listed future tests are not all implemented.
**Source reviewed:** `refactor/tiny-emu` at `c8654181aa3b9fd455f750a7f422408e90f029e0` and accepted `main` at `40c5d0ce543ebbbc3690591695afddc18def8232`.
**Existing runs:** preserve the champion, 20M shared foundation, four independent 80M runs and original training commit; DO NOT edit their lineage.
**User field observation:** the old 60M champion could not manage a 1° slope. Treat this as a historical user observation, not a reproducible measured result until retested.

## 1. North star and the current defects

One six-servo dynamically unstable biped should obey `(vx, vy, yaw_rate)` and requested body posture while adapting to ground inclination. Do not manufacture separate uphill/sidehill/downhill brains. Also do not compare unrelated teaching targets as though they were competing solutions to the same task.

Actual v5015 behaviour discovered from source:

- `domain_randomization.MAX_SLOPE_DEG = 6.0`; severity scales it by `terrain_radius`. Current `slope` arm `0.12` trains with **at most ±0.72° per axis**, and `gentle_slope` evaluation `0.20` uses **at most ±1.2° per axis**. Both randomise roll and pitch; sampled magnitude and gradient direction must be logged, not inferred from the radius. The current common exam measures survival/tracking/height, not signed uphill progress.
- The fast MJX model simulates an **infinite planar incline equivalence** by tilting gravity over a flat contact plane; this is suitable for isolated constant-tilt physics but NOT ramp approaches, changing normals, bumps or steps. It also means `data.qpos[:,2]` is **not** uphill altitude; measure signed planar progress along the uphill basis and convert it to equivalent physical elevation via `progress * sin(slope_angle)`.
- v5015 `horizontal_tracking_motion` measures commanded velocity in the robot's **current heading**. Current `cmd_yaw` specifies **angular velocity**, not absolute heading. A robot can rotate downhill and later meet `cmd_yaw=0` while travelling forward in its new (incorrect) direction.
- `step_evidence` is foot lift, not completed strides. `support` is a foot-height proxy, not ground reaction force. Current survivor-weighted promotion cannot certify directed walking.
- The **existing v5015** deployable actor uses **55 × 4 = 220 inputs, six outputs**. The **v5016-candidate** contract is explicitly **61 × 4 = 244 inputs, six outputs**, in `emu/observation_versions.py`. These extra inputs are purposeful (ground grade forward/lateral, ground validity, heading error sin/cos, heading estimate validity), not a prohibition against improving perception. v5016-candidate is **not wired into PPO/export/viewer yet**. The existing 220-input weights cannot be loaded as a 244-input actor by renaming/padding a file; training and policy export require explicit versioning.

## 2. What counts as a successful test episode

Record uninterrupted pre-reset trajectories with explicit initial/final simulator positions, yaw, command stream, incline vector, true first-fall time, surface contacts and servo effort. No blending across auto-resets.

Independent reported fields per exam:
`slope_angle_deg`, `slope_bearing_world`, `requested_course_world`,
`requested_yaw_rate`, `actual_yaw_rate`, `actual_yaw_unwrapped`,
`desired_heading_unwrapped`, `heading_error_mae_deg`,
`heading_error_p95_deg`, `yaw_rate_mae_rad_s`,
`yaw_abs_error_1s`, `yaw_abs_error_3s`,
`signed_course_progress_m`, `uphill_progress_m`,
`equivalent_elevation_gain_m`, `cross_track_drift_m`,
`vx_mae`, `vy_mae`, `full_episode_survival`,
`first_fall_time_s`, `height_shortfall_m`,
`foot_lift_evidence`, `completed_stride_count` (only after calibrated),
`servo_shaft_abs_power_w`, `servo_over_rated_frac`,
`spring_potential_energy_j`.

The requested *course* is a world-fixed track or a properly integrated world-frame path for each command. A reversal or deliberate yaw turn is not automatically wrong; compare it to the appropriate evolving requested course. Uphill progress is positive only when the task actually requests climbing. For an explicit downhill or sideways task, grade progress along **that** requested direction; never force uphill progress regardless of operator intent.

### Score rules, before any weight tuning

- **Hard requirements:** enough uninterrupted survival to execute the task; signed progress in the requested direction; heading deviation bounded for zero-yaw tasks; vx/vy/yaw command quality. Missing or nonfinite metrics fail closed.
- **Report separately:** downhill heading exploits, lateral drift, falling, excessive torque/load, torso crouching and foot-lift-only behavior.
- No winning based on survival alone, instantaneous yaw rate alone, reward alone, or motor power alone.
- Engineer-defined threshold candidates, **not observed passes:** 7 s exam (350 steps at 20ms), then 14 s (700 steps) to expose delayed yaw drift. Consider 1° directed trials the initial milestone; do NOT set an unconditional 10° pass gate.

## 3. Proposed slope/command exam matrix (small slopes first)

Start with `0, ±0.25, ±0.5, ±1.0, ±1.5, ±2.0` degrees along commanded track; only after progress at 1° consider `±3°, ±5°`. Do **not** interpret 10–15° as today's baseline. Where feasible examine near-field equivalent slopes separately from actual rotated geometric slopes.

For each absolute slope angle run:
- A. requested **uphill**, zero yaw rate, modest `vx = +0.10 m/s`; independently `+0.20 m/s`.
- B. requested **downhill**, same speeds.
- C. requested **cross-slope left and right**, `vx = +0.10`, `vy = 0`; then `vy = ±0.05` while preserving requested heading.
- D. requested **backwards** (`vx < 0`) and sidestep (`vy != 0`) against the same independent direction oracle.
- E. requested **constant-turn** (`yaw_rate = ±0.2 rad/s`) and **turn-stop-hold**; intended course is a trajectory, not a fixed heading.
- F. **command transitions** between straight, lateral, turning and stop; the controller must never reward an unrevealed future command.
- G. **positive and negative slope gradient orientations**, including robot initially facing sideways, uphill and downhill; world progression oracle must not depend on robot's current heading.
- H. **8 independent reset/physics seeds per condition** for an initial pilot; expand to 32–64 after narrowing promising policies, balancing slope signs and start headings.

***Pilot size:*** fixed 256–512 world equivalent evaluations on GPU after CPU oracle tests, with deterministic seeded conditions. Monitor compile/memory; avoid launching a huge `angle×heading×seed` combination in one JAX compilation. Preserve conventional held-out flat-ground `core`, `dynamic50`, `robust40` and recovery panels as regression controls.

## 4. CPU-first anti-cheat oracle and heading-loop tests

These are specific tests to implement; names indicate *proposals*, not tests currently passing.

### Pure geometry and directional score — proposed `tests/test_v5016_direction.py`

| ID | Test | Expected result |
|---|---|---|
| D01 | `flat_world_progress`: level slope + commanded straight travel | course progress equals signed XY displacement; uphill gradient undefined/zero, no NaN |
| D02 | `uphill_positive_downhill_negative`: opposite traversals on 1° gradient | +progress for uphill, -progress for downhill; yaw changes cannot flip world gradient |
| D03 | `rotates_then_descends`: policy turns 180°, walks forward downhill, ends with zero yaw rate | **FAIL** requested uphill progress/heading despite good body `vx` and current yaw rate |
| D04 | `sideways_traverse`: deliberate cross-slope task | passes own course progress while showing ~zero uphill progress; must not be penalized for obeying command |
| D05 | `explicit_reverse`: backward command executed uphill without turning | correctly credited; uphill bearing separate from body forward axis |
| D06 | `gravity_equivalence`: 0°, ±0.5°, ±1°, ±2° slope vectors | downhill gradient is signed gravity tangent projected onto contact plane; zero angle is well-defined |
| D07 | `elevation_equivalence`: planar translation on tilted-gravity ground | uphill equivalent elevation gain = signed slope-distance × sin(absolute slope angle); **not** world-Z |
| D08 | `wrap_heading`: 179°→-179° and multiple turns | unwrapped heading continuous, wrapped error takes shortest angular difference |
| D09 | `oscillatory_yaw`: +0.3/-0.3 yaw-rate errors alternate | signed mean near zero but 1s/3s mean absolute error and p95 reveal failure |
| D10 | `reset_boundary`: auto-reset after early fall | no teleport displacement / newborn heading blended into previous episode; missing progress fails closed |
| D11 | `spawn_velocity_grace`: nonzero inherited initial qvel | exclude nominated settle window from progress qualification; don't reward the initial ballistic launch |
| D12 | `negative_controls`: frozen standing / foot tapping / heading-circles / ballistic jump / backwards downhill | none may earn an uphill-walk pass despite survival, foot lift or short-term body speed |
| D13 | `positive_controls`: perfect directed walking at 0°, 0.5°, 1° and controlled turns | pass with expected signs and numerical tolerances independent of policy |
| D14 | `same_exam_conditions`: compared policies | exact common inclination, start heading, commands, seeds, duration, noise and drift; fail on differing protocol or stale hash |

### Heading hold P controller — proposed `tests/test_v5016_heading.py`

**Outer-loop law:** `desired_heading += commanded_yaw_rate × dt`, `error=wrap(desired_heading-measured_heading)`, `policy_yaw_command=clip(commanded_yaw_rate + kp*error, ±safe_yaw_limit)`.

| ID | Test | Expected result |
|---|---|---|
| H01 | zero-rate command + no disturbance | zero correction |
| H02 | zero-rate command + 5°, 15° drift | correction rotates toward target; gain sign correct |
| H03 | ±90° offset and ±π wrap | shortest correction and safe saturation |
| H04 | constant commanded turn | target rotates at exact requested rate, without forcing original heading |
| H05 | turn command stops | freezes current target; robot can settle on intended final direction |
| H06 | time-step changes / 20ms tick / command ramp | integral heading evolution consistent with timestamps; no double integration |
| H07 | reset, new operator heading reference, lost yaw-estimate quality | defined non-kicking reinitialisation; no stale heading or abrupt kick |
| H08 | gyro bias + sample delay + noisy yaw estimate | bounded correction; log estimate bias and authority/saturation |
| H09 | clamp/actuator authority and low speed | yaw command within actor's trained envelope; controller cannot demand impossible instantaneous turns |
| H10 | controller on/off on identical evaluation seeds | compare improvement **and** damage to tracking/turn responsiveness; keep off-mode to inspect underlying policy |
| H11 | policy input causality | PPO sees issued command and, in the new versioned contract, causal heading/terrain state; never future target information. Existing v5015 policies remain 220/6, future v5016 policies may be 244/6 |
| H12 | deployment parity | same heading calculation in simulator and physical interface; yaw is relative/gyro-integrated unless an absolute heading sensor is added |

**Important:** A gyro+accelerometer does not give drift-free absolute compass yaw. Heading hold from integrated gyro is a **relative-heading** stabiliser over bounded windows, not a guaranteed lifelong north-hold. Reset references and assess synthetic gyro bias/latency; don't use sim-truth orientation for the deployed controller. The sim oracle may use true orientation only to **score** the policy independently.

Use `Kp` as a bounded, tested experiment setting, not a new pile of tuning parameters; sample conservative trial gains (e.g. 0, 0.5, 1.0, 2.0 per second), selecting after on/off evidence. Distinguish heading-controller competence from underlying PPO yaw competence. Do NOT gate activation during training on an arbitrary yaw milestone until checked.

## 5. Terrain observability / triangle of downward ToF sensors (read-only first)

Three noncollinear downward beams with calibrated sensor origins and pointing vectors define up to three ground-intersection points. Fit local plane normal via a cross product and combine it with IMU-estimated gravity to recover ground pitch/roll relative to gravity and a local uphill bearing.

Proposed `tests/test_v5016_ground_plane.py`:
- G01 perfect flat/±0.5°/±1° planar rays: orientation and bearing accurate within tight numerical tolerance;
- G02 chassis leans 10° on flat ground: recovered **ground slope ~0°**, not chassis pitch;
- G03 tilted ground + independently tilted chassis: recovered slope equals ground, not torso;
- G04 missing hit, nearly-collinear beams, rear beam intersects leg, out-of-range returns: invalid confidence; never NaN;
- G05 one biased/noisy sensor: detect unreasonable plane estimates; with only three hits redundancy is limited, optional fourth sensor allows robust fitting;
- G06 ToF delay and IMU accel dynamics: time alignment and filtering bounded; accelerometer alone is unreliable during fast acceleration;
- G07 actual geometric ramp transition versus gravity-tilt infinite plane: record expected disagreement and use physical ramp tests before hardware-readiness claims;
- G08 v5015 versus v5016-candidate obs schema: the declared 244-input contract may include slope and heading measurements only once **the complete environment→normaliser→PPO→export→viewer→hardware chain** is wired and tested. Keep the 220-input v5015 policy intact; no smuggling terrain data into old command slots.

**Do not make hardware installation a prerequisite for v5016's first heading tests.** We can start with read-only simulation oracles, then compare observability of IMU-only versus IMU+ground-plane once sensor dynamics are validated.

## 6. Real MJX, evaluation and regressions (after CPU oracles)

- P01 real 0° and ±0.5°/±1° gravity-vector worlds match expected sign of forces and intended climb; test both forward/uphill and yaw-reversed/downhill controls.
- P02 full-state rollouts 350 then 700 control frames with true first-fall/timeout separation; no reset contamination.
- P03 commanded turn/stop, sidestep, reverse and transitions preserve command semantics with explicit **v5015 220→6 versus v5016 244→6** schema selection; verify both paths independently.
- P04 counterfactual stress cases: aggressive heading P saturates, steep slope 2° causes fall, poor velocity tracking, large lateral drift, crouched policy; fail honest task-specific gates.
- P05 current v5015 fixed-panel baseline regression: hold `core`, `dynamic50`, `robust40`, recovery against untouched policy with and without P; no silent performance regression.
- P06 measure computational/compile overhead of batched heading state + separate exam telemetry, no training-process slowdowns beyond an explicit reviewed threshold.
- P07 policy hash and source identity persisted for all screens/results; `results.md` and `results_bundle.zip` include by-angle/direction matrices, heading errors, signed progress and representative worst-case trace references.
- P08 compare **original 60M and all four 80M policies** on identical v5016 read-only exams **before** training a new one. Treat model/schema incompatibilities as blockers, not something to bypass.
- P09 real physical bench validation after sensor integration: slope standstill IMU bias, ToF errors, servo stalls, disconnects; no autonomous steep-slope trials without controlled fall protection.

## 7. v5016 work passes / no code sprawl

1. **Pass A — examiner truth:** one slope/direction task specification and pure CPU numeric oracles (D01–D14), no PPO edits. World-relative requested course, real first-fall horizon, anti-cheat controls. Gate: synthetic bad controllers demonstrably fail, good controls pass.
2. **Pass B — observation/heading loop:** heading P as a tiny well-scoped pure controller within existing command ownership, plus H01–H12. Initially **evaluation-only on/off**; no reward changes. Gate: yaw/velocity commands remain correct for turn, reverse, sidestep, transition and level ground.
3. **Pass C — bounded real physics:** P01–P08 MJX/GPU tests at 0°, 0.25°, 0.5°, 1°, 1.5°, 2° (3°/5° stretch). Benchmark 60M champion + four 80M candidates, report actual competence boundary and failure modes. No 10° claims.
4. **Pass D — sensor estimator and actor v2:** G01–G08, three-ray/IMU ground plane first as read-only; then wire the declared 244-input v5016 actor with explicit policy schema, observation normalisation, exporter, viewer and real-hardware parity. Gate: tilted robot on level ground cannot be misdiagnosed as inclined ground; old policies remain loadable in their original runtime.
5. **Pass E — curriculum experiment:** if exams show need, compare four ways to learn **the same directional slope tasks**: mixed inclines, progressive incline ladder, competence-adaptive frontier, interleaved level+incline rehearsal. Fix reward, actor, commands, terrain envelope, physics, parent, seeds, budget and held-out exam; vary only training schedule. Isolate P on/off as a **separate ablation**, not an arm confound.
6. **Pass F — staged GPU budget:** shared source-compatible foundation, short pilot forks at a few million steps, checkpoint review, then longer equal-budget forks only if learning is real. An example later 20M shared + four 60M continuations costs 260M total, not a default automatic commitment. Run multiple seeds for finalists and keep old checkpoints untouched; source-commit/optimizer lineage must be preserved. No pretending a new-source optimizer can transparently resume a schema-mismatched v5015 checkpoint.
7. **Pass G — release:** eliminate superseded components rather than wrapping them; report ownership in `WHERE_THINGS_LIVE.md`; tests and real WSL GPU acceptance; trained viewer and physical-controller parity; optional policy schema migration only if independently proven.

## 8. Release evidence and success claims

- **Read-only examiner proven:** negative-control downhill-turn cheat fails; positive commanded climb/turn/backward/sidestep controls pass; unit tests + GPU physics exercises run.
- **Heading controller worthwhile:** on/off paired evaluations improve heading/course following *without unacceptable flat-ground survival or turn tracking loss*. If not, don't enable it.
- **First real terrain milestone (engineering target, not guaranteed outcome):** meaningful directed progress and survival at **1°**, with no downhill-turn trick, across both slope signs and at least several reset seeds; preserve level-ground quality.
- **Later:** 2°, 3°, 5° progression, actual geometric ramp transitions and sensor-equipped deployment. Ten degrees is NOT an initial benchmark.
- **One useful artifact:** `results.md` plus bundle with per-angle/direction matrices, raw traces, per-policy hashes, source commit, real gait/progress/fall metrics and known measurement limitations.
- **Protected:** `main`, deployed 60M champion, current four 80M run roots, legacy 220×6 v5015 policy contract and Pages manual publishing. **Future** v5016 trained policies may use 244×6 with explicit schema/compatibility tests.

No existing policy reward, simulator, actuator XML, command semantics, curriculum or running checkpoint is changed by this plan.
