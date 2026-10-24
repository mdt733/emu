# EMU — Unified Capability Growth (v5024 design contract)

**Status: canonical design, replacing the obsolete v5023 ten-hour A/B campaign.**
Owner decision 2026-10-24: stop the unchanged flat/1cm control run and do not
start any other run that excludes real slopes, commanded motion frontiers,
recovery, or finite physical terrain. This is a **single shared PPO actor**
that learns in **2048 simultaneous, specialist-emphasis physical worlds**.
No policy, checkpoint, main, showcase or champion is promoted by this design.

## Mental model

The policy is trained by ALL worlds every PPO update. Every world has a
persistent emphasis (a **teacher-only role**, not a special neural policy):
commands, resilience, continuous slopes, discrete steps, transitions/posture,
or **control+efficiency**. Every episode includes controlled exposure to other
dimensions, generally below their demonstrated frontiers. No specialist is
allowed to turn the others off globally. Specialisation is in challenge
severity, episode sampling and evidence attribution, not different networks
or incompatible observation schemas. Hardware physical/safety constraints
remain hard limits. The other limits are **adaptive competence frontiers**,
never silent permanent ceilings.

Illustrative minimum allocations; they can adjust to uncertainty and
learning evidence but cannot starve a capability.

| Emphasis | Fraction | Nominal 2048 worlds | Harder factor | Gentler but nonzero factors |
| --- | ---: | ---: | --- | --- |
| Command | 20% | ~410 | vx / vy / yaw and mixtures | recovery, slopes, finite steps, posture |
| Resilience | 20% | ~410 | pushes and bad starts / recovery | command combinations, terrain |
| Continuous slopes | 20% | ~410 | grade (up / down / sideways) and direction | commands, pushes, steps |
| Discrete steps | 20% | ~410 | real riser + descent + upper support | slope, velocity, resilience |
| Transitions/posture | 10% | ~204 | acceleration / reversals / commanded chassis posture | the other axes |
| Control / efficiency | 10% | ~204 | better walking at known-safe tasks | gentle commands, recovery, slope and steps |

Control worlds are **not idle easy worlds**. They practise safe combinations
with a dedicated, *bounded* smoothness/motor-effort teaching signal, stable
recovery, joint safety and command accuracy. The quality payment is conditional
on meaningful motion and survival so freezing cannot save "energy" for free.
Same gait/no privileged channels. Efficiency measurements accompany *all*
cohorts; control worlds receive the stronger lesson and provide a stable
comparable longitudinal reference. Mechanical shaft work is NOT calibrated
battery joules; evaluate work/m, movement, tracking, gait jerk, support,
slip and recovery simultaneously.

## Preserve main's successful cold-start gait

Direct audit of `main` (Git SHA `2ec9e9db0a816c658131311129ef89ec7a595a12`):

- `foundation_config` sets **20,000,000 environment steps**, **2048 MJX worlds**, `seed=733`, 32-frame PPO rollout, **4 epochs ×8 minibatches**, LR **2e-4**, at most 700 physical control frames per episode. The same shared PPO model and 8 transparent reward payments are NONNEGOTIABLE foundation invariants.
- The controller uses **70 steady-only bootstrap updates** (~4,587,520 env-steps at 2048 worlds), then gradually mixes STEADY, TRANSITION, RESILIENCE with a per-regime allocation floor. It does NOT make the robot master everything at frame zero.
- Gentle *initial* command seed `vx=+0.020…+0.060 m/s`; `foundation_config` ultimate per-task command scale is `vx_hi=0.40`, `vx_lo=-0.12`, `vy=0.14`, `yaw=0.90`. Those are **bounds on curriculum sampling, not a declaration that the newborn policy can track them**. Physical command tracking and supported gait develop progressively. Foundation `phase_speed_gain=0.75` and `push_max_force_n=0.70`, not the much more aggressive mature-profile gain/push.
- The main governor uses closed-loop pooled episode-count fall EMA (not noisy per-rollout ratio), proportional difficulty growth and faster retreat, a **shared total per-update difficulty growth budget** so all aspects cannot jump together, and hard global rollback only for sustained collapse. These protect the learned gait from an exploding task distribution.
- From the first update, the policy learns locomotion against *real* contact, with contact/clearance/smoothness/stepping/activity and early-death shaping. The first foundation policy had **no v5023 finite-step/360 actor**; therefore there can be no exact neural/observation parity with a future terrain_360 cold run. Preserve *physics, reward, algorithm and progression doctrine*; report differences explicitly.

**How unified improves, rather than discards, main:** the six-world-group shares are the eventual target, not an immediate cold-start recipe. The actual activation fraction is a **smoothed earned ramp** derived from the proven steady difficulty: 0 while steady difficulty ≤0.10; rising linearly toward 100% as it reaches 0.32. Early nonactivated births keep the main-style task, and their rewards have no new control-specific motor lesson. Once activated, each episode receives ONE teacher-only emphasis without replacing the network. When competence retreats, exposure can retreat too. Unlike a hard 20M terrain unlock, extremely easy real slope and small step practice should begin **when steady support and recovery are demonstrated**, without starving the control gait. The exact ramp is an experimental design choice to validate against main-style 20M cold-start quality, not a proven result.

**First 20M acceptance gate** (non-negotiable): compare a true fresh unified 20M actor against historical foundation reference on matched flat-gait, speed tracking, recovery, action jerk, attitude and mechanical shaft work/m; require no substantial collapse. Also verify specialist occupancy grows from near 0 to useful nonzero values, all six specialty roles receive actual completed episodes by 20M, and real incline+steps were present once earned. If the actor walks worse, correct the curriculum before extending to 100M+. Never assume a theoretical architecture automatically performs better.

## Diamond / octahedral command boundary (not a rectangle)

For a requested vx,vy,yaw, compute independent signed axis limits:
positive and negative vx are distinct. Normalized command magnitude in
each axis: x = |vx| / Vx(sign), y = |vy| / Vy, z = |yaw| / Vyaw.
The **L1 envelope** is x+y+z<=1, an octahedron (vx/vy slice: diamond).
Maintain directional *local* evidence: passing at half forward + half
sideways must not automatically expand both pure-axis maxima. Draw samples
across the interior, at the edge, and just outside (e.g. radius 0.85–1.12),
with a mix of pure axis, 2-axis and 3-axis directions to avoid starving
vertices. Signed directions/heading and yaw combinations are real commands.
Cap extremely large requests by actual mechanical constraints, not arbitrary
training-profile defaults. Frontier growth earns evidence; headroom exists.

## One evidence loop per emphasized dimension

1. On each episode birth, attach teacher-only birth context: focus, requested
   command vector and severity, real floor angle/direction, step height/up-down
   and disturbance. These are NOT new actor observation channels.
2. The policy sees only existing realistic 360 inputs (command, servo/IMU/
   body-rate/flow, delayed 4x4 ToF with independent validity).
3. On episode end, score **real achieved result**, never PPO scalar reward:
   enough survival/time, accurate command following, supported feet, no falls,
   acceptable drift, smoothness/effort. A terminated episode supplies failure
   evidence rather than an optimistic partial success.
4. Attribute evidence to the actual relevant direction + tested severity
   **within the episode's context**. Per-bin accumulators accumulate multiple
   completed episodes; no single lucky crossing grows a frontier.
5. Increase only the demonstrated boundary slightly if sustained, high-quality
   performance near/above it beats a confidence threshold. Hold when there
   is not enough evidence, retreat on sustained inability/collapse. No
   20M/34M/60M "unlock" and no global core gate vetoing one independent
   well-supported local success. Stable easy control remains a global safety
   diagnostic, not a reason to disable every other capability.
6. Keep a small randomized inter-specialist crossover budget so learning
   becomes coherent. After individual frontiers mature, increase combinations.

Command quality must be scored over episode time, not based on final root
position alone or the requested speed. Genuine steps demand supported
crossing and upper-platform contact, not just root x>lip. Inclines require
real MJX *contact* and normal evidence, not tilted gravity or a UI label.

## The single real terrain geometry (implemented, still under acceptance)

**No triangle-mesh soup.** The proposed six-triangle illustration has become
two *solid, convex oriented boxes* (lower + higher slab), always present in
the unified MJX topology. Their upper faces are co-planar at zero step
height and differ by a proper surface-normal displacement `h` when a
sharp finite step appears. Both slabs share the same pitch/roll normal and
independently chosen edge yaw, with movable MJX `mocap_pos/mocap_quat` sampled
per episode. Thus an actual foot, contact solver and all 16 depth rays see
the same non-flat world, with no per-episode model recompilation. A dormant
raised slab is buried well below the contact range, and slab-slab collisions
are excluded via complementary contact bits.

The source supports *exact reproducible exam parameters* for slope
`terrain_pitch_deg`, `terrain_roll_deg`, step
`terrain_fixed_height_m`, and rotated lip `terrain_edge_angle_deg`; omit
those to sample episodes, including mixed pitch+roll+step. Current early
frontier is intentionally conservative: physical geometry supports up to
15° **test** inclination (not a claim the controller can walk it); the
ordinary adaptive teaching frontier still needs robust slope evidence.

**No physical tripod in the NEW unified policy.** The earlier three ToFs
are retained only to load/replay historical 232-input and schema-9 360-input
policies; the new unified actor still has 90×4=360 observations but now
generates raw 4×4 distance/valid cells, body IMU and optical flow only.
Three existing grade/confidence channels are estimated from the delayed
4×4 returns with a residual-checked planar fit (reject mixed riser/top).
Optical flow scale uses actual collision-box rays, never a fabricated infinite
plane. Source provenance has a NEW `array-only` sensor revision, so old
policies cannot be silently interpreted as this new sensor model.

**Important correction:** the active array is **4×4 = 16 cells** today, not
8×8. An actual 8×8 array would change the observation interface, runtime
budget, policy weights and viewer parser, and needs its own explicit decision.

**Acceptance still pending**: current `scripts/smoke_unified_focus.py` must
pass 12-world mixed real contact, 2cm step height, exact 3° pitch /
−2° roll / 17° edge yaw, 4×4 rays and one full PPO gradient on Ubuntu CPU.
Then require a 4060Ti MJX GPU smoke and a live-viewer observation match.
Neither a source commit nor a drawn floor counts as physical proof.

## Physical engineering prerequisite: mixed real slopes and steps

The old trainer's source forbids `terrain_step_max_m>0` alongside
`contact_slope_deg>0` or randomized contact-domain terrain; the 2048-world
previous campaign consequently displayed `slope=0.0°` and stayed there.
Changing the printed number is NOT a fix. We need:

- Shared MJX topology for physical ground and finite step, oriented correctly
  relative to each sampled floor normal (including height and foot contact)
- Genuine 2D uphill/cross-slope orientation per world with controllable
  angle severity; real floor contact normal varies across worlds and, where
  supported by MJX, episode resets, without recompilation per episode
- Consistent birth height, actor ToF ray intersection, local-ground foot
  clearance, slip, support, edge crossing and fall measurements under
  mixed incline+step contact
- Absent platforms that do not produce ghost contacts in flat/control worlds
- A tiny physical acceptance smoke: flat + incline + rise + combined,
  with 2–4 worlds, real contacts, JAX PPO gradient, sensor ranges/validity
- NO actual long GPU training until this acceptance is demonstrated

The controller cannot expand a slope frontier against a static per-world
floor if the commanded difficulty exceeds that floor. Either real floor
slope must be adjustable at reset, or the target cohort must be *persistently
calibrated to its physical world* and curriculum exposure must account for
that limitation. This is a physics constraint to solve openly, not hide.

## Curriculum safety and performance

- Foundation from cold start may begin with gentle samples of every family
  once the actor can walk; no arbitrary 20M wait to enable real terrain.
- Preserve the policy's existing 360 observation ordering and physical 6
  action contract; teacher focus/severity must NEVER appear in actor inputs.
- No uncontrollable exponential command / push / terrain growth; physical
  hardware bounds, JAX capacity and conservative trust-region increments
  remain constraints. But every adaptive frontier should have *actual
  room to move* and report its current/maximum.
- Every console report includes focus occupancy, evidence denominators,
  sampled physical slope-angle distribution (not just config), upward/downward
  real step attempts/passes, command-frontier probe outcomes, and control
  mechanical shaft work/m plus tracking/jerk. No hidden zero-mode.
- Train 2048 worlds sequentially in ONE process / PPO trainer. Evaluate held-out
  frozen joint contexts in parallel with ordinary reference gait, not isolated
  flat-only models. One 10h run will now be a **unified policy campaign** with
  checkpoints and adaptive duration, not multiple divergent old-source arms.

## Priorities and the concrete reimplementation order

- [x] Owner-approved architectural direction and protected feature-branch
      specification (this document).
- [x] Fail-close obsolete ten-hour A/B-only launcher and mark old runbooks retired.
- [x] Teacher-only 20/20/20/20/10/10 world assignment; real low-dose step crossover,
      command/resilience/transition/control role emphasis. **Real slope crossover remains blocked.**
- [x] Opt-in command-specialist L1 probes and persisted episode-quality/sector
      evidence (24-edge-sample gate, signed directions); needs CUDA smoke and held-out audit.
- [x] Bootstrap specialist activation as steady difficulty is earned (0 at ≤.08;
      gradually to 100% by .30), preserving the main learning sequence.
- [ ] Build and physically smoke real mixed slope+step topology plus valid
      4x4 actor rays; this is the main launch blocker.
- [ ] Link resilience, slope, step, transition/posture growth to one
      evidence contract with sampled difficulty; retain legacy state for
      existing frozen policies, start NEW source-pinned lineage.
- [x] Opt-in stronger bounded motor/smoothness lesson in easy, moving control worlds,
      mechanical-work/m and episode-quality reports; **true long-run efficiency not verified**.
- [ ] Add focused diagnostics plus an actual production MJX PPO CUDA smoke.
- [ ] Replace old A/B overnight runner with a single unified 10h run,
      time-budget gate, checkpoints and live schema-9 viewer.
- [ ] Verify held-out composite challenges, preserve old champion and ask
      for explicit promotion before any main merge.

**Legacy note:** the retired v5023 dual-arm runbooks and launcher were deleted from the feature branch; their history remains in Git. Only this unified charter is authoritative. There is no unified GPU overnight launcher yet.
