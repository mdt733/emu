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

**Legacy note:** `docs/V5023_OVERNIGHT_10H_PLAN.md` and
`docs/V5023_OVERNIGHT_RUNBOOK.md` represent a superseded experiment.
DO NOT follow their launch instructions for this design.
