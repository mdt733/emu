# EMU speed regression — top-down source and 80M evidence audit

**Audit:** 30 October 2026. **Do not edit the ongoing experiment lineage.**
**Frozen 360-input training source:** `38713791023681dc9883f6c374d92e46261b3baf`.
**Reference:** protected `main` at `2ec9e9db0a816c658131311129ef89ec7a595a12`.
**Evidence:** user-supplied `physical_comparison_14p9_16worlds.zip`, with A/B
`results_data.json` and both 16-world frozen physical exams.

## Executive diagnosis

The discrepancy is **primarily task-distribution and frontier governance**, not a new
global velocity/energy reward penalty. The new training recipe asks nearly every
world to execute a specialist-adjusted *low-speed* command, and starts its
signed forward frontier at 0.06 m/s rather than main's 0.60 m/s mature value.
Both completed A/B arms have the exact same final frontier,
`0.3691672572776858 m/s = 0.06 * 1.15**13`: thirteen net 15% promotions.
They have not reached the 1.4 m/s configured safety ceiling. A second,
independent obstacle is the **finite 8 m-long flat support slab**, which can
punish sustained 1 m/s walking by sending the robot off-world in a 14-second
episode. This is a geometry issue, not a PPO deficiency, and requires a
diagnostic before claims about learned maximum speed.

**Important:** the `frontier_vx_hi` console value is the *requested command
frontier*, not a measurement of maximum actual speed. The 16-world physical
exams use 0.15 m/s commands; they prove low-speed supported locomotion and
step crossing, not capability at 0.6–1 m/s. No source-matched, actual-speed
comparison of the 232-input Pinnacle and the 360-input successor was supplied;
do not claim main's measured 1 m/s from these artefacts.

## A/B 80M evidence

| Metric | A coherent | B considered uphill |
|---|---:|---:|
| Final steps | 80,019,456 | 80,019,456 |
| Fixed-eval broad | 0.759 | 0.742 |
| Forward command frontier, m/s | 0.369167 | 0.369167 |
| Forward near-edge precision | 0.0735 | 0.0972 |
| Current forward boundary attempts/successes | 3 / 0 | 1 / 0 |
| Command role share | 20% | 20% |
| Actual athletic-rehearsal occupancy | 0% | 0% |
| Saved config athletic-rehearsal share | 10% | 10% |
| Average forward command over completed episodes, m/s | 0.0645 | 0.1018 |
| Average actual forward velocity over completed episodes, m/s | -0.0033 | -0.0122 |
| Forward weighted locomotion reward | 0.868 | 0.843 |
| Mechanical cost reward term | -0.00722 | -0.00692 |

The signed training speed averages are not peak-speed tests; they mix
directions, terrain, episodes and falls. The held-out flat 0.15 m/s exams
give mean forward signed error about -0.029 (A) and -0.022 m/s (B).
They do **not** probe the 1 m/s question.

Real supported uphill step crossings, eight independent worlds each:
2.0 cm: A 7/8, B 7/8; 2.5 cm: A 4/8, B 6/8;
3.0 and 3.5 cm: both 0/8. The 2.5 cm difference is promising but
too small a sample to establish superiority. B has markedly worse
downhill mission scores at 5–12 degrees, so there is a tradeoff.

## Source-to-source cause inventory

1. **Specialist frontier bootstrap is 10x smaller.** Main's
   `RegimeParams.frontier_vx_hi` starts at `0.60`; unified v5028
   explicitly overrides it to `.06` in `RegimeGovernor.__init__`.
   It promotes successful signed command probes by `*1.15`, requiring
   at least four completed edge episodes at >=60% success per decision,
   and currently reports insufficient data/poor near-edge precision.
   Thirteen net promotions explain both final values exactly.
   `emu/curriculum.py`, plus the archived main version.

2. **Fast `main-mature` requests are overwritten at episode birth.**
   `EmuMJXEnv._command_envelopes` still calculates main-style steady
   interpolation, which could reach about `0.06+(.60-.06)*.93≈.562`
   at A's final steady difficulty. But `specialist_mix=1` after the
   20M foundation + 10M blending, and `_specialist_task` rewrites
   the episode command. With balanced focus allocation, 20% command
   worlds sample the small frontier, 20% step and 20% slope get
   coherent 0.10–0.22 m/s requests, *then* the auxiliary attenuation
   `0.04 + 0.5*(vx - 0.04)` makes these ~0.07–0.13 m/s.
   The 10% CONTROL role receives easy velocities; resilience/transition
   nonfocus motions also receive attenuation. No active role preserves
   a substantial unchanged fast steady-walking cohort.
   `emu/environment.py` `_specialist_task`, `_specialist_enabled`;
   `emu/capability_focus.py` weights and `teaching_profile`.

3. **The advertised athletic rehearsal is absent.** A and B both
   save `athletic_rehearsal_frac=.10` in `training_config`, but the
   final metrics report `rollout_athletic_rehearsal_frac=0` and
   `occupancy_athletic_rehearsal=0`. The trainer constructs the
   governor using the new configuration, then loads the older
   20M ancestor's full `curriculum_state` without reasserting
   changed training-time rehearsal weights. The saved state appears
   to replace the desired .10 with the foundation's zero. Even if
   restored, the current specialist rewrite would override many
   rehearsal target commands. These are two separate fixes:
   enforce/reconcile intentional fork config and preserve a genuine
   fast gait cohort through the specialist assignment.
   `emu/training.py` around governor restore and `params_with_field`.

4. **The physical world is no longer an infinite floor.**
   Original `robot.xml` is byte-identical in main/v5028 (Git blob
   `6bbe55ea44a75a7383920fa78ef49824256f5e4d`).
   Main uses its infinite plane. Unified v5028 replaces it with
   two finite `box` collision slabs; flat episodes bury the upper
   box, leaving the lower slab half-length 4.0 m, width 3.0 m.
   The lower slab is placed at the origin in flat episodes.
   A straight 1 m/s walk from the centre covers 4 m in 4 seconds,
   while a full 700-step episode at 20 ms lasts 14 seconds.
   Even a valid fast gait can run off the finite slab and be
   punished as falling. In active step worlds the two surfaces
   give a longer course but remain finite. This is a **structural
   problem that must be removed or demonstrated benign** before
   sustained 1 m/s training; it is not yet a proven explanation of
   the *current* ~0.37 command frontier, because actual trained speeds
   are much lower. `emu/environment.py` `add_unified_terrain`
   and `_unified_terrain`.

5. **The ordinary reward is largely unchanged.**
   Both versions pay `0.58 alive + 1.55 locomotion + 0.68 posture
   + 0.50 contact + 0.18 anti-slip + 0.16 clearance
   + 0.12 smoothness + 0.17 joint safety` (weighted),
   and the same early-death penalty
   `65 + 0.10*(max_episode_steps - episode_age)` for falls.
   Their normal velocity command tracking shares the quadratic
   exponential `exp[-(62*vx_error² + 40*underspeed² + ...)]`.
   For zero other errors and a >0.025m/s forward shortfall, the
   multiplier is approximately 0.43 at 0.10 m/s error,
   0.025 at 0.20, 0.00018 at 0.30 and 2.2e-11 at 0.50.
   This **steep signal saturates** when jumping far ahead of the
   competence frontier; however, it did not newly appear in v5028.
   B discounts part of this only for near-lip uphill STEP worlds
   (max 120 frames), not general walking. `emu/rewards.py`.

6. **A small new motor cost exists, but is not the main cause.**
   Although `motor_guidance_weight=0` in both A/B training configs,
   six-focus mode injects per-world proxy weight 0.006 generally,
   0.055 in CONTROL; the observed weighted reward term is ~-0.007,
   versus locomotion payment ~+0.84–0.87. Keep measuring it, but
   do not blame speed loss principally on this term.
   `emu/capability_focus.py` `control_effort_weights`
   and `emu/rewards.py` `motor_guidance_per_world`.

7. **Other important confounders.** Main reached its showcase
   after much longer training (~260M), while this study finishes
   at 80M; the actor is 232 vs 360 inputs with a changed terrain/ToF
   observation meaning. Fixed-eval `dyn80` survival is not an
   actual 1 m/s exam. Do not use one as proof of the other.

## Proposed minimal studies and order of operations

**First: cheap, read-only speed ladder on pinned A/B and main.**
For each policy's native observation schema, evaluate flat requested
forward speeds `0.10,0.15,0.20,0.30,0.40,0.50,0.60,0.80,1.00` m/s,
e.g. 16–32 worlds, with common seeds and separately tracked
100/200/350/700-frame horizons. Record *measured* speed, signed bias,
MAE, first fall position x, support, survival, action amplitude,
shaft J/metre, and amount of usable collision surface remaining.
Explicitly mark unlike sensor contracts/physics. A short-horizon
exam distinguishes lack of motor/gait competence from later
walking off the edge.

**Second: fix the finite-world opportunity.** Design a real,
sufficiently long continuous collision surface for sustained fast
flat walking while preserving the *actual* upper/lower 2-slab
discontinuity and unchanged 16-ray sensor semantics. This is a
physics-contract change: require its own source/actor acceptance and
fresh matched training, not a silent continuation of either 80M run.
Alternatively terminate/rebase deliberately at a distance boundary
without counting it as a fall, but that changes time distribution
and must be justified/measured. Never introduce teleporting actor
traces or free completion rewards.

**Third: restore genuine high-speed practice.** Keep one common PPO
policy and six role families, but reserve a meaningful (proposed
20–30%) source-visible normal steady/fast role that is **not**
rewritten into low-speed auxiliary commands. Explicitly reassert
the chosen specialist/athletic rehearsal settings after a permissible
experiment fork, with tests confirming the actual JAX rollout's
high-speed command histogram/occupancy. Preserve a separate near-edge
probe cohort plus easier baseline so expansion doesn't require
dangerous large one-shot jumps. Keep current step/slope roles intact.

**Fourth: gate performance on observed speed, not requested frontier.**
The promotion and validation metrics should print
`requested_vx_p50/p90/p99` and
`achieved_vx_p50/p90/p99`, duty fractions of actual demand
`>0.3,>0.5,>0.8`, per-band tracked survival/support,
and completed near-edge attempt counts. Report exact teacher
cohort allocations and actual executed commands rather than
assuming the configured .60/1.40 limits reached the policy.
Do not let a 1m/s training frontier stand in for achieved 1m/s.

**Fifth: only then ablate the reward curvature.** If sufficient
requested fast episodes actually occur on a physically continuous
floor but ascent toward higher speed still fails, compare the
original exponential tracking reward with a bounded-gradient or
Huber-like continuation; keep underspeed and overspeed penalties
symmetric in validity and maintain strict physical fall penalties.
Use matched seeds/checkpoints/frozen tests, with a command-following
metric and power/foot contact guardrails. Changing reward weights
before fixing exposure/terrain would obscure the cause.

**Decision point:** No evidence yet supports making either 80M
policy a replacement for the 232-input showcase. The A/B files
show real growing step skill, but not competitive fast commanded
motion. Avoid editing `main`, the original training SHA or
the finished checkpoints.
