# EMU architecture — current H40

**One physical robot. One 360-input policy. One batched MJX trainer. One source-pinned run.**

This page describes code ownership, not historical capability. The checked-in
`showcase/champion.emupolicy` remains the **120M H40 mainline actor**. The new
**Experiment C 500M cold two-lip run** is an unpromoted research candidate;
a passed source test does not make its trained behavior known.

## Read from the top

```text
Operator / lesson request (vx, vy, yaw, pitch)
    |
emu/cli.py               -> parses train, report, prism-exam, viewer, doctor
    |
emu/run.py               -> resolves ONE immutable TrainingConfig; checks source,
    |                       output directory and resume ancestry
emu/training_config.py   -> serialized recipe, H40 physical/sensor identity
    |
emu/training.py          -> one 2048-world PPO lineage and periodic frozen exams
    |                       (source-pinned checkpoints outside Git)
    +-- emu/curriculum.py         -> six lesson types; capability evidence;
    |                               stairs unlock only after walking competence
    +-- emu/capability_focus.py  -> shared-policy cohort allocation/commands
    +-- emu/environment.py      -> reset, actuator delivery, MJX physics,
    |       |                       sensors, real contact and reward facts
    |       +-- emu/robot.xml + emu/physical_variants.py -> physical H40 model
    |       +-- emu/terrain_carpet_mjx.py -> four 1 m source tiles, true elevations
    |       +-- emu/carpet_prism_mjx.py -> eight convex triangular colliders
    |       +-- emu/contracts.py + emu/tof_geometry.py + emu/directional.py
    |       |                       -> 4x4 ToF/IMU/flow/command contract
    |       +-- emu/rewards.py   -> PPO teaching payments, no actor-only truth
    |       +-- emu/step_goal.py -> reward/evidence-only stair conditions
    +-- emu/ppo.py              -> one 360→256→256→6 actor, critic, GAE, PPO
    +-- emu/checkpoint_io.py    -> optimizer, normalizer, curriculum, provenance
    +-- emu/training_health.py  -> optimizer/run safety
    +-- emu/console_dashboard.py -> read-only console
    |
    +-- emu/evaluation.py / emu/prism_exam.py -> held-out frozen exams
    +-- emu/results.py         -> observed evidence, not exaggerated claims
    +-- emu/policy_export.py   -> inference-only .emupolicy
            |
       emu/viewer_server.py + emu/viewer_src/
            |
       Infinite Meadow OR physical specialist observation worlds
       (browser CPU MuJoCo; distinct from training's MJX prisms)
```

## Non-negotiable boundaries

- **Actor:** 360 inputs = four 90-field sensor/command/history samples; six
  servo-position outputs at 50 Hz. No privileged contact map, hidden
  Academy position goal, exact height, episode identity or teacher reward
  reaches it. One shared actor serves every specialist.
- **Physics:** H40 ~40 cm, six chassis-mounted 12V C018 servos; actuator
  output is limited to the modeled rated ±0.980665 N·m, not stall torque.
  Carbon limbs, passive knee support, shin deflection, model mass and
  transmission assumptions remain subject to real hardware measurements.
  Physics 200 Hz; one control frame is four physics substeps.
- **Terrain:** MJX uses four recycled one-metre source tiles and eight real
  convex triangular prisms. The same geometric source supplies terrain
  sensing and contact. No legacy box/slab/floor cheat or second solver.
- **Teaching:** the six focused lesson cohorts are *allocations of one
  learner*, not six policies. Curriculum can use perfect physical facts to
  choose challenges/rewards; held-out exams remain separate.
- **Inference:** the exported policy is NOT an optimizer checkpoint.
  A source SHA, physical XML, sensor contract, reward recipe and saved
  curriculum together define a reproducible training lineage.
- **Viewer:** the browser's 225 mutable CPU heightfields give real browser
  contacts. Meadow is generalized free roaming; Commands, signed Slopes,
  Academy UP/DOWN, Resilience, Transitions and Control are dedicated
  observation fixtures. The untrained sensor lab also uses these same
  heightfields, never a retired log/ramp/box course. Their behavior is
  qualitative and is not a frozen MJX step-certification result.

## Active stair school — Experiment C (unpromoted)

The cold C source adds **two real signed lip seams at x=1 m and x=2 m** to
the same eight-prism world. The first supported two-foot transfer is only
an *introductory teaching milestone*. The completion event requires two
verified signed lip crossings, first-lip bilateral transfer, 50 cm of
continued travel after the second, a recent controlled forward-`vx`
window, upright contact and no fall. Ordinary command tracking applies
between lips; limited approach relief fades completely as each direction
graduates. The destination and contact labels remain teacher-only.

The training campaign lives on
[PR #84](https://github.com/mdt733/emu/pull/84), and the running,
source-pinned 500M checkout must not be edited as part of housekeeping.
**No claim of successful C training or mainline replacement is implied.**

## Keeping the skeleton clean

Module ownership is enforced by `scripts/audit_core_dependencies.py`.
Keep these boundaries rather than splitting the simulator into versioned
experiment engines or leaving parallel implementations. For a new study:
branch, test, record evidence, then remove superseded scaffolding. Remove
only branches proved unreachable by current CLI paths and physical tests;
never change the in-flight C numerical experiment behind its saved SHA.

See [Operations](OPERATIONS.md) for installation, running and recovery;
[Validation](VALIDATION.md) for evidence, promotion and release gates.
