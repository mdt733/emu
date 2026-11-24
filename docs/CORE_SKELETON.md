# EMU — core skeleton (H40 mainline)

**Purpose:** one page for understanding which piece owns what. The **H40 / six C018 12 V / rated-torque** robot and immutable 120M actor remain on `main`. The opt-in **`h40-spring-only`** research profile changes passive knee springs while preserving the motor, ROM and 360-input contract; it is *not* the deployed mainline model or a build-ready spring cartridge. The current draft research experiment tests **one survival-aware teacher trained from scratch**, not a new champion. See [current handover](../handover.md).

## Loop

```text
C018 motors + knee support + shin flex + two spherical feet
                            |
             MuJoCo/MJX: robot and real prism contacts
                            |
         ┌──────────────────┴──────────────────┐
         |                                     |
  noisy, delayed sensors                teacher's true facts
  + requested commands                  contact, reward, fall
         |                                     |
  90 readings × 4 frames                       |
         |                                     |
       ONE 360→256→256→6 actor                 |
         |                                     |
   6 servo-position targets                    |
         └───────────── physics ───────────────┘
                           |
     PPO actor+critic learns; governor chooses next episodes
                           |
    source-pinned checkpoint → independent exams → report
                           |
          validated .emupolicy → browser viewer
```

**Boundaries:** privileged contact/terrain ground truth belongs only to teaching; the actor sees 360 sensor/command/history values. Frozen exams do not feed PPO or curriculum. The browser runs its own MuJoCo/WASM world, not the batched MJX solver; a pretty viewer walk is not an independent exam.

## Source ownership

| Layer | Current owner |
| --- | --- |
| Physical model and morphology | `emu/robot.xml`, `emu/physical_variants.py` |
| Deployment contract, sensor geometry, coordinate conventions | `emu/contracts.py`, `emu/tof_geometry.py`, `emu/directional.py` |
| Physics, reset, sensors, rewards' measurements | `emu/environment.py` |
| 2×2 rolling terrain, triangular-prism collisions | `emu/terrain_carpet_mjx.py`, `emu/carpet_prism_mjx.py` |
| Mass/friction/servo domain variation | `emu/domain_randomization.py` |
| Six teaching cohorts, lesson selection, evidence | `emu/capability_focus.py`, `emu/curriculum.py`, `emu/capability_field.py`, `emu/step_goal.py`, experimental `emu/developmental_school.py` |
| Eight base reward terms and optional hypotheses | `emu/rewards.py` |
| Actor/critic, on-policy rollouts, GAE, PPO | `emu/ppo.py` |
| Training orchestration, immutable recipe, learner health | `emu/training.py`, `emu/training_config.py`, `emu/training_health.py` |
| Single run CLI, checkpoints and resume | `emu/cli.py`, `emu/run.py`, `emu/checkpoint_io.py` |
| Independent evaluations, reports, held-out stair evidence | `emu/evaluation.py`, `emu/prism_exam.py`, `emu/results.py`, `emu/step_autopsy.py` |
| Actor export, browser and read-only field atlas | `emu/policy_export.py`, `emu/viewer_server.py`, `emu/viewer_src/`, `emu/capability_viewer.py` |
| Hardware / environment diagnosis | `emu/doctor.py` |

## Hardware and timing

- `h40` is the **default/mainline** model (~400 mm neutral height, ~506 g provisional mass, 36 mm ball feet, six position actuators, passive knees and two unpowered shin-flex hinges). `h40-spring-only` is the active **opt-in research** physical profile; its 150 N/m preloaded knee tendon/rocker needs an actual manufacturable design before real-world claims.
- C018 12 V rated actuator force limit: **±0.980665 N·m**; stall torque is *not* continuously available. Actual torque-speed, thermal behaviour and the provisional actuator kv=0.624 N·m·s/rad speed-feedback term are **not yet hardware calibrated**. Unlike the retired passive motor-joint damping, kv is now inside the ±0.980665 N·m actuator force cap.
- Each knee uses a scaled **compression** support strut; passive shin bending and 2 N·m/rad hip-roll centring are separate from the motor force limit.
- Physics timestep 5 ms, four substeps per 20 ms actor frame (200 Hz physics / 50 Hz control). A one-frame servo-command delivery delay is modelled.
- Real terrain: **four 1 m tiles** in a recycled 2×2 midpoint-controlled window; **eight convex triangular prisms**, not 18 and not nine boxes. The hfield tiles are non-contact source samplers.
- One 360-field actor = four 90-field frames. IMU/proprioception, optical flow estimate, forward 4×4 ToF ranges/validity, estimated ground grade, shin flex and commanded velocity/posture; no teacher contact map.
- PPO: two 256-unit hidden layers, Gaussian exploration, a separate critic, 32-frame rollout × 2048 environments, four epochs/eight minibatches.

## Learning / reporting

The six specialist focuses are **command, resilience, slope, step, transition, control**. They are episode assignments sharing one actor and optimiser. In the current 0→180M research study, the same source-pinned teacher schedules a 20M foundation, later Academy exposure and continued generalist rehearsal. Its optional `--stair-survival-curriculum` requires two-foot first-lip crossing **and survival through the same completed life** to graduate uphill height. The **nominal** ordinary step frontier is bounded to 40 mm; Academy can earn **up to 55 mm**. The desired 80 mm step is tested as a stretch target, not yet within the explicit teaching ceiling. Researchers may still use authenticated warm-start forks via the generic trainer; this study simply chooses a cold birth.

**Where lesson decisions actually live:** `RegimeGovernor` (`emu/curriculum.py`) owns physical difficulty and recovery/competence evidence. One pure `birth_lesson_weights` function in the existing `emu/capability_focus.py` now combines six-role schedule, retention, athletic reservation and Academy admission into **one** episode birth decision. `CapabilityField` records command/recovery/maneuver evidence; in the current `shadow` setting, it **does not select lessons or compute unused teaching anchors**. It is not another teacher, policy or optimiser. A future school that actively teaches from the field must replace an existing allocation decision, not stack a second director on top.

Every run lives outside Git at `~/emu-data/runs/<name>/` and includes resolved config, source/robot hash, policy+optimizer checkpoints, normalizer, teacher state, `metrics.csv`, independent `fixed_eval.csv`, and reproducible `results.md`/bundle. Exported `.emupolicy` is inference-only, not a complete learner rewind.

## Known fidelity boundaries

1. **VL53L8CX physical contract updated:** current h40 training and viewer use one 45°×45° square optical frustum (16 zone-centre rays), 70° down, 30mm below torso, zero forward/lateral/yaw; the 360 observation width is unchanged but its cryptographically-pinned geometry revision is new. The ST 4×4 zone-ID map explicitly accounts for the **two-axis lens inversion** in UM3109; verify PCB rotation before feeding ranges to the actor. A zone-centre ray is still an approximation of the sensor's multi-return optical histogram; validate foot occlusion against actual hardware.
2. **Terrain upgraded:** current MJX training uses four 1m rolling source tiles and eight convex triangular prisms, with midpoint recycling and foot-clearance protection. The **viewer** independently runs a CPU MuJoCo 15×15m ring of 225 mutable 2×2 heightfields, representing infinite deterministic world coordinates without reloading the robot. Its generated 1m surfaces and sensor height samples agree, and no occupied tile moves. Difficulty depends on world +Y (north); east–west varies the seed at fixed latitude. The viewer heightfield collider is **not** the trainer's convex prism and cannot certify matched MJX stepping.
3. **Mainline and research are distinct:** the original H40 seed-735
120M champion remains protected on `main`; its viewer is not a matched
Spring-only exam. One research cold student on source `1fa832fc` (seed
735) and a revised source `4048e845` (seed 736) each completed
120,061,952 steps, but their teacher and held-out exam seeds differed.
The revised student traversed 64/500 uphill 4–40 mm and 0/500
44–80 mm on exact held-out prisms (seed 5701); this is not proof of
superiority over `main` or robust stair competence. Post-training
maintenance commits cannot resume those frozen checkpoints.
4. **Contact acceleration:** non-foot colliders are disabled in the fast solver and monitored by virtual safety probes. 2/4 solver iterations is experimental, not a certified physically faithful engine.
5. **Software debt:** `environment.py`, `training.py`, `curriculum.py`, `ppo.py`, `results.py` still contain optional historical teaching controls and compatibility code. The active cold study selects one recipe rather than deleting the generic warm-fork path. Separate verified dead teaching branches from useful experiment knobs; freeze the new source before cleaning shared arithmetic. Retired standalone stair-campaign drivers were removed from this research branch, while Git preserves them.

## Quality gates

Run `bash scripts/check.sh --full` for Python, actual H40 policy contract and browser tests; CI additionally runs CPU prism/contact checks. Before modifying physical training, require 2048-world GPU PPO smoke, source-pinned restore/export and independent matched MJX exams. The 120M campaign **already completed**; do not rerun it to validate documentation changes. Publication of the new actor to GitHub Pages requires a separate checkpoint-backed release approval; no merge triggers deployment.

Find the detailed work discipline in [MAINTENANCE](MAINTENANCE.md).

## Physical identity and history

The currently trained H40 has all six servos in the chassis (not two knee-mounted motors), a provisional ~506 g all-up mass, its rated actuator output enforced inside the servo controller, and one 45°×45° sensor contract. Its complete mass/actuator ledger, original 120M source SHA, policy hash, outstanding real-hardware assumptions and release status live in the [H40 handover](../handover.md). Do not duplicate that ledger or launch historical experiments from this architecture page.

Older v5058 and v5059 research remains in Git and its original PR discussions. Different morphology/ToF/actuator models require new checkpoint identities.
