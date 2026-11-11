# EMU — core skeleton (H40 mainline)

**Purpose:** one page for understanding which piece owns what. The **H40 / six C018 12 V / rated-torque** robot is the active `main` implementation, promoted through [PR #53](https://github.com/mdt733/emu/pull/53). Its exact 120M actor is checked in; publication to GitHub Pages remains a separate approval. See [current handover](../handover.md).

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
| Six teaching cohorts, lesson selection, evidence | `emu/capability_focus.py`, `emu/curriculum.py`, `emu/capability_field.py`, `emu/step_goal.py` |
| Eight base reward terms and optional hypotheses | `emu/rewards.py` |
| Actor/critic, on-policy rollouts, GAE, PPO | `emu/ppo.py` |
| Training orchestration, immutable recipe, learner health | `emu/training.py`, `emu/training_config.py`, `emu/training_health.py` |
| Single run CLI, checkpoints, resume and read-only parent console | `emu/cli.py`, `emu/run.py`, `emu/checkpoint_io.py`, `emu/console_dashboard.py` |
| Independent evaluations, reports, held-out stair evidence | `emu/evaluation.py`, `emu/prism_exam.py`, `emu/results.py`, `emu/step_autopsy.py` |
| Actor export, browser and read-only field atlas | `emu/policy_export.py`, `emu/viewer_server.py`, `emu/viewer_src/`, `emu/capability_viewer.py` |
| Hardware / environment diagnosis | `emu/doctor.py` |

## Hardware and timing

- `h40` is the sole active and default model (~400 mm neutral height, ~506 g provisional total (460 g chassis with all six C018 servos, ~40 g hip carriers/feet, ~6 g carbon rods), 36 mm ball-foot diameter, six position actuators and two unpowered shin-flex hinges).
- C018 12 V rated actuator force limit: **±0.980665 N·m**; stall torque is *not* continuously available. Actual torque-speed, thermal behaviour and the provisional actuator kv=0.624 N·m·s/rad speed-feedback term are **not yet hardware calibrated**. Unlike the retired passive motor-joint damping, kv is now inside the ±0.980665 N·m actuator force cap.
- Each knee uses a scaled **compression** support strut; passive shin bending and 2 N·m/rad hip-roll centring are separate from the motor force limit.
- Physics timestep 5 ms, four substeps per 20 ms actor frame (200 Hz physics / 50 Hz control). A one-frame servo-command delivery delay is modelled.
- Real terrain: **four 1 m tiles** in a recycled 2×2 midpoint-controlled window; **eight convex triangular prisms**, not 18 and not nine boxes. The hfield tiles are non-contact source samplers.
- One 360-field actor = four 90-field frames. IMU/proprioception, optical flow estimate, forward 4×4 ToF ranges/validity, estimated ground grade, shin flex and commanded velocity/posture; no teacher contact map.
- PPO: two 256-unit hidden layers, Gaussian exploration, a separate critic, 32-frame rollout × 2048 environments, four epochs/eight minibatches.

## Learning / reporting

The six specialist focuses are **command, resilience, slope, step, transition, control**. They are episode assignments sharing one set of actor weights. A fresh **20M** cold-start is the minimum standard foundation test. The Academy exposure grows after its configured introduction age; an Academy frontier is a teaching target, not proven step competence.

Every run lives outside Git at `~/emu-data/runs/<name>/` and includes resolved config, source/robot hash, policy+optimizer checkpoints, normalizer, teacher state, `metrics.csv`, independent `fixed_eval.csv`, and reproducible `results.md`/bundle. Exported `.emupolicy` is inference-only, not a complete learner rewind.

## Known fidelity boundaries

1. **VL53L8CX physical contract updated:** current h40 training and viewer use one 45°×45° square optical frustum (16 zone-centre rays), 70° down, 30mm below torso, zero forward/lateral/yaw; the 360 observation width is unchanged but its cryptographically-pinned geometry revision is new. The ST 4×4 zone-ID map explicitly accounts for the **two-axis lens inversion** in UM3109; verify PCB rotation before feeding ranges to the actor. A zone-centre ray is still an approximation of the sensor's multi-return optical histogram; validate foot occlusion against actual hardware.
2. **Terrain upgraded:** current MJX training uses four 1m rolling source tiles and eight convex triangular prisms, with midpoint recycling and foot-clearance protection. The **viewer** independently runs a CPU MuJoCo 15×15m ring of 225 mutable 2×2 heightfields, representing infinite deterministic world coordinates without reloading the robot. Its generated 1m surfaces and sensor height samples agree, and no occupied tile moves. Difficulty depends on world +Y (north); east–west varies the seed at fixed latitude. The viewer heightfield collider is **not** the trainer's convex prism and cannot certify matched MJX stepping.
3. **New cold start completed and source promoted:** seed 735, 2048 worlds,
120,061,952 steps on the central-servo/carbon/1m/physical-ToF model.
The frozen `.emupolicy` is current in `main`. Independent matched
real-prism exams have **not** been supplied; stepping remains unsolved.
Earlier h40 checkpoints are incompatible.
4. **Contact acceleration:** non-foot colliders are disabled in the fast solver and monitored by virtual safety probes. 2/4 solver iterations is experimental, not a certified physically faithful engine.
5. **Software debt:** `environment.py`, `training.py`, `curriculum.py`, `ppo.py`, `results.py` contain historical experimental options. Remove *proven unreachable* branches as their last source-pinned study finishes, without changing this active baseline's arithmetic.

## Quality gates

Run `bash scripts/check.sh --full` for Python, actual H40 policy contract and browser tests; CI additionally runs CPU prism/contact checks. Before modifying physical training, require 2048-world GPU PPO smoke, source-pinned restore/export and independent matched MJX exams. The 120M campaign **already completed**; do not rerun it to validate documentation changes. Publication of the new actor to GitHub Pages requires a separate checkpoint-backed release approval; no merge triggers deployment.

Find the detailed work discipline in [MAINTENANCE](MAINTENANCE.md).

## Physical identity and history

The currently trained H40 has all six servos in the chassis (not two knee-mounted motors), a provisional ~506 g all-up mass, its rated actuator output enforced inside the servo controller, and one 45°×45° sensor contract. Its complete mass/actuator ledger, original 120M source SHA, policy hash, outstanding real-hardware assumptions and release status live in the [H40 handover](../handover.md). Do not duplicate that ledger or launch historical experiments from this architecture page.

Older v5058 and v5059 research remains in Git and its original PR discussions. Different morphology/ToF/actuator models require new checkpoint identities.
