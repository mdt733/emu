# EMU — core skeleton (v5058 research source)

**Purpose:** one page for understanding which piece owns what. This describes the executable system, not the history of experiments. The only active successor candidate is the **h40 / six C018 12V / rated-torque** robot; `main` remains protected. See [current handover](../handover.md).

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
| Deployment contract, coordinate conventions | `emu/contracts.py`, `emu/directional.py` |
| Physics, reset, sensors, rewards' measurements | `emu/environment.py` |
| 2×2 rolling terrain, triangular-prism collisions | `emu/terrain_carpet_mjx.py`, `emu/carpet_prism_mjx.py` |
| Mass/friction/servo domain variation | `emu/domain_randomization.py` |
| Six teaching cohorts, lesson selection, evidence | `emu/capability_focus.py`, `emu/curriculum.py`, `emu/capability_field.py`, `emu/step_goal.py` |
| Eight base reward terms and optional hypotheses | `emu/rewards.py` |
| Actor/critic, on-policy rollouts, GAE, PPO | `emu/ppo.py` |
| Training orchestration, immutable recipe, learner health | `emu/training.py`, `emu/training_config.py`, `emu/training_health.py` |
| Single run CLI, checkpoints and resume | `emu/cli.py`, `emu/run.py`, `emu/checkpoint_io.py` |
| Independent evaluations, reports, held-out stair evidence | `emu/evaluation.py`, `emu/prism_exam.py`, `emu/results.py`, `emu/step_autopsy.py` |
| Actor export, browser and read-only field atlas | `emu/policy_export.py`, `emu/viewer_server.py`, `emu/viewer_src/`, `emu/capability_viewer.py` |
| Hardware / environment diagnosis | `emu/doctor.py` |

## Hardware and timing

- `h40` selects ~400 mm neutral height, ~508 g provisional total (460 g chassis with all six C018 servos, ~40 g hip carriers/feet, ~8 g carbon rods), 36 mm ball-foot diameter, six position actuators and two unpowered shin-flex hinges.
- C018 12 V rated actuator force limit: **±0.980665 N·m**; stall torque is *not* continuously available. Actual torque-speed, thermal behaviour and the existing 0.624 N·m·s/rad powered-joint damping are **not yet hardware calibrated**.
- Each knee uses a scaled **compression** support strut; passive shin bending and 2 N·m/rad hip-roll centring are separate from the motor force limit.
- Physics timestep 5 ms, four substeps per 20 ms actor frame (200 Hz physics / 50 Hz control). A one-frame servo-command delivery delay is modelled.
- Real terrain: **four 75 cm tiles** in a recycled 2×2 midpoint-controlled window; **eight convex triangular prisms**, not 18 and not nine boxes. The hfield tiles are non-contact source samplers.
- One 360-field actor = four 90-field frames. IMU/proprioception, optical flow estimate, forward 4×4 ToF ranges/validity, estimated ground grade, shin flex and commanded velocity/posture; no teacher contact map.
- PPO: two 256-unit hidden layers, Gaussian exploration, a separate critic, 32-frame rollout × 2048 environments, four epochs/eight minibatches.

## Learning / reporting

The six specialist focuses are **command, resilience, slope, step, transition, control**. They are episode assignments sharing one set of actor weights. A fresh **20M** cold-start is the minimum standard foundation test. The Academy exposure grows after its configured introduction age; an Academy frontier is a teaching target, not proven step competence.

Every run lives outside Git at `~/emu-data/runs/<name>/` and includes resolved config, source/robot hash, policy+optimizer checkpoints, normalizer, teacher state, `metrics.csv`, independent `fixed_eval.csv`, and reproducible `results.md`/bundle. Exported `.emupolicy` is inference-only, not a complete learner rewind.

## Known fidelity boundaries

1. **ToF mismatch to intended ST VL53L8CX:** present policy ray spread uses **60° vertical** and **45° horizontal** despite the sensor's **45°×45°** physical FoV. At h40, two rows aim partly behind the robot. The dedicated CPU `viewer --sensor-lab` shows the h40 robot and adjustable 45°×45° physical sensor preview (foot occlusion included) alongside the old actor-ray contract; the training sensor contract remains unchanged.
2. **Viewer geometry is illustrative:** current browser terrain uses a separate nine-box/50 cm preview, not the trainer's four 75 cm / eight real prisms. Do not call it a physics-fidelity exam.
3. **1 m tile option not yet accepted:** a 2×2 window of four 1 m tiles may improve preview margin for the taller robot, but changes contact geometry/recycler and therefore needs new source and acceptance. Preserve the rated v5058 20M baseline.
4. **Contact acceleration:** non-foot colliders are disabled in the fast solver and monitored by virtual safety probes. 2/4 solver iterations is experimental, not a certified physically faithful engine.
5. **Software debt:** `environment.py`, `training.py`, `curriculum.py`, `ppo.py`, `results.py` contain historical experimental options. Remove *proven unreachable* branches as their last source-pinned study finishes, without changing this active baseline's arithmetic.

## Quality gates

Run `bash scripts/check.sh --full`, then real CPU prism/recycler checks; on the operator's GPU require 2048-world full PPO smoke, restart/checkpoint validation and a full **20M** cold start. Compare frozen-policy performance against an equivalent independent exam of the current `main` actor before any promotion. `main` and published actor stay frozen until explicit review.

Find the detailed work discipline in [MAINTENANCE](MAINTENANCE.md).

## v5058-successor h40 physical mass correction (new model identity)

**All six C018 servos are chassis-mounted**, including both remotely driven
knees. Former 300g torso plus massive old training rods was not representative.
The shared `robot.xml` now uses a **460g chassis box** (330g servos, ~12g
Pi Zero, ~60g three AA/14500 cells, ~18g wiring/regulators, ~40g structure),
with its centre 5mm ahead of the hip axes and rear edge at x=-45mm. Both
hip pivots are physically within the chassis, with no cosmetic brackets.

Each 40cm leg has **two solid 2mm carbon thigh rods** and **one solid 2mm
carbon shin rod** split at the pre-existing passive flex joint. The
nominal all-up robot is ~508g including the unchanged 10g hip
carrier and 10g ball foot per side. All masses and material density are
estimates requiring direct measurement. The electrical/servo drive is still
an ideal joint-space actuator; mechanical remote-knee transmission losses,
backlash and inertia have not been validated.

The previous v5058-rated 20M experimental lineage remains pinned to its
old physical model hash. This new model requires a fresh cold-start and
separate exams before `main` promotion. The legacy knee compression spring,
shin-flex stiffness, and rated torque limit are unchanged intentionally.
