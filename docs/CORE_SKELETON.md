# EMU — core architecture (H40)

The goal is **one physical model, one deployable actor and one fully reproducible learning program**. Each source module owns one concept; experiments don't create alternate simulators or separate neural networks.

## Data and control loop

```text
H40 XML + rated motors/springs + four real-prism MJX tiles
    |                            |
    | noisy IMU / velocity /     | teacher-only true contact,
    | joints / 4x4 ToF / cmd     | terrain, survival and effort
    v                            v
90 readings x 4-frame history   outcome rewards + independent
    |                           capability schools/governor
    v                            |
ONE 360 -> 256 -> 256 -> 6 actor <--- PPO actor/critic update
    |
6 bounded servo targets --> MJX physics --> next sensor frame
    |
full optimiser/checkpoint/normaliser + source provenance
    |
frozen actor --> independent physical examiner --> evidence ZIP
    |
validated .emupolicy --> separate MuJoCo/WASM browser viewer
```

**Critical division:** the trained policy receives sensor/command history only. The teacher may use exact physical contact or terrain heights to calculate rewards, stage selection and evidence. These privileged values are never smuggled into the 360 sensor channels. Evaluation reads frozen weights and performs no PPO updates.

## One H40 physical identity

- 40 cm nominal hip height, six chassis-mounted Feetech ST-3215-C018 12V servos; ±0.980665 N·m rated output cap in the simulated motor controller.
- Passive knee support struts and flexible carbon-leg design; Spring V1 model retains 150 N/m knee support and 2 N·m/rad hip-roll centring. This is **simulated** and needs physical measurement.
- Forward 4×4 ToF sensor approximation, 70° downward, 30 mm below torso centre, 45°×45° optical span. Actual zone histogram, board orientation, latency and occlusion need calibration against hardware.
- Four rolling 1 m terrain tiles represented by eight true convex triangular prisms in MJX; no invisible box steps. Feet-only fast-contact solver and virtual non-foot intrusion detection remain deliberate approximations.

## Holistic random-init program

`scripts/run_h40_holistic_cold_wsl.sh` sets an immutable recipe and runs one uninterrupted **checkpoint lineage** through 20M, 40M, 120M, 140M and 160M environment steps. The stages are age-scheduled **teacher settings**, not independent trained models, additional actor observations or runtime code swaps.

| Age | Teacher emphasis | Prevented failure |
| --- | --- | --- |
| 0–20M | Stand/walk, supportive gait guidance | No stable basic gait |
| 20–40M | Fade gait/phase-specific rewards to zero | Overfit prescribed gait |
| 40–120M | Commands, slopes, recovery and earned UP/DOWN schools | Flat-only skill |
| 120–140M | Signed progress and near-lip approach | Avoidance instead of learning steps |
| 140–160M | Verified two-foot transfer, true post-cross continuation, extra control rehearsal | Leading-foot-only credit; downhill retention collapse |

The uphill Academy difficulty cap (55 mm) differs from downhill (80 mm) by design. **Both remain individually earned competency frontiers**; the separate examiner still probes 4–80mm. A robot is not promised to clear any height just because a teacher presents it.

## Module ownership

| Responsibility | Current source |
| --- | --- |
| Robot and deployment sensor contract | `robot.xml`, `physical_variants.py`, `contracts.py`, `tof_geometry.py` |
| Real signed prisms, physical contacts and observations | `environment.py`, `terrain_carpet_mjx.py`, `carpet_prism_mjx.py`, `directional.py` |
| Manufacturing spread and motion limits | `domain_randomization.py`, `physical_variants.py` |
| One learner and loss/reward accounting | `ppo.py`, `rewards.py`, `training_health.py` |
| Age stages, allocation and earned challenges | `capability_focus.py`, `curriculum.py`, `capability_field.py`, `step_goal.py` |
| Run, checkpoint, metrics and source locks | `cli.py`, `run.py`, `training.py`, `training_config.py`, `checkpoint_io.py`, `results.py` |
| Frozen evaluation and exam | `evaluation.py`, `prism_exam.py`, `scripts/v5053_matrix_exam.py` |
| Viewer, exported actor and dashboard | `viewer_src/`, `policy_export.py`, `viewer_server.py`, `console_dashboard.py` |

Large modules warrant eventual responsibility-based refactoring, but **never change traced JAX operations or physics during a source-pinned GPU run**. Any refactor requires contact, reward, observation, export and training-lifecycle parity tests. The current clean-integration proposal carries training-core module bytes unchanged from the original holistic cold experiment.

Read [installation](INSTALL.md), [physical quality](PHYSICAL_QUALITY.md), [maintenance rules](MAINTENANCE.md) and [current handover](../handover.md).
