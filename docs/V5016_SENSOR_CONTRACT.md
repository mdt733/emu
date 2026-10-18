# v5016 sensor actor: gated observation interface

The existing Pitch15 contract stays `pitch15_220`: 55 named channels per
frame, four historical frames, six actions. The default environment and
PPO normalizer still use that exact order and geometry. Archived actors are
**not** reinterpreted.

The opt-in `sim2real_232` interface is 58 channels per frame and four
frames of history. The first 55 names retain their original ordering, but
their gyro/velocity/range/velocity-confidence *sources* change. Because a
field can retain a name but change meaning, the **schema string is mandatory**
when training or exporting this sensor policy.

- `gyro_x/y/z`: measured body-local gyroscope, not the legacy twice-rotated rate.
- `vel_x/y`: downward optical-flow angular velocity corrected with measured
  gyro and scaled by ToF triangulated depth; validity is a separate field.
- `tof_height`: ToF-derived *boresight depth*, not perfect simulator Z.
- `vel_confidence`: zero when ToF/flow invalid (not "speed is zero").
- `ground_forward_grade_rad`, `ground_sideways_grade_rad`: grades relative to
  gravity-levelled body heading, from the tripod fit.
- `ground_visibility_confidence`: visibility/geometry confidence; it is NOT
  a guarantee against uneven terrain or specular reflections.

Sensor sample noise is Gaussian (1.5 mm ToF, 0.035 rad/s optical flow,
0.025 rad/s gyro at noise scale 1). These are provisional assumptions,
not verified optical-flow hardware specifications. The four actor history
frames are separate from the asynchronous sensor acquisition queue.

Schema-8 232-input learning, matched fixed evaluators and the read-only
local viewer now have explicit versioned input contracts. The public
champion and the default 220-input training mode remain unchanged.

The MuJoCo floor normal and true root velocity are used only in the forward
measurement simulator and objective rewards, never supplied to the estimator.
The first simulated ToF mount points are provisional and require calibration
against final real robot geometry.

## Acquisition timing and deliberate failures

The opt-in MJX path now carries a per-world, four-control-frame raw
sensor acquisition ring in the live environment state; it resets at episode
birth and never joins the schema-7 checkpoint (which deliberately restarts
episodes). The defaults at 50 Hz control are ToF at 25 Hz, optical flow at
50 Hz, gyro at 50 Hz, with **one 20 ms frame delivery latency** for both ToF
and flow. Flow de-rotation uses the gyroscope reading from the camera's
own acquisition frame, not the latest independent IMU sample.

Configurable constructor-only diagnostic knobs: `tof_period_frames`,
`flow_period_frames`, `tof_delay_frames`, `flow_delay_frames`,
`tof_dropout_prob`, `flow_dropout_prob`, and
`gyro_bias_std_rad_s`. The default dropout probabilities are 1% for
new ToF samples and 2% for new flow samples. The standard deviation for
the episode-static gyro bias is 0.015 rad/s. Gaussian sample noise scales
per world with the preexisting `sensor_scale` curriculum parameter.

On missed updates, the last raw measurement is held and marked older;
on a dropped or invalid acquired sample, it becomes invalid and the
corresponding confidence goes to zero when delivered. A missing reading
must not be interpreted as level terrain or a confidently stationary robot.
Freshness attenuates confidence, but confidence does not yet represent
uncertainty from texture, rolling-shutter artefacts or nonplanar terrain.
The sensor cadence is a JAX static environment configuration rather than a
training knob: v5016 training/export identity must lock it before GPU runs.

**Known limitations:** this remains a pinhole, optical-axis flow model,
not a texture-based camera renderer. There is no photometric failure model,
body occlusion, pixel quantisation, continuous gyro integration or drift
correction. The 232-input interface remains experimental pending validation against
calibrated physical ToF, IMU and optical-flow hardware.

## Checkpoint identity — schema 8 (guarded)

Legacy schema-7 Pitch15 pickle checkpoints remain 220 inputs and load
without any newly required metadata. New 232-input **checkpoint schema 8**
requires `observation_schema=sim2real_232`, its 232/6 weight shapes,
232-element observation-normalizer statistics, and a canonical
`sensor_contract` containing immutable field order, ToF extrinsic geometry,
the sensor-model revision and exact cadence/delay/dropout/noise settings.
The contract has a SHA-256 fingerprint. It is independently regenerated on
loading so self-consistent but altered source data cannot redefine its meaning.

Legacy learning/resume default to schema 7; the 232-input learner opts
into schema 8. The browser exporter converts validated schema-8 checkpoints
to browser policy schema 2; legacy policies still export as schema 1.
The independent physical directional examiner remains schema-1-only
pending a dedicated sensor-aware physical examiner.

## Standalone 232-input PPO acceptance (opt-in)

`emu doctor --v5016-smoke` compiles two genuine MJX sensor worlds,
steps them, builds the **232-input** PPO network, performs one 2-frame
rollout/optimizer update, and rejects nonfinite parameters/rewards.
This test neither starts a long training run nor writes checkpoints.
For a CPU-only testing host, add `--no-require-gpu --no-mjx-smoke`.
Passing the smoke doesn't prove locomotion. The completed 20M sensor
foundation is now replayable with a separate local browser observation
emulator, which is not a held-out exam.

## End-to-end learner wiring (v5016 experimental)

`emu train --profile sim2real --sensor-actor` explicitly chooses the
232-channel policy. The unadorned sim2real profile still uses the protected
legacy actor convention, permitting A/B tests rather than silent replacement.
The training configuration records `observation_schema`, and the training
coordinator supplies the same observation schema to all five matched held-out
worlds and to the cloud evaluator. PPO running normalizers are 232-wide; every
checkpoint family (latest, rotating snapshot, recovery, best, final) includes
schema 8 plus the exact canonical sensor contract.

Auto-resume, explicit parent resume, recovery and final-step promotion all
validate the sensor-contract fingerprint *before* using weights. Legacy
schema-7 functions continue to default to the protected 220-input contract.
A v5016 checkpoint is exported as a separate browser policy schema and
remains incompatible with the Pitch15-only physical directional examiner.

The first GPU acceptance run can use 512 worlds and 2,097,152 steps (128 PPO
updates at 32 frames per update, 16,384 env steps per update). A fresh run is
appropriate since an archived 220 policy cannot safely bootstrap a 232 policy.
Do not claim the sensor-trained actor is deployable until the live robot sensor
pipeline and exporter/viewer can interpret the same sensor timestamps, ranges,
gyro reference and field ordering. Fixed evaluator remains simulation based.

Example from a clean source checkout:

```bash
emu doctor --v5016-smoke
emu train --profile sim2real --sensor-actor \
  --run-dir ~/robot_lab/emu_runs/v5016_sensor_2m \
  --steps 2097152 --envs 512 --seed 733 --device cuda
```

## Foundation first, then mature sim2real

For a brand-new actor, `--profile foundation --sensor-actor` uses the
known stable low-disturbance stepping curriculum with **the new 232-channel
sensor inputs**. Optional `--pitch-deg 15` retains the v5015 posture task.
The existing `--profile sim2real --sensor-actor` uses mature graph-based
curriculum and should generally be reserved for a sensor-trained lineage
continued from a completed foundation. Both profiles pin the same
observation schema and sensor hardware model, but can differ in lesson
difficulty.

Since experiment configurations are immutable at a run path, begin with a
**new directory** for each phase; resume the schema-8 foundation by passing
its exact final checkpoint as `--resume` to the mature phase (do not silently
change a saved phase's curriculum).

## v5016 browser live replay

The viewer accepts browser policy schema 2, with 58x4 channels and the
validated schema-8 sensor contract. Its distinct browser sensor module
models three downward ToF rays, gyro-compensated optical flow, delayed
sample queues, missed updates, dropout and episode gyro bias.

Browser replay is *approximate*, not bit-exact MJX. A rendered park
heightfield is not the same as the tilted contact planes used for some
training phases; browser random noise differs from JAX's PRNG. Use
held-out MJX evaluations for promotion, not the browser demonstration.

**When Pinnacle is currently running, do not pull into the training
checkout.** Start the viewer from a separate detached Git worktree:

    cd ~/emu
    git fetch origin feature/v5016-sim2real
    git worktree add --detach ~/emu-viewer-5016 origin/feature/v5016-sim2real
    cd ~/emu-viewer-5016
    PYTHONPATH="$PWD" emu viewer-build
    PYTHONPATH="$PWD" emu viewer --live --root ~/robot_lab/emu_runs/v5016_pinnacle

The watcher follows the most recently atomically published policy across
tournament branches, reporting stage name and validated step count.
The terrain-park grass receives a separate, depth-tested high-contrast
10-cm overlay with bold 1-m borders, switched on by default.

For the standalone foundation run, change --root to
~/robot_lab/emu_runs/v5016_sensor_foundation_20m.
