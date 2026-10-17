# v5016 sensor actor: gated observation interface

The existing Pitch15 contract stays `pitch15_220`: 55 named channels per
frame, four historical frames, six actions. The default environment and
PPO normalizer still use that exact order and geometry. Archived actors are
**not** reinterpreted.

The opt-in `sim2real_232` interface is 58 channels per frame and four
frames of history. The first 55 names retain their original ordering, but
their gyro/velocity/range/velocity-confidence *sources* change. Because a
field can retain a name but change meaning, the **schema string is mandatory**
when training/export of this policy is eventually enabled.

- `gyro_x/y/z`: measured body-local gyroscope, not the legacy twice-rotated rate.
- `vel_x/y`: downward optical-flow angular velocity corrected with measured
  gyro and scaled by ToF triangulated depth; validity is a separate field.
- `tof_height`: ToF-derived *boresight depth*, not perfect simulator Z.
- `vel_confidence`: zero when ToF/flow invalid (not "speed is zero").
- `ground_forward_grade_rad`, `ground_sideways_grade_rad`: grades relative to
  gravity-levelled body heading, from the tripod fit.
- `ground_visibility_confidence`: visibility/geometry confidence; it is NOT
  a guarantee against uneven terrain or specular reflections.

Initial noise uses reproducible per-world/phase perturbations (1.5 mm ToF,
0.035 rad/s optical angular flow, 0.025 rad/s gyro at noise scale 1).
These are provisional engineering hypotheses, not verified specifications
of an optical-flow module. Four *policy* history frames do not simulate an
asynchronous sensor's sampling interval or latency.

This pass exposes the opt-in schema from `EmuMJXEnv` and adapts PPO's
normalization to its width. **Production training, checkpoint resume/export,
live viewer and the public champion are intentionally still locked to 220.**
Before enabling a 232 training profile: version checkpoint/provenance
validation, enforce no cross-schema resume or export, wire matched evaluator
environments and browser observation transforms, and test sensor update
rates, dropouts, biases and delayed readings.

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
correction. The 232-input interface is experimental until matching versioned
training/checkpoint/export and viewer input paths have been implemented.

## Checkpoint identity — schema 8 (guarded)

Legacy schema-7 Pitch15 pickle checkpoints remain 220 inputs and load
without any newly required metadata. New 232-input **checkpoint schema 8**
requires `observation_schema=sim2real_232`, its 232/6 weight shapes,
232-element observation-normalizer statistics, and a canonical
`sensor_contract` containing immutable field order, ToF extrinsic geometry,
the sensor-model revision and exact cadence/delay/dropout/noise settings.
The contract has a SHA-256 fingerprint. It is independently regenerated on
loading so self-consistent but altered source data cannot redefine its meaning.

All existing viewers, exporters and training-resume utilities still load
**only schema 7** by default. Reading schema 8 requires explicit schema and
sensor contract arguments; passing a merely shape-compatible policy is never
enough. This is an interface-safety gate, **not permission to launch 232-input
training**; trainer configs, held-out evaluators and the browser runtime
must be updated together before that gate is enabled.

## Standalone 232-input PPO acceptance (opt-in)

`emu doctor --v5016-smoke` compiles two genuine MJX sensor worlds,
steps them, builds the **232-input** PPO network, performs one 2-frame
rollout/optimizer update, and rejects nonfinite parameters/rewards.
This test neither starts a long training run nor writes checkpoints.
For a CPU-only testing host, add `--no-require-gpu --no-mjx-smoke`.
Passing this smoke does NOT establish that a new actor can walk or that
the legacy browser viewer can reproduce its observations.

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
A v5016 checkpoint is deliberately not exported by the legacy browser writer
or consumed by the original Pitch15-only directional examiner.

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
