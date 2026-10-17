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
