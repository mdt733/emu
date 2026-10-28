# EMU surface laboratory — local MuJoCo viewer

The variable-terrain viewer is part of the current source tree. No
old v5023/v5024 feature branch needs merging to inspect the browser
terrain laboratory. Use a separate terminal; never modify the source
checkout or checkpoints of an actively running experiment.

This is a **manual browser simulation**, not an automatic certificate
of policy competence. Browser MuJoCo uses the trained actor and real
inclined/stepped collision geometry. Its independent exam counterpart is
`emu capability-exam --terrain-matrix` (MJX), with fixed cases and
separate stable-versus-tracking metrics.

## Start the viewer

Use a separate terminal and existing installed checkout. For watching
a source-pinned training run, keep its training shell/checkpoint directory
untouched. Open the local viewer from the **same frozen training checkout**
if possible:

```bash
cd "$HOME/emu-v5028"
source .venv/bin/activate

# Static/manual policy: any verified .emupolicy file
emu viewer showcase/champion.emupolicy --no-open

# OR live inspection of the first v5024 arm:
emu viewer --live \
  --root "$HOME/robot_lab/emu_runs/v5024_cold_steering/main_balanced_s733" \
  --no-open
```

Open **http://127.0.0.1:8765/**. If that port is in use, append
`--port 8766` and use **http://127.0.0.1:8766/**.

A 360-input policy can be loaded using the viewer's **Load policy**
control after being exported with `emu export-policy`. The approved
published 232-input Pinnacle remains available for reference only.
The live watcher is read-only; it selects validated saved checkpoints,
never trains/resumes, modifies metadata or consumes the learner's GPU
by itself. Browser simulation runs locally using MuJoCo/WASM. The
viewer HTTP service has no authentication: keep it local.

## Manual physical comparison

The **FRONT DEPTH 4×4** display shows what the new array-only actor
observes. Choose a surface under the laboratory controls:

| Surface | Useful manual checks |
| --- | --- |
| Flat | Baseline standing, forward/reverse, sideways and yaw commands |
| Continuous uphill/downhill/cross-left/cross-right | 1°, 2°, 3°, 5° grades first; compare fall, signed drift and tracking |
| Step up / step down | 0.3, 0.5, 1 and 2 cm rises/drops; inspect genuine far-side support |
| Combined commands | Walk uphill or across slope, then strafe/yaw instead of drifting under gravity |

Inclines use a real rotated support surface, not tilted-gravity physics
or a painted floor. Steps are actual discontinuous supports. Try a
modest `vx≈0.04 m/s`, then independently vary `vy` and yaw. Reset
between cases and compare heading assistance ON/OFF, sensor noise
ON/OFF and matched commands at matched grades.

The **essential distinction**: stable on a grade, actual progress
against gravity, and accurate three-axis command following are three
different results. Neither an expanded training slope target nor a
visually upright robot proves the latter two.

## Frozen matched assessment

The new 360-input examiner provides read-only 1/2/3/5° directional
incline comparisons and 0.3/0.5/1/2 cm up/down steps:

```bash
JAX_PLATFORMS=cpu emu capability-exam --terrain-matrix \
  --policy /path/to/new360.emupolicy \
  --output "$HOME/robot_lab/emu_runs/terrain_matrix_unique" \
  --worlds 8 --horizon 350 --device cpu
```

It writes `unified_exam.json` and `unified_exam.md`, separating
supported moving stability from complete command tracking. Inspect
uphill velocity, lateral downhill drift, yaw deviation and actual
foot-contact completion, not only teacher difficulty.

See the [v5024 experiment design](history/V5024_RUN_PLAN.md) and
[release gates](RELEASE_READINESS.md). The published Pinnacle and
existing training runs remain unchanged until an independently
validated 360-input champion is approved.
