# EMU v5022 — Physical terrain and forward depth

**Branch:** `feature/v5022-forward-terrain` from protected `main`
(`2ec9e9db0a816c658131311129ef89ec7a595a12`).
**Status:** physically grounded geometry and **viewer-only diagnostic**; NOT connected to actor,
PPO, new NN observations or training runs. The published public showcase
and protected `main` remain untouched.

## Single-owner architecture

- `robot.xml`: unchanged 6-actuator robot, links and 18mm ball feet.
- `environment.py`: will own actual MJX world generation, independent
  sensor acquisition/delay/dropout, observation assembly. No second timing
  queue or sensor framework.
- `terrain.py`: pure JAX/NumPy geometry helpers for *real collidable
  support boxes*, continuous grade transitions, signed steps, and physically
  measured tripod/front depth. No curriculum, PPO or checkpoint logic.
- `contracts.py`: will own **versioned** deployable observation fields and
  sensor contract hash. Existing 220 and 232 identities stay immutable.
- `tests/test_terrain.py`: one physical/optical regression suite.
- Existing reward, PPO, checkpoint and exporter owners remain as-is.
  The **existing viewer owners** now contain a read-only optical preview;
  they do not modify the champion's neural inputs.

No additional production module, standalone experimental harness, or
version-specific training script should be needed.

## Sensor selected

**ST VL53L8CX** (wide-angle discussion referred to ~63°; ST's specified
detection FoV is **65° diagonal, 45° by 45°**), 8×8-capable hardware
**operating in 4×4 mode**, delivering 16 range zones.

- Reference: https://www.st.com/resource/en/datasheet/vl53l8cx.pdf
- Driver: https://www.st.com/content/st_com/en/technical-documents/UM3109.html
- `terrain.py` holds **one optical mounting point** at body XYZ
  (0, 0, -18) mm, tilted approximately **70° downward**. Provisional
  mounting geometry: verify leg swing, chassis clearance, PCB placement,
  wiring and actual usable FoV using CAD / a real robot.
- Zone ordering: internal rows are **far to near**; lateral ordering is
  defined in the robot frame. Check orientation mapping and lens inversion
  against the real VL53L8CX 4×4 driver before connecting hardware.
- Range validity: explicit per-zone validity, missing return represented
  by zero **plus invalid**. The 1.5m simulator cap is a training region of
  interest, not the published max ranging capability (4m).
- The Python reference simulator samples five rays per zone (one centre
  plus four quarter-zone points) to approximate a finite footprint.
  The viewer offers one, four or five ray samples per zone, with a measured
  scan cost. All sample directions share the same geometry definition.
  Neither path emulates ST's histogram
  algorithms, reflectance response, crosstalk and multiple-target selection
  are **not** modelled exactly.
- Actual **moving ball feet** must intersect front rays. Never erase a
  visible foot from the simulated scan to make training easier. Later
  include shins, joints, wiring and enclosure if physical CAD indicates
  they occlude the optics.
- Range data are *line-of-sight slant ranges*, not 16 heights or a terrain
  classification. Chassis IMU pitch and tracked history provide context.

Nominal target acquisition for the upcoming opt-in interface: around
**25–30 Hz** (not guaranteed every 50 Hz control frame), buffered with
capture/delivery timestamps, noise, dropouts and stale-sample age in the
**existing `SensorState`**. No duplication of timing math. A further
physical performance benchmark must determine whether 5×16 ray sampling
is affordable at 2,048 worlds.

## Observations, not privileged terrain truth

New actor schema to be designed and independently versioned from
`sim2real_232`. Starting candidate: 16 measured ranges + 16 validity
flags + one scan-age signal **per history frame**, adding 132 inputs to
the existing 232 => **364 total**, *if* all 4 historical frames retain
all 33 new channels. Neither 296 nor 364 is mandated before the end-to-end
batch/performance tests. Do not silently reuse 232 checkpoint weights.

Physical world generator knows slope/step height for evaluation only;
the policy receives no ground-geometry labels, invisible footprints,
future feature distance or true horizontal velocity.

## Terrain geometry

Consecutive `Surface` segments have real solid MuJoCo BOX volumes,
a top slope and height. Signed `Seam.jump_m` independently represents
vertical step-up/down. Zero jump with changing slope creates a crest,
valley, slope onset or exit. Training difficulty may vary absolute grade,
grade transition and step jump independently.

A downstream physical MJX scene must use *the same support geom poses*
for collision and ToF forward modelling. Both front sensor and down
tripod now have such an optional forward model; legacy `sim2real_232`
keeps its original single-plane interpretation for checkpoint identity.
Tripod beams straddling a step must **not** be interpreted as a perfectly
valid local slope just because three intersection points define a plane.

## Proof gates

1. CPU geometric and JAX parity tests, 4×4 optical centre, range validity,
   step/crest profiles and moving-foot occlusion.
2. Real MJX physical collision and ray-distance parity with a step, crest,
   and two worlds standing at distinct elevations. Measure batch memory
   and compilation/runtime overhead.
3. Replace single-plane sensor acquisition **only in the new opt-in
   terrain mode**, adopt 4×4 range/validity/age in existing async queue.
4. Freeze schema across observation, PPO, checkpoints, export, CPU viewer
   and browser; prove a fresh cold-start rollout and its lineage.
5. Controlled matched-policy runs versus down-tripod-only walking; assess
   anticipatory pre-contact movement and real successful landings.

**Do not start an overnight GPU run before gates 1–4.**

## Pre-existing general CI problem

The old full Python test suite passes the approved 232-input current
showcase to a legacy **220-input/schema-1-only directional examiner**
in five tests. Those failures are inherited from `main`, not failures
of the forward sensor. The protected showcase must not be rewritten
to satisfy a stale test. The dedicated terrain check was green before
consolidation; final consolidated results are tracked in GitHub Actions.


## Viewer acceptance lab (champion untouched)

The *existing* `viewer_src` now offers the **FRONT DEPTH · 4×4**
diagnostic, attached to any loaded policy. It DOES NOT connect this
diagnostic to the policy observations, learned weights, or servo commands.

Controls:
- **Downward tilt:** 0–90° in 5° increments, provisional default 70°.
- **Forward sensor mount:** 0–180 mm from root, provisional default 0 mm.
  These are body coordinates, not world-X or feet-relative positions.
- **Show rays:** 16 representative visible rays plus the optical mount.
- **Ground:** existing viewer park, or switch into one real MuJoCo contact
  course: flat, 5cm step up, 5cm step down, crest, trough,
  half-cylinder log, or angled ridge.
- **4×4 cells:** raw line-of-sight distances in **centimetres**,
  far/top row to near/bottom row. Invalid = "—", NOT zero height.
  Each optical zone defaults to four subrays over simple courses;
  the larger park heightfield uses one centre ray per zone to keep the
  viewer responsive. Neither emulates a calibrated ST histogram engine.
- Sensor raycast includes MuJoCo-visible, **moving robot geometry**,
  so swing feet/legs may occlude returns; no silent filtering.
- In depth courses autonomous roaming is disabled, leaving the exact
  champion actor under manual joystick control. Existing navigation
  remains available in the normal park mode. The champion has NOT learned
  these steps; falls here are expected, not evidence of sensor failure.

To try it without replacing your working checkout:

```bash
git clone --branch feature/v5022-forward-terrain https://github.com/mdt733/emu.git ~/emu-v5022
cd ~/emu-v5022
bash scripts/setup_wsl.sh --install-node --viewer-only
source .venv/bin/activate
emu viewer showcase/champion.emupolicy --no-open
# In your browser: http://127.0.0.1:8765/
```

A running local viewer must rebuild with `emu viewer-build` after pulling
code updates. The public GitHub Pages showcase continues to use `main`;
opening that public URL will NOT show v5022 until explicitly promoted.

Feedback requested: choose a course; walk toward the feature with
vX about 0.04–0.08 m/s; compare tilt 35/50/65° and mount 85/120/160 mm.
Watch the far row, first reliable range change, dropout at the edge,
and whether a swinging foot occludes a cell. Keep speed and starting
position the same when comparing configurations.

**Important caveat:** the browser renders actual MuJoCo geom positions for
ray intersection, but its ray engine is Three.js visual geometry and its
zone aggregation is simplified. It is a *qualitative* hardware validation,
not a bit-identical readout from a real VL53L8CX. Also the loaded 232-input
champion still uses its protected historical down-tripod floor estimator,
which can become inaccurate on discontinuous new courses. The front panel
reports independently observed geometry; no trained step-crossing claim.


### Preflight review and copyable measurements

The viewer has an **Auto / 1 / 4 / 5** rays-per-zone selector:
- **Auto:** 16 rays on the costly park heightfield, 64 rays on simple courses.
- **1:** 16 total centre rays, fastest.
- **4:** 64 total quarter-zone rays, a spatial 8×8 proxy reduced to 4×4.
- **5:** 80 total centre + quarter-zone rays, matching the Python reference.

All configurations deliver exactly **16 raw zone ranges and validity bits**.
Measured **milliseconds per scan** is shown; scan update is intentionally
throttled to about 13 Hz (or slower if a scan proves expensive). This is a
*viewer display clock*, **not** the real VL53L8CX 4×4 acquisition clock.

In the display, orange indicates **some rays struck a visible robot part**
(which can be shin/foot), and yellow means subrays hit more than one visual
object (e.g. two support geoms across a step seam). Those classifications use
viewer-only geometry IDs for debugging. **They are privileged diagnostics,
never allowed into the NN.** A flat slope should not be flagged as a "mixed
edge" merely because its near and far rays have different slant ranges.

The 3D ray endpoints are sampled **physical surface intersections** rather
than a fictional projection of a median depth onto the centre ray. The
**Copy 4×4 reading + settings** action copies the raw zone grid, configuration,
scan cost, and occluded/mixed zone indices for bug reporting.

Known limits before first interactive acceptance:
1. Browser colour/rendered geometry is used for ray intersection, not the
   real sensor histogram. The real instrument needs target-distance/status
   bench measurements on a physical step before the NN is final.
2. The **Python training candidate models ball-foot spheres**, while the
   browser diagnostic sees all visible robot/shin meshes. Align physical
   occluder geometry before closing the sim-to-real gate.
3. A step course may make the existing champion fall or corrupt its old
   tripod-derived ground estimate. That's expected: this is a sensor
   inspection run, not proof of step locomotion.
4. Existing full CI has five pre-existing tests that demand a schema-1
   champion while the published champion is schema-2; do not "fix" them
   by replacing the protected policy.
5. Browser CI confirms JS contracts and builds; full visual WASM scene
   rendering and intuitive sensor behaviour still require the user's
   interactive PC run.

### First PC acceptance sequence

1. Open **Current viewer terrain** with **Auto**, confirm the 4×4 panel
   changes during movement and the visible optical mount is outside the
   torso mesh.
2. Switch to **Flat reference** and compare tilt 35°, 50°, 65° with a
   stationary robot. Watch far/near row ordering and missing cells.
3. Set tilt 50° / forward 90 mm; approach **5 cm step up**, then
   **5 cm step down** at a controlled slow joystick speed. Pause just
   before the edge, **Copy 4×4 reading + settings**, record any missing or
   robot-occluded cells. A fall is not evidence of bad optical behaviour.
4. Try **Hill crest** then **Hill trough** and **angled ridge**.
5. Compare **1 / 4 / 5** on the same paused physical scene and record
   milliseconds/scan. Try **4** on the park only once to check whether
   64 casts cause a visible slowdown, and return to Auto afterwards.
6. Repeat your favourite settings with sensor noise off to separate
   geometry mistakes from mild added range noise.

**Do not freeze NN input shapes or begin a big cold-start training campaign
until the above visible behaviour is accepted.**


### Copy-paste verification on a WSL PC (no GPU run)

The `--viewer-only` switch is opt-in: it reuses the existing setup script,
runs the consolidated physical MJX terrain tests and protected sensor schema
test, builds the browser from the lockfile, and runs existing Node tests.
It does **not** call the known-incompatible legacy schema-1-only examiner or
run training. Normal setup flags retain their previous semantics.

**Fresh separate checkout, installation, targeted tests and viewer launch:**

```bash
git clone --branch feature/v5022-forward-terrain \
  https://github.com/mdt733/emu.git ~/emu-v5022
cd ~/emu-v5022
git branch --show-current
bash scripts/setup_wsl.sh --install-node --viewer-only
source .venv/bin/activate
emu viewer showcase/champion.emupolicy --no-open
```

**Optional repeatable checks after setup** (for example, after a later branch update):
```bash
cd ~/emu-v5022
source .venv/bin/activate
python -m pytest -q tests/test_terrain.py
python -m pytest -q tests/test_physics.py::test_versioned_sensor_actor_schema_keeps_pitch15_frozen
npm test --prefix emu/viewer_src
emu viewer-build
```

**If the separate checkout already exists**, replace just the initial clone
with:

```bash
cd ~/emu-v5022
git switch feature/v5022-forward-terrain
git pull --ff-only origin feature/v5022-forward-terrain
```

The viewer's copy-reading button exports 16 centimetre values and valid/null
markers with tilt/forward mount, geometry category, number of rays, measured
scan milliseconds, and zone indices hit by robot parts or multiple visible
meshes. Use those snapshots for reproducible before/after comparisons.

**Manual acceptance questions (not automated assertions):**

- Flat: far-to-near monotonic geometry at typical orientation; no fake seam
  from different oblique ranges on one surface; chassis pitch changes scan.
- Step up/down: observe the edge at a distance before the feet arrive;
  physical contact and front measurements refer to the *same* boxes.
- Crest/trough: changing grade is continuous, not a vertical step.
- Moving legs: orange self-occluded cells appear when real visible geometry
  crosses the optical frustum; do not suppress them in PPO.
- Benchmark: pause on the same scene and compare 16/64/80 rays/scan
  and FPS; warm up several scans and report typical cost rather than a
  single first-use value (browser JIT, shader, cache, GC effects).
- Disable depth entirely and check the champion walks just as previously;
  the sensor must never modify its NN observation history or actuator loop.


### Stage 2 — tests ready after your FIRST viewer results

These options remain **opt-in**, and reuse the two maintained test files.
They do not start training, modify a checkpoint, or generate a new lab module.

**1. Confirm the sensor contract + solid-world parity after tweaking the mount:**

```bash
cd ~/emu-v5022
source .venv/bin/activate
python -m pytest -q tests/test_terrain.py
npm test --prefix emu/viewer_src
```

This checks the new 16/64/80 geometry modes, 35°/50°/65° angle
and 85/120/160mm position candidates, true contact on steps/crests,
batched JAX, and exact browser/Python beam directions on two rotated
robot poses. Node-less CPU-only Python installs skip only the
cross-language comparison; WSL viewer-only setup installs Node.

**2. Inspect the range of visibility *before* contact, with no GPU needed:**

```bash
EMU_SENSOR_SWEEP=1 python -m pytest -q -s tests/test_terrain.py -k optional_post_viewer_mount_sweep
```

This prints a JSON report of how many of the 16 depth zones distinguish
a real +5cm or -5cm step from level ground over five approach positions,
at 64 rays/scan. The default compares all 9 mount candidates. For the
mount you identify as promising in the viewer, specify it exactly:

```bash
EMU_SENSOR_SWEEP=1 EMU_SENSOR_TILT_DEG=50 EMU_SENSOR_FORWARD_MM=120 \
  python -m pytest -q -s tests/test_terrain.py -k optional_post_viewer_mount_sweep
```

Replace 50 and 120 with **your actual chosen tilt and forward mount**.
The output is a diagnostic, **not a pass/fail terrain-learning score**.
Ranges are line-of-sight measurements, and changed-zone counts are
geometric contrast, not proof that the network will use the signal.

**3. Measure prospective GPU cost (optional; JAX CUDA needed):**

```bash
EMU_SENSOR_BENCH=1 python -m pytest -q -s tests/test_terrain.py -k optional_gpu_2048_world_ray_cost
```

This JIT-compiles the pure forward sensor model for 2,048 worlds,
compares **16 / 64 / 80 rays**, includes two moving-foot spheres,
and prints compilation seconds separately from average steady-state
scan milliseconds. It does **not** include PPO, whole-robot MJX
contact or end-to-end training FPS, and does not start a training run.
CPU-only installations explicitly skip the GPU benchmark.

**4. Decide the next code change using viewer + test results:**

- If up/down steps are visible early, flat surfaces read sensibly, and
  self-occlusion is tolerable: lock forward position, tilt, valid range
  and 4×4 orientation in a NEW sensor-contract revision. Do not mutate
  `sim2real_232` or reinterpret its checkpoints.
- If previews are too distant/close: run the same angle/forward sweep
  around your preferred candidate before freezing it.
- If the leg obscures frequent cells: investigate vertical/lateral
  mount position or orientation and update simulator occluders.
  Never filter leg returns from simulated NN inputs without a
  matching physical sensor intervention.
- If GPU/park scans prove costly: compare 16/64/80 options and measure
  the *full* 2,048-world trainer before selecting the training footprint.

**Following this stage:** integrate raw 16-zone ranges, 16 validity
flags and a scan-age signal into the existing asynchronous sensor queue,
define/fingerprint a new 4-frame actor schema, then test a from-scratch
256-world PPO update, 2,048-world memory/performance smoke and checkpoint
export/resume. Only then start a 10–20M cold-start foundation and a
short branch tournament. The 364-wide observation concept is a **candidate**,
not a validated or deployed contract yet.


### Viewer feedback follow-up: zero mount, functioning courses and park ring

After the first hands-on viewer run, the physical root-relative sensor's
forward range is now **0–180 mm** instead of an arbitrary 80mm lower bound.
Zero is a valid simulated optical-centre position, but might sit *inside*
the torso enclosure: raycasts now use two-sided surfaces so the user
can see this chassis self-occlusion rather than see through plastic.
The ultimate real bracket/PCB geometry still needs validating.

**Why depth disappeared in non-park modes:** the raw raycast still ran,
but the viewer's underfoot grade telemetry threw on mode `sensor`,
interrupting display updates. `surfaceSlope('sensor')` now explicitly
reports "not supported" rather than pretending a smooth grade exists on
discontinuous steps or raising a fatal error. A regression lives in the
existing Node viewer suite.

The default **radial terrain park** now has a wider **4.2 m level-centre
radius** and six deterministic *collidable* features around its outer
inner-flat ring: two different-direction 5cm raised ledges, a 6cm
ramp/crest/ramp, a short transverse log, an angled ridge and a relative
trough between raised banks. The centre (radius under ~1.7 m) is clear
for safe initial standing/walking, and trees spawn further out.
Features are **enabled by default** on v5022, with an independent
`Park obstacle ring (solid)` toggle in the settings drawer.
The existing pose-lattice mission planner gets approximate obstacle
keep-outs; it has NOT learned to climb them, so manual testing and
unexpected falls are still separate concerns.

These are browser viewer changes only: PPO, trained policy and
the protected `main` demonstration retain their existing contracts.

**Update an existing v5022 WSL checkout:**

```bash
cd ~/emu-v5022
git pull --ff-only origin feature/v5022-forward-terrain
source .venv/bin/activate
python -m pytest -q tests/test_terrain.py
npm test --prefix emu/viewer_src
emu viewer-build
emu viewer showcase/champion.emupolicy --no-open
```

Restart the viewer process after rebuilding and do a hard reload in the
browser so older JavaScript doesn't survive the update. Start in
**Flat reference**, then try **Step up**; confirm live depth still
updates even though the slope telemetry reads "No terrain underfoot".
Next choose **Current viewer terrain** and explore the six new features
with the **Park obstacle ring** toggle on and off. Compare forward
mounts **0/30/60/90 mm** at the same tilt to find whether a near-root
mount is obstructed by the body/legs.


### Optical footprint correction after first viewer inspection

The earlier diagnostic ray builder applied horizontal image-plane angle as
global yaw AFTER rotating each row's pitch. At downward tilt approaching 90°,
its horizontal component was multiplied by `cos(pitch)` and therefore
collapsed: the field appeared as a V/triangle rather than a 2D matrix.
**That was incorrect for this sensor.**

Both `emu/terrain.py` and `viewer_src/src/sensor.js` now use the same
square **45° × 45° optical image plane** and rotate the entire optical
basis downward together. Each of 16 zones remains a real 2D square
angular patch; 1, 4 or 5 subray samples produce 16 zone measurements.
At **90° straight down over flat ground**, 4×4 zone centres lie in a
regular square grid. At **50° oblique** incidence the ground projection
naturally becomes a stretched trapezoid. All beams still originate from
one physical sensor centre, so a pyramid of converging beams in a
side view is *not itself* a sensor fault.

Viewer changes:
- Tilt now ranges **0° (ahead) to 90° (straight down)**, including true
  90° rather than previously stopping at 75°.
- **Show beams from sensor** keeps the actual 16 depth directions.
- **Show 4×4 ground footprint** uses 25 extra boundary ray intersections
  to draw a surface-projected 4×4 outline. Missing or occluded corners
  remain open rather than being filled in with imaginary terrain.
- Milliseconds per scan **includes** that extra ray work. Copied readings
  and the HUD separately identify the 16/64/80 range samples and the
  optional 25 drawing-only footprint intersections. Disable footprint
  during ray-count benchmarking for comparable numbers.
- Python/JS cross-language direction parity and separate 90° flat-plane
  square-footprint tests now protect against the original regression.

**Recheck the viewer** after pulling/rebuilding. Start with *Flat
reference*, set **Tilt 90°**, **Mount 0 or 30mm**, leave footprint ON,
and orbit directly above the robot. Four rows and four columns must be
visible as a square grid on the ground (subject to real chassis
occlusion). Try Tilt 50° next: the patch elongates forward. If the grid
breaks or shows orange blocked readings at 0mm, the mounting origin
may be inside the chassis; move it a little forward and compare.


### Viewer selection round 2: provisional 0mm / 70° mount

The first interactive visual test preferred the sensor optical centre at **0mm
forward relative to the chassis/root frame**, **70° downward**, keeping the
existing **-18mm vertical** offset. The viewer and the standalone pure
Python terrain sensor now default to that candidate. This is an
**experimental setting**, not a frozen CAD bracket or actor policy schema.

The user's observed 4×4 pattern is useful:
- **Rear two middle zones:** frequently see moving feet/legs. These
  returns are preserved as real short-range returns, not silently erased.
  There is **no privileged simulator self/foot-identity bit** in the raw
  actor candidate. The viewer shows foot-hit debug colours only to assist
  human evaluation. A beam measuring a foot is measuring **camera-to-foot
  slant distance**, not directly foot-to-ground clearance.
- **Rear outside corners:** retain views of ground on opposite sides of
  the body and might detect cross-slope, especially when feet block the
  rear centre. The existing browser test now explicitly checks that a
  10° cross-slope produces distinguishable slant ranges on these corners.
- **Forward rows:** still project in front of the robot, but at 70° the
  farthest useful footprint is inherently closer than at 50°. Existing
  45° square FoV and robot pitch control affect exactly how far.
- **Required physical follow-up:** repeat with leg swing, chassis pitch,
  more than one terrain shape and sensor status/noise. Small reflectance
  or foot geometry differences on a real VL53L8CX could change which
  zones return a valid value.

A useful follow-up scan: leave **Forward 0mm**, try
**65° / 70° / 75°** tilt over flat, step-up and crest with representative
pitch commands. For optional candidate sweep use:

```bash
EMU_SENSOR_SWEEP=1 EMU_SENSOR_TILT_DEG=70 EMU_SENSOR_FORWARD_MM=0 \
  python -m pytest -q -s tests/test_terrain.py -k optional_post_viewer_mount_sweep
```

The default sweep now compares **65/70/75° × 0/30/60mm**.
Compare all 9 candidates only if it helps distinguish foot occlusion
from forward look-ahead. The goal is the best **multi-zone signal**, not
a fabricated per-foot height sensor and not maximum look-ahead alone.
No v5022 PPO inputs, archived actor contracts or checkpoints have been
changed in this stage.
