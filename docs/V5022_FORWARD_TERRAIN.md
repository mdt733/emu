# EMU v5022 — Physical terrain and forward depth

**Branch:** `feature/v5022-forward-terrain` from protected `main`
(`2ec9e9db0a816c658131311129ef89ec7a595a12`).
**Status:** ground and sensor **prototype only**; NOT connected to actor,
PPO, training runs, viewer or public showcase. No change to `main`.

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
- Existing reward, PPO, checkpoint, exporter and viewer owners remain as-is
  until a reviewed end-to-end schema is ready.

No additional production module, standalone experimental harness, or
version-specific training script should be needed.

## Sensor selected

**ST VL53L8CX** (wide-angle discussion referred to ~63°; ST's specified
detection FoV is **65° diagonal, 45° by 45°**), 8×8-capable hardware
**operating in 4×4 mode**, delivering 16 range zones.

- Reference: https://www.st.com/resource/en/datasheet/vl53l8cx.pdf
- Driver: https://www.st.com/content/st_com/en/technical-documents/UM3109.html
- `terrain.py` holds **one optical mounting point** at body XYZ
  (+90, 0, -18) mm, tilted approximately **50° downward**. Provisional
  mounting geometry: verify leg swing, chassis clearance, PCB placement,
  wiring and actual usable FoV using CAD / a real robot.
- Zone ordering: internal rows are **far to near**; lateral ordering is
  defined in the robot frame. Check orientation mapping and lens inversion
  against the real VL53L8CX 4×4 driver before connecting hardware.
- Range validity: explicit per-zone validity, missing return represented
  by zero **plus invalid**. The 1.5m simulator cap is a training region of
  interest, not the published max ranging capability (4m).
- The reference simulator samples five rays per zone to approximate a
  finite footprint, and uses a median of visible rays; ST's histogram
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
