# v5059 | 40 cm EMU carbon-rod leg: source-grounded physical proposal

Status: **engineering proposal, NOT a validated physical robot or ready-to-train model**.
Branch: `research/v5059-40cm-carbon-leg-physical-design`; parent v5058 rated-torque hard cap.
Preserve `main`, the v5057 200M result, and source/checkpoint hashes. No more mega training
until the hardware assumptions and MJX load paths pass simple physical tests.

## 1. Intended hardware (user sketch/CAD, 2026-11-08)

* Hip neutral height: 0.400 m.
* Each shin: one long **solid approximately 2 mm carbon rod**, from ball foot
  to the knee, extending about **40 mm beyond the knee** to anchor a spring
  pulling towards the thigh. This shin is an actual load-carrying column,
  not only a soft decorative hinge.
* An unloaded slender reference rod is mounted near the midpoint of the
  shin; a magnet near the knee relative to a Hall sensor measures bending.
  It must NOT become a second structural beam.
* Above the knee: a thigh carbon rod and a second knee-drive/tendon
  carbon rod. Both MAY carry meaningful loads. Model their real pinned/fixed
  endpoints and actual linkage geometry, not an arbitrary second knee spring.
* Foot: approximately 10 mm spherical ball; **working assumption: 10 mm
  DIAMETER, so radius 0.005 m**, not yet a finalized hardware measurement.
* Spring is a **tension/pulling** element from shin tail to thigh. Its
  CAD endpoints and preload are NOT established. Do not assume the old
  compression-tendon implementation has the right force sign.

The uploaded render shows the intended joint/linkage layout, not physical
section sizes or final dimensions. The assembled CAD, exact spring
attachment positions, real material grade and masses are still required.

## 2. What the current source ACTUALLY models

From `emu/robot.xml` and `emu/physical_variants.py`:
* `h40`: neutral hip 0.400 m, each nominal shin **0.27013 m**,
  each nominal thigh 0.27013 m, mass **0.56246 kg**.
* Shin is two thick-looking radius-8 mm collision-inertial capsules, each
  15g at base height and scaled linearly by h40 geometry: total **47.7 g
  per shin** at h40, PLUS a separate 10 g foot. The capsules are noncontacting
  but their mass and inertia remain in physics. Mass is NOT thin-carbon-realistic.
* Passive shin hinge: ONE fore-aft pitch DOF at shin midpoint,
  `k=4/1.5890 ≈ 2.517 N.m/rad`, `c=0.08 N.m.s/rad`,
  rigid limit ±10 deg; **no lateral compliance**.
* Knee support: spatial spring with 30/30mm attachment geometry and a
  **60 mm zero-force length**, `k≈1397 N/m` for h40. At neutral knee:
  compression ≈17.57mm, spring force ≈24.55 N, and extensor torque
  ≈0.521 N.m. At -55deg: about **1.68 N.m** extensor torque. This
  implemented COMPRESSION spring pushes sites apart; the proposed real
  spring PULLS between a different pair of attachments. These are NOT
  equivalent until actual CAD mechanics are measured.
* Existing foot sphere radius is **18 mm (36 mm diameter)**, not 5 mm.
* SIX C018 actuator limits on v5058 are ±0.980665 N.m at actuator output.
  However each driven hinge retains a legacy passive damping of 0.624
  N.m.s/rad, creating potentially unphysical braking while unpowered.
* `emu/contracts.py` ALREADY has `shin_flex_r/l`,
  `shin_rate_r/l`, `shin_load_r/l` within 360-input observation.
  `environment.py` makes the flex/rate estimates noisy but forms shin
  "load" from |bend angle|/0.10: it is a PROXY, not calibrated Hall load.
  Keep observation width and history contract unchanged unless a deliberate
  actor interface redesign is justified.
* The simulation currently aggregates .30kg at the torso; six real
  55g C018 servos ALONE weigh ~.330kg. This is a warning that actuator,
  battery, linkage and chassis masses are not yet itemized by actual
  moving body. Do NOT infer total finished mass by just deleting rod mass.

## 3. Source-backed 2mm material range

For a solid circular rod `I = π d⁴/64`; `EI` is bending rigidity.
Using free load-carrying shin length **L = 0.270 m**, nominal
density **1600 kg/m³** (an assumption: weigh actual rods).
A supplier's 2mm UD pultruded product states axial E≈28–40 GPa;
its separate material TDS states **FLEXURAL E≈20–30 GPa**.
High-modulus rods vary widely; CST lists ~134 GPa tensile modulus
for its standard grade, which is NOT proof of this rod's flexural E.
Sources:
* https://www.easycomposites.us/2mm-carbon-fiber-rod
* https://media.easycomposites.co.uk/datasheets/EC-TDS-Carbon-Fibre-Pultrusions.pdf
* https://www.cstsales.com/carbon_rods.html

| Candidate | d | assumed flexural E | mass of 270mm bare rod | EI | EI/L pure-moment angular equivalent | 0.75 EI/L midpoint-hinge tip-deflection equivalent | Euler pinned-pinned Pcrit, if rod alone |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C25 (ordinary material) | 2mm | 25GPa | 1.36g | 0.01963 N.m² | 0.0727 N.m/rad | **0.0545** N.m/rad | **2.66N** |
| C60 (intermediate hypothetical) | 2mm | 60GPa | 1.36g | 0.04712 N.m² | 0.1745 N.m/rad | **0.1309** N.m/rad | **6.38N** |
| C134 (premium hypothetical) | 2mm | 134GPa | 1.36g | 0.10524 N.m² | 0.3898 N.m/rad | **0.2923** N.m/rad | **14.25N** |
| D3-C30 (stiffer spar) | 3mm | 30GPa | 3.05g | 0.11928 N.m² | 0.4418 N.m/rad | **0.3313** N.m/rad | **16.15N** |
| D3-C134 (stiffer premium spar) | 3mm | 134GPa | 3.05g | 0.53279 N.m² | 1.9733 N.m/rad | **1.4800** N.m/rad | **72.13N** |

These are IDEALISED estimates, not exact MuJoCo joint constants. For one
midpoint hinge subjected to a transverse tip force: its end deflection
is `F L²/(4k)`, compared with an ideal cantilever `F L³/(3EI)`;
thus `k=0.75 EI/L`. For a pure bending moment and total angular
change, `k=EI/L`. Real knee/foot boundary conditions, axial compression,
large deflections, mounting sleeves and load-sharing produce different
effective parameters. Fit to a measured force-deflection curve.

The whole 310 mm cut length of the 2mm rod weighs ≈1.56g at 1600kg/m³,
BEFORE mount, foot, sensor, magnet and wiring. Never substitute 1.56g for
the entire leg subassembly.

### Buckling and spring-lever red flags: must prove before selecting 2mm

Simplified Euler pinned-pinned column estimate:
`Pcrit = π² EI / (K L)²` with K=1. Boundary stiffness, bows,
loading angle and bracing change Pcrit. At 0.562kg, 1g weight is
~5.52N; if **one inclined 45deg member** carried the entire load
axially, its force might be ~7.8N before landing impulses.
This is a *stress-case calculation*, NOT a validated load-path prediction.
Ordinary 2mm C25 with 2.66N Pcrit is therefore a **red flag**.
A 3mm high-modulus main shin would give much greater buckling margin
for only ~1.7g more bare rod than 2mm.

The old h40 knee support gives ~0.52N.m around neutral. If
transferred via an UNBRACED **40mm** extension of a 2mm shaft,
it takes ~13N transverse spring force and ~0.52N.m at the knee
section; the idealized outer-fibre bending stress is ~**663 MPa**.
At deep flex and 1.68N.m it would exceed 2GPa. One supplier's
ordinary 2mm rod TDS flexural strength is ~250–400MPa.
THIS IS NOT A STRENGTH-VALIDATED SPRING LEVER. Real mount geometry
may change the local bending moment: obtain exact attachment and
bearing locations. Prefer a short reinforced lever/sleeve or revised
knee spring leverage; DO NOT assume the bare rod can take it.

Before training: verify member axial forces, buckling, bending stress
and pin/adhesive loads under single-leg stance and >1g landings.
Do NOT count the Hall reference rod as structural reinforcement.
The two upper rods may share loads, but need their actual linkage
constraints to claim any safety benefit.

## 4. Proposed measurable v5059 parameters

| Parameter | First physical proposal | Confidence / dependency |
| --- | --- | --- |
| Hip neutral Z | **0.400 m** | user choice, source-proven |
| Primary shin rod d | **2.00 mm** for test candidate; include **3.00 mm** safety candidate | user approximate; strength/buckling unresolved |
| Nominal knee-to-foot length | **0.270 m**, re-solve if foot radius changes | geometry at original 18mm foot |
| Spring anchor tail | **0.040 m** beyond knee | user approximate; inspect knee moments |
| Rod material bending modulus | **25, 60, 134 GPa research scenarios**, NOT a declared fitted material | purchase-grade and flex test needed |
| Midpoint hinge k_pitch | **0.0545 / 0.1309 / 0.2923 N.m/rad** at 2mm; tip-deflection calibrated INITIAL VALUES only | match physical bending test and boundary |
| Shin flex damping | **derive** `c=2 ζ √(k J_eff)`; start ζ=0.05,0.15,0.30 from measured free decay, not old 0.08 | true effective inertia J_eff unknown |
| Example provisional damping | for k≈0.0545 and J_eff≈0.00018 kg.m²: **c≈0.00031 / 0.00094 / 0.00188 N.m.s/rad** | EXAMPLE only; recompute from new foot+mount masses |
| Shin hard flex stops | **do not presume ±10deg**; characterize from rod strain/clearance and limit actual mount contact | large bend is geometric not rigid hinge |
| Lateral shin compliance | **required**, likely second passive-axis DOF or validated equivalent | current model has none |
| Carbon rod mass | **~1.36 g** per 270mm bare 2mm rod (density 1600kg/m³), **3.05g** per 3mm | do not omit connectors and knee bearings |
| Foot ball | **5mm radius** provisional if 10mm diameter intended | collision geometry change; recompute neutral Z/length |
| Foot mass | **weigh ball + adhesive + mount** | cannot infer just from diameter |
| Hip/thigh motors | six C018 = **~330g** combined, all allocated to their real bodies | manufacturer specification; full mass ledger mandatory |
| Knee support spring | replace fixed legacy compression-only geometry with measured **tensile geometry** and force/preload, after checking stress | user's physical design differs |
| Peak actuator torque | **±0.980665 N.m hard cap** on 6 motors, via v5058 | must pass v5058 GPU acceptance |
| Powered joint 'free brake' | **legacy .624 N.m.s/rad damping must be removed or calibrated** | do not invent damping until unpowered backdrive measurement |
| Hall sensor observation | retain existing 360 schema, 2 shin flex/rate channels and load proxy; implement a proper magnet-distance measurement model later | no automatic privileged torque signal |
| ToF | same 4x4 real ray geometry, re-aim only if height changes | no sensor cheating |
| Training world | eight real prism colliders, 2048 environments, six actuators | preserve current single trainer |

**Important:** `c=0.01..0.04` suggested earlier was an unjustified
guess. Damping must scale with the much lower true lower-shin
inertia and actual flex stiffness; at the illustrative
`J_eff=0.00018` a `c=0.08` hinge is heavily over-damped.

**Important:** changing the foot ball radius from 0.018m to 0.005m
also changes the exact hip-to-foot neutral geometry. At a
0.400m neutral hip and 45deg folded leg, each nominal member length
would increase to approximately `(0.400 - 0.005)/√2 ≈0.2793m`,
before incorporating knee/foot offsets and rod-extension geometry.
Recompute neutral contact positions and actual eight-prism collision
tests; never just change the sphere radius without fixing pose.

## 5. Build order and acceptance gates (lean code, no scaffolding pile)

**Gate A — motor fix (v5058):** finish physical tests and actual
2048-world CUDA PPO smoke, verify motor over-rated fraction ZERO.
Do not claim acceptance yet; run must be performed on operator GPU.

**Gate B — mechanical bench/geometry:** procure or select exact
2mm rod grade and test a short coupon (length, mass, cantilever
load/deflection 2–3 forces, lateral and pitch, return-to-zero,
free-decay). Record knee pivot/upper-rod linkage CAD coordinates,
actual spring attachment locations and spring force at 2–3 lengths.
Check two axes' buckling and permanent deformation. Weigh each
moving part INCLUDING all six servos and battery separately.
This MUST precede claiming a 2mm 40cm load-bearing design is viable.

**Gate C — physical model, same code skeleton:** ONE 40cm model,
true masses and inertias for rod + fittings, foot radius, actual
tension-spring torque versus knee angle, measured shin flex and
damping, both load-bearing rods above knee, realistic passive
joint drag, six rated motor ceilings. Prefer two-axis shin
compliance if validated and affordable in MJX; do not add a
second actor, optimizer, or a museum of profile branches.
Observed Hall signal must arise from measured deflection, not
privileged contact force. Recenter mass only if physically
achievable by moving actual chassis/battery.

**Gate D — acceptance:** compile real MuJoCo/MJX; verify
left/right symmetry, actual full-load single-leg stance and
physical stability in both pitch+roll, spring sign and stored
energy, correct 10mm-ball contact and foot collision, buckling/
stress outside MuJoCo assessed from real loads, maximum C018
output ≤ rated even with randomization, no free passive brake.
Current tests budget stays ~24 core tests: REPLACE/extend the
existing relevant tests, no new battery of one-off tests.
2048-world GPU smoke, memory and post-compile steps/sec.

**Gate E — learning science:** first 20M cold-start stand and
0.1–0.2m/s, two seeds if viable. Only then 50–100M development
with command, slope, lateral, yaw, recovery and retention cohorts
(step Academy off). Fixed held-out 350-frame physical exams at
0.1/0.2/0.4/0.6/0.8m/s and ±3/6/10/12deg plus shoves;
report survival AND actual tracking, slip, motor saturation,
current/temperature when measured, knee stress and passive
spring energy. Compare identical seeds and fixed exam setups
against previous 40cm model while explicitly labeling that
the old 200M used excessive available stall torque.

## 6. What NOT to do

* Do not treat vendor axial modulus as measured flexural modulus.
* Do not treat Euler buckling as a substitute for full CAD
  link-load analysis; use it as a critical safety warning.
* Do not use the soft Hall reference rod as an extra stiffener.
* Do not carry old 47.7g simulated shin mass into a 1.36g bare
  spar, nor discard motor/mount/foot mass hidden in that total.
* Do not label old compression strut `k≈1397N/m` a faithful
  replica of the pictured tension spring.
* Do not reuse old v5057 200M weights with a changed physical
  model without an explicitly separate compatibility/transfer
  research exercise.
* Do not run a long 200M campaign until the cheap mechanical
  load tests and actual GPU sanity check pass.

### Primary evidence

Repository: `emu/robot.xml`, `emu/physical_variants.py`,
`emu/contracts.py`, `emu/environment.py`,
`emu/domain_randomization.py` on the v5058 branch.
Manufacturer C018 nominal ratings:
https://ftservo.github.io/ftservo-wiki/products/models/st-3215-c018/main/
MuJoCo actuator force limits:
https://mujoco.readthedocs.io/en/latest/modeling.html
