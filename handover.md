# handover.md — EMU 40 cm carbon-rod leg

Last updated 2026-11-08 (Australia/Sydney).
**Current research branch:** `research/v5059-40cm-carbon-leg-physical-design`.
**Full engineering proposal:** [docs/V5059_40CM_CARBON_LEG_PROPOSAL.md](docs/V5059_40CM_CARBON_LEG_PROPOSAL.md).

## Most important status

This v5059 branch is an **engineering proposal and handover**, NOT
completed simulator physics, a tested carbon-leg system, or hardware
validation. The user wants an *honest* 40 cm EMU that walks fast,
remains stable, can handle slopes and recovers from shoves. Multi-lip
stairs are not the immediate priority. Keep one six-actuator model,
360-input single actor, one JAX/MJX trainer, eight physical convex
prisms, 2048 environments, concise tests (~24 core tests), Git
research branches, and source-pinned experiments; main remains frozen.

## Branches and results

* `research/v5057-height-physics-matrix`: height profiles
  base26/h30/h34/h40; h40 20M seed 735 and 200M athletic continuation.
  The 200M h40 final training window achieved 0.258m/s vs requested
  0.314m/s, 16% falls. The independent 350-frame exam on the final
  policy: 200/200 fast survived up to 0.95m/s requests, 199/200
  uphill survived, 200/200 downhill survived across up to ±12deg.
  **NOT hardware-feasible evidence**: actuators were allowed up to
  STALL torque ±2.942Nm, and ~57.7% sampled motor output exceeded
  the rated ±0.981Nm. Keep old checkpoints immutable.
* `research/v5058-c018-rated-torque-hardcap`: draft
  [PR #49](https://github.com/mdt733/emu/pull/49).
  Limits all six motors directly via XML `forcelimited=true`
  and `forcerange=±0.980665Nm`, limits domain-randomized
  motor output too, and adds actual MuJoCo tests + CUDA smoke
  fail-closed assertions. **CUDA acceptance has not yet run
  on the user's machine**. The altered physical SHA prevents
  silently resuming old v5057 checkpoints.
* `research/v5059-40cm-carbon-leg-physical-design`: child
  of v5058. Stores actual rod-design parameter proposal and
  this handover. NO 200M run has been attempted on it.

## User's physical leg design (see user sketch and CAD in chat)

* Approx 400mm neutral hip.
* A **solid ~2mm carbon-fibre rod runs all the way from
  a small (~10mm diameter, tentative) ball foot to knee**.
  The same rod continues ~40mm beyond knee; the end is connected
  by a spring pulling toward the thigh.
* Small reference rod attaches near halfway up shin and extends
  to magnet near knee, where a Hall sensor detects flex displacement.
  This reference rod is intended to be **unloaded/nonstructural**.
* Above knee are TWO long slim carbon members: thigh rod and
  actuation/tendon member. Either/both may carry significant loads.
* Important: this is deliberately a **bending beam in an
  articulated mechanism**, not a stand-alone pin-ended column.
  Carbon-rod bending strength can be high. The user's latest
  correction was: "pinned-column buckling is for columns, in
  flex these rods are extremely strong."
* Do not automatically disqualify the 2mm rod using an isolated
  Euler column formula. That formula only applies to a specific
  axial compression and support-condition scenario. Distinguish
  bending strength, bending stiffness and compressive instability;
  solve the actual linkage's axial and bending loads first.

The user-provided CAD and paint image are attachments in the
conversation and are **not committed in Git**.

## What source currently does, and why it is not hardware-faithful

* h40 source geometry (from `physical_variants.py`):
  ~270.13mm long thigh and shin each; overall ~562g.
* Shin upper/lower mass proxy combined ~47.7g **per shin** plus
  ~10g foot; one 2mm bare 270mm carbon rod at assumed
  density 1600kg/m3 weighs only ~1.36g, excluding fittings.
* Passive midpoint hinge is ~2.517 Nm/rad,
  0.08 Nms/rad damping, ±10deg hard stops, forward/back
  pitch only. Main carbon shaft can also bend laterally.
* Real rod modulus and attachment conditions need measured
  calibration. The initial analytic midpoint equivalent
  k_pitch candidate values for 2mm by hypothetical flexural E
  25/60/134 GPa are **0.0545/0.1309/0.2923 Nm/rad**.
  These are *not* established real-spring parameters and
  assume a cantilever tip-deflection model mapped to one
  midpoint hinge. Do not blindly use 0.30 Nm/rad.
* Shin damping must be estimated from free-decay tests or
  damping ratio and actual effective inertia; old 0.08
  is not a measured carbon property.
* Simulated knee support is a **compressed tendon** spring
  (~1397N/m h40) whereas user proposes a tensile spring
  from extended shin tail to thigh; geometry and force sign
  must be rebuilt from actual anchor coordinates.
* Current foot sphere has radius 18mm / diameter 36mm;
  user's likely ball diameter ~10mm requires neutral-stance
  geometry and real prism collision adjustments.
* The 300g torso lump plus heavy old rod proxies does
  not explicitly allocate the six 55g servos (330g), battery
  and fasteners; prepare a moving-body mass ledger.
* Actuated joint legacy damping 0.624 Nms/rad can act as
  an unpowered free brake. Need real backdrive data.
* **Do not invent new observation channels:** `emu/contracts.py`
  already has `shin_flex_r/l`, `shin_rate_r/l` and
  `shin_load_r/l` in the exact 360-input actor. Present
  shin load is a flex-derived proxy, not calibrated Hall load.

## Action plan

1. Complete v5058 CPU physical tests and 2048-world CUDA
   PPO smoke before asserting rated torque enforcement.
2. Obtain real linkage coordinates (knee, shin extension,
   spring endpoints, thigh and second rod), rod grade and
   cross-section, total foot size, and mass distribution.
   Measure rod flex/return/damping, spring force-vs-length.
3. Resolve load sharing through the actual articulated
   linkage. Inspect axial compression only if demonstrated,
   and calculate bending stress / local spring-lever moment;
   do not treat pinned Euler estimate as an observed failure.
4. Implement one honest, mass-corrected 40cm v5059
   physical profile, with calibrated shin compliance,
   tensile knee spring, foot contact, realistic motor
   damping and unchanged six motor rated hard limits.
   Prefer direct physical evidence over speculative constants.
5. Add/adjust only existing essential tests, validate
   with real MuJoCo/MJX physics and 2048-world PPO GPU.
6. Do 20M cold stand-and-walk gate; then expand speed,
   slope ±3/6/10/12deg, lateral, yaw, shove recovery.
   Evaluate held-out physical trials and rated-load
   saturation. Do not pour another 200M into a known
   physical shortcut.

## Source of truth

The specification `docs/V5059_40CM_CARBON_LEG_PROPOSAL.md`
contains all parameters, conditional calculations, vendor
material links, motor limits, design risks, evidence and
acceptance gates. Revise this handover whenever a material
design decision changes. Do not add duplicate historical
scaffolding or outdated branch docs.
