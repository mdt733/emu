# handover.md — EMU 40 cm carbon-rod leg

Last updated 2026-11-08 (Australia/Sydney).
**Current analysis branch:** `research/v5060-carbon-stand-diagnostics` (model remains v5059 `h40_carbon2`).
**Full engineering proposal:** [docs/V5059_40CM_CARBON_LEG_PROPOSAL.md](docs/V5059_40CM_CARBON_LEG_PROPOSAL.md).

## v5060: 20M carbon h40 failure, source-grounded diagnosis (2026-11-08)

**User-supplied artifacts:** physical_matrix(1).csv/md from one
`h40_carbon2` seed 735, finished **20,054,016 world steps**.
The new 478.3g carbon model did NOT learn to stand:
* At 5/10/20M: all final-window fall rates 100%;
  episodes average 19.87 / 23.81 / 26.50 control steps.
  At 20ms control DT this is ~0.40 / 0.48 / 0.53 seconds.
* End vx 0.638m/s vs cmd 0.043m/s => likely uncontrolled
  movement, not learned fast walking.
* Simulated actuator forces over rated: **0.0%**; maximum
  rated-force utilization 95.0%, so v5058 hard cap appears
  effective in TRAINING. Nominal no-load speed exceeded in
  24.7% of samples: a backdriven falling joint can exceed
  motor no-load speed; do not infer an actively over-speeding
  motor just from this aggregate statistic.
* Average knee spring energy 0.0535J; the neutral-geometry
  two-spring preload estimate is ~0.162J. This is consistent
  with being away from its initial knee pose, NOT a direct
  fall-cause measurement.
* No independent held-out evaluation in the CSV.

**IMPORTANT MECHANICS HYPOTHESIS, not yet proven:** at h40
neutral the knee-to-foot x lever is ~0.1975m, so total 478g
weight divided equally between two feet gives ~0.463Nm
gravity-induced knee moment per side if transmitted as a
vertical reaction. The newly proposed spring is 450N/m and
19mm past its 31mm slack length with ~24mm moment arm;
at neutral it produces only **0.205Nm** extensor torque
per knee (~44% of the simple equal-foot standing load).
The rated servo is capable of supplying more, but an
untrained policy gets little passive support. This is a
first-order STATIC load estimate; true contact and other
joint moment distribution may differ.

Further candidates: two extremely soft passive shin hinge
axes (0.06327Nm/rad), 5mm-radius feet, imaginary ±25°
hard stops, unmodelled upper knee-drive linkage. Do not
claim any individual factor caused all falls until the
fall-cause and support telemetry are read.

**New diagnostic source branch**:
`research/v5060-carbon-stand-diagnostics`.
Extends existing `scripts/v5057_physical_tournament.py --report`
to extract already-saved 20M `metrics.csv` fall-height,
tilt, angular-rate, virtual strikes, support, height error,
shin-near-stop, measured knee forces and motor saturation,
plus 5M/10M/20M progression. This is ZERO-GPU forensic
analysis and DOES NOT mutate the trained model or checkpoints.

Get report WITHOUT rerunning training:

```bash
cd ~/emu && git fetch origin \
  '+refs/heads/research/v5060-carbon-stand-diagnostics:refs/remotes/origin/research/v5060-carbon-stand-diagnostics'
git worktree add --detach ~/emu-v5060-diagnostics \
  refs/remotes/origin/research/v5060-carbon-stand-diagnostics
cd ~/emu-v5060-diagnostics
source ~/emu/.venv/bin/activate
export PYTHONPATH="$PWD"
export EMU_PHYSICAL_PROFILE=h40_carbon2
python scripts/v5057_physical_tournament.py \
  --root "$HOME/emu-data/runs/v5059-carbon-physical" \
  --profiles h40_carbon2 --seeds 735 --report
```

Outputs in existing tournament root:
`physical_matrix.md` and `physical_matrix.csv`.
Analysis-only output may overwrite those REPORT files,
but never the checkpoints or original `metrics.csv`.
If these source columns are unavailable, the report prints
`—` instead of inventing values.

**Do not launch another 20M/200M run yet.** Compare fall
causes at 5/10/20M, evaluate a short gravity/neutral
support trial, choose whether to increase passive knee
support or modify shin mechanics after evidence. A spring
change requires an independent source/model SHA and run
directory, not a silent checkpoint resume. Keep the
existing branch and old 20M run archived.

## Most important status

This v5059 branch now contains a **real implemented experimental**
`EMU_PHYSICAL_PROFILE=h40_carbon2` generated MuJoCo/MJX model, the
existing tests and GPU smoke entry point. It is NOT a measured hardware
digital twin or proof of stable 40cm training. No experimental GPU training
has been executed by the assistant. The user wants an *honest* 40 cm EMU that walks fast,
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
  of v5058, **implemented h40_carbon2** model + real physical tests,
  proposal and this handover. NO GPU run has been attempted on it.

## Implemented 40cm research model (v5059)

New named profile: `h40_carbon2`, added to **one**
`emu/physical_variants.py` parameter bank and content-addressed model cache.
The older `h40` and all previous checkpoint/model SHA references remain
intact.

Provisional hardware parameters (replace these estimates on the next
physical measurement, and start a new source-pinned lineage):

| Parameter | Implementation |
| --- | --- |
| Neutral hip Z | 0.400m |
| Bare rod | solid 2mm nominal, flexural modulus 30 GPa |
| Knee-to-foot segment | 0.2793m (foot shrunk to 10mm ball diameter) |
| Shin flex | two passive axes per leg, pitch and lateral roll |
| Midpoint-hinge k, each axis | 0.063269 N.m/rad (cantilever tip-deflection equivalent) |
| Shin damping, each axis | ~0.000600 N.m.s/rad (ζ=0.15, assumed distal inertia) |
| Flex extent | ±25° **assumed modeling guard; do not credit a physical hard stop** |
| Knee spring | tension-only spatial tendon; tail 40mm beyond knee; 30mm thigh anchor; 450N/m, 31mm slack/rest |
| Knee spring neutral extensor torque | estimated ~0.205N.m; NOT real measured support |
| Foot | 5mm sphere radius (10mm diameter), approx 3g ball+mount |
| Total simulated moving mass | ~0.47828kg itemized proxy |
| Torso | 0.420kg lump: six 55g C018 motors + 90g chassis/battery/electronics |
| Thigh | two separate slender mass proxy geoms (one thigh, one knee-drive), NOT an exact four-bar constraint |
| C018 force | hard rated ±0.980665N.m per motor, inherited from v5058 |
| Actuated passive damping | 0.018 N.m.s/rad; friction 0.025 N.m; reflected armature 0.003kg.m² (all unmeasured) |
| Sensors | 360-input actor unchanged, existing pitch-shin flex/rate/load Hall proxies; lateral passive bend is not directly observed |

The *old* heavy/rigid `h40` physical summary below is historical; it
does not describe the new profile.

### WSL run and cold-start gate

```bash
cd ~/emu &&
git fetch origin \
  '+refs/heads/research/v5059-40cm-carbon-leg-physical-design:refs/remotes/origin/research/v5059-40cm-carbon-leg-physical-design' &&
git worktree add --detach ~/emu-v5059-carbon \
  refs/remotes/origin/research/v5059-40cm-carbon-leg-physical-design &&
cd ~/emu-v5059-carbon &&
test -f emu/physical_variants.py &&
git rev-parse --short HEAD
```

```bash
(
  set -e
  cd ~/emu-v5059-carbon
  source ~/emu/.venv/bin/activate
  export PYTHONPATH="$PWD"
  export EMU_PHYSICAL_PROFILE=h40_carbon2
  export XLA_PYTHON_CLIENT_PREALLOCATE=false

  # Actual generated geometry, mass, torque, spring and left/right checks.
  python -m pytest -q tests/test_core_physics.py
  python scripts/probe_carpet_prism_mjx.py --worlds 4 --frames 2

  # GPU physical & PPO gate (requires user's RTX 4060Ti).
  python scripts/smoke_gpu_carpet.py \
    --device cuda --worlds 2048 --frames 2 --ppo \
    --full-update --benchmark-updates 3 --prisms \
    --solver-iterations 2 --solver-ls-iterations 4

  # Fresh independent 20M carbon-model cold start, no old checkpoint.
  python scripts/v5057_physical_tournament.py \
    --root "$HOME/emu-data/runs/v5059-carbon-physical" \
    --profiles h40_carbon2 --seeds 735 --execute
)
```

**Important:** GPU smoke and 20M training are not run/verified here.
The first 20M run is a **gate**, not evidence of an already capable
robot. If fall rate stays high, investigate actual shin hinge stop
occupancy and spring geometry before lengthening the run.

## Uncertainty and next engineering refinements

The material grade, Hall calibration, motor electrical dynamics, 2mm rod
end clamps, exact knee-drive four-bar, actual battery mass, rigid flex
stops, carbon yield, and spring characteristics have not been measured.
Current model allows **hard ±25° passive flex stops** as a numerical
guard; if the policy leans against them, it is exploiting an unverified
mechanism. The second upper rod is a **mass proxy** and does not transfer
the exact four-bar forces. Update the source (not an untracked editable
XML cache) as physical parts are measured, keeping the old run immutable.

## v5059 CUDA smoke recycler check (2026-11-08)

User's first actual 2048-world CUDA smoke from commit `4dd05e1`
loaded the new carbon model with **0.47828kg and k=0.0632689
Nm/rad**, built real eight prisms and passed surface/reset checks.
It stopped BEFORE training at
`Recycler moved a physical prism sidewall under a foot sphere`.
Root cause was the **smoke test's fixed +20mm overhang position**:
the former 18mm foot +4mm guard must block at 20mm; the current
5mm carbon foot +4mm guard is already CLEAR at 20mm, so the
actual physics recycler correctly moved the old prism. There was
**no demonstrated unsafe recycler bug**.

Fix committed on the v5059 branch: `scripts/smoke_gpu_carpet.py`
now tests the exact loaded foot radius+4mm margin at 2mm INSIDE
(must block) and 2mm OUTSIDE (must recycle). The existing
`tests/test_core_physics.py` footprint guard tests both
5mm and 18mm radii through the same JIT path. The actual
production guard `feet_clear_of_abandoned_prisms` was **not
weakened or changed**.

The CUDA timer warning about delayed warmup and optional missing
Warp modules are not the cause of this abort.

**Next:** fetch the latest v5059 branch into the *existing*
`~/emu-v5059-carbon` detached worktree (do not `git worktree
add` again), then rerun the same CUDA smoke. Only if that passes,
allow the fresh h40_carbon2 20M independent cold start. Do not
claim 2048-world acceptance or learned walking from the
aborted test.

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
4. Verify the new **estimated** h40_carbon2 40cm v5059
   physical profile against real MuJoCo/MJX, then improve
   shin compliance, tensile knee spring and damping with
   measurements. Six motor rated hard limits remain fixed.
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
