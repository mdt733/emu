# EMU — current handover (v5020 experiment, October 2026)

**Reference `main` is PROTECTED.** It contains the known-working v5016
Pinnacle 232-input actor and approved 260M showcase. Do not modify
`main`, the public Playground or its champion policy until a stronger
trained result has been independently demonstrated and the user
explicitly approves a reviewed merge.

**Sole current experimental source branch:**
[`feature/v5020-coldstart-terrain`](https://github.com/mdt733/emu/tree/feature/v5020-coldstart-terrain).
All changes in this handover refer **only** to that branch.

## Terrain implementation acceptance — 22 October 2026

**Overnight launch is blocked in source until the remaining acceptance gates
pass.** `--check` performs no PPO update and reports NOT OVERNIGHT READY;
`--run` exits before environment setup or run-directory creation.

Foundation qualification now uses an additional fixed +0.10 m/s forward
panel. At least 20% of original worlds must complete the full exam without
falling, travel at least 5 cm and 30% of commanded distance, show mean lift
evidence >=0.12, upright fraction >=0.80, support fraction >=0.50 and forward
tracking MAE <=0.08 m/s. These are initial basic-walking thresholds, not proof
of a strong policy or certified force-based gait. Advanced skills are not
required. Old survival-only CSVs are unknown and cannot auto-qualify or bypass
the gate through an explicit path.

Frozen summary metrics require full first-episode coverage for survival
(including non-fall timeouts) and exclude reset aftermath and report tracking/gait
means over observed first-episode frames. Walking qualification requires the
whole horizon, so incomplete evidence cannot qualify. Tournament metrics
reject negative errors/effort and out-of-range fractions.

Checkpoint and 312-input sensor formats are unchanged. Metric columns have
expanded: new source uses a new pinned run root; do not append to historical
CSV files. Same-revision full-state resume remains supported.

Validation: focused qualification/scoring/refork/launcher regressions pass,
including the actual JAX-compiled evaluator with controlled telemetry.
Physical training and real-GPU acceptance remain unproven here. This host
has Python 3.12 CPU dependencies, not the supported fresh-WSL Python 3.13/CUDA
stack; do not treat these tests as fresh-PC acceptance.

Elevation groundwork now includes `terrain.elevation_course`: static 5/10/20/
30/40 mm up/down exam models with a two-metre upper landing, the unchanged
lower plane and exact first-hit sensing through the existing named box.
Static box-surface births use the actual upper surface when the robot starts
on a down-step platform. Native MuJoCo tests check top contact without phantom
floor support, ray normals, and ToF/native first-hit parity at 5/10/20 mm in
both directions; a two-world 312-input MJX reset checks down-step birth height.
These are exam fixtures only: batched random elevation lessons, post-landing
walking certification, sensor-memory diagnostics and curriculum integration
are STILL PENDING. The old hurdle crossing classifier is not elevation
certification and no elevation training competence is claimed.

Frozen terrain exams now revoke a recorded crossing if the robot later falls
in that original episode. Efficiency summaries also exclude reset aftermath.
Training event reduction rejects fractional/contradictory completion,
contact, crossing and fall records.

Validation record: baseline full suite finished with 196 passing tests and
one existing launcher-help assertion failure (fixed and passing on rerun).
The focused qualification/scoring/refork/launcher suite now passes 69 tests;
elevation geometry/ToF/contact/MJX-birth checks pass 7 tests. An earlier
72-test focused integration run included real three-world 312-input MJX
terrain and frozen exams. These overlapping runs must not be added together
as a count of unique tests. The complete suite was not rerun after all edits.

Next: replace the 80 mm hurdle lesson with genuine elevation transitions,
validate sensor reconstruction, integrate browser 312 inference, and run the
small real-GPU lifecycle before removing the launch gate.

## Current status

- Physics: six-actuator biped in MuJoCo/MJX, real collidable boxes and
  contact slopes, 312-input four-frame rolling ToF/flow preview.
- Training: 2048/4096 worlds, 32-frame PPO rollouts, on-policy governor
  and capability-field evidence, complete resumable checkpoints.
- New campaign prototype: `emu campaign-refork`; five strategies
  (General / Turning / Recovery / Efficiency / Steps), one common
  parent per 20M generation, a 2M failure probe per contender,
  matched frozen exams and single-winner re-forking.
- Height: step specialist begins low and grows from verified real
  crossings to a **current simulator maximum of 150mm**; exam heights
  are 5/10/20/40/80/150mm for every contender.
- State of proof: source and smoke tests have been run, but **no
  qualified successor policy exists yet**. The earlier 20→80M
  v5020 four-arm run collapsed; do not mistake its 380M aggregate
  steps for learned walking.
- A known design gap remains: the Champion Ladder stages vx/vy/yaw
  physical command maxima and sets `open_frontier=False`, while
  physical step height adapts independently. Unify carefully in
  shadow mode before enabling multi-axis frontier control.

## Reading order

1. [Current architecture](docs/EMU_ARCHITECTURE.md) —
   physics, measurements, PPO, governor, independent examiner and
   champion selection
2. [Consolidation checklist](docs/V5020_CLEANUP_PLAN.md) —
   staged source cleanup and strict exit gates
3. [Curated historical findings](docs/RESEARCH_HISTORY.md) —
   old campaigns, what was learned and what failed
4. [WSL setup and safe CLI](docs/INSTALL.md) —
   current read-only plan, schema checks and bounded first GPU run
5. [Current five-way recipe](docs/V5020_WINNER_REFORK.md) —
   absolute step targets and independent screening

Earlier handovers are in Git history, not multiple competing front-page
instructions. Existing training and exam output roots are **not**
source-code artifacts and are never removed by repository cleanup.

## Outstanding validation before another substantial GPU run

1. Confirm the exact 312-input foundation candidate from the user's
   saved checkpoints (including performance near 16–20M).
2. Ensure a clean pinned source, matching world count and a brand-new
   output directory; never overwrite or cross-revision resume.
3. Run the full test suite and real MJX contact/sensor smokes.
4. Start with just the five short candidate probes.
5. Examine survival, gait, numerical health and frozen course results
   before spending the remaining steps to 40M.
6. Promote a candidate only if it beats the incumbent with no
   unacceptable regression. Stop if none qualifies.

Historical research code can be retired from *this experimental branch*
after dependency audits, but the known-working reference remains
untouched in `main` and every historical Git object remains available.


## 2026-10-22 — foundation + genuine elevation examiner integration

Development remains ONLY on `feature/v5020-coldstart-terrain`. `main`,
the Pinnacle policy and the public showcase are untouched. The five-arm
overnight launcher remains explicitly BLOCKED pending full acceptance.

The existing frozen 20M foundation forward exam now requires its *complete*
first-episode evidence: independent survival, commanded/actual displacement,
stepping, posture, contact support and velocity error. An isolated pass-rate
number is no longer an acceptable exam. The winner-refork's held-out
screen also has a dedicated easy commanded-forward panel; survival-only
policies cannot pass the eligibility gate or continue provisionally.

True elevation geometry is a SEPARATE examiner from the original short-box
hurdle curriculum. The CPU/MuJoCo fixtures provide long raised landings for
both step-ups and step-downs. The new JAX frozen elevation examiner checks
source-side support, real step contact, both feet and torso past the edge,
destination-surface contact, upright posture, at least eight supported
destination frames, post-landing gait, and >=40mm further progression.
It reports contacts, certified crossings, falls, timeouts and incompletes;
reset episodes cannot create a new attempt. The static geometry and physical
contact oracle are examiner-only and are never actor observations.

Small non-training regression command from a fresh checkout:

```bash
python -m pytest -q tests/test_foundation_readiness.py tests/test_v5020_refork.py tests/test_v5020_coldstart.py
bash scripts/overnight_v5020.sh --check
```

Once an ACTUAL 312-input checkpoint exists, run separate, read-only exams:

```bash
emu screen-elevation --checkpoint "$FOUNDATION_PATH" \
  --course-root "$HOME/robot_lab/emu_runs/v5020_elevation_checks" \
  --direction up --height-mm 10 --envs 8 --horizon 350 --device cuda
emu screen-elevation --checkpoint "$FOUNDATION_PATH" \
  --course-root "$HOME/robot_lab/emu_runs/v5020_elevation_checks" \
  --direction down --height-mm 10 --envs 8 --horizon 350 --device cuda
```

`FOUNDATION_PATH` must be set explicitly to a compatible saved 312-input
checkpoint. The command does not rewrite checkpoint, metrics or campaign
outputs. Geometry files are immutable: rerunning identical conditions is
safe, differing files fail closed.

**NOT YET ACCEPTED:** no verified GitHub Actions/MJX/GPU results for this
revision, no demonstrated successful real-model elevation crossing, no
integrated mixed-height per-world elevation TRAINING curriculum, and no
312-input browser policy execution parity. These remain hard gates against
enabling `overnight_v5020.sh --run` or promoting a Pinnacle successor.


### 20cm actor lookahead / observation contract

The forward ToF model nominally samples two spots 20cm ahead; previously
the 312 actor read out only through 15cm. Since the rolling map deposits
each real hit within a narrow 2cm window, a newly sighted edge could be
visible in the full 51-cell diagnostic while still absent from *all* the
actor's 20 terrain channels. The fifth distance on each side is now 20cm:
`[-5,0,5,10,20]` cm, still ten heights + ten confidences per 78-field
observation and four-frame 312-input actor history.

This intentionally changes the semantics of two channels on each frame.
`PREVIEW_RANGE_FIELDS`, `PREVIEW_VALID_FIELDS`, sample indices and the
rolling model revision were changed together, so the persisted sensor
fingerprint changes. Existing experimental 312 checkpoints with the older
15cm map contract are NOT silently accepted. Historical 220/232 observation
schemas remain unchanged. Browser 312 policy parity is still outstanding.

The new regression sends two forward rays and three tripod rays through
the *same* long-box collision geometry for true 20mm up/down transitions,
projects only sensor returns through the tripod estimator, inserts measured
signed heights into the V memory, checks the actual actor readout and
requires invalid rays to remain unknown (zero confidence). This is a
source-level numerical/MuJoCo test, not evidence of learned competence.
The elevation examiner also reports source-side exposure separately from
contact, crossing, falls, timeouts and incomplete attempts.
