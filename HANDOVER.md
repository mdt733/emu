# EMU — current handover (v5020 experiment, October 2026)

**Reference `main` is PROTECTED.** It contains the known-working v5016
Pinnacle 232-input actor and approved 260M showcase. Do not modify
`main`, the public Playground or its champion policy until a stronger
trained result has been independently demonstrated and the user
explicitly approves a reviewed merge.

**Sole current experimental source branch:**
[`feature/v5020-coldstart-terrain`](https://github.com/mdt733/emu/tree/feature/v5020-coldstart-terrain).
All changes in this handover refer **only** to that branch.

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
