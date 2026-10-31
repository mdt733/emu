# EMU — measured release gates for future successors

**Baseline status (31 October 2026): owner-approved v5032 260M,
360-input Pinnacle is the new `main` actor.** It has known limitations
explicitly recorded in the [release card](V5032_PINNACLE_RELEASE.md).
The previous v5016 232-input champion is retained at
`2ec9e9db0a816c658131311129ef89ec7a595a12`.

This checklist defines the bar for **future** successor policies;
release of v5032 as a reference milestone is not a claim that all
future capability targets have already been met.

The target is not just higher rewards, harder curriculum targets, or
a new neural input shape. It is a **genuinely better six-servo robot**:
measured fast dynamic locomotion approaching the incumbent's ~1m/s
**requested** forward-speed envelope, plus supported terrain crossing,
resilience and efficient control.

## 1. Training provenance and code hygiene

- [ ] Fresh training or provably compatible source-pinned lineage;
  log exact source SHA, seed, JAX/MuJoCo versions and checkpoint hashes
- [ ] Hold completed training checkpoints immutable; **never** resume
  the running source under a later cleanup branch
- [ ] One simple launcher path and one shared campaign-support module;
  dead experiments archived by immutable Git SHA, not active imports
- [ ] No unexplained duplicated rewards, alternate actor schemas or
  undocumented privileged observation fields
- [ ] No stale README, dead links, superseded "current" run instructions
- [ ] Automated Python checks, full 2048-world CUDA smoke, checkpoint
  resume/export and browser build tested on the actual candidate revision
- [ ] Current code and documentation pass peer review; no necessary
  capability or release test was deleted merely to reduce code size

## 2. Independent physical competence

Use **common physical tasks** to compare today's 360-input `main`
with any future challenger. For historical 232-input comparisons,
retain each actor's original sensor contract and use identical
body, ground, disturbances, command schedule, horizons and scoring.

- [ ] Signed forward tracking at 0.15, 0.3, 0.6, ~1.0 m/s; record
  actual mean velocity, steady-state error, falls and recovery. A
  1m/s teacher request is **not** a 1m/s pass.
- [ ] Reverse, lateral and yaw including direction changes and braking,
  without hidden turning-back shortcuts
- [ ] Four-bearing slopes (uphill, downhill, two cambers), record
  physical stability **and** commanded motion/heading separately
- [ ] Steps: support-verified **UP** and **DOWN** at 1, 2, 2.5, 3,
  3.5cm and onward; no mixed up/down headline scores
- [ ] Realistic combinations of incline + sharp ledge, orientation
  variation and terrain perturbations; no false geometric shortcuts
- [ ] Resilience conditioned on known shove, direction, command
  and remaining support (not merely training resistance target)
- [ ] Effort measured as meaningful shaft-work proxy per useful
  metre; do not call it electrical consumption
- [ ] Complete per-world JSON and time-resolved traces for difficult
  events, more than one seed and more than eight worlds for claims
- [ ] Retain already passed easier tasks when hard frontiers expand;
  certify beyond the specific held-out curriculum fixtures

## 3. Navigational safety and deployability

- [ ] Delayed/noisy 4×4 ToF observations provide measurable lead time
  to detect a high step or missing landing surface
- [ ] The dynamically unstable policy can **actually slow, hold,
  backstep and recover** under supervisor commands
- [ ] Prove a deliberate safe refusal/replan is not a mere stalled
  policy, and measure both false refusals and hazardous overcommitment
- [ ] No supervisor has access to secret terrain geometry at runtime;
  no hard-coded servo climbing animation or alternate actor
- [ ] Verify exact actor/export observation contract in browser replay,
  including all sensor timings and units

## 4. Release and rollback

- [ ] Share comprehensible README, install and safety docs, concise
  handover and historical links; avoid active v5024/v5025/26/27
  launcher clutter
- [ ] Leave the current approved 360-input `showcase/champion.emupolicy`
  unchanged until a *future* candidate has owner-reviewed evidence
- [ ] Explicit owner approval for the new champion and for merging to
  `main`; this checklist does not authorize either
- [ ] Record release source and checkpoint fingerprints, deterministic
  export, viewer smoke, and rollback to the prior 260M baseline
- [ ] Update the public Pages deployment only with a reviewed,
  independently tested actor + frontend combination

## Current candidate, release workflow and retained historical reference

The completed [v5032 runbook](PINNACLE_CANDIDATE.md) is historical
source-pinned reproducibility documentation. The [v5032 release card](V5032_PINNACLE_RELEASE.md)
is the current physical baseline, and
[issue #17](https://github.com/mdt733/emu/issues/17) holds proposed
future diagnostics. The v5031 study remains historical evidence,
not a second active training system.

Before publication, the owner must review the exact exported
actor and checkpoint SHA256, 360-input sensor contract, measured
flat/slope/step commands, energy/falls, browser playback and
rollback plan. The only approved release format is checked by
`scripts/verify_showcase_release.py`, and the GitHub Pages
workflow remains explicitly protected. Never infer approval
from a successful training exit, higher reward or merged commit.

[Maintenance](MAINTENANCE.md) ·
[Historical research](history/README.md) ·
[Public playground](PLAYGROUND.md) ·
[Previous candidate handover](history/HANDOVER_V5031_COMPLETE.md).
