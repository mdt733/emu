# EMU successor — measurable graduation and clean-release gates

**Candidate status:** NOT APPROVED. Protected `main` still hosts the
approved 232-input v5016 Pinnacle browser policy.

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

Use **common physical tasks** to compare 232-input `main` and the
360-input challenger; each actor keeps its own legitimate sensor
configuration, but has the same body, ground, disturbances, command
requests, horizon and comparable scoring.

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
- [ ] Leave the approved `showcase/champion.emupolicy` unchanged
  until a candidate passes the independent common exam
- [ ] Explicit owner approval for the new champion and for merging to
  `main`; this checklist does not authorize either
- [ ] Record release source and checkpoint fingerprints, deterministic
  export, viewer smoke, and a clear rollback path to v5016 Pinnacle
- [ ] Update the public Pages deployment only with a reviewed,
  independently tested actor + frontend combination

## Current study and historical information

[v5028 considered ascent](V5028_CONSIDERED_STEPS.md) is a **research
treatment**, not a candidate release. The owner reported the foreground
launcher running on 29 Oct 2026; its results still require review.

[Research history](history/README.md) ·
[Maintenance rules](MAINTENANCE.md) ·
[Current handover](../HANDOVER.md) ·
[Public playground](PLAYGROUND.md)

The original v5024-era release checklist is
[archived](history/V5024_RELEASE_READINESS.md).
