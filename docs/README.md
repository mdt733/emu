# EMU documentation map

This directory separates **current operating guidance**, the **unified learning
contract**, and **reproducibility records for completed studies**. Historical
recipes remain available but should not be mistaken for the current launch.

## Current / maintained

| Document | Purpose |
| --- | --- |
| [INSTALL.md](INSTALL.md) | Fresh WSL installation, v5024 campaign, smoke checks, safe resume |
| [V5024_RUN_PLAN.md](V5024_RUN_PLAN.md) | Three-arm cold-start experiment, held-out terrain matrix and evaluation |
| [UNIFIED_CAPABILITY_GROWTH.md](UNIFIED_CAPABILITY_GROWTH.md) | One actor, six teaching emphases, evidence-based target adjustments |
| [CAPABILITY_DIMENSIONS.md](CAPABILITY_DIMENSIONS.md) | Actual versus proposed independent physical capability targets |
| [SURFACE_LAB.md](SURFACE_LAB.md) | Real browser MuJoCo adjustable incline/step viewer |
| [PLAYGROUND.md](PLAYGROUND.md) | Protected public Pinnacle viewer and release safeguards |
| [RELEASE_READINESS.md](RELEASE_READINESS.md) | Candidate branch, preservation rules and promotion gates |
| [../HANDOVER.md](../HANDOVER.md) | Active experiment status and engineering handover |

## Historical / comparison material

- [OVERNIGHT_TEACHING.md](OVERNIGHT_TEACHING.md) — v5023 260M teaching
  tournament (read-only history). Its old Python launcher was removed from
  the polish branch and remains recoverable from the frozen v5023 Git ref.
- [FOUNDATION_EXPERIMENTS.md](FOUNDATION_EXPERIMENTS.md) — early alternative
  nursery trials and prior setup instructions.
- [TEACHING_ALLOCATION_EXPERIMENTS.md](TEACHING_ALLOCATION_EXPERIMENTS.md) —
  rationale and planned comparisons preceding v5024.

These are deliberately retained until the three-arm v5024 run and matched
archived-policy exams have been reviewed. Their earlier branch names are
historical, not defaults. Git history retains previously deleted experiment
scripts; recreating old pipelines is not a release requirement.

## Preservation rule

The 232-observation public Pinnacle and its viewer remain supported while
the 360-observation unified policy is developed. A test or code path with an
old version string is **not necessarily dead**. Remove it only after tracing
all live CLI, viewer, replay, checkpoint and CI references.
