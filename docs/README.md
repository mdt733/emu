# EMU documentation

Start at the [project README](../README.md) for scope,
[HANDOVER](../HANDOVER.md) for what is currently frozen,
or the [260M release card](V5032_PINNACLE_RELEASE.md) for the current actor.
The old [training runbook](PINNACLE_CANDIDATE.md) remains reproducibility history.

## Current, maintained

| Document | Purpose |
|---|---|
| [**PINNACLE 260M RELEASE**](V5032_PINNACLE_RELEASE.md) | **Current trained 360-input baseline**, physical results, limitations, SHA identities and rollback |
| [POST-PINNACLE RESEARCH](https://github.com/mdt733/emu/issues/17) | Measurements and possible experiments to revisit later; no active science run |
| [**INSTALL**](INSTALL.md) | New Windows/WSL2 GPU: Python 3.13, JAX CUDA, Node, pytest and 2048-world smoke |
| [**ARCHITECTURE**](ARCHITECTURE.md) | One-page system diagram, actor/teacher/exam separation and exact file ownership |
| [HANDOVER](../HANDOVER.md) | Current source/policy identity, honest v5032 physical baseline and immediate follow-up |
| [RELEASE READINESS](RELEASE_READINESS.md) | Owner-approval checkpoints and physical graduation criteria |
| [**PHYSICAL QUALITY**](PHYSICAL_QUALITY.md) | What genuinely defines a good robot; signed capability envelopes, path accuracy and no fake scalar |
| [UNIFIED CAPABILITY GROWTH](UNIFIED_CAPABILITY_GROWTH.md) | Six focus curricula, one actor and physically earned frontiers |
| [CAPABILITY DIMENSIONS](CAPABILITY_DIMENSIONS.md) | Detailed motion, resilience, terrain and J/m metrics |
| [VIEWER GRADUATED PARK](VIEWER_GRADUATED_PARK.md) | Eight real-collision viewer challenges and three difficulty levels |
| [PLAYGROUND](PLAYGROUND.md) | Protected public demo, source-safe viewer and GitHub Pages |
| [MAINTENANCE](MAINTENANCE.md) | Branch, checkpoint, code hygiene and clear ownership |
| [TRAVERSABILITY REFUSAL](TRAVERSABILITY_REFUSAL.md) | **Deferred design only**: ToF-informed slowdown, retreat and replan |

## Completed experiments — not active launch commands

- [v5028 considered step-ups](V5028_CONSIDERED_STEPS.md) and
  [speed-restoration audit](V5028_SPEED_FRONTIER_AUDIT.md)
- [v5030 temporal reward study](V5030_REWARD_EXPERIMENTS.md)
- [v5031 verified support-transfer A/B](V5031_VERIFIED_ASCENT.md)
- [Older research history](history/README.md)
- [Full v5031 handover snapshot](history/HANDOVER_V5031_COMPLETE.md)

Their original **source SHA** remains the reproducible reference.
Historical source code was removed from the *active release-candidate
tree*, not destroyed in Git. Do not launch old experiments from
today's branch or transplant incompatible training checkpoints.

## Deferred until after the baseline is settled

[Organic terrain transitions](TERRAIN_TRANSITION_GAP.md) and
[manual surface lab](SURFACE_LAB.md) remain reference material.
The 260M candidate is complete and promoted as the new baseline.
We will revisit supported 3cm-plus step-ups and organic terrain only
after targeted diagnostics from [issue #17](https://github.com/mdt733/emu/issues/17).

**Current baseline:** `main` now carries the 360-input 260M
Pinnacle and approved portable actor. The old 232-input champion
remains recoverable from commit `2ec9e9db0a816c658131311129ef89ec7a595a12`.
Any *future* replacement again requires measured evidence and approval.
