# EMU documentation index

**Start here:** [five-stage core skeleton](CORE_SKELETON.md)
(launch → physics → PPO → checkpoint/exam → viewer),
[project README](../README.md),
[current concise handover](../HANDOVER.md).

## Active experiment and engineering

| Guide | Status and purpose |
|---|---|
| [**v5044 140M true-prism cold start**](V5044_PRISM_140M_COLDSTART.md) | Source-pinned 2,048-world run, six specialists, live checkpoints, 10M physical evals and safe resume; independently verify run start/status |
| [**v5043 actual prism collision proof**](V5043_MJX_PRISM_CARPET.md) | 18 real convex meshes, true step walls, GPU memory and matched nine-box PPO throughput |
| [**Current core skeleton and retirement map**](CORE_SKELETON.md) | Keep / isolate / archive decisions, performance measurement programme and viewer parity boundary |
| [**v5041 140M terrain postmortem**](V5041_140M_TERRAIN_EXPOSURE_POSTMORTEM.md) | Why exposure probability must not multiply obstacle severity |
| [INSTALL](INSTALL.md) | WSL2/CUDA/JAX/MJX/Node prerequisites |
| [MAINTENANCE](MAINTENANCE.md) | Branches, checkpoints, code hygiene, source identity |
| [Architecture: expanded original overview](ARCHITECTURE.md) | Historical broader code map, retained for compatibility |

## Released champion / ongoing evaluation

| Guide | Purpose |
|---|---|
| [**Pinnacle 260M release**](V5032_PINNACLE_RELEASE.md) | **Released 360-input baseline**, exact policy identity and rollback |
| [260M physical baseline](PHYSICAL_BASELINE_260M.md) | Proven step crossing versus full controlled mission |
| [Physical quality](PHYSICAL_QUALITY.md) | Multi-axis movement, support, efficiency and survival measurements |
| [Capability dimensions](CAPABILITY_DIMENSIONS.md) | Capability envelopes / stability |
| [Release readiness](RELEASE_READINESS.md) | Owner approval and physical graduation gates |
| [Protected playground](PLAYGROUND.md) | Public viewer release protection |
| [Graduated terrain park](VIEWER_GRADUATED_PARK.md) | Browser-side real collision diagnostics; not an MJX certification |

## Historical / reference (do not treat as current launch commands)

- [**Complete v5044-era handover archived verbatim**](history/HANDOVER_V5044_FULL_ARCHIVE.md)
- [Older research history](history/README.md)
- [v5041 9-box trainer](V5041_GPU_CARPET_TRAINING.md) and
  [representative nine-box browser](V5041_TRAINING_CARPET_VIEWER.md).
  These **passed GPU engineering acceptance** and are a historical
  speed baseline. The v5044 prism runner is the present experiment.
- [v5042 browser-only prism geometry lab](V5042_PRISM_LAB_VIEWER.md).
- [v5040 separate three-slab Academy](V5040_REAL_MULTILIP_ACADEMY.md).
  Its original slabs are retired from v5044 *active collision physics*;
  the current Academy teaches on the shared carpet.
- [Old Pinnacle campaign plan](PINNACLE_CANDIDATE.md),
  [six-specialist design](UNIFIED_CAPABILITY_GROWTH.md),
  [traversability design](TRAVERSABILITY_REFUSAL.md),
  [terrain transition gap](TERRAIN_TRANSITION_GAP.md) and
  [surface lab](SURFACE_LAB.md) are reference material, not live
  commands.

Historical source SHA and original documents remain in Git.
**Do not resume checkpoints with a different source commit merely
because a later branch has a cleaner folder layout.**
