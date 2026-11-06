# EMU documentation — one current path

Start with [README](../README.md) for the robot,
[CORE_SKELETON](CORE_SKELETON.md) for which module owns each concept,
[INSTALL](INSTALL.md) for the single clean WSL checkout and
[HANDOVER](../HANDOVER.md) for the latest training/promotion state.

| Need | Maintained reference |
|---|---|
| How the one 360-input actor learns and how the eight-prism world works | [Live ownership / physics skeleton](CORE_SKELETON.md) |
| How to install, check CUDA, store runs and recover them | [WSL installation](INSTALL.md) |
| Which research knob and curriculum owns real steps / speed / balance | [Unified capability growth](UNIFIED_CAPABILITY_GROWTH.md) · [Capability dimensions](CAPABILITY_DIMENSIONS.md) |
| How to judge commanded movement and signed contacts | [Physical quality](PHYSICAL_QUALITY.md) · [Independent 260M baseline](PHYSICAL_BASELINE_260M.md) |
| Release approvals, rollback and promotion into main | [Release readiness](RELEASE_READINESS.md) · [260M Pinnacle release](V5032_PINNACLE_RELEASE.md) |
| GPU contact, speed and teaching experiments | [v5046 solver study](V5046_OVERNIGHT_RESULTS_20261105.md) · [v5048 550M shared-parent results](V5048_SHARED_PARENT_RESULTS_20261105.md) · [GPU audit](JAX_GPU_WORKLOAD_AUDIT.md) |
| Browser/WASM, park and protected publishing | [Playground](PLAYGROUND.md) · [Graduated park](VIEWER_GRADUATED_PARK.md) |
| Engineering and experiment clean-up policy | [Maintenance](MAINTENANCE.md) · [Issue #17](https://github.com/mdt733/emu/issues/17) |

## Completed 50M → eight forks → 200M research

The v5048 experiment has completed **550M unique steps** and **148
independent physical GPU panels** on
[the original frozen experimental source](https://github.com/mdt733/emu/tree/experiment/v5048-shared-parent-pinnacle).
The [measured v5048 result card](V5048_SHARED_PARENT_RESULTS_20261105.md)
documents the real physical step-height caveat, matched 100M
comparisons, the 200M temporal-speed policy and the unproven main
promotion gates. The future-main branch intentionally **omits its
temporary launcher** so the final architecture remains lean;
only the original experiment SHA should be used for reproduction.

The approved v5032 260M public policy remains unchanged.
A completed CUDA training tournament is not a released Pinnacle.

## Historical files

Old v5045 140M / v5046 20M / v5048 eight-arm launchers and the
completed v5051 fixed-wall/campaign and v5052 step-school wrappers, retired
workflows, earlier field geometries and research notebooks remain
recoverable at their **immutable original Git commits**. They do not
need to be in active `scripts/` or duplicated into a growing
`docs/history/` folder.

For the original detailed v5045 file census and pruned code, see
[the earlier cleanup commit](https://github.com/mdt733/emu/blob/feature/v5045-core-skeleton-audit/docs/CORE_SKELETON.md).
For experiments, consult the issue and source-pinned result cards,
not an old runbook presented as today's recommended launch.
