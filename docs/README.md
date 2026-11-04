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
| GPU contact and speed tradeoffs already measured | [v5046 solver study](V5046_OVERNIGHT_RESULTS_20261105.md) · [Solver and foot contact](V5046_FOOT_CONTACT_OPTIMIZATION.md) · [GPU audit](JAX_GPU_WORKLOAD_AUDIT.md) |
| Browser/WASM, park and protected publishing | [Playground](PLAYGROUND.md) · [Graduated park](VIEWER_GRADUATED_PARK.md) |
| Engineering and experiment clean-up policy | [Maintenance](MAINTENANCE.md) · [Issue #17](https://github.com/mdt733/emu/issues/17) |

## Current 50M → eight forks → 200M research

The v5048 campaign is source-pinned to
[its original experimental branch](https://github.com/mdt733/emu/tree/experiment/v5048-shared-parent-pinnacle).
Its shared 50M ancestor, eight matched 50M child continuations, physical
exams and final 100M winner extension are **research**, not evidence of
a successor until measured. The future-main release-prep branch
intentionally **omits** its temporary launcher so the final architecture
doesn't retain another experimental harness. To examine or resume that
run, use the **original experiment SHA and run directory**.

The frozen published 260M policy is unchanged while research runs.
Do not assume a CPU code check proves an 8GB CUDA training/exam pass,
and do not publish or merge a new actor without explicit release review.

## Historical files

Old v5045 140M / v5046 20M / v5048 eight-arm launchers, retired
workflows, earlier field geometries and research notebooks remain
recoverable at their **immutable original Git commits**. They do not
need to be in active `scripts/` or duplicated into a growing
`docs/history/` folder.

For the original detailed v5045 file census and pruned code, see
[the earlier cleanup commit](https://github.com/mdt733/emu/blob/feature/v5045-core-skeleton-audit/docs/CORE_SKELETON.md).
For experiments, consult the issue and source-pinned result cards,
not an old runbook presented as today's recommended launch.
