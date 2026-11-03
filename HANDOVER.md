# Current EMU engineering handover — short version

**3 November 2026.** The active large experiment is the pinned
**v5044 140M from-scratch 18-prism curriculum**, on
`feature/v5044-prism-140m-coldstart`, draft PR #33.
**Never update that source checkout mid-run.** New housekeeping lives
exclusively on **`feature/v5045-core-skeleton-audit`**.

For the current code journey, ownership and retirement plan:
**[Core skeleton](docs/CORE_SKELETON.md)**.

For exact current v5044 GPU run, first/continue/verify, restart and
checkpoint instructions:
**[v5044 140M runbook](docs/V5044_PRISM_140M_COLDSTART.md)**.

For source status, 18-prism physics tests and matched GPU
throughput:
**[v5043 prism physical proof](docs/V5043_MJX_PRISM_CARPET.md)**.

For the frozen public champion: **[release card](docs/V5032_PINNACLE_RELEASE.md)**.
The public `main` branch and published champion remain unchanged.

## What is true now

- One six-servo, 360-input, single-PPO actor; six teaching cohorts.
- v5044: one 2,048-world full 140M target, 20M foundation,
  dynamic six-specialist growth, streaming same-carpet Academy.
- 18 real 3×3 convex-prism colliders, deterministic X/Y slopes and
  signed step offsets. In v5045 the inert 9 old box bodies are **removed
  from prism scenes**; safe recycling uses the real prism XY footprint.
  Noncontact 2×2 hfields supply physical vertex/sensor samples.
- Matched 2048-world post-JIT PPO: **7,492 steps/s prisms**,
  **17,774 steps/s boxes**; prisms use **1.90 GB peak**.
- The viewer is local MuJoCo/WASM **replay**, not live MJX GPU state.
  Its illustrative prism/terrain view must not be described as
  identical to the actual training instances.
- v5045 removes superseded Python/JS terrain implementations and
  unused historical probes, archives old notes, batches prism mocap
  updates, and removes the inert legacy boxes from the prism scene.
  **No FPS improvement claimed until identical CUDA benchmarks.**

## Non-negotiables

Preserve the running v5044 SHA, training checkpoint and resume
semantics, saved physical/actor interfaces, physically independent
exams, backward compatibility with the released model and the
published public playground. Cleanup and optimisations belong to a
new branch with fresh tests and matched benchmarks.

**Full original handover (all previous decisions retained verbatim):**
[docs/history/HANDOVER_V5044_FULL_ARCHIVE.md](docs/history/HANDOVER_V5044_FULL_ARCHIVE.md).
