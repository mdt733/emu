# EMU release-candidate cleanup and promotion gates

**Date:** 27 October 2026  
**Training source (frozen):** `feature/v5024-terrain-control` at
`d26ae0c2c4097d7c7124759197a94504abc44995`.  
**Engineering cleanup:** `feature/v5024-release-polish`, forked from that commit.  
**Protected:** `main`, public Pages/showcase, approved 260M Pinnacle,
archived v5023 experiment checkpoints, and the running v5024 run directories.

## Why two branches?

v5024's 3 × 60M campaign pins its exact source revision. Even a harmless
documentation commit changes the SHA. **Do not pull the polish branch into
the running training checkout**, and do not resume its checkpoints under
polish code. Review and integrate cleanup only after collecting results,
and start a new source-pinned run for any changes that alter learning.

## Cleanup policy

1. Preserve actor contracts (legacy 220/232 and unified 360), training
   lineage validation, physical simulation, failure recovery, policy export
   and the protected browser champion until independently replaced.
2. Keep tests that protect a real behaviour or boundary, regardless of
   their original v5015/v5016/v5023 names. Remove only redundant or
   unreachable cases **after** mapping coverage and replacing their checks.
3. Keep the core read-first path: README → INSTALL → V5024_RUN_PLAN →
   UNIFIED_CAPABILITY_GROWTH → CAPABILITY_DIMENSIONS.
4. Clearly label completed experiment scripts/docs as historical. On the
   frozen *training* branch, the v5024 runner still imports the older
   tournament helpers and must remain untouched. On the **polish branch**,
   those helpers have been consolidated into its v5024 launcher and the
   v5023 tournament entrypoint removed. That is a changed source revision:
   never use it to resume the frozen 180M experiment.
5. Run small tests before GPU and hold behavioural changes until frozen
   examiner results justify them. Reduce accidental complexity, not just
   source line count.

## Evidence required before promoting to main

- [ ] Three 60M arms finish or failures are diagnosed without hiding the
      missing evidence. Save `overnight_results.zip`, source SHA and configs.
- [ ] Frozen generalist **and** 1/2/3/5° directional terrain matrix on the
      candidates; report supported motion separately from precise tracking.
- [ ] Check uphill commanded velocity, lateral downhill drift, yaw,
      step-up AND down support, falls and work per useful distance.
- [ ] Repeat promising arms with further seeds before claiming broad gains.
- [ ] Independently verify cold-start, 2048-world CUDA update,
      same-source resume/extension, export, and local-browser playback.
- [ ] Current candidate CI green (Python/MJX CPU/browser/viewer); avoid
      claiming that CPU CI verifies CUDA capacity.
- [ ] Review diff versus v5024; ensure no stale branch or file paths in
      public entry docs; identify actual unreferenced code/tests for deletion.
- [ ] Reviewed PR, explicit owner approval for **main merge**.
- [ ] Separate explicit release approval for any public showcase update.

## Public release readiness — 360-input policy plumbing

The previous release gate accepted only 232 observations. The polish
branch now validates **360-input schema-9 array-only actor policies** in
the stdlib-only public release tool, including the 4×4 physical array
extrinsics, sensor settings, exact field order and independently matched
sensor SHA256. The already approved 232-input Pinnacle approval remains
valid and unchanged. **No public deployment is triggered by this work.**
The old untrained zero-action fixture, fallback and redundant
`--require-champion` switch have been deleted: trained policy and
valid provenance are unconditional build requirements.

The current `showcase/deployment_request.json` remains pinned to the
approved v5017 pose-lattice release. Future 360 publication still
requires a separate owner-reviewed release request, policy artifact,
checkpoint evidence, and GitHub Pages deployment.

## Legacy retirement — explicit exit plan

The old 220/232-input policy family, sim2real trial profiles, legacy-only
examiners and their tests are **temporary scaffolding**, not a permanent
second product. Retire them by audited deletion after:

1. A 360-input champion and its saved observation/normalizer contract are
   selected using held-out real-contact exams and replicated seeds.
2. The local 360-input viewer and exported-policy replay are verified against
   the Python actor, including terrain sensing and all controls.
3. An owner-approved public release actually switches the showcase to the
   new champion, with the old one preserved as an archived release/blob.
4. Call-site and checkpoint format inventory shows the old schema is no
   longer required for the supported trainer, exporter, CI or viewer.
5. Remove obsolete compatibility branches, old CLI flags and their now-
   irrelevant tests together; retain small backward-compatible readers
   only when actual user data needs migration.

The desired final `main` has **one active actor contract, one learner and
one current release path**, not two maintained generations. Do not delete
legacy runtime dependencies *before* the public release migrates.

**This cleanup is not a new champion.** Keep the main policy and public
showcase until matched, physically grounded evaluation supports promotion.

## Audit decisions (after standalone launch consolidation)

Source-level cleanup is underway, but none of these changes is claimed as
GPU/behavioural validation. Do not choose a champion before the matched
frozen comparisons. The future deletion of old-schema data paths depends
on successful public migration, not on a desire for a smaller diff.

| Item | Decision | Reason |
| --- | --- | --- |
| `scripts/overnight_teaching.py` | **Removed on polish** | v5024 campaign now owns its needed launch/report helpers; the v5023 tournament remains in Git history |
| `emu/directional.py` / `terrain_spike.py` | Retain | Frozen exam, legacy courses and CI physical probes |
| Legacy 220/232 contracts and corresponding tests | Retain | Public Pinnacle and checkpoint-export compatibility |
| `scripts/smoke_training_lifecycle.py` | Retain | Checkpoint/resume/export acceptance |
| `scripts/install_viewer_wsl.sh` | Retain | CPU-only standalone viewer installation |
| v5023 tournament documents | Removed from active docs | Complete historical versions remain in Git history; current run has its own plan |
| Surface-lab instructions | Consolidate | v5024 already includes the real MuJoCo viewer; no third checkout needed |
| Source-check workflow naming and launch plans | Refresh | Historical version labels and stale plan options confuse the current release |
