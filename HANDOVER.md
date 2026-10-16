# EMU — living handover

**Current stable release:** `main` at merge commit `252f6102dd813a451daad29fecf2fbebd586da6b`, containing the refactored trainer, single browser viewer, and real **Pitch15 120,061,952-step** `showcase/champion.emupolicy`. The public GitHub Pages site deploys only after manually dispatching the Pages workflow; do not confuse a code merge with a published site.

**Current refactor:** `refactor/emu-minimal`. The goal is *genuine reduction*, not making dozens of ownership-labelled files. Keep the working original available in Git history; do not rewrite `main` or active WSL run roots before CI and local acceptance.

## The essential ideas

Robot and universe → rewards → PPO/training → curriculum / capability evidence → branch experiment design → independent results → frozen browser viewer. Every retained file needs an understandable purpose. No one-off pass documents, old CSV result bundles or import-only compatibility stubs.

## Proven experiment and lessons

User's study from source commit `a5996b8cda2c394c7c65e3e938282a3808ded121`, seed 733, 2048 MJX worlds: shared 20M intro, four independent 80M arms, two 100M finalists and Pitch15 120M winner, total 320M aggregate environment steps.

Pitch15 matched frozen-screen scores: .683903 at 80M; .712062 at 100M; .723497 at 120M (91.992% hard survival, 86.328% growth survival). Shadow second at earlier rounds; slope practice did not dominate the common gravity-equivalent slope exam. **These rankings do not prove actual hardware walking, and one seed isn't a universal result.** Late Pitch15 servo over-rated-torque sample fraction ~74.12%, aggregate simulated shaft |power| ~12.81 W. Real hardware motor limits/efficiency and signed footfall progress remain unresolved.

The selected raw PKL checkpoint (local-only) SHA-256:
`ff115931b085792eb36f729b3d38248a58a9258cc8c3cd0e5601db5f5413be28`.
Trained browser file tracked at `showcase/champion.emupolicy`, Git blob SHA
`067d6156f0f9de629eabb1a69cf1b0ceb86db24c`, policy actor SHA-256
`c836f0f179445c309aea3b9c87eb6b63c75a7cd2666801e262087b7663d5849c`.
Preserve exact 220-input/six-action contract and checkpoint schema 7.

## Operational safety

- Keep large checkpoint roots outside Git; preserve source commit and normaliser/optimiser lineage. Never open an active trainer's branch for destructive experiments.
- Changes to MJX environment, PPO, observation ordering, checkpoint storage and exporter need real CUDA save/resume/export acceptance, not just unit tests.
- The static browser uses `emu/viewer_src/`; local live watcher uses validated read-only checkpoints. The GitHub Pages champion is a frozen exported brain, never the original `.pkl`.
- Pages deployment is intentionally manual and pins the exact champion Git blob SHA.

## Current minimalisation work

- Prune retired docs and checked-in historical result copies (preserved in Git history).
- Merge related helper modules into the *existing owner* rather than retain alias shims. Run all Python/JS tests and a source-import audit after each useful cluster.
- Target ~7 conceptual source layers; a few additional files for configuration, checkpoint contracts and browser boundary are acceptable when they avoid giant circular modules.
- Finish with one readable front page, this handover, `docs/INSTALL.md`, and `docs/PLAYGROUND.md`; the mandatory third-party notice is not optional.

## Continuing in another chat

Inspect `git diff main...refactor/emu-minimal` and `git status`; don't trust the summary alone. Read `README.md`, `docs/INSTALL.md` and this file. Make actual, CI-verified reductions, not docs-only passes. Before any merge, check 16 Python + four browser tests, real WSL CUDA acceptance and browser model playback. Keep the 120M champion's file bytes unchanged.

