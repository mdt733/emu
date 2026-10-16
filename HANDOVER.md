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

## Minimalisation status and honest scope (17 October 2026)

**PR #6:** [refactor/emu-minimal](https://github.com/mdt733/emu/pull/6), branched from the working 120M `main` release. **Do not merge yet.** The independent GitHub Pages champion Git blob remains `067d6156f0f9de629eabb1a69cf1b0ceb86db24c`, byte-for-byte the same as `main`.

Tracked inventory: **132 → 56 files** (after empty media removal), `emu/` **42 → 15 Python files** (including `__init__.py`), `docs/` **30 → 2 Markdown files**, five maintained setup/check/showcase scripts, four test files, 20 browser source/dependency assets, one protected champion. All previous pass notes and tracked legacy result bundles can be found in Git history.

**Crucial honesty:** Python source volume only changed **609,546 → about 599,518 bytes**. Most gains are fewer module boundaries and discoverability, **NOT** a 70% reduction in logic. To shrink further, audit live usage and eliminate or simplify real training/exam features after careful CUDA acceptance, rather than coalescing into unreadable multi-thousand-line modules.

### Current code ownership (no import-only alias files)

- `robot.xml` + `environment.py`: physical universe, task commands, per-world domain variation, 220-dim actor observation.
- `rewards.py`: eight training payments, with servo power diagnostic rather than a hidden PPO penalty.
- `ppo.py`: PPO actor/value optimisation, JAX backend and performance timing.
- `training.py`: one learning loop, rolling measurements, explicit checkpoint calls.
- `curriculum.py`: regime governor, field teacher/evidence, held-out cloud probes, training health.
- `branch_lab.py`: readable 20/80/100/120M hypothesis recipe, single-run and tournament/expt executor.
- `evaluation.py`: frozen evaluation, fair matched branch screens, capability atlas, signed course/contact/gait and multi-branch batch plans.
- `results.py`: rollout scoring metrics, CSV/log artefacts, training stories and optional posture drift diagnostics.
- `viewer_server.py` + `viewer_src/`: actor export, local snapshot watching, field visualizer, browser physics/UI.
- `directional.py`: independent pure geometry/heading/contact oracles imported by PPO; kept separate to avoid circular evaluation dependencies.
- `contracts.py`, `training_config.py`, `checkpoint_io.py`, `cli.py`: explicit small boundaries for shared shapes/constants, user-controlled knobs, complete rewind/run lineage, public commands.

### Failures caught while truly moving source

- Removed-field import in `try_restore_recovery`: corrected after checkpoint round-trip CI failed. **Reason:** changing a class module does not automatically update every lazy import.
- Duplicate `summarize()` in moved posture/report logic silently overrode the campaign summary. Corrected by naming the optional function `summarize_posture()`; existing matched-scorecard roundtrip passed again.
- Source audit found old `capability_field` and `domain_randomization` doctor imports, stale atlas `paths.py` import, and a stale policy-export import in the real CUDA acceptance verification path. Corrected; added explicit CPU doctor, CLI planning and import-path smoke to GitHub Actions.
- Never declare the branch ready based on green tests alone. Exercise real WSL CUDA: 256→512 steps save/resume/verify, 20M cold start only if desired, export a tested actor and inspect the browser locally, including independent signed course/footfalls.

**No change to PPO reward weights, physical motor parameters, trained actor source files or schema-7 checkpoint payload is intended.** Source relocation still changes training source provenance; use a fresh run root, do not resume a historical lineage with this revision.

## Continuing in another chat

Inspect `git diff main...refactor/emu-minimal` and `git status`; don't trust the summary alone. Read `README.md`, `docs/INSTALL.md` and this file. Make actual, CI-verified reductions, not docs-only passes. Before any merge, check 16 Python + four browser tests, real WSL CUDA acceptance and browser model playback. Keep the 120M champion's file bytes unchanged.

