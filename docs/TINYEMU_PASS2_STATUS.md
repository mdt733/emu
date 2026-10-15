# tinyEmu refactor — Pass 2 implementation/validation checkpoint

**Date:** 15 October 2026. **Status:** in-progress; **not merge-ready**.
**Review:** [Draft PR #3](https://github.com/mdt733/emu/pull/3).
**Source base:** accepted main `40c5d0ce543ebbbc3690591695afddc18def8232`.

## What actually changed

| Path | Concrete change |
|---|---|
| `docs/TINYEMU_PASS1_COMPARISON.md` | Candidate ZIP versus current main, every runtime Python file and missing capability |
| `emu/scoring.py` | Pure single-owner table of the eight existing positive PPO reward payments |
| `emu/environment.py` | Reuses that table; exposes the eight weighted payments and the terminal fall payment in the rollout metric packet |
| `emu/rollout_metrics.py` | Copies named weighted payments into report/CSV schema, leaving existing reward and curriculum metrics intact |
| `tests/test_reward_ledger.py` | Original NumPy arithmetic parity and real `EmuMJXEnv._reward` to reducer tests for both v5014/v5015 posture profiles |
| `emu/single_run.py` | Safe one-run config, source/run-directory checks and delegation to existing `run_support.run_training` |
| `emu/cli.py` | Exposes `emu train`, `--plan`, `--resume`; retains existing ladder and every previous command |
| `tests/test_single_run.py` | Pure planning, invalid knobs, source/provenance mismatch, protect existing data, partial-resume and runner delegation checks |
| `docs/WHERE_THINGS_LIVE.md` | Direct reward → PPO → CSV → curriculum → exam → branch selection reference and one-run usage |

## What was retained versus deleted

**No existing runtime modules were deleted or renamed.** Current accepted schema-7 checkpoint handling, 20M/fork ladder, capability field/cloud teaching, domain randomisation, model XML, 220-input/six-action actor and viewer source remain intact. The Pass 9 candidate is **not** used as a replacement or permitted to overwrite accepted run history. The public champion and Pages manual-dispatch workflow are not touched. This pass intentionally adds a small, understandable facade *before* riskier internal consolidation.

The command uses two **existing** physics/curriculum profiles: `foundation` (shadow field) and `mature` (graph field). It does not claim that Pass 9 field-free learning matches main. The new CLI uses `v5015` posture commands in both profiles, consistent with the accepted smoke acceptance contract.

## Actually executed GitHub evidence

- [CI run 67](https://github.com/mdt733/emu/actions/runs/38681970773): **153 Python passed + 5 subtests**, **17 browser passed**, source compiled, viewer production build and protected-policy/WASM verification passed.
- [CI run 68](https://github.com/mdt733/emu/actions/runs/38682106135): after adding real-`_reward` regression, **154 Python passed + 5 subtests**, **17 browser passed**. Python and browser jobs finished successful.
- Tests executed in ephemeral GitHub Actions Ubuntu environments; these results prove CPU-level contract and browser static-package compatibility, **not** physical locomotion or the user's WSL GPU lifecycle.
- Earlier candidate ZIP checks (105 Python pass, 2 skip; 20 JS pass) are evidence for the ZIP, not the main-derived branch.

## Required before Pass 2 is considered fully accepted

- Run on the user's actual clean RTX 4060 Ti WSL checkout of this **specific branch commit**: `emu doctor --v5015-smoke --domain-smoke` and the existing `scripts/smoke_training_lifecycle.py` 0→256→512 acceptance; verify held-out exam, no-op resume, normaliser count, export and loaded browser.
- Separately exercise the new `emu train` real-command lifecycle from a unique scratch root outside Git (the unit tests mock the trainer subprocess).
- Confirm latest main source, protected champion Git blob, and no dirty-after-build regression before any merge.
- Decide whether to remove the tracked `viewer/` and `emu/viewer_src/` mirror; first prove packaged assets and Pages build with one canonical source. This has **not** happened yet.
- No schema-7→9 migration. Source and checkpoints from a different commit must use a new run root or dedicated tested migrator.

## Next small steps, in dependency order

1. **Pass 3:** complete real WSL GPU acceptance and read-only viewer comparison on this branch. Do not start a costly ladder.
2. **Pass 4:** move reward factor construction out of `environment.py` into a pure, named physical-measurement → scoring boundary, with original v5014/v5015 numerical comparisons and *actual* JAX/MJX smoke before removal of old code.
3. **Pass 5:** create interrupted-episode / stride-completion read-only scorecards with bad-controller control cases before winner gates.
4. **Pass 6:** derive per-servo torque, velocity, work, heating proxy, spring energy as *read-only* diagnostics before any efficiency shaping.
5. **Pass 7–10:** reduce curriculum governance, compose a simple ladder, run independent matched experiments, finish viewer single-source and docs; decide explicitly what to retire.

**Merge gate:** requires explicit human review and GPU evidence; never auto-merge, never publish Pages, and never mutate existing long experiment run roots.
