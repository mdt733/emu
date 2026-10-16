# tinyEmu: structural cleanup checkpoint (15 October 2026)

**Branch:** `refactor/tiny-emu` | **Draft review:** [PR #3](https://github.com/mdt733/emu/pull/3) | **Do not merge or deploy yet.**

## Goal, not a file-count competition

Working `main` at `40c5d0ce543ebbbc3690591695afddc18def8232` is the verified reference. The supplied Pass 9 ZIP is an unverified source of simplification ideas. This branch is a third design, derived from working main; it is **not** a Pass 9 migration. It is intended to make the relationships among observed physics, reward, PPO, curriculum, independent evaluation and experiment selection explicit.

## Changes actually made

| Area | Changed files | Retained / removed |
|---|---|---|
| PPO scoring responsibility | `emu/scoring.py`, `emu/environment.py`, `emu/rollout_metrics.py` | **Retained:** all eight original positive payments, v5014/v5015 posture profiles, anti-freezing gate, early-fall penalty. **Removed:** `environment.EmuMJXEnv._reward` and reward-only constants from simulation code. `scoring.training_reward` and `apply_early_fall_penalty` now own these operations. |
| Scoring contract tests | `tests/test_reward_ledger.py`, `tests/test_core_contracts.py` | Independent original payment expression, terminal penalty, JAX JIT compilation, two posture profiles and reducer checked. **Not proof of full MJX parity.** |
| Single experiment interface | `emu/single_run.py`, `emu/cli.py`, `tests/test_single_run.py` | **Added** one CLI wrapper around existing validated trainer, with `--plan`, clean-source/nonoverwrite guard, same-lineage resume; **no additional PPO engine**. |
| Viewer source ownership | `emu/viewer_server.py`, `scripts/setup_wsl.sh`, `scripts/prepare_showcase.py`, `scripts/verify_showcase_dist.py`, `.github/workflows/emu-v5015-checks.yml`, `.github/workflows/emu-playground.yml`, browser-related tests and docs | **Retained:** identical policy fixture, XML/browser simulation, JS runner, npm lockfile, packaged standalone viewer, local read-only live watcher, static champion and manually dispatched Pages. **Removed:** 26 duplicate `viewer/` tracked files (815,601 bytes), the mirror synchronization/copy-on-build path, and obsolete mirror test. The sole source is `emu/viewer_src/`. |
| Understandability | `docs/WHERE_THINGS_LIVE.md`, Pass 1 comparison, Pass 2 status and this report | Physics measurements, PPO scalar reward, telemetry, curriculum and branch selection connected in a short guide. |

## Actual automated evidence

- [CI run 76](https://github.com/mdt733/emu/actions/runs/38683350011): reward extraction with JAX compilation — **159 Python passed plus five subtests; 17 JS passed**, static build and asset checks passed.
- [CI run 87](https://github.com/mdt733/emu/actions/runs/38683817195): package-owned single viewer source after removal of duplicate tree — **159 Python passed plus five subtests; 17 JS passed**, Vite build and showcase asset verification passed.
- Subsequent CI adds a new check: **a browser build may not change tracked source or create unignored files**. See the latest PR checks for the final SHA's result.
- Original `main`'s separate WSL RTX 4060 Ti 0→256→512 acceptance was successful **before these changes**. Do not copy that PASS label onto this new source commit.

## Practical source-to-behaviour trace

```
robot.xml -------------------> MuJoCo/MJX physics
                                   |
environment.py::_metrics ----------+-- measured values / honest proxies
                                   |
scoring.py::training_reward <------+-- rewards only
scoring.py::apply_early_fall_penalty --- terminal cost
                  |
environment.py::_step_impl -> StepOut.reward ----> ppo.py::rollout_once
                      |                                 |
                      +-> named reward_* telemetry     +-> compute_gae -> PPO update
                               |
                     rollout_metrics.py -> training.py -> metrics.csv

held-out evaluations: evaluation.py / screening.py (not PPO training reward)
curriculum: curriculum.py (on-policy rollouts, not exam feedback)
experiment promotion: branch_lab.py::score_row / promotion_choices
```

**Important limitations:** foot-height `support` is not a measured contact force. Foot-lift `step_evidence` is not a verified completed stride. Action smoothness is not electrical or mechanical power. No spring work, thermal strain or cost-of-transport objective is yet trained or independently scored. The promotion score still emphasises survival.

## Remaining work / gate before merge

1. Verify current branch in **its own** clean WSL worktree, without changing long-run roots or `main`. Run `emu doctor --v5015-smoke --domain-smoke`.
2. Run the accepted `scripts/smoke_training_lifecycle.py` three stages (`first`, `continue`, `verify`) in a **new scratch folder outside Git**, demonstrating 0→256→512, held-out exams, completed resume no-op, source identity, observation normalizer and actor export.
3. Independently run a bounded new `emu train` experiment with a *different* scratch root and confirm safe restart/plan and absence of dirty-after-build.
4. Interactively load the protected 60M actor and bounded exported actor in the browser, including WASM loading, control axes and zero-weight fixture. Browser appearance is not a scientific evaluation.
5. **Then** make real gait-cycle/displacement and servo-work telemetry read-only and calibrate them against known-bad controllers; keep PPO reward/promotions unchanged until thresholds can be justified.
6. Finish removing obsolete governance/teacher complexity only after equal-budget experiments demonstrate the new small training/curriculum/ladder design. Do not interpret Pass 9 checkpoint schema 9 as resumable from main's schema 7.

### Protected artifacts and invariants

- `showcase/champion.emupolicy` must retain Git blob `4743d8d3cd4706edce04809fe1b61df014e57d5e`.
- `emu/robot.xml` remains unchanged from accepted main.
- Actor contract remains 55 sensor/command entries × four frames = **220** observations and six actions.
- GitHub Pages stays **manual workflow_dispatch only**; no implicit publishing on merge.
- No long-running experiment roots were opened, copied or modified by this work.
