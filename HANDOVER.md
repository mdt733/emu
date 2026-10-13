# EMU — live handover (14 October 2026)

## Project state

- **GitHub:** [PR #1](https://github.com/mdt733/emu/pull/1) was correctly merged into `main` at commit `b0f31d8`. The protected 60M policy is still `showcase/champion.emupolicy`; GitHub Pages deployment is **manual** and requires its pinned Git hash. The new viewer source is in main, but a manual Pages deploy is a separate decision.
- **Current maintenance work:** [draft PR #2](https://github.com/mdt733/emu/pull/2), branch `refactor/ladder-human-first`, is deliberately **not merged**. It simplifies training orchestration; it must not be pulled into a running experiment checkout.
- **GPU:** Operator verified JAX CUDA/MJX and an actual v5015 PPO optimiser-update smoke on WSL (Python 3.13). The first 72M-step *aggregate* lab subsequently **failed during training**. Error/logs have not been supplied; root cause is unknown. Do not make claims about its performance, or alter/restart checkpoints before inspecting logs.
- **Older long run:** v5014 400M lineage was intentionally stopped after gradual gait crouching around **100–200M absolute steps per policy**. A 46M final ladder branch cannot test that long-horizon failure.

## Design contract

EMU is a dynamically unstable six-servo biped, trained for walking, steering and recovery using JAX/MJX/PPO—not a fixed animation or static pose controller. Deployable actor: **55 features × 4 frames = 220 observations, six motor outputs**. Policy export is actor-only; schema-7 `.pkl` checkpoints preserve actor/critic, optimiser, normaliser, curriculum, health and capability field. Resuming a checkpoint does not reproduce an identical simulator episode.

Physical variants include gravity-equivalent slopes and robot/hardware domain perturbations. The browser's actual MuJoCo ramp geometry is separate; its success has not been established as learned locomotion. Do not confuse rigid chassis pitch with the line from foot midpoint to chassis (robot lean). Hold-out evaluations are read-only.

## Clean architecture (draft PR #2)

```
emu/ladder_recipe.py   human-edited hypothesis & absolute step list
       |
emu/branch_lab.py      fork/checkpoint, matched screens, eligibility, promotion
       |
emu/phase_config.py    foundation & mature training presets, strict compatibility
emu/run_support.py     validated checkpoint I/O, Git provenance, subprocess launch
       |
emu/training_config.py typed knobs and validated per-arm overrides
       |
emu/training.py        stable GPU learning orchestrator
   ├─ ppo.py           optimiser/actor/critic
   ├─ environment.py   MuJoCo physics/reward/observations
   ├─ curriculum.py    difficulty governor
   └─ capability_field.py, health.py, rollout_metrics.py, evaluation.py
```

Current recipe: **20M shared foundation → 4 × 26M arms (baseline, shadow, equivalent slope, 15° pitch) → matched fixed examination → 2 × 34M → 1 × 46M**. About 72M environment steps total. Each future arm can override named `TrainingConfig` fields in `ladder_recipe.py`; run identity fields cannot be overridden. The current phase schedule is not automatically a 400M marathon.

The obsolete `forge` v5014/v5015 single-lineage runners, `tournament` v5013 experiment, and old `train/compare` CLI were **removed from the draft source tree**; their code remains in Git history at `b0f31d8`, with historical evidence in `results/v5013_tournament/`. Reusable machinery was moved to neutral modules. The `lab5015` spelling and old shell launcher survive as compatibility aliases, not recommended commands.

## Commands for a NEW experiment (only after reviewing the failed run)

```bash
cd ~/robot_lab/emu
unset PYTHONPATH
source .venv/bin/activate
emu ladder --stage plan
emu doctor --v5015-smoke --domain-smoke
bash scripts/run_ladder.sh
```

The launcher clears stale `PYTHONPATH`, checks source location, verifies real GPU/MJX/PPO execution and only then starts the ladder. It stores runs outside Git at `~/robot_lab/emu_runs/v5015_lab1`; **a new experiment with a different Git commit/recipe must use a new run root**. Do not simply reuse the failed run's root with changed training settings.

To watch completed validated checkpoints from the current ladder, use `emu viewer --live --root ~/robot_lab/emu_runs/v5015_lab1 --no-open` in a separate terminal. `docs/tournament.md` details the experiment and scoring; `docs/PLAYGROUND.md` covers manual public deployment.

## Safety and next work

1. Obtain the failed v5015 console error and read its `console.log`, `lab_context.json`, `training_config.json` and latest checkpoint details **before** diagnosing or resuming. No root cause is confirmed yet.
2. Keep draft PR #2 independent; review CI and refactor equivalence. No merge is authorised yet.
3. Do not re-export/replace the 60M public `showcase/champion.emupolicy` from an unproven training run. Static Pages publishing is manual, separately guarded.
4. After recovering early training, investigate the **100–200M** crouching issue with timed matched held-out evaluation and height-command tracking. Scientific lessons live in `docs/history.md`; old audit and results remain historical records.

For current code ownership see [architecture](docs/architecture.md), [training](docs/training.md), [experiment ladder](docs/tournament.md), [installation](docs/INSTALL.md), [history](docs/history.md), and [document map](docs/README.md).
