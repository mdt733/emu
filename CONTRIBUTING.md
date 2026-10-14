# Contributing to EMU

EMU is experimental robotics software. A useful change is **small, reproducible and testable**, with an explicit explanation of what behaviour it changes.

## Before editing

Read [architecture](docs/architecture.md), [training](docs/training.md) and [project status](docs/STATUS.md). The actor contract is 55 observation fields × four frames → six motor targets. Changing field order, action meaning, robot XML or checkpoint schema is not a routine refactor.

Use a **separate branch** and run root for experiments. Do not pull source changes into an active training checkout or reuse its checkpoints under a changed training configuration. Never commit .venv, node_modules, GPU run logs or `.pkl` training checkpoints.

## Everyday checks

```bash
python -m compileall -q emu tests
python -m pytest -q
npm ci --prefix viewer --no-audit --no-fund
npm test --prefix viewer
npm run build --prefix viewer
git diff --check
```

The Python suite includes a **NumPy ↔ JavaScript exported-policy inference parity** check; Node 24 must be available. CI also checks the static showcase packaging and protected policy asset.

For physics, PPO, JAX, observation or checkpoint changes, additionally run:

```bash
emu doctor --v5015-smoke --domain-smoke
```

That is a **small GPU integration smoke**, not evidence of locomotion quality. A training-loop change also needs a bounded new run and resume comparison before a major experimental campaign.

## Change boundaries

- **New hypothesis:** edit `emu/ladder_recipe.py` using existing, typed `TrainingConfig` fields. Document the baseline and one variable changed.
- **Training algorithm:** keep updates inside `emu/ppo.py`; `training.py` orchestrates state transitions, evaluations and checkpoint publication.
- **Evaluation:** score the *same actor and observation normaliser* that you checkpoint. Rollout telemetry describes a different (rollout-generating) actor and must remain distinguishable.
- **Persistence:** publish checkpoints atomically; never unlink the only validated resume point before a replacement is ready. A resume restores learning state but not simulator episodes.
- **Browser:** edit `viewer/` and build the package mirror with `emu viewer-build`; never edit the mirror as the source of truth.
- **Research claims:** cite the run source commit, seed, checkpoint step, evaluator cohorts and results. Do not infer robustness from training reward or the browser alone.

## Pull-request checklist

Explain *why* the change is needed, *which invariant it protects*, test commands/results, and whether it affects learned weights, checkpoint compatibility, viewer assets or the public champion. Add a regression test for a demonstrated failure, and remove obsolete duplicate assertions when appropriate. Distinguish CPU-only contract checks from actual GPU validation.

**Do not merge an unverified change solely because CI is green.** See [validation](docs/validation.md) for the separate release gates.
