# EMU handover — current VERSION: v5012

## Purpose

EMU is a MuJoCo/MJX PPO trainer for a dynamically unstable six-actuator biped. The project goal is a trainer that repeatedly builds a capable policy from random weights and maintains a trustworthy live model of the policy's command, maneuver and recovery capability.

## Current release

v5012 is a measurement/trustworthiness release following the v5011 cold-start result.

Key changes:

- capability-field schema 3;
- one-second command-node events use mean tracking error across the gait window;
- fast post-settle falls count as node failures;
- one-frame command jumps are visible to maneuver-edge and acceleration-frontier instrumentation;
- node calibration uses a held-out node-window exam rather than recovery rate;
- missing field metrics and incompatible field state fail loudly;
- resume is strict and cannot silently restart from zero;
- checkpoint discovery prefers `final.pkl`, then `best_eval.pkl`, then latest;
- view/evaluate are seed-aware;
- `emu compare` runs matched shadow-vs-graph branches for seeds 733 and 1742 by default;
- forced final trainer evaluation includes the longer hard suites.

## Main commands

```bash
emu doctor
emu train
emu compare
emu view --seed 733
emu evaluate --seed 733
```

`emu train` remains the recommended single graph run. `emu compare` is the scientific v5012 experiment.

## Important invariants

- actor observation remains 220 values (55 × 4 history);
- held-out evaluations never update the live capability field;
- runtime partial node/edge/recovery events are not serialized across a process restart;
- resume restores learned network, optimizer, governor, observation normalizer and capability field, but starts fresh MJX episode buffers;
- old field schemas are viewable but not resumable as schema 3.

## Repository layout

See the top-level `README.md` for the human-facing overview and `docs/v5012-design.md` for the experiment rationale. Historical design notes remain under `docs/` and `docs/history.md`.
