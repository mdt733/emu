# EMU v5011 results

Generated 2026-10-12 01:44:05

- Source version: `v5011`
- Git commit: `52b6df4f76f2125198e501a6e1e320834596ff1b`
- Git branch: `main`
- Source dirty at launch: `False`
- Raw run store: `/home/mdt733/robot_lab/emu_runs/v5011`

v5011 continuously turns ordinary PPO experience into semantic command/recovery/maneuver events. Held-out evaluators remain quarantined calibration/exam evidence and never update the live field.

## Shared foundation

- run: `trainer_foundation20_seed733`
- target steps: `20,000,000`
- seed: `733`
- checkpoint used by all phase-1 branches: `/home/mdt733/robot_lab/emu_runs/v5011/rung_runs/trainer_foundation20_seed733/checkpoints/latest_mjx.pkl`

## Run summary

| run | phase | seed | ok | step | durability | heldout recovery | worst-dir | cal MAE | nodes known/mastered | edges known/mastered | vx+ | vx- | seconds |
|---|---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| trainer_foundation20_seed733 | trainer_foundation | 733 | yes | 20054016 | -0.384 | -1.000 | -1.000 | - | 2/0 | 2/0 | 0.600 | -0.200 | 1237.7 |
| trainer_graph | trainer_20_to_120 | 733 | yes | 120061952 | 0.816 | 0.565 | 0.214 | 0.631 | 310/0 | 1250/0 | 0.809 | -0.600 | 4357.2 |

* Field node/edge counts are **training-derived** self-map evidence, useful for trainer efficiency and diagnosis. Held-out recovery/worst-direction columns remain independent exam evidence.

## Trainer selection

- selected trainer: `trainer_graph`
- rule: regression-safe lexicographic scheduler: held-out recovery first; continuous field coverage is training-derived tie/efficiency evidence, not an exam score

| trainer | rank tuple | heldout recovery | competent nodes | competent edges | durability |
|---|---|---:|---:|---:|---:|
| trainer_graph | `[1, 0.5654296875, 0.21354166666666666, 0.6, 0.0, 1250.0, 0.0, 0.8162122914508326]` | 0.565 | 0 | 0 | 0.816 |

## Errors

No branch errors recorded.

## Raw evidence location

- `/home/mdt733/robot_lab/emu_runs/v5011/ladder_summary.csv` — one-row-per-run comparison.
- `/home/mdt733/robot_lab/emu_runs/v5011/run_meta.json` — full launch flags, provenance, selection and error metadata.
- `/home/mdt733/robot_lab/emu_runs/v5011/rung_runs/<name>/fixed_eval.csv` — complete held-out trajectory.
- `/home/mdt733/robot_lab/emu_runs/v5011/rung_runs/<name>/metrics.csv` — full training/frontier telemetry.
- `/home/mdt733/robot_lab/emu_runs/v5011/rung_runs/<name>/performance.json` — cumulative wall-time attribution when profiling is enabled.
- `/home/mdt733/robot_lab/emu_runs/v5011/rung_runs/<name>/heldout_capability_cloud.csv` — held-out sparse combined-command probe points accumulated during training.
- `/home/mdt733/robot_lab/emu_runs/v5011/rung_runs/<name>/capability_field_latest.npz` — continuously updated training-derived node/recovery/maneuver field.
- `/home/mdt733/robot_lab/emu_runs/v5011/rung_runs/<name>/capability_field_summary.json` — compact current field diagnostics.
- `/home/mdt733/robot_lab/emu_runs/v5011/rung_runs/<name>/checkpoints/best_eval.pkl` — public alias for the preferred fixed-regression checkpoint; historical filenames remain readable.

The raw paths above are intentionally outside Git. This directory contains the compact evidence worth preserving with the source history.
