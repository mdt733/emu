# EMU v5007 results

Generated 2026-10-10 02:15:05

- Source version: `v5007`
- Git commit: `8239fd9a0fea3f9908173cdb9624af6dc6c6a13f`
- Git branch: `main`
- Source dirty at launch: `False`
- Raw run store: `/home/mdt733/robot_lab/emu_runs/v5007`

Fixed held-out evaluators are the main comparison ruler. Training fall rate and live frontier values are supporting evidence, not branch-selection scores by themselves.

## Shared foundation

- run: `foundation20`
- target steps: `20,000,000`
- seed: `733`
- checkpoint used by all phase-1 branches: `/home/mdt733/robot_lab/emu_runs/v5007/rung_runs/foundation20/checkpoints/best_broad_eval.pkl`

## Run summary

| run | phase | seed | ok | step | durability | dyn80 | combo70 | res75 | robust40 | growth110 | growthcombo110 | vx MAE | yaw MAE | vy frontier | yaw frontier | combo radius | push N |
|---|---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| foundation20 | foundation | 733 | yes | 20054016 | -0.383 | 0.040 | 0.165 | 0.048 | 0.197 | 0.037 | 0.074 | 0.3405 | 1.1955 | 0.2200 | 1.5000 | 1.000 | 0.700 |
| slew_static | phase1 | 733 | yes | 60030976 | 0.523 | 0.938 | 0.999 | 0.713 | 0.967 | 0.584 | 0.935 | 0.0575 | 0.2670 | 0.2200 | 1.5000 | 1.000 | 0.700 |
| noslew_static | phase1 | 733 | yes | 60030976 | 0.729 | 0.991 | 1.000 | 0.793 | 0.991 | 0.789 | 0.986 | 0.0305 | 0.1426 | 0.2200 | 1.5000 | 1.000 | 0.700 |
| frontier_axis | phase1 | 733 | yes | 60030976 | 0.731 | 0.996 | 0.997 | 0.892 | 0.996 | 0.993 | 0.994 | 0.0311 | 0.2646 | 0.2200 | 1.5000 | 1.000 | 0.700 |
| frontier_repaired | phase1 | 733 | yes | 60030976 | 0.763 | 0.999 | 1.000 | 0.865 | 0.996 | 0.969 | 0.999 | 0.0354 | 0.1942 | 0.2200 | 1.5000 | 1.000 | 0.700 |
| frontier_no_rehearsal | phase1 | 733 | yes | 60030976 | 0.769 | 0.999 | 1.000 | 0.898 | 0.999 | 0.929 | 0.999 | 0.0300 | 0.2275 | 0.2200 | 1.5000 | 1.000 | 0.700 |
| frontier_push | phase1 | 733 | yes | 60030976 | 0.770 | 1.000 | 1.000 | 0.858 | 0.994 | 0.987 | 1.000 | 0.0314 | 0.2074 | 0.2200 | 1.5000 | 1.000 | 2.403 |
| frontier_accel | phase1 | 733 | yes | 60030976 | 0.772 | 0.998 | 1.000 | 0.856 | 0.995 | 0.957 | 0.999 | 0.0307 | 0.2113 | 0.2200 | 1.5000 | 1.000 | 0.700 |
| frontier_push_accel | phase1 | 733 | yes | 60030976 | 0.744 | 1.000 | 1.000 | 0.849 | 0.996 | 0.965 | 1.000 | 0.0402 | 0.2399 | 0.2200 | 1.5000 | 1.000 | 2.407 |
| frontier_lower_kl | phase1 | 733 | yes | 60030976 | 0.777 | 1.000 | 1.000 | 0.858 | 0.996 | 0.954 | 1.000 | 0.0348 | 0.1849 | 0.2200 | 1.5000 | 1.000 | 0.700 |
| resilience_focus | phase1 | 733 | yes | 60030976 | 0.780 | 1.000 | 1.000 | 0.869 | 0.997 | 0.996 | 1.000 | 0.0342 | 0.2212 | 0.2200 | 1.5000 | 1.000 | 2.396 |
| sensor_frontier_repaired | phase1 | 733 | yes | 60030976 | 0.779 | 1.000 | 1.000 | 0.876 | 0.999 | 0.958 | 1.000 | 0.0415 | 0.2003 | 0.2200 | 1.5000 | 1.000 | 0.700 |
| frontier_beast2 | phase1 | 733 | yes | 60030976 | 0.747 | 1.000 | 1.000 | 0.870 | 0.999 | 0.975 | 1.000 | 0.0408 | 0.2463 | 0.2200 | 1.5000 | 1.000 | 2.400 |
| resilience_focus__confirm | confirm | 1742 | yes | 60030976 | 0.774 | 1.000 | 1.000 | 0.889 | 1.000 | 0.974 | 1.000 | 0.0321 | 0.2004 | 0.2200 | 1.5000 | 1.000 | 2.403 |
| sensor_frontier_repaired__confirm | confirm | 1742 | yes | 60030976 | 0.777 | 1.000 | 1.000 | 0.892 | 0.999 | 0.986 | 1.000 | 0.0347 | 0.1962 | 0.2200 | 1.5000 | 1.000 | 0.700 |
| sensor_frontier_repaired__phase2 | phase2 | 733 | yes | 120061952 | 0.826 | 1.000 | 1.000 | 0.959 | 1.000 | 0.990 | 1.000 | 0.0305 | 0.1771 | 0.2200 | 1.5000 | 1.000 | 0.700 |

## Cross-seed family scores

| family | mean durability |
|---|---:|
| sensor_frontier_repaired | 0.778 |
| resilience_focus | 0.777 |
| frontier_lower_kl | 0.777 |
| frontier_accel | 0.772 |
| frontier_push | 0.770 |
| frontier_no_rehearsal | 0.769 |
| frontier_repaired | 0.763 |
| frontier_beast2 | 0.747 |
| frontier_push_accel | 0.744 |
| frontier_axis | 0.731 |
| noslew_static | 0.729 |
| slew_static | 0.523 |

## Phase 2 selection

- family: `sensor_frontier_repaired`
- source run: `sensor_frontier_repaired`
- selection: mean family durability across available training seeds

## Errors

No branch errors recorded.

## Raw evidence location

- `/home/mdt733/robot_lab/emu_runs/v5007/ladder_summary.csv` — one-row-per-run comparison.
- `/home/mdt733/robot_lab/emu_runs/v5007/run_meta.json` — full launch flags, provenance, selection and error metadata.
- `/home/mdt733/robot_lab/emu_runs/v5007/rung_runs/<name>/fixed_eval.csv` — complete held-out trajectory.
- `/home/mdt733/robot_lab/emu_runs/v5007/rung_runs/<name>/metrics.csv` — full training/frontier telemetry.
- `/home/mdt733/robot_lab/emu_runs/v5007/rung_runs/<name>/checkpoints/best_broad_eval.pkl` — preferred policy checkpoint.

The raw paths above are intentionally outside Git. This directory contains the compact evidence worth preserving with the source history.
