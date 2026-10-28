# v5027 expanded physical-frontier exam — 29 October 2026

**Evidence basis:** owner-pasted completed `capability-exam --terrain-matrix` console log from 29 Oct, folder `~/robot_lab/emu_runs/v5027_coherent_retention_s735_r2/extended_physical_fixed_20261029_065743`. Three actors A/B/C each completed the same **16-world per condition** frozen MJX physical assessment at seed **1729**, requested vx **0.15m/s**, horizon **350** control frames, 5/8/10/12/15-degree four-direction grades, and 2/2.5/3/3.5cm paired up/down steps. Original training source remains **`5dec60c4`**, first examiner had an exact-15° float32 boundary error fixed at results-audit SHA `5d44f02`. The user pasted the full terminal log, but **the per-world JSON files have not yet been provided**; up/down crossing splits, velocity biases and case-failure diagnoses remain unavailable.

## Directionally commanded slope missions — out of 16 each

| Grade / bearing | A original | B direct | C direct+25% moving rehearsal |
|---|---:|---:|---:|
| 5° uphill | 16 | 16 | 16 |
| 5° downhill | 16 | 8 | 15 |
| 5° left camber | 16 | 16 | 16 |
| 5° right camber | 13 | 14 | 15 |
| **5° subtotal /64** | **61** | **54** | **62** |
| 8° uphill | 15 | 13 | 14 |
| 8° downhill | 14 | 0 | 9 |
| 8° left camber | 16 | 15 | 16 |
| 8° right camber | 7 | 12 | 14 |
| **8° subtotal /64** | **52** | **40** | **53** |
| 10° uphill | 13 | 10 | 16 |
| 10° downhill | 11 | 0 | 2 |
| 10° left camber | 15 | 13 | 16 |
| 10° right camber | 6 | 5 | 12 |
| **10° subtotal /64** | **45** | **28** | **46** |
| 12° uphill | 9 | 2 | 16 |
| 12° downhill | 1 | 0 | 0 |
| 12° left camber | 14 | 10 | 16 |
| 12° right camber | 0 | 0 | 5 |
| **12° subtotal /64** | **24** | **12** | **37** |
| 15° uphill | 8 | 0 | 8 |
| 15° downhill | 0 | 0 | 0 |
| 15° left camber | 13 | 3 | 11 |
| 15° right camber | 0 | 0 | 0 |
| **15° subtotal /64** | **21** | **3** | **19** |
| **ALL 320 physical missions** | **203/320 = 63.4%** | **137/320 = 42.8%** | **217/320 = 67.8%** |

**Interpretation:** C marginally leads pooled terrain mission performance over A, and strongly beats B, especially at 12°. A slightly leads C specifically at 15° (21/64 vs 19/64); neither has mastered 15° commands. B's 0/16 downhill at 8°, 10°, 12° and 15° strongly argues against adopting **both** direct step and direct slope commands wholesale. The training step-command adjustment also changed the slope sampler: do not attribute all B degradation to step climbing alone. All 15° strict all-axis slope scores were **0/16** in every direction and policy; mission scores use the separately documented low-yaw bias/jitter criterion, not strict instantaneous yaw-MAE. Physical support stability was frequently 15–16/16 even where commanded downhill/right-camber motion failed. This is *control limitation*, not simply inability to stand.

## Actual paired discontinuities — each 16 worlds contains 8 up + 8 down

| Absolute step height | A crossing / mission | B crossing / mission | C crossing / mission |
|---|---:|---:|---:|
| 2.0cm | **16 / 9** | **16 / 5** | 14 / 5 |
| 2.5cm | **13 / 4** | 7 / 5 | 10 / 3 |
| 3.0cm | 5 / 0 | 5 / 1 | **6 / 1** |
| 3.5cm | 4 / 0 | 2 / 0 | **5 / 0** |
| **All crossings /64** | **38** | 30 | 35 |
| **All complete missions /64** | **13** | 11 | 9 |

3cm **physical crossing capability exists intermittently**, but complete 3cm motion is almost absent. Not enough evidence to say the robot can reliably climb 3cm: output totals aggregate UP and DOWN. Full per-world JSON must separate up/down via `worlds[].direction` and `crossing_pass`, and inspect `movement_tracking_pass`, `heading_control_pass`, velocity MAE and signed vx/vy/yaw bias.

### Directional non-terrain commands, strict all-axis score

| Command (16 worlds) | A | B | C |
|---|---:|---:|---:|
| Lateral left | 9 | 12 | **15** |
| Lateral right | 4 | 13 | **16** |
| Yaw left | **15** | 12 | 14 |
| Yaw right | 14 | 13 | **15** |
| **Total /64** | **42** | 50 | **60** |

C again materially improves lateral/yaw command performance versus A/B on an independent evaluation seed. This makes its **25% moving rehearsal a useful control component**, despite its weaker previous broad fixed exam and resilience75 survival. It does not establish overall superiority or protective recovery retention.

## What the evidence says to change next

1. **Stop treating increased forward command on BOTH slope and step focus as a proven fix.** The B−A contrast is negative on the 8–15° physics matrix and for >2cm crossing. Under one-seed/same-parent A/B, a direct step-only arm and a separate slope-direct arm would clarify the trade-off.
2. **Use A's original coherent approach as the terrain reference** and C as evidence for *targeted* low-stress directional movement rehearsal. Explore 10–15% rehearsal (not a promise of improvement); also consider preserving 25% for directional control only while separately enhancing push recovery, because C did not preserve previous 60M resilience75.
3. Diagnose failures from detailed JSON: **downhill** and **right camber** fall apart before uphill and left camber. Determine whether the main failure is mean vx tracking, lateral drift, yaw drift/oscillation, insufficient path progress, or actual falls. Do not modify the reward/controller until this is known.
4. For steps, first split 2.5/3cm **step-UP vs step-DOWN** and verify 5-frame supported far-side evidence on rising lip. 3cm physical crossing 5–6/16 pooled is not demonstrated 3cm ascent mastery.
5. Repeat the most discriminating cases with a **new evaluation seed and 32 worlds** at matching speed and geometry. Multiple trials on one trained seed are still not multi-training-seed replication.
6. Only after control diagnosis, perform bounded feature-branch experiments (e.g. original coherent + 10–15% moving rehearsals vs corrected STEP-only command treatment vs clean A baseline); don't spend 400M steps because training frontiers say 15°/3cm.

The complete artifacts are at the run-root path above and can be bundled without rerunning simulation:

```bash
ROOT="$HOME/robot_lab/emu_runs/v5027_coherent_retention_s735_r2"
EXAM="$ROOT/extended_physical_fixed_20261029_065743"
cd "$ROOT"
tar -czf "$HOME/v5027_15deg_3cm_physical_results.tar.gz" \
  -C "$ROOT" "$(basename "$EXAM")"
echo "$HOME/v5027_15deg_3cm_physical_results.tar.gz"
```

That archive is for further forensic analysis in ChatGPT; do not add large policy binaries or checkpoints. The evaluation source `~/emu-v5027-eval` and original frozen learner `~/emu-v5027` must remain pinned to their historical commits until this comparison is complete. Viewer-only adjustments remain on separate `feature/v5027-viewer-frontier`. No protected `main` or showcase publication changes.
