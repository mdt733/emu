# v5027 expanded physical-frontier exam — 29 October 2026

**Evidence basis:** owner-provided finished per-world physical archive `v5027_15deg_3cm_physical_results.tar.gz`, in addition to the complete terminal log. The archive contains all three full `unified_exam.json` and `.md` files plus A/B/C console logs. Original output directory: `~/robot_lab/emu_runs/v5027_coherent_retention_s735_r2/extended_physical_fixed_20261029_065743`. Three policies were tested with identical **16 worlds/condition**, CPU MJX, seed **1729**, forward command **0.15m/s**, horizon **350** control frames, 5/8/10/12/15° four-direction grades and 2/2.5/3/3.5cm paired up/down steps. Original training source **`5dec60c4`** remains frozen; the exam source has the exact-15° boundary fix from results-audit SHA `5d44f02`. The per-world JSON has now been inspected. All evidence is simulated; not a hardware reliability claim.

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

**Critical correction from per-world JSON: all 3.0 and 3.5 cm successes were STEP-DOWNS. None of the three policies ascended a 3.0 or 3.5 cm ledge in any of the eight uphill worlds.** 2.5cm step-up is possible for A, but still not reliably commanded after climbing.

### Verified supported step-UP / step-DOWN splits (8 worlds each)

| Height | A UP / DOWN | B UP / DOWN | C UP / DOWN |
|---|---:|---:|---:|
| 2.0cm | **8/8 · 8/8** | **8/8 · 8/8** | 6/8 · 8/8 |
| 2.5cm | **6/8 · 7/8** | **0/8 · 7/8** | 2/8 · 8/8 |
| 3.0cm | **0/8 · 5/8** | **0/8 · 5/8** | **0/8 · 6/8** |
| 3.5cm | **0/8 · 4/8** | **0/8 · 2/8** | **0/8 · 5/8** |

3.0cm uphill did *not* generally cause immediate falls: A/B/C all survived **8/8** uphill worlds, but **0/8** crossed or satisfied movement tracking. The signed forward vx command error (actual minus requested, averaged across the eight uphill worlds) is **A −0.132m/s**, **B −0.135m/s**, **C −0.137m/s**, at a requested **+0.15m/s**. This is an obstacle approach/progression failure: roughly ~0.013–0.018m/s mean achieved forward speed, not a successful three-centimetre climb.

The **2cm uphill** episodes all crossed for A and B, but *full mission* scores were **A 1/8, B 0/8**, with movement tracking **8/8 each** and heading control **A 1/8, B 0/8**. So the missing post-crossing quality is largely yaw/heading control, not inability to produce forward travel. C crosses 6/8 uphill, passes the complete uphill mission 1/8, and has the same general yaw bottleneck.

At **2.5cm uphill**, A crosses **6/8**, C **2/8**, B **0/8**. The two C crossings and all six A crossings fail the *full mission* on the heading gate; B's eight uphill worlds all fail survival. This makes A the clearest experimental STEP-UP reference, and disproves a simple monotonic benefit from fully preserving the direct terrain command.

### Actual steep-slope failure mechanism in the per-world JSON

A and C often maintain plausible forward velocity but fail mean yaw control; B additionally shows severe downhill velocity drift. Example **12° uphill**, vx signed biases (actual minus requested) A **−0.018m/s**, B **−0.034m/s**, C **−0.034m/s**, yet provisional mission passes **9/16, 2/16, 16/16**; heading gate passes exactly **9/16, 2/16, 16/16**. C is controlling rotation better, *not just going faster*.

At **15° downhill**, all mission counts are 0/16. A survives 16/16 with forward vx MAE ~0.022m/s but yaw signed bias ~**+0.117rad/s**; B falls to 11/16 survival with mean forward **+0.247m/s bias** against +0.15m/s request and yaw bias **+0.388rad/s**; C survives 12/16 with vx MAE ~0.117m/s and yaw bias ~**−0.076rad/s**. This separates yaw limitation from true overspeed/loss of commanded motion. Right camber also produces severe sideways drift (A +0.110m/s signed vy bias at 15°, C +0.126m/s). Some strict passes remain 0 on all slopes because the legacy instantaneous yaw-MAE gate is much tighter than the provisional mission's bias/jitter test; both scores need retaining.

**Scope caution:** The examiner's `path_travel_m` is an *absolute accumulated distance*, not signed goal progress; it is not a standalone proof of downhill course completion. The supported far-side crossing test is genuine physical foot-contact progression, but the “mission” score still needs a signed-path/heading audit before public promotion.

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

1. **Treat main replacement, not isolated terrain scores, as the real objective.** The protected v5016 Pinnacle reportedly handles ≈1m/s on its trained slope-only curriculum. v5027's earned forward frontier at 60M is only ~0.37–0.43m/s, despite a configured safety headroom of 1.4m/s. Merely extending training does not guarantee frontier promotion or true 1m/s tracking. Require an identical physical command-speed sweep from .2 through 1.0m/s, with signed velocity MAE, heading/path, survival, recovery and mechanical J/m, compared to Pinnacle under a schema-neutral actual-contact protocol. Existing v5027 unified frozen terrain exams stop at 0.30m/s and do not certify speed.
2. **Stop treating increased forward command on BOTH slope and step focus as a proven fix.** The B−A contrast is negative on the 8–15° matrix and for >2cm uphill ascent. Under same-parent A/B, a direct STEP-only arm and separate SLOPE-only arm would isolate the trade-off.
3. **Use A's original coherent approach as the terrain reference** and C as evidence for *targeted* low-stress directional movement rehearsal. Explore 10–15% rehearsal (not a promise of improvement); also consider preserving 25% for directional control only while separately enhancing push recovery, because C did not preserve previous 60M resilience75.
4. Focus the known failure mechanisms: downhill command overspeed and yaw drift (especially B), right camber lateral bias, steep uphill yaw stability (A/B), and 3cm upward ledge approach rather than merely standing in front of the lip. Repair physics-aware control and shaping without turning the policy into a scripted gait.
5. The split is now known: 3cm upward is 0/8 across A/B/C; 2.5cm A 6/8. Make upward 3cm crossing with supported far-side contact a genuine graduation target, and report post-cross speed/heading separately.
6. Repeat the most discriminating cases with a **new evaluation seed and 32 worlds** at matching speed and geometry. Multiple trials on one trained seed are still not multi-training-seed replication.
7. Only after control diagnosis, perform bounded feature-branch experiments (e.g. original coherent + 10–15% moving rehearsals vs corrected STEP-only command treatment vs clean A baseline); don't spend 400M steps because training frontiers say 15°/3cm.

The complete raw per-world results were bundled by the owner and inspected; the following remains a reproducible WSL archive command:

```bash
ROOT="$HOME/robot_lab/emu_runs/v5027_coherent_retention_s735_r2"
EXAM="$ROOT/extended_physical_fixed_20261029_065743"
cd "$ROOT"
tar -czf "$HOME/v5027_15deg_3cm_physical_results.tar.gz" \
  -C "$ROOT" "$(basename "$EXAM")"
echo "$HOME/v5027_15deg_3cm_physical_results.tar.gz"
```

That archive is for further forensic analysis in ChatGPT; do not add large policy binaries or checkpoints. The evaluation source `~/emu-v5027-eval` and original frozen learner `~/emu-v5027` must remain pinned to their historical commits until this comparison is complete. Viewer-only adjustments remain on separate `feature/v5027-viewer-frontier`. No protected `main` or showcase publication changes.
