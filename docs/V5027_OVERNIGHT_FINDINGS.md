# EMU v5027 — complete 140M A/B/C overnight analysis

**Observed 29 October 2026.** Evidence: owner-supplied `overnight_results(4).zip`; this report uses the attached campaign metadata, full `metrics.csv`, `fixed_eval.csv`, `run_meta.json`, and the three final CPU physical `unified_exam.json` files. The ZIP does **not** include training checkpoint bytes; archived outputs report the final checkpoint files present, but independent hash revalidation requires the originals on the WSL machine.

## Provenance

- Training source SHA: **`5dec60c4ed22843d91c6aeef9cfdb90d73ef4c45`** (not this later results-documentation commit).
- Branch: `feature/v5027-coherent-retention`; seed 735; 2,048 CUDA MJX worlds; 360-channel array-only ToF policy; 20,054,016-step shared cold start.
- All A/B/C resumed from the SAME foundation final checkpoint at step 20,054,016. Each final policy reached **60,030,976 steps**, with zero `rollback_count` in the final metrics. Aggregate training ≈140M world steps across shared+forked runs; no policy is 140M old.
- Archived first failed import-path preflight did not begin training; completed campaign is under `~/robot_lab/emu_runs/v5027_coherent_retention_s735_r2`.
- Three clean CPU preflight variants and 2048 CUDA full-update preflight are recorded under the source-matched `preflight_pass.json`. Frozen terrain exam: CPU, 8 worlds per case, 350 control frames, 0.15m/s command, shared seed 1701, matched physical collision topology; the new v2 examiner distinguishes original strict yaw-rate score, physical mission, supported crossing, and verified slope stability.

## Experimental arms — one variable at a time

| Arm | Task | Moving retention |
|---|---|---:|
| **A original** | Original coherent STEP/SLOPE: forward intention 0.10–0.22m/s then half-strength auxiliary attenuation around +0.04m/s | None extra |
| **B direct** | Same coherent STEP/SLOPE but preserves the sampled forward crossing command and updates gait rate | None extra |
| **C retention** | Same direct command as B | Extra 25% moving CONTROL rehearsal |

Same shared parent and seed are useful controls, **not multi-seed replication**. The retention cohort redistributes specialist focus allocation; for example command/slope/step roles are 16.7% each in C versus 20% in A/B, so C's treatment also means fewer terrain and command births.

## Final same-age frozen generalist exams (60,030,976)

| Metric | A original | B direct | C + retention |
|---|---:|---:|---:|
| Broad fixed score | **0.76919** | 0.76485 | 0.74474 |
| Easy core survival | 64/64 (100%) | 64/64 (100%) | 64/64 (100%) |
| Dynamic80 survival | **64/64 (100%)** | **64/64 (100%)** | 62/64 (96.9%) |
| Resilience75 survival | **55/64 (85.9%)** | **55/64 (85.9%)** | 48/64 (75.0%) |
| Robust40 survival | 62/64 (96.9%) | 62/64 (96.9%) | 62/64 (96.9%) |
| Combo70 survival | 64/64 | 64/64 | 64/64 |
| Core forward tracking MAE | **0.01404m/s** | 0.01562m/s | 0.02002m/s |
| Core yaw-rate MAE | 0.08860rad/s | 0.08926rad/s | **0.07804rad/s** |

At 40M C led broad exam (A 0.6845, B 0.6969, C **0.7259**). At 50M C still narrowly led (A 0.7361, B 0.7456, C **0.7489**). But by 60M, **C's broad score declined to 0.7447**, while A/B improved, and C retained less resilience75 survival. This is a *retention trade-off*, not a demonstrated comprehensive no-forgetting improvement.

C's final-window on-policy falls (~14.0%) were lower than B (~18.4%) and A (~28.0%); those windows faced different specialist/command mixes and do **not** overturn the common fixed-exam comparison. Averaged over the last ten training windows: shaft proxy A ~105.1J/m, B ~95.8J/m, C ~88.7J/m; C looks promising for efficiency but energy comparisons also depend on policy-task exposure.

## Frozen physical terrain: independently supported results

### Same-height step crossings (four up + four down per case)

| Absolute height | A crossing / full mission | B crossing / full mission | C crossing / full mission |
|---|---:|---:|---:|
| 3mm | 8/8 · 8/8 | 8/8 · 8/8 | 8/8 · 8/8 |
| 5mm | 8/8 · 8/8 | 8/8 · 8/8 | 8/8 · 8/8 |
| 10mm | 8/8 · 8/8 | 8/8 · 7/8 | 8/8 · 8/8 |
| **20mm** | **6/8 · 3/8** | **7/8 · 4/8** | **7/8 · 3/8** |

At 20mm, uphill supported crossings **A 2/4, B 3/4, C 3/4**; all three crossed downhill **4/4**. For *full* 20mm moving missions, uphill passes were A **0/4**, B **2/4**, C **0/4**. Crossing and full post-cross command tracking must not be conflated.

Direct commands gained one supported 20mm uphill crossing in eight-world screening, with almost no change in broad score. This suggests directional forward intent helps, but is **not strong evidence of a large improvement at 2cm**; both B and C replicate the previous v5026 coherent learner's 7/8 physical crossing outcome. **No fixed 30mm step was tested**.

### Four-direction grades 1°, 2°, 3°, 5°

| Frozen grade outcome | A | B | C |
|---|---:|---:|---:|
| Verified moving missions over 128 worlds | **127/128** | 126/128 | **127/128** |
| 5° four-direction subset | 31/32 | 30/32 | 31/32 |
| Slope physical stability | 128/128 | 128/128 | 128/128 |

All three coherently taught policies perform well up to the **tested 5°**. No experiment has yet independently certified 15° command tracking; the `slope≤15.0°` console display is curriculum exposure.

### Directional commands: retention did learn something

| Strict frozen command, eight worlds each | A | B | C |
|---|---:|---:|---:|
| Lateral left | 4/8 | 3/8 | **8/8** |
| Lateral right | 1/8 | 4/8 | **7/8** |
| Yaw left | **8/8** | 6/8 | 5/8 |
| Yaw right | 3/8 | 4/8 | **8/8** |
| **Total four commands** | **16/32** | **17/32** | **28/32** |

C improved **sideways and turning coverage in the fixed physical examiner**, although yaw-left still fell behind A. It also reduced core yaw-rate tracking error, but lost fixed recovery survival and some core forward accuracy. It is **not** an overall generalist winner.

## Frontier and moving-cost observations (not independent mastery)

| End-of-run teaching/metric | A | B | C |
|---|---:|---:|---:|
| Step practice target | 30.8mm | 29.4mm | 31.4mm |
| Slope target | 15.0° | 15.0° | 13.6° |
| Resilience difficulty (unitless) | 0.515 | 0.489 | 0.491 |
| Positive vx teaching frontier | 0.369m/s | 0.425m/s | 0.425m/s |
| Last 10 windows: observed vx average | ~0.064m/s | ~0.107m/s | ~0.069m/s |
| Last 10 windows: operator cmd vx average | ~0.083m/s | ~0.119m/s | ~0.107m/s |
| Last 10 windows: control J/m average | ~105 | ~96 | ~89 |

Curricula differ, so the on-policy vx and J/m rows must **not** be described as equal-command speed/efficiency superiority. The positive teaching frontier `vx=[...]` is a curriculum parameter, not observed velocity. Mechanical shaft J/m is not electrical battery cost.

## 29 Oct follow-up — 15° validator roundoff, not a robot fall

In the extended 16-world, seed1729 0.15m/s exam, A printed all
5/8/10/12° cases then stopped on `uphill_15deg` due to
`physical terrain total inclination exceeds 15-degree exam cap`.
The exact requested fixture was 15.0°; the old float32
trigonometric roundtrip produced **15.00000095°**. The new
results-audit branch validates the float32 input in float64,
with a 1e-9° guard tolerance. The physical 15° cap and
MuJoCo geometry are **not** increased.

Pre-crash A **provisional physical mission** pass counts from
the console (not final saved JSON):

| Grade | Uphill | Downhill | Left camber | Right camber |
|---|---:|---:|---:|---:|
| 5° | 16/16 | 16/16 | 16/16 | 13/16 |
| 8° | 15/16 | 14/16 | 16/16 | 7/16 |
| 10° | 13/16 | 11/16 | 15/16 | 6/16 |
| 12° | 9/16 | 1/16 | 14/16 | 0/16 |

Most conditions stayed physically stable even when tracking
failed. The abort prevented full A JSON and all B/C exams;
the scorer writes final JSON only on success. Rerun in a
new unique output root and capture intermediate stdout with
`tee`, rather than overwriting the first attempt.
No trained-policy or public viewer files were changed.

## New read-only high-frontier physical examiner (results-audit branch)

The source-pinned v5027 experiment **did not** test whether the 31mm/15°
practice target was actually mastered. On `feature/v5027-results-audit`
only, `capability-exam --terrain-matrix` now accepts optional
`--matrix-heights-cm` (unique 0.1cm increments up to 4cm) and
`--matrix-grades-deg` (unique grades up to 15°). Defaults remain the
old comparable 3/5/10/20mm and 1/2/3/5° fixtures. Same actual rotated
MJX contact slabs, actor schema and frozen scorer; no optimizer, no
checkpoint migration, no teacher/curriculum changes. The examiner
records the requested grade/height arrays in output protocol metadata.

Run from a **fresh separate evaluation checkout**, not the finished
source-pinned `~/emu-v5027` learner. Use its existing exported actors:

```bash
git clone --single-branch --branch feature/v5027-results-audit \
  https://github.com/mdt733/emu.git "$HOME/emu-v5027-eval"
cd "$HOME/emu-v5027-eval"
source "$HOME/emu-graduation/.venv/bin/activate"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -c 'import emu.environment as e; print(e.__file__)'
RUN="$HOME/robot_lab/emu_runs/v5027_coherent_retention_s735_r2"
OUT="$RUN/extended_physical_$(date +%Y%m%d_%H%M%S)"
for ARM in A_coherent_original B_coherent_direct C_direct_retention; do
  python -m emu.cli capability-exam --terrain-matrix \
    --policy "$RUN/physical_policies/$ARM.emupolicy" \
    --output "$OUT/$ARM" --worlds 16 --horizon 350 --seed 1729 \
    --speed 0.15 --device cpu \
    --matrix-grades-deg 5 8 10 12 15 \
    --matrix-heights-cm 2 2.5 3 3.5
done
echo "Extended physical exams: $OUT"
```

**Caution:** this is an independent examiner-only revision with a
different Git source SHA from the actual training. It loads immutable,
contract-verified `.emupolicy` exports, and retains each actor's
trained policy hash and age. Never `--resume` an old training
checkpoint under this changed examiner SHA. Neither the extended
exams nor their behavior on the owner's CPU have yet been verified.
Treat 16-world results as screening, then repeat interesting
conditional results using 32 worlds / additional seeds. **Do not
mistake slope stability for directed locomotion or step crossing for
post-cross speed/heading control.**

**New 29 Oct evaluation result:** The complete independent 16-world, seed1729, 5–15°/2–3.5cm A/B/C physical matrix is documented in [v5027 extended physical results](V5027_EXTENDED_PHYSICAL_FINDINGS.md). B's direct command adjustment significantly regressed steep grades relative to A; C improves steering and 12° uphill but not 3cm full mission. Preserve the original finished experiment, await per-world JSON for ascent-vs-descent analysis.

## Interpretation and next decision

**A** remains the marginal winner on 60M frozen broad score and familiar recovery, but **B** preserves nearly all that competence while achieving more on-policy directed locomotion and one more verified 20mm uphill crossing. **C** demonstrates a meaningful learnable *control benefit* from moving retention, but a 25% allocation is too large or insufficiently targeted to claim recovery/no-forgetting protection.

For subsequent engineering, treat **B as the reference for further coherent-directed terrain research**, **A as the measured broad generalist control**, and **C as evidence that moving, lower-difficulty command rehearsal can improve signed sideways/yaw tasks**. None should replace the proven `main` Pinnacle on this evidence. The differences are single-seed and eight-world physical samples.

Recommended next work, with **no 400M run yet**:

1. Repeat frozen *same-speed* 10/20mm UP/DOWN and 3/5° grades with multiple evaluation seeds and 32 worlds. Distinguish supported crossing from full command/heading tracking; don't waive yaw accuracy.
2. Extend only the **exam** to 25/30/35mm and 8/10/12/15° with careful safety/physical-lip validation; test B/A/C and old v5026 E on equal policy-compatible geometry. These difficulties **have not been tested**.
3. Add genuine 0/0.5/1/2/3N recovery comparisons at multiple commanded speeds so the retention hypothesis is tested under equal task distributions.
4. Explore *smaller or targeted* moving-retention share (e.g., 10–15%), preserving coherent terrain, and include targeted yaw-left and forward-command tracking. This is a proposed experimental knob, not an established improvement.
5. Keep any new code/training on a **new source-pinned experiment branch** with fresh run directory; never swap a completed 60M checkpoint to a different source SHA. Respect protected `main`, public champion and single-brain 360-input interface.

**Artifact handling:** this is a **documentation-only results branch**. The original GPU source commit `5dec60c` and `~/emu-v5027` should remain untouched if further same-source evaluation or restart is intended. Source-pinned checkpoint portability to a new training implementation is not assumed.
