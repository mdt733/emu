# v5030 temporal-reward A/B — physical findings and next gate

**Finished, source frozen:** `a7be11b587f0912e642e100b22ec6d617f548f5b`, 60,030,976 steps per arm; shared cold 20M foundation, seed 747. This document lives on the *separate reporting branch*, which must not be used as a checkpoint-compatible training source merely because it carries the same exam records.

## Decision

**Promote neither arm to Pinnacle.** B's 0.5-second forward reward shows a credible benefit worth retaining as a candidate, not proof it is globally superior.

- At commanded **0.6m/s**, A reaches 0.544m/s, survives 14/16 worlds; B reaches 0.558m/s, survives 16/16. B's forward MAE is 0.051 versus A's 0.098m/s.
- At 0.6m/s, per-world mean *absolute signed yaw-rate bias* improves from **0.563rad/s A to 0.121rad/s B**. This is distinct from stride jitter; avoid teaching the walker to look motionless. B's residual bias still needs measured long-term heading/path stability.
- Both complete supported **2cm upsteps 6/8**. At **2.5cm up**, A 0/8 versus B 2/8; at **3.0cm up**, both 0/8. Downsteps must be reported separately.
- B's slope *mission* has regressions: uphill 8° A 10/16 vs B 1/16 and downhill 12° A 11/16 vs B 0/16. Survival in these cases remains 16/16 for both at uphill 8° and downhill 12°; inspect heading and speed before calling this inability to climb.
- At commanded **1.0m/s**, neither policy survives any world. The mean first episode is under 1s; any transient peak is **not** sustained 1m/s evidence.
- The 0/16 speed-ladder mission flag at 0.3m/s and above includes the current jitter threshold even when physical forward movement survives. Preserve both the mission score and its separate components rather than claiming all higher speeds fail physically.

## Minimal next decision path

1. **Evaluator-only, no retraining:** add stride-aware net heading/path displacement (net heading drift, lateral displacement and progress over whole stride windows); keep oscillation as separately reported telemetry. Add an optional progressive start/ramp in the 0.8–1.0m/s ladder to isolate launch transient from running competence, without replacing direct command tests.
2. **One step-specific reward experiment** if evidence still shows failed supported ascent: compare unchanged control with *bounded*, actual-support-verified ascent credit. No repeated height rocking, no teacher-only actor inputs, no stacked free-credit with the old near-lip allowance. Prefer B as a candidate baseline but continue retaining A as an independent control.
3. Before a 200–300M next-Pinnacle cold start, require step-up preservation, 0.6+ actual velocity/survival, heading path control, and acceptable grade/downstep losses under matched physical exams. Select best checkpoint by held-out tests rather than final age.

The following generated read-only comparison retains traceable numbers, protocols and hash provenance. Run `python scripts/v5030_physical_report.py /path/to/v5030_temporal_physical_comparison.zip --output-dir /tmp/emu_report` to regenerate it.

---

# EMU — matched physical comparison (decision report)

**Source SHA:** `a7be11b587f0912e642e100b22ec6d617f548f5b`  
**Arms:** A = `A_original25`, B = `B_halfsecond25`.  
**Interpretation:** physical mission, supported crossing, survival and tracking are separate measures. This is a single-seed screen, not a statistical guarantee.

**A_original25:** 60,030,976 steps; actor SHA256 `c3803ef1c1327a1c9fa0208533ee95c2153756765ea9d2e8440e60ede163a884`.  
**B_halfsecond25:** 60,030,976 steps; actor SHA256 `351fafd361c6196d1cc58964b491e61e07f4480d1880f7a73974408e2678789c`.  

Fixture: two_32m_half_length_sharp_slabs_v5029; 16 worlds per speed, 700 frames, seed 1701.

## Frozen evaluation (broad cohort)

| Arm | Broad score | Fall rate (training) |
|---|---:|---:|
| A_original25 | 0.7432138744907784 | 0.2688172161579132 |
| B_halfsecond25 | 0.7525993854748045 | 0.25 |

## Actual forward walking — same commands

| Request | A actual | B actual | A vx MAE | B vx MAE | A survived | B survived | A mission | B mission |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.080 | 0.070 | 0.022 | 0.030 | 16/16 | 16/16 | 15/16 | 16/16 |
| 0.20 | 0.171 | 0.161 | 0.030 | 0.039 | 16/16 | 16/16 | 12/16 | 15/16 |
| 0.30 | 0.268 | 0.259 | 0.034 | 0.042 | 16/16 | 16/16 | 0/16 | 0/16 |
| 0.40 | 0.359 | 0.351 | 0.045 | 0.051 | 16/16 | 16/16 | 0/16 | 0/16 |
| 0.50 | 0.451 | 0.460 | 0.055 | 0.045 | 16/16 | 16/16 | 0/16 | 0/16 |
| 0.60 | 0.544 | 0.558 | 0.098 | 0.051 | 14/16 | 16/16 | 0/16 | 0/16 |
| 0.80 | 0.785 | 0.819 | 0.205 | 0.158 | 7/16 | 8/16 | 0/16 | 0/16 |
| 1.00 | early fall | early fall | — | 0.106 | 0/16 | 0/16 | 0/16 | 0/16 |

Average speed/MAE use available per-world values and include early-terminated worlds where reported. "Early fall" suppresses misleading average speed when the **mean episode lasts under 100 control frames**; it is not a sustained speed result.

## Straight-line directional control — separate drift from oscillation

| Requested vx | A abs yaw bias | B abs yaw bias | A yaw jitter | B yaw jitter | A heading pass | B heading pass |
|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.021 | 0.023 | 0.126 | 0.116 | 15/16 | 16/16 |
| 0.20 | 0.031 | 0.021 | 0.184 | 0.180 | 12/16 | 15/16 |
| 0.30 | 0.033 | 0.050 | 0.298 | 0.288 | 0/16 | 0/16 |
| 0.40 | 0.239 | 0.156 | 0.519 | 0.422 | 0/16 | 0/16 |
| 0.50 | 0.450 | 0.204 | 0.776 | 0.587 | 0/16 | 0/16 |
| 0.60 | 0.563 | 0.121 | 0.936 | 0.737 | 0/16 | 0/16 |
| 0.80 | 0.755 | 0.615 | 1.165 | 1.265 | 0/16 | 0/16 |
| 1.00 | early fall | early fall | early fall | early fall | 0/16 | 0/16 |

Yaw bias = mean of each world’s **absolute signed** yaw-rate bias (rad/s), not the mean of absolute instantaneous yaw rates. Jitter = the exam’s measured yaw-rate fluctuation (rad/s). Stride-scale jitter is a diagnostic, not an automatic reason to penalise the gait. Bias can accumulate into meaningful heading drift; verify with path traces.

## Physically supported steps — UP versus DOWN

| Height | A UP | B UP | A DOWN | B DOWN | A mission total | B mission total |
|---:|---:|---:|---:|---:|---:|---:|
| 2.0 cm | 6/8 | 6/8 | 8/8 | 8/8 | 7/16 | 4/16 |
| 2.5 cm | 0/8 | 2/8 | 7/8 | 8/8 | 4/16 | 2/16 |
| 3.0 cm | 0/8 | 0/8 | 4/8 | 7/8 | 1/16 | 1/16 |
| 3.5 cm | 0/8 | 0/8 | 2/8 | 1/8 | 0/16 | 0/16 |

Supported crossings require real far-side contact and five supported frames. "Mission" also requires motion/heading tracking, and is deliberately different from physical crossing.

## Slope / steering missions and survival

| Test | A mission | B mission | A survived | B survived | A vx actual | B vx actual |
|---|---:|---:|---:|---:|---:|---:|
| flat | 16/16 | 16/16 | 16/16 | 16/16 | 0.124 | 0.115 |
| uphill_5deg | 16/16 | 15/16 | 16/16 | 16/16 | 0.107 | 0.104 |
| downhill_5deg | 16/16 | 15/16 | 16/16 | 16/16 | 0.132 | 0.132 |
| cross_left_5deg | 15/16 | 15/16 | 16/16 | 16/16 | 0.128 | 0.119 |
| cross_right_5deg | 16/16 | 16/16 | 16/16 | 16/16 | 0.122 | 0.112 |
| uphill_8deg | 10/16 | 1/16 | 16/16 | 16/16 | 0.111 | 0.109 |
| downhill_8deg | 14/16 | 16/16 | 16/16 | 16/16 | 0.130 | 0.138 |
| cross_left_8deg | 14/16 | 14/16 | 16/16 | 16/16 | 0.119 | 0.126 |
| cross_right_8deg | 15/16 | 14/16 | 15/16 | 16/16 | 0.138 | 0.111 |
| uphill_12deg | 0/16 | 0/16 | 16/16 | 15/16 | 0.113 | 0.120 |
| downhill_12deg | 11/16 | 0/16 | 16/16 | 16/16 | 0.148 | 0.173 |
| cross_left_12deg | 4/16 | 3/16 | 16/16 | 14/16 | 0.108 | 0.196 |
| cross_right_12deg | 9/16 | 11/16 | 13/16 | 15/16 | 0.045 | 0.061 |
| uphill_14p9deg | 0/16 | 0/16 | 16/16 | 14/16 | 0.111 | 0.214 |
| downhill_14p9deg | 0/16 | 0/16 | 15/16 | 12/16 | 0.201 | 0.328 |
| cross_left_14p9deg | 0/16 | 0/16 | 7/16 | 12/16 | 0.117 | 0.149 |
| cross_right_14p9deg | 2/16 | 4/16 | 13/16 | 13/16 | 0.028 | 0.005 |
| lateral_left | 16/16 | 16/16 | 16/16 | 16/16 | 0.002 | -0.012 |
| lateral_right | 16/16 | 14/16 | 16/16 | 16/16 | -0.006 | -0.013 |
| yaw_left | 13/16 | 13/16 | 16/16 | 16/16 | -0.002 | -0.011 |
| yaw_right | 10/16 | 10/16 | 16/16 | 16/16 | -0.002 | -0.014 |

## Evaluation contract and limitations

- The two arms share the same examined source commit, requested cases, seed, fixture and test durations; their checkpoint/actor hashes differ as expected.
- No automatic treatment winner: compare actual speed, drift, crossings, falls and grade performance by mission. The broad frozen score alone is insufficient.
- 0/16 mission at faster gait can result from yaw-jitter thresholds even when support, movement and speed are useful. Do not confuse gait oscillation with long-term directional error.
- The 1.0m/s test may include rapid failure; do not cite transient speeds as evidence of sustained walking. Consider a progressive-speed exam.
- Per-world evidence and uncertainty remain in the original JSON. Any 8/8 or 2/8 directional crossing count has wide uncertainty.
- Mechanical shaft work per travel metre is only a proxy for energy usage. It is not electrical or battery energy.
- This report does not touch training, rewards, actor data, checkpoints, or the main branch.

