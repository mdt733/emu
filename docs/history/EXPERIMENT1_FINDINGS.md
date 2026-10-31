# Experiment 1 — 290M-step graduation screening: results and decision

**Source:** `ab29c0270bf93b4c0339c5989d2fd23c6976f8e4` on `feature/v5024-release-polish`  
**Evidence:** owner-provided `overnight_results(2).zip` (manifest, metrics, physical teacher counters, frozen generalist exams, nested evidence)  
**Seed:** 734  
**Completion:** 50,003,968 shared environment steps, then all 12 forks each ending at 70,057,984; all 13 run records complete, no reported rollbacks. **290M requested additional world steps in total.** The 50M parent is common to every fork.

## Executive verdict

The experiment successfully established a cold-start 360-input actor and explored twelve continuations. **Do not graduate a policy yet.** All 13 passed the fixed exam's easy-core survival cohort, but physical challenge competency is not close to the published Pinnacle's aspirations. The faster teacher **enlarged reported command/yaw/push practice frontiers much more quickly than it developed real uphill/step capability**.

### Held-out generalist exam at 70M

| Fork | Broad score | Core vx MAE (m/s) | Dynamic80 survival | Resilience75 survival |
| --- | ---: | ---: | ---: | ---: |
| Staged, integral OFF | **0.7343** | 0.0127 | 98.44% | 76.56% |
| Balanced, I OFF | 0.7139 | 0.0145 | 98.44% | 73.44% |
| Balanced, I 400ms/2× | 0.7231 | **0.0110** | 96.88% | 76.56% |
| Balanced, I 800ms/2× | 0.7131 | 0.0118 | 96.88% | **79.69%** |
| Balanced, I 200ms/2× | **0.6864** | 0.0113 | 98.44% | 75.00% |
| Terrain-early, I OFF | 0.7172 | 0.0126 | 98.44% | 73.44% |

These scores compare the **same current fixed generalist exam**. They are not a 1–5° physical directional terrain matrix, a step-height exam or a like-for-like Pinnacle comparison. One parent checkpoint/seed is a screen, not replication. Training fall rates are drawn from *different challenge distributions*.

### Actual practice frontier versus behaviour

- Shared 50M: final forward frontier **+0.279 m/s**, lateral **±0.093 m/s**, yaw **1.31 rad/s**, resilience frontier **4N**, real slope practice **0.35°**, step-height practice **1.5mm**. Final training average forward vx **0.0636 m/s** versus raw requested **0.0747 m/s**. Generalist frozen score **0.6659**.
- At 70M, balanced I OFF: forward frontier **+0.321 m/s**, average actual/raw requested vx **0.077/0.066 m/s**, slope **0.35°**, step **1.5mm**; fixed score **0.7139**.
- At 70M, balanced I 400ms/2×: frontier **+0.374 m/s**, actual/requested **0.067/0.069 m/s**, slope **0.36°**, step **1.5mm**; fixed score **0.7231**.
- At 70M, staged I OFF: **highest broad exam 0.7343**, but final actual/requested vx was **−0.0056/+0.0474 m/s** on its own training distribution. It is *not* demonstrated faster commanded walking.
- ALL TWELVE 70M forks ended with step-height target at the **1.5mm retreat floor**. Ten ended at **0.35° slope floor**; remaining two ended at approximately 0.36° and 0.44°. This is poor terrain progress, irrespective of the broader generalist exam scores.  
- All 70M forks had their resilience target at the **4N ceiling** and yaw near **3.5 rad/s**, despite substantial ordinary final-rollout fall rates (roughly 15.7–36.4%). These are **practice targets**, not certified externally demonstrated recovery or turning.

### Why simply making promotions larger is insufficient

1. **Sparse/difficult command proof.** The shared learner logged 35 completed forward near-edge trials over its sampled reports and only 9 passed. During the final 20M, balanced I OFF recorded 30 forward near-edge completions but **only 4 passed**. Forward command frontier increasingly outran reliable edge tracking. Concentrate training samples near a physically useful speed band; do not claim +0.32m/s is achieved just because its frontier expanded.
2. **Step teacher trapped at floor.** Balanced I OFF had near-height uphill **9/57** and downhill **22/69** supported, complete-episode successes during the final 20M; the terrain-specific I-OFF fork had **17/84 up** and **18/77 down**. Its teacher retreated on failing *easier* real step episodes **21 times in 31 reporting rows** (terrain-specific: 24 times). More terrain exposure didn't fix the crossing skill. Real crossing has a separate `terrain_cross_new` signal, but certification currently requires both a far-side crossing **and surviving to episode timeout**; decompose contact/crossing, post-cross survival and walking speed before redesigning rewards.
3. **Slope teacher oscillates to minimum.** A single failed short easier cohort in **any of four orientations** can retreat the ONE global slope target, clearing evidence. Balanced I OFF registered two such retreats during 50→70M and ended at the 0.35° floor. The terrain-specific fork registered eight. A shared worst-direction frontier gives little room for repeated graduated uphill/camber practice.
4. **Resilience practice maxed too soon.** 8% multiplicative increases, gated on general resilience competence/fall EMA rather than explicit per-force held-out proof, rapidly drove the push frontier to the 4N safety ceiling. Do not interpret this as robot competence at 4N. A higher 4N envelope can affect survival and dilute terrain learning. Separately qualify recovery at specific physical force values.
5. **Integral trim is promising but modest.** The controlled balanced 400ms/2× arm improves the generalist exam from 0.7139 to 0.7231 and core vx MAE from 0.0145 to 0.0110, while final rollout falls declined 22.2%→18.8% on its own sampled tasks. This is evidence to **retain for further paired examination**, not proof on hills. Fast 200ms/2× was weaker (0.6864 exam; 36.4% final-rollout falls), whereas 800ms/2× showed resilience75 79.69%. Multiplicative I cannot reject zero-command `vy` drift on cambers.

## Experiment 2 — revised, **no 400M run yet**

**First: freeze and examine existing 70M policies (zero additional PPO steps)** on equal real MJX contacts: 0/1/2/3/5° uphill, downhill and both cambers; up/down steps 0/1/2/4cm; flat command tracking from 0.05 up to 1m/s in sensible bins; stop/reverse, lateral/yaw, 0/1/2/4N pushes, work per *useful* metre. Use normal sensor inputs and raw operator requests. Include staged OFF, balanced OFF, balanced I 400ms/2× and balanced I 800ms/2×. Export a fair policy-neutral comparator to Pinnacle before claims of superiority.

**Then: run deliberately isolated short training forks**, not twelve more uncontrolled recipes. Recommended factors:

- **Terrain teacher repair:** do not let one sparse easier cohort repeatedly reset the challenge frontier. Track separately (a) exposure/difficulty being *sampled*, (b) supported physical crossing/grade stability, (c) full-episode survival, and (d) command tracking. Maintain a nonzero proportion of taller/steeper stretch probes beyond the certified frontier. Require sustained easy failure for retreat; retain signed up/down and four-way slope evidence. Improve foothold/progress feedback only if contact traces justify it.
- **Speed/steering teacher repair:** add enough targeted pure-forward/negative/lateral/yaw trials to make near-edge proof useful; challenge beyond demonstrated speed in a bounded band, but gate *qualification* on actual tracking. Prevent yaw/push target saturation from being misread as achieved multi-axis competence.
- **Resilience dose control:** evaluate genuinely held-out force-conditioned recoveries and grow challenge frequency/intensity from their outcomes, with gentler coexistence on difficult steps/slopes.
- **Control trial:** staged OFF vs the same lesson recipe + bounded sensor-based velocity-I (prefer 400ms/2×; 800ms/2× secondary). Keep raw operator commands as scoring reference and check zero-vy camber separately (bounded additive controller is an independent hypothesis).

Any changed source must use an **explicitly reviewed migration/fork mechanism** for the source-pinned frozen policy, proving identical actor/input/physical state shape and serializing new controller/curriculum state. Do NOT quietly disable Git provenance or force-read an `ab29c027` checkpoint under a different SHA; if migration parity cannot be established, run the foundation cold. Keep the public Pinnacle and `main` untouched.

Only after credible real terrain, command-envelope and recovery improvement should the single 400M cold-start graduation run begin.

## Engineering record

- Owner report: `overnight_results(2).zip`, verified 13 completed records, training config, run_meta, heldout and nested capability evidence.
- Linked campaign source: `ab29c0270bf93b4c0339c5989d2fd23c6976f8e4`.
- This document is a **result analysis and proposed Experiment 2**, not a claim that repairs, physical matrix or GPU training are implemented.
