# EMU v5025 forensic comparison: why main learned better

Status: **source-and-telemetry audit; no training changes**  
Baseline source: protected `main` at `2ec9e9db0a816c658131311129ef89ec7a595a12`  
Failed-growth screen source: `ab29c0270bf93b4c0339c5989d2fd23c6976f8e4`  
Evidence: owner-provided `overnight_results(2).zip`: one 50M common cold-start learner plus twelve 20M forks ending at **70M per actor**, 290M aggregate world steps.  
Related: [experiment results](EXPERIMENT1_FINDINGS.md).

## Verdict

The main gait learner, PPO and eight-payment reward largely survived the refactor. The new actor **learned foundation walking**: at ~19–20M, actual and commanded average vx were both ~0.04–0.05m/s, settled vx precision reached ~0.96, and fall rate was low. PPO gradients were finite, no NaN gradient skips were reported, and the value estimator was healthy. This does not rule out more subtle bugs, but rules out a blanket claim that the core optimiser cannot learn.

The most strongly supported failure is **the training distribution after foundation**: the newer system did **not** implement the working main's mature locomotion curriculum. It kept low foundation gait settings and replaced broad high-speed rehearsal with sparse, fragmented boundary challenges. Terrain specialist labels were added to random commands and regimes rather than generating coherent supported *terrain tasks*. A weak, sometimes inaccessible terrain reward then had little leverage against whole-episode survival and unrelated disturbances. The physical frontier governor repeatedly backed down to near-zero terrain challenges while push/yaw command target numbers expanded.

These findings identify **design regressions**, not a definitive single causal bug: comparisons between new 70M actors and published 260M Pinnacle are currently **age-, schema-, and test-incompatible**. The reported ~1m/s main behaviour has not been measured in the same physical examination as the 70M candidate. Run a same-physics common exam before quantifying performance ratios.

## Source-level causes, ordered by confidence

### A. Missing main-style maturation — CONFIRMED CONFIG REGRESSION

| Setting | Main mature_config | New unified one_run_config through 70M |
|---|---:|---:|
| Ordinary +vx practice envelope | 0.60m/s | 0.40m/s |
| Reverse envelope | −0.20m/s | −0.12m/s |
| Sideways envelope | ±0.22m/s | ±0.14m/s |
| Yaw envelope | ±1.50rad/s | ±0.90rad/s |
| Phase-speed gain | 1.25 | 0.75 |
| Base difficulty rate | 0.0024 | 0.0014 |
| Athletic rehearsals | 10% | **0%** |
| Observation-normalizer freeze | from 25M | never (0) |

The unified specialist vx frontier may grow beyond the ordinary maximum, so 0.40 is **not** an absolute prohibition on all forward requests. But most ordinary STEADY/TRANSITION/RESILIENCE training births use the reduced default envelope and low-gain gait-phase setting. The policy seldom experiences persistent, appropriately paced faster walking. A bigger *governor report number* does not make the ordinary policy adept at that speed.

Check `main:emu/training_config.py:mature_config`, `ab29c027:emu/run.py:one_run_config`, `emu/environment.py:_command_envelopes` and `command_phase_rate`.

At 50M, the governor advertised +0.279m/s but average actual vx was only +0.0636m/s against +0.0747m/s average command. In ~20–50M, the common learner logged **801** command-specialist completed episodes but only **35** forward near-edge completed probes and **9** successes. No amount of large promotion percentage replaces missing training exposure.

### B. Step/slope specialists inherited incompatible tasks — CONFIRMED CODE PATH

`environment.py:_specialist_task` explicitly changes command focus to STEADY but does **not** make STEP or SLOPE worlds consistently forward-crossing, heading-held, or steady. It leaves their sampled regime/commands and applies contact geometry separately in `_unified_terrain`.

At 50M the regime allocation was STEADY **28.3%**, TRANSITION **21.3%**, RESILIENCE **50.4%**. Therefore many STEP/SLOPE-labelled worlds were also random transition/resilience trials, while the step-crossing criterion specifically needs sustained forward root travel beyond the lip and supported far-side contacts. The additional terrains can independently overlap.

Fixing teacher *labels* or increasing their allocation cannot solve task/metric mismatch. A valid primary STEP lesson should repeatedly request signed forward travel toward a real lip, near-zero yaw, with geometry and sensors visible, and reasonable disturbances; later mix in lateral, turning, pushes and compound inclines.

### C. Terrain reward and graduation criteria are extremely sparse — CONFIRMED CODE + OBSERVATION

`_terrain_ascent_guidance` only pays on ascent with **operator vx ≥0.08m/s**, small yaw/lateral requests and a near-lip supported forward motion. The young command frontier was seeded at **0.06m/s** and its ordinary average command was ~0.04–0.05m/s until well into growth. Much of the early step practice therefore had **no route to the local progress reward**. The recorded total ascent-guidance contribution remained ~0.0001–0.0006 per frame during much of the shared learner's growth versus ordinary reward ~3/frame.

A separately paid **+10 once** on physical first crossing is positive, but very rare in the 65,536 physical frames per update. The teacher's `terrain_episode_success` additionally requires the actor to **survive until episode timeout**, even if the supported far-side crossing happened earlier. This is defensible as a *graduation exam* but a poor sole local teaching signal. Separately record (1) approach and foot placement, (2) verified supported lip crossing, (3) post-cross survival and (4) command obedience. Do not fake step success from a foot lift or edge touch.

The shared 50M lesson began at 4.0mm and ended 1.5mm. Every 70M fork finished at 1.5mm. Balanced OFF final-20M supported episode success: near up **9/57**, near down **22/69**; the terrain-focused fork only up **17/84**, down **18/77**.

### D. Aggressive governor repeatedly destroyed terrain practice — CONFIRMED

The four-direction slope teacher promotes a single slope ceiling only when **all four** orientations each supply four near-grade completed episodes with ≥65% stable movement. One failing easier-direction cohort can reduce the **global** grade by 12% and clear every directional evidence window. One failed easy-step cohort can reduce the single shared step target by **18%** and similarly clear up/down data. Because easy failures were frequent, targets were hammered down rather than maintaining a gradient of instructive challenges.

Across the shared learner slope went as high as 1.01° but ended **0.35°**; all but two forks also ended 0.35°. At the same time, very low near-height steps got **harder** to finish: the balanced branch showed easy-step outcomes around 23–28%, not better despite the 1.5mm target. This is consistent with task interference and/or poorly aligned reward, not merely promotion steps being too small.

Retain a certified frontier **AND** a separate exploration distribution. Don't let an unproductive certification batch remove the robot's chance to practise a visible 3mm–1cm lip or 1–3° grade. Use sustained, severity-aware failure/recovery statistics and per-direction evidence; retain a stable easy-band anchor and nonzero stretch probes.

### E. Resilience practice overwhelmed the maturing terrain curriculum — CONFIRMED EARLY SATURATION; INTERFERENCE CAUSALITY UNTESTED

The new push strength reached its **4N target by ~24M**, 4M after the foundation, without a physical force-conditioned competence test; at 50M half the ordinary regime allocation was RESILIENCE. Since STEP/SLOPE specialists inherit randomly sampled regimes, some are also RESILIENCE births. Their task resilience is softened, not necessarily removed, so real pushes and disturbed initial states can accompany their first attempts at a small step.

Simultaneous telemetry:
- ~20.3M: fall **2.9%**, 1.03N frontier, 4mm steps, 0.7° slope.
- ~24.2M: fall **15.3%**, 4N frontier, 3.3mm steps.
- ~40.0M: fall **28.9%**, 4N frontier, 1.5mm steps, 0.69° slope.
- 50M: fall **22.6%**, 4N frontier, 1.5mm steps, 0.35° slope.

The timing is consistent with harmful interference but does **not** by itself prove pushes caused the regression. Isolate matched no-push/small-push vs high-push terrain forks before assigning causation. Reward/command success should never be conflated with mere requested push strength.

### F. Newly added 360 observation schema — PLAUSIBLE SECONDARY FACTOR, NOT CONVICTED

Main's deployed actor uses 232 history inputs; the new actor uses 360, including 4×4 simulated ToF and delayed optical flow. The fundamental six-action PPO and old reward structure were otherwise mostly intact. Normalizer treatment and visual-footprint/occlusion need examination. The 20M foundation's healthy accuracy argues against a completely broken actor/sensor pipeline; it does not prove the new ToF is informative enough to resolve step geometry. Before blaming the network size or sensors, use a **matched 232-vs-360 flat mature recipe** and record real sensor visibility when the physical lip is 25–40cm ahead.

### G. Training-age confound — CONFIRMED LIMITATION

Every tournament child was just **70M steps old**, sharing one 50M ancestor. The earlier successful main champion was **260M** steps old, with different observation contract and curriculum. The tournament used 290M **aggregate training**, not a 290M-deep candidate. Current 0.73 fixed generalist scores should NOT be compared with historical main/Pinnacle metrics as if they share exam schema and physics. The public demo's ~1m/s is a user-observed behaviour until a common physical trace quantifies it.

## Controlled diagnosis before another long GPU campaign

1. **Read-only frozen exam FIRST:** On existing 70M checkpoints, test flat speeds 0.04/0.1/0.2/0.4/0.6/0.8/1.0m/s; zero- and nonzero-vy/yaw commands; 0/1/2/3/5° inclines in all directions; 0/3/5/10/20mm up/down, with source-appropriate observations. Log actual signed root progress, support contacts, foot clearance, command MAE, survival, real applied force and J/m. Compare the pinned public Pinnacle on identical physics using its genuine 232 inputs. No teacher metric counts as independent mastery.
2. **Minimal main-parity treatment:** Start from the working main's 20M flat birth, then deliberately **restore mature main command/phase/difficulty/athletic settings** in the new 360 learner, without 4N pushes or compulsory high-density terrain. Retain source-pin/new branch. At age 40/60/80M examine speed and falls against matched 360 control on unchanged foundation settings. This separates lack of mature speed teaching from exotic terrain/sensor issues.
3. **One coherent terrain intervention:** Keep STEP and SLOPE worlds in purposeful near-straight supported movement episodes: forward vx meaningfully positive, yaw held, low initial disturbances, visible ledge and enough time to cross. Keep all six lessons alive at low exposure, but emphasise a **single primary competency** per focus. Sample both easy and above-frontier heights/grades rather than annihilating the distribution after failed certification. Use confirmed far-side touchdown and separate survival outcome.
4. **One force-isolation intervention:** Cross the coherent terrain lesson with low vs fast-growing push severity; certify recovery at specified real forces. If no-push terrain learns and 4N terrain doesn't, the interference mechanism is supported rather than assumed.
5. **Controller trial only after walking catches up:** OFF vs 400ms/2× velocity-I at identical commands/terrain, raw operator tracking as score. Sensor body-frame estimate must be converted to the heading frame when assessing slopes/cambers; multiplicative gain cannot cancel lateral drift when original vy=0.
6. **Only then recombine and attempt an uninterrupted 400M cold run.** Preserve first 20M repeatability and the original main policy. Do not make an unreviewed single gigantic learner which mixes curriculum, reward, sensors, actuator changes and controller tuning.

## Acceptance signals for claiming repair

- At 40–80M, **achieved signed velocity** improves in the held-out 0.2/0.4/0.6m/s bins; a higher frontier number alone does not count.
- Real step-up at ≥3/5/10mm and four-direction slopes ≥1/2/3° show monotonic improvement on identical fixed physical cases and the robot does not merely survive by drifting downhill.
- Recoveries conditioned on actual 0/1/2/4N pushes improve without sacrificing flat basic gait.
- Reward/teacher reports distinguish supported crossing vs episode survival; no unsupported geometry/foot-lift claims.
- At least one matched control demonstrates the causal effect of each proposed change; avoid premature twelve-arm speculation.

**Do not edit protected main, release policy, or source-pinned results checkout without explicit owner permission.**
