# EMU v5023 — 10-hour overnight GPU campaign proposal

**Status: experimental design, NOT yet an executable launcher or approved model promotion.** Written 2026-10-24 using the new governor policy from source `411b841`, the 0.10–0.30m/s frozen speed-sweep console logs, and uploaded `results_bundle(8).zip`. The per-world ZIP of speed sweep is still absent; inspect it BEFORE freezing new teaching rules. All source work stays off `main`; PIN the existing `~/emu-v5023-window` / `411b841` source and all existing checkpoints.

## Objective and hypotheses

**Both, not either-or.** Spend a well-controlled fraction of overnight compute (A) extending the unchanged 34M network with the old governor as a reference, AND (B) building a new train-from-flat, *speed-conditioned physical step-up curriculum* to teach the actor to merge velocity, ToF, foot position, stance and placement. A claimed win must cross raised contact geometry *without excessive falls, edge-skirting, lost command tracking, or a 25% increase in shaft work per travelled metre*.

The frozen `411b841` actor at command +0.30m/s passed **all 8/8** paired 1cm-ceiling cases on seed1701 (including uphill steps of actual 0.87 and 0.95cm), vs 6/8 overall at +0.15m/s. Across two seeds and all courses, supported overall crossings improved 24/48 at +0.15 to 28/48 at +0.25 and +0.30; falls also rose in 2/4cm cases. This proves *evaluation approach speed matters*, not yet that the network is using an anticipatory ToF response or intentionally clearing the foot.

## Empirical GPU budget, not a fantasy target

Machine: RTX 4060 Ti 8GB in WSL, measured 2,048 MJX parallel worlds. Same run source `411b841` measured **13,959,168** incremental steps from 20,054,016 to 34,013,184. GPU PPO update time **448.90s**, effective training-only **31,096 env-steps/second** (similar older run 31.5k); independent final fixed evaluation ~**418.65s**, final capability-cloud exam **104.28s**, plus compilation/other overhead. Final dashboard `steps_per_sec` ~14k reflected the long final exam; it is NOT sustained PPO FPS. Fixed eval can recur about each **60M** steps; cloud about **40M** steps; new/forked process pays additional startup and sometimes compilation.

- Arithmetic ceiling: 31.1k × 36,000s ≈ **1.12 billion** environment steps *if continuously doing PPO without a single evaluation or restart*. **Never promise this as a 10h campaign output.**
- Realistic mixed-workload design budget: **~500–700M incremental world steps** spread among several runs, allowing 7–10min each final fixed+cloud exam, intermediate checks, compilation, disk, CPU physical exams, and contingency. The below BASE sequence is **616M steps**, ~5.5h pure PPO plus ~2.5–3.5h overhead. The extra 80–100M is **optional**, not a requirement.
- Live throttle: compare total actual wall FPS (not train FPS), eval duration and GPU memory after first long block; keep >= **45–60min contingency**. Never schedule two simultaneous 2,048-env PPO jobs on an 8GB GPU; run sequentially, WSL GPU sanity check first.

## Candidate overnight timetable (incremental steps, *not* cumulative step quotas)

| Priority / arm | Parent -> cumulative target | Incremental env-steps | Rough wall allocation | Reason and decision |
| --- | --- | ---: | ---: | --- |
| 1. **Control: unchanged source** | `411b841` archived **34→110M** | 76M | ~1.0–1.3h | Test the user's 'maybe only more time' hypothesis with original teacher, preserving independent baseline |
| 2. **Common foundation** | NEW source, cold **0→20M** | 20M | ~0.35–0.5h | Flat-only actor / optimizer / normalizer shared by all NEW-source teaching arms |
| 3. **Arm A: speed × terrain coverage** | shared 20→**100M** | 80M | ~1.0–1.3h | Stratified uphill approach-speed training, held-out evaluations, unchanged intermediate foothold reward |
| 4. **Arm B: +supported touchdown** | same shared 20→**100M** | 80M | ~1.0–1.3h | Exactly A's sampling/governor, plus anti-farming upper-foot contact/transfer milestone |
| 5. **Top performer grows** | winner 100→**260M** | 160M | ~1.9–2.3h | Confirm skill increases with experience and inspect 160/260M snapshots |
| 6. **Second performer follow-up** | other arm 100→**180M** | 80M | ~1.0–1.2h | Check that fast early progress wasn't a lucky short-training result |
| 7. **Best performer extended** | winner 260→**380M** | 120M | ~1.4–1.8h | Longer slow improvement or learning plateau; stop if quality regresses |
| **Base total** | one control + one foundation + two arms + continuation | **616M** | **~7.7–9.7h** | Budget *not* a deadline guarantee |
| **Optional stretch** | winner or promising alternative +80M | **80M** | ~0.9–1.3h | Only if time, health and independent evidence justify |

At ~31.1k update FPS, 616M PPO-only steps would take ~5.50h; the wall allocations above are deliberately wider because each fork can pay a final 7min fixed eval, 1.7min capability-cloud exam, periodic exams, and compilation. Exact timing MUST be calibrated after the first long block. If slower than forecast, **drop #6 and #7 before truncating the comparative A/B**. Stop scheduling new work once <=60min remains, or if thermal/VRAM/finite-PPO health fails.

## NEW-source implementation doctrine, before any overnight launch

**A/B must share one exact new Git commit and one identical 20M flat foundation**. Implement a small explicit experiment enum/profile in existing `training_config.py` / `environment.py` / `curriculum.py`, not yet another trainer, and capture it in saved run configs. `Arm A` and `Arm B` differ by ONE toggle: the touchdown teaching payment. The new cohort law, direction balance, difficulty gates and ordinary PPO are identical.

1. **Training speed + obstacle sampling:** on actual exposed *up-step births*, stratify command vx into **0.10–0.15**, **0.16–0.22**, **0.23–0.30m/s** bins with enough examples in each; 10–15% terrain births initially, capped conservatively at ~20–25% while core walking is healthy. Guarantee matched fast **flat** births, so speed is not a privileged obstacle label. Keep some slow ascents: solve a usable skill *envelope*, not only running starts. Full vx/vy/yaw, posture, resilience and transitions remain in the mix. Do not silently assign terrain only to easy STEADY regime unless expressly tested.
2. **Truthful success and teacher:** only actual MJX ball-foot/platform/floor contacts and sustained support count as climbing. Arm B adds bounded *once per episode* teacher bonus for first supported upper-foot touchdown near the edge, while the other foot is in real stance and the root has healthy forward progress. No replay rewards for stationary foot tapping, body clipping, repeated rocking or successful falling. Do NOT feed teacher ground geometry or exact contact success into the 360-input actor; its 4×4 ToF raw ranges+validity, existing delayed foot/joint/IMU and vx request remain actor-only input.
3. **Curriculum evidence:** persist completed attempt/success/falls by direction **and velocity band** across PPO rollouts; avoid the proven one-rollout starvation. Practice exposure and height are different controls. Begin with actual ~0.3–1.0cm heights; grow *real* height conservatively but not by an unreasonably tiny 0.025cm after an enormous 25% exposure gate. Require supported UP mastery at relevant speeds and DOWN safety; local retreat when speed/height cells fail. Avoid growth on only fast successes hiding slow-step inability.
4. **No loss of deployability:** no learned gait generator, privileged Foothold ID, new actor inputs, raw ground-truth root velocity or unreal sensors. Keep robot XML, 360 observation order/normalizer, 6-action actuator layout, PPO optim, parameter shapes and safety. If any of these change, it is a separately labelled research arm with incompatible checkpoints.
5. **Provenance:** NEW experiment from scratch because repository SHA differs. Only after exact review could explicit migration be considered; by default code's `inspect_run` rejects cross-SHA resume. Existing `411b841` **34→110M** control must run **using the ORIGINAL pinned `~/emu-v5023-window` checkout**, NOT new code. Every A/B continuation must have compatible saved source and config; each phase has its own unique run directory.

## Independent scoring, physical gates, and branches

At shared-foundation and 100/180/260/380M decision points save frozen `.emupolicy` plus validated `.pkl` lineage. Read **real physical performance**, not PPO scalar reward. For each contender: evaluate seed1701/1702 for historic comparability, PLUS precommitted unseen seeds such as **2417 and 3011**; command .10/.15/.20/.25/.30m/s, `--worlds 8` or up to 16/32 for better sample, 0/1/2/4cm ceilings *but report actual heights not ceiling names*. Full horizon 350 frames, flat and dormant topology parity, confirmed ≥100ms far-side contact and full survival. If cost is material, screen with 8 worlds/two seeds then expand winner exam to 16–32 and unseen seeds.

**Preserve four evidence layers**: physical crossing (UP and DOWN separately, height/speed bins); safety (falls, tilt, foot support, lateral escapes); command quality (actual velocity at lip and flat tracking, phase timing); mechanics (shaft-work per forward metre, servo saturation, landing impulse, ToF valid-ray range, and the first true upper-foot touchdown). Do not reward simply reaching far-side root position without foot support. Broad core gait/resilience evaluator remains a mandatory gate.

Default select a contender only if it improves *upward crossing across multiple speed/height bins* without significant down-safety/fall regression, retains ~95–100% of old broad core walking quality, and does not degrade shaft work per metre beyond an explicit justified tradeoff. Otherwise prefer the control or spend next slot on diagnosis. Never auto-promote `main`/champion/showcase; compare independently and request owner approval.

### Escalation / alternative arms, mundane to long shots

**Low risk / high signal:** straight 34→110M old control; terrain-exposure dose 10/15/20% with preserved flat matching; speed-balanced births vs terrain-independent sampling; near-lip speed / foot-contact event tracing; assessment of 0.95cm edge where the foot sphere radius is 1.8cm; fixed exam command pace .10–.30 so 'faster' does not just earn more distance in 7sec.

**Medium risk:** first-upper-foothold touchdown reward (Arm B); heel/ball clearance relative to step top; tie cadence to speed *after* actual commands are set, compare with legacy sampled phase; randomize approach phase at birth and step-edge yaw while preserving paired physical terrain; speed-specific curriculum gates; gentle friction/servo asymmetry after first reliable 1cm ups; mild shaft-work conditional-on-progress objective for efficiency.

**High risk / isolate in new source/arm:** temporarily attenuate `phase_sin/cos` input or phase-sync reward near detected obstacles so actor can leave a rhythmic gait without a forced new trajectory; short teacher-guided pretraining for an actual supported first foothold followed by normal PPO (must retain no teacher leakage); alternative physical depth-sensor angle/geometry if v2 4x4 rays fail to see the lip early enough (new sensor contract; cold-start); small recurrent policy only if sensor/partial-observability diagnosis shows an unrecoverable need for memory. Do not throw these together at night as a confounded soup.

## Launch readiness / operator protocol

- [x] Establish baseline and speed hypothesis from frozen policy / JSON; speed-sweep aggregate supports faster 1cm course. **Per-world speed-sweep ZIP still needs inspection.**
- [x] Estimate throughput from `results_bundle(8).zip`, including ~9min final exam overhead.
- [ ] Implement/configure and review new A/B experiment toggles on feature branch, document source SHAs, verify CPU source-level run/config and CUDA doctor.
- [ ] Verify governor window, direction/speed bin denominators, first-foot-on-top events, anti-reward-farming and sensor visibility **in code** before 20M cold run. Avoid sprawling test collections.
- [ ] Verify exact command lines and dry-run `--plan`; ensure clean checkout, 2,048-world GPU budget, sufficient free disk, checksums and unique output directories.
- [ ] Write a **fail-closed sequential overnight shell orchestration script** with output manifest and log tail; it must not choose random winners or overwrite checkpoints, must stop scheduling new jobs at wall budget, gracefully preserve valid completed stages, and have reproducible manual resume instructions. Do not claim it exists until committed.
- [ ] Record every run's wall duration, source, parent step/hash, policy hash, fixed exams and failing cases; select continuation at explicit success gates, not reward alone.
- [ ] No `main` merge or champion update without explicit owner approval.

**Handover:** latest `HANDOVER.md` is authoritative for exact next state. This document is a proposal awaiting implementation and validation; do NOT give a copy-paste overnight launch command until runnable.
