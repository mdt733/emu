# EMU v5039 — Integrated Step Academy design contract

Status: **DESIGN ONLY; NOT IMPLEMENTED.** Preserve frozen v5033/v5037 source commits, parent checkpoints and main. The eight Stage A arms used real single-lip contacts, not a continuous staircase.

## Non-negotiable architecture

One six-servo physical robot, one terrain_360 observation contract, one actor/critic, one optimizer, one observation normalizer, one adaptive capability governor and one resumable source-pinned lineage. Academy is a physical episode/teacher role, not a second PPO trainer or model. Actor receives the existing real sensor/proprioceptive history; true geometry/contact remains reward/teacher-only. A future policy must train from zero without any dependency on the current 280M checkpoint.

## What already exists

- Real two-slab single-lip MJX contact, actual 4×4 ToF, six capability birth cohorts, and supported far-side crossing detector.
- Single-lip ascent progress, raised-foot and optional verified upper-support shaping, bounded once per episode.
- Stage A eight experimental continuations from a matched full-state parent, including ±3-degree obliqueness, 85% STEP allocation and 300-frame rapid repeat.
- Independent historical single-lip exams and Stage A reporting/evaluation tooling.

## Unfinished before training in a dedicated continuous-step world

1. **Real multi-lip physics**: build four physical lips first, then six after profiling. Fixed MJX model topology with multiple collidable slabs/mocap bodies; per-world pose, height and edge angle sampled on episode reset. Real up-up, up-down, down-up and mixed-level courses. Geometry cannot overlap, leave collision holes, create phantom contacts or prevent the generalist's flat worlds. Benchmark memory/performance at 2048 envs on RTX 4060 Ti 8GB. Preserve original two-slab exams and model.

2. **Per-lip state machine**: next lip index, signed ascent/descent, edge normal, near/contact/clearance/verified support/weight transfer/crossed/failed states, independent evidence counters. Current code uses world-index parity to infer step direction and one terrain_crossed latch for the entire episode: neither supports mixed up/down lips or repeated rewards correctly. Do not double pay repeated foot taps or backwards recrossings.

3. **Sensor fidelity on new geometry**: verify all 16 ToF rays measure the same physical surfaces as feet and torso; ensure approaching both oblique and consecutive lips produces anticipatory range readings despite latency, occlusion, dropout and pitch. Keep the 360-input actor schema exactly unchanged; no target-lip privileged observation.

4. **Conserved milestone reward**: controlled comparison with original reward, then per-lip one-time grounded clearance, true upper contact, supported body weight-transfer and independent far-side success. Retain the existing reward total rather than simply inflating points. Reject stationary toe-tapping, leaning into lip, airborne CoG lift and repeated rocking as successes.

5. **Correct episode scheduling**: assign Academy at world birth within the SAME six-role sampler. Only Academy births see repeated lips; ordinary command, slope, recovery, transition, control and fast-retention practice stay meaningful. Stage A's shortened 300-frame max affects all worlds and must not be copied into the integrated design. Instead use per-world/course end conditions while preserving PPO shapes/horizon.

6. **Fixed physical examiner**: unchanged old one-lip physical suite plus independent matched multi-lip sequences. Per-height and per-lip uphill/downhill completed attempts, real supported foothold, weight transfer, speed and pauses, contact/torque saturation, sensor-to-action response, and retention in vx/vy/yaw/slope/recovery. Distinguish frontier data from held-out 2/2.5/3/3.5/4cm success; independent seeds.

7. **Turnkey launcher + automatic report**: clean source SHA and same-import worktree checks, immutable full-state parent if continuing; exact manifest, restartable partial checkpoints, validation of completed arms, auto physical exams, per-arm and campaign Markdown/CSV/JSON/ZIP even if an exam fails. Never end an expensive campaign with only a note to run exams later.

## Cold-start integration schedule (proposed, tunable)

- 0–20M: existing reliable foundation; Academy physical topology present but dormant, all surfaces are flat or inactive. No split trainer.
- 20–60M: existing general capability growth; ordinary single lips continue at teacher-controlled severities.
- 60–80M: optionally start 5–15% EASY multi-lip nursery episodes without starving retention. Heights remain evidence-gated; a time threshold should never imply competence.
- 80–100M: gradual ramp of Academy birth share toward 50%; never instantaneously seize half the GPU worlds. Keep the remaining 50% for command, resilience, slope, transition, control and explicit faster-gait rehearsal.
- 100M onward: test 50% Academy/50% generalist balance; reduce Academy allocation adaptively if matched physical retention drops. Heights and course complexity move only after genuine supported crossing success.
- A 50% birth allocation means approximately 1024 of 2048 worlds in expectation, not separate optimizer updates, a separate policy or a fixed mask. Assign on episode births, avoid synchronized resets.
- Academy can be disabled in a clean from-scratch run, reproducing the previous curriculum with compatible source and explicit run identity.
- These are suggested default study settings, NOT hard-coded doctrine. Tune start age, ramp duration, share, early share, lip count, ±3-degree angle and weight-transfer reward recipe through named source-pinned config fields.

## Critical proof gates before long training

**P Physics:** multi-step real contact, ToF visibility and heterogeneous up/down direction correct on CPU and GPU; 2048-world peak GPU memory and throughput viable.

**E Event/reward:** two real uphill lips scored twice, one failed lip not scored as crossing, stepping back cannot farm reward, signed direction and per-lip reset are correct.

**L Learnability:** initial matched 5–10M continuations from one source-pinned PPO checkpoint with old two-slab control, multi-lip repeats, and verified rewards. Improvement measured at a FIXED held-out 3cm uphill step, not only a larger curriculum frontier.

**I Integration:** a new cold-0M smoke traverses foundation to dynamic birth-role schedules under a simulated age trigger at 60M/80M/100M; no optimizer fork, hidden sensor or schema change, reproducible config and resume.

**R Report:** each experiment always writes measured results and report ZIP and retains a clear distinction between trained, tested, untested and failed. No report-less completion message.

## Work order and release discipline

Build and verify multi-lip physical course plus event state first. Integrate the existing sensor with it, add the evidence and reward ledger, then the fixed examiner, then short matched GPU runs, then the cold-start governor schedule. Do not merge any unfinished Academy work into main or disturb the frozen source of completed 280–290M checkpoints. The dedicated Academy experiment is a research mode of the same architecture, not a permanent fork of locomotion development.
