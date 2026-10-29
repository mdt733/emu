# v5030 reward laboratory — post-v5029 proposed experiments

**Design only; do not interrupt or patch the live v5029 120M training SHA.**
Prototype arithmetic lives in `emu/rewards.py` and is **not wired** into
`training_reward`, teacher governance, actor observations or PPO.
This document selects hypotheses, not a winner. Final launch variants
must be chosen from the pending v5029 evidence.

## Why this is a reward study and not yet another curriculum rewrite

V5028's slow speed was largely a real exposure/finite-floor problem:
the 4m-half-length flat slab and suppressed fast birth commands made an
honest sustained 1m/s walk impractical. V5029 fixes both and puts a
measured 25% fast-flat curriculum alongside the six terrain specialists.
**Only after its speed ladder and physical matrix complete** can we
decide whether further reward shaping is useful. Do not compare v5028
and v5029 reward totals as if their physical worlds or command sampling
were identical.

Important: `vx=[-...,+...]` in the terminal is a **teacher target
frontier**, not a measured maximal speed. The new `cmd_vx_over_03_frac`,
`actual_vx_at_cmd_over_03` and frozen measured speed ladder are stronger
evidence.

## Candidate questions and small matched studies

| Arm | Forward reward / step transfer | Question |
|---|---|---|
| **A — control** | Unchanged v5029 reward (including careful near-lip ascent) | What can the fixed fast/terrain curriculum achieve unmodified? |
| **B — half-second velocity** | 50% instantaneous forward tracking + 50% causal 25-frame forward tracking, preserving instant lateral/yaw/pitch, contact, fall and effort terms | Does removing gait-phase speed noise improve real pace and recovery without cheating? |
| **C — supported step transfer** | B plus verified signed CoG/ground-level transfer; use a **fixed reallocated existing step guidance budget** rather than unbounded additive rewards | Can the robot make more supported 2.5–3.5cm crossings without encouraging jumps/squats or sacrificing speed? |
| **D — optional lip allowance** | B, with brief, one-shot 0.5–0.8s forward underspeed relief only near a verified real lip and while maintaining support; no extra velocity debt repayment | Does deliberate foothold preparation need extra temporal slack beyond rolling tracking? |

Do **not** automatically run all four: start with **A/B** (small
matched diagnostic continuation or fresh common 20M foundation under one
new source contract), select C *or* D based on actual failures. Avoid a
four-arm 400M-step blind search. Test each proposed source/shape in CPU
MJX and 2048 CUDA before a full overnight run.

### B: smoothing mechanics (offline prototype already coded)

20 ms control means **25 frames ≈ 0.5 s**. Compare a causal
mean of **achieved** and **requested** forward velocity, not an average
of tracking reward samples. For each episode and uninterrupted command
segment:

```
v25(t) = mean(actual_vx[t-24:t+1])
c25(t) = mean(requested_vx[t-24:t+1])
F(e) = exp(-(62 e² + 120 max(e-0.025,0)²
             + 40 max(-e-0.025,0)²))
forward_B = 0.5 F(actual_vx - requested_vx)
          + 0.5 F(v25 - c25)
```

The helper uses the actual `F` coefficients already in rewards;
the expression above is *forward-axis tracking only*, not the full
locomotion payment (which still multiplies all-axis tracking, upright
height, contact and gait quality).

**Must reset on episode birth and velocity-command discontinuity.**
Do not average between worlds, across falls, or through unrelated
transition commands. Maintain the existing instantaneous control
tracking for starts, stops, reversals and commanded turns unless a
later dedicated experiment tests those too. No new actor observation.
Adding reward-only 25-frame state to actual MJX training changes the
training state/checkpoint contract; cannot silently resume old
checkpoints without an explicitly accepted same-source fork.

Smoothing alone **does not permit a 0.3-second standstill** at 0.6m/s:
the averaged error will still be substantial. Hence the separate
step-specific hypothesis D.

### C: physically verified step elevation, not rewarding squats

The current uphill lesson already has non-repeatable approach/progress
reward and a +10 physical crossing payment. An additional bonus must
**reallocate** some of that finite lesson budget, not stack endlessly.

```
signed_surface = direction * (target_support_level - source_support_level)
signed_cog     = direction * (current_cog_height - approach_cog_height)
eligible_m = min(clip(signed_surface,0,cap),
                 clip(signed_cog,0,cap))
if target_surface_contact AND stable_foot_support AND NOT terminal:
    new_peak = max(previous_peak, eligible_m)
    one_time_credit = budget * (new_peak - previous_peak)/cap
else:
    one_time_credit = 0
```

Up requires *higher actual* support; down requires a *lower actual*
landing **with stable contact**, not just free-fall. Keep a per-obstacle
high-water mark and a single crossing credit. Use contact pairs and
actual collider geometry; never feed those teacher-only facts to
the actor. No reward for torso height without corresponding genuine
terrain height, or for lifting/rocking the same foot repeatedly. The
pure `verified_support_elevation_credit` helper expresses this
anti-cheating contract; real PPO integration would also enforce
episode/obstacle lifecycle and cap all incremental payments.

### D: finite real-lip pause budget

Cap at 25–40 control frames (0.5–0.8s) **once per obstacle**, not
per proximity-frame or near-lip re-entry. Start only after real
approach-to-lip evidence; keep lateral, yaw, posture, overspeed,
bad-contact, stability and fall penalties unchanged. Smoothly restore
forward tracking when the walker has transferred support or its
budget expires. A body rocking indefinitely at the lip must run out
of allowance; standing on an ordinary flat floor never gets it.
Do not demand a dangerous sprint to 'repay' speed debt.

B and D partly overlap with the existing `coherent-careful`
120-frame narrow uphill relief: the D trial should **replace or
shorten** the old relief to keep total forgiveness controlled,
not layer two independent free-speed windows.

## Decision gate: choose after v5029, not beforehand

Use its exact-source `overnight_results.zip`,
`physical_exams/speed_ladder/unified_exam.json` and
`physical_exams/terrain_matrix/unified_exam.json`.

- **Speed teaching still absent**: if fast-gait occupancy or >.3m/s
  requested share is missing, first fix birth sampling/config—not reward.
- **Fast requests present, measured speed poor**: prioritise A/B and
  examine fixed speed ladder, first-fall location and torque/shaft work.
  The exponential error curve may then merit a separate *bounded*
  curvature ablation; don't bundle it with smoothing by default.
- **Speed tracks well but 2.5–3.5cm supported crossings lag**:
  prioritise A/C, or A/D if traces show careful foot placement is
  aborted by instantaneous underspeed pressure. Include both UP and
  DOWN, lateral drift and post-step speed recovery.
- **Both fast speed and terrain good**: avoid experimental changes
  until a second seed or matched exam reveals an actual deficiency.

At every branch record:
actual requested and achieved vx by band (.3/.5/.8/1.0m/s),
per-speed survival/support, recovery to commanded pace after a lip,
supported step-crossing fraction by height/direction,
heading/yaw error, downhill stability, falls, actuator J/m.
New-world A/B exams must share exact model/observations, seeds,
requested velocities, stop horizons and evaluator SHA; no cross-schema
source score substitution. Treat uncertainty from 8 or 16 worlds
seriously; absence of a speed or step pass is not automatic failure
without opportunities.

## Implementation scope deliberately postponed

The following are **not** enabled on any training branch yet:
25-frame reward-state ring, per-step surface-reference state, budget
lifecycle, new actor observation, changed reward weights, new PPO arm
launcher, or checkpoint migration. Once the v5029 outcomes select the
best small A/B, build opt-in switches with the baseline bitwise
unchanged, a source-pinned same-config pair and tiny CPU/GPU acceptance
before committing to a large GPU campaign.

Protected reference: `main`.
Protected completed study: `feature/v5028-considered-steps`.
Running experiment: `feature/v5029-speed-restoration` at
`8288dc0b02c2b97f9a915cd14c0b65958e6b77c4`.
Housekeeping and prototypes only: `feature/v5030-reward-lab`.
