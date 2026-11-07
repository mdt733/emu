# v5055: diagnose before another cold-start campaign

Status: **instrumentation only**, not an improved trainer and not a
go-ahead for a 160M cold start. Research parent: v5054
`74d04d24473dc6d216001b7eac7c835f5e53e3c8`. Main is untouched.

## Reproduced evidence

The independently measured seed-1701 2,048-world stair matrix contains
2,000 uphill trials at exact heights 2.5–50 mm, 100 trials per height.
The old v5052 80,019,456-step actor and v5054 182,059,008-step actor:
- Each foot independently on the target prism top for five frames:
  **1,348/2,000 → 1,096/2,000**.
- Controlled five-frame downstream traversal:
  **531/2,000 → 421/2,000**.
- Strict five-frame parking: **13/2,000 → 1/2,000**.
- Controlled traversal at 32.5–40 mm: **70/400 → 11/400**.
- Controlled traversal at 2.5–10 mm: **193/400 → 221/400**.
This is regression on harder heights, not zero climbing ability. The
original 80M result schema is v2 and the 182M result schema v3; compare
only common metric definitions. The 2,048-world zero-height matched-key
flat control and a full downhill matrix are **still missing**.

The 182M metrics bundle (157 update rows, 80.67M–182.06M) records
**899** completed short-climb events, **358/2,440** near-uphill successes,
**0/2,413** near-downhill successes, and a final Academy up/down
frontier of approximately **2/114 mm**. It reports zero PPO KL-stop
events in those rows. This does NOT prove a healthy learning objective
or adequate actuator/perception capability.

## Additional observed control regression: one-sided hip-roll saturation

Reanalysis of the complete 182M v5054 metrics.csv finds that the sampled
**left hip-roll** action (joint 3, 0-indexed) clipped against [-1,1] at
about **19.4%** across the first four recorded intervals and **67.2%**
across the last four; the right hip-roll action (joint 0) increased only
about **10.5% → 13.3%**. The 0.25m/s held-out generalist broad score
declined **0.672 → 0.446** during the same continuation. This is a
correlation, NOT proof of causal lane-drift or a motor torque metric.
The new read-only keys `ppo_hip_roll_clip_{left,right}_{academy,other}`
and `ppo_hip_roll_raw_{left,right}_{academy,other}` separate Academy
sampling from generalist exposure and reveal the signed control bias.
If saturation exists only in Academy, prioritize reward/contact
incentives; if it spreads to other worlds, assess actor-wide control
authority and generalist forgetting.

## Perception limit worth checking, not an established failure

The raw 4x4 ToF range synthesizer adds Gaussian range noise of **2mm
sigma × sensor scale**. The actual sensor-scale frontier in the observed
182M bundle was **1.5**. The task sampler assigns that same frontier
to the birth sensor-scale field, yielding nominal **3mm sigma** unless
another task path overrides it.
The Academy's 2–3mm target lips are therefore near this noise scale
before geometric beam placement, body oscillation, occlusion and
sensor delay. That does NOT prove invisibility; a top-surface height
change maps to a beam-range difference depending on ray inclination.
A paired flat/positive-lip sensor audit at fixed pose, x distance,
body attitude and sample keys is needed before changing the sensor.

## Curriculum hazard: min lip threshold versus generated difficulty

The actual `make_profiles` first-lip amplitude contains factors
`0.90 * (0.75..1.0) * (0.95..1.5)` on the nominal frontier,
while the short-climb teacher counts a signed lip only if
`abs(birth_lip_m) > 1.5mm`. At the minimum 1.5mm frontier,
the first lip is **always below that threshold** for the first
global x=0→1 seam when slope is zero: at the exact 1.5mm minimum,
the maximum offset is
`1.5 * 0.90 * (0.95 + 0.55/14) + (1.5 * 0.10) = 1.486 mm`,
still smaller than the strict `>1.5mm` contact-classifier threshold.
The smooth slope cancels across the shared seam; the independent
height jitter has at most the shown 0.15mm difference.

A NumPy reproduction of the current deterministic source-hash profiles
over 200,000 random 31-bit terrain seeds found:
- 1.5mm nominal: **0.0%** classifiable first lips; 0.0% near;
- 2mm nominal: **63.1%** classifiable; 49.5% near;
- 3mm nominal: **100%** classifiable; 49.5% near;
- 4mm nominal: **100%** classifiable; 49.5% near.
Here "near" uses the existing physical actual/requested ratio
`0.78..1.05`. These are **physics-generator samples, not training
success rates**, and refer to the first +X seam with equal uphill/downhill
frontiers. A requested 3mm does NOT mean a realised 3mm lip.
The minimum reachable governor frontier is therefore a **deterministic
training dead zone**, not a safe resting difficulty for failed novices.
This must be corrected before a cold run, but changing it now would
contaminate the logging-only causal audit.

## Hypothesis A: task completion can have negative policy credit

Current PPO has `gamma=.99`, `GAE lambda=.95`. An Academy short climb
receives `+8` and an absorbing `done`; its transition bootstraps
`V(next)=0`. Meanwhile healthy locomotion produces persistent
per-frame rewards, including `+0.58` for being alive. Therefore a
nominally positive terminal bonus can have negative TD error:
`delta_terminal = r_terminal - V(current)`.

The **actual actor uses normalized batch advantages**, so inspecting
`delta` alone does not establish the direction of the policy update.
This branch adds seven low-cost scalar metrics extracted from the
already-computed GAE and original PPO advantage-normalization:
- `ppo_stair_credit_finish_events`: count (zero is a valid count);
- `ppo_stair_credit_finish_reward`: mean final transition reward;
- `ppo_stair_credit_finish_value`: mean pre-step value estimate;
- `ppo_stair_credit_finish_td`: mean `reward - value`;
- `ppo_stair_credit_finish_adv_raw`: mean exact raw GAE advantage;
- `ppo_stair_credit_finish_adv_normalised`: mean batch-normalised
  advantage (the optimiser's actual signal);
- `ppo_stair_credit_finish_negative_fraction`: fraction of success
  events whose *normalised* advantage is negative.

Every conditional metric is **-1 (unknown)** when its event count is
zero. The logger normally publishes only 1 of every 10 PPO updates: the
audit therefore accumulates event-WEIGHTED credit over EVERY rollout
seen by the current process, not just the current log tick. Keys begin
`ppo_stair_audit_`. The cumulative count and means are strictly
`since_process_start` (a mid-run process restart begins a new audit
window; it does not silently pretend to include prior rollouts).
This includes the **first** 32-frame GPU rollout; the metrics are evaluated
on the pre-update critic and policy, so later optimiser updates cannot
rewrite which action produced an earlier terminal. All metrics are summaries, not actor inputs and not new reward.
No trainer, sampler, physics, optimizer, architecture or sensor changes.

Separate from the direct per-rollout metrics, `ppo_stair_audit_*`
contains the cumulative, event-weighted evidence over all 32-frame
rollouts in the **current process**, including those between 10-update
CSV log intervals. Use `ppo_stair_audit_finish_events` and
`ppo_stair_audit_finish_adv_normalised` for the decision, rather than
a per-row event that may have only zero or one success.

The same audit publishes two *episode-ended* teacher gate counts,
`academy_episodes_first_lip_armed` and
`academy_episodes_two_foot_ready`, against
`academy_episodes` and `academy_short_climb_successes`. These
show whether the first-lip confirmation or independent-five-frame foot
latch is starved. It also reports the ACTUAL Academy mean requested and
achieved velocities (`academy_actual_cmd_vx_mps`,
`academy_actual_vx_mps`) to test the approach-speed mismatch. These
are post-step measurements only and change no actor behaviour.

### Stage 1: short **diagnostic** PPO fork

Fork the saved v5054 182M checkpoint into a NEW run directory using
cross-source hash guards, and collect roughly 4M more environment
steps (absolute target 186M). This is not a new treatment: reward,
policy actions, observation schema and physical teacher stay unchanged.
**Critical resume hygiene:** the original v5054 code would automatically
discard Academy/ordinary STEP evidence on any cross-source fork, even
when only logging changed. The v5055 branch now checks the actual
parent `training_config.json` for the same success/terrain/solver recipe
and preserves compatible evidence. It still resets when an older or
unverifiable success recipe is used. The parent's recovery.pkl is not
inherited across source commits, even for diagnostic-only changes, so
monitor rollback decisions separately. Run under same 2048-world, seed-735,
Academy, obstacle and reward configuration as v5054.

Use the existing `python -m emu.cli train` with
`--resume /path/to/v5054.../checkpoints/final.pkl`,
`--research-parent-source 74d04d24473dc6d216001b7eac7c835f5e53e3c8`
and `--research-parent-sha256 "$(sha256sum ... | awk '{print $1}')"`.
First execute the same command with `--check-resume-only`.
Never call the diagnostic source an improved policy.

Interpret credit only when enough finish events were sampled; report
the distribution across several rollouts rather than extrapolating a
single mean. If normalized PPO advantage is mostly negative, the
completion incentive is **demonstrably wrong** for current value
estimates. If mostly positive, the terminal-bonus hypothesis is NOT
the dominant explanation: audit the eligibility and physics gates next.

### Stage 2: frozen perception / lane controls (no PPO)

Use existing `scripts.v5053_matrix_exam` with identical seed=1701,
worlds=2048, horizon=350 and the same exported actor:
1. `--mode flat --trials-per-level 2000` (same indexed reset keys).
2. `--mode stairs` (100 trials x 20 exact uphill heights).
3. A separate downhill-focused physical cohort: existing balanced mode
   covers downhill, but it has fewer trials per height than the stair
   mode and **is not yet a paired high-powered downhill curve**.

Compare first-life physical foot contact, upstream and downstream
upright crossing, lane error and heading, **at the same commanded
0.25 m/s**, and subtract flat drift before blaming step geometry.
The actor has local rates and terrain ranges but no direct world-frame
goal bearing or persistent yaw reference. Do not change the actor
observation schema based on inference alone.

## Stage 3: decide one actual training change

Only after the diagnostic outcomes:
- If negative normalized credit dominates: A/B completion incentives
  and terminal objective accounting from a **common frozen 80M parent**.
- If flat and step lane drift match: test deployable yaw-hold/odometry
  rather than step-specific lane reward.
- If matching flat succeeds but stair lane fails: analyze contact,
  lateral impulse, servo torque saturation and ToF approach timing.
- If reward signs are fine and foothold rates stall: inspect the
  exact `_carpet_events` gate for first-tile lip confirmation versus
  independent target-top foot proof; the held-out examiner and teacher
  do **not** require identical intermediate events.

Do not change reward, terrain, curriculum, PPO and actor observations
at once. One comparison, then remove the losing/temporary paths.
Keep <=24 core tests, no new campaign launcher, no changes to `main`.
