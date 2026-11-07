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
zero.
This includes the **first** 32-frame GPU rollout; the metrics are evaluated
on the pre-update critic and policy, so later optimiser updates cannot
rewrite which action produced an earlier terminal. All metrics are summaries, not actor inputs and not new reward.
No trainer, sampler, physics, optimizer, architecture or sensor changes.

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
steps (absolute target 186M). This is not a new treatment: the new
branch changes **logging only**. Run under same 2048-world, seed-735,
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
