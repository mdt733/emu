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

## Manufacturer motor calibration: existing simulator is NOT a confirmed 6V motor

`emu/robot.xml` currently labels STS3215 at **6V** yet assumes
19.5 kgf.cm stall (1.913 N.m), 6.5 kgf.cm rated (0.638 N.m), and
42 rpm no-load (0.238s/60°).

The actual Feetech STS3215/C001 official 2023 A/1 datasheet gives:

| Supply | Stall kgf.cm | Rated kgf.cm | no-load RPM | I_stall |
|---|---:|---:|---:|---:|
| 6V | 16.5 | 4.0 | 42 | 2.0A |
| 7.4V | 19.5 | 5.0 | 52 | 2.5A |
| ST-3215-C018 @12V | 30 | 10.0 | 45 | 2.7A |

Source PDFs (manufacturer):
- https://files.seeedstudio.com/products/Feetech/108090023_STS3215-C001_Datasheet.pdf
- https://cdn.robotshop.com/media/F/Fit/RB-Fit-155/pdf/feetech_12v_30kg_cm_magnetic_encoding_servo_sts321_specification_pdf.pdf

Interpretation: model peak torque reflects a **7.4V** C001, but model
max speed is for **6V**, and model continuous rated torque is optimistic
at *either* supply. At 6V rated is 0.392 N.m; at 7.4V 0.490 N.m.
The model's 0.638 N.m over-rated threshold overestimates real
published continuous load headroom by ~63% / 30% respectively. A
reported 68% of actuator samples over *model* rated is therefore NOT
a reliable thermal-duty estimate for either voltage.

C018 12V motor nominal stall/rated are 2.942/0.981 N.m, same
45.2x24.7x35mm case and 55g. Its 45 RPM is actually slower than the
C001 at 7.4V (52 RPM), although torque improves substantially. Power
must remain separately regulated/verified; do **not** apply 12V to C001.
The official C018 spec describes an over-current shutdown at >2A for
2 seconds (customizable) and a 70°C shutdown; stall-current figure is
not a continuous operating target. Six C018 can theoretically demand
6 * 2.7A = **16.2A** simultaneously at stall; no supply is sized by
the per-servo idle or normal current alone.

### Physical spring and action limits may explain one-sided saturation

Robot's hip-roll springs have stiffness **2 N.m/rad**. If the physical
joint actually follows a clipped roll target of `TARGET_SPAN=0.38 rad`,
the *centering spring* exerts 0.76 N.m (~7.75kgf.cm), beyond both C001
published continuous torque ratings. At ±25° joint limit it reaches
~0.873 N.m. **Action clipping alone does NOT imply the joint reached
its commanded angle, nor that the motor is fighting the spring**:
the direction of actual motor and spring moments matters. This doesn't imply the motor has reached stall in simulation;
the spring may be doing useful energy storage, and the motor may work
with gravity rather than against it. Need physically-signed moment
and motor-load data before changing the springs.

The 6-axis policy clips the left hip-roll action at ±1 increasingly
often (19.4%→67.2% early→late), but a command-position limit is
**not** a torque limit. The v5055 read-only diagnostic now reports
physical MJX actuator load fraction and hip-roll spring moment by
Academy versus non-Academy worlds, separate from the already reported
network action clipping. Motor actuator force omits MuJoCo passive
damping. The 0.435 N.m*s/rad joint damping is a heuristic for a
torque–speed slope; it ALWAYS dissipates power, including when no
motor torque is commanded, so it is not a faithful active servo
torque-speed limiter and must not be confused with motor electrical
work. Do not change the solver or motor physics in this credit audit.

### Out-of-the-box shape threshold: 36mm diameter ball foot

Each foot is a sphere with radius **18mm**. A 36mm diameter rounded
contact can exploit rolling/rocking interactions against lips
shallower than the diameter without a clean pre-lip swing lift.
Around and above **36mm**, this mechanism cannot alone make the
sphere bridge from lower floor to upper lip; an actual lifted step
becomes critical. v5054's sharp physical traverse failure from
32.5mm to 40mm may reflect a *change of required mechanism*, not just
curriculum aggression. This is a hypothesis, not a proven geometric
causal threshold; re-examine 28–44mm with dense 1.25mm height levels
(`--levels 40 --trials-per-level 50`), extract first upper-foot
height and contact normal BEFORE lateral-accuracy gates.

### Other falsifiable engineering questions

- **Mirrored trials:** identical sign-mirrored left/right robot states,
  commands and local step normals should give symmetric dynamics.
  Actor has no enforced chirality equivariance, and a persistent
  one-sided roll preference could be a policy bias, not weak motors.
- **Sight geometry:** torso origin ~0.258m above foot ground; a 40°
  downward ToF beam hits flat terrain ~0.308m ahead, 60° ~0.149m,
  80° ~0.046m, while 100° looks slightly backward. At a birth only
  0.22m before a lip, essentially the shallow 40° row (four rays)
  samples the upper platform. 3mm top-height change causes only
  ~4.7mm idealized ray-range shift (3mm normal-range noise at
  sensor_scale=1.5); body pose, dropout and the 8mm planar-fit
  residual may obscure the event. Validate with actual fixed-pose
  ray samples; don't infer invisibility from this approximation.
- **Real robot mass:** C001 and C018 each weigh 55g, six motors
  already contribute 330g. The current XML total mass is about
  0.5kg; check **where those motor masses are physically located**
  before treating simulated lower-leg inertia and available torque
  as hardware-calibrated. 12V motor swap may retain nominal mass
  but changes motor dynamics.

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
