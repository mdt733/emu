# EMU history — archaeology and lessons worth keeping

This file is deliberately the one place where old versions, dead ends, bugs and
historical reasoning live.  Current code comments should explain the current
code.  They should not become a museum of every previous implementation.

If a future result contradicts something here, prefer the new experiment and
update this history rather than preserving an old belief by inertia.


## Durable project rules

1. **EMU is dynamically, not statically, stable.**
   Ball-like feet and six actuated joints make stepping/catching the useful
   primitive.  A motionless stand is not a prerequisite for locomotion.

2. **Use fixed held-out rulers.**
   A curriculum metric changes because both the policy and the task change.
   Fixed deterministic evaluators are the longitudinal capability ruler.

3. **Verify measurements before tuning controllers around them.**
   Several apparent RL collapses were telemetry/curriculum-accounting problems.

4. **Change one causal layer at a time when possible.**
   Broad overnight matrices are fine when each branch asks a crisp question;
   simultaneous architecture/reward/curriculum rewrites are not.

5. **PPO bookkeeping correctness is not clerical detail.**
   Timeout bootstrapping, raw/clipped action likelihoods, reward timing,
   checkpoint provenance and recovery state all became important once episodes
   got long and competent.

6. **Recovery means restoring causal state, not just network weights.**
   Policy, optimiser, curriculum, governor, normaliser and LR-controller state
   belong together.

7. **Do not fake domain randomisation.**
   Sensor noise is not random mass, geometry, friction or servo physics.

8. **Survival is necessary but not sufficient.**
   A robot can remain upright while ignoring a command.  Command capability
   needs both survival and obedience/tracking.


## Robot / training context

EMU is a small reverse-knee biped, roughly half a kilogram in the simulation
lineage, with six position-controlled joints and ball-like feet.  The intended
physical robot does not rely on magical simulator contact signals.

Modern training uses three live regimes:

- STEADY — hold a commanded vx/vy/yaw;
- TRANSITION — change commands during an episode and adapt quickly;
- RESILIENCE — keep moving through rough starts and external pushes.

POSTURE is a shared difficulty dial rather than a fourth scenario.  Different
environments occupy different regimes simultaneously so PPO sees a mixture
rather than the whole vectorised population changing task at once.

The modern deployable actor packet is 55 channels x four history frames = 220
inputs.  It contains sensor/estimator-style information: gravity/gyro,
servo/joint state, previous command/control information, foot geometry,
noisy optical-flow/ToF-style horizontal velocity estimate and confidence,
height/command/phase and shin-flex proxies.  Simulator truth may be used by
rewards and evaluators but should not quietly leak into the deployable actor.


## v3xxx — the trainer becomes real

The v3xxx family established the JAX/MJX PPO stack, curriculum controller,
viewer/dashboard and much of the practical infrastructure.

Expensive bugs discovered in this era included:

- reusing JAX PRNG keys for independent random draws;
- recovery streaks reset by the wrong event;
- resume paths dropping curriculum/update clocks;
- observation dimensions drifting after encoding changes;
- viewers reading stale/renamed curriculum fields;
- argparse help strings breaking on unescaped `%` characters;
- best-checkpoint metadata referring to a different temporal policy than the
  parameters actually saved.

The lasting lesson is not "be careful".  Expensive bugs should become permanent
regression checks or explicit invariants.


## The abandoned v4xxx rewrite

One generation changed curriculum philosophy, reward shape, termination logic,
network structure and other assumptions together.  The result hovered around a
~30-step falling plateau across many configurations while a known-working
ancestor could escape an equivalent plateau in roughly 10-15M steps.

The family was abandoned.  Compound uncertainty made the experiment
uninterpretable.


## Why v500x happened

Late v3xxx curricula had several independently scheduled difficulty tracks.
Each could decide that its own dimension was ready to increase.  Several
individually safe pushes could therefore land simultaneously and create a much
larger on-policy distribution jump than PPO was meant to absorb.

The v500x redesign moved toward coherent regimes and a coordinated total
curriculum push budget.  Retreat remained uncapped: if several dimensions are
clearly too difficult at once, making all of them safer is desirable.


## The synchronised-timeout measurement failure

A particularly educational failure came from fall telemetry.

After a full-batch reset, timeout clocks were aligned.  Successful long episodes
therefore completed in waves while short failed episodes completed between the
waves.  A per-rollout fall ratio could alternate between apparently terrible
and apparently excellent even when the actual number of falls per environment-
step had barely changed.

Permanent fixes:

- stagger timeout clocks after full vector resets;
- keep true episode age separate from the staggered timeout clock;
- aggregate falls using pooled event counts/EMAs rather than equal averages of
  per-rollout ratios;
- use the same physically meaningful signal for collapse/recovery logic.

This remains the project's canonical "measure the thing before tuning the
controller that consumes the measurement" lesson.


## v5002 — long-horizon collapse still existed

The best v5002 policy could learn a useful gait and expand curriculum, then show
a large performance/fall-rate crash around the 70-75M region, recover later,
and continue oscillating.  Because the network could recover, this did not look
like a simple NaN or permanently destroyed policy.

The audit before v5003 identified shared correctness seams:

- timeout GAE bootstrapped from an auto-reset observation instead of the
  physical pre-reset next state;
- PPO likelihood bookkeeping could compare raw and clipped actions
  inconsistently;
- reward could be evaluated against a command/phase that had already advanced
  beyond what the action saw;
- early-fall shaping used the staggered timeout clock instead of true episode
  age;
- resilience force had a double-scaling risk;
- rollout health could nominate a post-gradient policy that did not generate
  those metrics;
- recovery did not restore every causal controller/training state.

v5003 fixed these as shared correctness changes rather than trying to assign the
historical collapse to one magic hyperparameter.


## v5003 — the old collapse disappears

After the shared correctness fixes, the characteristic late-training collapse
vanished.  Fixed `core` and `stress30` evaluators became permanent project
infrastructure.

Important v5003 lessons:

- heavy curriculum cohort throttling was too conservative;
- forced permanent easy-core rehearsal was unnecessary for that corrected
  experiment;
- a fixed low learning rate was slower rather than meaningfully safer;
- observation normalisation can drift toward a narrow stationary curriculum and
  alter held-out representations, so normaliser state belongs in checkpoint and
  recovery state;
- healthy training fall rate on a narrow curriculum does not imply broad
  held-out ability.

Do not reopen the v5002 "mystery collapse" without new evidence.  It was a
correctness-generation problem, not the current research question.


## v5004 -> v5005 — capability becomes the bottleneck

The corrected learner had ample stability margin, so attention shifted from
"make PPO survive" to "make the robot athletic and hardware-plausible".

The actor was stripped of simulator-teacher observations and converted to the
220-value deployable packet.  The shin-flex concept was restored as mid-shin
compliance with Hall-sensor-style flex/rate/load proxies.  Exact root vx/vy was
replaced in the actor by a noisy optical-flow/ToF-style horizontal velocity
estimate.  Reward-side MuJoCo truth remained allowed.

A bad stance-slip concept was corrected: penalising body horizontal velocity
was not the same thing as measuring foot slip.  The modern reward uses actual
tangential near-ground foot movement from consecutive exact poses.

The actuator/command envelope was widened.  A key discovery was that increased
**target authority** mattered more than merely asking for a wider velocity box.
The v5005 `athletic_torture` lineage became the foundation, with very strong
core/dynamic/resilience/robust held-out performance and compatibility with the
clean deployable actor.


## v5006 — adaptive frontier experiment

v5006 asked a more ambitious question: instead of declaring final vx/vy/yaw
ceilings, could the system discover its own physical command frontier?

It added independent physical frontiers for forward/reverse vx, vy, yaw,
mixed-command radius, transition hold/ramp timing, push force and sensor noise.
Mixed commands used a p-norm blob rather than assuming the simultaneous command
space was a rectangular box.

### What v5006 genuinely established

The command-frontier measurement was mis-specified.

The old precision gate used tight absolute all-step tolerances.  That meant
normal gait oscillation, command transitions and estimator noise could vote
against a boundary.  Lateral/yaw precision was especially problematic.  Across
open-frontier branches, vy and yaw retreated to tiny floors and most combo
frontiers retreated as well.  Reduced exposure then encouraged real forgetting
of lateral/yaw athletic behaviour.

This did **not** mean all adaptive frontiers were useless:

- push-frontier branches expanded from 0.70 N to roughly 1.66 N;
- acceleration/transition frontiers tightened hold/ramp timing strongly;
- `open_frontier` expanded reverse vx substantially;
- `servo_physics_only` pushed forward vx to about 0.604 m/s.

### The no-software-slew clue

`servo_physics_only` effectively removed the artificial target slew limiter and
let simulated geared-servo dynamics determine realised motion.  It produced
particularly strong tracking and recovery/robustness numbers, including very
low vx/yaw tracking error through much of its trajectory.

That result was not perfectly isolated because the branch still lived inside
the broken command-frontier generation.  v5007 therefore carries a clean
`slew_static` vs `noslew_static` fixed-envelope A/B.

### Long continuation / specialisation

The automatically selected phase-2 family was `open_frontier`, not `servo44`.
The continuation became exceptionally strong around 100-120M absolute steps:
resilience and robust transfer were excellent while dynamic capability was
still broad.

Past roughly 130M, yaw tracking deteriorated and dynamic performance began to
fall while resilience could remain high.  By 150M there was a clear broad
capability collapse.  The leading interpretation is curriculum-driven
specialisation/interference rather than a simple exploding-KL failure.

Permanent lesson: save frequently and select the broadest held-out checkpoint,
not simply the final/latest one.

### Combo evaluator lesson

`combo70` survival was near-perfect for many branches even when tracking was
mediocre.  Survival under a mixed command only proves the robot did not fall;
it does not prove it obeyed vx/vy/yaw accurately.  Mixed-command evaluators must
report tracking error as well as survival.

### Viewer combo-envelope hotfix

The original viewer sampled vx, vy and yaw independently.  Combo-trained
branches did not train a rectangular command cube; they trained a p-norm mixed
command region.  Random/manual/dynamic viewer commands could therefore request
corners the policy had never been trained to execute and make a good policy look
broken.

The viewer was fixed to read combo geometry/frontier state from the checkpoint
and project combined manual/random/dynamic commands inside the learned p-norm
region.  Axis-only branches retain rectangular semantics.  This was a viewer-
only fix, not a training change.

### Preflight hotfix lesson

Historical preflight tests accidentally allowed JAX to see the training GPU and
could fight a live job for VRAM.  Preflight work should be CPU-only.  v5007's
clean ladder therefore runs a small CPU JAX/MuJoCo/XML smoke test rather than a
large separate regression suite from the runtime package.


## v5007 — repaired frontier + anti-forgetting

v5007 keeps the 220-value actor architecture and checkpoint compatibility with
the v5005 lineage. The original v5005 champion checkpoint was later lost during
the repository/data-layout cleanup, so v5007 was deliberately changed to rebuild
one common locomotion ancestor instead of allowing every experimental branch to
rediscover walking independently.

The ladder now trains a conservative `foundation20` run from scratch to 20M
steps, then warm-starts every phase-1 branch from that exact policy + observation
normalizer while giving each branch fresh optimizer/curriculum/frontier state.
This preserves the key experimental requirement: all branches share one common
starting policy. The foundation uses the target no-slew actuator/hardware
contract but none of the v5007 experimental frontier/combo levers.

v5007 is primarily a curriculum/measurement/evaluation repair, not another
network rewrite.

Core changes:

- true no-target-slew environment mode;
- one clean old-slew control for causal comparison;
- settled, near-edge frontier evidence;
- absolute + command-relative tracking tolerances;
- EMA/bad-streak hysteresis for tracking-based retreat;
- command floors at the already proven v5005 athletic envelope;
- 20% full-envelope athletic rehearsal, excluded from frontier certification;
- repaired combo/sensor/transition frontier gating;
- two fixed deterministic evaluation cohorts by default;
- mixed-command tracking metrics;
- `growth110` and `growthcombo110` beyond-v5005 exams;
- rolling broad held-out selection into `best_broad_eval.pkl`;
- confirmation of the top two branch families on a second PPO seed;
- phase 2 from the best broad checkpoint, stopping at 120M by default.

The default phase-1 matrix has 12 purposeful branches.  The extra GPU work is
intentional: it buys causal A/Bs and cross-seed evidence rather than speculative
branch-count inflation.


## True physics randomisation remains an honest boundary

The current fast MJX architecture vectorises environment data around a shared
model.  It can cheaply vary state, observations and external forces, but it does
not honestly give every environment a separately randomised robot model.

Do not label observation noise as morphology DR.  True per-environment
randomisation of link lengths, mass/inertia, floor friction and servo constants
needs a batched-model architecture.

If v5007 solves command-frontier/forgetting issues and `robust40` is still the
main gap, true model-domain randomisation is the obvious next engineering
candidate.


## Regression/invariant checklist for future rewrites

Even though the clean runtime archive no longer ships a pile of historical test
files, these invariants should survive future work:

- never reuse a JAX PRNG key for independent draws;
- assert actor observation dimensions from constructed components;
- timeout truncation and true terminal falls are different in GAE;
- PPO likelihoods must use the same raw sample for old/new log probability;
- execution clipping must not silently redefine the probability sample;
- compute reward against the task/phase the action actually observed;
- keep true episode age separate from timeout staggering;
- stagger full-batch timeout clocks;
- save/restore observation normaliser state;
- save/restore optimiser, curriculum, governor and LR-controller state together;
- best-checkpoint metrics and saved parameters must refer to the same temporal
  policy;
- compare actual regime occupancy with requested allocation;
- fixed held-out evaluators must remain stationary across a comparison;
- combined-command capability requires tracking, not survival alone;
- live exploratory frontiers must not silently erase already demonstrated
  capability without an explicit experiment allowing that ablation.


## Source/provenance infrastructure introduced with v5007

The project stopped using copied source folders as its primary version-history
mechanism. The source tree is now intended to live in one Git repository. A
root `VERSION` file is the operational generation identifier. `run_ladder.py`,
`environment.py`, default raw-run paths, tracked result paths and metadata all
derive from it.

Heavy experiment artefacts are deliberately separated from source history:

- Git repository: code, robot model, documentation and compact results;
- `../emu_runs/<VERSION>/`: checkpoints, full metrics, fixed-eval trajectories,
  logs and other multi-gigabyte run evidence.

The ladder snapshots Git provenance at launch and passes it into training so
checkpoints carry the source version, commit, branch/describe state and dirty
flag. This complements the older temporal checkpoint-provenance work: temporal
provenance says which policy state produced a metric; Git provenance says which
source produced the entire run.

The previous standalone provenance note, viewer-hotfix note, preflight-hotfix
note, reference-result files, plan files, build notes, duplicate READMEs and
standalone result generator remain intentionally absent. Their durable lessons
are condensed in this file and `handover.md`.

Permanent operational lesson: **do not auto-increment the experiment generation
for every run.** Advance `VERSION` deliberately when the experiment definition
changes, commit it, and create the matching Git tag. Seeds/retries/confirmations
of one design belong to the same VERSION.

## v5008 — performance hardening + reference-start capability work

v5008 begins from the completed v5007 `sensor_frontier_repaired__phase2`
terminal ~120M checkpoint by default. The exact terminal checkpoint and the
v5007 best-broad checkpoint are copied out of the raw run tree into the
protected sibling store `../emu_references/v5007_120m/`; hashes and compact raw
evidence travel with them. The zero-start/shared-foundation route remains a
first-class supported start mode.

The first v5008 engineering pass is performance instrumentation. v5007's 10M
boundaries were visibly expensive because a full fixed evaluation performs ten
legacy suites across two cohorts (roughly 14.3M held-out environment steps at
2048 envs and horizon 350) before checkpoint work. v5008 records wall time for
individual evaluator suites, compiled training, host diagnostic transfer,
rollout summarisation/governor work, normalisation, logging and checkpoint
transfer/write costs.

Two semantics-preserving efficiency fixes landed with the instrumentation:
rollout/PPO diagnostic pytrees are device-fetched once per update instead of
many scalar/array fetches, and checkpoint finite validation now reuses the same
host state copy that is written rather than paying a duplicate parameter
transfer.

The legacy v5007 held-out ruler remains unchanged. A less-frequent hard tier was
added for mature-policy headroom: dynamic100, resilience100, robust70,
growth140 and growthcombo140. A fresh hard evaluation contributes to
`eval_v5008_score` and may save `best_v5008_eval.pkl`; `best_broad_eval.pkl`
remains for longitudinal continuity.

The focused capability matrix contains reference_continue, frontier_extended,
resilience_extended, rehearsal10, lower_kl_confirm and sensor_margin2. The top
three receive a second seed. A top-two family-score gap below 0.010 triggers a
third seed for both before long continuation selection.

## v5008 stage 2 — evaluator efficiency pass

Stage-1 profiling showed the training kernel itself sustaining roughly 29–30k environment steps/s, while fixed evaluation dominated wall time. With 2048 environments, horizon 350 and two cohorts, the 10-suite legacy ruler simulates 14,336,000 held-out environment steps at every 10M training boundary; the forced final legacy+hard exam adds 21,504,000 evaluation steps. In the 12M profile run, the first evaluation boundary added about 7.6 minutes and the final broad+hard exam about 9.8 minutes.

Stage 2 therefore optimises the evaluator before touching PPO/physics maths:

- routine held-out checks use one deterministic 2048-env cohort; forced/final evaluations retain two cohorts for direct final-score comparability with v5007 and v5008 stage 1;
- fixed evaluation now aggregates the nine required summary metrics on-device and returns only compact scalars instead of materialising/transferring a full horizon-by-environment metric history;
- evaluator telemetry records cohorts used, equivalent simulated evaluation steps, wall time and evaluation steps/s;
- the second profiling run is stored as `perf_probe2`, preserving the original `perf_probe` evidence in the same v5008 run store.

No reward, observation, physics, actuator, PPO, command-envelope or final evaluator definition is changed by this pass.

## v5008 stage 3 — physical capability and recovery atlas

The stage-1 hard probes (`dynamic100`, `resilience100`, `robust70`, etc.) were useful for proving that the old fixed evaluator suite had saturated, but their names/numbers were programmer-chosen coordinates, not physical capability limits. Stage 3 demotes them to optional legacy regression probes and removes them from branch selection.

Added `atlas.py`, a non-training evaluator that sweeps interpretable physical quantities and writes the full curves plus 95/80/50% read-off frontiers. Command probes cover fixed forward/reverse/lateral/yaw commands. Recovery probes cover initial body tilt, linear-velocity mismatch, angular-velocity mismatch, independent six-joint pose disorder, and a combined-chaos sweep. Recovery is stricter than survival: the robot must regain near-upright attitude, controlled roll/pitch rate, command tracking, support and no bad body contact continuously for 0.20 s. Recovery-time mean/p50/p90 are retained at every point.

The environment gains evaluator-only probe controls in the params vector. They do not enter the actor observation and do not change ordinary training when left at their default sentinels. The zero-start path remains supported.

Routine/final arbitrary hard probes are now disabled unless `--legacy-hard-probes` is explicitly requested. Longitudinal v5007 fixed evaluators remain for regression continuity; physical capability claims should come from the atlas.

## v5008 Stage 3 hotfix — checkpoint loader compatibility

- Fixed `atlas.py` loading checkpoints created by `training.py` when it runs as `__main__`.
- Such checkpoints record the `PPOState` NamedTuple as `__main__.PPOState`; atlas now exposes the compatible `training.PPOState` definition before unpickling, matching the existing viewer compatibility path.
- No physics, policy, evaluator, curriculum, checkpoint contents, or atlas definitions changed.

## v5008 stage 4 — sparse combined-command capability cloud

Stage 3 proved that a dense axis-at-a-time atlas is useful as an occasional
held-out audit but is the wrong shape for continuous evaluation. Stage 4 adds a
sparse capability-cloud representation with deliberately capped dimensionality:

- command is always the real 3-vector `(vx, vy, yaw)` and ordinary probes are
  genuine combinations rather than one-axis tests;
- disturbance context is categorical (`none`, `pose`, `momentum`, `impact`,
  `chaos`), not another stack of map axes;
- each disturbance family exposes one scalar severity while nuisance details
  such as signs, directions, joint-by-joint offsets and shove direction are
  randomized across the vectorized evaluation cohort;
- outcome stores survival, sustained dynamic recovery probability and recovery
  time rather than a single aggregate score.

`capability_cloud.py` provides a standalone `--stage cloud` survey. It begins
with deterministic low-discrepancy exploration and spends a second refinement
pass near empirically borderline points. The raw CSV remains authoritative; a
compact family × command-load × severity reduction is written only for human
inspection.

Capability-training runs can now accumulate the same cloud continuously at
normal held-out-evaluation boundaries. The default ladder budget is 12 sparse
points per regular evaluation and 32 at final evaluation, using only 256 envs
per point and a 220-step horizon. This evidence is observational in stage 4: it
is not yet allowed to select branches or drive the curriculum.

The first dense atlas also exposed a recovery-detector bug in concept rather
than mechanics: requiring very low body rates and a ~12-degree upright gate
made a healthy dynamic stepping gait fail the recovery definition even after
tiny disturbances it survived easily. Recovery now means sustained return to
broadly upright, supported, three-axis command tracking; roll/pitch rate is not
required to become statue-like. The cloud reports transient recovery events
separately from sustained recovery through the evaluation horizon.

An evaluator-only exact shove probe was added. It applies one controlled
horizontal force pulse shortly after birth; direction is randomized across envs
and the cloud records force, duration and impulse in physical units. Ordinary
training retains its existing stochastic resilience pushes when the probe
sentinel is disabled.


## v5008 stage 4.1 cloud audit corrections
- The sparse cloud is explicitly a **held-out probe cloud**, distinct from the future passive training-experience cloud / near-fall replay bank.
- Raw 3-D command vectors are now retained in direction-aware command sectors in reports; scalar command load is only a coarse marginal and may not hide asymmetric mixed-command corners.
- Impact/chaos recovery timing cannot be credited until the controlled shove has finished.
- Footing, actuator/model uncertainty, passive rollout harvesting, command-transition/acceleration probing and near-fall replay remain deliberate later layers; they are not faked into this survey.

## v5008 stage 5 — paired recovery campaign

Stage 4.1 proved the low-dimensional cloud representation useful, but also exposed
an evaluator-design alias: disturbance family had become correlated with command
yaw direction. Stage 5 removes that class of error by generating complete matched
command bundles. Every held-out command anchor is evaluated as the exact same
`(vx, vy, yaw)` under `none`, `pose`, `momentum`, `impact`, and `chaos` contexts.
Family, command direction and severity can therefore no longer masquerade as one
another.

Nuisance disturbance direction is stratified into eight equally represented
within-point bins for held-out probes. The map still has no extra direction axis;
each point simply retains mean recovery plus worst/best direction diagnostics.
Impact/chaos recovery time is measured from the end of the shove rather than from
episode birth.
Routine held-out panels now combine two permanent command anchors with rotating
matched scout anchors. That gives exact longitudinal coordinates while gradually
filling new parts of the 3-D command envelope. All branches see the same scout
schedule; the 60-point final panel densely rechecks the first 12 command anchors.

Disturbed results are only treated as valid resilience evidence when the matched
undisturbed command itself recovers >=80% and survives >=95%. This separates
command incompetence from recovery incompetence without inventing another
aggregate physical-capability score.

Training keeps one resilience difficulty scalar. Stage 5 exposes the physical
maxima that scalar expands into: tilt, root-position offset, horizontal velocity
mismatch, angular velocity mismatch, and independent actuator-joint pose disorder.
Historical defaults reproduce the v5007/v5008 reference; new linear/joint
disorder are opt-in. Recovery-specialist branches can reset only the resilience
difficulty after warm-starting the champion, preserving learned command/frontier
state without instantly dropping the policy into the hardest new recovery world.

The serious capability ladder now runs nine 60M branches from the protected v5007
120M reference: control, combined-command/maneuver, pose recovery, momentum
recovery, impact recovery, chaos recovery, chaos with 10% rehearsal, chaos with
lower KL, and an integrated maneuver+recovery branch. Top three branches buy a
second seed; close top-two recovery results buy a third seed; the selected family
continues to 120M absolute branch steps.

Automatic seed/phase-2 selection is explicitly regression-safe and lexicographic:
preserve reference longitudinal durability within 0.04 and undisturbed command
competence within 0.05, require matched valid disturbed evidence, then prefer
held-out disturbed recovery and worst-direction recovery. This selection rule is
an experiment scheduler only, not a claim that one scalar describes physical
capability.

## v5008 stage 5 hotfix — preflight reset contract

- Fixed the new recovery-aware preflight smoke test to unpack `EmuMJXEnv.reset()` as `(state, obs)` before calling `step()`; the Stage 5 environment/training path was not reached by the original failed smoke test.
- Capability-campaign confirmations now always include the `reference_continue` control seed in addition to the top ranked experimental families, preventing multi-seed experiments from being compared to a one-seed baseline.
- Phase 2 now resumes the terminal `latest_mjx.pkl` policy that produced the final paired capability cloud (with best-broad checkpoints only as fallback), keeping selection and continuation provenance aligned.

## v5008 stage 5.2 — evaluation throughput hardening

Early Stage-5 profiling showed the PPO kernel was not the wall-clock bottleneck. On the 60M reference branch, train_update consumed about 2121 s, fixed regression evaluation about 1436 s, and the capability cloud about 1938 s. The printed cumulative FPS therefore collapsed after evaluation even though raw train_update throughput remained near 28–30k env-steps/s.

Stage 5.2 changes evaluation execution without changing PPO/reward/physics/training semantics:

- logs both `train_fps` (compiled training only) and `wall_fps` (whole-process effective throughput);
- decouples cloud cadence from legacy fixed-eval cadence;
- legacy fixed regression evaluation defaults to every 30M steps, while the paired capability cloud remains every 10M;
- cloud panels are control-gated: exact undisturbed command controls are evaluated first and disturbance families are only run where baseline command competence is established;
- cloud points are evaluated in outer-vmap batches (default 4 points × 256 envs) to improve GPU occupancy while retaining 256 robots and eight nuisance-direction strata per point;
- recovery_not_before is dynamic so impact/chaos points can share the same batched evaluator compile;
- phase 2 can resume an existing `__phase2/checkpoints/latest_mjx.pkl` in place, enabling a performance-only source update without discarding an interrupted continuation;
- preflight now includes a tiny batched-cloud JAX/MJX execution smoke test.

The old fixed rulers remain regression coordinates. The speed pass deliberately removes wasted invalid resilience probes rather than lowering the evidence threshold.


## v5009 — evaluator throughput + clean cross-version continuation

Version boundary created after v5008 Stage 5 performance evidence showed evaluation dominating wall time. v5009 retains the Stage-5 training/recovery architecture but changes evaluator execution and cadence: separate training/effective FPS, control-gated disturbance probes, point batching, and less-frequent legacy regression evaluation. A dedicated `--import-v5008-phase2` path imports the real in-progress v5008 phase-2 latest checkpoint into a fresh v5009 run directory so provenance remains clean across the evaluator change. Legacy metric/checkpoint field names containing `v5008` remain readable for compatibility and are not claims about the current source version.

## v5010 — continuous self-mapping trainer

- Added `capability_field.py`: 21×15×25 command lattice with exponentially-forgetting command competence, recovery-severity evidence and 26-sector maneuver edges.
- Ordinary PPO rollout metrics now continuously update the live field; no extra simulator review is required.
- Added fixed-size node/edge teaching packets to `environment.py`. The field chooses geography while `RegimeGovernor` retains global safety/frontier authority.
- Added `shadow`, `nodes` and `graph` field modes. Shadow computes the same map and packet with zero steering for a fair control.
- Periodic held-out cloud now acts as calibration: the live field chooses questions, but held-out answers never update the field.
- Added `run_ladder.py --stage trainer`: genuine 0→20M cold foundation, exact-state fork into three 20→120M trainer variants, automatic independent zero-start winner confirmation, optional 220M extension.
- Capability-field state is checkpointed and restored with governor state, including auto-recovery temporal rollback.
- VERSION bumped to v5010 because this changes the trainer/experiment definition rather than merely patching evaluator performance.

### v5010 final hardening before first GPU run

- Node teaching anchors now carry a locally useful recovery-severity proposal as well as `(vx, vy, yaw, weight)`. Resilience births selected by the live field use that severity, clipped to the governor-earned global resilience difficulty; the field therefore steers both command geography and *where along the recovery frontier to practise* without gaining authority to exceed safety limits.
- Routine held-out reviews remain field-selected calibration questions, but the forced 120M final panel is deterministic and identical across trainer branches. This prevents branch selection from comparing policies on different exams while still annotating the common controls with each field's own competence prediction.
- Trainer-mode legacy fixed evaluation is intentionally sparse (one mid-run check plus forced final). Continuous field telemetry does the broad review work; held-out calibration remains every 20M.
- Routine calibration defaults to 15 requested points (three matched command bundles); the common final panel defaults to 50 requested points (ten matched command bundles), with control gating still avoiding meaningless disturbance work.


## v5011 — Honest Map

- Promoted the v5010 graph trainer to the recommended public training path.
- Capability field schema 2: event-based node/recovery/maneuver competence, separate continuous quality, conservative confidence bounds and explicit staleness/trend metadata.
- Added complete maneuver target/ramp telemetry to the training metrics without changing actor observations or rewards.
- Re-enabled mastered-boundary exploration using statistically meaningful mastery; added evidence-aware under-coverage pressure.
- Replaced the final sparse Halton recovery panel with a deterministic moderate common recovery panel; dense atlas remains the physical-limit ruler.
- Added calibration MAE + Brier-style error while keeping all held-out outcomes quarantined.
- Added checkpoint schema/model/observation metadata plus public `latest.pkl`, `best_eval.pkl`, `final.pkl` aliases.
- Added `emu` public CLI, editable packaging, `emu doctor`, expanded viewer, tests and cleaned documentation layout.
