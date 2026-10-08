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
