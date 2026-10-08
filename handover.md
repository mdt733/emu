# EMU handover — current VERSION: v5007

## What EMU is

EMU is a small six-servo reverse-knee biped with ball-like feet.  It is meant
to balance dynamically by stepping and catching itself, not by first learning
a rigid motionless stand.  The control goal is practical joystick locomotion:
forward/reverse/lateral velocity, yaw, rapid command changes and recovery from
disturbances while using observations that could plausibly exist on the real
robot.

The current actor input remains **220 values**: 55 deployable-style channels
with four frames of history.  The policy does not receive simulator-only contact
truth or perfect root velocity.  Simulator truth is still allowed for rewards,
training diagnostics and held-out evaluation.

`robot.xml` is the MuJoCo robot/actuator model.  `environment.py` owns physics,
tasks, observations, rewards and curriculum/frontier state.  `training.py` owns
PPO, normalisation, checkpointing, recovery and fixed evaluation.  `viewer.py`
is the interactive/manual showcase.  `run_ladder.py` owns the current experiment
matrix, raw/output routing, Git provenance capture and overnight orchestration.
`VERSION` is the single operational version source; the current value is `v5007`.


## Where we are now

v5005's `athletic_torture` policy established the modern baseline: the cleaned
deployable actor could move aggressively and tolerate substantially more target
authority and uglier conditions than the older conservative curriculum allowed.

v5006 then tried to discover the robot's physical command frontier instead of
hard-coding a final vx/vy/yaw ceiling.  It produced useful evidence but exposed
a governor design error:

- lateral and yaw frontier precision used tight absolute, all-step tracking
  thresholds;
- transient command changes and normal estimator noise were counted against the
  boundary;
- vy/yaw frontiers retreated to their floors;
- most combo-trained branches also lost their combo radius;
- reduced exposure then encouraged genuine forgetting of parts of the already
  proven athletic envelope.

The adaptive-frontier idea itself was not disproved.  Other dimensions did
expand: push force reached about 1.66 N, transition timing tightened strongly,
reverse vx expanded in `open_frontier`, and the no-software-slew experiment gave
very strong tracking/recovery results.

The long v5006 continuation also showed a useful training-time pattern.  Around
100-120M absolute steps the policy was exceptionally broad and resilient.  Past
roughly 130M it increasingly specialised: resilience could remain excellent
while yaw tracking and dynamic80 deteriorated, with a clear broad collapse by
150M.  That is why v5007 does not equate "latest checkpoint" with "best policy".


## What v5007 is testing

v5007 repairs measurement/distribution before making another architectural
rewrite.

### 1. True no-software-slew baseline

Most branches bypass the artificial joint-target slew limiter.  The policy asks
for a target and simulated geared-servo physics decides what motion is actually
possible.  `slew_static` remains as one clean old-slew control against
`noslew_static`.

This does **not** claim that real-servo thermal limits are solved.  The current
MuJoCo actuator model still needs separate sustained-torque/thermal validation
before hardware transfer.

### 2. Repaired command frontier

A frontier can grow only from evidence near that actual boundary and after the
command has had time to settle.  Easy interior commands cannot vote that the
edge is ready to expand.

Tracking tolerance combines an absolute allowance with a command-relative
allowance.  Tracking-based retreat uses an EMA/bad-streak hysteresis instead of
letting one ugly rollout eat the envelope.

The command frontier may retreat toward, but not below, capability already
proven in v5005:

- vx forward: +0.60 m/s
- vx reverse: -0.20 m/s
- |vy|: 0.22 m/s
- |yaw|: 1.50 rad/s
- mixed-command radius: 1.00

### 3. Athletic rehearsal

Repaired-frontier branches normally reserve 20% of episode births for the full
known v5005 athletic envelope.  This is anti-forgetting practice, not frontier
evidence, so rehearsal samples are excluded from frontier certification.

`frontier_no_rehearsal` exists specifically to test whether this is causal.

### 4. Better fixed evaluation

The moving curriculum is not the ruler.  Fixed evaluators remain the primary
comparison:

- core / stress30 / dynamic50: continuity
- dynamic80: athletic edge
- combo70: mixed-command survival **and tracking error**
- resilience50 / resilience75: disturbance recovery
- robust40: transfer to a deliberately uglier actuator/latency/sensor model
- growth110 / growthcombo110: commands beyond the old v5005 envelope

Fixed evaluation averages two deterministic cohorts by default.  A rolling
broad held-out score selects `best_broad_eval.pkl`; the long continuation resumes
that checkpoint instead of blindly resuming the final policy.


## Default overnight ladder

`python run_ladder.py --list` is authoritative.  The default matrix is:

1. `slew_static` — old 4 rad/s software slew, fixed athletic envelope.
2. `noslew_static` — true no-slew, same fixed envelope.
3. `frontier_axis` — repaired single-axis frontier before combo geometry.
4. `frontier_repaired` — primary repaired frontier + p=1.35 combos + rehearsal.
5. `frontier_no_rehearsal` — anti-forgetting ablation.
6. `frontier_push` — repaired command frontier + adaptive pushes.
7. `frontier_accel` — repaired command frontier + adaptive transition timing.
8. `frontier_push_accel` — composition of the two v5006 frontiers that worked.
9. `frontier_lower_kl` — same primary experiment with target KL 0.025.
10. `resilience_focus` — harder manoeuvring/recovery emphasis.
11. `sensor_frontier_repaired` — sensor-margin retest after command fix.
12. `frontier_beast2` — repaired all-levers combination.

Phase 1 runs each for 60M additional steps from the v5005 champion.  The top two
branch families are rerun at a second training seed.  The best family across the
available seeds is then continued to 120M absolute steps from its best broad
checkpoint.


## How to judge the results

Do not pick a winner from final training fall rate or one pretty survival
number.

A convincing branch should:

1. preserve near-perfect core/dynamic continuity;
2. preserve or improve vx/vy/yaw command tracking;
3. show real `growth110` / `growthcombo110` evidence beyond v5005;
4. raise resilience without trading away yaw/command fidelity;
5. remain strong in `robust40`;
6. stay broad through the phase-2 tail rather than specialising late;
7. preferably reproduce on the confirmation seed.

`combo70` survival alone is not enough.  v5006 proved that a policy can stay
upright under mixed commands while obeying them poorly.

Likewise, frontier retreat must not be fall-only: a robot that survives a hard
left-turn request by walking straight ahead has not demonstrated that command
as part of its capability frontier.


## What remains intentionally out of scope

True per-environment morphology/mass/friction/servo-parameter domain
randomisation still requires a batched-model MJX architecture.  Observation
noise is not a substitute for model randomisation.

If v5007 preserves/grows nominal command capability but `robust40` remains the
clear bottleneck, that is the strongest argument for making true model-domain
randomisation the next engineering project.


## Repository and data layout

EMU now uses one source repository rather than one copied source folder per generation.

```text
~/robot_lab/
├── emu/                    # Git: source, docs and compact results
│   └── results/v5007/      # results.md + small summary evidence
└── emu_runs/v5007/         # external: checkpoints and full raw telemetry
```

The ladder derives both versioned locations from the root `VERSION` file. The
default external base can be moved to another disk with `EMU_RUN_ROOT`. Raw
checkpoints are intentionally outside Git; compact experiment evidence is
written back into `results/<VERSION>/`.

Every ladder launch captures the Git commit, branch, describe/tag state and
dirty-source status. That provenance is stored in run metadata and checkpoints.
The intended workflow is therefore: commit clean source -> run training ->
commit the small completed results -> tag the completed generation. The run
metadata preserves the exact source commit used for training.

## First commands on the training machine

Read `install.txt` for a blank-machine setup.  For an already working machine:

```bash
python run_ladder.py --list
python run_ladder.py --device cuda --base-checkpoint /full/path/to/v5005/checkpoint.pkl
```

When the run is finished, start with `results/v5007/results.md`. Full branch
`fixed_eval.csv`, `metrics.csv`, logs and checkpoints remain in the external
`../emu_runs/v5007/rung_runs/` tree for deeper diagnosis.

See `install.txt` for the Git workflow, blank-Ubuntu setup, raw-run relocation,
viewer paths, and the exact procedure for advancing to v5008.
