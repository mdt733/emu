# EMU handover — current VERSION: v5011

## v5011 — Honest Map + public cleanup

v5011 promotes the v5010 graph trainer to the recommended default and changes the live field from frame-weighted pseudo-probabilities to semantic event evidence. One stable command window produces one node verdict; one integrated resilience birth produces one recovery verdict; one source→target command ramp plus settling period produces one maneuver verdict. Continuous quality is tracked separately from competence probability.

Field schema is now 2 and intentionally cannot silently resume a schema-1 v5010 map. Network/governor/normalizer checkpoints remain readable; if a historical checkpoint carries an old field, v5011 starts a fresh map rather than reinterpreting it.

The public front door is `emu`: `emu doctor`, `emu train`, `emu resume`, `emu view`, `emu evaluate`, `emu benchmark`. `run_ladder.py` and `training.py` remain available for research/legacy experiments.

Forced final recovery evaluation now uses a deterministic moderate 3-D command panel so matched controls remain useful; physical limit discovery stays in the atlas. Routine calibration remains quarantined and never updates the live field.


## Robot / learning doctrine

EMU is a small six-servo dynamically unstable biped with ball-like feet. It is
supposed to stay up by stepping and catching itself; static standing is not a
prerequisite or primary target. The practical goal is joystick locomotion:
forward/reverse/lateral velocity, yaw, rapid command changes and disturbance
recovery using observations that could plausibly exist on the real robot.

The actor still uses the deployable-style 220-value input: 55 channels with four
frames of history. Simulator truth is allowed for reward/evaluation, not as a
hidden actor shortcut.

`robot.xml` owns the MuJoCo model. `environment.py` owns task/physics/reward and
observations. `training.py` owns PPO, curriculum/governor, checkpointing,
evaluation and v5008 timing. `run_ladder.py` owns experiment orchestration,
Git provenance and raw/tracked output routing. `VERSION` is the operational
source of truth.

## v5007 result that v5008 starts from

v5007 successfully rebuilt from one shared 20M cold-start locomotor and ran all
branches from that common ancestor. The important conclusions are:

- the artificial software target slew limiter is retired; servo physics should
  own actuator bandwidth;
- repaired settled edge-tracking frontiers did not reproduce v5006's catastrophic
  vy/yaw shrink/forgetting loop;
- survival alone is still not enough: command MAE remains part of competence;
- 20% athletic rehearsal was useful but not sacred; 10% deserves testing;
- lower target KL remained competitive and deserves proper multi-seed confirmation;
- sensor-frontier and resilience-focused families were essentially tied at the
  top rather than one being conclusively superior;
- most importantly, long training no longer inevitably destroys the controller.

The selected `sensor_frontier_repaired` continuation reached ~120M with perfect
(or effectively perfect) dynamic80/combo70 survival, resilience75 around 0.959,
growth110 around 0.990 and no v5006-style late collapse. This exact terminal
checkpoint is the default v5008 reference policy.

## Protected v5007 reference

Large checkpoints remain outside Git. Preserve the v5007 phase-2 ancestor under:

`~/robot_lab/emu_references/v5007_120m/`

Minimum important files:

- `latest_mjx.pkl` — exact terminal ~120M state; v5008 default reference start;
- `best_broad_eval.pkl` — preserve the v5007 held-out champion too;
- `fixed_eval.csv` and `metrics.csv` — behavioural/training evidence;
- `envelope_summary.json`;
- `rung_run_meta.json` and `ladder_run_meta.json`;
- `SHA256SUMS.txt`.

See `install.txt` for exact copy/hash commands.

The protected reference is an accelerator, not a dependency. `run_ladder.py`
retains `--start zero`, which rebuilds a shared foundation from scratch before
branching. Periodic from-zero sanity runs are required so EMU never becomes a
system that only works because an ancient checkpoint already knows how to walk.

## v5008 stage A — performance hardening

v5008 first measures where wall time actually goes before making invasive JAX
changes. `training.py --profile-performance` records cumulative/last/mean wall
time for:

- compiled train/update work;
- device-to-host transfer of rollout/PPO diagnostics;
- host rollout summarisation;
- governor/parameter bookkeeping;
- observation-normaliser/synchronisation work;
- each fixed evaluator suite;
- full fixed-evaluation boundaries;
- checkpoint device-to-host transfer;
- checkpoint pickle/write time;
- metrics/envelope logging.

`performance.json` is the compact live summary; timing columns also appear in
`metrics.csv` and `fixed_eval.csv`.

Two low-risk performance cleanups are already in the v5008 core:

1. rollout/PPO diagnostics are fetched from the GPU once as a pytree per update
   instead of many small `np.asarray` device gets;
2. checkpoint validation reuses the same host copy that is about to be pickled,
   instead of transferring/checking parameters and then transferring the state
   again to save it.

These change neither physics nor PPO/reward maths.

The first recommended v5008 command is deliberately small:

`python run_ladder.py --stage profile --device cuda`

It uses the exact v5007 winning recipe and the protected 120M reference. The
point is to obtain real timing evidence before further optimisation.

## v5008 stage B — longer evaluators

The v5007 rulers are retained unchanged for longitudinal comparison, but mature
policies now saturate several of them. v5008 therefore adds a less-frequent hard
tier rather than deleting the old ruler:

- `dynamic100`;
- `resilience100`;
- `robust70` using a harsher fixed actuator/latency/sensor/push world;
- `growth140` with a materially wider command design envelope;
- `growthcombo140` for mixed commands in that wider envelope.

Hard exams run every 20M steps by default plus the final evaluation. The new
`eval_v5008_score` combines the old broad score with the hard tier. A separate
`best_v5008_eval.pkl` is saved when a fresh hard evaluation sets a new v5008
high-water mark; `best_broad_eval.pkl` is still preserved for continuity.

## v5008 stage C — focused capability ladder

After the performance profile/pass is accepted, use:

`python run_ladder.py --stage capability --device cuda`

Current branches are intentionally fewer and more causal than v5007:

1. `reference_continue` — exact winning v5007 recipe/control;
2. `frontier_extended` — raises only competence-earned command guardrails;
3. `resilience_extended` — expands the disturbance-force envelope;
4. `rehearsal10` — tests 10% full athletic rehearsal;
5. `lower_kl_confirm` — proper multi-seed confirmation of target KL 0.025;
6. `sensor_margin2` — harsher sensor-noise start plus explicit sensor-frontier telemetry.

The top three receive a second seed. If the top two family means remain within
0.010 v5008 score, both receive a third seed automatically. Only then is the
winner continued long.

Do not build another all-knobs-on `beast` branch until individual improvements
have demonstrated compatibility.

## Open v5008 work after profiling

The next optimisation pass should be driven by `performance.json`, not guesses.
Likely candidates include evaluator batching/cadence, reducing host-governor
traffic by JIT-aggregating rollout statistics, avoiding recompilations, and
benchmarking env-count/rollout shapes for GPU occupancy.

The next realism work is intentionally not faked into this first v5008 core:

- sustained actuator thermal/duty behaviour should model transient torque plus
  continuous heating/derating, not simply replace stall torque with rated torque;
- true per-environment physical domain randomisation (mass/inertia/friction/
  servo constants) needs honest batched-model support. Observation noise alone
  must not be labelled morphology/domain randomisation.

## Invariants

- dynamically unstable stepping/catching remains intentional;
- no simulator-only actor truth;
- no artificial target slew unless explicitly resurrected as an A/B control;
- PPO old/new log-probability semantics must remain consistent;
- timeout truncation is not a true fall terminal in GAE;
- save/restore obs normaliser, optimizer, curriculum/governor and LR-controller
  state together for true resume;
- warm-start experiments may intentionally use policy+normaliser with fresh
  optimiser/governor histories;
- best-checkpoint metrics must refer to the exact policy being saved;
- fixed held-out evaluators remain stationary within an experiment generation;
- combined-command capability requires tracking as well as survival;
- exploratory frontier retreat must not erase the already-proven athletic floor;
- heavyweight checkpoints/runs remain outside Git;
- commit executable source before serious ladders so checkpoint provenance is
  reconstructible.

### Stage-2 performance rule

Routine fixed evaluation now uses one deterministic cohort, while forced/final evaluation uses two cohorts. This is intentional: routine evaluation is a screening/checkpoint signal; the final two-cohort result remains the authoritative cross-version comparison. The `perf_probe2` profile run exists specifically to measure this change against stage-1 `perf_probe`.

## v5008 stage 3: capability atlas doctrine

Do not describe a saturated fixed test (for example `dynamic100=0.999`) as "movement solved", and do not interpret a low arbitrary fixed-test score (for example `resilience100=0.05`) as a physical recovery limit. Those are coordinates on a historical regression ruler.

Use:

```bash
python run_ladder.py --stage atlas --device cuda
```

This loads the protected v5007 120M reference without training it and writes:

- `../emu_runs/v5008/atlas_reference/capability_atlas.csv`
- `../emu_runs/v5008/atlas_reference/capability_atlas.json`
- `../emu_runs/v5008/atlas_reference/atlas.md`

Small copies are also placed under `results/v5008/` for Git history.

The atlas reports full physical-unit curves and 95/80/50% frontiers for command capability and recovery from controlled initial-state disturbances. The percentages are transparent read-off conventions, not an aggregate "intelligence score". Always inspect the curves and recovery-time distributions.

Legacy `dynamic80/res75/...` evaluators remain valuable as regression alarms: they answer "did we break something the old robot could do?" rather than "what is the robot's maximum capability?".

The next recovery-training stage should be chosen from atlas weaknesses, then progress toward trajectory perturbation, mid-gait impulses/footing disturbances, and near-fall replay rather than blindly increasing one generic resilience scalar.

## v5008 stage 4 — Capability Cloud

Use the dense atlas as an occasional independent audit, not as the every-10M
workhorse. The continuous representation is now the sparse Capability Cloud:

`command (vx, vy, yaw) + disturbance family + severity -> outcome`

The active map intentionally stops there. Pose direction, angular-kick axis,
individual joint offsets and push direction are nuisance variables randomized
inside each vectorized cloud point, not dimensions to voxelise.

Standalone reference survey:

`python run_ladder.py --stage cloud --device cuda`

Default survey budget is 64 low-discrepancy exploration points + 24 refinement
points, 256 envs/point and 220 control steps/point. Outputs:

- `../emu_runs/v5008/cloud_reference/capability_cloud.csv` — raw point cloud;
- `capability_cloud.json` — compact family/load/severity bins;
- `capability_cloud.md` — human-readable summary.

During `--stage capability`, each training run also adds 12 cloud points at
regular held-out evaluation boundaries and 32 at the forced final evaluation.
These are written to each rung's `capability_cloud.csv/json`. They are currently
**observational only**: branch selection remains on the proven longitudinal
ruler until the cloud is demonstrated stable across policy age/seeds.

Disturbance families currently implemented honestly in the shared-model MJX
fast path:

- `none` — command-combination competence only;
- `pose` — randomized body tilt + six-joint pose disorder;
- `momentum` — randomized linear + angular velocity mismatch;
- `impact` — one controlled random-direction horizontal force pulse;
- `chaos` — a severity-scaled physically plausible mixture of all above.

Footing/friction and actuator-parameter families are intentionally absent for
now because true per-env friction/servo-constant randomisation needs batched
model parameters rather than pretending sensor noise or a lateral force is the
same thing.

Recovery is now dynamic: sustained broadly-upright support plus vx/vy/yaw
tracking. It no longer requires low roll/pitch rates, which incorrectly marked
normal walking as unrecovered in the first atlas.


## v5008 stage 4.1 cloud audit corrections
- The sparse cloud is explicitly a **held-out probe cloud**, distinct from the future passive training-experience cloud / near-fall replay bank.
- Raw 3-D command vectors are now retained in direction-aware command sectors in reports; scalar command load is only a coarse marginal and may not hide asymmetric mixed-command corners.
- Impact/chaos recovery timing cannot be credited until the controlled shove has finished.
- Footing, actuator/model uncertainty, passive rollout harvesting, command-transition/acceleration probing and near-fall replay remain deliberate later layers; they are not faked into this survey.

## v5008 stage 5 — next serious run

Stage 5 is the first full recovery/capability campaign built on the cloud evidence.
Run it from the protected v5007 120M reference:

`python run_ladder.py --stage capability --device cuda`

Default campaign:

- 9 primary branches × 60M branch steps;
- 20-point fixed matched held-out cloud panel at each 10M evaluation;
- 60-point final matched panel;
- 256 envs/point, 250 control steps/point;
- old hard100/70/140 suite disabled by default (regression-only, not capability);
- top 3 families repeated at seed +1009;
- third seed for top-two held-out recovery difference <0.03;
- selected family continues to 120M absolute branch steps.

The held-out schedule is identical across branches. Each 20-point routine panel
contains two permanent command anchors plus two rotating scout anchors, so known
coordinates are tracked through policy age while the map gains new command
combinations. Every command bundle contains identical `(vx, vy, yaw)` under all
five contexts: `none`, `pose`, `momentum`, `impact`, `chaos`. A disturbed point
only counts as valid recovery evidence when its matched `none` command is itself
competent (>=80% dynamic recovery and >=95% survival). Probe directions are
stratified over eight bins inside the vector batch; each point stores mean and
worst-direction recovery. The 60-point final panel rechecks the first 12 command
anchors densely for branch/seed comparison.

Per-rung evidence:

- `heldout_capability_cloud.csv` — complete longitudinal paired points;
- `heldout_capability_cloud.json` — compact navigation summary;
- `fixed_eval.csv` — historical regression ruler;
- `metrics.csv` — training/governor/frontier telemetry;
- `performance.json` — timing attribution when profiling is enabled.

Recovery branches still use ONE resilience difficulty scalar. New static physical
maxima define what difficulty=1 means for each experiment family:
`--recovery-tilt-max-deg`, `--recovery-position-max-m`,
`--recovery-linvel-max-mps`, `--recovery-angvel-max-rad-s`, and
`--recovery-joint-max-deg`. `--reset-resilience-difficulty` may reset only that
scalar after champion warm-start so new recovery worlds ramp in rather than
arriving at full checkpoint difficulty immediately.

Do not interpret the paired cloud as the final atlas. It is continuous held-out
navigation evidence. The dense atlas remains the final-boss physical audit after
the campaign winner is known. Passive ordinary-rollout harvesting and near-fall
replay are still future *training-cloud* layers and remain separate from the
held-out cloud.

## Stage-5.2 performance rule

Treat `train_fps` and `wall_fps` separately. `train_fps` measures the compiled PPO update path; `wall_fps` includes held-out evaluation and is expected to dip after an evaluation event. Do not infer a GPU slowdown from wall FPS alone.

Capability-cloud evaluation is now baseline-gated and batched. A disturbed probe is executed only after its exact no-disturbance `(vx, vy, yaw)` control meets the established competence gate (recovery >= 0.80 and survival >= 0.95). This both preserves causal interpretation and avoids spending GPU time on disturbance probes that would later be marked invalid anyway.

Default serious-ladder cadence is cloud every 10M steps and legacy fixed regression every 30M steps, with forced final evaluation retained. Phase-2 directories are resumable in place from `checkpoints/latest_mjx.pkl`; do not delete an interrupted phase-2 run merely to adopt evaluation-only performance changes.


## v5009 — evaluator-throughput generation

v5009 is a deliberate experiment-definition boundary from v5008 Stage 5. It keeps the learned-policy, reward and physical-training doctrine, but changes the cost and scheduling of evaluation: fixed regression checks are less frequent; capability-cloud controls gate pointless disturbance probes; compatible cloud points are batched; and logs distinguish raw training FPS from effective wall-clock FPS.

Use `--import-v5008-phase2` to continue the single existing v5008 phase-2 checkpoint into a fresh `emu_runs/v5009` directory. This preserves the policy/optimizer/governor state and source seed while keeping v5008 and v5009 evidence separate.

## v5010 trainer qualification

Primary experiment:

```bash
python run_ladder.py --stage trainer --device cuda
```

v5010's live capability field is **training-derived** and may steer future tasks. The held-out capability cloud is **evaluation-derived** and must never feed the live field. Preserve this separation when extending the system.

The 20M shared foundation is a true cold start and is forked with `--resume`, not warm-start, so actor/critic, optimizer, observation normalization, governor and capability-field state are identical at the three-way comparison split.

See `v5010_design.md` for the full architecture and experiment doctrine.


## v5010 — continuous self-mapping trainer

v5010 makes cold-start trainer quality the primary experiment. One genuine random-weight 0→20M nursery builds locomotion while a decaying vx×vy×yaw capability field observes in shadow mode. The exact 20M checkpoint then forks into shadow, node-steered and node+maneuver-graph trainers which each continue to 120M.

The live field is built continuously from ordinary PPO rollout metrics already transferred to the host. It stores current command competence, a generic command-conditioned recovery-severity layer, and 26-sector local command-transition/maneuver evidence with a 20M-step exponential half-life. `trainer_nodes` uses a bounded fraction of node anchors to choose useful command geography and locally useful resilience severity, always clipped by governor-earned safety limits. `trainer_graph` additionally supplies local source→target transition anchors.

The governor remains the high-level safety/controller authority. The field answers *where* practice is useful; it does not replace collapse detection, global physical frontiers, PPO/KL safeguards or rehearsal policy.

Held-out cloud reviews are quarantined. v5010 selects calibration commands from the live field, then tests them with fresh evaluator seeds/directions, but those outcomes never update the live field. The point is to test whether the trainer's self-map is honest, not to let it study the exam.

Primary command: `python run_ladder.py --stage trainer --device cuda`. Use `field_viewer.py` against a rung directory to inspect the live map without affecting training. See `v5010_design.md` for the full architecture and experiment doctrine.
