# EMU v5010 — Self-Mapping Trainer

## Thesis

v5010 treats the trainer as the product.  A good checkpoint is not enough: starting from random actor/critic weights, the current code must be able to rediscover locomotion and then spend a fixed GPU budget intelligently expanding command, recovery and maneuver capability.

The key architectural change is to reuse ordinary PPO experience as continuous capability review.  The held-out cloud remains separate and is used only to calibrate/check the trainer's self-map.

## The live capability field

The primary geometry is a low-resolution command lattice:

- `vx`: 21 bins from -0.65 to +1.20 m/s
- `vy`: 15 bins from -0.65 to +0.65 m/s
- `yaw`: 25 bins from -5 to +5 rad/s
- total: 7,875 command nodes

Each node stores decaying evidence for:

- ordinary command competence
- confidence / effective evidence
- slow competence reference, allowing learning vs forgetting to be distinguished
- recovery competence over six resilience-severity bins

Maneuverability is represented as edges, not extra map dimensions.  Every command node has 26 signed 3-D maneuver-direction sectors.  Ordinary transition ramps update edge evidence and normalized command-rate evidence.

Velocity therefore remains a node property; agility is an edge property.

## Continuous review

Every PPO update already transfers one rollout metric pytree to the host for logging/governor decisions. v5010 reuses it.

With the default 2048 envs × 32 frames, every update supplies 65,536 fresh simulator frames. Healthy settled frames add small positive/negative competence evidence. Real falls add a heavier negative impulse. Transition frames also update maneuver edges.

No extra simulator pass and no per-death Python callback is introduced.

Evidence has a default 20M-training-step half-life. Old mastery therefore becomes uncertain rather than remaining permanently true.

## Teaching packet

The full field never enters the actor observation and is never passed through the policy network. The host field selects a compact fixed-size packet each PPO update:

- 24 command-node anchors: `(vx, vy, yaw, weight, local recovery severity)`
- 24 local maneuver anchors: `(source vx/vy/yaw, target vx/vy/yaw, weight)`
- bounded node and edge influence fractions

This packs to a fixed 323-float environment parameter vector, preserving JAX shapes and avoiding recompilation.

The existing RegimeGovernor retains ownership of global safety, difficulty retreat, physical frontiers, collapse recovery and broad allocation. The capability field chooses geography inside those limits.

## Field modes

`off`
: no field behavior.

`shadow`
: builds exactly the same field/teaching packet but steering fractions are zero. This is the causal control and pays the same instrumentation overhead.

`nodes`
: after the cold-start nursery, a bounded fraction of births and transition targets come from productive command-space anchors. For resilience births, the same node anchor proposes a locally useful recovery severity, always clipped to the governor-earned global resilience ceiling.

`graph`
: node steering plus source-conditioned maneuver-edge targets. This is the full agility/self-mapping hypothesis.

## What the node sampler prefers

The live field distinguishes:

- productive frontier: roughly 35–80% current competence
- rapidly learning regions
- recently regressing/forgotten regions
- unknown cells adjacent to established competence
- a smaller amount of mastered rehearsal

Unknown exploration hugs the known competent boundary rather than teleporting to arbitrary command corners.

## Held-out review

The periodic held-out cloud does not train the field.

Instead, v5010 asks the field to nominate a few commands spanning its own predicted competence bands (roughly 90/75/60/45/30%). Fresh evaluator seeds then test those exact commands. The result is a calibration error plus the existing matched-control disturbance evidence.

Held-out answers remain quarantined: they are written to the cloud evidence but never update the live field.

Legacy fixed regression evaluation is deliberately sparse in trainer mode: one mid-run check after the 20M fork plus the forced 120M final. It is continuity insurance, not capability discovery.

Default held-out cadence is now 20M steps with a small 15-point requested routine panel (three matched command bundles) and 50-point common final panel. Routine calibration commands are field-selected; the final panel is deterministic and identical across trainer branches so branch ranking is apples-to-apples.

## Cold-start trainer experiment

Primary command:

```bash
python run_ladder.py --stage trainer --device cuda
```

The experiment is deliberately large.

### Shared seed-733 nursery

`0 → 20M`

- random actor and critic
- fresh Adam optimizer
- fresh governor
- fresh observation normalizer
- capability field observes in `shadow`
- conservative proven cold-start curriculum

At 20M the literal `latest_mjx.pkl` is the fork source.

### Fair 20M → 120M comparison

Three branches `--resume` the exact same checkpoint, preserving network, optimizer, normalizer, governor and capability field:

1. `trainer_shadow` — field observes, no steering
2. `trainer_nodes` — command geography steering
3. `trainer_graph` — command geography + maneuver-edge steering

Field influence ramps from 0 at 20M to at most 0.65 at 30M instead of switching on abruptly.

The 20→120M window is the actual trainer benchmark: how much capable robot can each teaching architecture produce with the same additional ~100M simulation budget?

### Independent cold confirmation

By default the winning strategy receives seed `733 + 1009 = 1742` and gets its *own* independent 0→20M nursery followed by 20→120M continuation. No checkpoint from the first seed is reused.

This is the proof that the winning architecture remains a trainer rather than a checkpoint-polishing recipe.

### Optional asymptotic extension

```bash
python run_ladder.py --stage trainer --trainer-extend-winner --device cuda
```

continues the selected confirmed trainer to 220M absolute steps, separating the 120M limited-compute benchmark from the “what can it eventually become?” question.

## What is deliberately not in v5010

- near-fall replay/reset bank
- per-env mass/friction/actuator-domain randomization
- footing/slip family as a dedicated field layer
- a learned neural curriculum controller

Those remain attractive next steps, but adding them now would make the self-mapping trainer experiment causally muddy.

## Files produced by each trainer run

- `metrics.csv` — ordinary PPO/governor/field telemetry
- `capability_field_latest.npz` — current command/recovery/edge arrays
- `capability_field_final.npz` — terminal field snapshot
- `capability_field_summary.json` — compact current field statistics
- `heldout_capability_cloud.csv/json` — quarantined calibration/exam evidence
- `fixed_eval.csv` — legacy regression continuity evidence
- normal checkpoints including the capability field state inside `governor_meta`

## Design invariant

Warm starts remain useful research shortcuts, but cold start is the product.

A feature is not considered a successful EMU trainer improvement merely because it improves a mature checkpoint. It must preserve the ability to build a strong robot from random network weights under a bounded compute budget.
