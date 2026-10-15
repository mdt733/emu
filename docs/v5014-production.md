# v5014 production lineage — historical reference

> Archived training recipe. The original `emu forge` implementation is preserved in Git history (e.g. commit `b0f31d8`), not in the current executable. These settings are **not** the active v5015 ladder. Do not use a historical checkout against a current v5015 run directory.


## Purpose

Train one deployable policy rather than select among a tournament of branches. The lineage preserves network, optimizer, observation normalizer, curriculum/health and capability-field state across physical-domain phases.

## Lessons carried from v5013

- Low capability-teacher dose (10%) was the clearest repeatable signal; higher 25/40/65% doses did not justify their extra influence.
- Nodes-only versus full-graph teaching was mixed across seeds, so v5014 does **not** canonize the nodes-only result. It uses full graph at only 10%.
- Teacher choice and physical-domain difficulty remain independent. The capability map can bias *what skill to practice* but does not choose how inaccurate the robot/world model is.
- Held-out evaluation remains quarantined.

## One lineage

| phase | absolute target | robot R | terrain T | hardware H |
|---|---:|---:|---:|---:|
| foundation | 20M | 0.00 | 0.00 | 0.00 |
| adapt | 60M | 0.10 | 0.08 | 0.12 |
| breadth | 140M | 0.30 | 0.25 | 0.35 |
| robust | 260M | 0.55 | 0.50 | 0.60 |
| generalist | 400M | 0.75 | 0.70 | 0.75 |

A phase boundary is a process/JIT boundary, not a learning reset. Resume restores learning state. Each phase compiles once and then runs at full MJX throughput.

The batched domain fast-path currently covers slope-equivalent gravity orientation, floor/foot friction asymmetry, body mass spread, actuated-joint damping/friction/armature, actuator gain/torque and servo zero offsets. Coherent morphology-bank variants and low-frequency heightfield roughness remain the next integration target because they need distinct performance treatment.

## Operator pitch

`cmd_pitch` already existed in the 55-channel deployable frame in v5013. v5014 makes it intentional: task generation uses up to ~8 degrees and reward softly tracks chassis pitch relative to local gravity. Survival and locomotion remain more important than exact camera pointing. The observation contract stays 55 x 4 = 220.

## Browser deployment

`emu export-policy checkpoint.pkl policy.emupolicy` strips training-only state and embeds actor weights, normalizer, control contract and robot XML. `emu viewer` serves browser assets plus a small **loopback-only JSON API** for an initial policy or validated live checkpoint; neural inference and MuJoCo physics execute locally in the browser. The public static Playground uses a bundled policy and has no Python API.

## Live training viewer

The local viewer can be launched alongside production training with `emu viewer --live`. The server searches each production phase in newest-first order for complete `latest.pkl`, `final.pkl` or rotated snapshots. Checkpoint I/O commits atomically; the read-only feed validates the schema and exports a cache only on a changed file. When a write is incomplete or the checkpoint is invalid, the last validated actor stays in the browser rather than being replaced by garbage.

The viewer polls roughly every 12 seconds. A compatible actor replaces weights without rebuilding the WebAssembly MuJoCo model; each replacement begins a clean simulator episode so visualization is not misrepresented as an uninterrupted rollout. Domain/control/robot XML changes trigger a full model reload. The browser's standalone policy selection and manual controls continue to work.

The demo sampler draws commands from a compact, training-field-evidenced node list whenever available, requiring at least three effective events and predicted success >=0.70. Only when that list is empty does it sample deliberately below command frontier limits. This is **a training-derived self-assessment, not held-out confirmation or safety certification**; transitions between nodes are not guaranteed even when the individual nodes looked competent in training.

Checkpoints carry no new browser-rewind state. The exported `.emupolicy` contains just up to 96 small command candidates alongside existing actor/normalizer/control data. Browser telemetry never teaches the field. The viewer is not an evaluation engine.

The live server listens only on loopback, because its checkpoint feed lacks authentication. The viewer must be built with Node/npm on a development machine. The operator has confirmed successful GPU domain smoke, Vite WASM build, and local browser/live operation; see [../RELEASE_VALIDATION.md](../RELEASE_VALIDATION.md). Those successes do **not** establish complete 400M convergence, held-out recovery, or physical-hardware transfer.

The Python watch process sets JAX to CPU before importing checkpoint validation. This deliberately prevents an otherwise innocent viewer from initializing or reserving the GPU memory used by the active learner.
