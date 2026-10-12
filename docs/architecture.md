# Architecture

EMU keeps six runtime responsibilities deliberately separate.

## Environment

`emu/environment.py` owns MJX physics, resets, task sampling, disturbances, reward, termination and deployable observations. The default trainer advances 2,048 worlds in parallel.

## PPO

`emu/ppo.py` owns the actor/critic network, observation normalisation, rollout collection and PPO update. A rollout is consumed once; EMU has no replay buffer.

## Curriculum

`emu/curriculum.py` owns regime difficulty/allocation and demonstrated physical frontiers. It answers **how hard is globally legal now?** Curriculum retreat is a task-distribution decision, not a neural-network rewind.

## Training health

`emu/health.py` owns PPO/KL stress and sustained-collapse detection. It may veto unsafe curriculum expansion or request rollback, but it cannot mutate the task distribution itself.

## Capability field

`emu/capability_field.py` turns ordinary training experience into semantic node, recovery and maneuver events. It answers **where inside the legal envelope would practice be useful?** The field may propose bounded teaching coordinates; held-out examination never updates it.

## Evaluation

`emu/evaluation.py` owns the deterministic fixed regression panels. `emu/capability_cloud.py` performs sparse held-out node/recovery exams, and `emu/atlas.py` reports physical command/recovery curves. Evaluators receive policy snapshots and return evidence; they have no path back into PPO, curriculum, health or field mutation.

## Trainer and experiment runner

`emu/training.py` is orchestration: it composes the owning modules into one learning loop, manages run-local artifacts and performs complete rollback when health requests it. `emu/experiments.py` knows only the modern experiment shape: random-start foundation, exact fork, then shadow/graph continuations. Git is the archive for historical ladders and ablations.

## Shared contracts

`emu/contracts.py` owns deployable observation dimensions and the fixed-shape JAX task-packet layout. `emu/tracking_contract.py` owns physical competence thresholds. One contract is defined once and consumed by every implementation that needs it.

## Configuration and process boundary

The public CLI exposes experiment-level choices. `emu/experiments.py` creates a `TrainingConfig`, serialises it, and starts a clean trainer subprocess for JAX/GPU isolation. `TrainingConfig` contains only values that genuinely vary between supported runs; invariant current doctrine is named in code and recorded in `run_meta.json`.

## Checkpoints

`emu/checkpoint_io.py` owns current-only schema 7. A rewind point contains exactly:

- policy/value network and optimiser state;
- observation normaliser;
- curriculum dynamic state;
- training-health state;
- capability-field state;
- metadata/provenance.

Resume is strict at every layer. Partial, unknown or incompatible state fails loudly. Canonical identities are `latest.pkl`, `best.pkl`, `recovery.pkl`, `final.pkl` plus rotating snapshots.
