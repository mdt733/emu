# EMU architecture — ownership and trust boundaries

The project separates components because they answer different questions; changes should flow through explicit contracts rather than cross-module mutation.

| Component / owner | Responsibility |
|---|---|
| `emu/environment.py` | MuJoCo/MJX physics, task packets, resets, rewards, termination and deployable observations |
| `emu/ppo.py` | Actor/critic, observation normalisation, PPO rollout and update |
| `emu/curriculum.py` | Legal difficulty/frontiers and retreat/advance policy |
| `emu/health.py` | PPO health, collapse detection, learning-rate stress and mature rollback recommendations |
| `emu/capability_field.py` | Semantic node, maneuver-edge and recovery evidence; bounded teaching proposals |
| `emu/evaluation.py`, `emu/capability_cloud.py`, `emu/atlas.py` | Read-only fixed tests, calibration and physical frontiers |
| `emu/training.py` | Composes the update loop and own-run artifacts; never lets held-out evaluators train the field |
| `emu/production.py` | The single v5014 20/60/140/260/400M learning lineage and exact phase-config compatibility |
| `emu/domain_randomization.py` | Batched JAX-valued robot/terrain/hardware model variation per training world |
| `emu/contracts.py`, `emu/tracking_contract.py` | Shared deployable 55×4 observation, six action targets, task packet and physical event thresholds |
| `emu/checkpoint_io.py`, `emu/checkpoints.py` | Validated schema-7 checkpoint save/read, atomic replacement and finite-policy protection |
| `emu/policy_export.py` | Actor-only JSON `.emupolicy` export; excludes critic/optimizer and held-out training state |
| `emu/live_viewer.py`, `emu/viewer_server.py` | Loopback-only, CPU/read-only validated checkpoint feed and viewer build/server |
| `viewer/` | Canonical MuJoCo WebAssembly + Three.js UI, local inference, manual/demo input and static showcase |
| `scripts/prepare_showcase.py` | Packages a static manifest and policy for browser-only hosting, without training or Python API |

## Important invariants

The **capability field** proposes *where* to practise; the **curriculum/governor** decides *what is currently permitted*; the **domain randomiser** samples *what physical conditions exist*. The **held-out examiner** never updates any of these. `emu/training_config.py` holds the narrow varying experiment surface and fixed doctrine; derived reporting-only state should not be duplicated in checkpoints.

The actor sees only the canonical **220-number** history vector; the inference actor is 220→256→256→6 with tanh hidden layers. `.emupolicy` is not a trainer resume file: only schema-7 `.pkl` checkpoints can continue PPO. See [hardware-model.md](hardware-model.md), [capability-map.md](capability-map.md) and [viewer/POLICY_FORMAT.md](../viewer/POLICY_FORMAT.md).

## Process boundaries

The trainer owns CUDA/JAX and writes under `../emu_runs/`. The local `emu viewer --live` is a **separate read-only CPU watcher** serving only on `127.0.0.1:8765`; browser JavaScript runs inference and nominal MuJoCo dynamics. GitHub Pages publishes a **different static bundle**, and has no private training API. The viewer demonstration is not an evaluator.

`emu/experiments.py` and the `train/compare/tournament` CLI remain useful for reproducing previous controlled experiments, but do not constitute the default production pathway. Git history and compact results are the archive; current code should not accumulate new switches solely to relive old experiments.
