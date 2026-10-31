# How EMU works — code map and ownership

**For someone opening this repository for the first time.** The
[README](../README.md) explains the project and gives runnable entry points.
This page answers the next question: *which file owns each concept?*

EMU is a six-servo **dynamically unstable biped**. It learns a single
neural controller by PPO in thousands of MuJoCo/MJX worlds. It is not
a collection of hand-programmed walking animations or six specialist
neural networks.

## The whole system at a glance

```text
 operator / experiment plan
   scripts/train_pinnacle.py        one 260M cold-start campaign, source guard
           |
           v
   emu/cli.py -> emu/run.py         validated train CLI, config + checkpoint guard
           |
           v
   emu/training.py                  one shared rollout/update/checkpoint loop
      |              |                  |
      v              v                  v
   environment.py   rewards.py         ppo.py
   robot.xml        teaching payment   actor, critic and optimiser
   actual physics       ^                  |
   sensors/contacts     |                  |
      ^                 |                  v
      |             curriculum.py <--- measured rollout outcomes
      |             capability_focus.py  teacher tasks, six focus groups
      |             capability_field.py  host-only competence evidence
      |
      +-------- commands for NEXT episodes, never exam results

 read-only held-out examination      saved outputs
   evaluation.py                      checkpoint_io.py
   physical_exam.py                   results.py
   capability_cloud.py                    |
          |                               v
          +--> measured physical facts -> .pkl --> export-policy --> .emupolicy
                                                                  |
                                                                  v
                                                viewer_server.py / browser JS
                                                local real-time MuJoCo replay
```

The diagram is conceptual, not an import graph. In particular, the
**teacher** chooses practice requests and uses simulator-only truth
to judge them. The **actor** gets only deployable observations.

## Code ownership: one place to look for each question

| Question | Primary owner | Scope |
|---|---|---|
| What is EMU's body? | `emu/robot.xml` | Geometry, joint mechanics, six motors, contact material |
| What does the actor see and output? | `emu/contracts.py` | Named observation fields, sensor revision, 4-frame history, 6 actions, hashes |
| What physical world is it in? | `emu/environment.py` | Batched MJX worlds, slopes/step geometry, disturbances, sensor emulation, contact facts, reset |
| What does a successful training frame pay? | `emu/rewards.py` | Scalar *teaching reward*, distinct from physical exam |
| What is practised next? | `emu/curriculum.py` and `emu/capability_focus.py` | Difficulty frontiers, retreats, six teacher emphases, task allocations |
| How is ongoing competence remembered? | `emu/capability_field.py` | Host-side training evidence; not a network input |
| How are the weights optimised? | `emu/ppo.py` | Shared actor/critic, observation normalisation, PPO gradient |
| What runs each update and saves progress? | `emu/training.py` | Rollout orchestration, logging, recovery, checkpoint cadence |
| Who permits a particular resume/source? | `emu/run.py`, `emu/training_config.py` and `emu/checkpoint_io.py` | CLI configuration, strict checkpoint validation, saved training state |
| How do we judge the bird independently? | `emu/evaluation.py`, `emu/physical_exam.py` and `emu/capability_cloud.py` | Fixed tests, real supported steps, requested-vs-achieved speed, held-out recovery |
| Where are measurable results written? | `emu/results.py` and `scripts/campaign_support.py` | CSV/JSON/console, campaign summaries, exportable *reports* |
| How is a robot played in the browser? | `emu/viewer_server.py` and `emu/viewer_src/` | Portable actor export, live checkpoint watching, MuJoCo/WASM, controls and terrain |
| Where does the training campaign start? | `scripts/train_pinnacle.py` | One pinned cold-start recipe; preflights, train, export, exams; **not** auto-release |
| How is the published demo protected? | `scripts/verify_showcase_release.py`, `scripts/prepare_showcase.py`, `.github/workflows/emu-playground.yml` | Approved policy identity and explicitly controlled GitHub Pages deployment |

Useful secondary files: `emu/directional.py` evaluates heading,
contacts and velocity/ground sensing; `emu/terrain_spike.py`
contains tiny real-physics acceptance experiments. They do not
define an alternative walking policy. `emu/viewer_src/src/main.js`
owns UI, `sim.js` owns browser MuJoCo physics, `sensor.js` owns
browser sensor emulation, and `terrain.js` owns viewer terrain
geometry. The viewer has its own physics and **does not secretly
steer the training actor**.

## Three kinds of information — do not confuse them

1. **Actor observations** are the 360 numbers fed into the policy:
   90 fields per 20 ms frame × 4 history frames. They include
   commands, proprioception, inertial/flow estimates and the 4×4
   forward ToF ranges *and validity bits*. The actor outputs
   **six servo targets**. See `contracts.py`, not a hard-coded
   interpretation of an array slice.
2. **Teacher-only facts** include exact physical contacts,
   terrain elevation and curriculum goals. They can affect
   *reward* and *next lesson*, never be smuggled into the actor.
   The six specialist focuses share **one** policy and optimizer.
3. **Held-out examination** runs a frozen exported policy on
   independent, specified physical tasks. It **never teaches the
   curriculum**. Teacher difficulty, reward, survival and
   successful supported terrain crossing are different metrics.

A 3 cm configured step is not a 3 cm step *climbed*. Requested
1 m/s is not 1 m/s *achieved*. A torso staying upright at the lip
is not a supported crossing. Yaw oscillation during walking can
be normal; sustained heading/command bias is what matters.

## Files produced, and why they are different

```text
 ~/robot_lab/emu_runs/v5032_pinnacle_s761/
   campaign_manifest.json      exact recipe, seed and source SHA
   preflight_pass.json          source-matched CPU and CUDA acceptance
   logs/                        readable raw training and exam output
   pinnacle_cold260/
     training_config.json       resolved PPO/curriculum settings
     run_meta.json              training Git/source and lineage identity
     metrics.csv                time series of observed training
     fixed_eval.csv             comparable frozen exams by age
     checkpoints/
       latest.pkl               *full* resumable learning state
       final.pkl                completed full training state
   pinnacle_candidate_260M.emupolicy
                               inference-only, browser-compatible actor
   candidate_provenance.json    SHA256 source/checkpoint/export identity
   physical_exams/              independent command/slope/step reports
   overnight_results.zip        shareable logs/reports, no learned optimizer
```

A `.pkl` includes optimizer state, normalizer, curriculum/evidence,
step and provenance; treat it as a private research checkpoint.
A `.emupolicy` contains **deployable actor** weights, sensor
identity and its exported training metadata, *not* optimizer
state. Git holds the **code**, not the full checkpoints. To
reproduce an old learned policy you need its saved checkpoint
and its exact source SHA, not just the latest `main`.

## Trace one common task from start to finish

**Train:** `python scripts/train_pinnacle.py` displays a plan
(no writes) → `--run` checks source/manifest and full 2048 CUDA PPO
preflight → `emu.cli train` validates lineage → `training.train`
runs the one PPO learner → durable checkpoints and reports.

**Examine:** `emu capability-exam --policy ... --terrain-matrix`
loads the frozen actor, creates **real collision surfaces**, measures
supported up/down crossing and actual tracking, writes independent
results and does not change policy weights.

**Watch:** `emu viewer --live --root ...` observes saved checkpoint
updates and serves validated snapshots to the local browser.
The JavaScript viewer runs a separate MuJoCo simulation; its
graduated park is an interactive diagnostic, not the production
training world and not an independent certification.

**Release:** `export-policy` creates a candidate. Only an explicitly
reviewed actor with source/checkpoint hashes and matching physical
evidence can replace `showcase/champion.emupolicy`. A protected
GitHub Pages workflow keeps publication separate from training.

## Editing rules for the next contributor

- Change **one owner** for a concept, don't add a second trainer,
  reward evaluator or policy schema for a minor experiment.
- Reward/physics/actor changes start a **new source-pinned training
  lineage**. Never pull code into a checkout mid-training or
  transplant an incompatible optimizer checkpoint.
- Test the smallest relevant behavior, real MJX contacts,
  checkpoint resume/export, independent exams and browser parity.
  For actual 2048-world CUDA use the source-specific acceptance smoke.
- Keep prior public policy and release approval untouched until
  the owner explicitly signs off the successor.
- Put completed studies in historical docs/Git refs; keep
  [README](../README.md), [HANDOVER](../HANDOVER.md), this page and
  the [current runbook](PINNACLE_CANDIDATE.md) about **today**.

The [engineering hygiene rules](MAINTENANCE.md) make the same
ownership and release boundaries operational.
