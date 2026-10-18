# EMU v5016 pinnacle: staged 12-hour sensor training

The 20M 232-input foundation is already trained. This is a **new** campaign
that borrows its exact schema-8 network, value function, optimizer, observation
normalizer, curriculum, field and learning-rate state; it never initializes
another zero-weight policy. The original foundation and the old 220-input
showcase remain untouched.

## One-command launch on WSL

From a **clean** `feature/v5016-sim2real` Git checkout:

```bash
cd ~/emu
git switch feature/v5016-sim2real
git pull --ff-only

# Read-only, validates original 20M checkpoint:
emu campaign5016 --plan

# GPU: automatic 12-hour soft-budget campaign:
emu campaign5016 --run
```

Default root: `~/robot_lab/emu_runs/v5016_pinnacle/`. Default source:
`~/robot_lab/emu_runs/v5016_sensor_foundation_20m/`.

GPU training uses **2048 MJX worlds, seed 733**, 32-frame PPO batches and
the existing v5015 learning loop. Deterministic held-out exams use 128
worlds at 300 frames and use the exact same actor sensor schema.
The 12-hour budget is measured using successful process wall time and
screening wall time, and checked **between stages** (no dangerous GPU
signal or partial file write in the middle of PPO). Training can take
less or modestly more than 12 hours depending on CUDA and compilation.

## Campaign plan

| Phase | Actors | Absolute steps | New steps |
|---|---:|---:|---:|
| Baseline control vs three hypotheses | 4 | 80M | ~240M |
| Best two guarded finalists | 2 | 200M | ~240M |
| Successive, score-gated pinnacle | 1 per phase | 300/400/500/600/700/800M | up to ~600M |

Total nominal additional policy steps: **~1.08 billion**, at ~35k frames/s
corresponding to ~8.6 GPU training hours, plus held-out tests, compilations,
checkpoint writes and any retakes or fallbacks. The time guard may stop after
a smaller number of phases if the computer is slower than expected.

The independent first-round hypotheses share the same original foundation:

- `baseline`: mature forward/side/yaw/posture teaching, no added motor cost
- `grace`: the same, with mild motor effort weight 0.035
- `contact`: genuine varied contact-plane inclines, initial radius 0.40
- `combo`: extra simultaneous sideways/yaw command teaching

The final champion then faces progressively more real ground-angle variation
(terrain radius 0.12 to 0.90), combined movement commands, and modest
hardware randomisation in the last three stages. Training retains the existing
PPO's collapses/recovery safeguards and continuously logs servo mechanical
power, nominal torque overload and joint safety. These are **not actual
battery power or temperature measurements**.

## Champion selection

All stage comparisons use a single deliberately identical held-out condition:
a 15-degree posture command envelope, shared phase speed, fixed seeds, identical
sensor observation semantics, and matched `robust`, `growth`, `core`,
`contact_slope`, and `contact_harder` panels. The two latter suites use
**real contact-normal variation**, not tilted gravity pretending to be terrain.
The independent motor exam adds simulated shaft watts, overload exposure and
action smoothness; its weight is deliberately modest, so an immobile robot
cannot win merely by saving power.

Core survival, severe-disturbance survival, real incline survival, minimum
step evidence, healthy body height and foot support are hard eligibility
guards. A poor continuation doesn't overwrite the best previous policy.
Every tested checkpoint gets a recorded SHA-256 and scorecard. The winner
can therefore be at an earlier step than the longest trained lineage.

## Files, resume and recovery

```
~/robot_lab/emu_runs/v5016_pinnacle/
    campaign.json          # persistent campaign source + time budget
    champion.json          # exact chosen .pkl path and SHA-256
    results.md             # full leaderboard and rollout provenance
    results_data.json      # machine-readable campaign state
    results_bundle.zip     # shareable metrics, NOT model weights
    scoreboards/           # named evaluation panels by immutable policy SHA
    round1/                # baseline/grace/contact/combo checkpoints
    round2/                # selected finalists and checkpoints
    polish/                # each 100M refinement and checkpoint
```

Every stage has its own `console.log`, `metrics.csv`,
`fixed_eval.csv`, `checkpoints/{latest,best,recovery,final}.pkl`
(where applicable), `training_config.json`, and `results.md`.
The artifact ZIP intentionally excludes the potentially large trained weights.

```bash
# Read-only summary:
emu campaign5016 --status

# Safe continuation after reboot, failure or Ctrl+C:
emu campaign5016 --run

# Read-only planning without any files or GPU:
emu campaign5016 --plan
```

The campaign pins the **feature branch source commit**, the foundation
checkpoint SHA-256, the v5016 sensor schema fingerprint, seed, environment
count, test difficulty and GPU budget. Do not edit code, change settings,
move the parent or overwrite checkpoints mid-campaign. Starting a different
experiment means choosing a new `--root`.

The `--run` flag is mandatory to start GPU work; all normal stage outputs
are outside the Git checkout. This launcher never commits or merges `main`.
The v5016 232-input actor remains incompatible with the existing 220-input
legacy 220-input physical examiner. Local browser replay now uses a separate schema-2 sensor emulator; it is visual inspection, not a bit-exact held-out physics evaluation.

**Important:** screen-test improvements are simulated evidence; physical
robot torque/current and real ground sensors still need hardware validation.
