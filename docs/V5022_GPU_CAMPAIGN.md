# v5022: matched real-terrain GPU campaign

**Status: experimental branch-only learning trial.** Neither `main`, the
protected 60M/400M policies nor existing 220/232-channel actor contracts
are edited. No auto-promote and no simulated-only "step" reward.

## The proven main is the starting point

Protected `main` is our behavioural reference (`2ec9e9db0a816c658131311129ef89ec7a595a12`).
This branch changes only opt-in terrain schema/environment paths and a separate
experimental runner. Every `pitch15_220` and `sim2real_232` environment
continues to load the original `robot.xml`, receive the old observations,
and use the original PPO, reward, task and controller code.

The v5022 experiment now obtains its command limits, phase dynamics, push
strength, difficulty steps and regime allocation from **main's own
`foundation_config()`** rather than reinventing easier/harder settings.
The deliberately selected `v5015` posture profile is retained (main's
general-purpose foundation helper defaults to the older v5014 profile),
and new sensor/collision geometry is the explicit experimental change.

**Long GPU runs now call main's existing `training.train()` lifecycle**,
rather than a newly abbreviated PPO loop. The existing `TrainingHealth`
adaptive KL controller, curriculum retreat/recovery, validated checkpoints,
capability-field *shadow* recorder, snapshot retention, normalisation and
fixed-evaluation machinery are reused. The small A/B harness remains ONLY
for optimizer and memory smokes. Shadow evidence does not steer lessons.

A positive smoke or terrain result does not make this new observation
schema a production release: the experiment still needs independent
flat-ground baseline comparisons, terrain held-outs and a version-aware
viewer/export before any promotion.

**Baseline gate:** the existing baseline Python and browser suites must
still run; a new experiment doesn't justify rewriting older policy
files or old contract tests. CI currently has five previously known
legacy schema-1 examiner failures against the protected schema-2 champion,
which must be reported separately rather than papered over.

## Main-baseline continuity: mandatory invariant

This experiment is an **add-on to main's 700-commit learned engineering
foundation**. `emu.training.train` is the only supported long-run
learner. It owns the original RegimeGovernor/TrainingHealth, KL adaptation,
frontier retreats, capability field, full-schema validated checkpoints,
automatic recovery and canonical fixed examinations. The A/B wrapper
selects the visibility flag and strict source identity; it **must not**
implement a second long-term PPO update, governor or checkpoint loop.
Older 220/232 training remains byte-for-byte identical on its original
schema/geometry paths; the new 364 schema is separately fingerprinted
as checkpoint schema 9 and can't load as either old schema.

A clean run should begin with level-ground survival and walk competence.
A raised-step crossing score has no meaning if one arm never learns to
stand or walk on the baseline. Do not disable TrainingHealth retreat/
recovery to increase apparent reward. Stop/review if the core learner
fails its survival and stepping gates, rather than paying millions of
steps for a physically incapable policy.

## The actual questions

1. **Does physical forward terrain sensing help a robot learn to cross a
   real 5cm step-up and step-down ledge?** Compare matched visible/blinded
   policies with the SAME 364-input network and real collision world.
2. **Can it keep the old skills while using this new information?**
   Compare held-out survival, forward/lateral/yaw tracking, height,
   support and walking/step evidence; log early retreats/collapses.
3. **Will feet in the near-centre zones poison or help training?**
   Keep their *actual raw range returns*, plus validity/age. No oracle
   foot tags or privileged floor-height inputs. A separate later
   ablation may test whether their presence is informative.
4. **Can the new 64-ray model run efficiently on the RTX 4060 Ti 8GB?**
   Run one genuinely full-sized PPO optimizer update before investing
   millions of robot-steps. Report throughput; use 1024 worlds if 2048
   does not fit.

## Physics and observations

- **Fixed geometry, first phase:** common MuJoCo scene with a raised
  **5cm platform** (one real step-up, one real step-down edge), a nearby
  **4.5cm crest with finite up/down ramps**, and the ordinary ground.
  All worlds have identical topology; this first campaign does not yet
  randomly distribute logs, holes, slopes or arbitrary-height steps.
- **Chosen optical direction:** 0mm forward, 70° downward. Optical
  centre is 40mm below the torso-root Z plane, rather than the
  earlier 18mm (which was physically *inside* torso_box). The current
  physical box has 25mm half-height, leaving 15mm geometric underside
  clearance. This provisional bracket is **not a real CAD guarantee**.
- Raw 16 slant ranges + 16 validity flags + one capture-age channel per
  frame; 4 history frames = **364 inputs**. The existing 232 channels
  remain in their original order and meanings. Older actor/checkpoint
  formats cannot be reinterpreted as this schema.
- The true collidable ground, boxes and live two foot-ball positions
  generate geometric measurements. Noise: about 1.5mm; independent
  per-zone dropouts: 2%; capture: 50Hz; delivery delay: 1 control frame.
  Both arms generate identical sensor packets, but **blinded** replaces
  the new channels with zero/invalid values. Same PPO network size and
  same seed, curriculum and real physical obstacles.
- The original down-looking ToF and flow forward models are also given
  physical-course intersections. Only reward/exams can read exact
  support heights or labels. No future terrain heights in observations.
- The full **schema-9 checkpoint managed by main's checkpoint_io** stores
  actor, value function, optimizer, observation normalizer, teacher
  governor, training-health, capability field and physical sensor
  fingerprint. The A/B wrapper additionally enforces clean branch and
  exact source identity at resume.
  A resumed process starts NEW robot episodes; it does not pretend to
  replay the exact mid-episode simulator state.

## Copy-paste sequence (from the v5022 checkout)

```bash
cd ~/emu-v5022
git pull --ff-only origin feature/v5022-forward-terrain
source .venv/bin/activate

# Confirm independent 364-input contract, true MJX geometry and ray parity:
python -m pytest -q tests/test_terrain.py

# Print the experiment specification (NO learning):
python -m emu.v5022_experiment --plan --envs 2048 --steps 10000000

# CPU correctness gate: actual PPO optimizer and held-out course exam,
# on two tiny worlds/learner arms. No artifact/run is created:
python -m emu.v5022_experiment --smoke --device cpu --envs 4

# GPU gate A: one full-shaped PPO update (32 rollout frames, 4 epochs,
# 8 minibatches) with 256 worlds. No checkpoint or long run:
python -m emu.v5022_experiment --smoke --device cuda --envs 256 --arm visible

# GPU gate B: same real batch and network as the planned big run:
python -m emu.v5022_experiment --smoke --device cuda --envs 2048 --arm visible

# Full canonical main-engine gate, ONE 32-frame rollout in each arm.
# Unlike --smoke, this exercises main's real training loop, checkpoint
# writer, health/curriculum, source validation and held-out evaluation.
# Keep separate roots because identical manifests require identical git SHA.
python -m emu.v5022_experiment --run \
  --root ~/robot_lab/emu_runs/v5022_main_engine_gate01 \
  --arm visible --envs 256 --steps 8192 --seed 733 --device cuda
python -m emu.v5022_experiment --run \
  --root ~/robot_lab/emu_runs/v5022_main_engine_gate02 \
  --arm blinded --envs 256 --steps 8192 --seed 733 --device cuda

# Only after the smoke succeeds: two 10M-step cold-start arms.
# Each arm is a separate complete run with its own checkpoint/CSV/eval.
python -m emu.v5022_experiment --run \
  --root ~/robot_lab/emu_runs/v5022_terrain_ab01 \
  --arm both --envs 2048 --steps 10000000 --seed 733 --device cuda
```

If the 2048-world GPU smoke fails with OOM, do **not** start a larger
run anyway. Re-run the full-size smoke and **both** A/B arms at 1024:
```bash
python -m emu.v5022_experiment --smoke --device cuda --envs 1024 --arm visible
python -m emu.v5022_experiment --run \
  --root ~/robot_lab/emu_runs/v5022_terrain_ab01_1024 \
  --arm both --envs 1024 --steps 10000000 --seed 733 --device cuda
```

To resume an interrupted arm or extend a completed pair to 20M
cumulative steps, run the **same** command with the same root/seed/envs
and `--steps 20000000`. The source commit and actor fingerprint must
still match. Both arms continue their independent PPO lineages.

## What the output can, and cannot, say

Each arm writes an experiment identity JSON, a metrics CSV, the **canonical** `checkpoints/latest.pkl`, snapshots, recovery and
`checkpoints/final.pkl` (through `emu.training.train`), established
`fixed_eval.csv`, plus a separate `v5022_crossing.json`. A completed pair also writes a `summary.json`
with **visible-minus-blinded** measured deltas.

The held-out test uses an identical fixed-seed forward command. A
**true crossing** requires physically passing x=0.96m (beyond the
raised ledge) while alive and supported, without a reset contaminating
the result. Also inspect support, first-fall, command tracking and
forward progress. Curriculum rollout reward is not an independent
success criterion. This early fixed arena is NOT a comprehensive
generalisation test: logs, different obstacle placements, headings,
and harder cold-start robustness remain subsequent gates.

Do NOT crown a new champion, export a demo as if it were schema-2, or
promote to main from a positive training-reward delta. Review the
per-arm cross-step statistics and traces first; the exporter/viewer
must explicitly learn the new 364-input version before public promotion.
