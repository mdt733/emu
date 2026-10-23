# v5022: matched real-terrain GPU campaign

**Status: experimental branch-only learning trial.** Neither `main`, the
protected 60M/400M policies nor existing 220/232-channel actor contracts
are edited. No auto-promote and no simulated-only "step" reward.

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
- The full **experimental** checkpoint stores actor, value function,
  optimizer, observation normalizer, teacher governor, RNG key, source
  commit/branch/cleanliness and physical sensor fingerprint.
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

Each arm writes an experiment identity JSON, a metrics CSV, an atomic
`checkpoint.pkl` (after up to 1M steps and at termination), and a
`fixed_eval.json`. A completed pair also writes a `summary.json`
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
