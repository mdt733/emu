# v5053 — a one-rollout 2048-world physical capability screen

This is **read-only frozen-policy examination**, not another PPO run and
not an alternative robot. It does not change main or v5052 training.

## Why
In the v5052 eight-scenario trial, every arm recorded **zero completed
uphill destinations** (48 step worlds each). Some learned a valid first
top-contact, some independently landed both feet; the old simultaneous
double-stance examiner reported zero for all. There is insufficient evidence
to know whether insufficient detection, leg reach, dynamic balance, or
trailing-foot recovery is the dominant obstacle.

The scientifically useful next experiment is therefore a physical,
**known-height**, per-world stage exam, including an ordinary-walking
retention screen. A larger reward number is not stair-climbing proof.

## The batched physical protocol
- Same **360 actor input**, 6 servo commands, real 4x4 ToF, real
  eight convex MJX prisms; no privileged actor observations or optimizer.
- Each world starts centred in Y, **0.53m in X**. One signed
  0.75m tile boundary at X=0.75m. No procedural jitter, no random
  extra steps, and a fixed destination X=1.35m/Y=0.375m.
- Actual physical X-edge height is independently computed from **both
  real source-tile top surfaces**, not from requested frontier.
- Optional terrain families have an exact signed X grade, with
  shared vertex continuity at tile boundaries.
- All worlds share the same physical topology/model shape and one
  frozen actor but independently vary commands, terrain and small
  reset kicks.
- Results accumulate on GPU as **per-world scalars**, not an entire
  [frame,world,prism contact] trace, and the host groups them only after
  identifying the assignment of each world.

### Balanced screen
10 families × 20 difficulties × 10 attempts = **2,000 active worlds**
of the default 2,048. The other 48 are explicitly labelled padding.
Families: idle/kicks, slow forward, fast forward, lateral, yaw, combined
forward/steer, uphill grade, downhill grade, uphill step, downhill step.

### Stair-specific screen
20 exact **uphill** heights from **2.5mm to 50.0mm**, spaced 2.5mm,
100 independently seeded attempts at each = 2,000 worlds.
This is a real 2,000-world width and 350-frame JAX scan, not
20 sequential evaluations of 100 worlds.

### Outcomes
For each signed stair height, in order:
1. At least one genuine upper-tile ball-foot contact against the
   actual prism top (normal and matched foot bottom).
2. Each foot independently demonstrates **five consecutive** such top
   contact frames; simultaneous double stance is NOT required.
3. Body reaches 0.60m beyond the first lip and stays upright, supported,
   heading-controlled and low-slip for five consecutive frames.
4. Record survival, commanded velocity/yaw errors and actual progress.

The independent scoring code reads physical contacts and body pose,
not the Academy teaching reward or its completion flags. It refuses
the exam if requested and real surfaces disagree by >0.5mm, or
grades differ by >0.30 degrees. First-life flags prevent reset contamination.
Results include all trials, grouped height curves and Wilson
95% trial confidence intervals; those do not quantify PPO-seed uncertainty.

## Run — SAFE while v5052 training continues
**Do not switch the source-pinned directory used by a running v5052
trainer.** Use a **separate Git worktree** at the same repo:
```bash
cd ~/emu
# This clone may fetch only a narrow set of branches by default.
git fetch origin +refs/heads/research/v5053-capability-matrix:refs/remotes/origin/research/v5053-capability-matrix
git worktree add --detach ~/emu-v5053-matrix refs/remotes/origin/research/v5053-capability-matrix
cd ~/emu-v5053-matrix
# Reuse the existing working EMU venv. Don't refer to ~/emu-graduation.
source ~/emu/.venv/bin/activate
export PYTHONPATH="$PWD"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python -m pytest -q tests/test_core_capability_matrix.py

# Use the real v5052 trained frozen actor. Export on the v5052 checkout:
# Already exported on this workstation: ~/emu-data/actors/v5052_01_goal.emupolicy
python -m scripts.v5053_matrix_exam \
  --policy ~/emu-data/actors/v5052_01_goal.emupolicy \
  --output ~/emu-data/exams/v5053-01-goal-balanced-s1701 \
  --device cuda --worlds 2048 --horizon 350 \
  --mode balanced --seed 1701

python -m scripts.v5053_matrix_exam \
  --policy ~/emu-data/actors/v5052_01_goal.emupolicy \
  --output ~/emu-data/exams/v5053-01-goal-stairs-s1701 \
  --device cuda --worlds 2048 --horizon 350 \
  --mode stairs --seed 1701
```

Use a new output directory for each actor/seed/mode. The examiner refuses
overwriting a nonempty report. For a tiny acceptance before 2,048 worlds:
```bash
python -m scripts.v5053_matrix_exam \
  --policy ~/emu-data/actors/v5052_01_goal.emupolicy \
  --output ~/emu-data/exams/v5053-01-goal-smoke \
  --device cuda --worlds 8 --horizon 80 \
  --mode stairs --levels 2 --trials-per-level 3
```

## Interpretation
A slope trial is not a step. A brief leading-foot contact is not a full
climb. A policy's foot-specific recovery is separate from its command
tracking. Existing v5052 PPO checkpoints **must not** be resumed on
this changed-source branch: export actors from that training source,
then examine their immutable exported `.emupolicy` bundles here.

The goal is a fair, source-grounded physics screen. It is not yet
integrated into the four-panel periodic training evaluator: that change
requires on-GPU throughput, physical agreement, and memory benchmarks.
