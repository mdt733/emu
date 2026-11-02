# v5040: improve STEP training, not the training architecture

Research branch: `feature/v5040-multi-lip-academy-world`.
`main`, published actors and prior experiment checkpoints are untouched.

## The actual change

The existing STEP specialist can now receive a short course of **four real
physical lips** instead of a single step. Three intermediate collision boxes
join the existing first/final slabs. Each episode chooses randomly among all
six two-up/two-down orders; individual heights (up to the Academy frontier)
and course yaw (±3°) also vary. No sequence or terrain truth is exposed to
the actor: its existing 4×4 ToF, optical flow, IMU and proprioception are
the only inputs. The observation size remains 360, with one actor and PPO.

Each lip pays once for genuine physical progress: uphill foot clearance,
top-supported foothold, weight transfer and five-frame supported crossing.
Downward lips pay for their supported crossing, not imaginary foot lift.
A lip that was never approached cannot be counted as a failed attempt.
Academy difficulty has its own small checkpointed evidence window:
both upward and downward near-height success must be demonstrated to grow
by 0.5mm; repeated failures at easier heights retreat by 0.5mm.
This does not change the ordinary single-step difficulty governor.

The proven 20M gait foundation stays intact. Academy births start ramping
at 60M, reach 10% at 80M and up to 50% at 100M. Fast-gait rehearsal and
other specialties retain their allocation. These are birth fractions;
the longer 1200-frame (24s) Academy course may occupy more than half the
active frame batch. Other worlds retain 700 frames (14s). Completing four
verified lips ends the course immediately. No Stage A shortened-episode
settings are used.

## Keep it small

Old v5033 science and v5037/v5038 single-lip research scripts,
their reports/tests and duplicate design documents are left on their
original Git branches, rather than copied into this implementation.
No new test suite or Academy CI matrix. There is one optional quick MJX
smoke/memory probe and one useful read-only four-lip examiner.

Earlier CPU physics probing observed real support, ToF visibility and a
finite PPO step; this is **not** proof of learning. The simplified revision
has not yet been run on the 8GB GPU or trained. Leave `main` alone.

## On the GPU PC

From a separate checkout, with the existing compatible JAX/MJX venv:

```bash
git clone -b feature/v5040-multi-lip-academy-world --single-branch \
  https://github.com/mdt733/emu.git ~/emu-multilip
cd ~/emu-multilip
source ~/emu-plateau/.venv/bin/activate
git status --short
PYTHONPATH="$PWD" python scripts/smoke_step_course.py \
  --device cuda --worlds 128 --frames 3 --ppo
PYTHONPATH="$PWD" python scripts/smoke_step_course.py \
  --device cuda --worlds 2048 --frames 3 --ppo --full-update
```

The 2048×32 update may OOM the 4060 Ti 8GB. It writes no checkpoint.
Only launch real training after this gate; do not silently reduce the
production batch. Compare a trained candidate using both the ordinary
single-step exam and `scripts/exam_step_course.py`. Success is supported
crossings and preserved general walking, not reward numbers alone.
