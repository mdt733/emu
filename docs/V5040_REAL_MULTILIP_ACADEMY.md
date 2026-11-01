# v5040 — REAL four-lip Step Academy MJX world (experimental)

**This implements physical consecutive steps, not the v5037 single-lip screen.**
Research only, draft branch feature/v5040-multi-lip-academy-world.
Original main, v5033/v5037 source commits, policy weights, saved checkpoints
and historic physical exams are unchanged.

## What has been implemented

- The two original long slabs plus three REAL intermediate 0.35m-wide
  MuJoCo collision-box platforms; four contiguous sharp lips at 0.25, 0.60,
  0.95 and 1.30m of course travel, with genuine signed up/down profiles.
- Four balanced course patterns: up-up-down-down, up-down-up-down,
  down-up-down-up, down-down-up-up. Independent per-world lip heights,
  bounded cumulative elevation, a default 4cm difficulty ceiling
  (terrain frontier still starts small), and symmetric ±3-degree edge yaw, preserving
  physically continuous platform seams.
- Correct contact filtering: intermediate platforms touch the six-servo
  robot but do NOT generate platform-vs-platform self-contacts. Ordinary
  non-Academy worlds retain the historical two-slab physics. Dormant middle
  slabs are buried, without phantom 4x4 ToF returns.
- Existing authentic 360-input ToF array and optical-flow depth intersect
  the actual intermediate physical boxes. Existing proprioceptive/sensor
  actor observation order and width remain unchanged: no hidden lip-index
  or privileged terrain-height inputs.
- At each STEP-Academy birth, per-world lip index and anti-repeat flags
  begin at zero. Per-lip real foot contact, clearance, upper foothold and
  supported body weight transfer are rewarded once each. A lip is credited
  only after five consecutive frames of verified far-side support. Separate
  uphill/downhill crossing counts and complete four-lip course events are
  reported in metrics.csv and terminal logs.
- Full normal PPO/curriculum pipeline. The original foundation, commands,
  slopes, turning, resilience, transition, control and fast gait all retain
  representation. Academy role share ramps on world-step age, not on a
  separately optimized specialist:
  - 0–60M: Academy dormant, original 20M gait foundation untouched
  - 60–80M: easy nursery exposure rising to 10% of births
  - 80–100M: smooth ramp from 10% to 50% Academy births
  - 100M+: 50% Academy, 10% ordinary STEP and 40% other generalist roles
    (for the balanced original focus mix). Configurable experiment settings.
  - The entire source can train fresh from 0M with --academy-course-lips 4;
    there is no dependency on the completed v5033 checkpoint.

## Actual proof and gates

- The dedicated academy-cpu GitHub job instantiates four real MJX worlds,
  checks the 4 physically abutting contact-box seams, verifies mixed
  up/down heights and source sensor width, checks a real ray intersection
  to each type of physical platform, advances physics, and optionally
  evaluates one PPO gradient. See live Actions run and smoke script.
- Never claim actual *learned* climbing, 2048-env viability or real-hardware
  motor sufficiency based on a successful CPU smoke.
- Before a long training run, use a dedicated new worktree and the user's
  local CUDA device to run the production-shape 2048×32 PPO/memory gate.
- The independent four-lip frozen actor examiner is in
  scripts/exam_step_course.py and reports physical uphill/downhill successes
  and complete courses. It does not update PPO or curriculum.
- Run both the standard historic one-lip physical capability exam and this
  four-lip exam when comparing a trained candidate against the 280M control.

## Safe WSL physical acceptance (no training outputs)

The frozen old training worktree must not be switched or modified.

~~~bash
cd ~/emu-plateau
git fetch origin '+refs/heads/feature/v5040-multi-lip-academy-world:refs/remotes/origin/feature/v5040-multi-lip-academy-world'
git worktree add -b feature/v5040-multi-lip-academy-world ~/emu-multilip origin/feature/v5040-multi-lip-academy-world
cd ~/emu-multilip
source ~/emu-plateau/.venv/bin/activate
git status --short
PYTHONPATH="$PWD" python scripts/smoke_step_course.py --device cpu --worlds 4 --frames 3 --ppo
PYTHONPATH="$PWD" python scripts/smoke_step_course.py --device cuda --worlds 128 --frames 3 --ppo
PYTHONPATH="$PWD" python scripts/smoke_step_course.py --device cuda --worlds 2048 --frames 3 --ppo --full-update
~~~

The last command can OOM on 8GB: it is the explicit *gate*, not
permission to start training. If it fails, reduce contact pair topology
or adjust the fixed geometry implementation; do not silently reduce the
production batch or substitute fake contacts.

## Later two-policy comparison

Build a small matched controlled continuation from the **same frozen
training ancestor**, one with academy_course_lips=0 and one with
academy_course_lips=4 and early Academy role fraction 0.5. Do not merge
their checkpoint weights or pretend curricula with different physical
episode mixes have directly comparable training reward. Use the same
unmodified read-only one-lip exam, plus the new real four-lip examiner
with independent held-out seeds. Full report must include all cases and
the distinction between a training frontier and supported physical step
success.

This v5040 PR is NOT automatically a new flagship policy. It is the
physical course and curricular machinery to learn one.
