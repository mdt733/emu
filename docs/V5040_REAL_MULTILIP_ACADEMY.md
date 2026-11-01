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
  (Academy's OWN height frontier starts at 3mm and retreats as low as 1.5mm
  on failed easy supported lips), and symmetric ±3-degree edge yaw, preserving
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
  only after five consecutive frames of verified far-side support. A contact
  against a vertical side wall does NOT count as an upper foothold: true
  support checks contact normal AND the sphere's foot-bottom proximity
  to the upper collision face. Separate uphill/downhill crossings and
  complete course events appear in metrics.csv and terminal logs.
- Four-lip course difficulty uses a checkpointed, distinct near/easy
  approached-lip evidence window. At least 12 genuine attempted lips
  PER signed direction, >=55% supported crossing success near the
  independent Academy frontier, and healthy PPO are needed for +0.5mm.
  Repeated failures even at easy heights retreat -0.5mm, floor 1.5mm.
  Up-only progress cannot graduate both directions. Old single-lip STEP
  evidence cannot counterfeit Academy difficulty success.
- Stage A's global 300-frame shortened episodes are explicitly forbidden
  in the integrated multi-lip mode. The ordinary generalists keep their
  proven 700-frame/14s timeout, whereas Academy worlds receive a
  configurable 1200-frame/24s budget and end immediately on the fourth
  real supported crossing. The longer time never amplifies the old
  early-fall penalty. Coherent-careful training gets at most 120 frames
  of crouch/velocity allowance PER genuine uphill lip, not per
  entire course or for downhill/flat worlds.
- Full normal PPO/curriculum pipeline. The original foundation, commands,
  slopes, turning, resilience, transition, control and fast gait all retain
  representation. Academy role share ramps on world-step age, not on a
  separately optimized specialist:
  - 0–60M: Academy dormant, original 20M gait foundation untouched
  - 60–80M: easy nursery exposure rising to 10% of births
  - 80–100M: smooth ramp from 10% to 50% Academy births
  - 100M+: configurable maximum 50% of ALL new births assigned to Academy;
    the remainder retains ordinary single-lip STEP, command, slope,
    resilience, control, transition and fast rehearsal. Fast rehearsal's
    explicitly reserved fraction is accounted for BEFORE sampling STEP
    worlds, and >=15% of births is reserved for other nonfast roles.
    This is a target of BIRTHS: longer Academy episodes make the observed
    proportion of PPO frames larger. Both the birth target and ACTUAL
    frame occupancy are reported; tune the birth quota by measured
    retention, not by guessing '1024 of 2048' from the label.
  - The entire source can train fresh from 0M with --academy-course-lips 4;
    there is no dependency on the completed v5033 checkpoint.

## Actual proof and gates

- The academy-cpu GitHub job instantiates real MJX worlds, verifies all
  four physically abutting platforms and alternating signed heights,
  a genuine robot-foot upper contact, anticipatory visibility of all
  FOUR lips by the ACTUAL forward 4x4 ToF sensor before root crossing,
  correct five-frame support events and no double payment, and a finite
  shared PPO update. This is physical simulation acceptance, NOT learning.
- The separate academy-exam-cpu job runs the original immutable 260M
  champion through the independent four-lip course and writes its honest
  signed support evidence. The first 4-world, 200-frame/1cm-cap baseline
  measured 1/8 supported uphill and 2/8 downhill opportunities, with
  ZERO full four-lip courses and zero falls. Larger fixed-seed exams
  are needed before interpreting those counts as performance.
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
PYTHONPATH="$PWD" python scripts/smoke_step_course.py --device cpu --worlds 4 --frames 3 --ppo --events --sensor-foresight
PYTHONPATH="$PWD" python scripts/smoke_step_course.py --device cuda --worlds 128 --frames 3 --ppo
PYTHONPATH="$PWD" python scripts/smoke_step_course.py --device cuda --worlds 2048 --frames 3 --ppo --full-update
~~~

The last command can OOM on 8GB: it is the explicit *gate*, not
permission to start training. Academy 2048-world CUDA performance,
long-course learnability, and actual hardware motor capability have
NOT been established yet. If it fails, reduce contact pair topology
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
