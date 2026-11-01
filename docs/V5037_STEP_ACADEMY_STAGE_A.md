# v5037 — Step Academy Stage A (launchable single-lip screening)

**Research only, not a release or a multi-step course.** Eight independent
~10M-world-step continuations (280,035,328 -> ~290M) from **v5033 arm 00**
full training state. All eight keep the identical 360-input terrain actor,
six STS3215-style actuators, physical step collider, PPO/normalizer, and
broader fixed evaluation. The unmodified 280M source/parent and main are
never edited.

### What is actually built

- A fixed 85% STEP-focused birth distribution with the other five roles
  sharing 15%; no hard-coded changes to other roles or the actor input.
- Forced physical single-lip exposure on focused STEP episodes.
- **Small symmetric edge yaw ±3 degrees**, or a 0-degree ablation. The
  historical source uses approximately ±20 degrees, left unchanged when
  Academy is disabled. Explicit examiner step-edge angles still win.
- Opt-in shorter 300-frame episodes to sample more distinct single-lip
  encounters, a 2cm nursery cap, approach-speed variations, pure-step
  specialisation, and a pre-existing verified-support reward reallocation.
- Eight treatment configs and an immutable manifest. Preflight validates
  selected full-state forks/own-run resumes, parent bytes and Git source
  before writing or training. Each run stays in a separate directory.
- Frozen physical exams remain the same two-slab benchmark.

**Not yet implemented:** 4–6 consecutive step lips, per-lip event reset,
sequential foothold/weight-transfer milestone rewards, and causal sensor
ablation. Those belong to Stage B after fixed-topology collision/sensor
validation. Do not call Stage A's rapid repeats a continuous staircase.

### Arms and hypotheses

| Arm | Purpose | Comparison |
|---|---|---|
| 00 | source control, normal single lip | original |
| 01 | dense step focus, 85%, every STEP birth | 00 |
| 02 | 300-frame repeated lip episodes | 01 |
| 03 | 2cm nursery cap | 02 |
| 04 | 100% STEP focus, no skill retention | 02 |
| 05 | reallocate 3 ascent points to verified upper support | 02 |
| 06 | approach speed varied ±0.11m/s | 02 |
| 07 | remove ±3° obliqueness | 02 |

### WSL launch on dedicated worktree

~~~bash
cd ~/emu-plateau
git fetch origin '+refs/heads/feature/v5037-step-academy-stage-a:refs/remotes/origin/feature/v5037-step-academy-stage-a'
git worktree add -b feature/v5037-step-academy-stage-a ~/emu-step-academy origin/feature/v5037-step-academy-stage-a
cd ~/emu-step-academy
source ~/emu-plateau/.venv/bin/activate
python scripts/run_step_academy.py --matrix
python scripts/run_step_academy.py --verify-only --all
python scripts/run_step_academy.py --run --all
~~~

Default parent:
~/robot_lab/emu_runs/v5033_plateau_260_to_280/arm_00_control_seed761/checkpoints/final.pkl.
Default output: ~/robot_lab/emu_runs/v5037_step_academy_stage_a.

Exact parent step 280,035,328 must record source SHA
52736f4d75b053f2bab57cfaa2b5f4061b59f426. If not,
preflight refuses rather than silently using the old 260M checkpoint.
Absolute target default is 290,000,000. Use --arm 1 --run to start only
the density experiment; --all --run skips validated completed arms.
Keep this foreground; no tmux.

### Success criteria

Measure physically supported uphill crossings per actual encounter at
fixed heights 2, 2.5, 3 and 3.5cm, true upper contacts, weight transfer,
retained flat command tracking, falls, and efficiency. Do not rank by the
curriculum frontier alone. Wider edge angles are reserved for held-out
evaluation; ±3° training is not a claim of diagonal-step mastery.
