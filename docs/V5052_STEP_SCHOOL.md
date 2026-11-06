# EMU v5052 — historical complete stair manoeuvre study

> **Archived experiment, not a runnable current-source campaign.** The v5052
> launcher and its v5051 helper have been retired from active `scripts/`,
> preserving the original files at their [immutable research snapshot](https://github.com/mdt733/emu/blob/6a3c9cc971936d12f95822a6561cc244abf896e7/scripts/v5052_step_school.py).
> The old `python -m scripts.v5052_step_school` commands below require that
> historical checkout. Today’s source uses `emu train`; v5054’s
> short-completion research is documented in [V5054](V5054_STAIR_SUCCESS_RESET.md).

## Question

**Why did fifteen v5051 PPO treatments yield 776 authentic upper/downhill
top-contact events, yet zero completed post-contact stair missions?**

v5051's report alone cannot distinguish failure to recover the trailing
foot, failure to sustain bilateral top support, or lack of controlled
continuation. It also does not prove that the sampled procedural terrain
ever supplied a particular exact-height wall.

The working hypothesis is that teaching a *velocity-tracking general walker*
across occasional signed height changes is not sufficiently focused to learn
the support transfer and continuation sequence. This study now tests a
**fixed goal 60cm beyond the first physical step**, separate control of
stair-speed tracking, genuine foothold/transfer guidance, gait-phase freedom,
and increased Academy exposure. Success is a one-time physical goal event,
not merely reaching a new tile or collecting velocity reward.

The papers that motivate these hypotheses are:
- [Siekmann et al., 2021, Cassie blind stair traversal](https://roboticsproceedings.org/rss17/p061.html):
  stair-like terrain variation and locomotion adaptation without explicitly
  changing the reward.
- [Chamorro et al., ICRA 2024, blind stair climbing](https://arxiv.org/abs/2402.06143):
  a **position-based task**, stair-mode observation and privileged asymmetric
  critic improved stair climbing relative to pure commanded-velocity control.
- [van Marum et al., IROS 2024, robust standing/walking](https://arxiv.org/abs/2404.19173):
  minimally constrained reward design and explicit physical comparison.

**New actual destination task (v2):** at every Academy episode birth,
the teacher calculates a fixed target **0.60m after the next +X 75cm
physical tile seam**, keeping the birth Y coordinate. With the shared
near-lip spawn 0.22m before the seam, the robot has to cover roughly
0.82m to arrive. The 360-input actor sees no privileged world coordinates
or height fields: it still receives ordinary forward commands and real
4×4 ToF/flow observations.

- Every frame earns a **signed change in actual Euclidean distance**
  to that same target, with bounded amplitude, upright/contact/low-slip
  conditions for **positive** credit. Backtracking is penalised even
  when unsupported, preventing bounce/retreat payment farming.
- **Arrival success** is NOT just reaching the coordinate. The trainee
  must (1) establish genuine top support with each foot on the
  **first target tile** for five uninterrupted frames per foot
  (sequential landings are allowed), (2) arrive within 7cm longitudinally and
  12cm laterally, (3) remain heading-aligned within 0.35rad, and
  (4) maintain upright supported low-slip balance for five consecutive
  frames. It gets one terminal bonus and a clean goal-completion episode,
  never a second bonus from the same birth.
- Academy goals at normal configured `--science-step-goal-weight 0`
  are **disabled**, so the unchanged control and velocity-relief-only
  treatment cannot secretly obtain new success rewards.
- The combined goal+relief arms use **100% Academy-only forward velocity
  tracking relief**: they can stop, lean, transfer and climb without
  having to maintain a fixed metres/second target. Yaw, lateral and
  pitch tracking, safety and large overspeed constraints remain.
  The 02 relief-only arm deliberately retains its 80% relief ablation.
- PPO differentiates **successful absorbing termination**, genuine
  physical falls and ordinary time-limit truncation. Success has
  zero terminal-value bootstrap; timeout still bootstraps. Source
  provenance and checkpoint schema prevent mixing prior versions.
- Only genuine sequential placement of BOTH feet on the FIRST goal tile
  counts toward completion. A later unrelated lip cannot unlock the target.
  The old strictly simultaneous bilateral-support check remains a
  separate diagnostic, not a compulsory component of a moving gait.
  Training geometry is still the actual dynamic eight-prism world,
  not a guaranteed exact-height riser.

**Scientific limitation:** the goal is fixed relative to the first physical
seam in the world-frame +X direction, but does not introduce operator
waypoint commands or an asymmetric critic. This is an explicit
destination-based **Academy lesson**, not arbitrary field navigation.
Some procedurally encountered lips are downhill; the independent
exam reports signed rise and separates uphill from downhill completion.
No matter what the terminal reward says, the independent examiner
must reproduce destination arrival from raw first-life prism contacts
and motion rather than trusting the teacher's completion flag.

## Registered 320M PPO study

One *NEW SOURCE-PINNED* 0→40M parent:
- 0–20M: ordinary gentle cold foundation; learn standing and walking.
- 20–40M: early low-height Academy introduction, same 8 real prisms and all
  six specialist/retention regimes; 20% Academy intro allocation.
  Academy-only births start **22cm before the next real 75cm prism seam**
  (within useful 4×4 ToF range), instead of wasting a novice episode
  walking 75cm before reaching its first challenge. The true tile surface,
  signed riser and collision are generated unmodified. Ordinary flat,
  resilience and other non-Academy births keep their original positions.
  This near-lip preparation is SHARED by all 7 arms, not a treatment factor.
- PPO optimizer, actor, critic, normalizer, curriculum and capability field
  saved in the identical `parent/checkpoints/final.pkl`.

Seven **40M PPO continuations from that exact same 40M parent**, each ending
at 80M absolute age (40 + 7×40 = **320M newly trained** world steps):

| Arm | Independent question |
|---|---|
| `00_control` | Does earlier consistent stair exposure alone improve outcomes? |
| `01_goal` | Can a fixed 60cm destination and signed distance reduction improve first-step recovery? |
| `02_relief` | Can temporarily relaxing Academy-only vx matching allow leg transfer? |
| `03_goal_relief` | Does fixed destination + removal of stair vx-pressure work better than either alone? |
| `04_contact_teacher` | Does true-prism reward-side foothold/transfer guidance help beyond the combination? |
| `05_phase_freedom` | Does slowing low-command phase and removing phase-sync *reward* improve the combination? The phase input still exists |
| `06_dense_school` | Does denser Academy allocation improve the combination? No new physics |

All arms keep the same 360-input actor, sensor contract, solver 2/4,
identical 2048 environments, seed 735, shared parent and PPO update count.
The combination treatments can reveal incremental benefit against
`03_goal_relief`, but are not a full factorial causality study.
The extra reward is gated by actual Academy assignment, support, low slip,
nonfalling posture, no lateral or yaw runaway, and measured *positive*
body-frame progress. Zero motion yields **zero** extra goal payment.
All new values default to zero; existing PPO policies maintain old economics
when the new knobs are unused.

## Independent physical stage autopsy

Each frozen final actor is examined, **without PPO updates**, on one held-out
terrain seed initially. Eight compact exam tasks per arm, 8 worlds per task
by default:
- Stand at zero command on flat.
- Fast walking at +0.70m/s on flat.
- Requested step sampling frontiers **0.5cm, 1.5cm and 3cm**, each at
  **+0.12m/s and +0.30m/s** commanded approach.

These are **procedural physical sampling frontiers, NOT exact risers**.
Two *different* physical outcomes are always distinguished:
(1) historical 50cm of additional travel after two-foot recovery, and
(2) NEW fixed-goal arrival 60cm beyond the initial seam, including five
stable frames and separate verified first-lip landing of EACH foot. The latter is the direct
v5052 stair objective and is independently re-scored without teacher
reward flags; the former is useful as a tougher post-contact retention
test. Uphill and downhill v5052 arrivals are counted separately.
For every verified signed prism top contact, the report separately counts:
1. Actual uphill contact height >= 1cm, 2cm and 3cm;
2. Confirmed five-frame simultaneous top support by both feet;
3. Post-support travel >= 15cm, 30cm and 50cm;
4. Completed 50cm heading-controlled missions, failures by named verdict;
5. Flat fast survival and achieved speed, zero-command footfall rate.

Do not report a staircase *success rate* divided by nominally requested
height: the generator did not impose a specific wall on every robot.
Until instrumented exact-height fixtures exist, these are
contact-conditional **mechanism diagnostics**, not step-success probability.

Forensic can also read original v5051 per-arm `prism_exam.json` files
without editing the old run. The old study's `SCIENCE_RESULTS.json` summary
is **not enough** to recover stages: supply the original `exams/` tree.

## Decision gates

- No >=1cm supported uphill contacts? First diagnose approach, swing
  clearance, ToF range and course geometry before adding more rewards.
- Upper contacts but no **independent trailing-foot upper support**?
  Teach trailing-foot recovery; consider foothold guidance and phase freedom.
  Simultaneous double stance is recorded separately, not mandatory.
- Bilateral support but <15cm continuing motion? Focus on a genuine
  endpoint-based task, controlled post-support progress and survival.
- 15cm/30cm progress but not 50cm? Investigate unrealistic grade horizon:
  at 0.12m/s the final 50cm requires >4 seconds *after* first recovery.
- Meaningful complete missions at 0.30m/s but not 0.12m/s? Dynamic gait
  timing/momentum, not just perceived height, is implicated.
- Greater contact counts with poor fast survival/standing? Do not promote.
- Flat speed and contact improve but no complete stage? Run exact-riser
  goal-conditioned/world-position task and privileged critic experiment next.

One PPO training seed and one held-out physical seed are **not**
statistical evidence of a deployable winner. Increase evaluator seeds for
promising arms and control, then independently repeat PPO seeds before any
promotion. Do not merge this exploratory branch into `main` simply because
a training reward improved.

## WSL launch

**This is a fresh source lineage**, not a resume of v5051. The experiment
refuses parent checkpoints from any different Git source commit. The old
results remain untouched.

```bash
cd ~/emu
git fetch origin
git switch research/v5052-step-learning 2>/dev/null || \
  git switch --track origin/research/v5052-step-learning
git pull --ff-only origin research/v5052-step-learning
source ~/emu-graduation/.venv/bin/activate
export PYTHONPATH="$PWD"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
git status --short      # MUST be blank
git rev-parse HEAD

# No training or file writes: inspect the exact proposed operations
python -m scripts.v5052_step_school --stage plan

# Small contract tests before the GPU training campaign
python -m pytest -q tests/test_core_step_goal.py

# Stage 1: 40M cold/common parent (watch balance by ~5M)
python -m scripts.v5052_step_school --stage parent

# Stage 2: 7×40M real PPO forks, all use the immutable parent
python -m scripts.v5052_step_school --stage arms

# Stage 3: independent physical exam of all 7 complete actors
python -m scripts.v5052_step_school --stage exams

# Stage 4: report the *failure stage*, not a composite reward
python -m scripts.v5052_step_school --stage report
```

Or single foreground run (320M PPO updates plus exams/report):

```bash
python -m scripts.v5052_step_school --stage all
```

All results are under
`~/emu-data/runs/v5052-fixed-destination-v2-s735/`. Stages resume or skip verified
completed checkpoints; a partial arm always resumes its **own** checkpoint
rather than restarting from the parent. Do not change Git SHA once training
starts.

For fast pilot research, use one matched pair first, without changing the
shared source or parent:

```bash
python -m scripts.v5052_step_school --stage arms \
  --only 00_control,03_goal_relief
python -m scripts.v5052_step_school --stage exams \
  --only 00_control,03_goal_relief
```

Then expand to the other arms if these show meaningful physical progress.

For a second held-out **environment** seed on finalists and unchanged
control (does not retrain PPO):

```bash
python -m scripts.v5052_step_school --stage exams \
  --only 00_control,03_goal_relief,04_contact_teacher \
  --exam-seeds 1701,1702
```

Forensics against old v5051 raw exams:

```bash
python -m scripts.v5052_step_school --stage forensic \
  --source-exams ~/emu-data/runs/v5051-science-s735
```

The terminal prints the old per-event verdict breakdown without modifying
the original study.

## Not built in this first research stage

True exact-height guaranteed risers, per-world heterogeneous physical
GPU exam assignment, explicit actor-visible target-position commands,
asymmetric-critic teacher/student, a completely clock-free 360-input
replacement actor, and matching browser eight-prism visualisation.
The next branch should implement those only if v5052's *measured*
failure-stage evidence identifies a need. Git remains the archive; avoid
permanently adding seven different training frameworks.
