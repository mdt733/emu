# v5054 — end the stair nursery on a real, successful climb

Research-only stacked on v5053's draft capability-matrix branch.
No policy architecture, sensor contract, or production `main` changes.
Opt-in source-pinned flag: `--science-step-finish-on-transfer` (OFF by default).

## Why

The paired 2,048-world v5053 exact-height exams of frozen `01_goal`
showed 1,997/2,000 flat worlds contacting both feet, 1,994 passing the
downstream line, but only 13 passing the precise five-frame park-and-hold.
20mm uphill steps still got both feet on top in 89/100 trials and
body downstream in 86/100. More precise parking is a separate skill.
Do not use the old strict destination score as the nursery teacher.

## Success → end episode → respawn next real step

Only active streamed Academy STEP specialist births are eligible.
The **first actual +X 75cm seam** height is computed from source
prism vertices and mocap elevations **on episode birth**. Zero-height
seams are not climbing trials; downhill and uphill are distinct signed
cohorts. The original actor never receives that height or a goal coordinate.

Each of the two feet must independently exhibit **5 consecutive frames**
of upward-facing real target prism-top contact, after an authentic
first-lip crossing. Once both-foot proof is latched:

- root X at least **10cm beyond** the first lip, still on that tile;
- lateral position within **25cm** of original birth lane;
- heading within **0.55rad** of course +X;
- gravity upright >=0.80, support >=0.5, no disallowed contact;
- above conditions sustained for **3 frames** (no stop/low-slip demand).

Success emits `academy_finish_new=1`, pays one bounded **+8** completion
reward and sets `done` immediately. The existing per-world reset path
then samples the next task, step profile, pose and RNG. The PPO terminal
mask treats this as **absorbing** (V=0), like an actual successful goal.
Falls remain failures; timeouts retain value bootstrap.
No duplicate payment is possible within an episode. Other specialists,
flat retention, and the original v5052 precise 60cm goal remain unchanged.

This is a **short climb**, not proof of reliable multi-step locomotion,
long-range navigation or exact stable destination arrival. The independent
v5053 evaluator remains authoritative for those separate scores.

## Honest per-encounter frontier evidence

In the opt-in short Academy, EACH completed trial is one signed real
first-lip verdict, recorded at success, fall or timeout. It is classified
against that direction's *actual current* frontier: near = 78–105%,
easy = 36–68%. The near success numerator is `academy_finish_new`;
the denominator includes failed episodes with a nonzero real birth seam.
Separate up/down frontiers still require >=12 near attempts and >=55%
short-climb success for a 0.5mm promotion. Easy failures retreat at <25%.
PPO health can still block expansion.

Legacy Academy's per-lip event curriculum now preserves the signed
height of an unsuccessful attempt rather than classifying its terminal
frame's transient zero. The ordinary STEP episode similarly latches
its first authentic attempted signed rise so a supported crossing at
frame 200 is not erased when the episode ends at frame 700. The
ordinary curriculum still requires survival to timeout; it is NOT
switched to instant success. Encounters where the robot never gets
close enough to physically register an approach are not scored by that
ordinary legacy mechanism; the new nursery instead has a real birth
lip recorded upfront and counts falls/timeouts as failed attempts.

## Acceptance and experiment hygiene

- Extends existing core tests; never exceeds 24 maintained test cases.
- Source changes require GitHub CI MJX physics check and a short CUDA PPO
  lifecycle on the operator's 8GB RTX 4060 Ti before any long experiment.
- `main` and original checkpoints are untouched.
- This new option modifies terminal/reward/teacher meaning and MUST use
  a fresh run directory, explicitly source-pinned. Historical config
  loading only defaults the newly introduced (formerly nonexistent)
  flag to False; all other missing keys continue failing.
- Retain one unchanged v5052 baseline arm and compare frozen-policy
  traversal curves after the exact same continuation budget.
- Monitor `academy_short_climb_successes`,
  `academy_near_attempts_up/down`, `academy_near_successes_up/down`,
  `academy_height_up_m/down_m`, `academy_event`, falls and velocity.
  Success rate should not increase merely by redefining the evaluator.

## Worktree setup (does not alter main)

```bash
cd ~/emu
git fetch origin +refs/heads/research/v5054-stair-success-reset:refs/remotes/origin/research/v5054-stair-success-reset
git worktree add --detach ~/emu-v5054-stair refs/remotes/origin/research/v5054-stair-success-reset
cd ~/emu-v5054-stair
source ~/emu/.venv/bin/activate
export PYTHONPATH="$PWD"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python -m pytest -q tests
```

To run a source-pinned PPO fork from the existing v5052 `01_goal`
final 80M checkpoint, use a **new** output folder and explicitly
provide the complete frozen parent checkpoint SHA256 and its original
Git source (as required by the existing research fork guard). Use the
v5052 campaign's COMMON training flags, plus:
`--science-step-goal-weight .30 --science-step-finish-on-transfer`,
and pass `--resume /path/to/01_goal/checkpoints/final.pkl`,
`--research-parent-source bfb235e2d5d4158180744125d20a479041937cb5`
and `--research-parent-sha256 $(sha256sum /path/to/final.pkl | cut -d' ' -f1)`.
Set `--steps` above the actual parent step (80,019,456), e.g. a small
acceptance continuation first. Supply a complete, matching set of
v5052 task/sensor/solver settings. Prefer `--plan` and
`--check-resume-only` before spending GPU time.
