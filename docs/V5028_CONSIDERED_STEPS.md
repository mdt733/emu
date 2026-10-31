> **Archived experiment (completed):** v5028 A/B source was frozen at
> `38713791023681dc9883f6c374d92e46261b3baf`. To reproduce its
> historical launcher, check out that exact commit in a separate worktree.
> This document is a record, **not a current launch instruction**.
> The v5029 study runs from its own pinned source; experimental reward
> designs follow in `docs/V5030_REWARD_EXPERIMENTS.md`.

# EMU v5028 — considered STEP-UP reinforcement experiment

**Status:** source-pinned experiment launched by owner on 29 October 2026;
full CPU/CUDA preflight outcome and PPO metrics are not yet supplied.
The live experiment uses `feature/v5028-considered-steps` at exact
`38713791023681dc9883f6c374d92e46261b3baf` and **must not be modified**.
This documentation is updated on the separate `feature/v5028-merge-polish`
branch, which is not a resume source.
Protected `main`, published Pinnacle and all historical v5027 checkpoints
are unchanged. Source changes require a **new cold 20M parent**; do not
transplant saved 5dec60c4/ab64cac training checkpoints.

## Research question

Does the old per-frame command-velocity reward discourage useful deliberate
foot placement on a tall step? Frozen v5027 A ascended **2.0cm 8/8**,
**2.5cm 6/8**, **3.0cm 0/8** at +0.15m/s command; 3+cm supported
crossings were only descents. This experiment tests *teaching*, not
deployment-time safe refusals.

## Exact change, only in B

- Both arms keep the same 360-input, six-servo, one PPO actor, real paired
  step contact, 4×4 delayed ToF, all focus teachers and unchanged gentle
  coherent forward commands (0.07..0.13 m/s in STEP lessons).
- `A_original_coherent` preserves v5027 original per-frame reward and
  five-frame moving far-side contact criterion.
- `B_considered_up` uses `--terrain-lesson coherent-careful`:
  for an *actively selected STEP specialist* approaching a **genuine
  upward rise** (>1mm physical height, real step slab, within 16cm
  before/12cm after the lip, within 22cm lateral), the reward multiplier
  for **forward UNDERSPEED only** becomes 10% of its ordinary loss.
  Maximum 120 qualifying 20ms frames (2.4s) **for the entire episode**,
  not renewable by rocking/retreating. All overspeed, lateral/yaw,
  posture, foot contact, smoothness, slip, joint and fall terms persist.
  Flat, step-down, slope, all other roles and post-crossing normal gait
  keep the original reward. Teacher-only true geometry gates the reward;
  it does NOT enter actor observations.
- B alone also removes the artificial per-frame `vx>0.02` condition
  from the **already real** five-frame far-side contact crossing,
  permitting supported brief settling. Actual far-side physical position
  + target MJX foot contact + healthy contact + survival remain mandatory.
  Timeouts without a crossing **cannot** promote the training frontier.
- This is a combined reward-plus-contact treatment, not proof of which
  subchange causes any gain. Do not claim tested safe refusal or precise
  ToF stopping distance. This source is a research hypothesis.

## Experiment and interpretation

Seed 735, 2048 CUDA envs, 20M common cold main-style foundation,
then two matched 20→80M forks, **140M aggregate world steps**. No
intermediate winner selection. Full 2048 CUDA PPO gradient acceptance,
two CPU variants, source/checkout guard and every-10M fixed exams
before/through training. The **original pinned launch** runs an independent **8-world/condition**
physical matrix: **2/2.5/3/3.5cm UP and DOWN** and four directed
**5/8/12/15-degree** slopes, 0.15m/s, 350 frames. The post-run polish branch updates a **future** launcher to 16 worlds; it does not alter the running training source or its examination count. It writes
`physical_summary.md`, independent per-world JSON and
`overnight_results.zip`.

Judge signed supported **UP** crossing and far-side heading/velocity
separately; success is not an improved training reward nor an
accidentally completed downward step. Require maintained flat forward
speed, lateral/yaw command, resilience75 and shaft energy performance,
against A as well as previous v5027 physical evidence. Fresh seed
and 32-world confirmation are needed before promotion.

## WSL foreground launch (no tmux)

```bash
set -euo pipefail
if [[ ! -d "$HOME/emu-v5028/.git" ]]; then
  git clone --branch feature/v5028-considered-steps --single-branch \
    https://github.com/mdt733/emu.git "$HOME/emu-v5028"
fi
cd "$HOME/emu-v5028"
git switch feature/v5028-considered-steps
git pull --ff-only
test -z "$(git status --porcelain)"
source "$HOME/emu-graduation/.venv/bin/activate"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -m py_compile emu/rewards.py emu/environment.py emu/evaluation.py \
  emu/cli.py scripts/smoke_unified_focus.py scripts/v5028_considered_steps.py
python -m pytest -q tests/test_considered_steps.py
python -u scripts/v5028_considered_steps.py --run
```

Once running, do not pull/rebase/commit into that checkout.
On Ctrl+C or unexpected interruption, restart the **exact unchanged
source** with `--run --skip-smoke` after the same-SHA CUDA preflight
stamp exists. The launcher uses its own source-pinned checkpoint
recovery, terminal console output and log files. No tmux.
Results: `~/robot_lab/emu_runs/v5028_considered_steps_s735/`.

**Known limitation:** this is a teacher-only reward experiment.
Sensor detection lead distance, safe stop/retreat and deployment
refusal are separate and remain unimplemented. Read
[TRAVERSABILITY_REFUSAL](TRAVERSABILITY_REFUSAL.md) before adding a
navigation safety controller.
