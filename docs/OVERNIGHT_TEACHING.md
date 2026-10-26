# EMU overnight teaching tournament

**Purpose:** a substantial, reproducible, source-pinned GPU experiment
testing both early foundation recipes and later specialist allocations.
All policies are trained from scratch with ONE actor, ONE PPO optimiser
and the existing physical MJX terrain, rewards and sensor contract.

This campaign does not modify `main`, the approved Pinnacle, the showcase
or any prior checkpoint. It does not secretly promote a winner.

## What runs

Preflight: actual 12-world CPU contact/episode smoke followed by a
**2048-world full CUDA PPO update**; these run in independent processes.

| Run name | Early recipe | Teaching allocation | Total world steps |
|---|---|---|---:|
| growth_balanced_s733 | curious, 10M foundation | 20/20/20/20/10/10 after ramp | 80M |
| growth_staged_s733 | curious, 10M foundation | Recovery/grace → slope → step | 80M |
| growth_terrain-early_s733 | curious, 10M foundation | 35% slope / 30% step focus after handover | 80M |
| screen_main_s733 | Main-style adaptive | balanced; ends before growth | 10M |
| screen_command-heavy_s733 | 90/5/5 gait/transition/recovery | balanced; ends before growth | 10M |

**Total: 260M environment steps**, sequentially, with independent frozen
physical exam panels and checkpoint/diagnostic logs. Each long run is
independent cold-start and uses the same seed 733. The first screen is a
separate experiment, not silently transferred into another run.

The long experiments run FIRST, so an interrupted night still produces useful development data. Their first 10M already represents the curious foundation, so a separate curious screen would duplicate it. Use `--no-screens` to run only the 240M long comparisons.

The 10M foundation is deliberately a **test**, not proof of mature walking.
All longer policies continue beyond it. The six-focus strengths remain
physically evidence-controlled, with easy/near/hard practice. Lesson share
is interpolated smoothly and affects only NEW episode births.

## Complete safe WSL setup + PREVIEW (no GPU training)

The owner's old `~/emu` may be a plain folder without `.git`.
Prefer the clean checkout `~/emu-capability-growth` without deleting it.

```bash
(
set -euo pipefail
BRANCH="feature/v5023-capability-growth"
REPO="$HOME/emu-capability-growth"

if ! git -C "$REPO" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  if [ -e "$REPO" ]; then
    echo "STOP: $REPO exists without valid Git. Leave it untouched."
    exit 1
  fi
  git clone --single-branch --branch "$BRANCH" \
    https://github.com/mdt733/emu.git "$REPO"
fi

cd "$REPO"
test -z "$(git status --porcelain)" || {
  echo "STOP: source has uncommitted changes"; exit 1;
}
git switch "$BRANCH"
git pull --ff-only
test -z "$(git status --porcelain)"

if [ ! -f .venv/bin/activate ]; then
  bash scripts/setup_wsl.sh --cuda13 --install-node
fi
source .venv/bin/activate
python -m pip install -e .

git rev-parse HEAD
nvidia-smi

python scripts/overnight_teaching.py \
  --root "$HOME/robot_lab/emu_runs/overnight_teaching_v5023"
)
```

Preview lists the five commands but creates no campaign or GPU policy.
**Make sure the exact source SHA is green on GitHub Actions** before the
actual campaign. The complete campaign command follows.

## RUN ALL FIVE experiments overnight (starts training!)

```bash
(
set -euo pipefail
cd "$HOME/emu-capability-growth"
test "$(git branch --show-current)" = "feature/v5023-capability-growth"
test -z "$(git status --porcelain)"
source .venv/bin/activate
nvidia-smi

python scripts/overnight_teaching.py \
  --root "$HOME/robot_lab/emu_runs/overnight_teaching_v5023" \
  --run
)
```

Keep the WSL terminal/process running. Each candidate has its own directory,
validation checkpoints, regular metrics/frozen exams, terminal log and
source-pinned training config. GPU sessions are sequential, not simultaneous.
This runner does **not** estimate how long your particular computer will
take and does not silently overwrite an existing experiment.

## What happens on interruption / errors

If one training arm fails, the runner **records that failure, continues the other arms, and exits nonzero at the end**. CPU or CUDA preflight failure aborts the campaign.
Successful partial results are still summarized in
`overnight_summary.md`, `overnight_summary.json` and
`overnight_results.zip`. If interrupted partway through training,
re-run the **same** command using the same Git SHA, target and
campaign root; it uses each run's own `--resume auto` checkpoint.
Already completed runs are detected and validated by `emu train`.
If run metadata is corrupt or source/config differs, the campaign stops;
do not manually delete/replace saved checkpoints to force a resume.

A successful CPU/full CUDA preflight writes `preflight_pass.json`.
`--skip-smoke` on resumption is allowed only when that file certifies the
**same source SHA**. Never skip an untested 2048-world CUDA preflight.

To shorten/restructure the experiment deliberately, choose a NEW campaign
root. The campaign manifest refuses changes to an existing seed or target.

## What to share next morning

Under `~/robot_lab/emu_runs/overnight_teaching_v5023`:

- `overnight_results.zip` — shareable small bundle with config, rollout
  metrics, fixed exams and per-run reports; **no giant policy files**.
- `overnight_summary.md` and `.csv` — direct five-arm comparison.
- `frozen_exams_by_age.csv` — held-out comparable panels every ~10M (actual PPO boundaries), not mixed with curriculum-dependent training scores.
- `campaign_status.json` — records failed versus completed arms; one independent failure does not cancel the remainder.
- `progress_by_age.csv` — available same-age rollout observations
  at 10M, 20M, 40M, 60M and 80M, without inventing missing milestones.
- `logs/*.log` — process logs (open on failures).
- Each candidate's `checkpoints/final.pkl`, `results_bundle.zip` and
  `fixed_eval.csv`.

At launch `pip_freeze.txt`, `nvidia_smi.txt` and `disk_space.txt` record the actual software/driver environment. After a completed/active training row the campaign estimates remaining hours from *measured* throughput. That is a training-only estimate; fixed exams, compilations and failures add overhead. Check free disk space, Windows sleep and automatic restart settings, and avoid GPU-heavy applications.

**Keep the process alive with `tmux`:** run `tmux new -s emu-overnight`, execute the entire training command inside, detach with Ctrl-B then D, and reattach using `tmux attach -t emu-overnight`. Do not Git-pull while any candidate is running.

The scoreboard is **descriptive**, not an automatic champion election:
compare frozen performance and physical near-target command/slope/step
attempts, not just mean PPO reward. At 10M an infant may fall often.
Repeat promising arms across seeds 734 and 735 before drawing strong
reproducibility conclusions.

## Important limits

No cloud-side assistant can run your RTX 4060 Ti locally from this chat.
GitHub CPU CI doesn't prove 2048-world CUDA memory stability or long-run
quality. The campaign must finish its preflight, and each candidate should
be evaluated before we consider promoting any policy. `main` stays frozen.
