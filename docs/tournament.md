# Experiment lab — v5015 and historical tournaments

## v5015 short experiment lab

**Recommended now:** edit/inspect [`emu/ladder_recipe.py`](../emu/ladder_recipe.py), run `emu ladder --stage plan`, and start the campaign with `bash scripts/run_ladder.sh`. The former `lab5015` command remains an alias. This uses schema-7 checkpoints and the normal PPO training engine, with hard eligibility gates after matched evaluations.

| Milestone | Cumulative step per policy | Extra GPU steps |
|---|---:|---:|
| Shared foundation | 20M | 20M |
| Four first-round forks | 26M each | 4 × 6M = 24M |
| Two selected finalists | 34M each | 2 × 8M = 16M |
| One finalist continuation | 46M | 12M |
| **Total** | | **72M** |

All four forks inherit **the exact same validated 20M checkpoint**: actor, critic, optimiser, normaliser, curriculum, training-health and capability-field state. Step counts are absolute environment steps, rounded to the PPO update boundary. Forks run sequentially on one GPU; extra experiment overhead includes JAX compilation and held-out screens.

The candidate hypotheses are intentionally easy to understand:

| Arm | Exactly what it tests |
|---|---|
| `baseline` | v5015 rewards, 10% graph teacher, flat collision ground, 8° pitch cap |
| `shadow` | Same baseline except no active capability teaching |
| `slope` | Same baseline except gentle gravity-equivalent slopes, `T=0.12` |
| `pitch15` | Same baseline except 15° maximum requested chassis pitch |

The baseline and slope arms are **not** training on actual ramps or heightfields. Browser ramps remain a separate held-out physics challenge.

### Before running

Use the development Git worktree on one frozen commit. Update it **before** starting a run, not between forks. Run roots are outside Git, and an immutable `lab_context.json` prevents silent source-commit, seed and environment-count changes. Do not reset or erase old runs; do not edit a completed arm's training config.

```bash
cd ~/robot_lab/emu
git status --short   # should be empty
git pull --ff-only
source .venv/bin/activate
emu doctor --v5015-smoke
emu ladder --stage plan
```

The new `doctor --v5015-smoke` compiles an actual MJX reset/step and an 11-field held-out summary with the new posture contract. **Do not begin the 20M run if this fails.**

### Unattended tournament — one command

The 20M foundation is a **branch checkpoint**, not a mandatory pause. Let the GPU continue through the complete short experiment, including scores and promotions:

```bash
cd ~/robot_lab/emu
bash scripts/run_ladder.sh
```

The wrapper unsets any stale `PYTHONPATH`, verifies Python is importing **v5015 from this checkout**, and runs a real GPU/MJX/PPO optimizer-update smoke test **before training**. It then starts `emu ladder --stage auto`: foundation → four 26M forks → common 15° and gentle-slope held-out tests → two *eligible* 34M finalists → new tests → one *eligible* 46M finalist → final score.

The runner automatically chooses the top eligible policies, but **never overrides** hard-survival, pitch-tracking, height-shortfall and missing/nonfinite data gates. If fewer than two round-1 candidates or zero round-2 candidates qualify, it stops safely with all checkpoints intact. Selections are locked to those evaluated checkpoint fingerprints, and resuming only reuses screening results if the fingerprints and evaluation settings match.

To inspect the first matched comparison before the finalists continue:

```bash
bash scripts/run_ladder.sh --stop-after round1
```

Restart the default command to continue from validated checkpoints and matching scorecards, **without changing Git commit, seed or environment count**.

**Long-horizon caution:** the observed crouch started around **100–200M steps in one policy lineage**. This first 72M-step *aggregate* funnel only extends one lineage to 46M, so it cannot detect that late collapse. The strongest candidates need staged 100M/150M/200M follow-up measurements.

### Stop-and-review workflow

```bash
emu ladder --stage foundation  # 20M once
emu ladder --stage round1       # sequential 6M additions per arm
# Or run one at a time: emu ladder --stage round1 --arm baseline
emu ladder --stage score --round round1

# AFTER reviewing round1_scorecard.md, example names only:
emu ladder --stage round2 --select baseline slope
emu ladder --stage score --round round2

# AFTER reviewing round2_scorecard.md, example only:
emu ladder --stage round3 --select slope
emu ladder --stage score --round round3
```

**The names in the examples are illustrative, not recommendations.** Round2 requires two eligible arms; Round3 requires exactly one previously selected eligible finalist. Selection is recorded and cannot be silently rewritten after a run starts. Each new stage gets its own directory, so earlier checkpoints are preserved.

### How the competition is judged

A **common, fixed 15° v5015 evaluation world** is used for every contender, independent of the domains/pitch ceiling used during training. It runs the same deterministic stress panels (`resilience100`, `robust70`, `growth140`, `growthcombo140`) using a single batched GPU screening process. It measures recovery/growth survival, **an extra fixed gentle slope-equivalent stress panel (T=0.20, up to ~1.2°)**, chassis pitch tracking error, height shortfall, and motion error. Scores are transparent; any missing/non-finite metric, severe crouching, poor pitch tracking, or major hard-survival regression marks a candidate ineligible. An eligible score is **not proof** of learned 15° pitch or new terrain mastery.

A scoreboard and readable Markdown report are written under:

```text
~/robot_lab/emu_runs/v5015_lab1/
  foundation20/checkpoints/final.pkl
  round1/<arm>/checkpoints/final.pkl
  round1_scoreboard.json
  round1_scorecard.md
  round2/<arm>/checkpoints/final.pkl
  round2_scorecard.md
  round3/<arm>/checkpoints/final.pkl
```

Scores include a SHA-256 fingerprint of each evaluated checkpoint, and promotion verifies the same fingerprint. Never promote on a stale report. Screening is read-only with respect to PPO weights and never feeds teaching data into the capability field.

### What the GPU time cannot tell us yet

A single seed is useful for choosing **hypotheses worth more compute**, not establishing statistical superiority. Once the first funnel identifies promising changes, confirm a pair on a different independent seed or on additional held-out tasks. Watch actual-minus-commanded height in `metrics.csv`; if everything starts crouching again, investigate rather than letting a global reward average choose the winner.

The retired `forge5015` 400M single-lineage recipe is available from Git history only; it is not part of the current CLI. The user-reported 100–200M crouch and first failed v5015 lab run both require separate logged investigations.

---

## Archived experiment: v5013 capability-teacher tournament

The v5013 tournament compared graph-teacher doses, node/edge/recovery mechanisms and staged teacher schedules. Seed confirmation was mixed; it did **not** prove one universal winning teaching mechanism. The original source, ranking rules, prompts and runner remain inspectable in [Git history](https://github.com/mdt733/emu/blob/b0f31d865198f451414ec0d6e69e1793c69478a7/emu/tournament.py), and its actual evidence is preserved in [results/v5013_tournament](../results/v5013_tournament/README.md). Key lessons are retained in [history.md](history.md).

Historical `emu tournament` commands in older docs do **not** exist in the current executable. Use a separate historical Git checkout for exact reproduction; do not point it at current run roots.
