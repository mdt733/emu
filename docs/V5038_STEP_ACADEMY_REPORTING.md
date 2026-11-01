# v5038 — Stage A reporting and held-out physical examination

**This is a read-only post-run tool. It cannot and must not change the source
commit of the completed v5037 training runs.** It is maintained on a separate
feature branch. No PPO training takes place, and no policy is promoted.

## Immediate complete comparison report (no GPU)

From the exact **existing** source-pinned training checkout:

~~~bash
cd ~/emu-step-academy
set -o pipefail
git fetch origin '+refs/heads/feature/v5038-step-academy-report-and-exams:refs/remotes/origin/feature/v5038-step-academy-report-and-exams'
git show origin/feature/v5038-step-academy-report-and-exams:scripts/report_step_academy.py | python -
~~~

This is safe despite the completed checkpoint SHA guard: git-show streams only
the reporter to Python stdin. The current Git HEAD and source tree are not
edited. It writes only under
~/robot_lab/emu_runs/v5037_step_academy_stage_a/analysis.

Files:
- step_academy_report.md: eight-arm scores and direct control deltas, actual
  near-frontier UP/DOWN successful encounters and matched physical-height grid
- step_academy_report.csv and .json: complete machine-readable study tables
- step_academy_results.zip: all lightweight CSV/JSON/Markdown and per-arm
  evidence; **does not include full model checkpoint binaries**

## One-command physical science completion

After the report command, use the same git-show with:

~~~bash
git show origin/feature/v5038-step-academy-report-and-exams:scripts/report_step_academy.py | python - --exams
~~~

The tool exports each complete full-state checkpoint to a frozen evaluation
actor, then executes read-only matched CPU MuJoCo exams with identical:
- seeds 1701 and 1702 (matched across every arm)
- 16 worlds and 350 control frames per case
- sharp step heights 1.0, 2.0, 2.5, 3.0, 3.5 and 4.0 cm
- supported up/down crossings and supplementary 3-degree graded floor,
  flat / lateral / yaw skill-retention cases.

Each completed exam is skipped on restart after exact protocol verification.
An existing incomplete exam directory is not silently overwritten. A failed
exam stops the campaign rather than labeling it as success. GPU remains
available while the CPU exam runs. The report and ZIP refresh at the end.

The training frontier UP/DOWN attempt rates must NOT be compared across arms
as if their obstacle heights are fixed. The new physical matrix separates
genuine held-out heights from those curriculum-relative observations.

## Important scope

Stage A trained single-lip physical contacts only. The originally proposed
continuous 4–6-lip Step Academy Stage B and new 2/3/4-point sequence rewards
were **not implemented in v5037**. A report is not a replacement for that work.
The present tool completes Stage A evaluation, not Stage B construction.
