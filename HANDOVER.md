# EMU engineering handover — v5046, 5 November 2026

**Current source:** `mdt733/emu`, draft [PR #35](https://github.com/mdt733/emu/pull/35),
branch `experiment/v5046-75cm-four-tile-midpoint`.
[PR #36](https://github.com/mdt733/emu/pull/36) is a **separate draft
cleanup** stacked on #35; do not advance source-pinned experiment
checkouts without starting a new run lineage. The released 260M
360-input Pinnacle remains in `main`, unchanged.

The [historical v5045 cleanup handover](https://github.com/mdt733/emu/blob/97d419d48340173dad8af3e477b1c3ee333cd02e/HANDOVER.md)
and older Git commits retain the per-file archaeology; this handover
only describes the live system and next scientific decisions.

## Current runnable skeleton

- **One six-servo biped**, one PPO actor/critic, 360 input channels from
  sensor and command history (4×4 ToF, IMU, flow and proprioception).
  Six specialists are *birth cohorts*, not six policies.
- **One rolling 2×2 carpet of four 75cm tiles**, each two exact
  convex triangular-prism colliders = eight true physical contact solids.
  No legacy slab/box colliders. Terrain geometry and policy inputs
  remain distinct.
- **One source-verified trainer**: `emu train` → `emu/training.py`,
  `emu/environment.py`, `emu/curriculum.py`, `emu/rewards.py`;
  `emu/evaluation.py` and `emu/prism_exam.py` are the independent
  examiners; `emu/checkpoint_io.py` owns checkpoint integrity.
- Solver iterations/linesearch are **explicit opt-in** saved training
  physics. `robot.xml` and `main` remain unchanged.
- **21 focused Python tests**, browser checks and actual MJX geometry
  acceptance. The 34-arm solver sweep was temporary experimental
  infrastructure; its code is removed in PR #36, results retained.

## What has actually been measured

### 34-arm speed study, two seeds, RTX 4060 Ti 8GB

Default MuJoCo solver 100/50: **31,974** steady full-PPO world-steps/s.
4/4: **89,921**; 2/4: **119,662**; 1/1: **162,930**
but the last has unacceptable early contact/gait warning signals.
The 2/4 setting is a *research candidate*, not yet signed-off
contact/step physics. Re-enabling torso collision cut 4/4
throughput by 43%; the resulting feet-only physical approximation
requires virtual-strike safety exams. Reset batch cap 128 stays
because the 192/256 speed results were inconsistent across seeds.

On the operator's CUDA GPU, compact/full reset `mjx.forward`
disagreed in `.qacc` by up to 0.01096 in the diagnostic;
CPU preflight passed. **Do not loosen tolerances** until later-step
contact and trajectory consequences are examined.
[Measurements and caveats](docs/V5046_OVERNIGHT_RESULTS_20261105.md).

### First 20M continuous 2/4 learning test

At 20,054,016 world-steps (seed 735): latest rollout falls **2.1%**,
achieved vx **0.094 m/s** vs commanded **0.119 m/s**,
held-out score **+0.530** (from -0.089 at 10M).
Core/stress30/dynamic50/combo70 independent 350-frame panels
survived 100%; **no verified up/down terrain lip successes**.
This is evidence of a robust slow learner, not yet parity with
the champion's speed, yaw/heading authority or stair crossing.

Steady PPO was around 119k world-steps/s; setup compilation and
independent frozen evaluations dominated wall time. Do not replace
`main` on training success alone. The policy's full `final.pkl`
was **not** present in the uploaded measurement bundle, and can
be discarded by user choice if training anew.

## Next actions, in order

1. **Current clean WSL install:** use [the canonical INSTALL guide](docs/INSTALL.md).
   One checkout `~/emu` with local `.venv`, data under
   `~/emu-data/runs/<study>-s<seed>/`, disposable audits under
   `~/emu-data/scratch/`. No extra roots, no zip-unpacking into source.
2. **Step discovery study:** start from the same valid 10M foundation,
   move streaming Academy introduction from 60M to near the end of the
   foundation, keep real signed lips starting ~3–5mm and ample control
   rehearsal. First establish genuine foot-on-upper-surface support,
   continued forward progress and safe downstep placement. Reward
   reallocation is a *second independent hypothesis*, not combined
   blindly with changed curriculum.
3. **Slow and zero command tests:** record world-frame drift, commanded
   vx/vy/yaw versus actual, duty cycle, support and shaft J/m; distinguish
   standing stabilisation and small deliberate corrective steps.
4. **Fast gait study:** measure stride length, cadence, speed, swing
   clearance, torque and heading control. Only after baseline profiling
   investigate temporal velocity tracking, changed rehearsal dose or
   relaxed crouch-to-release when they address an observed failure.
5. **Promotion:** compare a frozen contender with the `main` champion
   on same physical missions. Engineering throughput and terrain
   maintainability **count** even if speed only matches; maintain
   physics fidelity, steering and supported step control. Merge only
   after explicit user approval. Keep PR #35 and PR #36 draft meanwhile.

The detailed unanswered backlog is [GitHub issue #17](https://github.com/mdt733/emu/issues/17).
The 260M reference measurements are [here](docs/V5032_PINNACLE_RELEASE.md).

## v5048 — shared-parent Pinnacle development tournament

The aborted v5047 **independent cold-start** tournament is retired
because it was not the requested experiment and its physical examiner
retained a stale **9×50cm** reshape. The new [draft PR #38](https://github.com/mdt733/emu/pull/38) fixes the frozen observer to use
**4×75cm** actual prism source tiles (including correct rise/run
normals). The previous v5047 CUDA gate stopped BEFORE any 100M policy
training, so it produced no reusable walking checkpoint. Its old
temporary script was removed on this branch.

### One ancestor, eight controlled forks, one continued winner

| Stage | Cumulative age | Unique simulation work | Physical purpose |
|---|---:|---:|---|
| Cold foundation (seed 735) | 50M | **50M** | Learn one coherent basic walker and normalizer on the full four-tile carpet |
| Eight independent treatment forks | 100M each | **8×50M** | All inherit the **same immutable 50M final.pkl** actor, optimizer, normalizer, curriculum, health and capability field |
| Winning fork grows | 200M total | **+100M** | Continue the **same** best 100M checkpoint and training run, not another fresh cold-start |
| **Total** | — | **550M real world-steps** | One shared-PPO trunk, eight matched research comparisons, one ambitious Pinnacle candidate |

The one-predetermined-change A/B bracket:

| Arm | Change after 50M |
|---|---|
| `no_academy` | Control: continue normal mixed-world training without dedicated Academy |
| `early_lips` | Activate true signed repeated-lip Academy immediately at the 50M fork; ramp to 40% by ~60M |
| `precise_frontier` | Same Academy plus 1mm signed promotion/retreat increments and 12 evidence attempts |
| `zero_balance` | Same Academy plus 10% true zero-vx/vy/yaw steady births and safe zero-command posture credit |
| `temporal_speed` | Same Academy plus 50% rolling 0.5-second forward-command reward blend |
| `fast_rehearsal` | Same Academy plus 30% rather than 10% fast-command rehearsal |
| `heavy_academy` | Same Academy plus 65% rather than 40% eventual dedicated step births |
| `solver4_control` | Same Academy but Newton/linesearch solver **4/4** instead of **2/4** for the 50M research fork |

All other physical terrain, observation/history structure, PPO, reward
and specialist curriculum stay identical. A single source SHA and seed
pin all eight parent-trained policies; solver4 is deliberately labeled
an explicit post-fork physical change, and never silently substitutes
one physics mode in another checkpoint or independent examination.

At 100M, each frozen actor receives eight independent real-physical
commands/terrain modes on TWO held-out seeds and 16 GPU worlds:
0m/s stillness, .06 slow, .20 cruise, .70 fast, signed .5cm/2cm/3cm
up/down steps and 8° grade. Each mode is 500 control frames.
The original 50M policy gets four selected baseline protocols.
After winning-branch extension to 200M, repeat all eight two-seed
frozen exams. **148 planned independent physical exam invocations**,
counting the four foundation checks, all 128 fork checks and the final
16; these are not 148 permanent pytest files.

The independent examiner accepts only **real top-prism contacts**
with sustained supported foot placement and actual signed terrain
transition; it records command error, gait cadence, stride length,
stillness drift, survival, power and heading. Training reward and
curriculum counters do NOT grant stair mastery. Step results are
downweighted if only one held-out seed crosses. There are **two**
predeclared selection gates: a strict physical contender threshold
(flat ≥65% survival, step ≥50%, fast ≥0.20m/s), and a lower but still
meaningful research-continuation gate (flat ≥30%, step ≥20%, fast
≥0.05m/s). If the strict gate fails but an exploratory fork survives,
the best one may gain the extra 100M steps to see whether it matures;
that is **not a claim of main parity or Pinnacle status**. If all
fail even the lower gate, the campaign stops and reports the failure.

### Exact clean-WSL launch, one checkout / one virtualenv

After the current source-pinned v5047 experiment has stopped, switch
the existing `~/emu` checkout. Because the initial Ubuntu clone was
`--single-branch`, explicitly add the new remote tracking ref first:

```bash
cd ~/emu
git status --short    # must show no uncommitted tracked edits
git remote set-branches --add origin experiment/v5048-shared-parent-pinnacle
git fetch origin
git switch --track origin/experiment/v5048-shared-parent-pinnacle
git branch --show-current
git rev-parse --short HEAD
source .venv/bin/activate

# No training, no GPU and no folders: resolves/validates all 9 real
# CLI stage recipes, 50M parent and eight 100M fork targets.
PYTHONPATH="$PWD" python scripts/v5048_shared_parent_tournament.py plan

# Only AFTER the matching PR GitHub CI is green:
PYTHONPATH="$PWD" python -u scripts/v5048_shared_parent_tournament.py run --hours 9
```

The runner gates the first 50M with a real 2,048-world CUDA PPO
update, then tiny CUDA checkpoint→actor-export→100-frame independent
exam lifecycles for 2/4, 4/4 and the new zero-command sampler.
The latter specifically detects the previously failing four-tile
frozen-policy reshape **before spending GPU-hours on a foundation**.
It checks each external research parent's exact checkpoint SHA256
and same-source actor/sensor compatibility using the trainer's
existing `--check-resume-only` path before running a child.
Changed-teacher forks do not inherit the parent's old rollback policy.
A missing checkpoint from an interrupted *first compilation* is
safely retried from the original immutable ancestor, while
completed/partial children use their own validated `--resume auto`.
The script never kills a running PPO update when its nine-hour
soft budget expires; it checks between stages and can resume only
on the identical source/recipe fingerprint.

Everything under two established home-level roots:

```text
~/emu/                                   # source + one .venv
~/emu-data/runs/v5048-foundation-s735/  # 50M immutable parent final.pkl
~/emu-data/runs/v5048-<arm>-s735/       # all training checkpoints,
                                          # step_100M/200M actor exports,
                                          # exact physical exams and reports
~/emu-data/scratch/v5048-shared-parent/  # temporary harness state, logs,
                                          # scoreboard and results_bundle.zip
```

`status` reads already completed evidence:

```bash
cd ~/emu
python scripts/v5048_shared_parent_tournament.py status
```

The upload-ready `results_bundle.zip` contains **metrics, metadata,
frozen examiner JSON, and summary**, intentionally NOT giant
`final.pkl` blobs. If a genuine 200M contender emerges, compare its
actual gait/command/control/steering to the preserved Pinnacle on
equivalent physical missions before promoting the engineering
architecture to `main`. The 2/4 acceleration-parity issue still
needs bounding. Do **not** automatically merge any branch.

**Cleanup obligation:** once measured, remove
`scripts/v5048_shared_parent_tournament.py`, save its findings in
issue #17 and the current report, and keep only working trainer,
examiner, checkpoint and compact tests. The old v5047 study script
has already been deleted on this branch.

## Run hygiene and release rules

```text
~/emu/                            Git source + single .venv
~/emu-data/runs/<study>-s<seed>/   run_meta, config, console, metrics,
                                   fixed exams, reports and checkpoints
~/emu-data/scratch/               throw-away A/B experiment output
```

For real training, require a new empty run directory, a clean Git SHA,
one `emu train --run-dir <absolute path> --plan` then the matching
approved training command. `--resume auto` uses **the exact original
source** and unchanged config. Do not `git pull` mid-run. Export
a browser `.emupolicy` alongside the originating run when needed.

**No more one-off launcher libraries.** The CLI owns training, the
current evaluator owns physical evidence, and completed research
is a concise result under the existing report/issue rather than
another permanent script. Full maintenance rules:
[docs/MAINTENANCE.md](docs/MAINTENANCE.md).
