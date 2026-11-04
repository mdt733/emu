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
