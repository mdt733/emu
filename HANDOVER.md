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

## Live v5047 promotion tournament (temporary, then delete runner)

[Draft PR #37](https://github.com/mdt733/emu/pull/37) is stacked
on the clean PR #36. New **one-optimizer** research includes the same
2,048 CUDA worlds, 360 actor inputs, four recycled 75cm terrain tiles,
eight true convex-prism colliders, 10M genuine walking foundation and
constant 10% fast / 15% CONTROL rehearsal except when explicitly tested.

The disposable `scripts/v5047_pinnacle_tournament.py` runs **seven
independent 100M-step cold-start arms** from identical ancestry:

| Arm | One hypothesis relative to `early_lips` |
|---|---|
| `no_academy` | Control: no concentrated Academy exposure in 100M |
| `early_lips` | Actual multi-lip course born after ~11M, 40% by ~28M |
| `precise_frontier` | ±1mm progression/retreat with 12 real signed attempts |
| `zero_balance` | 10% true zero-vx/vy/yaw steady births; posture credit without artificial stepping requirement **only** for zero commands |
| `temporal_speed` | 50% rolling 0.5s vx tracking blend |
| `fast_rehearsal` | 30% high-speed rehearsal instead of 10% |
| `solver4_control` | 4/4 solver rather than 2/4, for trained-policy physics consistency |

Each policy undergoes **16 independent physical exam protocols**
(two fixed held-out seeds × eight terrain/command cases) with 16
worlds and a 500-control-frame horizon: still (0m/s), slow .06, cruise
.20, fast .70, signed .5cm/2cm/3cm lips, and 8° incline. The frozen
exam reads only raw foot-solid contacts, measured world heading,
footfall timing, stride length and achieved speed. `prism-exam` now
inherits the actor's exact 2/4 or 4/4 solver **from the export**.
An eligible contender requires held-out flat and step survival and
actual forward movement, not raw training reward.

An objective-reported finalist starts **fresh at 300M** with the
winning source-pinned teacher hypothesis, and a **200M different-seed**
replication is attempted if wall time allows. Total potential
**1.2 billion world-steps**, up to 144 frozen exams. A 9h
soft wall-clock budget is enforced **between arms**; an already
running training/exam isn't killed or left with a silently truncated
checkpoint. Actual duration depends on JAX compilation/physics.
No automatic merge or champion release.

### After the current fresh WSL installation

The install guide originally checked out v5046 with `--single-branch`.
Use this explicit fetch once; it won't depend on having fetched every
remote branch or require a second Python environment:

```bash
cd ~/emu
git status --short
git fetch origin \
  experiment/v5047-pinnacle-stairs-gaits:refs/remotes/origin/experiment/v5047-pinnacle-stairs-gaits
git switch --track origin/experiment/v5047-pinnacle-stairs-gaits
source .venv/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python -m pip check

# Plans resolve all seven real CLI configs and do ZERO GPU work.
PYTHONPATH="$PWD" python scripts/v5047_pinnacle_tournament.py plan

# After plan + exact SHA CI and 2,048-world CUDA acceptance,
# foreground nine-hour soft-budget tournament, one CUDA process at a time:
PYTHONPATH="$PWD" python -u scripts/v5047_pinnacle_tournament.py run --hours 9
```

If `git status --short` is nonempty with tracked edits, commit/stash
rather than switching over active work. The local `.venv` is editable
from this same checkout; it does **not** need a second JAX download
after the existing CUDA installation. If pip dependencies were changed
then rerun the maintained installation from this new SHA.

Outputs are strictly canonical:
`~/emu-data/runs/v5047-<arm>-s<seed>/` for run-owned
checkpoints/metrics/exams/exports and
`~/emu-data/scratch/v5047-pinnacle-tournament/` for the
temporary manifest, logs, continuous `SUMMARY.md`, and an
upload-ready **`results_bundle.zip` without enormous `.pkl` files**.
Reissue the identical `run --hours 9` against **the same Git commit**
to resume an interrupted campaign; don't pull or merge mid-run.
If `status` is desired: `python scripts/v5047_pinnacle_tournament.py status`.

### What would earn a new Pinnacle?

Not a high PPO reward or 120k GPU steps/s by itself. We want
stable two-foot slow progress and near-zero commanded drift; high-speed
gait with real stride-length, speed, steering and bounded effort;
**repeated up AND down supported signed lips on independent seeds**;
preserved slopes, turn control and recovery. A promising result can
become a replacement-main candidate if the trained actor is roughly
champion-equivalent and the 75cm carpet/faster trainer are measurably
better. The old champion remains frozen and available for rollback.

**Explicit release hold:** 2/4 solver penetration and CUDA compact/full
`.qacc` acceleration mismatch still require investigation before
claiming hardware-faithful dynamics. The original main actor uses a
historical training world, so comparing its old physical exam with
v5047 requires a clearly documented equivalent-mission bridge.
Final merge/publish needs explicit user approval.

**Cleanup contract:** this one experimental runner is deleted after
the results are analysed and committed as concise permanent findings
in issue #17 / HANDOVER. Keep any successful general-purpose
sampler/reward/exam improvement; discard one-off tournament
orchestration and its extra state.

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
