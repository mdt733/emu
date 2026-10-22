# EMU — current handover (v5020 experiment, October 2026)

**Reference `main` is PROTECTED.** It contains the known-working v5016
Pinnacle 232-input actor and approved 260M showcase. Do not modify
`main`, the public Playground or its champion policy until a stronger
trained result has been independently demonstrated and the user
explicitly approves a reviewed merge.

**Sole current experimental source branch:**
[`feature/v5020-coldstart-terrain`](https://github.com/mdt733/emu/tree/feature/v5020-coldstart-terrain).
All changes in this handover refer **only** to that branch.

## Terrain implementation acceptance — 22 October 2026

**Overnight launch is blocked in source until the remaining acceptance gates
pass.** `--check` performs no PPO update and reports NOT OVERNIGHT READY;
`--run` exits before environment setup or run-directory creation.

Foundation qualification now uses an additional fixed +0.10 m/s forward
panel. At least 20% of original worlds must complete the full exam without
falling, travel at least 5 cm and 30% of commanded distance, show mean lift
evidence >=0.12, upright fraction >=0.80, support fraction >=0.50 and forward
tracking MAE <=0.08 m/s. These are initial basic-walking thresholds, not proof
of a strong policy or certified force-based gait. Advanced skills are not
required. Old survival-only CSVs are unknown and cannot auto-qualify or bypass
the gate through an explicit path.

Frozen summary metrics require full first-episode coverage for survival
(including non-fall timeouts) and exclude reset aftermath and report tracking/gait
means over observed first-episode frames. Walking qualification requires the
whole horizon, so incomplete evidence cannot qualify. Tournament metrics
reject negative errors/effort and out-of-range fractions.

Checkpoint and 312-input sensor formats are unchanged. Metric columns have
expanded: new source uses a new pinned run root; do not append to historical
CSV files. Same-revision full-state resume remains supported.

Validation: focused qualification/scoring/refork/launcher regressions pass,
including the actual JAX-compiled evaluator with controlled telemetry.
Physical training and real-GPU acceptance remain unproven here. This host
has Python 3.12 CPU dependencies, not the supported fresh-WSL Python 3.13/CUDA
stack; do not treat these tests as fresh-PC acceptance.

Elevation groundwork now includes `terrain.elevation_course`: static 5/10/20/
30/40 mm up/down exam models with a two-metre upper landing, the unchanged
lower plane and exact first-hit sensing through the existing named box.
Static box-surface births use the actual upper surface when the robot starts
on a down-step platform. Native MuJoCo tests check top contact without phantom
floor support, ray normals, and ToF/native first-hit parity at 5/10/20 mm in
both directions; a two-world 312-input MJX reset checks down-step birth height.
These are exam fixtures only: batched random elevation lessons, post-landing
walking certification, sensor-memory diagnostics and curriculum integration
are STILL PENDING. The old hurdle crossing classifier is not elevation
certification and no elevation training competence is claimed.

Frozen terrain exams now revoke a recorded crossing if the robot later falls
in that original episode. Efficiency summaries also exclude reset aftermath.
Training event reduction rejects fractional/contradictory completion,
contact, crossing and fall records.

Validation record: baseline full suite finished with 196 passing tests and
one existing launcher-help assertion failure (fixed and passing on rerun).
The focused qualification/scoring/refork/launcher suite now passes 69 tests;
elevation geometry/ToF/contact/MJX-birth checks pass 7 tests. An earlier
72-test focused integration run included real three-world 312-input MJX
terrain and frozen exams. These overlapping runs must not be added together
as a count of unique tests. The complete suite was not rerun after all edits.

Next: replace the 80 mm hurdle lesson with genuine elevation transitions,
validate sensor reconstruction, integrate browser 312 inference, and run the
small real-GPU lifecycle before removing the launch gate.

## Current status

- Physics: six-actuator biped in MuJoCo/MJX, real collidable boxes and
  contact slopes, 312-input four-frame rolling ToF/flow preview.
- Training: 2048/4096 worlds, 32-frame PPO rollouts, on-policy governor
  and capability-field evidence, complete resumable checkpoints.
- New campaign prototype: `emu campaign-refork`; five strategies
  (General / Turning / Recovery / Efficiency / Steps), one common
  parent per 20M generation, a 2M failure probe per contender,
  matched frozen exams and single-winner re-forking.
- Height: step specialist begins low and grows from verified real
  crossings to a **current simulator maximum of 150mm**; exam heights
  are 5/10/20/40/80/150mm for every contender.
- State of proof: source and smoke tests have been run, but **no
  qualified successor policy exists yet**. The earlier 20→80M
  v5020 four-arm run collapsed; do not mistake its 380M aggregate
  steps for learned walking.
- A known design gap remains: the Champion Ladder stages vx/vy/yaw
  physical command maxima and sets `open_frontier=False`, while
  physical step height adapts independently. Unify carefully in
  shadow mode before enabling multi-axis frontier control.

## Reading order

1. [Current architecture](docs/EMU_ARCHITECTURE.md) —
   physics, measurements, PPO, governor, independent examiner and
   champion selection
2. [Consolidation checklist](docs/V5020_CLEANUP_PLAN.md) —
   staged source cleanup and strict exit gates
3. [Curated historical findings](docs/RESEARCH_HISTORY.md) —
   old campaigns, what was learned and what failed
4. [WSL setup and safe CLI](docs/INSTALL.md) —
   current read-only plan, schema checks and bounded first GPU run
5. [Current five-way recipe](docs/V5020_WINNER_REFORK.md) —
   absolute step targets and independent screening

Earlier handovers are in Git history, not multiple competing front-page
instructions. Existing training and exam output roots are **not**
source-code artifacts and are never removed by repository cleanup.

## Outstanding validation before another substantial GPU run

1. Confirm the exact 312-input foundation candidate from the user's
   saved checkpoints (including performance near 16–20M).
2. Ensure a clean pinned source, matching world count and a brand-new
   output directory; never overwrite or cross-revision resume.
3. Run the full test suite and real MJX contact/sensor smokes.
4. Start with just the five short candidate probes.
5. Examine survival, gait, numerical health and frozen course results
   before spending the remaining steps to 40M.
6. Promote a candidate only if it beats the incumbent with no
   unacceptable regression. Stop if none qualifies.

Historical research code can be retired from *this experimental branch*
after dependency audits, but the known-working reference remains
untouched in `main` and every historical Git object remains available.


## 2026-10-22 — foundation + genuine elevation examiner integration

Development remains ONLY on `feature/v5020-coldstart-terrain`. `main`,
the Pinnacle policy and the public showcase are untouched. The five-arm
overnight launcher remains explicitly BLOCKED pending full acceptance.

The existing frozen 20M foundation forward exam now requires its *complete*
first-episode evidence: independent survival, commanded/actual displacement,
stepping, posture, contact support and velocity error. An isolated pass-rate
number is no longer an acceptable exam. The winner-refork's held-out
screen also has a dedicated easy commanded-forward panel; survival-only
policies cannot pass the eligibility gate or continue provisionally.

True elevation geometry is a SEPARATE examiner from the original short-box
hurdle curriculum. The CPU/MuJoCo fixtures provide long raised landings for
both step-ups and step-downs. The new JAX frozen elevation examiner checks
source-side support, real step contact, both feet and torso past the edge,
destination-surface contact, upright posture, at least eight supported
destination frames, post-landing gait, and >=40mm further progression.
It reports contacts, certified crossings, falls, timeouts and incompletes;
reset episodes cannot create a new attempt. The static geometry and physical
contact oracle are examiner-only and are never actor observations.

Small non-training regression command from a fresh checkout:

```bash
python -m pytest -q tests/test_foundation_readiness.py tests/test_v5020_refork.py tests/test_v5020_coldstart.py
bash scripts/overnight_v5020.sh --check
```

Once an ACTUAL 312-input checkpoint exists, run separate, read-only exams:

```bash
emu screen-elevation --checkpoint "$FOUNDATION_PATH" \
  --course-root "$HOME/robot_lab/emu_runs/v5020_elevation_checks" \
  --direction up --height-mm 10 --envs 8 --horizon 350 --device cuda
emu screen-elevation --checkpoint "$FOUNDATION_PATH" \
  --course-root "$HOME/robot_lab/emu_runs/v5020_elevation_checks" \
  --direction down --height-mm 10 --envs 8 --horizon 350 --device cuda
```

`FOUNDATION_PATH` must be set explicitly to a compatible saved 312-input
checkpoint. The command does not rewrite checkpoint, metrics or campaign
outputs. Geometry files are immutable: rerunning identical conditions is
safe, differing files fail closed.

**NOT YET ACCEPTED:** no verified GitHub Actions/MJX/GPU results for this
revision, no demonstrated successful real-model elevation crossing, no
integrated mixed-height per-world elevation TRAINING curriculum, and no
312-input browser policy execution parity. These remain hard gates against
enabling `overnight_v5020.sh --run` or promoting a Pinnacle successor.


### 20cm actor lookahead / observation contract

The forward ToF model nominally samples two spots 20cm ahead; previously
the 312 actor read out only through 15cm. Since the rolling map deposits
each real hit within a narrow 2cm window, a newly sighted edge could be
visible in the full 51-cell diagnostic while still absent from *all* the
actor's 20 terrain channels. The fifth distance on each side is now 20cm:
`[-5,0,5,10,20]` cm, still ten heights + ten confidences per 78-field
observation and four-frame 312-input actor history.

This intentionally changes the semantics of two channels on each frame.
`PREVIEW_RANGE_FIELDS`, `PREVIEW_VALID_FIELDS`, sample indices and the
rolling model revision were changed together, so the persisted sensor
fingerprint changes. Existing experimental 312 checkpoints with the older
15cm map contract are NOT silently accepted. Historical 220/232 observation
schemas remain unchanged. Browser 312 policy parity is still outstanding.

The new regression sends two forward rays and three tripod rays through
the *same* long-box collision geometry for true 20mm up/down transitions,
projects only sensor returns through the tripod estimator, inserts measured
signed heights into the V memory, checks the actual actor readout and
requires invalid rays to remain unknown (zero confidence). This is a
source-level numerical/MuJoCo test, not evidence of learned competence.
The elevation examiner also reports source-side exposure separately from
contact, crossing, falls, timeouts and incomplete attempts.


## 2026-10-22 afternoon — opt-in training on genuine elevation platforms

Local evidence supplied by the user for earlier commit `d10dfd6`:
98/98 focused tests passed; `overnight_v5020.sh --check` passed its
111/111 focused tests, detected JAX 0.12 / CUDA GPU and MuJoCo 3.16,
and CORRECTLY refused the older 312-input 15cm-fingerprint foundation.
The resulting check reported NOT OVERNIGHT READY, as intended.

**New source after d10dfd6 still requires fresh regression and MJX runs.**
No subsequent tests, learned step competence, or large GPU campaign
are asserted by the GitHub edits alone.

The experimental `--terrain-elevation-mode up|down|mixed` now selects a
**fixed-topology 2m-long, 1m-wide physical raised platform**, not the
historical 80mm hurdle. The historical hurdle is still the default and
its generated XML remains byte-identical. A distinct
`v5020_adaptive_elevation_course.xml` is immutable and pinned in run
metadata; position and height vary independently across worlds only at
episode birth. Step-down births start at the actual raised collision
surface. Flat rehearsal births park the box out of the course.

The same sensor-visible collidable box is consumed by both forward
beams, the three tripod rays and now the optical-flow camera depth
on 312-input courses. No geometry height is directly given to the actor.
The sensor fingerprint was advanced to include the camera-depth
first-hit correction, invalidating older experimental 312 contracts.

The trainee's terrain crossing metric is now evaluated at the REAL
transition edge of the long platform, not the far end of a hurdle.
The existing episode-only scorer demands source support, obstacle
contact, upright two-foot destination passage, grounded support,
eight supported destination frames, >=40mm extra forward travel and
post-landing gait evidence. It is used for both teaching outcomes and
frozen evidence, while 220/232 and old hurdle defaults stay isolated.
This is instrumentation, NOT proof a neural policy learned to climb.

Before any extended run, execute:

```bash
cd ~/emu-v5020
git switch feature/v5020-coldstart-terrain
git pull --ff-only origin feature/v5020-coldstart-terrain
source .venv/bin/activate
python -m pytest -q tests/test_v5020_coldstart.py tests/test_scoring.py \
  tests/test_v5020_refork.py tests/test_foundation_readiness.py \
  tests/test_overnight_launcher.py
emu train --profile sim2real --forward-preview \
  --terrain-varied-placement --terrain-elevation-mode mixed \
  --terrain-height-ceiling-mm 20 --terrain-height-initial-mm 5 \
  --envs 64 --steps 131072 --seed 733 --device cuda \
  --run-dir "$HOME/robot_lab/emu_runs/v5020_mixed_elevation_plan" --plan
```

Only after those tests pass, a disposable *small* cold-start GPU
lifecycle (NOT an overnight campaign) is available:

```bash
cd ~/emu-v5020
RUN_ROOT="$HOME/robot_lab/emu_runs/v5020_platform_smoke_$(git rev-parse --short HEAD)"
XLA_PYTHON_CLIENT_PREALLOCATE=false emu train --profile sim2real \
  --forward-preview --terrain-varied-placement \
  --terrain-elevation-mode mixed --terrain-height-ceiling-mm 20 \
  --terrain-height-initial-mm 5 --terrain-height-exposure .65 \
  --envs 64 --steps 131072 --seed 733 --device cuda \
  --run-dir "$RUN_ROOT"
```

These early training steps are to prove reset, physical stepping, PPO,
finite gradients, saving and fixed exams compile and run, **not**
to assert competence. Do not reuse a smoke root after source changes.
The long overnight launcher `--run` remains blocked until browser 312
parity, independent elevation exams and real GPU learning are accepted.
The five-arm tournament still uses the historical hurdle by default:
it must NOT silently switch to new elevation lessons before acceptance.

 
## 2026-10-22 evening — requested full-scale 2048-world cold start

The earlier suggested **64-world GPU training smoke is superseded** by the
user's explicit decision to run real 2048-world experiments immediately.
Source is still ONLY `feature/v5020-coldstart-terrain`; the protected
Pinnacle main/showcase and `overnight_v5020.sh --run` stay untouched/locked.

New `scripts/run_v5020_2048.sh` is an **opt-in** shell entry point;
`--plan` has no side effects; `--run` actually trains synchronously on
the user's RTX GPU. No GPU training has been started remotely by ChatGPT.
Each training phase uses exactly 2048 environments, 32-frame PPO rollouts,
saved fixed examinations, `metrics.csv`, source SHA, logs, resumable
snapshots and exact parent checkpoint hashes. A failure stops the script
but leaves all output intact; rerun same command on the exact same
source revision to resume a valid same-stage snapshot.

Stages (absolute, cumulative training transitions):

| Stage | Target | Physical training |
| --- | ---: | --- |
| foundation | 20M | fresh, flat-first, 312 inputs; require exact-checkpoint basic walking exam |
| up | 32M | 2m raised platform, real up steps, 6mm height safety ceiling |
| mixed12 | 48M | raised step up/down, 12mm ceiling |
| mixed20 | 70M | raised step up/down, 20mm ceiling |

These are *ceilings*, not guaranteed earned step heights. All phases
retain flat rehearsal, while evidence-gated automatic height progression
should protect ordinary walking. Larger steps wait for evidence.

One copy/paste launcher command after synchronising the branch:

```bash
cd ~/emu-v5020
git switch feature/v5020-coldstart-terrain
git pull --ff-only origin feature/v5020-coldstart-terrain
bash scripts/run_v5020_2048.sh --run
```

For a safe read-only plan: `bash scripts/run_v5020_2048.sh --plan`.
To stop after a completed intermediate stage:
`bash scripts/run_v5020_2048.sh --run --stop-after up`.
Use `bash -n scripts/run_v5020_2048.sh` and
`python -m pytest -q tests/test_v5020_2048_launcher.py
tests/test_v5020_coldstart.py` for source tests before launching.

No compiled/genuine 2048-world mixed-elevation acceptance is claimed.
An OOM, compile exception, failed fitness gate, or physically incoherent
run are legitimate findings. The caller should provide the stage's
console.log, `metrics.csv`, and `fixed_eval.csv` from the default
`~/robot_lab/emu_runs/v5020_2048_<commit>` output root.


## 2026-10-22 evening — critical 20M→20.7M collapse and revised height doctrine

The user reported a real 2048-world 312-input experiment:
- Original 20M foundation: 700-step episodes, ~0% falls, reward +2.89.
- First `up` continuation: 13–15-step episodes, 100% falls,
  reward about -8.6; reported vx became unphysical, including -50m/s.
- **Stop the original v5020_2048_624bcb92db8c/up run; NEVER resume its
  damaged child checkpoint.** Its original 20M foundation best.pkl is
  still valuable and must not be overwritten.
- Root cause not yet isolated in GPU physics. Confirmed independent launcher
  mistake: old stage switched `foundation` to the aggressive mature
  `sim2real` curriculum (commands, push, phase speed, field and normaliser)
  exactly when introducing the long raised box. This violates continuity.
  A possible additional physics problem was the 2m kinematic platform
  deeply overlapping the static floor. Distinct collision masks now prevent
  terrain-terrain pairs while retaining real foot-plane/foot-box contacts.

User's **new binding curriculum doctrine**:
1. Every stage transition inherits policy/optimizer/normaliser, full governor
   and capability field state, command distribution, reward and perturbation
   settings. Only the newly introduced physical dimension changes.
2. The learned terrain frontier is a **skill rating**, not the highest
   height shown in training. Sample an explicit mix BELOW, NEAR and ABOVE
   the rating, with independent flat rehearsal and an immutable absolute
   hardware/physics safety ceiling.
3. At 0mm skill, start tiny above-rating probes (up to 0.25mm), not an
   unearned fixed 3mm jump, nor a free promotion to 0.25mm. The 0mm
   **rating stays zero until actual completed physical crossings succeed**.
4. Promotion requires enough completed >rating attempts with physically
   supported post-landing locomotion and tolerable falls; cap new rating
   to the MEASURED successful challenge height. Demotion is driven by
   failure at easier <rating heights; challenging failures alone HOLD.
5. Flat, slope (starts at 0 degrees), ascent and descent are separate
   skill axes. Adding descent cannot silently start at the previously
   earned ascent rating. Independent positive/negative slope ratings
   and descent-rated training still need to be integrated and tested.
6. **Never claim evidence is passed without user's WSL tests.**

Implemented feature-branch changes:
- `emu/terrain.py` new rated distribution and pure
  `rated_height_update`; old hurdle sampler/old governor retained.
- `emu/environment.py` birth-rated ability recorded as teacher-only
  rollout diagnostic; sampling clipped to explicit immutable course safety.
  All positive-height attempts (including easier practice) are eligible
  to count completed physical trials.
- `emu/scoring.py` independently counts DONE, crossing, contact, fall,
  and actual successful metres for below/above-rated trials. Long-lived
  episodes do NOT count as repeated attempts.
- `emu/curriculum.py` pools per-band evidence across PPO updates,
  runs the rating controller only in opted-in long-platform training,
  and migrates old source-pinned foundation checkpoint state.
- `emu/train_cli.py` supports `--profile terrain-bridge` with identical
  foundation PPO/gait profile instead of the mature switch, and requires
  new elevation plans to declare 0mm initial rating.
- `scripts/run_v5020_2048.sh` defaults only through the 32M up stage,
  can use `FOUNDATION_FROM` to reuse the original 20M foundation without
  rerunning it, and accepts a different-source parent only when its commit
  AND checkpoint SHA256 are explicitly pinned/validated.
  The mixed up/down stages are LOCKED rather than silently restarting
  descent at a nonzero rating. Terrain coaching reward is 0 in this
  continuity-first study.
- `emu/training.py` stops a catastrophic rated elevation experiment after
  four consecutive 2048-world short-episode 90%+ fall rollouts.
- Added `tests/test_v5020_rated_elevation.py`; the 2048 launcher now
  runs these and its own source tests before entering GPU training.
- Protected `main`, original 20M checkpoint, and the traditional
  `overnight_v5020.sh --run` lock remain unchanged.

**IMPORTANT: New source is UNTESTED ON THE USER'S GPU.** This branch only
contains source changes and authored regression tests. Run tests and
validate a flat zero-height equivalence/physics gate before spending
more substantial 2048-world GPU transitions. A mixed-slope and
independently-rated downhill campaign remains future work.


## 2026-10-22 — hardened non-breaking 2048-world restart

After the 100%-fall 20M->up catastrophe, do NOT run the old
`624bcb92db8c` continuation. That child is rejected as evidence of
learning. **Keep the exact qualified 20M best.pkl in place.**

The current branch's conservative launch now enforces, IN ORDER:

1. No dirty Git checkout; only `feature/v5020-coldstart-terrain`.
2. Fast source tests for rated-height distribution, separation of
   easy/challenge completed trials, quarantine and CLI handoff.
3. Look for the previously qualified 20M foundation at
   `~/robot_lab/emu_runs/v5020_2048_624bcb92db8c/foundation/checkpoints/best.pkl`.
   It is accepted automatically ONLY with SHA256
   `23ca16ecd7de3b9980108d3833bb47dd8276f02aab6a2db35a3f30208dd4fc23`.
   Any other source must be explicitly specified with `FOUNDATION_FROM`.
   If nothing is available, fail CLOSED; `COLD_START=1` opts in to a
   truly new foundation on a blank PC.
4. Verify exact-step saved flat walking evidence and unchanged 312-channel
   fingerprint, optimiser, normaliser and model schema.
5. **NEW frozen CPU verification before PPO updates:** same saved network,
   same normaliser, same steady +X command and seed on three independent
   worlds: original flat robot.xml, long-platform physics/sensors held at
   exactly 0mm, and genuine 0..0.25mm height probes. All use only 4 envs
   to check continuity (NOT a 64-env training smoke). Abort before GPU
   training for lost frozen forward walking or early falls. Each world
   has its OWN JAX-traced function: do NOT mutate one JIT closure between
   zero and positive height cases. Logs saved as `bridge_parity_*.log`.
6. Trainer compare saved original foundation config against bridge config
   and reject ANY changed command envelope, phase rate, push ceiling,
   reward, PPO economics, actor/normaliser, capability field mode, or
   any unrelated setting. Only the new physical terrain sampler/geometry,
   target step count/run location, and explicitly approved old-source
   parent hash may differ.
7. Remove the UNTRAINED historical `RegimeParams.frontier_terrain_height_m
   = 0.003` placeholder from the incoming flat checkpoint. Actual
   new elevation rating is exactly 0mm. All other saved learner/
   governor/health/field state survives intact.
8. Sample continuous step-up heights below/near/above earned skill,
   subject to the explicit 6mm safety limit. At initial 0mm rating,
   only 0..0.25mm positive probes enter. Promotion requires physical
   crossings at ABOVE-rated heights and cannot exceed the MEASURED
   successful height. Failure at easier heights permits retreat.
   No step completion is inferred from just proximity or box touching.
9. If 4 consecutive 2048-world PPO rollouts experience catastrophic
   short-episode falls, OR any physical vx exceeds 8m/s, write
   `UNSAFE_RESUME.json`, abort, and reject ALL future `--resume auto`
   attempts for that child lineage. The original parent is untouched.
10. The experimental default stops at the 32M absolute-step `up` stage.
    Random down steps/slope introduction remain BLOCKED until independent
    0mm/0-degree skill-rating continuity exists. Old
    `scripts/overnight_v5020.sh --run` and `main` are unaffected.

User's one command (runs source gate and frozen CPU parity, then starts
2048-world GPU PPO only if they pass):

```bash
cd ~/emu-v5020
git switch feature/v5020-coldstart-terrain
git pull --ff-only origin feature/v5020-coldstart-terrain
source .venv/bin/activate
bash scripts/run_v5020_2048.sh --run
```

No one has run this revised code on the user's WSL GPU yet. Do NOT claim
acceptance or guaranteed performance. If preflight fails, do not bypass
it. Fetch `~/robot_lab/emu_runs/v5020_2048_<gitshort>/logs/`, plus
the printed traceback and failure checkpoint provenance, to diagnose
before spending GPU time. Even successful frozen 4-world parity is
only a necessary short-horizon check, not proof of robust locomotion.


### Final launch scope clarification (do not overstate)
The `scripts/run_v5020_2048.sh` entry point now offers only
`--stop-after foundation|up`. The previously outlined mixed12 and
mixed20 stages are **removed from executable launcher code**, not merely
hidden after a run-time branch. The only new physical curriculum in
this reproducible invocation is the rated step-UP feature, with the
absolute safety cap fixed at 6mm. Down-step and slope experiments
require separately verified zero-rated axes and a new later study.

Before launch the script also runs the read-only frozen CPU transfer
exam `scripts/verify_v5020_bridge.py` (4 worlds, 160 frames, same saved
actor/norm/command on original flat, zero-height platform and actual
microstep platform). This is NOT a tiny training run and does not alter
the source checkpoint. It must pass before CUDA PPO begins.

Any existing `UNSAFE_RESUME.json` quarantines the continuation; never
remove it to dodge a failed gate. The original `main` branch and public
v5016 Pinnacle demo remain untouched.


## 2026-10-22 — missing clean results bundle restored (post-32M)

The protected 2048-world terrain-bridge run on source `6c22a5678442`
COMPLETED to 32,047,104 transitions with final logged reward +3.180,
episode length ~699, falls 1.1%, frozen broad score 0.095. This is
valuable **walk-preservation evidence**, not 6mm stair certification;
the last logged earned terrain rating was only ~0.231mm.

The 20M→32M launcher initially omitted the existing offline results
generation and printed two huge, unlabeled CSV rows instead. FIXED:
`scripts/run_v5020_2048.sh --run` now automatically calls
`emu report --root "$RUN_ROOT"` after the final checkpoint is saved.
It leaves the source checkpoint untouched and writes:
- `$RUN_ROOT/results.md`: readable narrative, trends, independent panels,
  separate earned stair rating vs immutable safety ceiling, warnings
- `$RUN_ROOT/results_data.json`: machine-readable summary and SHA256 lineage
- `$RUN_ROOT/results_bundle.zip`: both reports and supporting
  metrics/fixed_eval/config/metadata files, WITHOUT huge pickle models.

`emu/results.py` now discovers the v5020 stage-root layout
(`foundation/`, `up/`) in addition to older tournaments and
single-run `up/`. It includes the actual source Git commit and the
`lineage/up.parent.sha256` hash. Unit tests in
`tests/test_v5020_report.py` cover offline backfill and safe ZIP
exclusion of checkpoint bytes.

**Retroactive report generation for the user's ALREADY FINISHED run,
without training and without touching the old source checkpoint:**

```bash
cd ~/emu-v5020
git switch feature/v5020-coldstart-terrain
git pull --ff-only origin feature/v5020-coldstart-terrain
source .venv/bin/activate
emu report --root ~/robot_lab/emu_runs/v5020_2048_6c22a5678442
```

Then attach
`~/robot_lab/emu_runs/v5020_2048_6c22a5678442/results_bundle.zip`.
DO NOT rerun `scripts/run_v5020_2048.sh --run` to create a report.
The reporter is offline/read-only regarding training artifacts. No
GPU tests or PPO continuation were run by the assistant for these
reporting changes.


## 2026-10-22 evening — proposed larger continuous overnight rated-up campaign

**Evidence used:** user's uploaded 32M `results_bundle(3).zip`. Exactly
32,047,104 steps (2048 worlds, seed 733, source
`6c22a56784424c576c254c153b172a40849bf235`), final policy
SHA256 `ffd9c2126cc5567826ed2820a0fbdb830edea52597dd6f2e5a7fb4c0bc75c344`.
Frozen foundation walking pass ~99.976%; core survival 100%; final
training reward +3.180, episode length ~699, falls ~1.1%, held-out
broad score +0.095. The continuously earned up-step rating was
**0.231mm** versus a 6mm safety ceiling, and the last logged above
rating trials were only 3 (sparse). The report exposes very high
servo-over-rated time (~62.2% late): an issue for later servo-efficiency
research, NOT a reason to change reward weights mid-continuation.

New feature-branch entry point
`scripts/overnight_v5020_up.sh` has intentionally NO experimental
new terrain modes, profiles, teaching bonuses, optimizer changes or
servo reward shocks. It continues the exact *same* rated 312-input
up-step training from saved 32M final to cumulative milestones:
- `up64` 64M (32M additional)
- `up128` 128M (64M additional)
- `up224` 224M (96M additional)
- `up320` 320M (96M additional; total ~288M more)

Each is full 2048 MJX envs, seed 733, 8 degree pitch, same 0.4
height exposure and .30 challenge band, exact 6mm safety ceiling,
same flat rehearsal, same foundation PPO/gait profile and NO terrain
reward. The ACTUAL rating is **restored** from the source policy
checkpoint on every stage transition, not reset at a phase boundary.
Only absolute target transition count and run output directory change.

Safety:
- Launcher source commit pinned; no dirty working tree, no `main`;
  parent checkpoint exact SHA256 and expected original source SHA,
  step count and 312 channel/sensor/PPO/optimizer/curriculum contracts.
- `emu/train_cli.py::verify_rated_parent_continuation` rejects ANY
  new config difference beyond approved source-parent identity, output
  paths and absolute targets. It is called when inspecting rated parent
  handoffs, not only flat-foundation handoffs.
- Before first GPU training: tests
  `tests/test_v5020_overnight.py`,
  `tests/test_v5020_report.py`,
  `tests/test_v5020_2048_launcher.py`,
  `tests/test_v5020_rated_elevation.py`.
  Broken tests abort BEFORE GPU.
- Midstage sustained 90%+ catastrophic falls or impossible >8m/s
  forward speed creates `UNSAFE_RESUME.json` and forbids child resume.
- End-of-stage frozen exact-checkpoint exam and median last-three
  on-policy rows must preserve >=75% walking and core survival,
  <=25% last-three fall rate, >=400-frame length. Otherwise DO NOT
  start the next stage. Parent checkpoints are never modified.
- After EACH completed stage, automatically write
  `results.md`, `results_data.json`, `results_bundle.zip` from
  all completed available stages. Reporter excludes .pkl checkpoints.
  They remain available even if the next stage gate rejects.
- 6mm remains an absolute simulation safety limit, NOT earned skill.
  No descent or sloped-world transition is introduced overnight.

**Copy/paste** after feature branch checkout:
```bash
cd ~/emu-v5020
git switch feature/v5020-coldstart-terrain
git pull --ff-only origin feature/v5020-coldstart-terrain
bash scripts/overnight_v5020_up.sh --plan
bash scripts/overnight_v5020_up.sh --run
```
Keep WSL/PC awake. Optional earlier cutoff:
`bash scripts/overnight_v5020_up.sh --run --stop-after up128`.
After completion attach
`~/robot_lab/emu_runs/v5020_overnight_up_<HEAD12>/results_bundle.zip`.

**Not yet executed on the user's GPU by the assistant.** Source,
test and script commits live only on the protected feature branch.
No change to `main` or showcase; actual proof comes from user's
preflight and collected training logs.


## 2026-10-22 — concise operator console requested during overnight v5020

Feature-branch **NEXT RUN ONLY**, do not pull into or restart any
already-active source-pinned overnight campaign.

`emu/results.py::format_line` now prints ONE meaningful line:
`steps`, `rew`, `len`, `fall`, `fps` (actual wall fps),
`vx_cmd=[lo,hi]` (allowed command envelope, NOT observed min/max),
`slope` (fixed physical floor angle; 'var' if per-world randomized),
`step` (earned physical height rating, NOT cap; flat if no adaptive
step terrain), `resil` (governor resilience difficulty),
`shaft` (mechanical absolute servo J/net forward world-X metre),
`over` (fraction of joint-frame observations exceeding nominal
continuous servo torque). If no useful forward distance is measured,
shaft='n/a' instead of an unstable or infinite cost. Work uses only
existing transferred rollout measurements and a tiny NumPy reduction;
not electrical battery energy, and no simulator or neural change.
Warnings for nan-grad skips and global curriculum retreats survive.
Every detailed field is still preserved in the metrics CSV/reports.

Both periodic and final output use exactly the same formatter.
`tests/test_v5020_console_feedback.py` tests units, backward
slip penalty, no-motion sentinels, one-line and semantic labels;
both v5020 launchers include it in PRE-GPU pytest.

**Training already underway retains old console formatting until
the process ends.** Don't pull into its checkout mid-campaign:
the launcher strictly pins a source Git commit across stages.
This is purely presentation/CPU summary work for the NEXT launch,
not a reason to interrupt a healthy 320M overnight.
