# EMU validation — evidence before promotion

This document owns **what qualifies as improved real locomotion**. Training
reward, curriculum promotion, a nice browser animation and a GitHub PR
are not independent physical exams.

## What the robot is meant to do

Measure controlled movement in metres and world coordinates, under the
**operator's requested** `vx`, `vy`, yaw and posture commands. A capable
H40 should survive, track commanded velocity, maintain its intended heading,
avoid unsafe contacts and excessive rated motor load, clear signed slopes
and real steps, then **keep moving afterward**. Standing motionless,
turning away, diving over an edge, or touching the far side once is
insufficient.

Metrics belong to six measured capability families: command tracking,
slopes/camber, signed UP and DOWN stairs, disturbances/recovery, transitions
and movement efficiency (including supported useful distance per shaft
work). Report failures and 0-success bins as clearly as successes.

## Fixed evidence and fair comparisons

Use the real **eight-prism MJX geometry**, real model-rated torque and the
frozen export's correct 360-input observation/normalizer. Make A/B/C
comparisons on the SAME physical task matrix: seeds, command, horizon,
ground profile, sensor geometry, evaluator version and birth position.
No different simulator behind a prettified viewer score.

Report each relevant step stage separately:

1. Real measured lip and approaching position (not requested difficulty).
2. First foot reaches the target prism with actual support.
3. Other foot independently reaches the target and holds support.
4. Body crosses while upright and respecting heading/lane limits.
5. **After the lip:** 0.5–1 m of continued **commanded** motion without fall.
6. For C's two-lip school: both signed lips plus post-second stable
   continuation and a recent `vx` tracking window.

An "Academy success" from training is a different, historically changing
denominator. Never compare B's short-lip success percentage directly to
C's two-lip-plus-continuation percentage. Keep matched flat controls
and report uncertainty, especially for rare tall steps.

## Current frozen baselines

**Mainline, not replaced:** committed H40 120,061,952-step actor,
source `301d18b7b1cd1b165e6578fc53f9f8cda94fb680`, seed 735,
`showcase/champion.emupolicy`. Owner-observed stable gait and slopes
near 15° are qualitative browser reports, not hardware proof.

The independent November 11, 2026 2,000-pair signed-prism stair-versus-flat
exam showed a substantial limit near **+30–32.5 mm**. In +2.5..27.5 mm
ups, controlled traversal was **179/550** vs paired-flat **192/550**;
at +35..50 mm there were **0/350 first upper footholds**.
DOWN first-foot acquisition succeeded in 1000/1000 signed trials,
but far-side support and lane/heading control declined with depth.
The flat control itself managed only **745/2000 (37.25%)** strict
continuations. These numbers belong **only** to that actor and examiner.
Full evidence and original files are in
[issue #62](https://github.com/mdt733/emu/issues/62),
[PR #69](https://github.com/mdt733/emu/pull/69), and
[the historical findings ledger](https://github.com/mdt733/emu/blob/69c9d6a959147d5f02ebf2a024a28de00362a6b4/docs/RESEARCH_FINDINGS.md).

**Experiment B, not main:** the approximately 420M bidirectional Academy
run has qualitatively better coached signed-down step results than the
old single-direction school, but no matched, independent, two-lip
continuation exam has promoted it. Keep its export and checkpoint
separate. Draft B is [PR #81](https://github.com/mdt733/emu/pull/81).

**Experiment C, running/unproven:** 500M *requested* fresh steps,
single 2048-world H40 PPO, random weights at zero, seed 735.
Two real 1m-spaced signed lip lessons, first-lip contact for
introductory coaching only; full completion requires both lips and
50 cm of continued forward tracking. The new success rule is a
**training hypothesis**, not a demonstrated result. See
[PR #84](https://github.com/mdt733/emu/pull/84).

## Required gates for moving a candidate to main

- **Source/contract:** code checkout and model/actor/ToF/servo hashes
  fully recorded; frozen champion and its release manifest untouched.
- **Reproducibility:** correct cold source SHA, checkpoint+normalizer+
  optimizer+curriculum state, complete `results.md` and independent
  evaluation outputs. No incompatible parent policy or quiet reset.
- **Robotics:** matched frozen-command and terrain matrix vs current
  champion and B; genuine UP and DOWN contacts; both feet, recovery
  and sustained post-lip `vx`; no major regression in ordinary walking,
  reverse/lateral/yaw, resilience, slopes or control effort.
- **Physical safety:** no actuator command above the specified rated
  ±0.980665 N·m cap, and compare saturation, knee spring, shin clearance,
  fall modes and solver/CPU-versus-GPU parity. Simulation cannot certify
  real motor thermal, battery or link dynamics.
- **Software:** ≤24 core Python tests, real CPU MJX/prism CI, CUDA
  2048-world full PPO acceptance and checkpoint resume/export checks,
  current browser controls plus manual scenario review.
- **Promotion:** merge the proven source only after review, explicitly
  choose/export the new champion and verify its original source, then
  separately approve the manual public Pages release. A code merge
  must never silently change the checked-in public policy.

This page records acceptance conditions, **not approval**.
The Git history and source-pinned PRs retain old experimental detail;
the maintained tree needs no additional per-version study runbooks.


## Experiment D — learn a better coach, not a champion

**Research-only, not promoted.** D is a 16-treatment, 2×2×2×2 full
factorial study of the *teaching process*. Every arm independently
inherits the SAME cold 60M H40 walking foundation checkpoint, trains
another 100M world steps under the SAME source, seed, physical robot,
8-prism MJX collision engine, observer, actor, PPO, base curriculum
and normalizer. No arm is eliminated; there is no finalist or winner
continuation. The sample budget is 60M + 16 × 100M = **1.66B total**
environment steps distributed across 17 separate runs.

All 16 arms share the D correction to C's broken success timing:
1. first genuine bilateral first-lip top support,
2. separate second real signed lip, then verified bilateral support,
3. latched 50 cm downstream progress after that supported second step,
4. recent ≥20-frame genuine operator-velocity tracking,
5. upright, properly supported and not falling on completion.

Once 50 cm is achieved, a slightly late gait recovery can earn the
payment. Raw one-foot, unsupported, ballistic, backward and
single-lip shortcuts remain ineligible. The pre-existing C single-
frame course condition is still testable by switching the arm to
`off`, which is used in the pre-step foundation.

The four experimental factors:

| Factor | Treatment | Mechanism / hypothesis |
|---|---|---|
| P: progressive | One-lip+continuation introductory teaching before strict two-lip graduation | A staged goal is easier to discover |
| T: free-phase | Withdraw phase-synchrony shaping near *real* lips; the deployable gait clock itself does **not** change | Prescribed gait reward discourages deliberate step planning |
| F: foothold | Extra one-time real supported crossing, upper foothold, bilateral transfer and 50 cm progress payments | Dense physically grounded milestones improve exploration |
| C: commands | 0.08–0.30 m/s stair approach, half mixed ±0.05 lateral and ±0.12 yaw, ±14° birth heading | Straight-only Academy lessons underfit actual operator use |

Every on/off subset, including four singles, six pairs, four triples,
zero factors and all four together, is included. Interaction effects
matter: T may help alone, do nothing with C, but help with F. A high
overall actor score is **not** sufficient to declare a teaching
mechanism useful. Each branch is a separate causal observation, not
a production candidate.

**What every branch must report** (use matched exposure denominators):
- independently verified UP vs DOWN attempts, first-lip bilateral
  contact, second-lip arrival and second supported transfer;
- 50 cm continuation after the second, recent actual `vx` tracking,
  complete physical course and failures by stage;
- survival, commanded `vx`/`vy`/yaw error, slope/flat retention,
  useful work, rated-torque saturation, time to milestones and actual
  lessons offered (distribution shift);
- early learning slope and evidence accumulation, not just final
  achieved ability; frozen matched exams at fixed policy ages;
- a description of **what was learned about the teacher**, including
  negative findings, failure modes, likely measurement artifacts and
  confounding from interventions.

Exams are for **diagnosis**, not ranking. The next objective is to
consolidate compatible demonstrated improvements into one simple
teacher, clean out rejected experiment flags, then train a fresh
end-to-end learner with the final curriculum. Neither D nor C changes
the committed public champion or approves a Pages release.

From the pinned D source worktree:

```bash
bash scripts/run_h40_d_coach_lab_wsl.sh all
```

The launcher can also run `foundation`, `list` or
`arm 04-commands` independently. It validates complete checkpoint
ancestry before each fork, keeps each output in a distinct directory
under `~/emu-data/runs/h40-d-coach-lab-s735` and resumes interrupted
arms from their own checkpoints (never from the 60M parent after
partial work). All 16 branches use the original training source SHA
in their saved metadata.

**Deployment parity requirement for Factor T:** The actor's phase clock
is never advanced, slowed or reset using teacher-only true lip
coordinates. Only the *reward pressure* to synchronise foot movement
to that clock is relaxed near a physical stair. That is a teaching
factor which can be tested without inventing a new runtime stair
detector in the browser or hardware.

## D frozen-policy science examiner — use ALL 16, never select a winner

This source-pinned read-only examiner extracts as much as possible from
the completed 1.66B world-step teaching laboratory. It runs **no PPO,
no policy updates and no curriculum promotion**. Run in an independent
detached worktree. The training lineages and D's current chosen policies
are never rewritten.

```bash
# From a clean worktree at science/h40-d-coach-autopsy:
source ~/emu/.venv/bin/activate
export EMU_PHYSICAL_PROFILE=h40 PYTHONPATH="$PWD"
python -m scripts.examine_h40_d_coaches --smoke
python -m scripts.examine_h40_d_coaches
```

The `--smoke` mode exercises the full export → GPU matrix → true-MJX
two-lip trace → package chain with one actor and a different output
directory. Check that it finishes before running the complete scan.
The complete run resumes already verified cases and produces:

`~/emu-data/exams/h40-d-coach-science-v1/science-bundle.zip`

**This is the single file to upload for investigation.** It includes
all 16 training configs, full logged `metrics.csv`, fixed evaluations,
summary results, checkpoint SHA256 **but not checkpoint bytes**, actor
weight fingerprints **but not policy weights**, exact-source/robot
hashes, source-grounded treatment-integrity warnings and the independent
physical read-only test evidence, including per-world CSV and selected
event-level trajectory extracts.

One matched 2048-world CUDA batch per actor for each: a signed 0–50mm
stair matrix in 2.5mm increments (50 replicas per height/sign), the
same-index 2000-world zero-height seam, and a 10-family balanced
control/slope/command matrix. These tests use the existing
`scripts.capability_matrix_exam` real-prism first-life evaluator.

A separate genuine *two-riser* true-MJX Academy collision fixture
runs 128 worlds for up-up, sampled mixed signs and down-heavy first
lips at seed 1701 and independent seed 2701; it also tests an identical
mixed operator request, input-only ToF masking and phase holding at
seed 1701. Each trace spans 1050 control frames (21 seconds). It
records per-foot support **before chassis crossing**, clearance against
real target triangle surfaces, five-frame sequential and simultaneous
upper contacts **separately** at both lips, 50cm downstream, the
full 20-frame commanded-velocity window, falls, fatigue proxies,
torque-cap violations, 16-zone ToF validity and reasons for failure.

**Limits:** up/down Academy signed heights are physically generated
according to its saved shape, not exact across all levels; paired
seam matrices are exact but only single-riser. This does not supply
randomized training-seed confidence, measured real motor temperature,
hardware VL53L8CX data or proof a policy learned a new behaviour.
The input ablations are counterfactual policy-sensitivity tests.
Beware historical contact and reward label inconsistencies: the
independent score deliberately does not read Academy success flags.
The archive reports the original training reward evidence separately
from the independently measured capabilities.

**Resource discipline:** do not compile 2048 worlds plus 128 traces in
one JAX process. The driver launches one bounded child at a time so
GPU allocations return between scenarios. Interrupted stages preserve
logs and completed evidence; a complete ZIP is NEVER emitted while a
mandatory case is missing or has an incorrect policy/protocol hash.
