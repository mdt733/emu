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
